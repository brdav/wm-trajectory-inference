"""Camera intrinsics handling across frames.

A video comes from one physical camera, so every frame truly shares the same
K, but each backend model estimates K independently per submap chunk and per
loop-closure pair -- with no consistency constraint between estimates. That
lets ordinary model noise masquerade as frame-to-frame K drift, which then
leaks into point-cloud unprojection, inter-submap scale estimation, and GPS
fusion (see memory/backend-choice.md).

IntrinsicsLock gives Solver two modes:
  - fixed_intrinsics=<K>: the calibration is already known -- use it for every
    frame, ignore the backend's own estimate entirely.
  - neither set: unchanged current behavior, each frame keeps its own
    independently estimated K.

(K unknown but constant across the video -- e.g. --shared_intrinsics -- is
handled entirely outside this class, via da3_ray_pose_patch's two-phase joint
K+R refit: see main.py's --shared_intrinsics validation and
da3_ray_pose_patch.joint_refit. That path pools every submap's free K
estimate into a single whole-video median, which is far better-conditioned
than an online running median updated one submap at a time -- so there is no
running-consensus mode here.)

Resolution note: backends resize images internally before inference (e.g. DA3
resizes a 2896x1876 frame to 504x322), and `predictions["intrinsic"]`/depth are
self-consistent with THAT resized resolution, not the original file's. A
user-supplied fixed_intrinsics is calibrated against the original resolution
(e.g. from a calibration file), so it must be rescaled to match whatever
resolution the backend actually produced for THIS call before being used --
otherwise cx/cy end up centered far outside the actual pixel grid. This is why
fixed_intrinsics_size (the resolution the given K was calibrated for) is
required, and why `apply()` takes `actual_hw`.
"""
import numpy as np


class IntrinsicsLock:
    def __init__(self, fixed_intrinsics=None, fixed_intrinsics_size=None):
        if fixed_intrinsics is not None and fixed_intrinsics_size is None:
            raise ValueError("fixed_intrinsics requires fixed_intrinsics_size (the width,height it was calibrated for)")

        self._fixed_K = None
        self._fixed_ref_wh = None
        if fixed_intrinsics is not None:
            self._fixed_K = np.asarray(fixed_intrinsics, dtype=np.float64).reshape(3, 3)
            self._fixed_ref_wh = (float(fixed_intrinsics_size[0]), float(fixed_intrinsics_size[1]))

    @property
    def is_active(self):
        return self._fixed_K is not None

    def _fixed_K_for(self, actual_w, actual_h):
        ref_w, ref_h = self._fixed_ref_wh
        sx, sy = actual_w / ref_w, actual_h / ref_h
        K = self._fixed_K.copy()
        K[0, 0] *= sx  # fx
        K[0, 2] *= sx  # cx
        K[1, 1] *= sy  # fy
        K[1, 2] *= sy  # cy
        return K

    def apply(self, intrinsics, actual_hw=None):
        """intrinsics: (S, 3, 3) per-frame estimated K from the backend.

        actual_hw: (H, W) of the depth/images this batch of intrinsics actually
        corresponds to (e.g. predictions["depth"].shape[1:3]) -- required, to
        rescale the reference K to the backend's actual working resolution for
        this call.

        Returns a (S, 3, 3) array: the fixed K, rescaled to actual_hw and
        broadcast to all S frames (or, if not active, `intrinsics` unchanged).
        """
        if self._fixed_K is None:
            return intrinsics

        if actual_hw is None:
            raise ValueError("fixed_intrinsics mode requires actual_hw to rescale to the backend's working resolution")
        actual_h, actual_w = actual_hw
        K = self._fixed_K_for(actual_w, actual_h)
        return np.tile(K[None, :, :], (intrinsics.shape[0], 1, 1)).astype(intrinsics.dtype)

    def get_backend_lock(self):
        """Returns (K_pixel, ref_wh, "fixed") for backends that can lock K internally (currently
        only DA3Backend's --use_ray_pose path, see vggt_slam/da3_ray_pose_patch.py), or None if
        fixed_intrinsics isn't set."""
        if self._fixed_K is not None:
            return self._fixed_K, self._fixed_ref_wh, "fixed"
        return None


def parse_fixed_intrinsics(s):
    """Parse a "fx,fy,cx,cy,width,height" CLI string into ((3,3) zero-skew K, (width, height))
    -- width,height is the resolution the given K was calibrated for, required so IntrinsicsLock
    can rescale it to whatever resolution the backend actually processes a given call at."""
    if s is None:
        return None, None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 6:
        raise ValueError('--fixed_intrinsics expects 6 comma-separated values "fx,fy,cx,cy,width,height"')
    fx, fy, cx, cy, width, height = parts
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, (width, height)
