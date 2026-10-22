"""Pluggable feed-forward geometry backends (VGGT / Depth Anything 3).

`Solver.run_predictions` is written against this interface so the rest of the
pipeline (submap storage, SL(4) pose-graph optimization in graph.py, scale
alignment in scale_solver.py) never needs to know which model produced a given
submap's poses/depth. Both backends return the same torch-tensor contract --
`images`, `extrinsic`, `intrinsic`, `depth`, `depth_conf` -- carrying the same
leading dummy batch-of-1 dimension the raw VGGT model output has on
`extrinsic`/`intrinsic`/`depth`/`depth_conf` (but not `images`), so
`Solver.run_predictions`'s generic tensor->numpy `.squeeze(0)` loop applies
unchanged regardless of backend.
"""
import cv2
import numpy as np
import torch

from vggt_slam.undistort import undistort_frame


class VGGTBackend:
    """Wraps MIT-SPARK's VGGT fork: joint multi-view depth+pose, plus a custom
    2-view `compute_similarity` mode used for loop-closure verification.

    `known_intrinsics` (a Solver.IntrinsicsLock lock tuple) is accepted but ignored: VGGT has
    no equivalent of DA3's ray-pose internals to lock, so Solver's own post-hoc
    IntrinsicsLock.apply() override remains the only mechanism for this backend."""

    def __init__(self, model):
        self.model = model

    def infer_submap(self, image_names, device, known_intrinsics=None, frame_cache=None):
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = load_and_preprocess_images(image_names).to(device)
        with torch.no_grad():
            predictions = self.model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions["images"] = images
        predictions["intrinsics_locked_internally"] = False
        return predictions

    def infer_loop_closure_pair(self, query_frame, query_frame_path, retrieved_frame, retrieved_frame_path, device, known_intrinsics=None):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        lc_frames = torch.stack((query_frame, retrieved_frame), axis=0)
        with torch.no_grad():
            predictions_lc = self.model(lc_frames, compute_similarity=True)
        extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(predictions_lc["pose_enc"], retrieved_frame.shape[-2:])
        return {
            "match_score": float(predictions_lc["image_match_ratio"]),
            "extrinsic_lc": extrinsic_lc,
            "intrinsic_lc": intrinsic_lc,
            "depth_lc": predictions_lc["depth"],
            "depth_conf_lc": predictions_lc["depth_conf"],
            "frames_lc": lc_frames,
            "intrinsic_lc_locked_internally": False,
        }


def _da3_reprojection_match_ratio(depth, conf, extrinsic, intrinsic, conf_percentile=50.0, rel_depth_tol=0.1):
    """Geometric stand-in for VGGT-fork's learned `image_match_ratio`.

    DA3 has no trained match-quality head, so this unprojects the query
    frame's (index 0) high-confidence pixels using the pair's own jointly
    estimated depth+pose and checks what fraction land in-bounds and
    depth-consistent in the retrieved frame (index 1) -- i.e. whether DA3's
    own 2-view reconstruction is geometrically self-consistent, which it
    usually isn't for two unrelated images. Not calibrated against the VGGT fork's
    learned ratio, see memory/backend-choice.md.
    """
    H, W = depth.shape[1:3]
    K0, K1 = intrinsic[0], intrinsic[1]
    R0, t0 = extrinsic[0, :, :3], extrinsic[0, :, 3]
    R1, t1 = extrinsic[1, :, :3], extrinsic[1, :, 3]

    conf_thresh = np.percentile(conf[0], conf_percentile)
    ys, xs = np.nonzero(conf[0] > conf_thresh)
    if len(xs) == 0:
        return 0.0

    z0 = depth[0, ys, xs]
    pix0_h = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    cam0 = np.linalg.inv(K0) @ pix0_h * z0[None, :]

    world = R0.T @ (cam0 - t0[:, None])
    cam1 = R1 @ world + t1[:, None]
    z1_pred = cam1[2, :]
    valid_depth = z1_pred > 1e-3

    pix1 = K1 @ cam1
    pix1 = pix1[:2, :] / np.clip(pix1[2:3, :], 1e-6, None)
    x1, y1 = pix1[0, :], pix1[1, :]

    in_bounds = (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H) & valid_depth
    x1i = np.clip(x1.astype(np.int32), 0, W - 1)
    y1i = np.clip(y1.astype(np.int32), 0, H - 1)
    z1_obs = depth[1, y1i, x1i]

    depth_consistent = np.abs(z1_pred - z1_obs) < rel_depth_tol * np.maximum(z1_obs, 1e-3)
    return float((in_bounds & depth_consistent).mean())


class DA3Backend:
    """Wraps Depth Anything 3 (github.com/ByteDance-Seed/Depth-Anything-3).

    Unlike VGGT, DA3's `.inference()` does its own image loading/resizing, so
    it takes file paths directly rather than a pre-loaded tensor batch. DA3
    has no equivalent of VGGT-fork's `compute_similarity`/`image_match_ratio`
    loop-closure verification head, so `infer_loop_closure_pair` scores the
    pair geometrically instead (see `_da3_reprojection_match_ratio`).

    `known_intrinsics`, when given and use_ray_pose=True, is a (K_pixel, ref_wh, "fixed") tuple
    from Solver.IntrinsicsLock.get_backend_lock() -- it gets threaded into
    vggt_slam.da3_ray_pose_patch so the ray-pose rotation solve is fit directly against the
    known K instead of DA3's own free per-frame estimate (see that module's docstring for why).
    Ignored when use_ray_pose=False: that path uses a different internal head (the camera
    decoder) this patch doesn't touch, so Solver's post-hoc IntrinsicsLock.apply() override
    remains the only mechanism there. --shared_intrinsics (K unknown but constant) does NOT go
    through known_intrinsics at all -- see vggt_slam.da3_ray_pose_patch's capture/joint_refit and
    vggt_slam/pipeline.py's two-phase orchestration.
    """

    def __init__(self, model, use_ray_pose=True, ref_view_strategy="saddle_balanced"):
        self.model = model
        self.use_ray_pose = use_ray_pose
        self.ref_view_strategy = ref_view_strategy
        self._undistort_map1 = self._undistort_map2 = self._undistort_roi = None
        if use_ray_pose:
            from vggt_slam import da3_ray_pose_patch
            da3_ray_pose_patch.install()

    def set_undistort_maps(self, map1, map2, roi):
        """Called once per clip (vggt_slam/pipeline.py's run_video) when frames on disk are raw/distorted --
        makes _run's path-based fallback below (used by infer_loop_closure_pair, which re-reads
        by path instead of using frame_cache -- see its docstring) apply the same rectification
        the keyframe-selection producer thread applies, instead of silently reading a still-
        distorted frame while the rest of the pipeline assumes rectified geometry."""
        self._undistort_map1, self._undistort_map2, self._undistort_roi = map1, map2, roi

    def _resolve_image(self, p, frame_cache):
        if frame_cache and p in frame_cache:
            return frame_cache[p]
        if self._undistort_map1 is not None:
            img = undistort_frame(cv2.imread(p), self._undistort_map1, self._undistort_map2, self._undistort_roi)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return p

    def _run(self, image_paths, device, known_intrinsics=None, frame_cache=None):
        lock_active = known_intrinsics is not None and self.use_ray_pose
        if lock_active:
            from vggt_slam import da3_ray_pose_patch
            K_pixel, ref_wh, mode = known_intrinsics
            da3_ray_pose_patch.set_lock(K_pixel, ref_wh, mode)
        # Use the already-decoded (RGB, already-undistorted) array vggt_slam/pipeline.py's keyframe-selection
        # pass cached for this path when available -- DA3's own inference() accepts np.ndarray
        # directly (see class docstring), so this skips a second read+decode of the same file
        # from disk for every frame that was already read once for the disparity check. Falls
        # back to reading (and, if set_undistort_maps was called, rectifying) the path directly
        # for any frame not in the cache -- e.g. infer_loop_closure_pair, which re-reads by path
        # for a second, independent 2-view pass instead of reusing the submap's own preprocessed
        # tensors (out of scope to change here -- see infer_loop_closure_pair's call site).
        images_input = [self._resolve_image(p, frame_cache) for p in image_paths]
        try:
            pred = self.model.inference(
                images_input, use_ray_pose=self.use_ray_pose, ref_view_strategy=self.ref_view_strategy
            )
        finally:
            if lock_active:
                from vggt_slam import da3_ray_pose_patch
                da3_ray_pose_patch.clear_lock()
        images = torch.from_numpy(pred.processed_images).to(device).float().permute(0, 3, 1, 2).contiguous() / 255.0
        extrinsic = torch.from_numpy(pred.extrinsics).to(device).float().unsqueeze(0)
        intrinsic = torch.from_numpy(pred.intrinsics).to(device).float().unsqueeze(0)
        depth = torch.from_numpy(pred.depth).to(device).float().unsqueeze(0).unsqueeze(-1)
        depth_conf = torch.from_numpy(pred.conf).to(device).float().unsqueeze(0)
        return images, extrinsic, intrinsic, depth, depth_conf, pred, lock_active

    def infer_submap(self, image_names, device, known_intrinsics=None, frame_cache=None):
        images, extrinsic, intrinsic, depth, depth_conf, _, lock_active = self._run(
            image_names, device, known_intrinsics, frame_cache=frame_cache
        )
        return {
            # Leading dummy batch-of-1 dim, matching VGGT's own echoed "images"
            # (VGGT's forward operates on a (1, S, 3, H, W) batch internally),
            # so Solver.run_predictions's generic numpy .squeeze(0) loop -- which
            # raises on a non-1 axis, unlike torch's -- applies uniformly.
            "images": images.unsqueeze(0),
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth": depth,
            "depth_conf": depth_conf,
            "intrinsics_locked_internally": lock_active,
        }

    def infer_loop_closure_pair(self, query_frame, query_frame_path, retrieved_frame, retrieved_frame_path, device, known_intrinsics=None):
        images, extrinsic, intrinsic, depth, depth_conf, pred, lock_active = self._run(
            [query_frame_path, retrieved_frame_path], device, known_intrinsics
        )
        match_score = _da3_reprojection_match_ratio(pred.depth, pred.conf, pred.extrinsics, pred.intrinsics)
        return {
            "match_score": match_score,
            "extrinsic_lc": extrinsic,
            "intrinsic_lc": intrinsic,
            "depth_lc": depth,
            "depth_conf_lc": depth_conf,
            "frames_lc": images,
            "intrinsic_lc_locked_internally": lock_active,
        }
