"""Overlay each frame's future ego trajectory onto its own image, producing a dashcam-style
trajectory-overlay video (a curving path drawn on the road ahead).

Assumption (per user): the vehicle drives in the direction the front camera faces at that
instant - i.e. there's no separate camera-to-ego calibration, a frame's own ego pose IS treated
as its camera pose. So for frame i we take every future position from result.npz's fused_poses,
transform it into frame i's own local frame, and project it into the image with the front
camera's real calibration (OpenCV convention: x-right, y-down, z-forward, matching VGGT-SLAM's
own pose convention already -- see vggt_slam/gps_fusion.py's module docstring).

Camera calibration (intrinsics + lens distortion + mount height) comes from the clip's trip-level
FIXED_METADATA.json, NOT the per-clip FRONT_FOLDER/*.npz's 'intrinsics' field - that file is a
separate, later-generated artifact (different mtime/group/perms than the original delivered data)
that appears to belong to some other, unrelated tracking pipeline (it ships 21-object pose tracks
alongside it) rather than the vehicle's actual device calibration.

Ported from DVGT's tools/natix_inference/overlay_trajectory_on_video.py, adapted for VGGT-SLAM's
own result.npz (written directly by main.py -- see vggt_slam.slam_utils.save_result_npz) instead
of DVGT1's run_dvgt1_trajectory.py output -- the two aren't interchangeable in a way a key rename
alone would fix: DVGT1's poses.npz stores a 'c2w' array sampled at a truly fixed 2Hz (BASE_FPS
below, in the original script), so consecutive array rows are evenly spaced in time by
construction. VGGT-SLAM's keyframes are NOT evenly spaced -- frame_ids are whichever
extracted-frame indices survived keyframe selection, so the real time gap between consecutive
poses varies (confirmed on a real run: diffs of [4, 3, 2, 2, 2, 1, 1, ...] extracted-frame steps
between consecutive keyframes, at a fixed extraction_fps). So every spline/interpolation step
below is now parameterized by each pose's own real elapsed time (frame_id, extraction_fps) ->
seconds, not by its position in the array.

Also needs the *GPS-fused/road-aligned* trajectory (result.npz's fused_poses field, written
directly by main.py), not VGGT-SLAM's raw slam_poses: this script adds a literal camera_height_m
(real meters) to each position, which only makes sense once the trajectory's own scale is real
metric meters -- raw VGGT-SLAM poses are in an arbitrary "visual gauge" scale (see gps_fusion.py)
where that offset would be wrong by whatever the visual-gauge-to-meters scale factor happens to
be for that run.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


def extract_background_frames(clip_dir, fps, out_dir):
    """Extract front-camera frames directly from the clip's own raw video at `fps`, for the
    overlay's background. batch_infer.py's own extracted frames (frames/frame_%05d.jpg)
    are only as dense as --extraction_fps and, more importantly, keyframe-selected (most were
    dropped before VGGT-SLAM ever saw them) - reusing those and just holding each one for
    several output frames would make the *visible scene* jump at each surviving keyframe, even
    though the projected path's viewpoint is smoothly interpolated. The source video itself is
    ~36fps, so real frames at the requested output fps actually exist; this re-extracts them
    fresh rather than reusing either of those.
    """
    front_folder = os.path.join(clip_dir, "FRONT_FOLDER")
    mp4_candidates = glob.glob(os.path.join(front_folder, "*.mp4"))
    if not mp4_candidates:
        raise FileNotFoundError(f"No {front_folder}/*.mp4 found")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_candidates[0], "-vf", f"fps={fps}", "-q:v", "2",
         os.path.join(out_dir, "bg_%05d.jpg")],
        check=True,
    )
    frames = sorted(glob.glob(os.path.join(out_dir, "bg_*.jpg")))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames from {mp4_candidates[0]}")
    return frames


def find_fixed_metadata(clip_dir):
    """FIXED_METADATA.json lives one level above the clip (it's per-trip, shared by every
    HH-MM-SS clip in that trip): .../<trip_id>/FIXED_METADATA.json next to .../<trip_id>/<HH-MM-SS>/.
    """
    trip_dir = os.path.dirname(os.path.normpath(os.path.abspath(clip_dir)))
    candidate = os.path.join(trip_dir, "FIXED_METADATA.json")
    return candidate if os.path.isfile(candidate) else None


def load_camera_calibration(fixed_metadata_path, device_name="camera_front"):
    """Real per-vehicle calibration for `device_name`: intrinsics K, distortion coeffs
    (k1, k2, k3, p1, p2), mount height (m; tz is in cm - matches known Tesla mount heights,
    e.g. Model 3 ~130cm vs Model Y ~147cm), and forward mount offset (m; tx, from the
    vehicle's "ground_nominal" reference frame -- used by eval_pose_spline's heading
    correction below). r11..r33 is the identity for camera_front on every real clip checked
    so far (no mounting yaw/pitch/roll), so it's not read here -- the camera's own orientation
    is assumed to equal the vehicle body's exactly.

    K is assumed to apply directly at the clip's own native video resolution - that's simply
    how camera calibration works (a lens/sensor is calibrated at whatever resolution it
    actually captures), and our frames are extracted from that same video with no resizing.
    No separate "reference resolution" needs to be guessed from the file's own
    frame_width/frame_height field, which isn't reliable on its own (one real clip's
    frame_width/height named a resolution exactly half its fx/cx's actual native one -
    confirmed by checking that clip's own mp4 directly).
    """
    with open(fixed_metadata_path) as f:
        meta = json.load(f)
    for dev in meta.get("device_extrinsics", []):
        if dev.get("device_name") == device_name:
            K = np.array([[dev["fx"], 0, dev["cx"]], [0, dev["fy"], dev["cy"]], [0, 0, 1]], dtype=np.float64)
            dist = (dev.get("k1", 0.0), dev.get("k2", 0.0), dev.get("k3", 0.0),
                    dev.get("p1", 0.0), dev.get("p2", 0.0))
            height_m = dev["tz"] / 100.0
            forward_offset_m = dev["tx"] / 100.0
            return K, dist, height_m, forward_offset_m
    raise ValueError(f"No device '{device_name}' in {fixed_metadata_path}")


def build_pose_splines(trajectory, times):
    """Build smooth interpolators through the ENTIRE real trajectory, parameterized by each
    pose's own real elapsed time in seconds (`times`, strictly increasing), not by its position
    in the array (VGGT-SLAM keyframes aren't evenly spaced in time -- see module docstring).

    Translation: a cubic spline (C2-continuous) -- verified accurate against the road-alignment
    plots, and its own derivative is reused in eval_pose_spline below to get the direction of
    travel at any query time.

    Rotation: RotationSpline, a single global curve through every real sample (C2-continuous,
    like the translation spline above), not piecewise SLERP between just the two nearest samples
    -- piecewise SLERP has a real velocity kink exactly at every sample, which would show up
    directly in its angular-RATE (order=1) evaluation, not just its interpolated value.
    eval_pose_spline below uses that angular rate (the vehicle's own yaw rate, exact everywhere
    on a rigid body regardless of mount offset) to correct for camera_front's forward mount
    offset, so a kink there would inject per-keyframe noise straight into that correction, not
    just smoothness. RotationSpline can, in principle, overshoot a sample's own value between two
    samples with a sharp heading change, unlike SLERP -- checked against real turns in this
    codebase's own clips, where it was negligible (peaked at 0.17deg across a real ~99deg turn)
    -- so smoothing out sample-to-sample jitter is the more relevant tradeoff here than that
    theoretical risk.
    """
    translation_spline = CubicSpline(times, trajectory[:, :3, 3])
    rotation_spline = RotationSpline(times, Rotation.from_matrix(trajectory[:, :3, :3]))
    return translation_spline, rotation_spline


def build_linear_interpolators(trajectory, times):
    """The uncorrected, unsmoothed alternative to build_pose_splines: piecewise-linear position
    and piecewise-SLERP rotation directly between the two nearest real keyframes -- no C2-
    continuous curve fit through the whole trajectory, and (see eval_linear_pose) no heading
    correction of any kind. This is "whatever the reconstruction (+ GPS fusion) itself said" with
    the least possible additional processing, for a raw baseline to compare the corrected
    overlay against -- see main's --raw_linear.
    """
    translation_interp = interp1d(times, trajectory[:, :3, 3], axis=0)
    rotation_interp = Slerp(times, Rotation.from_matrix(trajectory[:, :3, :3]))
    return translation_interp, rotation_interp


def eval_linear_pose(translation_interp, rotation_interp, query_time, times):
    """The piecewise-linear/SLERP counterpart to eval_pose_spline, with none of its correction
    logic -- just the two nearest real keyframes interpolated directly, exactly as reconstructed
    (+ GPS-fused/road-aligned). See main's --raw_linear.
    """
    query_time = np.clip(query_time, times[0], times[-1])
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation_interp(query_time).as_matrix()
    T[:3, 3] = translation_interp(query_time)
    return T


def eval_pose_spline(translation_spline, rotation_spline, query_time, times, forward_offset_m,
                      min_speed_mps=3.0, correct_heading=True, heading_window_s=1.5,
                      max_turn_deg=8.0):
    """The single smoothly-interpolated camera pose at query_time (real seconds since the first
    pose). This is used as the viewpoint we project from at each output-video frame.

    With correct_heading=True (the default), the forward axis is derived from the trajectory's
    own direction of travel instead of the reconstruction's own spline-interpolated rotation --
    fixing the dominant error: VGGT-SLAM's reconstructed camera yaw and its own reconstructed
    translation disagreed by a persistent ~6deg for several real seconds on a real straight
    stretch (where they must agree -- a car can't hold a steady crab angle) -- graph.optimize()
    turned out to be a no-op for that whole run (zero loop closures -> zero-error graph, confirmed
    both empirically via --skip_bundle_adjustment and structurally in solver.py's add_edge), so
    this is most likely the backbone's own per-frame rotation output wobbling relative to its own
    translation output, not a pose-graph artifact -- invisible in any position-only view (e.g.
    the road-alignment plots) but throwing the projected path noticeably off to one side, worse
    close to the camera, since it's an angular error rather than a positional one. Trusting
    translation over rotation for heading (this function) sidesteps that ~6deg error entirely,
    rather than trying to detect or fade around it.

    Direction of travel is deliberately taken over a window of real elapsed time (the
    displacement from query_time to query_time+window, not translation_spline's instantaneous
    derivative at query_time alone): checked on a real run, the instantaneous derivative
    disagreed with this windowed direction by another ~1-2deg even after the rotation-based
    ~6deg error above was already sidestepped -- small next to that ~6deg, but still large enough
    to visibly tilt the near-field projected path, since CubicSpline's derivative at a single
    instant is a locally-fit quantity and can wobble a couple degrees off whichever way the real
    path is actually heading over the next second or two, especially once the real keyframes it's
    fit through carry any of their own small position noise. Averaging over heading_window_s of
    real time cancels most of that out on a genuine straight.

    That window is only valid on a genuine straight, though: averaging displacement over a real
    turn doesn't recover the instantaneous heading at query_time, it "cuts the corner" -- the
    average direction from the start to the end of the window blends the turn's entry and exit
    heading, matching neither. Checked on a real run: right at a clip's own start, where the
    vehicle is slowly turning onto the road, the reconstructed yaw rate implies as much as ~20deg
    of real heading change within a naive fixed 1.5s window -- exactly the corner-cutting failure
    this would otherwise reproduce. So the window is shrunk adaptively from the reconstructed
    yaw rate (omega, already needed below for the mount-offset correction) to never average over
    more than max_turn_deg of real heading change: min(heading_window_s, max_turn_deg / omega).
    This is what the old straightness_weight blend (removed) was ultimately also protecting
    against, just via a heuristic "how much does direction-of-travel vary over a lookahead
    window" proxy instead of this closed-form bound on the same underlying quantity.

    Before taking that windowed direction, a real, calibrated rigid-body effect is corrected for
    first: camera_front is mounted forward_offset_m ahead of the vehicle's "ground_nominal"
    reference point (see load_camera_calibration -- ty=0 and R=identity there too, so there's no
    lateral or rotational mount offset to account for, only this longitudinal one). For a rigid
    body, a point offset by r from another point on the same body has velocity
    v_point = v_ref + omega x r (omega = the body's own angular velocity, identical everywhere on
    a rigid body -- so the reconstructed rotation spline's own angular rate IS the vehicle's yaw
    rate exactly, mount offset or not). With r = forward_offset_m * forward, that means the
    camera's own velocity direction is offset from the vehicle's true heading by
    forward_offset_m * (omega x forward) -- exactly zero when omega is zero (a real straight),
    growing only during a real turn, and by a known amount (no free parameter fit here --
    forward_offset_m comes straight off the vehicle's own calibration).

    With correct_heading=False, the reconstruction's own spline-interpolated rotation is used
    unmodified everywhere -- kept only for side-by-side comparison against the (default)
    corrected version above (see main's --reconstructed_heading), since it reproduces the
    pre-fix projection error.

    Pitch/roll aren't recoverable from the direction of travel alone (a horizontal-plane
    quantity), so they're kept from the spline-interpolated reconstructed rotation: its own down
    axis seeds a Gram-Schmidt rebuild of the other two axes against the new forward axis.

    Below min_speed_mps the direction of travel is too noisy to trust, so the reconstructed
    rotation is used unmodified instead, same as correct_heading=False -- this isn't just a
    near-stationary guard (see also project_future_path's own z>1m cutoff for why those frames
    already show no path at all regardless): checked on a real run, right at a clip's own start
    (a slow turn out of a driveway/side street onto the road, a few m/s), the windowed direction
    disagreed with the reconstructed rotation by 20-55deg -- one to two orders of magnitude past
    the ~6deg this function otherwise exists to fix, and window-shrinking (max_turn_deg) alone
    didn't fix it either, so this isn't corner-cutting: at just a few m/s, real position noise of
    a few cm is already a meaningful fraction of the per-frame displacement itself, so ANY
    displacement-derived direction -- windowed or instantaneous -- gets noise-amplified into
    exactly this kind of large angular swing. 3.0 m/s (~11kph) was the lowest threshold that
    cleanly fell back to the reconstructed rotation across that whole slow-turn section on a
    real run; the reconstructed rotation's own real rotational cues (e.g. optical flow) make it
    the more trustworthy channel there anyway, unlike on a confident-speed straight.
    """
    query_time = np.clip(query_time, times[0], times[-1])
    R_ref = rotation_spline(query_time).as_matrix()

    R = R_ref
    if correct_heading:
        omega = rotation_spline(query_time, 1)  # world-frame angular rate (rad/s)
        omega_mag = np.linalg.norm(omega)
        max_turn_rad = np.radians(max_turn_deg)
        window = heading_window_s if omega_mag * heading_window_s <= max_turn_rad else max_turn_rad / omega_mag
        window_end = min(query_time + window, times[-1])
        dt = window_end - query_time
        if dt > 1e-3:
            windowed_velocity = (translation_spline(window_end) - translation_spline(query_time)) / dt
            lateral_offset_velocity = forward_offset_m * np.cross(omega, R_ref[:, 2])
            heading_aligned_velocity = windowed_velocity - lateral_offset_velocity
            forward_speed = np.linalg.norm(heading_aligned_velocity)
            if forward_speed >= min_speed_mps:
                forward = heading_aligned_velocity / forward_speed
                right = np.cross(R_ref[:, 1], forward)
                right /= np.linalg.norm(right)
                down = np.cross(forward, right)
                R = np.stack([right, down, forward], axis=1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = translation_spline(query_time)
    return T


def interpolated_future_positions(translation_source, rotation_source, query_time, end_time, fps):
    """Densely interpolated future positions and rotations, one per output frame from
    query_time up to end_time (real seconds), read off the same global translation/rotation
    source as the viewpoint. This is an alternative to the real, native keyframe waypoints (see
    main's --interpolate_future) - smoother-looking, but most of these points are synthesized
    (spline-fit), not measured. Rotations are returned alongside positions because
    project_future_path derives each point's own road-height offset from its own predicted
    down axis (see LOCAL_DOWN_AXIS) rather than a single fixed direction.
    """
    n_steps = max(1, round((end_time - query_time) * fps))
    query_times = np.linspace(query_time, end_time, n_steps + 1)
    positions = translation_source(query_times).astype(np.float64)
    rotations = rotation_source(query_times).as_matrix()
    return positions, rotations


def invert_se3(T):
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def distort_normalized(xn, yn, k1, k2, k3, p1, p2):
    """Brown-Conrady radial + tangential distortion, applied to pinhole-normalized (x/z, y/z)
    coordinates - same convention/order as OpenCV's camera model."""
    r2 = xn ** 2 + yn ** 2
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)
    yd = yn * radial + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn
    return xd, yd


LOCAL_DOWN_AXIS = 1  # column of a world_from_camera rotation matrix giving that camera's own local-down axis (x-right/y-down/z-forward convention), in world coords
NEAR_PLANE_Z = 1.0  # camera-local z (m) below which a point is behind/too close to project sanely


def _interp_at_z(p0, p1, z_target):
    t = (z_target - p0[2]) / (p1[2] - p0[2])
    return p0 + t * (p1 - p0)


def _clip_near_plane(pts, z_near=NEAR_PLANE_Z):
    """Clip an ordered polyline (camera-local coords) against the near plane z=z_near,
    inserting an interpolated point exactly at each crossing instead of just dropping
    sub-threshold points outright. A hard filter alone makes the drawn path jump straight to
    whichever raw sample happens to first clear z_near -- for the sparse real-keyframe path
    (main's default, no --interpolate_future) that can be several real metres further out than
    the vehicle's own current position, making the path look like it starts partway down the
    road instead of extending continuously from the car. Clipping at the exact crossing point
    instead anchors the drawn path at the closest point that's actually geometrically
    projectable, right at the near-plane boundary.
    """
    z = pts[:, 2]
    valid = z > z_near
    if valid.all():
        return pts
    out = []
    for i in range(len(pts)):
        if i > 0 and valid[i] != valid[i - 1]:
            out.append(_interp_at_z(pts[i - 1], pts[i], z_near))
        if valid[i]:
            out.append(pts[i])
    return np.array(out) if out else np.empty((0, 3))


def project_future_path(current_pose, future_positions_world, future_rotations_world, K, dist, camera_height_m):
    """Project the road trajectory - the real camera trajectory's positions, each shifted
    camera_height_m down along THAT point's own local-down axis (future_rotations_world[i]'s
    LOCAL_DOWN_AXIS column, rotated into world coords -- not a single direction fixed for the
    whole trajectory) - into current_pose's own camera view (current_pose may itself be smoothly
    interpolated between real poses - see eval_pose_spline - but future_positions_world are
    always the real measured positions, never fabricated).

    Deriving the down direction per-point like this means it inherits whatever noise is in that
    point's own predicted rotation (the same class of issue eval_pose_spline's heading correction
    exists to work around for yaw specifically -- see its docstring) -- accepted here as the more
    important property is a per-point, not globally-fixed, vertical reference. current_pose's own
    rotation is used for the anchor point (see below), since it already carries its own down
    axis unchanged from the reconstruction even after eval_pose_spline's heading correction (only
    forward/right are recomputed there, not down -- see its docstring).

    future_positions_world's own first point is expected to be current_pose's own position
    (see main's loop, which prepends it) - at zero distance that places the anchor at exactly
    (0, camera_height_m, 0) in the current camera's own frame, i.e. straight down from the lens,
    zero lateral/forward offset. That anchor itself is behind the near plane (see
    _clip_near_plane) and never directly drawn, but clipping the first segment against z_near
    (rather than dropping it) still pins the path's visible start to the closest point on that
    segment that's actually in front of the camera, instead of skipping ahead to wherever the
    next real sample happens to fall.

    Returns an (M, 2) array of pixel coordinates, only for points in front of the camera.
    """
    down_world = future_rotations_world[:, :, LOCAL_DOWN_AXIS]  # (M,3)
    road_points_world = future_positions_world + camera_height_m * down_world  # (M,3)

    ego0_to_egot = invert_se3(current_pose)
    pts_in_egot = (ego0_to_egot[:3, :3] @ road_points_world.T).T + ego0_to_egot[:3, 3]

    pts_in_egot = _clip_near_plane(pts_in_egot)
    if len(pts_in_egot) == 0:
        return np.empty((0, 2))
    z = pts_in_egot[:, 2]

    xn, yn = pts_in_egot[:, 0] / z, pts_in_egot[:, 1] / z
    xd, yd = distort_normalized(xn, yn, *dist)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * xd + cx
    v = fy * yd + cy
    return np.stack([u, v], axis=1)


MARKER_REFERENCE_WIDTH = 1280.0  # the pixel sizes below were tuned against this width (Latvia/NY's)


def draw_path(img_rgba, uv):
    """Draw the projected path onto img_rgba (uv is already in img_rgba's own pixel space).
    Line width and dot radius scale with the image's own resolution (relative to
    MARKER_REFERENCE_WIDTH) - otherwise the same absolute pixel sizes look proportionally
    much thinner/smaller on a higher-resolution clip (e.g. France's 2896px vs. 1280px).
    """
    draw = ImageDraw.Draw(img_rgba, "RGBA")
    px, py = uv[:, 0], uv[:, 1]
    n = len(uv)
    scale = img_rgba.width / MARKER_REFERENCE_WIDTH

    for i in range(n - 1):
        frac = i / max(n - 1, 1)
        alpha = int(255 * (1 - 0.6 * frac))
        width = max(1, round(5 * scale))
        draw.line([(px[i], py[i]), (px[i + 1], py[i + 1])], fill=(60, 170, 255, alpha), width=width)
    for i in range(n):
        if not (0 <= px[i] < img_rgba.width and 0 <= py[i] < img_rgba.height):
            continue
        frac = i / max(n - 1, 1)
        r = (7 - 3 * frac) * scale
        alpha = int(255 * (1 - 0.5 * frac))
        draw.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=(90, 190, 255, alpha))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result_npz", required=True,
                     help="main.py's result.npz -- uses its GPS-fused/road-aligned fused_poses "
                          "field, NOT raw VGGT-SLAM slam_poses, which is in an arbitrary unscaled "
                          "gauge, not real meters (see module docstring)")
    ap.add_argument("--clip_dir", required=True,
                     help="The raw clip folder (contains FRONT_FOLDER/...) - used both to find "
                          "its trip-level FIXED_METADATA.json for real camera calibration, and "
                          "to extract fresh background frames at --fps directly from its video")
    ap.add_argument("--out_path", required=True, help="Output overlay video path (.mp4)")
    ap.add_argument("--horizon_seconds", type=float, default=5.0,
                     help="How many seconds ahead of the current position to draw the path, "
                          "regardless of --fps or how densely real keyframes happen to fall in "
                          "that window")
    ap.add_argument("--camera_height_m", type=float, default=None,
                     help="Camera height above the road (m). If omitted, taken from "
                          "FIXED_METADATA.json's camera_front tz")
    ap.add_argument("--fps", type=float, default=10.0,
                     help="Output video fps. Real keyframes are unevenly spaced in time, so "
                          "between them the viewpoint we project from is smoothly interpolated "
                          "(so the drawn path doesn't wiggle/jump at each one) - the path's own "
                          "waypoints stay the real, native keyframe points, never fabricated. "
                          "Background frames are freshly extracted from the clip's own video at "
                          "this fps too, so the visible scene is genuinely this smooth, not just "
                          "the invisible projection math")
    ap.add_argument("--interpolate_future", action="store_true",
                     help="Plot points off the same global spline as the viewpoint (see "
                          "build_pose_splines) instead of the sparse, real native keyframe "
                          "waypoints. Smooth across the whole horizon (no kink at real sample "
                          "points), but most of the plotted points are then synthesized, not "
                          "measured")
    ap.add_argument("--reconstructed_heading", action="store_true",
                     help="Use the reconstruction's own spline-interpolated rotation as the "
                          "viewpoint's orientation everywhere, instead of the mount-offset-"
                          "corrected direction of travel (see eval_pose_spline's docstring). For "
                          "side-by-side comparison against the default -- this reproduces the "
                          "pre-fix projection error.")
    ap.add_argument("--heading_window_s", type=float, default=1.5,
                     help="Real seconds ahead of each frame to average the direction of travel "
                          "over on a genuine straight, instead of using translation_spline's "
                          "instantaneous derivative (see eval_pose_spline's docstring for why the "
                          "instantaneous derivative alone still visibly wobbles even after the "
                          "dominant rotation-vs-translation error is sidestepped). Shrunk "
                          "adaptively during a real turn -- see --max_turn_deg")
    ap.add_argument("--max_turn_deg", type=float, default=8.0,
                     help="Max real heading change (deg, from the reconstructed yaw rate) to "
                          "ever average direction-of-travel over -- caps --heading_window_s "
                          "shorter during a real turn so it can't 'cut the corner' through it "
                          "(see eval_pose_spline's docstring)")
    ap.add_argument("--min_speed_mps", type=float, default=3.0,
                     help="Below this speed, use the reconstruction's own rotation unmodified "
                          "instead of any direction-of-travel estimate -- at just a few m/s, "
                          "real position noise is a large enough fraction of the per-frame "
                          "displacement to swing the direction estimate by tens of degrees, "
                          "regardless of windowing (see eval_pose_spline's docstring)")
    ap.add_argument("--raw_linear", action="store_true",
                     help="Piecewise-linear position + SLERP rotation directly between the two "
                          "nearest real keyframes (see build_linear_interpolators) instead of the "
                          "smoothing CubicSpline/RotationSpline -- and no heading correction of "
                          "any kind, unlike even --reconstructed_heading (which still uses the "
                          "smoothing rotation spline, just without eval_pose_spline's correction "
                          "on top). This is the reconstruction's (+ GPS fusion's) own output with "
                          "the least possible additional processing, for a raw baseline; makes "
                          "--reconstructed_heading/--heading_window_s/--max_turn_deg/"
                          "--min_speed_mps moot")
    args = ap.parse_args()

    posez = np.load(args.result_npz)
    if "fused_poses" not in posez:
        raise ValueError(f"{args.result_npz} has no fused_poses -- this clip's GPS fusion "
                          "didn't produce a placement (see its gps_fusion_json field)")
    trajectory = posez["fused_poses"]  # (N,4,4), real meters (see --result_npz help)
    frame_ids = posez["frame_ids"]
    extraction_fps = float(posez["extraction_fps"])

    # Submap-boundary frames are logged twice under the same frame_id (the
    # same dedup is needed wherever poses are read from a result.npz) -- keeping
    # both would give a zero time gap between them, which CubicSpline/RotationSpline reject
    # outright (they require strictly increasing x).
    _, first_idx = np.unique(frame_ids, return_index=True)
    first_idx = np.sort(first_idx)
    if len(first_idx) != len(frame_ids):
        print(f"Dropped {len(frame_ids) - len(first_idx)} duplicate-frame_id pose(s) "
              f"(submap boundary nodes)")
    frame_ids = frame_ids[first_idx]
    trajectory = trajectory[first_idx]
    n = trajectory.shape[0]
    times = (frame_ids - 1.0) / extraction_fps  # real elapsed seconds, strictly increasing

    fixed_metadata_path = find_fixed_metadata(args.clip_dir)
    if fixed_metadata_path is None:
        raise FileNotFoundError(f"No FIXED_METADATA.json found above {args.clip_dir}")
    K, dist, meta_height_m, forward_offset_m = load_camera_calibration(fixed_metadata_path)
    camera_height_m = args.camera_height_m if args.camera_height_m is not None else meta_height_m
    print(f"Calibration from {fixed_metadata_path}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"camera_height={camera_height_m:.2f}m forward_mount_offset={forward_offset_m:.2f}m")

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    os.makedirs(out_dir, exist_ok=True)
    bg_tmp_dir = os.path.join(out_dir, "_bg_tmp")
    if os.path.exists(bg_tmp_dir):
        shutil.rmtree(bg_tmp_dir)
    bg_frames = extract_background_frames(args.clip_dir, args.fps, bg_tmp_dir)

    # Sanity check only (not used for correction): K should apply directly to the video's
    # actual resolution. A mismatch here would mean that assumption doesn't hold for this
    # clip and the overlay may be misprojected.
    with open(fixed_metadata_path) as f:
        declared_meta = json.load(f)
    declared_w, declared_h = declared_meta["frame_width"], declared_meta["frame_height"]
    first_img = Image.open(bg_frames[0])
    if abs(first_img.width / declared_w - 1.0) > 0.1:
        print(f"WARNING: the video is {first_img.width}x{first_img.height}, but "
              f"FIXED_METADATA.json declares frame_width/height={declared_w}x{declared_h} "
              f"(fx/fy assumed to apply to the video's own resolution - verify this clip's "
              f"calibration actually matches, e.g. via field_of_view_hor_deg)")

    total_duration_s = times[-1] - times[0]
    total_frames = min(int(total_duration_s * args.fps) + 1, len(bg_frames))
    print(f"Overlaying {total_frames} output frames ({n} real poses spanning "
          f"{total_duration_s:.1f}s, unevenly spaced, {len(bg_frames)} real background frames), "
          f"horizon={args.horizon_seconds}s...")

    if args.raw_linear:
        print("Viewpoint: --raw_linear (piecewise-linear/SLERP, no heading correction)")
        translation_source, rotation_source = build_linear_interpolators(trajectory, times)

        def get_pose(query_time):
            return eval_linear_pose(translation_source, rotation_source, query_time, times)
    else:
        print(f"Viewpoint: {'raw reconstructed rotation' if args.reconstructed_heading else 'mount-offset + windowed-direction-corrected'}")
        translation_source, rotation_source = build_pose_splines(trajectory, times)

        def get_pose(query_time):
            return eval_pose_spline(translation_source, rotation_source, query_time, times,
                                     forward_offset_m,
                                     correct_heading=not args.reconstructed_heading,
                                     heading_window_s=args.heading_window_s,
                                     max_turn_deg=args.max_turn_deg,
                                     min_speed_mps=args.min_speed_mps)

    tmp_dir = os.path.join(out_dir, "_overlay_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    for dense_idx in range(total_frames):
        query_time = times[0] + dense_idx / args.fps
        # index of the last real keyframe at or before query_time (i.e. the keyframe block
        # query_time currently falls within, for the non-interpolated future-path option below).
        i_real = min(max(np.searchsorted(times, query_time, side="right") - 1, 0), n - 1)
        current_pose = get_pose(query_time)

        img = Image.open(bg_frames[dense_idx]).convert("RGBA")

        end_time = min(query_time + args.horizon_seconds, times[-1])
        if args.interpolate_future:
            # Densely interpolated, off the same global position/rotation source as current_pose
            # (the smoothing CubicSpline/RotationSpline, or --raw_linear's piecewise-linear/SLERP
            # interpolators) - smooth across the whole horizon, but most of these points are
            # synthesized, not measured.
            future_positions, future_rotations = interpolated_future_positions(
                translation_source, rotation_source, query_time, end_time, args.fps
            )
        else:
            # Real, native keyframe samples up to end_time, starting from the NEXT real
            # keyframe after query_time (not i_real itself, which is at-or-BEFORE query_time,
            # i.e. spatially behind where the car actually is right now -- see the anchor point
            # prepended below). end is clamped to > i_real + 1 so this is never empty even if
            # horizon_seconds is smaller than the gap to that next keyframe.
            end = max(np.searchsorted(times, end_time, side="right"), i_real + 2)
            future_positions = trajectory[i_real + 1:end, :3, 3]
            future_rotations = trajectory[i_real + 1:end, :3, :3]

        # Anchor the path at the vehicle's own current (interpolated) position/rotation rather
        # than relying on whichever real/interpolated sample happens to be first -- see
        # project_future_path/_clip_near_plane's docstrings for why this is what lets the drawn
        # path extend continuously from the car instead of jumping ahead to the first sample
        # that clears the near plane on its own.
        future_positions = np.concatenate(
            [current_pose[:3, 3][None, :], future_positions.astype(np.float64)], axis=0
        )
        future_rotations = np.concatenate(
            [current_pose[:3, :3][None, :, :], future_rotations.astype(np.float64)], axis=0
        )

        uv = project_future_path(current_pose, future_positions, future_rotations, K, dist, camera_height_m)
        if len(uv) >= 2:
            draw_path(img, uv)

        img.convert("RGB").save(os.path.join(tmp_dir, f"frame_{dense_idx:05d}.jpg"), quality=92)

    print(f"Encoding into {args.out_path} at {args.fps} fps ...")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp_dir, "frame_%05d.jpg"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out_path],
        check=True,
    )
    shutil.rmtree(tmp_dir)
    shutil.rmtree(bg_tmp_dir)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
