"""GPS fusion for VGGT-SLAM trajectories.

VGGT-SLAM's own SL(4) pose graph (vggt_slam/graph.py) has no absolute scale
or metric grounding -- fusing GPS directly into it would mean adding a
partial (position-only) prior factor on the SL4 Lie group's tangent space,
whose parameterization lives in gtsam-develop's compiled SL4 C++ type with
no documented/inspectable convention from Python. Rather than guess at that,
this module fuses GPS as a *second*, decoupled stage: align_trajectory_to_roads
fits an initial scale/heading/translation to the raw GPS fixes, then
iteratively refines placement by map-matching the trajectory's own shape
against the real OSM road network (see vggt_slam/map_match.py) -- both stages
are rigid/similarity transforms applied uniformly to the whole trajectory, so
they can never change its own shape, only where and how it sits in the world.
"""
import os

import numpy as np


def _umeyama_similarity(src, dst):
    """Rotation + translation + scale mapping src (N,2) onto dst (N,2)."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src0, dst0 = src - mu_src, dst - mu_dst
    cov = dst0.T @ src0
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(src.shape[1])
    if np.linalg.det(cov) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    var_src = (src0 ** 2).sum() / src0.shape[0]
    scale = (S * np.diag(D)).sum() / (src0.shape[0] * var_src)
    t = mu_dst - scale * R @ mu_src
    return R, t, scale


def match_gps_to_frames(frame_ids, extraction_fps, gps, max_dt=0.5):
    """Greedy one-to-one matching between keyframes and GPS fixes by elapsed
    time (closest pairs first), so each GPS fix anchors at most one keyframe
    -- GPS fixes arrive slower than keyframes here (~1 Hz vs. denser optical-
    flow keyframes), so a nearest-neighbor-per-keyframe match would otherwise
    pull several consecutive keyframes toward the exact same point, fighting
    the odometry factors between them and distorting the trajectory instead
    of correcting it. Returns dict: frame_index -> (easting, northing, accuracy_m).
    """
    video_time_s = (np.asarray(frame_ids, dtype=float) - 1.0) / extraction_fps
    n_frames, n_gps = len(video_time_s), len(gps["elapsed_s"])
    dt = np.abs(video_time_s[:, None] - gps["elapsed_s"][None, :])  # (n_frames, n_gps)

    pairs = np.argwhere(dt <= max_dt)
    pairs = pairs[np.argsort(dt[pairs[:, 0], pairs[:, 1]])]

    matches = {}
    used_frames, used_gps = set(), set()
    for i, j in pairs:
        i, j = int(i), int(j)
        if i in used_frames or j in used_gps:
            continue
        used_frames.add(i)
        used_gps.add(j)
        matches[i] = (gps["easting"][j], gps["northing"][j], gps["accuracy_m"][j])
    return matches


def _rotation_between_vectors(a, b):
    """The minimal rotation matrix mapping unit vector a onto unit vector b (Rodrigues' formula:
    rotation axis = a x b, angle = angle between them) -- the unique such rotation with no extra
    twist about b, so it disturbs everything else as little as possible."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # a and b point exactly opposite -- any axis perpendicular to a gives a valid 180deg flip.
        perp = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def relevel_to_global_down(rotations, translations):
    """Rotate a whole raw VGGT-SLAM trajectory (about the origin) so the AVERAGE of every
    frame's own local-down axis (rotations[:, :, 1] -- the camera-local +y/down column of that
    frame's own world_from_camera rotation, in the x-right/y-down/z-forward convention used
    throughout this codebase) maps exactly onto canonical world +y.
    """
    global_down = rotations[:, :, 1].mean(axis=0)
    global_down /= np.linalg.norm(global_down)
    R_level = _rotation_between_vectors(global_down, np.array([0.0, 1.0, 0.0]))
    releveled_rot = np.einsum("ij,njk->nik", R_level, rotations)
    releveled_trans = translations @ R_level.T
    return releveled_rot, releveled_trans


def _rigid_align_2d(src, dst):
    """Rotation + translation (no scale) mapping src (N,2) onto dst (N,2), via Kabsch."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src0, dst0 = src - mu_src, dst - mu_dst
    cov = dst0.T @ src0
    U, _, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(cov) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    t = mu_dst - R @ mu_src
    return R, t


def _nearest_on_line(points_xy, line):
    """Each point's own nearest position on `line` (a shapely LineString) -- unlike
    map_match.py's _snap_points_to_line, this has no notion of travel direction (no monotonic
    arc-length clamp): it's used only inside _icp_rigid_to_line's own correspondence step, where
    points_xy is a rigid-transform hypothesis of the trajectory that may still be offset/rotated
    from the target line, so "nearest point so far along the line" isn't yet a meaningful
    constraint the way it is once the trajectory is already roughly correctly placed."""
    from shapely.geometry import Point

    out = np.empty_like(points_xy)
    for k, (x, y) in enumerate(points_xy):
        s = line.project(Point(x, y))
        p = line.interpolate(s)
        out[k] = [p.x, p.y]
    return out


def _icp_rigid_to_line(points_xy, line, R_init, t_init, max_iters, rot_tol_deg, trans_tol_m):
    """Iterative Closest Point: repeatedly re-derive each point's nearest-point correspondence on
    `line` and refit the best rigid (rotation + translation, no scale) transform onto those
    correspondences (Kabsch), starting from (R_init, t_init) -- NOT identity/zero. Scale is
    deliberately never touched here (see align_trajectory_to_roads's docstring for why floating
    scale in this kind of unpaired nearest-point loop is dangerous: it has a spurious global
    optimum at shrinking everything toward the target's local density, regardless of true scale).

    Initialization matters a lot: unlike the rigid-only refinement in align_trajectory_to_roads's
    old single-shot-per-iteration loop, this inner loop iterates many times per outer step, so a
    bad start (e.g. identity rotation) can walk into a wrong local optimum well before ever seeing
    the correspondences that would fix it -- confirmed directly: this was the actual cause of
    catastrophic misalignments on two real clips during validation, fixed by always initializing
    from the caller's own current best-guess transform rather than identity/zero.
    """
    R_cur, t_cur = R_init, t_init
    for _ in range(max_iters):
        transformed = points_xy @ R_cur.T + t_cur
        corr = _nearest_on_line(transformed, line)
        R_step, t_step = _rigid_align_2d(transformed, corr)
        center = transformed.mean(axis=0)
        shift = float(np.linalg.norm((center @ R_step.T + t_step) - center))
        rot_deg = np.degrees(np.arctan2(R_step[1, 0], R_step[0, 0]))
        R_cur = R_step @ R_cur
        t_cur = (R_step @ t_cur[:, None]).flatten() + t_step
        if abs(rot_deg) < rot_tol_deg and shift < trans_tol_m:
            break
    return R_cur, t_cur


def _runs(mask):
    """Contiguous index ranges [start, end) where mask is True."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _detect_primary_leg(translations, rotations, min_reverse_run=5):
    """Detects frames where the trajectory is moving backward relative to its own facing
    direction (dot(displacement, forward-facing direction) < 0), and returns a boolean mask
    (len == len(translations)) covering only the single longest contiguous NON-reversing run --
    the "primary leg" that align_trajectory_to_roads restricts its own fitting stages to.

    This is a genuine reverse-gear maneuver (e.g. a 3-point turn), not merely a wide forward
    U-turn -- confirmed directly on a real clip: ~40 consecutive frames with
    dot(displacement, facing) in [-0.99,-0.85], clearly distinguishable from scattered sign noise
    during near-zero motion elsewhere (hence min_reverse_run: a single flipped-sign frame is
    noise, not a real reversal). map-matching-2's routing model has no way to represent a genuine
    reversal (its own direction-change reward/probability factors penalize a route reversing on
    itself the same way they penalize noisy zig-zag GPS, since neither the model nor the raw GPS
    signal carries a heading input that could tell the two apart) -- fitting against it just
    contaminates the result, so it's better excluded than explained. This is safe because it's
    all one rigid body: a rotation/translation anchored on the reliable leg places the excluded
    segment correctly too, since it's carried along rigidly rather than independently fit.

    If no reversal is detected, the returned mask covers the whole trajectory (a no-op) --
    everything downstream behaves exactly as if this function didn't exist.

    Returns (mask, reverse_spans): reverse_spans is a list of (start_idx, end_idx) frame-index
    ranges that were excluded, for logging.
    """
    xz = translations[:, [0, 2]]
    forward_xz = rotations[:, [0, 2], 2]
    forward_xz = forward_xz / np.linalg.norm(forward_xz, axis=1, keepdims=True)

    disp = np.diff(xz, axis=0)
    disp_norm = np.linalg.norm(disp, axis=1)
    fwd_avg = forward_xz[:-1] + forward_xz[1:]
    fwd_avg = fwd_avg / np.linalg.norm(fwd_avg, axis=1, keepdims=True)
    dot = np.einsum("ij,ij->i", disp, fwd_avg) / np.maximum(disp_norm, 1e-9)

    is_reverse_interval = dot < 0
    reverse_runs = [(s, e) for s, e in _runs(is_reverse_interval) if (e - s) >= min_reverse_run]

    exclude = np.zeros(len(translations), dtype=bool)
    reverse_spans = []
    for s, e in reverse_runs:
        # interval i connects frame i and frame i+1, so a reversing interval run [s, e) excludes
        # frames [s, e] inclusive (both endpoints are part of the reversal).
        exclude[s:e + 1] = True
        reverse_spans.append((s, e + 1))

    legs = _runs(~exclude)
    if not legs:
        return np.ones(len(translations), dtype=bool), reverse_spans

    def leg_length(s, e):
        return float(np.linalg.norm(np.diff(xz[s:e], axis=0), axis=1).sum()) if e - s > 1 else 0.0

    best_s, best_e = max(legs, key=lambda se: leg_length(*se))
    mask = np.zeros(len(translations), dtype=bool)
    mask[best_s:best_e] = True
    return mask, reverse_spans


class _RoadNetwork:
    """The road network around one clip: the crop (centre, radius) is fixed up front, and the
    map-matching-ready network is built on demand per profile (build_network caches within network_dir)."""

    def __init__(self, center_lat, center_lon, buffer_m, to_utm, network_dir, region_pbf_path):
        self.center_lat, self.center_lon, self.buffer_m, self.to_utm = center_lat, center_lon, buffer_m, to_utm
        self.network_dir, self.region_pbf_path = network_dir, region_pbf_path

    def build(self, **kwargs):
        """(prepared_dir, edges) -- see map_match.build_network (kwargs: profile)."""
        from vggt_slam.map_match import build_network
        return build_network(self.center_lat, self.center_lon, self.buffer_m, self.to_utm, self.network_dir,
                             region_pbf_path=self.region_pbf_path, **kwargs)


def _apply_2d_transform(R2, t2, scale, rotations, translations):
    """Scale, then rotate/translate in the ground plane (x, z of the gravity-leveled frame), the whole trajectory."""
    R = np.eye(3)
    R[[0, 0, 2, 2], [0, 2, 0, 2]] = [R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1]]
    t = np.zeros(3)
    t[[0, 2]] = t2
    return np.einsum("ij,njk->nik", R, rotations), scale * (translations @ R.T) + t


def _select_region(center_lat, center_lon, buffer_m, to_utm, network_dir, region_pbf_path, fallback_region_pbf_path):
    """The region extract that has roads near the clip's GPS fixes: the labeled one, else the first fallback
    candidate that does. Returns (_RoadNetwork over that extract, region_used); region_used is None if none had roads
    (the network is then the labeled region's empty crop)."""
    net = _RoadNetwork(center_lat, center_lon, buffer_m, to_utm, network_dir, region_pbf_path)
    _, probe_edges = net.build()
    if probe_edges:
        return net, region_pbf_path
    if not fallback_region_pbf_path:
        return net, None
    candidates = ([fallback_region_pbf_path] if isinstance(fallback_region_pbf_path, (str, os.PathLike))
                  else list(fallback_region_pbf_path))
    for i, candidate_path in enumerate(candidates):
        print(f"align_trajectory_to_roads: no roads found near the GPS fixes in "
              f"{region_pbf_path if i == 0 else candidates[i - 1]} -- retrying "
              f"against fallback region extract {candidate_path} in case this clip is filed "
              f"under the wrong country.")
        candidate_dir = os.path.join(network_dir, f"fallback_region_{i}")
        os.makedirs(candidate_dir, exist_ok=True)
        candidate_net = _RoadNetwork(center_lat, center_lon, buffer_m, to_utm, candidate_dir, candidate_path)
        _, probe_edges = candidate_net.build()
        if probe_edges:
            return candidate_net, candidate_path
    print(f"align_trajectory_to_roads: no roads found in any fallback region extract "
          f"either ({candidates}) -- keeping the labeled region's (empty) crop.")
    return net, None


def _match_gps_route(net, gps_en_leg, gps_elapsed_leg, to_wgs84, to_utm, map_match_fallback_error_m):
    """Step 2: map-match the raw GPS track against the road network. Returns (gps_matched_line, prepared_dir,
    network_profile): the "car" profile unless its result looks bad and the "manual_drivable" network does better."""
    from vggt_slam.map_match import run_map_match

    prepared_dir, _ = net.build()
    network_profile = "car"
    _, dev_car, gps_matched_line = run_map_match(
        gps_en_leg, gps_elapsed_leg, prepared_dir, to_wgs84, to_utm, net.network_dir)
    mean_err_car = float(np.nanmean(dev_car)) if dev_car is not None else np.inf
    if gps_matched_line is None or mean_err_car > map_match_fallback_error_m:
        # the "car" profile's own access filtering can exclude the correct road on real access-restricted
        # streets; only switch to the weaker manual_drivable network if it measurably does better
        print(f"align_trajectory_to_roads: car-profile raw-GPS map-match looked bad (mean "
              f"deviation {mean_err_car:.2f}m{'' if gps_matched_line is not None else ', no match'}"
              f") -- retrying against a manual/drivable-only network in case car-profile's own "
              f"access filtering excluded the correct road.")
        prepared_dir_fb, _ = net.build(profile="manual_drivable")
        _, dev_fb, gps_matched_line_fb = run_map_match(
            gps_en_leg, gps_elapsed_leg, prepared_dir_fb, to_wgs84, to_utm, net.network_dir)
        mean_err_fb = float(np.nanmean(dev_fb)) if dev_fb is not None else np.inf
        if gps_matched_line_fb is not None and mean_err_fb < mean_err_car:
            print(f"align_trajectory_to_roads: manual/drivable-only fallback did better "
                  f"(mean deviation {mean_err_fb:.2f}m vs {mean_err_car:.2f}m) -- using it for "
                  f"the rest of this clip.")
            prepared_dir, gps_matched_line, network_profile = prepared_dir_fb, gps_matched_line_fb, "manual_drivable"
        else:
            print(f"align_trajectory_to_roads: manual/drivable-only fallback didn't improve on "
                  f"the car-profile result (mean deviation "
                  f"{mean_err_fb:.2f}m{'' if gps_matched_line_fb is not None else ', no match'}) "
                  f"-- keeping the car-profile result.")
    return gps_matched_line, prepared_dir, network_profile


def _refine_placement(net, prepared_dir, network_profile, R_cur, t_cur, full_xz_scaled_leg, video_time_s_leg,
                      to_wgs84, to_utm, max_iters, icp_max_iters, rot_tol_deg, trans_tol_m, patience):
    """Step 3: iterate map-match -> ICP on the primary leg from the current placement (R_cur, t_cur).
    Returns (R, t, prepared_dir) of the best-scoring iteration (design notes: memory/gps-fusion-notes.md)."""
    from vggt_slam.map_match import _snap_points_to_line, run_map_match

    best = None  # (mean_err, iteration, R, t)
    no_improve = 0
    # Each iteration is scored BEFORE its own correction is applied, against a fresh map-match of the placement it
    # started from, via _snap_points_to_line (the method behind map_match_error_mean_m); the best-scoring
    # iteration is returned, not the last. Stops when the correction is negligible or the best score has not
    # improved for `patience` iterations.
    for iteration in range(max_iters):
        placed = full_xz_scaled_leg @ R_cur.T + t_cur
        center_en = placed.mean(axis=0)
        prepared_dir, _ = net.build(profile=network_profile)

        traj_path_len = float(np.linalg.norm(np.diff(placed, axis=0), axis=1).sum())
        _, _, cur_matched_line = run_map_match(
            placed, video_time_s_leg, prepared_dir, to_wgs84, to_utm, net.network_dir)
        if cur_matched_line is None:
            print(f"align_trajectory_to_roads: iteration {iteration}: map-matching-2 returned "
                  f"an empty/aborted match on the leg -- stopping refinement here (keeping the "
                  f"best fit found so far).")
            break

        _, dev_pre = _snap_points_to_line(placed, cur_matched_line)
        mean_err = float(np.nanmean(dev_pre))
        improved = best is None or mean_err < best[0]
        if improved:
            best = (mean_err, iteration, R_cur.copy(), t_cur.copy())
            no_improve = 0
        else:
            no_improve += 1

        length_ratio = cur_matched_line.length / traj_path_len
        print(f"align_trajectory_to_roads: iteration {iteration}: center=({center_en[0]:.1f},"
              f"{center_en[1]:.1f}) leg path length={traj_path_len:.1f}m vs. matched route "
              f"length={cur_matched_line.length:.1f}m (ratio={length_ratio:.2f}), mean "
              f"deviation={mean_err:.2f}m{'' if improved else f' (no improvement x{no_improve})'}")

        # `placed` is already this iteration's placement, so ICP starts from identity here (unlike the seed fit)
        R_new, t_new = _icp_rigid_to_line(placed, cur_matched_line, np.eye(2), np.zeros(2),
                                           max_iters=icp_max_iters, rot_tol_deg=rot_tol_deg,
                                           trans_tol_m=trans_tol_m)
        rot_deg = np.degrees(np.arctan2(R_new[1, 0], R_new[0, 0]))
        shift_mag = float(np.linalg.norm(center_en @ R_new.T + t_new - center_en))
        print(f"align_trajectory_to_roads: iteration {iteration}: ICP step "
              f"rotation={rot_deg:.3f}deg, net shift at centroid={shift_mag:.3f}m")

        R_cur = R_new @ R_cur
        t_cur = (R_new @ t_cur[:, None]).flatten() + t_new

        if abs(rot_deg) < rot_tol_deg and shift_mag < trans_tol_m:
            print(f"align_trajectory_to_roads: converged after {iteration + 1} iteration(s)")
            break
        if no_improve >= patience:
            print(f"align_trajectory_to_roads: stopping after {patience} iterations without "
                  f"improvement (best mean deviation {best[0]:.2f}m at iteration {best[1]})")
            break
    else:
        print(f"align_trajectory_to_roads: WARNING reached max_iters={max_iters} without "
              "stopping via either convergence or patience -- the returned trajectory may still "
              "be misaligned; increase max_iters.")

    if best is not None and best[1] != iteration:
        print(f"align_trajectory_to_roads: using outer iteration {best[1]}'s placement "
              f"(mean deviation {best[0]:.2f}m) instead of the last-run iteration {iteration}'s -- "
              f"it scored better.")
    if best is not None:
        _, _, R_cur, t_cur = best
    return R_cur, t_cur, prepared_dir


def align_trajectory_to_roads(rotations, translations, frame_ids, extraction_fps, gps, network_dir,
                               osm_region_pbf_path, max_dt=0.5, max_iters=50, rot_tol_deg=0.02,
                               trans_tol_m=0.02, map_buffer_margin_m=100.0, fallback_region_pbf_path=None,
                               icp_max_iters=30, patience=2, map_match_fallback_error_m=8.0,
                               max_plausible_speed_mps=45.0):
    """Align a VGGT-SLAM trajectory to real-world roads without changing its shape.

    Only rigid/similarity transforms are applied (one uniform scale, then rotation + translation in the
    ground plane), so no angle or curvature of the trajectory can change. Stages (rationale, measured
    behaviour and failure cases: memory/gps-fusion-notes.md):

      1. Umeyama similarity fit of the matched keyframes onto their raw GPS fixes: the only place scale is
         estimated.
      2. Map-match the raw GPS track against the OSM road network, then ICP (rigid, scale fixed) the
         GPS-covered part of the trajectory onto that route, starting from step 1's rotation/translation.
      3. Repeat: map-match the trajectory from its current placement, ICP it onto the match, until the
         correction is negligible or the fit stops improving; the best-scoring iteration wins.

    Steps 2-3 fit only the longest leg that never moves backward (_detect_primary_leg), since map-matching
    cannot represent reversing; the transform still applies to the whole trajectory, which is map-matched
    in full at the end.

    Raises ValueError for the expected precondition failures (fewer than 2 keyframes matched to a GPS fix,
    implausible GPS-derived scale, no roads near the fixes, no usable raw-GPS route); vggt_slam/fusion.py
    maps their messages to manifest statuses.

    osm_region_pbf_path: pre-downloaded OSM region extract (scripts/download_osm_extracts.py) the road
        network is cropped from, locally.
    fallback_region_pbf_path: one path or a list tried in order if the labeled region has no roads near the
        fixes (clips are sometimes filed under the wrong country; vggt_slam/regions.py).
    max_dt: max seconds between a keyframe and a GPS fix to count as matched.
    max_iters / patience: cap on step 3's iterations / stop after this many without improvement.
    icp_max_iters, rot_tol_deg, trans_tol_m: ICP inner-loop cap and the convergence tolerances.
    map_buffer_margin_m: margin added to the network crop radius.
    map_match_fallback_error_m: retry the raw-GPS map-match on the "manual_drivable" network above this mean
        deviation, and keep it only if it does better.
    max_plausible_speed_mps: reject a step-1 fit implying a larger trajectory than this speed allows (before
        the expensive map-matching).

    Returns (fused_rot, fused_trans, scale, matches, matched_line, snapped, region_used, network_profile).
    matched_line / snapped are a fresh map-match of the FULL trajectory at its final placement (None if
    map-matching found no route). region_used is the extract that had roads near the fixes (None if none
    did). network_profile is "car" or "manual_drivable".
    """
    from pyproj import Transformer

    from vggt_slam.map_match import run_map_match

    rotations, translations = relevel_to_global_down(rotations, translations)
    matches = match_gps_to_frames(frame_ids, extraction_fps, gps, max_dt=max_dt)
    if len(matches) < 2:
        raise ValueError(
            f"Only {len(matches)} keyframe(s) matched to a GPS fix within {max_dt}s; "
            "need at least 2 to estimate scale."
        )

    # step 1
    idx = np.array(sorted(matches))
    R2, t2, scale = _umeyama_similarity(translations[idx][:, [0, 2]], np.array([matches[i][:2] for i in idx]))
    _, aligned_trans = _apply_2d_transform(R2, t2, scale, rotations, translations)

    video_time_s = (np.asarray(frame_ids, dtype=float) - 1.0) / extraction_fps
    to_wgs84 = Transformer.from_crs(gps["utm_crs"], "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", gps["utm_crs"], always_xy=True)
    os.makedirs(network_dir, exist_ok=True)

    primary_leg_mask, reverse_spans = _detect_primary_leg(translations, rotations)
    if reverse_spans:
        span_desc = ", ".join(f"{video_time_s[s]:.1f}-{video_time_s[e - 1]:.1f}s" for s, e in reverse_spans)
        print(f"align_trajectory_to_roads: detected {len(reverse_spans)} reverse-gear segment(s) "
              f"at [{span_desc}] -- restricting step 2/seed ICP/outer-loop fitting to the longest "
              f"non-reversing leg ({int(primary_leg_mask.sum())}/{len(primary_leg_mask)} frames); "
              f"the final result is still map-matched against the full trajectory.")

    # Network crop: centred on the raw GPS fixes (always correct, unlike a possibly badly fit trajectory), with a
    # radius covering both the fixes and the whole aligned trajectory, else map-matching can be forced through a
    # wrong nearby road (memory/gps-fusion-notes.md).
    gps_en = np.stack([gps["easting"], gps["northing"]], axis=1)
    gps_center_en = gps_en.mean(axis=0)
    crop_center_lon, crop_center_lat = to_wgs84.transform(*gps_center_en)
    gps_radius_m = float(np.linalg.norm(gps_en - gps_center_en, axis=1).max())
    traj_radius_m = float(np.linalg.norm(aligned_trans[:, [0, 2]] - gps_center_en, axis=1).max())

    # reject an implausible fit before it drives an expensive, doomed OSM crop + map-match
    video_duration_s = float(video_time_s.max() - video_time_s.min())
    max_plausible_radius_m = video_duration_s * max_plausible_speed_mps + map_buffer_margin_m
    if traj_radius_m > max_plausible_radius_m:
        implied_speed_mps = traj_radius_m / video_duration_s if video_duration_s > 0 else float("inf")
        raise ValueError(
            f"implausible GPS-derived scale ({scale:.1f}x) from {len(matches)} matched "
            f"keyframe(s): implies a trajectory radius of {traj_radius_m:.0f}m within a "
            f"{video_duration_s:.1f}s clip (~{implied_speed_mps:.1f} m/s average, vs. the "
            f"{max_plausible_speed_mps:.0f} m/s plausibility bound) -- refusing to map-match."
        )

    net, region_used = _select_region(
        crop_center_lat, crop_center_lon, max(gps_radius_m, traj_radius_m) + map_buffer_margin_m, to_utm,
        network_dir, osm_region_pbf_path, fallback_region_pbf_path)
    if region_used is None:
        # distinct from "raw-GPS map-match unavailable" below: no road network at all near the fixes
        raise ValueError(
            f"no roads found near the GPS fixes in {osm_region_pbf_path} or any fallback region "
            f"({fallback_region_pbf_path})"
        )

    # step 2: restricted to the primary leg's time window (GPS fixes during/after an excluded reversal would
    # contaminate the matched route)
    leg_lo_t = float(video_time_s[primary_leg_mask].min())
    leg_hi_t = float(video_time_s[primary_leg_mask].max())
    gps_leg_mask = (gps["elapsed_s"] >= leg_lo_t) & (gps["elapsed_s"] <= leg_hi_t)
    gps_en_leg, gps_elapsed_leg = gps_en[gps_leg_mask], gps["elapsed_s"][gps_leg_mask]

    extract_mask = np.zeros(len(frame_ids), dtype=bool)
    gps_matched_line = None
    network_profile = "car"
    prepared_dir = None
    if len(gps_en_leg) >= 2:
        gps_matched_line, prepared_dir, network_profile = _match_gps_route(
            net, gps_en_leg, gps_elapsed_leg, to_wgs84, to_utm, map_match_fallback_error_m)
        # keyframes inside the excluded reversal must not seed the ICP fit either
        extract_mask = ((video_time_s >= gps_elapsed_leg.min()) & (video_time_s <= gps_elapsed_leg.max())
                         & primary_leg_mask)

    if gps_matched_line is None or extract_mask.sum() < 10:
        # no weaker fallback method: a real precondition failure must show up in the manifest
        raise ValueError(
            f"raw-GPS map-match unavailable (matched_line="
            f"{'found' if gps_matched_line is not None else 'none'}, "
            f"{int(extract_mask.sum())} points in GPS-covered extract, need >= 10) -- cannot "
            "ICP-refine placement against the real route the GPS traces out."
        )

    print(f"align_trajectory_to_roads: raw-GPS map-match found a route "
          f"({gps_matched_line.length:.1f}m) -- ICP-refining the GPS-covered segment onto it "
          f"before iterating on the full trajectory.")
    # seed ICP from step 1's own (R2, t2), not identity: see _icp_rigid_to_line
    R_cur, t_cur = _icp_rigid_to_line(scale * translations[extract_mask][:, [0, 2]], gps_matched_line, R2, t2,
                                       max_iters=icp_max_iters, rot_tol_deg=rot_tol_deg, trans_tol_m=trans_tol_m)

    # step 3
    R_cur, t_cur, prepared_dir = _refine_placement(
        net, prepared_dir, network_profile, R_cur, t_cur,
        (scale * translations[:, [0, 2]])[primary_leg_mask], video_time_s[primary_leg_mask],
        to_wgs84, to_utm, max_iters, icp_max_iters, rot_tol_deg, trans_tol_m, patience)

    aligned_rot, aligned_trans = _apply_2d_transform(R_cur, t_cur, scale, rotations, translations)

    # fitting only touched the primary leg; this one pass over the FULL trajectory (reversal included) gives
    # the returned route and per-point snaps, and is the only place an excluded segment is map-matched
    snapped, _, matched_line = run_map_match(
        aligned_trans[:, [0, 2]], video_time_s, prepared_dir, to_wgs84, to_utm, net.network_dir)

    return aligned_rot, aligned_trans, scale, matches, matched_line, snapped, region_used, network_profile
