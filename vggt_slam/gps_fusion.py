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


def align_trajectory_to_roads(rotations, translations, frame_ids, extraction_fps, gps, network_dir,
                               osm_region_pbf_path, max_dt=0.5, max_iters=50, rot_tol_deg=0.02,
                               trans_tol_m=0.02, map_buffer_margin_m=100.0, fallback_region_pbf_path=None,
                               icp_max_iters=30, patience=2, map_match_fallback_error_m=8.0,
                               max_plausible_speed_mps=45.0):
    """Align a VGGT-SLAM trajectory to real-world roads without ever touching its own shape.

    All stages use only rigid/similarity transforms (rotation + translation, with a single
    one-time uniform scale) -- neither can change a single angle or introduce/remove curvature
    anywhere in the trajectory, by construction:

      1. One-shot scale + rotation + translation fit (Umeyama) of the trajectory's matched
         keyframes onto their *raw* GPS fixes -- this is the only place scale is ever estimated
         (never revisited afterward), from the true time-based keyframe<->GPS-fix correspondence.
         (A pairwise-distance-weighted-regression alternative was tried and validated against
         plain Umeyama on 23 real clips: the two agreed to within ~0.1-0.4% everywhere, i.e.
         Umeyama's scale was never actually the problem here, so it was not adopted.)

      2. Map-match the *raw GPS track itself* (not the trajectory) against the real OSM road
         network, then ICP (rigid, scale fixed, see _icp_rigid_to_line) the step-1-aligned
         trajectory's own GPS-time-covered segment onto that raw-GPS-implied route, initialized
         from step 1's own (rotation, translation) -- not identity/zero, see _icp_rigid_to_line's
         own docstring for why that initialization is load-bearing. This fixes heading/placement
         using the *unpaired* shape of the real route the GPS points trace out, rather than only
         the handful of paired keyframe<->fix correspondences step 1 has -- correspondence-free
         matching is what lets this succeed even where individual GPS fixes are noisy, as long as
         the route they collectively trace is recognizable.

      3. Iteratively map-match the trajectory's own shape (restricted to the primary leg -- see
         below) against the road network from its current placement, then ICP the leg onto that
         match (again rigid, scale fixed, again initialized from the current running transform),
         and repeat until either the correction is negligible or the best-seen fit stops
         improving (see the loop's own comment for why both checks matter). A better placement
         usually yields a better map-match, which yields a still-better placement.

    Before any of this, _detect_primary_leg restricts steps 2-3's own fitting to the single
    longest leg of the trajectory that never moves backward relative to its own facing direction
    -- a genuine reverse-gear maneuver (e.g. a 3-point turn), which map-matching-2's routing model
    has no way to represent (its direction-change reward/probability factors penalize a route
    reversing on itself the same way they penalize noisy zig-zag GPS), so fitting against it only
    ever contaminates the result. This is safe because it's all one rigid body: a transform
    anchored on the reliable leg still places the excluded segment correctly, carried along
    rigidly rather than independently fit. If no reversal is detected the mask covers the whole
    trajectory (a no-op). Confirmed directly on a real clip: excluding a detected reverse segment
    dropped the primary leg's own raw-GPS-route-length-vs-path-length ratio from 0.76 (the
    out-and-back trip's raw GPS path length double-counts the return leg, while a map-matched
    route only ever represents the road once) to 1.01. The final result below is still
    map-matched against the FULL trajectory (reversal included) -- only the fitting stages are
    leg-restricted, not the reported one.

    Steps 2-3 replace an earlier, simpler design that fit a single best-fit rigid transform
    (Kabsch) directly between the trajectory and its own map-matched *snapped points* each
    iteration. That design is gone rather than kept as a fallback: if step 2's raw-GPS map-match
    fails (no usable route, or too little of the trajectory overlaps the GPS time range), this
    function raises rather than silently degrading to the weaker method -- a real precondition
    failure should show up as a genuine gps_fusion failure in the manifest, not a quietly
    worse fit. ICP-with-unpaired-correspondence was deliberately NOT extended to also estimate
    scale: floating scale inside a nearest-point loop has a spurious global optimum at shrinking
    the whole trajectory toward the target's local point density regardless of the true scale --
    confirmed directly during validation (letting scale float here produced a trajectory ~44%
    shorter than the raw GPS implied, despite a *lower* residual-to-target metric, since ICP's own
    loss doesn't penalize that collapse). Scale is fixed at step 1's estimate throughout.

    Step 3's own outer loop is not guaranteed to improve monotonically -- each iteration re-matches
    the leg against the road network from its current placement, so it's a fixed-point iteration
    over a target route that can itself shift between iterations, not a plain minimization.
    Confirmed directly on a real clip: the loop converged cleanly (rotation/shift both settled
    near zero) but to a mean deviation 2.7x worse than an earlier iteration already had. To guard
    against this, every iteration is scored -- BEFORE its own correction is applied, against a
    FRESH map-match of whatever placement it started from -- and the best-scoring iteration is
    what's actually returned, not unconditionally the last one run. Scoring pre-correction is what
    lets the seed ICP's own result be a fair candidate too, with no special-casing: "iteration 0"
    is just the seed's placement, map-matched fresh over the whole leg instead of only ever being
    scored on the smaller GPS-covered subset it was fit against.

    The loop stops on two independent conditions, checked after each correction: the correction
    itself is negligible (nothing left to gain by continuing), or the best-seen score hasn't
    improved for `patience` iterations in a row -- necessary because, per the non-monotonicity
    above, "the correction became tiny" and "the fit is good" are different things; tying
    termination to the metric actually being reported catches a converged-to-a-worse-optimum case
    directly, instead of only trusting the correction's own magnitude. This also means the
    returned candidate is always one that was scored inside the loop's own normal course -- no
    separate closing pass is needed to capture "the truly final state" as its own candidate.

    This corrects *global* placement and heading (a single sparse Umeyama fit's rotation is a much
    weaker estimate than repeatedly matching the trajectory's full shape against real roads), but
    it cannot straighten out a turn that's genuinely rounded in VGGT-SLAM's own raw reconstruction
    -- similarity/rigid transforms preserve every angle exactly, so if that's the shape VGGT-SLAM
    actually produced, it's the shape this will still show, just correctly placed and oriented.

    osm_region_pbf_path: pre-downloaded whole-region OSM extract (a Geofabrik country/state
    .osm.pbf -- see scripts/download_osm_extracts.py) that build_network crops the network
    locally out of, via its spatial index. No network access at all.

    fallback_region_pbf_path: one path, or a list of candidate paths tried in order, instead of
    osm_region_pbf_path, if the GPS fixes' own crop finds zero roads there -- this dataset files
    some clips under the wrong country folder (e.g. an "Austria" clip whose GPS is actually near
    Stolac, Bosnia and Herzegovina, right on the Croatia border), so the clip's labeled region can
    genuinely have no nearby roads even though the GPS/trajectory themselves are fine. Each
    candidate is tried until one yields a non-empty crop (Geofabrik's per-country extracts can
    leave gaps right at a border, so the "obviously correct" neighboring country isn't always the
    one that actually has the nearby ways -- see vggt_slam/regions.py's FALLBACK_REGIONS).
    None/empty disables this (no fallback attempted).

    icp_max_iters: cap on _icp_rigid_to_line's own inner correspondence-refitting loop (steps 2-3
    above) -- separate from max_iters, which caps the outer map-match/ICP loop (step 3).

    patience: stop step 3's outer loop after this many consecutive iterations without an improved
    (lower mean deviation) candidate -- see the loop's own comment for why this is necessary, not
    just an efficiency knob.

    map_match_fallback_error_m: if step 2's raw-GPS map-match (against the default "car"-profile
    network) comes back with mean deviation above this, build_network is retried with
    profile="manual_drivable" (see its own docstring) and the raw-GPS map-match repeated against
    that network too -- confirmed on two real clips where "car"'s own undocumented access
    filtering excluded the correct road entirely (a real access-restricted lane the GPS trace
    still needed), forcing a detour that this catches. The fallback's own result is only actually
    used if its mean deviation is lower than the "car"-profile result's -- otherwise the flagged
    "car"-profile result is kept, since "manual_drivable" is weaker (no access-restriction
    awareness at all) and a bad primary result doesn't guarantee the fallback is the fix. Whichever
    network wins this comparison is reused for the rest of this clip's processing (step 3's outer
    loop and the final full-trajectory match below), not re-decided every iteration.

    max_plausible_speed_mps: sanity bound on the initial Umeyama fit, checked right after it's
    computed (before the expensive OSM crop/map-matching below ever runs): if scaling the
    aligned trajectory implies traveling further from the GPS center than this speed times the
    clip's own duration would allow, the fit is rejected outright (ValueError) rather than fed
    into map-matching. Confirmed on a real clip: only 2 keyframes matched (both from a ~15s
    stretch where the vehicle was actually stopped behind a bus, confirmed by inspecting the
    video -- near-identical frames throughout), so those 2 correspondences sat only ~5cm apart in
    SLAM's own ungauged units while the GPS fixes for that same window drifted ~119m apart (a
    urban multipath/canyon artifact, not real motion -- the GPS track itself looked smooth, with
    no jumps, so nothing upstream flags it as bad data). A 2-point similarity fit has zero
    redundancy to catch that mismatch: it just trusts the ratio, producing a ~1265x scale that
    projected the whole 50-keyframe trajectory's own modest ~4.5-unit extent out to nearly 6.8km
    -- physically impossible for a 60s clip, and (confirmed) an expensive dead end: map-matching
    was then given a crop 1-2 orders of magnitude larger than any real clip needs, taking minutes
    against a dense urban OSM region for a fit that could never place the trajectory correctly
    anyway. 45 m/s (~162 km/h) is a deliberately generous highway-speed bound so a genuinely fast,
    correctly-fit clip is never rejected -- this is meant to catch fits that are wrong by
    orders of magnitude, not to second-guess a merely-fast one.

    Returns (fused_rot, fused_trans, scale, matches, matched_line, snapped, region_used,
    network_profile):
    matched_line/snapped are always a fresh map-match of the FULL trajectory (reversal segment
    included) against the actually-returned fused_trans -- fitting above only ever touches the
    primary leg, so this final pass is not optional bookkeeping the way it might look; it's the
    only place the excluded segment (if any) and the reported route/snapped positions ever get
    computed together. Both None if map-matching-2 never found a match on the full trajectory.
    Saved by main.py alongside fused_rot/fused_trans so the road alignment can be plotted
    without re-running this whole function.
    region_used is whichever of osm_region_pbf_path / fallback_region_pbf_path actually had roads
    near the GPS fixes (None if none did) -- main.py records this in the manifest.
    network_profile is "car" or "manual_drivable" -- whichever build_network profile actually won
    (see map_match_fallback_error_m above); callers can record this in the manifest to see how
    often the fallback was actually needed across the dataset.
    """
    from pyproj import Transformer

    from vggt_slam.map_match import _snap_points_to_line, build_network, run_map_match

    n = len(frame_ids)
    rotations, translations = relevel_to_global_down(rotations, translations)
    matches = match_gps_to_frames(frame_ids, extraction_fps, gps, max_dt=max_dt)
    if len(matches) < 2:
        raise ValueError(
            f"Only {len(matches)} keyframe(s) matched to a GPS fix within {max_dt}s; "
            "need at least 2 to estimate scale."
        )

    idx = np.array(sorted(matches))
    src_xz = translations[idx][:, [0, 2]]
    dst_en = np.array([matches[i][:2] for i in idx])
    R2, t2, scale = _umeyama_similarity(src_xz, dst_en)

    R_align = np.eye(3)
    R_align[[0, 0, 2, 2], [0, 2, 0, 2]] = [R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1]]
    t_align = np.zeros(3)
    t_align[[0, 2]] = t2

    aligned_rot = np.einsum("ij,njk->nik", R_align, rotations)
    aligned_trans = scale * (translations @ R_align.T) + t_align

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

    # The network crop's CENTER comes from the raw GPS fixes' own bounding box, NOT from the
    # aligned SLAM trajectory: aligned_trans depends on the Umeyama fit above, which can be
    # poorly conditioned with few/noisy correspondences and land nowhere near the real location
    # -- while the GPS fixes themselves are ground truth and always correct. Centering on them
    # guarantees the query covers the right place regardless of how well the fit converges.
    #
    # The crop's RADIUS, however, must cover wherever aligned_trans (the whole trajectory, not just its
    # GPS-matched keyframes) goes, or map-matching can be forced through a wrong nearby road (case study:
    # memory/gps-fusion-notes.md). So: take the GPS radius or the distance from the GPS center to the
    # farthest point of the (possibly still poorly-placed) aligned trajectory, whichever is larger -- a bad
    # Umeyama fit only ever makes this radius bigger (a needlessly large but still correctly-centered area),
    # never smaller/wrong.
    gps_en = np.stack([gps["easting"], gps["northing"]], axis=1)
    gps_center_en = gps_en.mean(axis=0)
    crop_center_lon, crop_center_lat = to_wgs84.transform(*gps_center_en)
    gps_radius_m = float(np.linalg.norm(gps_en - gps_center_en, axis=1).max())
    traj_radius_m = float(np.linalg.norm(aligned_trans[:, [0, 2]] - gps_center_en, axis=1).max())

    # Reject an implausible fit here, before it drives an expensive (and doomed) OSM crop +
    # map-match below -- see max_plausible_speed_mps's own docstring for the concrete clip this
    # is modeled on (a ~1265x scale from a 2-point fit during a real stop, GPS drifting via urban
    # multipath). video_time_s's own span is used rather than a hardcoded clip length so this
    # still makes sense if this function is ever called on a longer/shorter recording.
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

    crop_buffer_m = max(gps_radius_m, traj_radius_m) + map_buffer_margin_m

    _, probe_edges = build_network(crop_center_lat, crop_center_lon, crop_buffer_m, to_utm,
                                    network_dir, region_pbf_path=osm_region_pbf_path)
    # Which region extract actually had roads near this clip's GPS fixes -- None means neither the
    # labeled region nor any fallback candidate did. Reported back to the caller (see main.py) so
    # it ends up in the manifest as structured, queryable data instead of only a log line: at
    # dataset scale, "was the fallback ever actually needed/used, and did it even help" needs to
    # be answerable without grepping every clip's log.
    region_used = osm_region_pbf_path if probe_edges else None
    if not probe_edges and fallback_region_pbf_path:
        fallback_candidates = (
            [fallback_region_pbf_path] if isinstance(fallback_region_pbf_path, (str, os.PathLike))
            else list(fallback_region_pbf_path)
        )
        for i, candidate_path in enumerate(fallback_candidates):
            print(f"align_trajectory_to_roads: no roads found near the GPS fixes in "
                  f"{osm_region_pbf_path if i == 0 else fallback_candidates[i - 1]} -- retrying "
                  f"against fallback region extract {candidate_path} in case this clip is filed "
                  f"under the wrong country.")
            candidate_dir = os.path.join(network_dir, f"fallback_region_{i}")
            os.makedirs(candidate_dir, exist_ok=True)
            _, probe_edges = build_network(crop_center_lat, crop_center_lon, crop_buffer_m, to_utm,
                                            candidate_dir, region_pbf_path=candidate_path)
            if probe_edges:
                network_dir, osm_region_pbf_path = candidate_dir, candidate_path
                region_used = candidate_path
                break
        else:
            print(f"align_trajectory_to_roads: no roads found in any fallback region extract "
                  f"either ({fallback_candidates}) -- keeping the labeled region's (empty) crop.")

    if region_used is None:
        # Distinct from step 2's "raw-GPS map-match unavailable" below: this is "there is no road
        # network at all near the GPS fixes, in the labeled region or any fallback" -- a different
        # root cause (bad/missing OSM coverage for this location) that a caller should be able to
        # tell apart from "roads exist but the GPS track itself couldn't be routed through them"
        # (see scripts/tools/refuse_gps.py's own classification of these two ValueErrors).
        raise ValueError(
            f"no roads found near the GPS fixes in {osm_region_pbf_path} or any fallback region "
            f"({fallback_region_pbf_path})"
        )

    prepared_dir, _ = build_network(crop_center_lat, crop_center_lon, crop_buffer_m, to_utm, network_dir,
                                     region_pbf_path=osm_region_pbf_path)

    # Step 2: map-match the raw GPS track itself (not the trajectory) -- an unpaired shape to ICP
    # the trajectory's GPS-covered segment onto, using far more of the GPS signal (the whole route
    # it traces) than step 1's handful of paired keyframe<->fix correspondences alone. Restricted
    # to primary_leg_mask's own time window: a GPS fix recorded during/after an excluded reversal
    # would contaminate gps_matched_line the same way it would contaminate the trajectory fit
    # (confirmed directly: this is exactly what produced a "matched route shorter than the raw GPS
    # path" symptom on a real clip -- the out-and-back GPS points were being explained as one
    # shortcut route instead of the reversal being excluded).
    leg_lo_t = float(video_time_s[primary_leg_mask].min())
    leg_hi_t = float(video_time_s[primary_leg_mask].max())
    gps_leg_mask = (gps["elapsed_s"] >= leg_lo_t) & (gps["elapsed_s"] <= leg_hi_t)
    gps_en_leg, gps_elapsed_leg = gps_en[gps_leg_mask], gps["elapsed_s"][gps_leg_mask]

    extract_mask = np.zeros(len(frame_ids), dtype=bool)
    gps_matched_line = None
    network_profile = "car"
    if len(gps_en_leg) >= 2:
        _, dev_car, gps_matched_line = run_map_match(
            gps_en_leg, gps_elapsed_leg, prepared_dir, to_wgs84, to_utm, network_dir
        )
        mean_err_car = float(np.nanmean(dev_car)) if dev_car is not None else np.inf
        if gps_matched_line is None or mean_err_car > map_match_fallback_error_m:
            # See map_match_fallback_error_m's own docstring above and build_network's
            # profile="manual_drivable" docstring: the "car" profile's own undocumented access
            # filtering can exclude the correct road entirely on real access-restricted streets.
            # Only actually switches to this weaker (no access-restriction awareness) network if
            # it measurably does better -- a bad "car"-profile result doesn't guarantee this is
            # the fix, so the flagged result is kept otherwise.
            print(f"align_trajectory_to_roads: car-profile raw-GPS map-match looked bad (mean "
                  f"deviation {mean_err_car:.2f}m{'' if gps_matched_line is not None else ', no match'}"
                  f") -- retrying against a manual/drivable-only network in case car-profile's own "
                  f"access filtering excluded the correct road.")
            prepared_dir_fb, _ = build_network(crop_center_lat, crop_center_lon, crop_buffer_m, to_utm,
                                                network_dir, region_pbf_path=osm_region_pbf_path,
                                                profile="manual_drivable")
            _, dev_fb, gps_matched_line_fb = run_map_match(
                gps_en_leg, gps_elapsed_leg, prepared_dir_fb, to_wgs84, to_utm, network_dir
            )
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
        # Also intersected with primary_leg_mask: a keyframe inside the excluded reversal
        # shouldn't seed the ICP fit either, even if it happens to fall within the leg-restricted
        # GPS window's own time range.
        extract_mask = ((video_time_s >= gps_elapsed_leg.min()) & (video_time_s <= gps_elapsed_leg.max())
                         & primary_leg_mask)

    if gps_matched_line is None or extract_mask.sum() < 10:
        # No fallback: an earlier design (fit a single best-fit rigid transform directly between
        # the trajectory and its own map-matched snapped points each iteration, no raw-GPS
        # matching at all) could still produce a placement here, but silently falling back to a
        # weaker method hides a real precondition failure -- treat it the same as the other
        # precondition checks in this function (e.g. "fewer than 2 matched keyframes" above): fail
        # loudly, with the reason logged, so it shows up as a genuine gps_fusion failure in the
        # manifest rather than a silently-degraded fit.
        raise ValueError(
            f"raw-GPS map-match unavailable (matched_line="
            f"{'found' if gps_matched_line is not None else 'none'}, "
            f"{int(extract_mask.sum())} points in GPS-covered extract, need >= 10) -- cannot "
            "ICP-refine placement against the real route the GPS traces out."
        )

    print(f"align_trajectory_to_roads: raw-GPS map-match found a route "
          f"({gps_matched_line.length:.1f}m) -- ICP-refining the GPS-covered segment onto it "
          f"before iterating on the full trajectory.")
    extract_xz = scale * translations[extract_mask][:, [0, 2]]
    # Initialized from step 1's own (R2, t2) -- NOT identity/zero, see _icp_rigid_to_line's own
    # docstring for why that matters.
    R_cur, t_cur = _icp_rigid_to_line(extract_xz, gps_matched_line, R2, t2,
                                       max_iters=icp_max_iters, rot_tol_deg=rot_tol_deg,
                                       trans_tol_m=trans_tol_m)

    full_xz_scaled = scale * translations[:, [0, 2]]
    full_xz_scaled_leg = full_xz_scaled[primary_leg_mask]
    video_time_s_leg = video_time_s[primary_leg_mask]

    best = None  # (mean_err, iteration, R, t)
    no_improve = 0
    # Scored BEFORE each iteration's own correction is applied, against a FRESH leg-restricted
    # map-match of whatever placement it started from -- this is what makes the seed a fair
    # candidate at all: "iteration 0"'s score is the seed's own (R2,t2)-seeded ICP result, extended
    # to the full leg and map-matched fresh, instead of only ever being scored on the smaller
    # GPS-covered subset it was fit against (an easier target it could never be fairly compared
    # against otherwise). Scored via _snap_points_to_line -- the SAME monotonic-snap method
    # run_map_match's own snapped/dev use and what map_match_error_mean_m in the manifest is
    # computed from -- rather than _icp_rigid_to_line's own nearest-point correspondence
    # (_nearest_on_line), which can disagree substantially at loops/self-intersections
    # (memory/gps-fusion-notes.md).
    #
    # Terminates on two independent conditions, checked after each correction: the correction
    # itself is negligible (nothing left to gain by continuing), or the best-seen score hasn't
    # improved for `patience` iterations in a row -- necessary, not cosmetic, per this function's
    # own docstring: the loop can converge cleanly (correction shrinking to ~0) to a WORSE fixed
    # point than an earlier iteration, so tying termination to the metric actually being reported
    # catches that directly. This also removes any need for a separate closing pass after the loop
    # to capture "the truly final state" as its own candidate: whichever iteration is actually best
    # already got scored in the loop's own normal course, since scoring happens before that
    # iteration's own correction is even computed.
    for iteration in range(max_iters):
        placed = full_xz_scaled_leg @ R_cur.T + t_cur
        center_en = placed.mean(axis=0)
        prepared_dir, _ = build_network(crop_center_lat, crop_center_lon, crop_buffer_m, to_utm,
                                         network_dir, region_pbf_path=osm_region_pbf_path,
                                         profile=network_profile)

        traj_path_len = float(np.linalg.norm(np.diff(placed, axis=0), axis=1).sum())
        _, _, cur_matched_line = run_map_match(
            placed, video_time_s_leg, prepared_dir, to_wgs84, to_utm, network_dir
        )
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

        # Re-initialized from identity/zero each iteration (unlike step 2's seed ICP call):
        # `placed` is already this iteration's current best-guess placement, so the
        # correspondence search starts from the right neighborhood regardless.
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

    R_final = np.eye(3)
    R_final[[0, 0, 2, 2], [0, 2, 0, 2]] = [R_cur[0, 0], R_cur[0, 1], R_cur[1, 0], R_cur[1, 1]]
    t_final = np.zeros(3)
    t_final[[0, 2]] = t_cur
    aligned_rot = np.einsum("ij,njk->nik", R_final, rotations)
    aligned_trans = scale * (translations @ R_final.T) + t_final

    # Fitting above only ever touched the primary leg (possibly a strict subset of the full
    # trajectory, if a reversal was excluded) -- this is the one map-matching pass that always
    # covers the FULL trajectory, so the returned matched_line/snapped correspond to what's
    # actually returned rather than to the leg alone. Not optional bookkeeping: it's the only
    # place the excluded segment (if any) is ever map-matched at all. "We can still do
    # map-matching at the end again" -- once (R_cur, t_cur) is chosen from the leg-restricted
    # candidates above, it applies correctly to the whole trajectory regardless, since it's all
    # one rigid body: the excluded segment is carried along rigidly rather than independently fit.
    final_traj_en = aligned_trans[:, [0, 2]]
    snapped, _, matched_line = run_map_match(
        final_traj_en, video_time_s, prepared_dir, to_wgs84, to_utm, network_dir
    )

    return aligned_rot, aligned_trans, scale, matches, matched_line, snapped, region_used, network_profile
