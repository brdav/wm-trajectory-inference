# GPS fusion notes

## Failure-status audit (why the statuses exist)
In one `manifest.jsonl` audit, ~72% of all "failed" `gps_fusion` records were really one of two
data-quality cases, which is why `batch_infer`/`vggt_slam/fusion.py` give them their own structured status
instead of the generic "failed" bucket:
- `no_real_gps_fixes` (every row of the sensor log is 'na'): 70.9% of the failures, nothing to fix in code.
- `insufficient_gps_matches` (< 2 keyframes within max_dt of a fix): 20.2%.
- `gps_log_too_short`: fewer than 2 distinct timestamps; 268 clips, each with exactly 1 sensor-log
  row (a natix logging gap). Previously an opaque `numpy.linalg.LinAlgError` ("SVD did not converge").
- `no_raw_gps_route_*`: `matched_line=found` means a route was matched but too few keyframes fall
  in its time window (sparse GPS: typically only 2-16 real fixes); `none` means no plausible route
  through the road network for the raw trace: a mix of sparse clips and a smaller tail with plenty
  of good GPS that still cannot be routed (one clip: 53/53 real fixes, realistic speed, 13 m
  accuracy, no match), a genuine map-matching limitation.

## Crop radius case study
The road-network crop's radius must cover wherever the whole aligned trajectory goes, not just the
spread of the raw GPS fixes. On a clip whose GPS covered only ~47% of its duration, a GPS-derived
radius left the real continuation of the road (way 207397504) outside the crop while an unrelated
nearby loop (way 207397505) was the only through-route; map-matching-2 routed the trajectory
through it (mean deviation 38.9 m, max 123.8 m). With the larger radius: 11.5 m / 24.2 m. Hence
radius = max(GPS radius, distance from the GPS centre to the farthest aligned point); a bad
Umeyama fit only ever enlarges the crop.

## ICP iteration scoring
Iterations are scored with `_snap_points_to_line`, the same monotonic-snap method that produces
`map_match_error_mean_m` in the manifest, not with `_icp_rigid_to_line`'s nearest-point
correspondence, which can disagree at loops/self-intersections (one clip: 3.25 m by nearest-point
vs 7.05 m as reported).

## Design notes for `align_trajectory_to_roads` (moved out of its docstring)

- **Rigid/similarity only.** Neither stage can change an angle or the curvature of the trajectory. If a turn is
  rounded in the raw SLAM reconstruction it stays rounded, just correctly placed and oriented.
- **Step 1 (Umeyama)** is the only place scale is estimated. A pairwise-distance-weighted regression was tried
  against plain Umeyama on 23 real clips: the two agreed to within ~0.1-0.4%, so Umeyama's scale was never the
  problem and the alternative was not adopted.
- **Steps 2-3 replaced a simpler design** (best-fit rigid transform, Kabsch, directly between the trajectory and
  its own map-matched snapped points each iteration). It is gone rather than kept as a fallback: if step 2's
  raw-GPS map-match fails, the function raises, so a real precondition failure shows up as a gps_fusion failure
  in the manifest instead of a quietly worse fit.
- **ICP does not estimate scale.** Letting scale float inside a nearest-point loop has a spurious optimum at
  shrinking the trajectory toward the target's local point density: it produced a trajectory ~44% shorter than
  the raw GPS implied, despite a lower residual. Scale stays at step 1's estimate.
- **Primary leg.** map-matching-2's routing model cannot represent reverse-gear manoeuvres (it penalizes a route
  reversing on itself like noisy zig-zag GPS), so fitting against one contaminates the result.
  `_detect_primary_leg` restricts steps 2-3 to the longest leg that never moves backward relative to its facing
  direction. Safe because it is one rigid body. On a real out-and-back clip excluding the reverse segment moved
  the raw-GPS-route-length / path-length ratio of the primary leg from 0.76 to 1.01 (the return leg double-counts
  in raw GPS path length, but a map-matched route represents the road once). The final result is still
  map-matched against the FULL trajectory. The GPS fixes used in step 2 are restricted to the leg's time window
  too; otherwise the out-and-back points were explained as one shortcut route ("matched route shorter than the
  raw GPS path").
- **Step 3 is a fixed-point iteration, not a minimization.** On one real clip the loop converged cleanly
  (rotation and shift near zero) to a mean deviation 2.7x worse than an earlier iteration. Hence every iteration
  is scored before its own correction, against a fresh map-match of the placement it started from, and the best
  scoring one is returned. This makes the seed ICP result a fair candidate ("iteration 0") and needs no closing
  pass. It stops when the correction is negligible or the best score has not improved for `patience` iterations;
  the second condition is needed because "the correction became tiny" and "the fit is good" differ.
- **manual_drivable fallback.** On two real clips the "car" profile's own undocumented access filtering excluded
  the correct road (an access-restricted lane the GPS trace needed). If the raw-GPS map-match on the "car"
  network has mean deviation above `map_match_fallback_error_m` (8 m), the "manual_drivable" network is tried and
  kept only if it does better (it has no access-restriction awareness at all). The winner is reused for the rest
  of the clip.
- **Plausibility bound (`max_plausible_speed_mps` = 45 m/s, deliberately generous).** One clip matched only 2
  keyframes, both from a ~15 s stretch where the vehicle was stopped behind a bus (near-identical frames), so the
  correspondences were ~5 cm apart in SLAM units while the GPS fixes drifted ~119 m (urban multipath). The 2-point
  fit trusted the ratio: ~1265x scale, a trajectory extent of nearly 6.8 km in a 60 s clip, and a map-matching crop
  1-2 orders of magnitude too large that took minutes. The bound catches fits wrong by orders of magnitude without
  second-guessing a merely fast one.
