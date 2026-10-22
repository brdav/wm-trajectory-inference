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
