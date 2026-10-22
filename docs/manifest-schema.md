# `manifest.jsonl` row schema

One JSON object per source video, **always with exactly these keys** (null when a key does not
apply; nested objects always have all of their keys too). Written by `batch_infer.py` (first
block) and completed by `scripts/metric_scale/update_manifest_metric_scale.py` (rest), which also normalizes
every row to this schema.

| key | content |
|---|---|
| `clip_id` | full path of the source video |
| `ok` | `batch_infer.py` finished the clip. `false` + `error`/`stage` otherwise (null when `ok`) |
| `seconds` | wall time of the clip's run (null for backfilled rows) |
| `fixed_intrinsics` | the camera calibration of `FIXED_METADATA.json` was used (otherwise self-calibrated). For older rows this was backfilled, see [memory/manifest-and-scale-history.md](../memory/manifest-and-scale-history.md) |
| `gps_fusion` | `status`, `error`, `region_used`, `used_fallback`, `network_profile_used`, `matched_keyframes`, `total_keyframes`, `map_match_error_mean_m`, `map_match_error_max_m`, `map_match_error_n_points`; the last eight are null unless the fusion produced a placement |
| `qc` | `route`, `slam_ok`, `slam_reason`, `n_keyframes`, `slam_valid_until_s`, `scale_ok`, `scale_reason`, `gps_extent_m`, `fused_over_gps_len`, `fused_max_window_mps`, `slam_max_step_ratio`, `version` (null where a metric does not apply, e.g. `slam_valid_until_s` unless SLAM broke) |
| `metric_scale` | `source` (`gps_fusion` / `metric_depth` / null), `scale`, then source-specific: `scale_p25`, `scale_p75`, `scale_rel_iqr`, `n_frames_used`, `model`, `scale_transferred`, `transfer_residual` (depth scale); `gps_fusion_scale` / `depth_scale` (when both exist); `status` (`failed`, `scale_contradicted`, `not_attempted`), `reason`, `error` (no usable scale) |
| `scale_check` | `depth_scale`, `ratio`, `verdict`, `confidence`, `n_frames_used`, `scale_rel_iqr`, `adjudication`; all null unless `qc.route == "needs_scale_check"` |
| `pose_source`, `pose_source_reason` | which poses to use, and why (see the README) |

## Notes on using the fields

- **`metric` poses are an estimate**, about +-10% in scale. Filter on `metric_scale.n_frames_used`
  (few frames = weak evidence) and `metric_scale.scale_rel_iqr` (spread of per-frame estimates; above
  0.25 is noisy).
- **`gps_fusion.status == "ok"` does not guarantee a good scale**; that is what the QC and the
  cross-check are for. How often QC is wrong: [memory/qc-validation.md](../memory/qc-validation.md).
- **`none` / `slam_jump`:** `qc.slam_valid_until_s` is the time of the last keyframe before the first
  jump, i.e. a clean prefix if you want to truncate (not validated).
- **`ego_failed` clips can still carry a `metric_scale`/`metric_poses`** (the first sweep ignored QC).
  Ignore them: `pose_source` already says `none`.
- **Where the scale lives:** `metric_scale` is in the manifest for every clip. Clips that got a depth
  scale also have it inside `result.npz` (`metric_scale`, `metric_poses`, `metric_scale_json`; load
  with `load_result_npz`, fields `metric_scale`/`metric_poses`/`metric_scale_info`). For plain `fused`
  clips the scale is only in the manifest (fused/slam step-length ratio).
- **Depth scale of a clip** was computed on a SLAM rerun; where its gauge differed from the stored
  run the scale is transferred through a similarity alignment (`metric_scale.scale_transferred`); where the
  reruns genuinely diverged there is no scale (`rerun_poses_mismatch`).
- **`static` clips** have `slam_poses` from < 4 keyframes: treat the trajectory as stationary
  (`qc.gps_extent_m` is the raw GPS span).
- Beware GPS timestamps: the GPS-vs-video time offset can be several seconds in a few percent of clips
  (poses are fine; see [memory/qc-validation.md](../memory/qc-validation.md)).
