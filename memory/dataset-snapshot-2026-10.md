# Dataset snapshot, 2026-10-20

Natix dataset, 120,026 source videos (one manifest row each). Counts by `pose_source`:

| `pose_source` | clips | share | `pose_source_reason` |
|---|---|---|---|
| `fused` | 109,230 | 91.0% | `gps_fusion` 105,063; after a cross-check: `gps_confirmed_by_depth` 3,161, `gps_ambiguous_vs_depth` 862, `gps_length_supports_fused` 112, `gps_unchecked` 32 |
| `metric` | 9,016 | 7.5% | `no_fused_poses` 8,681, `implausible_speed` 217, `longer_than_gps` 68, `no_route_match` 50 |
| `static` | 996 | 0.8% | `stationary` |
| `none` | 784 | 0.7% | `scale_contradicted` 328, `slam_jump` 226, `too_few_keyframes` 208 (moved, or no GPS to verify), `no_calibration` 18, `rerun_poses_mismatch` 4 |

Other facts at this date:
- Sample manifest: 120,026 unique `source_video_path`s.
- `metric` poses: scale about +-10% (checked against GPS on only 5 clips, within +-8%).
- `ego_failed` clips could still carry a `metric_scale`/`metric_poses` (the first sweep ignored QC); `pose_source` says `none` for them.
- 9 clips (all `slam_jump`) had a `result.npz` but no record from the original run (probably a crashed
  worker); backfilled as `ok: false`, `stage: "backfilled"`, `metric_scale.status: "not_attempted"`.
- Scale spread: ~11% of depth-scaled clips have `scale_rel_iqr` > 0.25.
