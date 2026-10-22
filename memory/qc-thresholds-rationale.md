# Evidence behind the QC thresholds (`scripts/metric_scale/clip_qc.py`, QC_VERSION 2026-10-21.1)

Measured on the 2026-10 dataset (see [dataset-snapshot-2026-10.md](dataset-snapshot-2026-10.md),
[qc-validation.md](qc-validation.md)). The code keeps only the rule; change a threshold -> bump
`QC_VERSION`.

- `SLAM_JUMP_RATIO = 50`: normal clips have p99 of the max-step ratio at ~27; 0.43% of clips exceed 50.
- `SLAM_RUN_RATIO = 10` / `SLAM_RUN_LEN = 6`: catches a sustained divergence whose individual
  steps stay under the jump ratio (e.g. a clip that only blows up in its last seconds).
- `SCALE_MAX_MPS = 50`: 180 km/h; p99.9 of the 5 s window speed over the dataset is ~41 m/s.
- `SCALE_MAX_GPS_RATIO = 1.5` (was 2.0): in a spot check, 17 of 38 passed clips with a ratio of
  1.5-2 had a depth-based scale that clearly disagreed.
- `SCALE_CHECK_LOW/HIGH = 0.8 / 1.25`: typical clips sit at 0.92-1.05; below 0.8 about half the
  spot-checked clips disagreed with the depth scale.
- `MAP_MATCH_CHECK_M = 10`: in a first spot check, 17 of 46 QC-passed clips above 10 m clearly
  disagreed with the depth scale, vs 7% below 5 m. The larger later sample found only 4% (102 of
  2,843) contradicted, so the trigger is conservative; see qc-validation.md.
- `STATIC_MAX_GPS_EXTENT_M = 30`: SLAM rarely adds keyframes without motion, so < 4 keyframes with a
  small GPS extent means "barely moved", not "SLAM failed".

## Cross-check verdicts (`update_manifest_metric_scale.py`)

Ratio = depth scale / GPS-fused scale: `confirmed` 0.8-1.25, `ambiguous` 0.67-0.8 or 1.25-1.5,
`contradicted` outside 0.67-1.5. Noisy per-frame depth estimates (`scale_rel_iqr` > 0.25, ~11% of
clips; about a quarter of the cross-checked ones) are contradicted only if the whole interquartile
range lies outside 0.67-1.5. For a contradicted clip it is unknown which estimate is wrong (in a
spot check the depth scale was sometimes the wrong one), so neither is trusted (`pose_source` none).
Exception: clips flagged only for map-matching error whose raw GPS track length agrees with the
fused scale (fused/GPS length 0.8-1.25) stay `fused` with `gps_length_supports_fused` (112 clips;
a judgement, not validated).
