# memory/

Project knowledge that is **not** part of the code contract: measurements, validation results,
investigation write-ups, dated dataset snapshots and decisions' rationale. They record what was
true when they were measured and may be stale; the code and `README.md` describe how things work
*now*. Code comments point here instead of carrying numbers.

| File | Contents |
|---|---|
| [qc-validation.md](qc-validation.md) | How good is the QC? Spot-check against depth scale (2026-10-21) |
| [qc-thresholds-rationale.md](qc-thresholds-rationale.md) | Evidence behind each threshold in `scripts/clip_qc.py` and the cross-check rules |
| [dataset-snapshot-2026-10.md](dataset-snapshot-2026-10.md) | Clip counts per `pose_source` / reason at 2026-10-20 |
| [manifest-and-scale-history.md](manifest-and-scale-history.md) | `fixed_intrinsics` backfill, undistortion-cache race, rerun scale transfer, backfilled rows |
| [gps-fusion-notes.md](gps-fusion-notes.md) | Failure-status audit, crop-radius case study, map-match scoring |
| [performance-investigations.md](performance-investigations.md) | Throughput / GPU / RAM investigations behind sbatch and Dockerfile settings |
| [backend-choice.md](backend-choice.md) | Why DA3-GIANT-1.1 + ray pose is the default, loop-closure caveat |
| [osm-bev-tag-reliability.md](osm-bev-tag-reliability.md) | Which OSM tags could be trusted for the map images |
