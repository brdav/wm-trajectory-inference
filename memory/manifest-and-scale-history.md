# Manifest and scale history

## `fixed_intrinsics` provenance
Recorded by `batch_infer.py` for 72,829 clips; for the other 47,197 it was backfilled: verified
against a rerun with fixed intrinsics for 7,612 of them (identical poses = fixed intrinsics were
used; evidence per clip in `_pipeline/fixed_intrinsics_evidence.csv`) and otherwise inferred from
the session zip (`true` iff it contains `undistort_maps.npz`, which agrees with 99.2% of the
recorded values).

## The undistortion-cache race
Two clips of one session extracted at the same moment shared a temp file; the second silently fell
back to self-calibration (fixed in `b722ac4`). Roughly 0.4-0.8% of the inferred `true` values
(about 160-320 of 39,585 rows) may in fact be self-calibrated; the effect on the fused
trajectories is small (median 5.6 m on ~485 m paths).

## Rerun scale transfer
The depth scale is computed on a SLAM *rerun* with fixed intrinsics. It is bit-identical to the
stored run (100 of 100 control clips, ~99.6% of all clips) except for clips the original run
processed with self-calibration because of the race above; for those the rerun's SLAM gauge
differs by 0-20%. These get the scale transferred through a similarity alignment (residual <= 5%
of path length, `metric_scale.transfer_residual`); where the reruns genuinely diverged there is no
scale (`rerun_poses_mismatch`).

## Repair scripts
Older manifest rows from before the fixed schema were repaired once by one-off scripts, removed
afterwards (last version: git commit `61e6655`).
