# Performance investigations

## Throughput of `run_full_dataset.sbatch` (4-GPU GH200 node, 288 cores, 450 GB host RAM)
- ~22-23 clips/min at `--workers_per_gpu 4`; the full ~120k-clip dataset takes on the order of 90 h,
  hence the self-resubmitting job (normal partition limit 12 h).
- `--workers_per_gpu 4` uses ~65 GB/GPU (cap 98 GB) but sits close to the host RAM ceiling (16
  worker processes each holding a model copy before GPU transfer). `--workers_per_gpu 5` was killed
  by a host-RAM OOM (SIGKILL, GPU memory still had headroom).
- `--workers_per_gpu 6` (36 workers): ~74-75% SM utilization but host RAM hit 81% of the cgroup cap
  during model loading and per-clip CUDA OOMs occurred (6 x ~15.7 GB packs a 95 GB GPU with ~0 headroom).
  Do not exceed 4 without re-validating on a small `--limit` run.
- **2026-08-24 profiling:** average SM utilization was only ~41% at 4 workers/GPU. Cause: each
  worker's PyTorch/MKL/OpenBLAS/OpenMP sized its thread pool off the full 288-core count. Capping
  threads to `max(1, cpu_count // (num_gpus*workers_per_gpu + 1))` (now in `batch_infer.gpu_worker`;
  `--cpu_threads_per_worker`) raised utilization to ~75% but **throughput stayed flat** at 22-23
  clips/min: a clip's wall-clock is dominated by a CPU-bound, largely serial path (per-submap gtsam
  `graph.optimize()` batch re-optimization, GPS/OSM road matching, keyframe selection). The cap is
  kept because it is free and removes contention risk (a 232 s outlier clip was seen pre-fix at 6
  workers/GPU). Real gains would need less per-clip CPU work (e.g. incremental graph optimization) or more nodes
  (safe: every job skips clips that already have a result).

## Self-resubmission and the SIGUSR1 trap
- 2026-08-27: job 3198885 caught its SIGUSR1 on schedule but the chain silently died; no follow-up
  job was ever submitted. The workers kept running full-tilt on the shared Lustre filesystem, so the
  resubmit counter's read+write almost certainly queued behind that contention and never returned
  before the hard TimeLimit kill. Fix: read the counter once at script start, write it only after
  `sbatch`, best effort. The warning time was raised from an original 120 s to 300 s for the same reason.

## OpenCV with CUDA (Dockerfile)
Keyframe selection (`vggt_slam/frame_overlap.py`: goodFeaturesToTrack + calcOpticalFlowPyrLK) was
found to cost about as much wall-clock per clip as the SLAM GPU forward pass (live profiling: 38% of
per-frame time vs 34% in the model). GH200 has a dedicated optical-flow block, so a CUDA-enabled
OpenCV build was added. Adding more GPU worker processes had been tried and made things worse (CPU
contention from duplicating the same CPU-bound work).

## OSM spatial index
A raw `.osm.pbf` has no spatial index; a bbox query is a linear scan (~30 s for a 139 MB country
file, minutes for multi-GB ones), so `download_osm_extracts.py` builds an SQLite R-tree per region once.
