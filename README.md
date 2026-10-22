# Trajectory inference for WM

## Quick start

A pre-built image is ready to use on Clariden — no build step needed. Register it as the
Pyxis environment `vggtslam_env` (image itself at
`/capstor/store/cscs/swissai/ab036/containers/vggtslam_image.sqsh`), so every command below just
runs as `srun --environment=vggtslam_env <command>`.

Below are the commands used to process Natix:

| Task | Command |
|---|---|
| Process every source video in the dataset's sample manifest (self-resubmitting until done, packs the output into session zips at the end) | `sbatch scripts/run_full_dataset.sbatch` |
| Then: QC, depth-based metric scale for clips without a usable GPS scale, merge into `result.npz` + `manifest.jsonl` | `sbatch scripts/run_metric_scale.sbatch` |
| Download OSM road-network extracts (once, and for any new country, before GPS fusion) | `python scripts/download_osm_extracts.py --out_dir <dir>` |
| Take a packed session apart again to edit it | `python scripts/tools/unpack_sessions.py ...` |

New data? See [Processing new data](#processing-new-data).

Every script's own `--help` / module docstring has the full flag list. The full run produces one
`result.npz` per clip — see [Loading a clip's trajectory](#loading-a-clips-trajectory) below.

**Other datasets (no intrinsics metadata, no GPS):** intrinsics and GPS already degrade
gracefully — with no sibling `FIXED_METADATA.json`, `batch_infer.py` automatically falls back to
`--shared_intrinsics`; with no sibling GPS `.csv` (or `--no-gps_fusion`), GPS fusion is just
skipped. Which videos to process comes from a sample manifest (below), so for another dataset provide a
parquet with a `source_video_path` column (paths relative to `--dataset_root`, laid out as
`<Country>/.../<session>/<HH-MM-SS>/FRONT_FOLDER/<video>.mp4`) and pass `--dataset_root`, `--out_root` and
`--sample_manifest` (or the `DATASET_ROOT` / `TRAJ_ROOT` / `SAMPLE_MANIFEST` environment variables of the sbatch).
The cluster paths live in one place, `vggt_slam/paths.py`, overridable by those same environment variables
(`REPO_DIR`, `CPU_PY`, `TORCH_HOME` and `HF_HOME` can likewise be overridden for the sbatch files).

## Repository layout

`vggt_slam/` is the library, `scripts/` the pipeline, `docs/` reference documentation, and `memory/` dated
measurements and investigation notes (validation results, dataset snapshots, rationale for thresholds) that are
deliberately kept out of the docs and the code comments; see [memory/README.md](memory/README.md).

## Scripts layout

```
scripts/                            the pipeline (what you run to process data)
  run_full_dataset.sbatch             stage 1: SLAM + GPS fusion for every video of the sample manifest, then packs session zips
  run_metric_scale.sbatch             stage 2: QC, depth-based scale, merge into result.npz + manifest.jsonl
  batch_infer.py  pack_sessions.py                                     called by stage 1
  clip_qc.py  estimate_metric_scale.py  integrate_metric_scale.py  update_manifest_metric_scale.py   called by stage 2
  download_osm_extracts.py            setup: OSM road extracts for GPS fusion (once, and for any new country)
scripts/tools/                      maintenance and visualization, not needed to process data
  unpack_sessions.py                  restore a packed session zip to a loose folder
  refuse_gps.py                       rerun GPS fusion only, on already processed clips
  overlay_trajectory_on_video.py      draw a clip's future trajectory onto its video
scripts/osm_maps/                   separate product: ego-centred OSM map images for the clips (consumes the fused trajectories)
  osm_bev.py  make_osm_maps.py  make_osm_report.py  run_osm_prepare.sbatch  run_osm_maps.sbatch
```

## Processing new data

The pipeline has two sbatch jobs, run one after the other; everything is resumable and skips what is done.

**What drives it: the sample manifest.** `/capstor/store/cscs/swissai/ab036/processed_datasets/natix/manifest/samples_24fps.parquet`
(also available hive-partitioned as `manifest/samples_24fps/country=*/part.parquet`) lists the dataset's 5 s samples;
its `source_video_path` column names the full source video of each. The pipeline processes whole source videos,
so the work list is the set of unique `source_video_path`s. The dataset tree is never walked and
no path cache exists. Read via `vggt_slam/sample_manifest.py`.

**To process more data**
1. Put the videos on the dataset root and get them into the sample manifest (same columns). New countries also
   need their OSM extracts (`scripts/download_osm_extracts.py`), otherwise GPS fusion is skipped for them.
2. `sbatch scripts/run_full_dataset.sbatch` (SLAM + GPS fusion, `batch_infer.py`). A video is skipped if it already
   has a loose `result.npz`, an ok row in `trajectories/manifest.jsonl`, or its `result.npz` inside its session
   zip. The job resubmits itself every 12 h until nothing is left (stop it with
   `touch trajectories/_pipeline/STOP`; counter in `_pipeline/.resubmit_count`, max 40). When everything is
   done it **packs the output into session zips** (`scripts/pack_sessions.py`): one zip per session, only for
   sessions in which every manifest video has a result; clips of a session that already has a zip are
   **merged into it**. It prints the next command.
3. `sbatch scripts/run_metric_scale.sbatch`: QC of every clip, depth-based scale for the clips that need one,
   and the `qc` / `metric_scale` / `pose_source` fields of `manifest.jsonl`. See
   [Which poses to use](#which-poses-to-use-downstream).

Logs of both jobs: `trajectories/_pipeline/logs/`. To try the pipeline on a handful of clips without touching
the real output, give the first job a small manifest and a scratch output folder:
`sbatch --export=ALL,TRAJ_ROOT=$SCRATCH/test_out,SAMPLE_MANIFEST=$SCRATCH/small.parquet scripts/run_full_dataset.sbatch`
(link `_osm_extracts` into `$TRAJ_ROOT` for GPS fusion).

**Model weights.** The SLAM loop closure needs `dino_salad.ckpt` and the DINOv2 `torch.hub` repo in a torch hub
cache: `$TORCH_HOME/hub/checkpoints/dino_salad.ckpt` (download from
`https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt`) and
`$TORCH_HOME/hub/facebookresearch_dinov2_main/` (cloned by `torch.hub` on first use, needs internet). The sbatch files
set `TORCH_HOME` to `/iopsstor/scratch/cscs/dbrggema/metric_scale/torch_home`, which holds both; that is scratch,
so it can be purged, and `~/.cache/torch/hub` does not have them. The depth models (DA3) download into `HF_HOME`
(`/iopsstor/scratch/cscs/dbrggema/hf_home`) on first use.

## Tests

CPU-only tests of the QC routing and manifest logic (synthetic data, a few seconds):
`python -m pytest tests` (the `wm_env` conda env is enough; `test_result_npz.py` is skipped unless torchvision is
installed, i.e. inside the container). Refactors that touch the SLAM / GPS-fusion path need a GPU run instead: process
a few clips before and after and diff them with `python tests/compare_results.py <old_out> <new_out>`.

## Rebuild image (Podman + Enroot)

A `Dockerfile` is provided for GPU clusters using Slurm with Pyxis/enroot, for when you need to
change dependencies (edits to `vggt_slam/*.py`/`main.py` alone don't need a rebuild, just a `pip install -e .`).

Build the image once and convert it to a squashfs file for enroot:
```
podman build -t vggtslam_image .
enroot import -o vggtslam_image.sqsh podman://vggtslam_image
```
(To make this the new pre-built image everyone else uses, copy the resulting `.sqsh` over
`/capstor/store/cscs/swissai/ab036/containers/vggtslam_image.sqsh` -- that path is what
`vggtslam_env`'s Pyxis definition, `~/.edf/vggtslam_env.toml`, points at.)


## Where everything lives (`trajectories/`)

All pipeline output is under `/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories/`. Nothing is
written anywhere else except transient files on scratch.

```
trajectories/
  <Country>/.../<Session>.zip   one zip per recording session; per clip <HH-MM-SS>/FRONT_FOLDER/FRONT_<ts>/result.npz
                                with slam_poses always, fused_poses (GPS-fused) and/or metric_poses (depth-scaled) when
                                the clip has them -- WHICH ONE TO USE is `pose_source` in manifest.jsonl
  manifest.jsonl                one row per clip, fixed schema (see "manifest.jsonl row schema") -- start here
  qc_config.json                provenance of those fields: QC version, git commits, every threshold, counts
  packed_sessions.jsonl, _osm_extracts/    bookkeeping of pack_sessions, OSM extracts used by GPS fusion
  _pipeline/                    records of the post-processing (stages 2-5), not needed to USE the poses:
    clip_qc.csv (+ .config.json)     stage 2 output, all QC metrics per clip (also copied into manifest.jsonl's qc)
    manifest_metric_scale.jsonl      stage 3 record per clip: depth scale, spread, frames used, or why it failed
    integration_log.jsonl            stage 4 record per session: which clips got metric_scale / transferred / refused
    fixed_intrinsics_evidence.csv    per clip: why its fixed_intrinsics value is definitive (rerun comparison, session maps)
    backups/                         copies of files replaced in place (see its README.txt); deletable when no longer wanted
    logs/                            job logs of stages 2-5 (sbatch output)
```

Transient: the per-clip depth-scale files (`metric_scale.npz`, with the SLAM rerun's poses and per-frame scales) are
written to `$SCRATCH/natix_metric_scale_work/` (`--work_dir`) by stage 3 and merged into the zips by stage 4; delete
that folder whenever you like. To run on new data: run `batch_infer.py`, then `sbatch scripts/run_metric_scale.sbatch`.

## Which poses to use (downstream)

Processing is a pipeline; each stage can drop or reroute a clip, and the manifest records the
outcome so consumers never need to re-derive it:

| # | Stage | Script | Outcome per clip |
|---|---|---|---|
| 1 | SLAM + GPS/road fusion | `batch_infer.py` | `result.npz` with `slam_poses`; plus `fused_poses` if fusion worked (`gps_fusion.status`) |
| 2 | Quality control | `clip_qc.py` | `qc.route`: `use_fused` / `needs_scale_check` / `needs_metric_scale` / `static` / `ego_failed` |
| 3 | Depth-based scale | `estimate_metric_scale.py` | meters-per-SLAM-unit for `needs_metric_scale` and `needs_scale_check` clips, from DA3METRIC-LARGE depth vs the SLAM point cloud |
| 4 | Merge into clip files | `integrate_metric_scale.py` | adds `metric_scale`, `metric_poses`, `metric_scale_json` to the clip's own `result.npz` (inside the session zips) |
| 5 | Merge into manifest | `update_manifest_metric_scale.py` | adds `qc`, `metric_scale`, `scale_check`, `pose_source`, `pose_source_reason` to every `manifest.jsonl` row |

Stages 2-5 are `scripts/run_metric_scale.sbatch`. QC routes (`scripts/clip_qc.py`; thresholds are in the script,
their evidence in [memory/qc-thresholds-rationale.md](memory/qc-thresholds-rationale.md)):

- `ego_failed`: SLAM itself is unusable: a jump/divergence, or < 4 keyframes in a clip that did move.
- `static`: < 4 keyframes **and** the raw GPS fixes span < 30 m, i.e. the vehicle barely moved. Not a failure;
  there is no meaningful trajectory or scale.
- `needs_metric_scale`: SLAM is fine but there is no trusted GPS scale: no `fused_poses`, or they fail a hard
  check (implied speed > 50 m/s, fused path > 1.5x the raw GPS track, GPS fusion found no road route).
- `needs_scale_check`: `fused_poses` pass the hard checks but sit near a limit or cannot be verified
  (fused/GPS path-length ratio < 0.8 or > 1.25, or no ratio computable, or mean map-matching error > 10 m).
  They are **cross-checked** against the depth-based scale.
- `use_fused`: everything else.

**Downstream code should branch on one field, `pose_source` in `trajectories/manifest.jsonl`:**

| `pose_source` | use | `pose_source_reason` |
|---|---|---|
| `fused` | `fused_poses` (GPS/road-fused, metric) | `gps_fusion`; after a cross-check: `gps_confirmed_by_depth`, `gps_ambiguous_vs_depth`, `gps_length_supports_fused`, `gps_unchecked` |
| `metric` | `metric_poses` (SLAM poses scaled by a depth-based estimate, about +-10%) | why GPS wasn't used: `no_fused_poses`, `implausible_speed`, `longer_than_gps`, `no_route_match` |
| `static` | nothing to use: ~stationary | `stationary` |
| `none` | nothing usable | `scale_contradicted`, `slam_jump`, `too_few_keyframes`, `no_calibration`, `rerun_poses_mismatch` |

The cross-check (`scale_check` in the manifest) compares the depth scale with the fused one: ratio depth/GPS
0.8-1.25 `confirmed`, 0.67-0.8 or 1.25-1.5 `ambiguous`, outside 0.67-1.5 `contradicted`, `unavailable` without a
usable depth scale. Clips stay `fused` unless contradicted; for a contradicted clip neither estimate is trusted
(`pose_source` `none`), except when GPS fit and GPS length outvote the depth scale. Rules in detail:
`scripts/update_manifest_metric_scale.py`; how well it works and clip counts per category:
[memory/qc-validation.md](memory/qc-validation.md), [memory/dataset-snapshot-2026-10.md](memory/dataset-snapshot-2026-10.md).

The full manifest schema and notes on using the fields: [docs/manifest-schema.md](docs/manifest-schema.md).

```python
import json
for line in open(".../trajectories/manifest.jsonl"):
    r = json.loads(line)
    if r["pose_source"] == "fused":    ...   # result.npz["fused_poses"]
    elif r["pose_source"] == "metric": ...   # result.npz["metric_poses"], r["metric_scale"]["n_frames_used"], ...
    # "static" / "none": skip
```

### Traceability

- Each manifest row's `qc.version` and `trajectories/qc_config.json` (QC version, git commit, every threshold,
  counts, depth-scale settings) say which QC produced it.
- `_pipeline/integration_log.jsonl` records, per session zip, which clips got `metric_scale`, which were transferred
  from a SLAM rerun, and which were refused; `_pipeline/manifest_metric_scale.jsonl` the depth step.
- Files replaced in place are copied to `_pipeline/backups/` first (see its README.txt); the manifest scripts keep a
  timestamped `manifest.jsonl.bak_<time>` next to the manifest, which you can move there or delete.
- The per-clip depth-scale files are transient work files in `$SCRATCH`; deleting them changes no behaviour.

## Loading a clip's trajectory

Use `vggt_slam.slam_utils.load_result_npz` rather than parsing `result.npz` by hand — it returns a plain dict (`frame_ids`, `slam_rotations`, `slam_translations`, `extraction_fps`, `video`, and `fused_rotations`/`fused_translations`/`snapped`/`matched_route_wkt`/`gps_fusion_info`, the last five `None` if that clip's GPS fusion never ran or didn't reach that stage; and `metric_scale`/`metric_poses`/`metric_scale_info`, `None` unless a depth-based scale was merged in -- see [Which poses to use](#which-poses-to-use-downstream)):

```python
from vggt_slam.slam_utils import load_result_npz

result = load_result_npz("/path/to/out_dir/result.npz")
print(result["slam_translations"].shape)   # (n_frames, 3)
print(result["gps_fusion_info"])           # None, or {"status": ..., "map_match_error_mean_m": ..., ...}
```

At batch-pipeline scale (`scripts/batch_infer.py --dataset_root ...`), a whole finished recording session's directory (every clip that shares one `FIXED_METADATA.json`/undistort cache) may have since been archived into a single `Session.zip` by `scripts/pack_sessions.py`, to keep the output tree's inode count down — see that script's docstring. A packed clip's own directory no longer exists on disk, so reach it through the zip instead of assuming the loose path is always there:

```python
import io
import zipfile
from pathlib import Path

from vggt_slam.slam_utils import load_result_npz
from vggt_slam.undistort import session_cache_dir
from vggt_slam.session_pack import session_zip_path


def load_clip_trajectory(out_dir):
    """out_dir: a clip's own output directory (mirrors its path under --dataset_root, with the
    .mp4 suffix dropped). Works whether or not the clip's session has since been packed."""
    out_dir = Path(out_dir)
    result_path = out_dir / "result.npz"
    if result_path.exists():
        return load_result_npz(result_path)
    session_dir = session_cache_dir(out_dir)
    arcname = str(out_dir.relative_to(session_dir) / "result.npz")
    with zipfile.ZipFile(session_zip_path(out_dir)) as zf:
        return load_result_npz(io.BytesIO(zf.read(arcname)))
```

**Gotcha when analyzing `result.npz`:** the frame shared between two adjacent submaps (the last pose of one, the first pose of the next) is logged **twice** under the same `frame_id`, one entry per submap. Any script computing per-step velocity/acceleration from consecutive rows must drop one of the duplicate pair first — otherwise the ~0 elapsed time between them divides into a spurious, enormous velocity/acceleration spike at every single submap boundary.