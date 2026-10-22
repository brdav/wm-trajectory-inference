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
| Take a packed session apart again to edit it | `python scripts/unpack_sessions.py ...` |

New data? See [Processing new data](#processing-new-data).

Every script's own `--help` / module docstring has the full flag list. The full run produces one
`result.npz` per clip — see [Loading a clip's trajectory](#loading-a-clips-trajectory) below.

**Other datasets (no intrinsics metadata, no GPS):** intrinsics and GPS already degrade
gracefully — with no sibling `FIXED_METADATA.json`, `batch_infer.py` automatically falls back to
`--shared_intrinsics`; with no sibling GPS `.csv` (or `--no-gps_fusion`), GPS fusion is just
skipped. Which videos to process comes from a sample manifest (below), so for another dataset provide a
parquet with a `source_video_path` column (paths relative to `--dataset_root`, laid out as
`<Country>/.../<session>/<HH-MM-SS>/FRONT_FOLDER/<video>.mp4`) and pass `--dataset_root`, `--out_root` and
`--sample_manifest` (or the `DATASET_ROOT` / `OUT_ROOT` / `SAMPLE_MANIFEST` environment variables of the sbatch).

## Processing new data

The pipeline has two sbatch jobs, run one after the other; everything is resumable and skips what is done.

**What drives it: the sample manifest.** `/capstor/store/cscs/swissai/ab036/processed_datasets/natix/manifest/samples_24fps.parquet`
(also available hive-partitioned as `manifest/samples_24fps/country=*/part.parquet`) lists the dataset's 5 s samples;
its `source_video_path` column names the full source video of each. The pipeline processes whole source videos,
so the work list is the set of unique `source_video_path`s (120,026 today). The dataset tree is never walked and
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
`sbatch --export=ALL,OUT_ROOT=$SCRATCH/test_out,SAMPLE_MANIFEST=$SCRATCH/small.parquet scripts/run_full_dataset.sbatch`
(link `_osm_extracts` into `$OUT_ROOT` for GPS fusion).

**Model weights.** The SLAM loop closure needs `dino_salad.ckpt` and the DINOv2 `torch.hub` repo in a torch hub
cache: `$TORCH_HOME/hub/checkpoints/dino_salad.ckpt` (download from
`https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt`) and
`$TORCH_HOME/hub/facebookresearch_dinov2_main/` (cloned by `torch.hub` on first use, needs internet). The sbatch files
set `TORCH_HOME` to `/iopsstor/scratch/cscs/dbrggema/metric_scale/torch_home`, which holds both; that is scratch,
so it can be purged, and `~/.cache/torch/hub` does not have them. The depth models (DA3) download into `HF_HOME`
(`/iopsstor/scratch/cscs/dbrggema/hf_home`) on first use.

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

Stages 2-5 are `scripts/run_metric_scale.sbatch`. QC routes (`scripts/clip_qc.py`):

- `ego_failed`: SLAM itself is unusable: a jump/divergence, or < 4 keyframes in a clip that did move.
- `static`: < 4 keyframes **and** the raw GPS fixes span < 30 m, i.e. the vehicle barely moved (keyframe
  selection needs motion). Not a failure; there is no meaningful trajectory or scale.
- `needs_metric_scale`: SLAM is fine but there is no trusted GPS scale: no `fused_poses`, or they fail a
  hard check (implied speed > 50 m/s, fused path > 1.5x the raw GPS track, GPS fusion found no road route).
- `needs_scale_check`: `fused_poses` pass the hard checks but sit near a limit or cannot be verified
  (fused/GPS path-length ratio < 0.8 or > 1.25, or no ratio computable, or mean map-matching error > 10 m).
  They are **cross-checked** against the depth-based scale.
- `use_fused`: everything else.

**Downstream code should branch on one field, `pose_source` in `trajectories/manifest.jsonl`:**

| `pose_source` | use | clips (2026-10-20) | `pose_source_reason` |
|---|---|---|---|
| `fused` | `fused_poses` (GPS/road-fused, metric) | 109,230 (91.0%) | `gps_fusion` 105,063; after a cross-check: `gps_confirmed_by_depth` 3,161, `gps_ambiguous_vs_depth` 862, `gps_length_supports_fused` 112, `gps_unchecked` 32 |
| `metric` | `metric_poses` (SLAM poses scaled by a depth-based estimate) | 9,016 (7.5%) | why GPS wasn't used: `no_fused_poses` 8,681, `implausible_speed` 217, `longer_than_gps` 68, `no_route_match` 50 |
| `static` | nothing to use: ~stationary | 996 (0.8%) | `stationary` |
| `none` | nothing usable | 784 (0.7%) | `scale_contradicted` 328, `slam_jump` 226, `too_few_keyframes` 208 (moved, or no GPS to verify), `no_calibration` 18, `rerun_poses_mismatch` 4 |

The cross-check (`scale_check` in the manifest: `depth_scale`, `ratio` = depth/GPS, `verdict`): `confirmed`
(ratio 0.8-1.25), `ambiguous` (0.67-0.8 or 1.25-1.5), `contradicted` (outside 0.67-1.5), `unavailable`
(no depth scale, or < 5 frames). `scale_check.confidence` is `high` when the per-frame depth estimates are
consistent (`scale_rel_iqr` <= 0.25); then the verdict uses the median ratio. For noisy (`low`) estimates a clip is
contradicted only if the whole interquartile range of its per-frame scales lies outside 0.67-1.5. Clips stay
`fused` unless contradicted. For a contradicted clip we cannot tell which estimate is wrong (in a spot check the
depth scale was sometimes the wrong one), so neither is trusted: `pose_source` is `none` -- with one exception:
for a clip flagged only for its map-matching error, the raw GPS track length already agrees with the fused scale
(fused/GPS length 0.8-1.25), so GPS fit and GPS length outvote the depth scale: `fused`,
`gps_length_supports_fused` (112 clips; a judgement, not validated).

### manifest.jsonl row schema

One JSON object per source video, **always with exactly these keys** (null when a key does not apply; nested objects
always have all of their keys too). Written by `batch_infer.py` (first block) and completed by
`scripts/update_manifest_metric_scale.py` (rest), which also normalizes every row to this schema. Older rows from
before the schema were repaired by `scripts/backfill_manifest.py`.

| key | content |
|---|---|
| `clip_id` | full path of the source video |
| `ok` | `batch_infer.py` finished the clip. `false` + `error`/`stage` otherwise (null when `ok`) |
| `seconds` | wall time of the clip's run (null for backfilled rows) |
| `fixed_intrinsics` | the camera calibration of `FIXED_METADATA.json` was used (otherwise self-calibrated). Recorded by `batch_infer.py` for 72,829 clips; for the other 47,197 it was backfilled: verified against a rerun with fixed intrinsics for 7,612 of them (identical poses = fixed intrinsics were used; evidence per clip in `_pipeline/fixed_intrinsics_evidence.csv`) and otherwise inferred from the session zip (`true` iff it contains `undistort_maps.npz`, which agrees with 99.2% of the recorded values). A race in the undistortion cache (two clips of one session extracted at the same moment shared a temp file; the second silently fell back to self-calibration; fixed in `b722ac4`) means roughly 0.4-0.8% of the inferred `true` values (about 160-320 of 39,585 rows) may in fact be self-calibrated; the effect on the fused trajectories is small (median 5.6 m on ~485 m paths). |
| `gps_fusion` | `status`, `error`, `region_used`, `used_fallback`, `network_profile_used`, `matched_keyframes`, `total_keyframes`, `map_match_error_mean_m`, `map_match_error_max_m`, `map_match_error_n_points`; the last eight are null unless the fusion produced a placement |
| `qc` | `route`, `slam_ok`, `slam_reason`, `n_keyframes`, `slam_valid_until_s`, `scale_ok`, `scale_reason`, `gps_extent_m`, `fused_over_gps_len`, `fused_max_window_mps`, `slam_max_step_ratio`, `version` (null where a metric does not apply, e.g. `slam_valid_until_s` unless SLAM broke) |
| `metric_scale` | `source` (`gps_fusion` / `metric_depth` / null), `scale`, then source-specific: `scale_p25`, `scale_p75`, `scale_rel_iqr`, `n_frames_used`, `model`, `scale_transferred`, `transfer_residual` (depth scale); `gps_fusion_scale` / `depth_scale` (when both exist); `status` (`failed`, `scale_contradicted`, `not_attempted`), `reason`, `error` (no usable scale) |
| `scale_check` | `depth_scale`, `ratio`, `verdict`, `confidence`, `n_frames_used`, `scale_rel_iqr`, `adjudication`; all null unless `qc.route == "needs_scale_check"` |
| `pose_source`, `pose_source_reason` | which poses to use, and why (table above) |

Things to know:

- **`metric` poses are an estimate**, about +-10% in scale (checked against GPS on only 5 clips, within +-8%).
  The manifest's `metric_scale` entry lets you filter: `n_frames_used` (clips with few frames are weak
  evidence) and `scale_rel_iqr` (spread of the per-frame estimates; ~11% of clips are above 0.25).
- **`gps_fusion.status == "ok"` does not guarantee a good scale** -- that is what the QC checks and the
  cross-check are for. Of the 4,495 cross-checked clips, 50% were confirmed by the depth scale, 17-21% were
  contradicted for the length-ratio triggers but only 4% for the map-matching trigger (102 of 2,843, all of
  which were then overruled by the GPS track length); noisy depth estimates (about a quarter) are judged on their
  interquartile range and only 32 clips remain unverifiable. Measured error rate of the
  unflagged clips: see [QC validation](#qc-validation-how-good-is-the-qc).
- **`none` / `slam_jump`:** `qc.slam_valid_until_s` is the time of the last keyframe before the
  first jump, i.e. a clean prefix if you want to truncate (not validated).
- **`ego_failed` clips can still carry a `metric_scale`/`metric_poses`** (the first sweep ignored
  QC). Ignore them: `pose_source` already says `none`.
- **Scale of a depth-scaled clip** was computed on a SLAM rerun. For clips where the rerun's
  gauge differed from the stored run, the scale is transferred through a similarity alignment
  (`metric_scale.scale_transferred`, residual <= 5% of path length in `transfer_residual`; these are clips the original run processed with self-calibration because of that race, see `fixed_intrinsics`); where
  the reruns genuinely diverged there is no scale (`rerun_poses_mismatch`).
- **Where the scale lives:** `metric_scale` is in the manifest for every clip. Clips that got a depth
  scale also have it inside `result.npz` (`metric_scale`, `metric_poses`, `metric_scale_json`; load with
  `load_result_npz`, fields `metric_scale`/`metric_poses`/`metric_scale_info`), alongside `fused_poses`
  where those exist. For plain `fused` clips the scale is only in the manifest (fused/slam step-length ratio).
- **`static` clips** have `slam_poses` from < 4 keyframes: treat the trajectory as stationary rather than
  reading motion from it (`qc.gps_extent_m` is the raw GPS span).
- 9 clips (all `slam_jump`) had a `result.npz` but no record from the original run, probably a crashed worker; they got a row
  with `ok: false`, `stage: "backfilled"` and `metric_scale.status: "not_attempted"`. The manifest has exactly one row per
  source video of the sample manifest (120,026).

```python
import json
for line in open(".../trajectories/manifest.jsonl"):
    r = json.loads(line)
    if r["pose_source"] == "fused":    ...   # result.npz["fused_poses"]
    elif r["pose_source"] == "metric": ...   # result.npz["metric_poses"], r["metric_scale"]["n_frames_used"], ...
    # "none": skip
```

## QC validation (how good is the QC?)

All numbers 2026-10-21, measured with `scripts/qc_validation/` (the raw clip lists, scores and figures were not kept; rerunning
needs a GPU run, see "Reproduce"). The only independent reference
for a clip's scale is the depth-based one (DA3METRIC-LARGE vs the SLAM point cloud), so every statement below
is "the two estimates agree / clearly disagree" (ratio depth/GPS in 0.8-1.25 / outside 0.67-1.5), never ground
truth. A disagreement does not say which estimate is wrong; in some clips it was the depth scale.

- **The reference is usable.** On a random sample of 363 unflagged `use_fused` clips with a confident depth scale
  the depth/GPS ratio has median 0.98 and a robust spread of 8.3% (these bands are about 2 and 4 sigma).
- **Unflagged clips (105,063).** 6 of 363 (1.7%, 95% CI 0.8-3.6%) clearly disagree, i.e. at most ~1,700 clips
  (800-3,700) may have a wrong scale that QC cannot see; another 3.3% are ambiguous (0.67-0.8 / 1.25-1.5).
  9% of the sample had no confident depth scale and is not covered.
- **QC mostly errs by passing, not rejecting.** Among clips QC rejected for the scale check, 74 of 248 with a
  confident depth scale (30%, CI 24-36%) have a depth scale that agrees with the GPS one: probable false
  positives (~70-100 clips), against up to ~1,700 clips passed that disagree.
- **Where the cross-check tier pays off** (clearly contradicted): fused/GPS length ratio 1.5-2 (now a hard
  reject): 17 of 40; ratio < 0.8: 21% (184 of 881); no ratio computable: 17% (102 of 583); ratio 1.25-1.5: 18%
  (33 of 188); mean map-match error > 10 m: 4% (102 of 2,843; flat across 10-15 / 15-25 / > 25 m; an earlier
  spot check suggested 37% but was confounded by the other edge-case clips). Speed 30-50 m/s and SLAM step
  ratio 25-50 (near-threshold) were fine (0 of 40, 1 of 30).
- **Placement is fine; timing is not always.** The distance from raw GPS fixes to the fused path (cross-track,
  insensitive to timing) has median 3.1 m per clip; only 10 of 1,496 QC-passed clips exceed 20 m and 2 exceed
  50 m. The fused path at the *same time* as a fix is > 50 m off in 42 clips, but those lie 5 m from the path:
  an implied GPS-vs-video time offset of a median 3 s (p90 9 s) in ~3% of clips. Not a pose error, but beware
  when using GPS timestamps.
- **Limits:** one sample, one pool, depth scale +-8-10%; the rates above were measured on confident depth
  estimates only (9% of the sample had noisy ones and are not covered); `static` and `too_few_keyframes` clips were checked against GPS extent only, not against depth.

Reproduce: draw a stratified sample of clips into a `clip_list.txt` (dataset-relative `.mp4` path, TAB, stratum label; the
strata and sample sizes are in `scripts/qc_validation/qc_depth_spotcheck.py`), score it with
`scripts/estimate_metric_scale.py --clip_list ... --pipeline_dir $QC_VALIDATION_DIR/records --work_dir $QC_VALIDATION_DIR/metric`
(about 25 clips/min on a 4-GPU node), then `QC_VALIDATION_DIR=... python scripts/qc_validation/qc_depth_spotcheck.py`
(and `qc_placement.py`, which needs no GPU).

### Traceability

- Each manifest row's `qc.version` and `trajectories/qc_config.json` (QC version, git commit, every threshold,
  route and `pose_source` counts, depth-scale settings) say which QC produced it; `clip_qc.csv.config.json` is
  the same for the QC step alone.
- `_pipeline/integration_log.jsonl` records, per session zip, which clips got `metric_scale`, which
  were transferred from a SLAM rerun, and which were refused; `_pipeline/manifest_metric_scale.jsonl` the depth step.
- Files replaced in place are copied to `_pipeline/backups/` first (see its README.txt); the manifest scripts keep a
  timestamped `manifest.jsonl.bak_<time>` next to the manifest, which you can move there or delete.
- The per-clip depth-scale files (rerun poses and per-frame scales) are transient work files in `$SCRATCH` (see
  "Where everything lives"); the estimator skips finished clips via the QC csv and `integration_log.jsonl`, so deleting
  them changes no behaviour.

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