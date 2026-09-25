# Trajectory inference for WM

## Quick start

A pre-built image is ready to use on Clariden — no build step needed. Register it as the
Pyxis environment `vggtslam_env` (image itself at
`/capstor/store/cscs/swissai/ab036/containers/vggtslam_image.sqsh`), so every command below just
runs as `srun --environment=vggtslam_env <command>`.

Below are the commands used to process Natix:

| Task | Command |
|---|---|
| Full dataset, self-resubmitting until done | `sbatch scripts/run_full_dataset.sbatch` |
| Download OSM road-network extracts (should be done first once, before GPS fusion) | `python scripts/download_osm_extracts.py --out_dir <dir>` |
| Archive / restore finished sessions (inode cleanup) | `python scripts/pack_sessions.py ...` / `scripts/unpack_sessions.py ...` |

Every script's own `--help` / module docstring has the full flag list. The full run produces one
`result.npz` per clip — see [Loading a clip's trajectory](#loading-a-clips-trajectory) below.

**Other datasets (no intrinsics metadata, no GPS):** intrinsics and GPS already degrade
gracefully — with no sibling `FIXED_METADATA.json`, `batch_infer.py` automatically falls back to
`--shared_intrinsics`; with no sibling GPS `.csv` (or `--no-gps_fusion`), GPS fusion is just
skipped. The one real Natix-specific piece is clip *discovery*: `--dataset_root` batch mode only
finds videos inside directories literally named `FRONT_FOLDER`. For another layout,
generalize `_iter_front_folder_mp4s` in `scripts/batch_infer.py`.

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


## Loading a clip's trajectory

Use `vggt_slam.slam_utils.load_result_npz` rather than parsing `result.npz` by hand — it returns a plain dict (`frame_ids`, `slam_rotations`, `slam_translations`, `extraction_fps`, `video`, and `fused_rotations`/`fused_translations`/`snapped`/`matched_route_wkt`/`gps_fusion_info`, the last five `None` if that clip's GPS fusion never ran or didn't reach that stage):

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