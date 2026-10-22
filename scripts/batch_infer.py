#!/usr/bin/env python3
"""Run VGGT-SLAM trajectory inference on a video clip, or batch over many clips using all GPUs
on the node in parallel. The sole entry point for turning a natix-style clip (or dataset of
them) into poses: a single clip (--video/--out_dir) or every video of the sample manifest.
Designed to be launched
once inside an allocation that already has the vggtslam_image container attached, e.g.:

    # Single clip:
    srun --environment=vggtslam_env python scripts/batch_infer.py \\
        --video /capstor/.../FRONT_FOLDER/FRONT_2025-09-10_17-04-42.mp4 \\
        --out_dir /capstor/store/cscs/swissai/ab036/.../poses_out/france \\
        --num_gpus 1

    # Everything listed in the dataset's sample manifest (what scripts/run_full_dataset.sbatch does; the
    # manifest, dataset root and output root default to the natix locations, see vggt_slam/sample_manifest.py):
    srun --environment=vggtslam_env python scripts/batch_infer.py --num_gpus 4

Architecture:
  - One persistent GPU worker process per GPU (--num_gpus x --workers_per_gpu), each loading
    the geometry model ONCE and then looping over many clips pulled from a shared queue --
    this avoids the model-reload cost that a fresh process per clip would pay on every
    single clip (still paid once, unavoidably, in single-clip mode).
    Per-clip GPU compute is bursty
    (~1s forward pass per submap; most of a clip's wall-clock is CPU-side pipeline/
    orchestration work, not the GPU), so a single worker per physical GPU tends to leave it
    idle between calls -- raise --workers_per_gpu (multiple processes sharing one GPU id) if
    nvidia-smi shows low utilization; each GPU here has 98GB, comfortably fitting several
    model copies.
  - A CPU-side thread pool (--num_extract_workers) extracts frames (ffmpeg) for upcoming
    clips concurrently with GPU inference on already-extracted ones, so the two stages
    overlap instead of running as two sequential passes over the whole dataset. Undistortion
    no longer happens here as a bulk pass over every frame -- see below.
    ffmpeg decode (--ffmpeg_threads) parallelizes internally per clip too, since a single
    high-resolution clip (e.g. 2896x1876 vs 1280x960 for other natix clips seen so far) can
    otherwise dominate one extract-worker slot for far longer than a same-length clip at lower
    resolution.
  - Which clips to process comes from the dataset's sample manifest (--sample_manifest, parquet of 5 s
    samples; its `source_video_path` column names the full source videos), not from walking the
    dataset tree. Already-processed clips are skipped, so a run that hits the Slurm wall-time limit can
    simply be resubmitted to pick up where it left off: a clip counts as done if it has a loose
    result.npz, an ok row in <out_root>/manifest.jsonl, or its result.npz inside its session's
    zip (scripts/pack_sessions.py archives finished sessions).
  - GPS/road-fusion (vggt_slam/fusion.py) is ON by default: it map-matches against the
    pre-downloaded OSM region extracts, using each video's sibling FRONT_*.csv automatically (no registry --
    just same basename, .csv extension). Pass --no-gps_fusion to skip it
    if it bottlenecks a very large batch.

Per-clip intrinsics policy (always on, not configurable; rationale: memory/backend-choice.md):
  - Every clip is checked for a sibling FIXED_METADATA.json (natix trip-dir convention:
    .../<trip_dir>/<timestamp>/FRONT_FOLDER/video.mp4, metadata one level above <timestamp>),
    cross-checked against the video's own encoded resolution (FIXED_METADATA.json's own
    frame_width/height has been observed to be exactly half the true encoded resolution for at
    least one clip, so it's never trusted on its own).
  - If found: undistortion remap tables are computed once (vggt_slam/undistort.py) from its
    camera_front entry's distortion coefficients and cached; frames themselves are undistorted
    in-memory by vggt_slam/pipeline.py, only for the ones it selects as keyframes -- no second copy of the
    frames is ever written to disk. The RESULTING rectified K (not the pre-distortion one --
    undistortion changes it) is passed to the SLAM as its fixed intrinsics.
  - If not found (or its declared resolution can't be verified against the actual video):
    frames are left as-is and --shared_intrinsics is used instead (self-calibrate a single
    consensus K for the whole clip via the two-phase joint fit -- see
    vggt_slam/da3_ray_pose_patch.py).
  - use_ray_pose=True, overlapping_window_size=2, and ref_view_strategy="middle" are always
    used (not exposed as flags): --use_ray_pose is required for the above to have any effect
    (only DA3's ray-pose path can be locked/jointly refit; the camera-decoder path can't), and
    the other two are this project's own settled defaults for batch runs.
"""
import argparse
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

REPO_DIR = Path(__file__).resolve().parent.parent

# open3d's compiled pybind extension can fail to load in this container with "cannot allocate
# memory in static TLS block" (glibc's default static-TLS surplus is too small for aarch64 +
# torch/gtsam/open3d all claiming static TLS slots). Must be set here, before any gpu_worker
# subprocess is spawned: GLIBC_TUNABLES is read by the dynamic linker at process exec time, so
# setting it later (e.g. inside gpu_worker() itself, after its process already started) is too
# late to have any effect.
os.environ.setdefault("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=4000000")

sys.path.insert(0, str(REPO_DIR))
from vggt_slam.map_match import ensure_osmium
from vggt_slam.paths import DATASET_ROOT, SAMPLE_MANIFEST, TRAJ_ROOT
from vggt_slam.pipeline_params import SlamParams
from vggt_slam.regions import region_candidates
from vggt_slam.sample_manifest import source_videos
from vggt_slam.session_pack import session_zip_path
from vggt_slam.undistort import compute_undistort_maps, save_undistort_maps, session_cache_dir

_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd", b"edts"}
_VIDEO_SAMPLE_ENTRY_TYPES = {b"avc1", b"hvc1", b"hev1", b"mp4v", b"vp09", b"av01"}


def _walk_mp4_boxes(f, start, end, in_stsd=False):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            return None
        size = struct.unpack(">I", header[:4])[0]
        btype = header[4:8]
        header_len = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header_len = 16
        if size == 0:
            size = end - pos
        box_end = pos + size

        if in_stsd and btype in _VIDEO_SAMPLE_ENTRY_TYPES:
            # VisualSampleEntry: box header(8) + reserved(6) + data_ref_idx(2) + pre_defined(2)
            # + reserved(2) + pre_defined[3](12) = 24 bytes in, then width/height (2 uint16 BE).
            f.seek(pos + 8 + 6 + 2 + 2 + 2 + 12)
            dims = f.read(4)
            if len(dims) == 4:
                return struct.unpack(">HH", dims)
            return None
        if btype == b"stsd":
            result = _walk_mp4_boxes(f, pos + header_len + 8, box_end, in_stsd=True)
            if result:
                return result
        elif btype in _CONTAINER_BOXES:
            result = _walk_mp4_boxes(f, pos + header_len, box_end, in_stsd=in_stsd)
            if result:
                return result
        pos = box_end
    return None


def _read_mp4_dims(video_path):
    """Reads the encoded video stream's (width, height) straight from the MP4 box structure,
    without needing ffprobe. Only seeks/reads box headers (not frame data), so this stays cheap
    regardless of file size. Returns None if unparseable."""
    try:
        with open(video_path, "rb") as f:
            return _walk_mp4_boxes(f, 0, os.path.getsize(video_path))
    except Exception:
        return None


def _resolve_camera_meta(video_path):
    """Looks up this clip's camera_front intrinsics + distortion from a sibling
    FIXED_METADATA.json. Returns {"fx","fy","cx","cy","k1","k2","k3","p1","p2"} (used as-is, NOT
    rescaled) or None if metadata is missing, unparseable, or fails a sanity check against the
    actual video.

    FIXED_METADATA.json's own declared frame_width/height field cannot be trusted: for at least
    one clip it reported exactly half the true encoded resolution. But cam["fx"/"cx"/etc] were
    STILL correct as-is for the actual (not the declared) resolution in that case -- e.g. a
    clip with declared frame_width=1448 but actual encoded width 2896 had cx=1448, which is
    actual_width/2 exactly, not declared_width/2. So the fix is not to rescale fx/fy/cx/cy at
    all -- it's to validate them directly against the actual video (principal point should sit
    near the actual frame's center) instead of trusting the declared reference resolution.
    """
    try:
        meta_path = video_path.parents[2] / "FIXED_METADATA.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        cam = next((d for d in meta.get("device_extrinsics", []) if d.get("device_name") == "camera_front"), None)
        if cam is None:
            return None
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        actual = _read_mp4_dims(video_path)
        if not actual:
            return None
        aw, ah = actual
        if not (0.3 * aw < cx < 0.7 * aw and 0.3 * ah < cy < 0.7 * ah):
            return None  # principal point isn't near-centered on the actual frame -- don't trust it
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "k1": cam.get("k1", 0.0), "k2": cam.get("k2", 0.0), "k3": cam.get("k3", 0.0),
            "p1": cam.get("p1", 0.0), "p2": cam.get("p2", 0.0),
        }
    except Exception:
        return None


def _processed_clip_ids(out_root):
    """clip_ids (full source-video paths) with an ok row in <out_root>/manifest.jsonl."""
    done = set()
    path = Path(out_root) / "manifest.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("ok") is True:
                    done.add(r["clip_id"])
    return done


_ZIP_MEMBERS = {}


def _result_in_session_zip(out_dir):
    """True if out_dir's session was packed (vggt_slam.session_pack) and the clip's result.npz is
    inside that zip. A packed session can still be missing a clip that was added to the dataset
    later -- that clip is NOT done and gets processed (pack_sessions.py then merges it into the zip)."""
    zp = session_zip_path(out_dir)
    if not zp.exists():
        return False
    names = _ZIP_MEMBERS.get(zp)
    if names is None:
        with zipfile.ZipFile(zp) as zf:
            names = _ZIP_MEMBERS[zp] = set(zf.namelist())
    return str(out_dir.relative_to(session_cache_dir(out_dir)) / "result.npz") in names


def discover_clips(dataset_root, out_root, scratch_dir, countries, limit, force=False, osm_extracts_dir=None,
                   sample_manifest=SAMPLE_MANIFEST):
    """Streams clip dicts for every source video in the sample manifest that is not processed yet (a
    generator, so extraction can start on the first clip right away). A clip is already processed if it has
    a loose out_dir/result.npz, an ok row in <out_root>/manifest.jsonl, or its result.npz inside its
    session zip; --force reprocesses everything. Skipped clips never reach the _resolve_camera_meta lookup.

    frames_dir is rooted under scratch_dir (node-local), NOT under out_dir (the shared output
    filesystem) -- see --scratch_dir's help text for why.
    """
    dataset_root, out_root = Path(dataset_root), Path(out_root)
    videos = source_videos(sample_manifest, countries)
    done = set() if force else _processed_clip_ids(out_root)
    print(f"discover_clips: {len(videos)} source videos in {sample_manifest}"
          f"{' (countries: ' + ', '.join(countries) + ')' if countries else ''}; "
          f"{len(done)} clips have an ok row in {out_root / 'manifest.jsonl'}", flush=True)
    n_done = n_zip_only = n_new = n_into_packed = 0
    for rel_str in videos:
        rel = Path(rel_str)
        video_path = dataset_root / rel
        out_dir = out_root / rel.with_suffix("")
        if not force:
            if (out_dir / "result.npz").exists() or str(video_path) in done:
                n_done += 1
                continue
            if _result_in_session_zip(out_dir):  # result is in a session zip but has no manifest row
                n_done += 1
                n_zip_only += 1
                continue
        if session_zip_path(out_dir).exists():
            n_into_packed += 1  # new clip in an already-packed session; pack_sessions.py will merge it
        region_pbf_path, fallback_region_pbf_paths = region_candidates(rel, video_path, osm_extracts_dir)
        yield {
            "video_path": video_path,
            "out_dir": out_dir,
            "frames_dir": Path(scratch_dir) / rel.with_suffix("") / "frames",
            "camera_meta": _resolve_camera_meta(video_path),
            "osm_region_pbf_path": region_pbf_path,
            "osm_fallback_region_pbf_paths": fallback_region_pbf_paths,
        }
        n_new += 1
        if limit and n_new >= limit:
            return
    print(f"discover_clips: {n_done} already processed ({n_zip_only} of them only found inside a session zip, "
          f"without a manifest row), {n_new} to process ({n_into_packed} belong to an already-packed session)",
          flush=True)


def extract_one(clip, fps, ffmpeg_threads=0):
    frames_dir = clip["frames_dir"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not any(frames_dir.iterdir()):
        _t0 = time.time()
        try:
            subprocess.run(
                # -threads before -i sets ffmpeg's own decode parallelism (0 = let it pick;
                # decode is what actually scales with source resolution, e.g. France's
                # 2896x1876 vs 1280x960 for the other natix clips -- this was the single
                # biggest per-clip CPU cost observed for that clip).
                ["ffmpeg", "-y", "-threads", str(ffmpeg_threads), "-i", str(clip["video_path"]),
                 "-vf", f"fps={fps}", str(frames_dir / "frame_%05d.jpg")],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            return clip, e.stderr.decode(errors="replace")[-2000:]
        print(f"extract_one DIAG: ffmpeg extraction for {clip['video_path'].name} took "
              f"{time.time()-_t0:.2f}s", flush=True)

    cam_meta = clip["camera_meta"]
    if cam_meta is None:
        clip["image_folder"] = str(frames_dir)
        clip["fixed_intrinsics_str"] = None
        return clip, None

    # No second (undistorted) copy of the frames is ever written to disk: the remap tables only
    # depend on resolution + known distortion, not frame content, so they're computed once here
    # from a single frame's shape and cached; vggt_slam/pipeline.py loads them and undistorts in-memory, only
    # for the frames it actually keeps as keyframes (see vggt_slam/undistort.py's docstring).
    #
    # Cached at the SESSION level (session_cache_dir), not per-clip: every clip sharing one
    # FIXED_METADATA.json has identical camera_meta and (in practice) identical resolution, so
    # this ~10MB file was previously being redundantly recomputed and re-saved once per clip
    # (confirmed: avg 6.0 clips/session in Croatia's data) -- a major contributor to this
    # pipeline's inode/storage footprint at ~120k-clip scale. Concurrent workers processing
    # different clips of the same session may race here; save_undistort_maps writes atomically
    # so that's harmless (redundant compute, not corruption).
    session_dir = session_cache_dir(clip["out_dir"])
    session_dir.mkdir(parents=True, exist_ok=True)
    new_k_path = session_dir / "undistorted_intrinsics.json"
    maps_path = session_dir / "undistort_maps.npz"
    clip["image_folder"] = str(frames_dir)
    try:
        if not new_k_path.exists() or not maps_path.exists():
            first = cv2.imread(sorted(frames_dir.iterdir())[0].as_posix())
            h, w = first.shape[:2]
            map1, map2, roi, result = compute_undistort_maps(
                w, h,
                fx=cam_meta["fx"], fy=cam_meta["fy"], cx=cam_meta["cx"], cy=cam_meta["cy"],
                k1=cam_meta["k1"], k2=cam_meta["k2"], k3=cam_meta["k3"],
                p1=cam_meta["p1"], p2=cam_meta["p2"],
            )
            save_undistort_maps(str(maps_path), map1, map2, roi)
            # Same atomic-write rationale as save_undistort_maps -- a plain write_text here
            # would be racy now that this cache is shared across a session's concurrent workers.
            tmp_k_path = new_k_path.with_name(new_k_path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
            tmp_k_path.write_text(json.dumps(result))
            tmp_k_path.replace(new_k_path)
        else:
            result = json.loads(new_k_path.read_text())
        clip["undistort_maps_path"] = str(maps_path)
        clip["fixed_intrinsics_str"] = (
            f"{result['fx']},{result['fy']},{result['cx']},{result['cy']},"
            f"{result['width']},{result['height']}"
        )
    except Exception:
        # Undistortion failed (bad coefficients, corrupt frame, degenerate crop, ...) -- fall
        # back to raw frames + self-calibration rather than losing the clip entirely.
        clip["undistort_maps_path"] = None
        clip["fixed_intrinsics_str"] = None
        clip["undistort_warning"] = traceback.format_exc()[-2000:]
    return clip, None


def gpu_worker(gpu_id, run_cfg, job_queue, result_queue, keep_frames, gps_fusion, cpu_threads):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Without this, each worker process's PyTorch/MKL/OpenMP/OpenBLAS defaults to sizing its
    # intra-op thread pool off the node's FULL core count (288 here), not knowing it's one of
    # many worker processes sharing the node -- confirmed live at --workers_per_gpu 6 (24
    # workers): load average 364 on 288 cores, ~40k OS threads total, and individual clips
    # ballooning to 130-230s (vs a 40-90s norm) from pure scheduling contention, even though
    # nvidia-smi showed GPU SM% *rising* (workers' CPU-bound stages were just smeared across
    # more wall-clock, not actually finishing faster). Must be set before `import torch` --
    # these are read by the native libraries at first use, not by Python, so setting them any
    # later (e.g. via torch.set_num_threads after import) only fixes PyTorch's own pool, not
    # MKL/OpenBLAS/numexpr calls made underneath it.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(cpu_threads)
    sys.path.insert(0, str(REPO_DIR))

    import torch
    from vggt_slam import pipeline
    torch.set_num_threads(cpu_threads)

    backend = pipeline.build_backend(run_cfg["params"], "cuda")

    while True:
        _t_wait0 = time.time()
        job = job_queue.get()
        _wait_s = time.time() - _t_wait0
        if job is None:
            break
        print(f"gpu_worker DIAG: gpu{gpu_id} waited {_wait_s:.2f}s for next job "
              f"(extraction/undistortion not keeping up if this is large)", flush=True)

        video_path, out_dir, frames_dir = job["video_path"], job["out_dir"], job["frames_dir"]
        clip_id = str(video_path)
        pid = os.getpid()
        t0 = time.time()
        # Reported BEFORE any risky work, so the main process can tell which clip was in flight
        # if this whole worker process dies uncaught (OOM/SIGKILL, a fatal CUDA/driver error, a
        # native segfault, ...) -- none of which the try/except below can ever catch, since the
        # process itself is gone, not just the current Python call. Confirmed to happen in
        # practice: several clips ended up with a partial result.npz on disk (from the pipeline's
        # own SLAM-stage save) but no manifest.jsonl record at all, silently un-retriable forever
        # after since discover_clips's skip-check only looks at result.npz existing. See
        # _reap_dead_workers in main() for the other half of this.
        result_queue.put({"clip_id": clip_id, "gpu": gpu_id, "pid": pid, "status": "started",
                           "out_dir": str(out_dir)})
        try:
            if job.get("undistort_warning"):
                print(f"[WARN] {clip_id}: undistortion failed, using raw frames + "
                      f"--shared_intrinsics instead:\n{job['undistort_warning']}",
                      file=sys.stderr, flush=True)

            gps_csv = osm_region_pbf_path = osm_fallback_region_pbf_paths = None
            if gps_fusion:
                sibling_csv = video_path.with_suffix(".csv")
                if sibling_csv.exists():
                    gps_csv = str(sibling_csv)
                    if job.get("osm_region_pbf_path"):
                        osm_region_pbf_path = str(job["osm_region_pbf_path"])
                    if job.get("osm_fallback_region_pbf_paths"):
                        osm_fallback_region_pbf_paths = [str(p) for p in job["osm_fallback_region_pbf_paths"]]

            gps_fusion_info = pipeline.run_video(
                run_cfg["params"], backend, job["image_folder"], str(out_dir / "result.npz"), str(video_path),
                run_cfg["extraction_fps"], fixed_intrinsics=job["fixed_intrinsics_str"],
                shared_intrinsics=job["fixed_intrinsics_str"] is None,
                undistort_maps_path=job.get("undistort_maps_path"), dense_log=run_cfg["dense_log"],
                gps_csv=gps_csv, osm_region_pbf_path=osm_region_pbf_path,
                osm_fallback_region_pbf_paths=osm_fallback_region_pbf_paths, gps_max_dt=run_cfg["gps_max_dt"])

            if not keep_frames:
                # frames_dir's parent is the clip's whole scratch subtree (just frames/ now --
                # undistorted frames are never written to disk -- node-local, never under out_dir).
                shutil.rmtree(frames_dir.parent, ignore_errors=True)

            result_queue.put({"clip_id": clip_id, "gpu": gpu_id, "pid": pid, "ok": True,
                               "seconds": time.time() - t0,
                               "fixed_intrinsics": job["fixed_intrinsics_str"] is not None,
                               "gps_fusion": gps_fusion_info})
        except Exception:
            result_queue.put({"clip_id": clip_id, "gpu": gpu_id, "pid": pid, "ok": False,
                               "seconds": time.time() - t0, "error": traceback.format_exc()})


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=None, help="Single-clip mode: process just this one video file (requires --out_dir). Without --video, batch mode runs over --sample_manifest")
    p.add_argument("--out_dir", default=None, help="Single-clip mode: output dir for --video's result.npz/frames/etc")
    p.add_argument("--dataset_root", default=DATASET_ROOT, help="Batch mode: dataset root that the sample manifest's relative video paths are resolved against")
    p.add_argument("--out_root", default=TRAJ_ROOT, help="Batch mode: root dir to mirror the dataset structure into, one subdir per clip (manifest.jsonl is appended here)")
    p.add_argument("--sample_manifest", default=SAMPLE_MANIFEST, help="Batch mode: parquet (or hive-partitioned parquet dir) listing the clips to process; its source_video_path column names the source videos. Nothing walks the dataset tree")
    p.add_argument("--countries", nargs="*", default=None, help="Batch mode: restrict to these countries (first path component, e.g. France Latvia). Default: all")
    p.add_argument("--limit", type=int, default=None, help="Batch mode: only process the first N unprocessed clips of the manifest (for testing)")
    p.add_argument("--fps", type=float, default=10.0, help="ffmpeg frame-extraction fps")
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--workers_per_gpu", type=int, default=1,
                    help="GPU worker processes sharing each physical GPU (total workers = "
                         "num_gpus * workers_per_gpu). Per-clip GPU compute is bursty (~1s "
                         "forward pass per submap, the rest is CPU-side pipeline/orchestration "
                         "work -- see the module docstring), so a single worker per GPU often "
                         "leaves it idle between calls; raise this if nvidia-smi shows low "
                         "utilization with workers_per_gpu=1. Each worker loads its own full "
                         "model copy, so watch GPU memory (98GB/GPU on this cluster, comfortably "
                         "fits several)")
    p.add_argument("--num_extract_workers", type=int, default=64, help="Concurrent ffmpeg extraction workers")
    p.add_argument("--cpu_threads_per_worker", type=int, default=None,
                    help="OMP/MKL/OpenBLAS/torch intra-op thread cap for each GPU worker process. "
                         "Without this, every worker independently sizes its thread pool off the "
                         "node's FULL core count, unaware it's one of many co-resident worker "
                         "processes -- confirmed to cause severe scheduling contention (load "
                         "average far above core count, individual clips taking 2-4x longer) once "
                         "more than a few workers run concurrently (see gpu_worker's docstring). "
                         "Default: max(1, cpu_count // (num_gpus*workers_per_gpu + 1)), reserving "
                         "one worker's-worth of cores for the ffmpeg extraction pool")
    p.add_argument("--extract_lookahead", type=int, default=None,
                   help="Max clips allowed to be extracted-but-not-yet-GPU-consumed at once. "
                        "Extraction (num_extract_workers concurrent threads) is much faster "
                        "per-clip than a GPU worker's SLAM pass, so without a cap it races far "
                        "ahead and piles up hundreds of clips' worth of frames on --scratch_dir "
                        "before any are cleaned up (observed: 36GB/5min at the default settings). "
                        "Default: max(4, 2 * num_gpus * workers_per_gpu) -- enough that a GPU "
                        "worker never waits on extraction, without unbounded buildup.")
    p.add_argument("--ffmpeg_threads", type=int, default=0, help="ffmpeg -threads for decode (0 = let ffmpeg auto-pick)")
    p.add_argument("--keep_frames", action="store_true", help="Don't delete extracted/undistorted frames after a clip's poses are saved")
    p.add_argument("--scratch_dir", default="/dev/shm/vggt_slam_frames",
                   help="Node-local scratch dir for extracted/undistorted frames -- deliberately "
                        "NOT under --out_root/--out_dir. At dataset scale (~120k clips x hundreds "
                        "of frames each), writing/deleting that many small per-frame files on the "
                        "shared output filesystem would burn through its inode budget and hammer "
                        "its metadata server; a node-local tmpfs (RAM-backed, no shared inode cost) "
                        "avoids that entirely and is faster besides. Only small per-clip outputs "
                        "(result.npz, manifest.jsonl, gps fusion artifacts) ever reach --out_root.")
    p.add_argument("--gps_fusion", action=argparse.BooleanOptionalAction, default=True,
                    help="Also road-align via each clip's sibling FRONT_*.csv (extra per-clip OSM lookups). "
                         "On by default; pass --no-gps_fusion to skip if this bottlenecks a very large batch")
    p.add_argument("--force", action="store_true", help="Reprocess clips even if out_dir/result.npz already exists")
    p.add_argument("--dense_log", action="store_true",
                    help="Also write the full dense point cloud (result_points.pcd) per clip. Off by default "
                         "(too much storage across a full batch); meant for single-clip debugging runs")
    # SLAM settings (vggt_slam.pipeline_params.SlamParams); use_ray_pose / overlapping_window_size /
    # ref_view_strategy and the per-clip intrinsics handling are fixed policy for this script, not flags
    d = SlamParams()
    p.add_argument("--model_backend", default=d.model_backend, choices=["vggt", "da3"])
    p.add_argument("--da3_checkpoint", default=d.da3_checkpoint)
    p.add_argument("--submap_size", type=int, default=d.submap_size)
    p.add_argument("--max_loops", type=int, default=d.max_loops)
    p.add_argument("--min_disparity", type=float, default=d.min_disparity)
    p.add_argument("--conf_threshold", type=float, default=d.conf_threshold)
    p.add_argument("--lc_thres", type=float, default=d.lc_thres)
    p.add_argument("--gps_max_dt", type=float, default=0.5)
    p.add_argument("--osm_extracts_dir", type=str, default=None,
                   help="Dir of pre-downloaded whole-region OSM extracts, one per dataset country "
                        "(and per US state), named e.g. Austria.osm.pbf / 'United States_New "
                        "York.osm.pbf' -- see scripts/download_osm_extracts.py. GPS fusion requires "
                        "a matching extract for a clip's own region; if none is found here, that "
                        "clip's GPS fusion is skipped (non-fatally). Default in --dataset_root/"
                        "batch mode: {out_root}/_osm_extracts; in --video/single-clip mode: none "
                        "(use --osm_region_pbf_path directly instead, since there's no "
                        "dataset_root to resolve a country from)")
    p.add_argument("--osm_region_pbf_path", type=str, default=None,
                   help="--video/single-clip mode only: explicit region extract path to use "
                        "(bypasses --osm_extracts_dir's auto country-detection, which needs a "
                        "--dataset_root path to resolve a country from). Ignored in batch mode.")
    p.add_argument("--osm_fallback_region_pbf_path", type=str, default=None,
                   help="--video/single-clip mode only: tried once instead of "
                        "--osm_region_pbf_path if the GPS fixes' own crop finds zero roads there "
                        "(see FALLBACK_REGIONS for batch mode's per-country equivalent). Ignored "
                        "in batch mode.")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.video and not args.out_dir:
        parser.error("--video requires --out_dir")
    # without --video this is batch mode over --sample_manifest (dataset root / out root default to the natix locations)

    if args.gps_fusion:
        # GPS fusion's road-network crop needs osmium; a failed install only loses fusion for the clips
        # (each degrades gracefully), it must not abort a long batch
        ensure_osmium()

    osm_extracts_dir = args.osm_extracts_dir
    if osm_extracts_dir is None and not args.video:
        osm_extracts_dir = str(Path(args.out_root) / "_osm_extracts")

    if args.video:
        video_path = Path(args.video)
        out_dir = Path(args.out_dir)
        if not args.force and (out_dir / "result.npz").exists():
            print(f"{out_dir / 'result.npz'} already exists (use --force to reprocess)")
            return
        clips = [{
            "video_path": video_path,
            "out_dir": out_dir,
            "frames_dir": Path(args.scratch_dir) / out_dir.name / "frames",
            "camera_meta": _resolve_camera_meta(video_path),
            "osm_region_pbf_path": Path(args.osm_region_pbf_path) if args.osm_region_pbf_path else None,
            "osm_fallback_region_pbf_paths": (
                [Path(args.osm_fallback_region_pbf_path)] if args.osm_fallback_region_pbf_path else []),
        }]
        manifest_dir = out_dir
    else:
        # A generator, not a list: discover_clips streams clips as it goes, so extraction (see the
        # ThreadPoolExecutor submission loop below) can start on the first clip immediately.
        clips = discover_clips(args.dataset_root, args.out_root, args.scratch_dir, args.countries,
                                args.limit, args.force, osm_extracts_dir, args.sample_manifest)
        manifest_dir = Path(args.out_root)

    run_cfg = dict(
        params=SlamParams(
            model_backend=args.model_backend, da3_checkpoint=args.da3_checkpoint, submap_size=args.submap_size,
            max_loops=args.max_loops, min_disparity=args.min_disparity, conf_threshold=args.conf_threshold,
            lc_thres=args.lc_thres),
        extraction_fps=args.fps, gps_max_dt=args.gps_max_dt, dense_log=args.dense_log,
    )

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()

    total_workers = args.num_gpus * args.workers_per_gpu
    cpu_threads = args.cpu_threads_per_worker or max(1, (os.cpu_count() or 1) // (total_workers + 1))
    print(f"Capping each of {total_workers} GPU worker processes to {cpu_threads} CPU threads "
          f"(OMP/MKL/OpenBLAS/torch)", flush=True)

    workers = [
        ctx.Process(target=gpu_worker, args=(gpu_id, run_cfg, job_queue, result_queue,
                                              args.keep_frames, args.gps_fusion, cpu_threads))
        for gpu_id in range(args.num_gpus)
        for _ in range(args.workers_per_gpu)
    ]
    for w in workers:
        w.start()

    manifest_path = manifest_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_f = open(manifest_path, "a")

    done = 0
    submitted = 0  # clips submitted for extraction so far -- keeps growing until discovery finishes
    in_progress = {}  # worker pid -> {"clip_id", "out_dir"} for whatever job it's currently on,
                       # if any -- see _reap_dead_workers below.

    lookahead = args.extract_lookahead or max(4, 2 * len(workers))
    # Bounds how many clips can be extracted-but-not-yet-GPU-consumed at once: acquired before
    # submitting a clip for extraction, released once that clip's GPU worker finishes (its frames
    # are deleted at that point) or its extraction fails outright. Without this, extraction races
    # far ahead of the (much slower per-clip) GPU workers and piles up unboundedly on scratch_dir.
    extract_slot = threading.Semaphore(lookahead)

    def _reap_dead_workers():
        # Catches the one failure mode gpu_worker's own try/except can't: the whole process dying
        # uncaught (OOM/SIGKILL, a fatal CUDA/driver error, a native segfault, ...) while a clip
        # was in flight -- that clip's job never gets a completion message at all, and without
        # this it would just be silently dropped: no manifest.jsonl record, and whatever partial
        # result.npz the pipeline's own SLAM-stage save left behind on disk would make every future
        # run's discover_clips skip-check treat it as already done, forever. Confirmed to happen
        # in practice this way.
        nonlocal done
        for w in workers:
            if w.is_alive():
                continue
            lost = in_progress.pop(w.pid, None)
            if lost is None:
                continue  # died cleanly after its last job (or never got one) -- nothing lost
            done += 1
            extract_slot.release()
            err = (f"gpu_worker process (pid={w.pid}) died without completing this clip "
                   f"(exitcode={w.exitcode}) -- likely an OOM/SIGKILL or a fatal CUDA/driver "
                   f"crash, neither catchable as a Python exception")
            manifest_f.write(json.dumps({"clip_id": lost["clip_id"], "ok": False,
                                          "error": err}) + "\n")
            manifest_f.flush()
            print(f"[{done}/{submitted}] FAIL (worker died) {lost['clip_id']}: {err}",
                  file=sys.stderr, flush=True)
            # So a future run's discover_clips naturally retries this clip instead of silently
            # treating the crash's own partial output as a completed run.
            out_dir = Path(lost["out_dir"])
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)

    def drain_results(block=False):
        nonlocal done
        while True:
            try:
                res = result_queue.get(block=block, timeout=1 if block else None)
            except Exception:
                break
            if res.get("status") == "started":
                in_progress[res["pid"]] = {"clip_id": res["clip_id"], "out_dir": res["out_dir"]}
                continue
            in_progress.pop(res["pid"], None)
            done += 1
            extract_slot.release()
            # gpu / pid only identify the worker process (used for the progress line below and for crash
            # bookkeeping); they are not part of the manifest row
            manifest_f.write(json.dumps({k: v for k, v in res.items() if k not in ("gpu", "pid")}) + "\n")
            manifest_f.flush()
            status = "OK" if res["ok"] else "FAIL"
            print(f"[{done}/{submitted}] {status} gpu{res['gpu']} {res['seconds']:.1f}s {res['clip_id']}", flush=True)
            if not res["ok"]:
                print(res["error"], file=sys.stderr, flush=True)
        _reap_dead_workers()

    print(f"Discovering clips and extracting frames with {args.num_extract_workers} workers "
          f"(ffmpeg_threads={args.ffmpeg_threads} each), "
          f"feeding {len(workers)} GPU workers ({args.num_gpus} GPUs x {args.workers_per_gpu}, "
          f"extract_lookahead={lookahead})...", flush=True)

    fixed_count, shared_count = 0, 0

    def _tap(c):
        nonlocal fixed_count, shared_count
        if c["camera_meta"]:
            fixed_count += 1
        else:
            shared_count += 1
        return c

    # extract_slot.acquire() (in feeder, below) can only unblock once a GPU result comes back
    # through drain_results -- so submission must run on its own thread, concurrently with the
    # main thread draining extraction results, not gated behind a single fully-submitted batch
    # (submitting everything upfront in one comprehension would deadlock: nothing could ever
    # release a permit until submission finished, and submission can't finish without a permit).
    extract_done_q = queue.Queue()

    def feeder(pool):
        nonlocal submitted
        for c in clips:
            extract_slot.acquire()
            submitted += 1
            pool.submit(extract_one, _tap(c), args.fps, args.ffmpeg_threads) \
                .add_done_callback(lambda fut: extract_done_q.put(fut.result()))
        extract_done_q.put(None)  # sentinel: discovery + submission is fully done

    with ThreadPoolExecutor(max_workers=args.num_extract_workers) as pool:
        feeder_thread = threading.Thread(target=feeder, args=(pool,), daemon=True)
        feeder_thread.start()

        discovery_done = False
        extract_completed = 0
        while not (discovery_done and extract_completed >= submitted):
            # A plain, untimed extract_done_q.get() deadlocks once all extract_lookahead slots
            # are in flight and their GPU results finish before the corresponding NEXT extraction
            # (which can't even start yet -- feeder is blocked on extract_slot.acquire()) --
            # drain_results (the only thing that releases a slot) was only ever called from right
            # here, so nothing wakes this loop to call it once there's no *new* extraction to
            # react to. Confirmed live: all 361 threads in this process sitting idle, extraction
            # pool completely quiet, feeder permanently blocked on acquire(). A timeout makes this
            # loop periodically drain result_queue and free up slots even with no extraction
            # event to react to, which is what actually keeps the feeder unblocked.
            try:
                item = extract_done_q.get(timeout=1.0)
            except queue.Empty:
                drain_results(block=False)
                continue
            if item is None:
                discovery_done = True
                print(f"{submitted} clips discovered total ({fixed_count} --fixed_intrinsics, "
                      f"{shared_count} --shared_intrinsics)", flush=True)
                continue
            extract_completed += 1
            clip, err = item
            if err:
                extract_slot.release()
                done += 1
                manifest_f.write(json.dumps({"clip_id": str(clip["video_path"]), "ok": False,
                                              "stage": "extract", "error": err}) + "\n")
                manifest_f.flush()
                print(f"[{done}/{submitted}] FAIL extract {clip['video_path']}: {err}", file=sys.stderr, flush=True)
                continue
            job_queue.put(clip)
            drain_results(block=False)

        feeder_thread.join()

    for _ in workers:
        job_queue.put(None)

    while done < submitted and any(w.is_alive() for w in workers):
        drain_results(block=True)
    drain_results(block=False)

    for w in workers:
        w.join()
    _reap_dead_workers()  # final catch-all in case the very last worker died right at this boundary

    manifest_f.close()
    print(f"Done. {done}/{submitted} accounted for. See {manifest_path}")


if __name__ == "__main__":
    main()
