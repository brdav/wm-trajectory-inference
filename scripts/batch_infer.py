#!/usr/bin/env python3
"""Run VGGT-SLAM trajectory inference on a video clip, or batch over many clips using all GPUs
on the node in parallel. The sole entry point for turning a natix-style clip (or dataset of
them) into poses -- scripts/run_video_to_poses.sh is now a thin wrapper around this for the
single-clip case (clip-name shortcuts + optional GPS-fusion plotting). Designed to be launched
once inside an allocation that already has the vggtslam_image container attached, e.g.:

    # Single clip:
    srun --environment=vggtslam_env python scripts/batch_infer.py \\
        --video /capstor/.../FRONT_FOLDER/FRONT_2025-09-10_17-04-42.mp4 \\
        --out_dir /capstor/store/cscs/swissai/ab036/.../poses_out/france \\
        --num_gpus 1

    # Whole dataset:
    srun --environment=vggtslam_env python scripts/batch_infer.py \\
        --dataset_root /capstor/store/cscs/swissai/ab036/datasets/natix \\
        --out_root /capstor/store/cscs/swissai/ab036/.../poses_out \\
        --num_gpus 4

Architecture (see conversation / README for the "why"):
  - One persistent GPU worker process per GPU (--num_gpus x --workers_per_gpu), each loading
    the geometry model ONCE and then looping over many clips pulled from a shared queue --
    this avoids the model-reload cost that a fresh `python main.py` per clip would pay on every
    single clip (still paid once, unavoidably, in run_video_to_poses.sh's single-clip mode).
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
  - Already-completed clips (an existing result.npz) are skipped, so a run that hits the
    Slurm wall-time limit can simply be resubmitted to pick up where it left off.
  - GPS/road-fusion (main.py's --gps_csv path) is ON by default: it map-matches against live
    OSM data per clip, using each video's sibling FRONT_*.csv automatically (no registry --
    just same basename, .csv extension). This is a per-clip external-network dependency, so
    pass --no-gps_fusion if it bottlenecks a very large batch.

Per-clip intrinsics policy (always on, not configurable -- see conversation notes):
  - Every clip is checked for a sibling FIXED_METADATA.json (natix trip-dir convention:
    .../<trip_dir>/<timestamp>/FRONT_FOLDER/video.mp4, metadata one level above <timestamp>),
    cross-checked against the video's own encoded resolution (FIXED_METADATA.json's own
    frame_width/height has been observed to be exactly half the true encoded resolution for at
    least one clip, so it's never trusted on its own).
  - If found: undistortion remap tables are computed once (vggt_slam/undistort.py) from its
    camera_front entry's distortion coefficients and cached; frames themselves are undistorted
    in-memory by main.py, only for the ones it selects as keyframes -- no second copy of the
    frames is ever written to disk. The RESULTING rectified K (not the pre-distortion one --
    undistortion changes it) is passed to main.py as --fixed_intrinsics.
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
import csv
import json
import math
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
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
from vggt_slam import country_regions, us_regions
from vggt_slam.session_pack import is_session_packed
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


def _iter_front_folder_mp4s(root):
    # os.walk + per-directory sort instead of sorted(root.rglob(...)): rglob's result must be
    # fully materialized before sorted() can return anything, which forces a complete walk of
    # the entire (~120k-clip, 28-country) tree before even one clip is available -- with
    # --limit or without, that leaves all 4 GPUs idle for the whole scan. Sorting only within
    # each directory keeps traversal order deterministic at negligible cost and lets the caller
    # start processing the first clip immediately.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if os.path.basename(dirpath) == "FRONT_FOLDER":
            for fname in sorted(filenames):
                if fname.endswith(".mp4"):
                    yield Path(dirpath) / fname


# Some clips are filed under the wrong country folder in this dataset (confirmed: an "Austria"
# recording whose GPS fixes are actually near Stolac, Bosnia and Herzegovina, right on the
# Croatia border -- its own crop finds zero roads in Austria's extract). Mapping is
# dataset-folder-name -> ordered list of fallback region_keys to try (same keying as
# _region_pbf_path/download_osm_extracts.py); add more entries as mislabeled clips are found.
# Croatia is listed first for Austria: confirmed via direct spatial-index query that this specific
# border area's roads are covered by Croatia's extract, not Bosnia's (Geofabrik's per-country
# extracts can leave a gap right at a border) -- Bosnia stays as a second candidate for any other
# mislabeled "Austria" clip that lands further from that border.
FALLBACK_REGIONS = {
    "Austria": ["Croatia", "Bosnia and Herzegovina"],
}


def _region_name(rel_path):
    country = rel_path.parts[0]
    return f"{country}_{rel_path.parts[1]}" if country == "United States" else country


def _region_pbf_path(rel_path, osm_extracts_dir, region_name=None):
    """rel_path: a clip's video path relative to dataset_root. Returns the pre-downloaded
    region extract's path (see scripts/download_osm_extracts.py's naming -- "United States"
    clips are keyed by "United States_<State>", every other country by its own name directly)
    if osm_extracts_dir is set AND that region's extract actually exists there, else None (GPS
    fusion is required to have a region extract -- see main.py's --osm_region_pbf_path -- so a
    clip whose region isn't covered here just has its GPS fusion skipped, non-fatally).

    region_name: override the region key resolved from rel_path -- used to look up a fallback
    region (see FALLBACK_REGIONS) instead of the clip's own labeled region.
    """
    if not osm_extracts_dir:
        return None
    region_name = region_name or _region_name(rel_path)
    candidate = Path(osm_extracts_dir) / f"{region_name}.osm.pbf"
    return candidate if candidate.exists() else None


def _fallback_region_pbf_paths(rel_path, osm_extracts_dir):
    """Returns the ordered list of candidate fallback extract paths that actually exist on disk
    (see FALLBACK_REGIONS) -- align_trajectory_to_roads tries each in turn until one has roads
    near the clip's GPS fixes. Empty list if this region has no configured fallback."""
    fallback_regions = FALLBACK_REGIONS.get(_region_name(rel_path), [])
    paths = (_region_pbf_path(rel_path, osm_extracts_dir, region_name=r) for r in fallback_regions)
    return [p for p in paths if p is not None]


def _read_latlon_centroid(csv_path):
    """Cheap best-effort mean (lat, lon) of a natix FRONT_*.csv's real (non-'na') GPS fixes, or
    None if the file is missing/unreadable or has no real fixes. Deliberately NOT
    vggt_slam.gps.load_gps_fixes -- that also does frame-number/fps alignment and a UTM
    projection that this doesn't need, and discover_clips calls this for every not-yet-processed
    US clip on every run, so keeping it a minimal single-pass CSV scan matters.
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            lats, lons = [], []
            for row in reader:
                if row.get("GPS_latitude_deg", "na") == "na":
                    continue
                lats.append(float(row["GPS_latitude_deg"]))
                lons.append(float(row["GPS_longitude_deg"]))
    except (OSError, KeyError, ValueError):
        return None
    if not lats:
        return None
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _all_region_candidates(video_path, osm_extracts_dir):
    """GPS-centroid-based region resolution across BOTH US states and every other downloaded
    country, searched TOGETHER as one combined, distance-ranked list -- regardless of which
    folder (US state or country) the clip happens to be filed under. Some clips are filed under
    the wrong region entirely, including across the US/country boundary itself (confirmed on real
    clips: several "United States"-labeled clips actually in Puerto Rico -- which lives in
    vggt_slam/country_regions.py's table since its extract is named like a country ("Puerto
    Rico.osm.pbf"), not a state, so a search restricted to US states alone could never reach it;
    also a "Canada"-labeled clip sitting right on the US border). Searching one merged space
    instead of two separate ones (US-only vs. country-only) means neither direction of that
    mislabeling needs its own special case.

    Returns (primary_path, fallback_paths) ordered by distance from the clip's own GPS centroid,
    restricted to regions that actually have a downloaded extract; (None, []) if the clip's GPS
    centroid can't be determined (missing/empty/all-'na' CSV) -- caller falls back to folder-based
    resolution in that case.
    """
    centroid = _read_latlon_centroid(video_path.with_suffix(".csv"))
    if centroid is None:
        return None, []
    lat, lon = centroid
    coslat = math.cos(math.radians(lat))

    def dist2(clat, clon):
        dlat, dlon = lat - clat, (lon - clon) * coslat
        return dlat * dlat + dlon * dlon

    entries = [(dist2(clat, clon), f"United States_{state}.osm.pbf")
               for state, (clat, clon) in us_regions.US_STATE_CENTROIDS.items()]
    entries += [(dist2(clat, clon), f"{country}.osm.pbf")
                for country, (clat, clon) in country_regions.COUNTRY_CENTROIDS.items()]
    entries.sort(key=lambda e: e[0])

    candidates = []
    for _, name in entries:
        p = Path(osm_extracts_dir) / name
        if p.exists():
            candidates.append(p)
    if not candidates:
        return None, []
    return candidates[0], candidates[1:]


def _region_candidates(rel_path, video_path, osm_extracts_dir):
    """Unified region-selection entry point used by both discover_clips (batch_infer.py) and
    refuse_gps.py's --resume re-fusion path. Returns (primary_path_or_None, fallback_paths).
    """
    if not osm_extracts_dir:
        return None, []
    primary, fallback = _all_region_candidates(video_path, osm_extracts_dir)
    if primary is not None:
        return primary, fallback
    # No usable GPS centroid (e.g. all-'na' CSV) -- degrade to the folder-labeled region rather
    # than skipping region resolution entirely; main.py/refuse_gps.py's own "No real GPS fixes"
    # handling already covers the case where GPS fusion can't run at all, so this fallback (and
    # FALLBACK_REGIONS, e.g. Austria->Croatia/Bosnia) is only actually exercised for the rarer
    # partial-GPS-data case now that the GPS-centroid path above handles the common one.
    return _region_pbf_path(rel_path, osm_extracts_dir), _fallback_region_pbf_paths(rel_path, osm_extracts_dir)


VIDEO_PATHS_CACHE_NAME = "all_video_paths.txt"


def _discover_all_video_paths(dataset_root, out_root):
    """Streams every FRONT_*.mp4 under dataset_root, same order as _iter_front_folder_mp4s.
    Uses a cached flat-text manifest at out_root/all_video_paths.txt if present instead of
    walking dataset_root -- run_full_dataset.sbatch resubmits itself roughly every 12h until the
    whole ~120k-clip backlog is done (dozens of times over a full run), and dataset_root is the
    delivered, effectively-immutable source dataset, so re-walking its entire tree on every one
    of those resubmissions re-pays the same cost for the same, unchanging answer. If the dataset
    ever grows with genuinely new clips, delete this cache file to force a fresh walk.

    Written incrementally as the walk actually happens (line-by-line, flushed) rather than
    materializing the whole list before yielding the first path -- so a cache MISS doesn't lose
    _iter_front_folder_mp4s's own streaming property (starting extraction on the first clip
    before the rest of the tree is walked, see its docstring). The cache file itself is only
    committed (atomic os.replace) once the walk actually finishes, so a process killed mid-walk
    (confirmed to happen on this cluster) leaves no cache at all rather than a silently
    incomplete one -- the next invocation just re-walks from scratch, same as today.
    """
    cache_path = Path(out_root) / VIDEO_PATHS_CACHE_NAME
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield Path(line)
        return

    print(f"discover_clips: no cached video-path manifest at {cache_path} -- walking "
          f"{dataset_root} once and caching the result for every future invocation.",
          flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + f".tmp-{os.getpid()}")
    n = 0
    with open(tmp_path, "w") as f:
        for video_path in _iter_front_folder_mp4s(Path(dataset_root)):
            f.write(str(video_path) + "\n")
            f.flush()
            n += 1
            yield video_path
    os.replace(tmp_path, cache_path)
    print(f"discover_clips: cached {n} video paths to {cache_path}", flush=True)


def discover_clips(dataset_root, out_root, scratch_dir, countries, limit, force=False, osm_extracts_dir=None):
    """Streams clip dicts as they're found (a generator, not a list) so extraction can start on
    the first clip while the rest of the tree is still being walked (see
    _discover_all_video_paths). Already-processed clips (out_dir/result.npz exists) are skipped
    here, before the _resolve_camera_meta lookup, unless --force is set.

    frames_dir is rooted under scratch_dir (node-local), NOT under out_dir (the shared output
    filesystem) -- see --scratch_dir's help text for why.
    """
    dataset_root = Path(dataset_root)
    country_set = set(countries) if countries else None
    count = 0
    for video_path in _discover_all_video_paths(dataset_root, out_root):
        rel = video_path.relative_to(dataset_root)
        if country_set is not None and rel.parts[0] not in country_set:
            continue
        out_dir = Path(out_root) / rel.with_suffix("")
        # is_session_packed: a clip whose whole session got archived by scripts/pack_sessions.py
        # has no out_dir on disk at all any more (see vggt_slam/session_pack.py) -- without this
        # check that looks identical to never-processed and this would silently redo it.
        if not force and ((out_dir / "result.npz").exists() or is_session_packed(out_dir)):
            continue
        region_pbf_path, fallback_region_pbf_paths = _region_candidates(rel, video_path, osm_extracts_dir)
        yield {
            "video_path": video_path,
            "out_dir": out_dir,
            "frames_dir": Path(scratch_dir) / rel.with_suffix("") / "frames",
            "camera_meta": _resolve_camera_meta(video_path),
            "osm_region_pbf_path": region_pbf_path,
            "osm_fallback_region_pbf_paths": fallback_region_pbf_paths,
        }
        count += 1
        if limit and count >= limit:
            return


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
    # from a single frame's shape and cached; main.py loads them and undistorts in-memory, only
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
            tmp_k_path = new_k_path.with_name(new_k_path.name + f".tmp-{os.getpid()}")
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


def gpu_worker(gpu_id, base_args, job_queue, result_queue, keep_frames, gps_fusion, cpu_threads):
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

    import argparse as _argparse
    import torch
    import main as vggt_main
    torch.set_num_threads(cpu_threads)

    device = "cuda"
    args = _argparse.Namespace(**base_args)
    backend, clip_model, clip_preprocess, clip_tokenizer, processor = vggt_main.build_backend(args, device)

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
        # practice: several clips ended up with a partial result.npz on disk (from vggt_main's
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

            clip_args = _argparse.Namespace(**base_args)
            clip_args.image_folder = job["image_folder"]
            clip_args.fixed_intrinsics = job["fixed_intrinsics_str"]
            clip_args.shared_intrinsics = job["fixed_intrinsics_str"] is None
            clip_args.undistort_maps_path = job.get("undistort_maps_path")
            clip_args.log_path = str(out_dir / "result.npz")
            clip_args.video = str(video_path)
            clip_args.gps_csv = None
            clip_args.osm_region_pbf_path = None
            clip_args.osm_fallback_region_pbf_path = None
            if gps_fusion:
                sibling_csv = video_path.with_suffix(".csv")
                if sibling_csv.exists():
                    clip_args.gps_csv = str(sibling_csv)
                    if job.get("osm_region_pbf_path"):
                        clip_args.osm_region_pbf_path = str(job["osm_region_pbf_path"])
                    if job.get("osm_fallback_region_pbf_paths"):
                        clip_args.osm_fallback_region_pbf_path = [
                            str(p) for p in job["osm_fallback_region_pbf_paths"]]

            gps_fusion_info = vggt_main.run_video(
                clip_args, backend, device, clip_model, clip_preprocess, clip_tokenizer, processor)

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
    p.add_argument("--video", default=None, help="Single-clip mode: process just this one video file (requires --out_dir). Mutually exclusive with --dataset_root/--out_root")
    p.add_argument("--out_dir", default=None, help="Single-clip mode: output dir for --video's result.npz/frames/etc")
    p.add_argument("--dataset_root", default=None, help="Batch mode: root dir to search for */FRONT_FOLDER/*.mp4 (requires --out_root). Mutually exclusive with --video/--out_dir")
    p.add_argument("--out_root", default=None, help="Batch mode: root dir to mirror dataset_root's structure into, one subdir per clip")
    p.add_argument("--countries", nargs="*", default=None, help="Batch mode: restrict to these dataset_root subdirs (e.g. France Latvia). Default: all")
    p.add_argument("--limit", type=int, default=None, help="Batch mode: only process the first N discovered clips (for testing)")
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
    # Passthrough knobs mirroring main.py's own defaults; batch runs never use
    # vis_map/vis_imgs/run_os/vis_flow (headless) or dense point-cloud logging (too much
    # storage across many clips). use_ray_pose/overlapping_window_size/ref_view_strategy and
    # per-clip intrinsics handling are fixed policy for this script, not exposed as flags --
    # see the module docstring.
    p.add_argument("--model_backend", default="da3", choices=["vggt", "da3"])
    p.add_argument("--da3_checkpoint", default="depth-anything/DA3-GIANT-1.1")
    p.add_argument("--submap_size", type=int, default=16)
    p.add_argument("--max_loops", type=int, default=1)
    p.add_argument("--min_disparity", type=float, default=25)
    p.add_argument("--conf_threshold", type=float, default=25.0)
    p.add_argument("--lc_thres", type=float, default=0.95)
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

    if bool(args.video) == bool(args.dataset_root):
        parser.error("pass either --video (with --out_dir) for a single clip, or --dataset_root "
                      "(with --out_root) for batch mode")
    if args.video and not args.out_dir:
        parser.error("--video requires --out_dir")
    if args.dataset_root and not args.out_root:
        parser.error("--dataset_root requires --out_root")

    if args.gps_fusion:
        # Needed by vggt_slam.map_match._extract_local_bbox for any clip whose region has a
        # pre-downloaded extract (see --osm_extracts_dir). This container's package installs
        # don't persist across separate `srun` invocations, but DO persist for every process
        # (including multiprocessing-spawned GPU workers) within this one -- so install once,
        # here, before those workers start. Not check=True: a compute node with no/flaky
        # internet (observed: PyPI DNS failures here) must not crash the entire multi-hour batch
        # over this one optional package (map_match.py's own docstring: "not a hard dependency")
        # -- gpu_worker's per-clip try/except already degrades a single clip gracefully if
        # `import osmium` ends up failing, so worst case here is losing GPS fusion, not the run.
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "--index-url", "https://pypi.org/simple", "--no-deps", "osmium"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] pip install osmium failed ({e}) -- GPS fusion will fail per-clip if "
                  "osmium isn't already importable in this container", file=sys.stderr, flush=True)

    osm_extracts_dir = args.osm_extracts_dir
    if osm_extracts_dir is None and args.dataset_root:
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
            "osm_fallback_region_pbf_path": (
                Path(args.osm_fallback_region_pbf_path) if args.osm_fallback_region_pbf_path else None),
        }]
        manifest_dir = out_dir
    else:
        # A generator, not a list: discover_clips streams clips as os.walk finds them, so
        # extraction (see the ThreadPoolExecutor submission loop below) can start on the first
        # clip immediately instead of waiting for the whole ~120k-clip tree to be enumerated
        # first -- see discover_clips's docstring.
        clips = discover_clips(args.dataset_root, args.out_root, args.scratch_dir, args.countries,
                                args.limit, args.force, osm_extracts_dir)
        manifest_dir = Path(args.out_root)

    base_args = dict(
        model_backend=args.model_backend, da3_checkpoint=args.da3_checkpoint,
        use_ray_pose=True, ref_view_strategy="middle",
        vis_map=False, vis_imgs=False, vis_voxel_size=None, run_os=False, vis_flow=False,
        headless=True,
        log_results=True, skip_dense_log=True, log_path="result.npz",
        submap_size=args.submap_size, overlapping_window_size=2,
        max_loops=args.max_loops, min_disparity=args.min_disparity,
        conf_threshold=args.conf_threshold, lc_thres=args.lc_thres,
        gps_csv=None, extraction_fps=args.fps, gps_max_dt=args.gps_max_dt,
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
        ctx.Process(target=gpu_worker, args=(gpu_id, base_args, job_queue, result_queue,
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
        # result.npz vggt_main's own SLAM-stage save left behind on disk would make every future
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
            manifest_f.write(json.dumps({"clip_id": lost["clip_id"], "gpu": None, "ok": False,
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
            manifest_f.write(json.dumps(res) + "\n")
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
