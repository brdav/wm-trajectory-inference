#!/usr/bin/env python3
"""Run VGGT-SLAM trajectory inference on a video clip, or batch over many clips using all GPUs
on the node in parallel. The sole entry point for turning a natix-style clip (or dataset of
them) into poses: a single clip (--video/--out_dir) or every video of the sample manifest.
Designed to be launched
once inside an allocation that already has the vggtslam_image container attached, e.g.:

    # Single clip:
    srun --environment=vggtslam_env_ext python scripts/batch_infer.py \\
        --video /capstor/.../FRONT_FOLDER/FRONT_2025-09-10_17-04-42.mp4 \\
        --out_dir /capstor/store/cscs/swissai/ab036/.../poses_out/france \\
        --num_gpus 1

    # Everything listed in the dataset's sample manifest (what scripts/run_full_dataset.sbatch does; the
    # manifest, dataset root and output root default to the natix locations, see vggt_slam/sample_manifest.py):
    srun --environment=vggtslam_env_ext python scripts/batch_infer.py --num_gpus 4

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
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent

# open3d's compiled pybind extension can fail to load in this container with "cannot allocate
# memory in static TLS block" (glibc's default static-TLS surplus is too small for aarch64 +
# torch/gtsam/open3d all claiming static TLS slots). Must be set here, before any gpu_worker
# subprocess is spawned: GLIBC_TUNABLES is read by the dynamic linker at process exec time, so
# setting it later (e.g. inside gpu_worker() itself, after its process already started) is too
# late to have any effect.
os.environ.setdefault("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=4000000")

sys.path.insert(0, str(REPO_DIR))
from vggt_slam.clip_prep import extract_one, resolve_camera_meta
from vggt_slam.paths import DATASET_ROOT, SAMPLE_MANIFEST, TRAJ_ROOT
from vggt_slam.pipeline_params import SlamParams
from vggt_slam.regions import region_candidates
from vggt_slam.sample_manifest import source_videos
from vggt_slam.session_pack import session_zip_path
from vggt_slam.undistort import session_cache_dir

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


def clips_to_process(dataset_root, out_root, scratch_dir, countries, limit, force=False, osm_extracts_dir=None,
                      sample_manifest=SAMPLE_MANIFEST):
    """Streams clip dicts for every source video in the sample manifest that is not processed yet (a
    generator, so extraction can start on the first clip right away). A clip is already processed if it has
    a loose out_dir/result.npz, an ok row in <out_root>/manifest.jsonl, or its result.npz inside its
    session zip; --force reprocesses everything. Skipped clips never reach the resolve_camera_meta lookup.

    frames_dir is rooted under scratch_dir (node-local), NOT under out_dir (the shared output
    filesystem) -- see --scratch_dir's help text for why.
    """
    dataset_root, out_root = Path(dataset_root), Path(out_root)
    videos = source_videos(sample_manifest, countries)
    done = set() if force else _processed_clip_ids(out_root)
    print(f"clips_to_process: {len(videos)} source videos in {sample_manifest}"
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
            "camera_meta": resolve_camera_meta(video_path),
            "osm_region_pbf_path": region_pbf_path,
            "osm_fallback_region_pbf_paths": fallback_region_pbf_paths,
        }
        n_new += 1
        if limit and n_new >= limit:
            return
    print(f"clips_to_process: {n_done} already processed ({n_zip_only} of them only found inside a session zip, "
          f"without a manifest row), {n_new} to process ({n_into_packed} belong to an already-packed session)",
          flush=True)


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
        # after since clips_to_process's skip-check only looks at result.npz existing. See
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


class BatchRun:
    """One batch: frames are extracted on a thread pool, fed to the GPU worker processes, and every outcome
    (ok, failed, worker died, extraction failed) is appended to <manifest_dir>/manifest.jsonl."""

    def __init__(self, args, clips, run_cfg, manifest_dir):
        self.args = args
        self.clips = clips
        self.run_cfg = run_cfg
        self.manifest_path = manifest_dir / "manifest.jsonl"
        self.done = 0
        self.submitted = 0  # clips submitted for extraction so far -- keeps growing until the manifest list is exhausted
        self.fixed_count = self.shared_count = 0
        self.in_progress = {}  # worker pid -> {"clip_id", "out_dir"} of the job it is on, see _reap_dead_workers
        self.feeder_error = None
        self.extract_done_q = queue.Queue()
        self.workers = []
        self.manifest_f = None
        self.extract_slot = None

    # -- workers and manifest ------------------------------------------------------------------

    def _start_workers(self):
        import multiprocessing as mp
        args = self.args
        ctx = mp.get_context("spawn")
        self.job_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        total_workers = args.num_gpus * args.workers_per_gpu
        cpu_threads = args.cpu_threads_per_worker or max(1, (os.cpu_count() or 1) // (total_workers + 1))
        print(f"Capping each of {total_workers} GPU worker processes to {cpu_threads} CPU threads "
              f"(OMP/MKL/OpenBLAS/torch)", flush=True)
        self.workers = [
            ctx.Process(target=gpu_worker, args=(gpu_id, self.run_cfg, self.job_queue, self.result_queue,
                                                  args.keep_frames, args.gps_fusion, cpu_threads))
            for gpu_id in range(args.num_gpus)
            for _ in range(args.workers_per_gpu)
        ]
        for w in self.workers:
            w.start()

    def _record(self, row):
        self.manifest_f.write(json.dumps(row) + "\n")
        self.manifest_f.flush()

    def _reap_dead_workers(self):
        # Catches the one failure mode gpu_worker's own try/except can't: the whole process dying
        # uncaught (OOM/SIGKILL, a fatal CUDA/driver error, a native segfault, ...) while a clip
        # was in flight -- that clip's job never gets a completion message at all, and without
        # this it would just be silently dropped: no manifest.jsonl record, and whatever partial
        # result.npz the pipeline's own SLAM-stage save left behind on disk would make every future
        # run's clips_to_process skip-check treat it as already done, forever.
        for w in self.workers:
            if w.is_alive():
                continue
            lost = self.in_progress.pop(w.pid, None)
            if lost is None:
                continue  # died cleanly after its last job (or never got one) -- nothing lost
            self.done += 1
            self.extract_slot.release()
            err = (f"gpu_worker process (pid={w.pid}) died without completing this clip "
                   f"(exitcode={w.exitcode}) -- likely an OOM/SIGKILL or a fatal CUDA/driver "
                   f"crash, neither catchable as a Python exception")
            self._record({"clip_id": lost["clip_id"], "ok": False, "error": err})
            print(f"[{self.done}/{self.submitted}] FAIL (worker died) {lost['clip_id']}: {err}",
                  file=sys.stderr, flush=True)
            # so a future run's clips_to_process retries this clip instead of treating the crash's
            # partial output as a completed run
            out_dir = Path(lost["out_dir"])
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)

    def _drain_results(self, block=False):
        while True:
            try:
                res = self.result_queue.get(block=block, timeout=1 if block else None)
            except Exception:
                break
            if res.get("status") == "started":
                self.in_progress[res["pid"]] = {"clip_id": res["clip_id"], "out_dir": res["out_dir"]}
                continue
            self.in_progress.pop(res["pid"], None)
            self.done += 1
            self.extract_slot.release()
            # gpu / pid only identify the worker process (progress line, crash bookkeeping); not part of the row
            self._record({k: v for k, v in res.items() if k not in ("gpu", "pid")})
            status = "OK" if res["ok"] else "FAIL"
            print(f"[{self.done}/{self.submitted}] {status} gpu{res['gpu']} {res['seconds']:.1f}s {res['clip_id']}",
                  flush=True)
            if not res["ok"]:
                print(res["error"], file=sys.stderr, flush=True)
        self._reap_dead_workers()

    # -- extraction ----------------------------------------------------------------------------

    def _tap(self, clip):
        if clip["camera_meta"]:
            self.fixed_count += 1
        else:
            self.shared_count += 1
        return clip

    def _feed(self, pool):
        """Thread: submit each selected clip for extraction, at most `extract_lookahead` ahead of the GPUs.
        Must run concurrently with the main thread draining results: slots are only released by
        _drain_results, so submitting everything upfront would deadlock. Always ends with the None
        sentinel, also when selecting clips fails (the error is re-raised by run())."""
        try:
            for c in self.clips:
                self.extract_slot.acquire()
                self.submitted += 1
                pool.submit(extract_one, self._tap(c), self.args.fps, self.args.ffmpeg_threads) \
                    .add_done_callback(lambda fut: self.extract_done_q.put(fut.result()))
        except BaseException:
            self.feeder_error = traceback.format_exc()
        finally:
            self.extract_done_q.put(None)  # sentinel: selection + submission is fully done

    def _collect_extractions(self):
        """Main thread: hand each extracted clip to the GPU workers until the clip list is exhausted and every
        submitted clip has come back."""
        selection_done = False
        extract_completed = 0
        while not (selection_done and extract_completed >= self.submitted):
            # The timeout matters: a plain get() deadlocks once all lookahead slots are in flight and their GPU
            # results finish before the next extraction (which cannot start, the feeder waits for a slot) --
            # _drain_results, the only thing that frees a slot, would never be called. Periodic draining
            # keeps the feeder unblocked even without an extraction event to react to.
            try:
                item = self.extract_done_q.get(timeout=1.0)
            except queue.Empty:
                self._drain_results(block=False)
                continue
            if item is None:
                selection_done = True
                print(f"{self.submitted} clips selected from the manifest ({self.fixed_count} --fixed_intrinsics, "
                      f"{self.shared_count} --shared_intrinsics)", flush=True)
                continue
            extract_completed += 1
            clip, err = item
            if err:
                self.extract_slot.release()
                self.done += 1
                self._record({"clip_id": str(clip["video_path"]), "ok": False, "stage": "extract", "error": err})
                print(f"[{self.done}/{self.submitted}] FAIL extract {clip['video_path']}: {err}",
                      file=sys.stderr, flush=True)
                continue
            self.job_queue.put(clip)
            self._drain_results(block=False)

    # -- run -----------------------------------------------------------------------------------

    def run(self):
        args = self.args
        self._start_workers()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_f = open(self.manifest_path, "a")

        lookahead = args.extract_lookahead or max(4, 2 * len(self.workers))
        # Bounds how many clips are extracted-but-not-yet-GPU-consumed: acquired before a clip is submitted for
        # extraction, released when its GPU worker finishes (frames deleted) or its extraction fails. Without it
        # extraction races far ahead of the slower GPU workers and piles up on scratch_dir.
        self.extract_slot = threading.Semaphore(lookahead)

        print(f"Discovering clips and extracting frames with {args.num_extract_workers} workers "
              f"(ffmpeg_threads={args.ffmpeg_threads} each), "
              f"feeding {len(self.workers)} GPU workers ({args.num_gpus} GPUs x {args.workers_per_gpu}, "
              f"extract_lookahead={lookahead})...", flush=True)

        with ThreadPoolExecutor(max_workers=args.num_extract_workers) as pool:
            feeder_thread = threading.Thread(target=self._feed, args=(pool,), daemon=True)
            feeder_thread.start()
            self._collect_extractions()
            feeder_thread.join()

        for _ in self.workers:
            self.job_queue.put(None)
        while self.done < self.submitted and any(w.is_alive() for w in self.workers):
            self._drain_results(block=True)
        self._drain_results(block=False)
        for w in self.workers:
            w.join()
        self._reap_dead_workers()  # final catch-all in case the very last worker died right at this boundary

        self.manifest_f.close()
        print(f"Done. {self.done}/{self.submitted} accounted for. See {self.manifest_path}")
        if self.feeder_error:
            # clips selected before the failure were processed; a non-zero exit also stops the self-resubmitting job
            raise RuntimeError(f"clip selection failed:\n{self.feeder_error}")


def _single_clip(args):
    """Clip list and manifest dir for single-clip mode (--video), or None if it is already processed."""
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    if not args.force and (out_dir / "result.npz").exists():
        print(f"{out_dir / 'result.npz'} already exists (use --force to reprocess)")
        return None
    clips = [{
        "video_path": video_path,
        "out_dir": out_dir,
        "frames_dir": Path(args.scratch_dir) / out_dir.name / "frames",
        "camera_meta": resolve_camera_meta(video_path),
        "osm_region_pbf_path": Path(args.osm_region_pbf_path) if args.osm_region_pbf_path else None,
        "osm_fallback_region_pbf_paths": (
            [Path(args.osm_fallback_region_pbf_path)] if args.osm_fallback_region_pbf_path else []),
    }]
    return clips, out_dir


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.video and not args.out_dir:
        parser.error("--video requires --out_dir")

    if args.video:
        single = _single_clip(args)
        if single is None:
            return
        clips, manifest_dir = single
    else:
        # batch mode over --sample_manifest; a generator, so extraction can start on the first clip immediately
        osm_extracts_dir = args.osm_extracts_dir or str(Path(args.out_root) / "_osm_extracts")
        clips = clips_to_process(args.dataset_root, args.out_root, args.scratch_dir, args.countries,
                               args.limit, args.force, osm_extracts_dir, args.sample_manifest)
        manifest_dir = Path(args.out_root)

    run_cfg = dict(
        params=SlamParams(
            model_backend=args.model_backend, da3_checkpoint=args.da3_checkpoint, submap_size=args.submap_size,
            max_loops=args.max_loops, min_disparity=args.min_disparity, conf_threshold=args.conf_threshold,
            lc_thres=args.lc_thres),
        extraction_fps=args.fps, gps_max_dt=args.gps_max_dt, dense_log=args.dense_log,
    )
    BatchRun(args, clips, run_cfg, manifest_dir).run()


if __name__ == "__main__":
    main()
