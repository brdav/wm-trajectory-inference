#!/usr/bin/env python3
"""Estimate a metric scale for clips whose GPS fusion did not succeed, using DA3METRIC-LARGE.

Clips whose gps_fusion status in the trajectory manifest is not "ok" have no `fused_poses`, only
`slam_poses`, which are in an arbitrary visual-gauge scale. For each such clip this script:

  1. extracts frames and (like batch_infer.py) undistorts them with the trip's calibration,
  2. re-runs VGGT-SLAM with dense point-cloud logging (no GPS fusion) -- the point cloud is
     needed to get SLAM-gauge depth, and batch runs don't store it,
  3. runs DA3METRIC-LARGE on --n_frames evenly spaced keyframes and converts its output to
     meters with the *calibrated* rectified focal (metric = net_output * focal_px / 300, focal
     rescaled to the processed width),
  4. compares that metric depth with the SLAM point cloud projected into the same frame (z-depth,
     nearest point per pixel, sky/hood/far pixels masked): scale = median(metric / slam).

Output per clip: <out_root>/<same relative path as the video, no .mp4>/metric_scale.npz with
frame_ids, slam_poses and metric_poses (= slam_poses with the translation multiplied by scale; the
rerun's own poses, which are the ones the scale belongs to -- they can differ slightly from the
original result.npz), scale, per-frame scales and quality stats. Already-done clips are skipped,
so an interrupted run can simply be rerun. One line per clip goes to <out_root>/manifest_metric_scale.jsonl.

Validated on 5 clips with GPS: metric-depth scale within about +-8% of the GPS-derived scale
(no systematic bias). Treat scale as ~+-10% accurate.

Run inside the container, e.g.:
    srun --environment=vggtslam_env python3 scripts/estimate_metric_scale.py \\
        --out_root /path/to/metric_scale_out --num_gpus 4 --workers_per_gpu 2
Needs dino_salad.ckpt + the DINOv2 hub repo in torch.hub's cache (same as batch_infer.py) and the
DA3METRIC-LARGE weights (HF_HOME, or network access to download them).
"""
import argparse
import json
import os
import queue as queue_mod
import random
import shutil
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent

# open3d fails to load in this container ("cannot allocate memory in static TLS block") unless
# this is set before the worker processes are spawned -- same fix and reason as batch_infer.py.
os.environ.setdefault("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=4000000")

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "scripts"))

DEFAULT_TRAJ_ROOT = "/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories"
DEFAULT_DATASET_ROOT = "/capstor/store/cscs/swissai/ab036/datasets/natix"
MANIFEST_NAME = "manifest_metric_scale.jsonl"
MIN_FRAMES_USED = 1
MAX_METRIC_DEPTH_M = 80.0
BOTTOM_CROP_FRAC = 0.85  # ignore the bottom 15% of the image (hood)


def candidate_clips(args):
    """(clip_id, label) for every clip that should get a depth-based scale.

    With --qc_csv (the normal pipeline, see run_metric_scale.sbatch): every clip scripts/clip_qc.py
    routed to needs_metric_scale (SLAM is fine but there is no usable GPS scale: fusion
    failed/missing, or fused_poses look implausible) or needs_scale_check (fused_poses pass but
    are near a limit or unverifiable: the depth scale cross-checks them). Clips with a diverged SLAM trajectory
    (route ego_failed) are never scaled. Without it: every clip whose gps_fusion status in the
    batch manifest is not "ok" (older behaviour, ignores the quality checks).
    """
    if args.clip_list:
        # explicit list (one dataset-relative .mp4 path per line, optional tab + label), e.g. to
        # spot-check clips the QC passed; ignores QC routes and any existing result
        with open(args.clip_list) as f:
            for line in f:
                if line.strip():
                    rel, _, label = line.rstrip("\n").partition("\t")
                    yield f"{args.dataset_root.rstrip('/')}/{rel}", label or "clip_list"
        return
    if args.qc_csv:
        import csv
        with open(args.qc_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["route"] in ("needs_metric_scale", "needs_scale_check") and (args.force or row.get("has_metric_scale") != "True"):
                    yield f"{args.dataset_root.rstrip('/')}/{row['clip']}", row.get("scale_reason") or "needs_metric_scale"
        return
    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") is not True:
                continue
            status = (r.get("gps_fusion") or {}).get("status", "no_gps_fusion_record")
            if status != "ok":
                yield r["clip_id"], status


def integration_state(out_root, dataset_root):
    """(integrated, refused) clip_id sets from integrate_metric_scale.py's append-only log, latest record per clip
    winning. integrated = merged into the clip's result.npz; refused = the SLAM rerun did not reproduce the stored
    poses (re-estimating would be refused again). Lets this script skip finished clips without the per-clip
    metric_scale.npz files, which can be archived once integrated."""
    integrated, refused = set(), set()
    log = Path(out_root) / "integration_log.jsonl"
    if log.exists():
        with open(log) as f:
            for line in f:
                r = json.loads(line)
                cid = lambda inner: f"{str(dataset_root).rstrip('/')}/{r['session']}/{inner[:-len('/result.npz')]}.mp4"
                for inner, why in r.get("skipped", {}).items():
                    if why == "mismatch":
                        refused.add(cid(inner)); integrated.discard(cid(inner))
                for inner in r.get("updated_inner", []):
                    integrated.add(cid(inner)); refused.discard(cid(inner))
    return integrated, refused


def select_clips(args):
    dataset_root = Path(args.dataset_root)
    statuses = set(args.statuses) if args.statuses else None
    countries = set(args.countries) if args.countries else None
    out_root = Path(args.out_root)
    # Clips that already failed once (e.g. too few usable frames) are not retried automatically:
    # they never get a metric_scale.npz, so without this every rerun / sbatch resubmission would
    # redo them forever. --retry_failed overrides.
    failed = set()
    prev_manifest = out_root / MANIFEST_NAME
    if prev_manifest.exists() and not args.retry_failed:
        with open(prev_manifest) as f:
            failed = {json.loads(l)["clip_id"] for l in f if l.strip() and not json.loads(l).get("ok")}
    integrated, refused = integration_state(out_root, dataset_root)
    if not args.retry_failed:
        failed |= refused
    clips = []
    for clip_id, status in candidate_clips(args):
        if clip_id in failed or (clip_id in integrated and not args.force):
            continue
        video = Path(clip_id)
        rel = video.relative_to(dataset_root)
        if countries is not None and rel.parts[0] not in countries:
            continue
        if args.num_shards > 1 and zlib.crc32(str(rel).encode()) % args.num_shards != args.shard:
            continue
        if not args.force and (out_root / rel.with_suffix("") / "metric_scale.npz").exists():
            continue  # (also skipped via the QC csv's has_metric_scale and the integration log, see candidate_clips)
        if statuses is not None and status not in statuses:
            continue
        clips.append({"video_path": str(video), "rel": str(rel.with_suffix("")), "status": status})
    if args.shuffle:
        random.Random(0).shuffle(clips)
    return clips[:args.limit] if args.limit else clips


def read_pcd_xyz(path, stride):
    """x,y,z of a binary PCD (fields x y z rgb, float32 each), every `stride`-th point."""
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline()
            if line.startswith(b"POINTS"):
                n = int(line.split()[1])
            if line.startswith(b"DATA"):
                assert b"binary" in line, "expected a binary PCD"
                break
        data = np.fromfile(f, dtype=np.float32, count=n * 4).reshape(n, 4)
    return np.ascontiguousarray(data[::stride, :3])


def estimate_scale(slam_poses, frame_ids, points_xyz, frames_dir, K, undistort, metric_model, n_frames):
    """Median metric/SLAM depth ratio over n_frames evenly spaced poses. K = rectified
    (fx, fy, cx, cy, width, height) -- the geometry SLAM actually ran in."""
    import cv2
    import torch

    fx0, fy0, cx0, cy0, w0, _ = K
    map1, map2, roi, undistort_frame = undistort
    P = torch.from_numpy(points_xyz).cuda()

    _, first = np.unique(frame_ids, return_index=True)
    first = np.sort(first)  # drop duplicate frame_ids (submap boundaries)
    sel = first[np.linspace(0, len(first) - 1, min(n_frames, len(first))).round().astype(int)]

    per_frame = []
    for k in sel:
        img = cv2.imread(str(Path(frames_dir) / f"frame_{int(frame_ids[k]):05d}.jpg"))
        if img is None:
            continue
        img = cv2.cvtColor(undistort_frame(img, map1, map2, roi), cv2.COLOR_BGR2RGB)
        pred = metric_model.inference([img], process_res=504)
        H, W = pred.depth.shape[1:]
        sc = W / w0
        fx, fy, cx, cy = fx0 * sc, fy0 * sc, cx0 * sc, cy0 * sc
        metric = pred.depth[0] * ((fx + fy) / 2) / 300.0
        sky = pred.sky[0] > 0.3 if pred.sky is not None else np.zeros((H, W), bool)

        w2c = torch.from_numpy(np.linalg.inv(slam_poses[k]).astype(np.float32)).cuda()
        pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        ok = z > 0.05
        zs = torch.where(ok, z, torch.ones_like(z))
        u = torch.round(fx * pc[:, 0] / zs + cx).long()
        v = torch.round(fy * pc[:, 1] / zs + cy).long()
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        D = torch.full((H * W,), float("inf"), device="cuda")
        D.scatter_reduce_(0, (v * W + u)[ok], z[ok], reduce="amin")  # nearest point per pixel
        D = D.view(H, W).cpu().numpy()

        rows = np.arange(H)[:, None] < BOTTOM_CROP_FRAC * H
        m = np.isfinite(D) & ~sky & (metric < MAX_METRIC_DEPTH_M) & rows
        if m.sum() < 200:
            continue
        per_frame.append((int(frame_ids[k]), float(np.median(metric[m] / D[m])), int(m.sum())))
    if len(per_frame) < MIN_FRAMES_USED:
        raise RuntimeError(f"only {len(per_frame)} usable frames (< {MIN_FRAMES_USED})")
    s = np.array([p[1] for p in per_frame])
    stats = {"scale": float(np.median(s)), "scale_p25": float(np.percentile(s, 25)),
             "scale_p75": float(np.percentile(s, 75)), "n_frames_used": len(s)}
    stats["scale_rel_iqr"] = (stats["scale_p75"] - stats["scale_p25"]) / stats["scale"]
    return stats, np.array(per_frame)


def worker(gpu_id, cfg, job_q, res_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(cfg["cpu_threads"])
    import argparse as _ap
    import torch
    import main as vggt_main  # after torch, same order as batch_infer.gpu_worker
    from depth_anything_3.api import DepthAnything3
    import batch_infer
    from vggt_slam.undistort import load_undistort_maps, undistort_frame
    torch.set_num_threads(cfg["cpu_threads"])

    # SLAM settings = batch_infer.py's own defaults, minus GPS fusion, plus dense logging
    bi = batch_infer.build_arg_parser().parse_args(["--video", "x", "--out_dir", "y"])
    slam_args = dict(
        model_backend=bi.model_backend, da3_checkpoint=bi.da3_checkpoint,
        use_ray_pose=True, ref_view_strategy="middle",
        vis_map=False, vis_imgs=False, vis_voxel_size=None, run_os=False, vis_flow=False,
        headless=True, log_results=True, skip_dense_log=False, log_path="result.npz",
        submap_size=bi.submap_size, overlapping_window_size=2, max_loops=bi.max_loops,
        min_disparity=bi.min_disparity, conf_threshold=bi.conf_threshold, lc_thres=bi.lc_thres,
        gps_csv=None, extraction_fps=cfg["fps"], gps_max_dt=bi.gps_max_dt,
    )
    backend, clip_model, clip_preprocess, clip_tokenizer, processor = vggt_main.build_backend(
        _ap.Namespace(**slam_args), "cuda")
    metric_model = DepthAnything3.from_pretrained(cfg["metric_checkpoint"]).to("cuda")

    while True:
        job = job_q.get()
        if job is None:
            break
        t0 = time.time()
        work_dir = Path(cfg["scratch_dir"]) / "work" / job["rel"]
        res = {"clip_id": job["video_path"], "gpu": gpu_id, "prev_status": job["status"]}
        try:
            video = Path(job["video_path"])
            clip = {
                "video_path": video, "out_dir": work_dir, "frames_dir": work_dir / "frames",
                "camera_meta": batch_infer._resolve_camera_meta(video),
            }
            if clip["camera_meta"] is None:
                raise RuntimeError("no_calibration: no usable FIXED_METADATA.json for this clip")
            clip, err = batch_infer.extract_one(clip, cfg["fps"], cfg["ffmpeg_threads"])
            if err:
                raise RuntimeError(f"frame extraction failed: {err}")
            if clip.get("fixed_intrinsics_str") is None:
                raise RuntimeError(f"undistortion failed: {clip.get('undistort_warning')}")

            a = _ap.Namespace(**slam_args)
            a.image_folder = clip["image_folder"]
            a.fixed_intrinsics = clip["fixed_intrinsics_str"]
            a.shared_intrinsics = False
            a.undistort_maps_path = clip["undistort_maps_path"]
            a.log_path = str(work_dir / "result.npz")
            a.video = str(video)
            a.osm_region_pbf_path = None
            a.osm_fallback_region_pbf_path = None
            vggt_main.run_video(a, backend, "cuda", clip_model, clip_preprocess, clip_tokenizer, processor)

            z = np.load(work_dir / "result.npz")
            frame_ids, slam_poses = z["frame_ids"], z["slam_poses"]
            fx, fy, cx, cy, w, h = (float(x) for x in clip["fixed_intrinsics_str"].split(","))
            map1, map2, roi = load_undistort_maps(clip["undistort_maps_path"])
            points = read_pcd_xyz(work_dir / "result_points.pcd", cfg["pcd_stride"])
            stats, per_frame = estimate_scale(
                slam_poses, frame_ids, points, clip["frames_dir"], (fx, fy, cx, cy, w, h),
                (map1, map2, roi, undistort_frame), metric_model, cfg["n_frames"])

            # recorded so integrate_metric_scale.py can store the full provenance with the scale
            stats.update(model=cfg["metric_checkpoint"], n_frames_requested=cfg["n_frames"],
                         pcd_stride=cfg["pcd_stride"], process_res=504)
            metric_poses = slam_poses.copy()
            metric_poses[:, :3, 3] *= stats["scale"]
            final_dir = Path(cfg["out_root"]) / job["rel"]
            final_dir.mkdir(parents=True, exist_ok=True)
            tmp = final_dir / f".metric_scale.{os.getpid()}.npz"
            np.savez(tmp, frame_ids=frame_ids, slam_poses=slam_poses, metric_poses=metric_poses,
                     extraction_fps=z["extraction_fps"], video=np.asarray(str(video)),
                     scale=np.asarray(stats["scale"]), per_frame_scale=per_frame,
                     stats_json=np.asarray(json.dumps(stats)), prev_gps_status=np.asarray(job["status"]))
            os.replace(tmp, final_dir / "metric_scale.npz")
            res.update(ok=True, **stats)
        except Exception:
            res.update(ok=False, error=traceback.format_exc()[-2000:])
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)  # frames + temp result/pcd
        res["seconds"] = time.time() - t0
        res_q.put(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, help="Where per-clip metric_scale.npz files go")
    ap.add_argument("--clip_list", default=None,
                    help="Process exactly these clips (dataset-relative .mp4 paths, optional TAB label per "
                         "line) instead of the QC/manifest selection. Use a separate --out_root for such "
                         "spot checks so they are not mixed into the production outputs.")
    ap.add_argument("--qc_csv", default=None,
                    help="scripts/clip_qc.py output: process the clips it routed to needs_metric_scale "
                         "(recommended). Default: every clip whose gps_fusion status in --manifest is not 'ok'")
    ap.add_argument("--manifest", default=f"{DEFAULT_TRAJ_ROOT}/manifest.jsonl")
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--statuses", nargs="*", default=None,
                    help="Only clips with these labels: gps_fusion statuses without --qc_csv, "
                         "clip_qc scale_reason values (e.g. no_fused_poses) with it")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N selected clips (testing)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle the selection (with --limit: a random sample)")
    ap.add_argument("--shard", type=int, default=0, help="This job's shard index (with --num_shards)")
    ap.add_argument("--num_shards", type=int, default=1, help="Split clips across this many independent jobs")
    ap.add_argument("--force", action="store_true", help="Redo clips that already have a metric_scale.npz")
    ap.add_argument("--retry_failed", action="store_true",
                    help="Also retry clips that previously failed (recorded ok=false in the output manifest)")
    ap.add_argument("--num_gpus", type=int, default=4)
    ap.add_argument("--workers_per_gpu", type=int, default=1)
    ap.add_argument("--cpu_threads_per_worker", type=int, default=None)
    ap.add_argument("--ffmpeg_threads", type=int, default=0)
    ap.add_argument("--fps", type=float, default=10.0, help="Must match the fps the trajectories were made at")
    ap.add_argument("--n_frames", type=int, default=50, help="Keyframes per clip to compare depth on")
    ap.add_argument("--pcd_stride", type=int, default=3, help="Use every Nth point of the dense cloud")
    ap.add_argument("--metric_checkpoint", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--scratch_dir", default="/dev/shm/vggt_slam_metric_scale",
                    help="Node-local scratch for frames and the temporary dense point cloud (~0.5GB/clip)")
    args = ap.parse_args()

    clips = select_clips(args)
    print(f"{len(clips)} clips to process", flush=True)
    if not clips:
        return
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    job_q, res_q = ctx.Queue(), ctx.Queue()
    n_workers = args.num_gpus * args.workers_per_gpu
    cfg = dict(vars(args))
    cfg["cpu_threads"] = args.cpu_threads_per_worker or max(1, (os.cpu_count() or 1) // (n_workers + 1))
    for c in clips:
        job_q.put(c)
    for _ in range(n_workers):
        job_q.put(None)
    workers = [ctx.Process(target=worker, args=(g, cfg, job_q, res_q))
               for g in range(args.num_gpus) for _ in range(args.workers_per_gpu)]
    for w in workers:
        w.start()

    done = n_ok = 0
    crashed = False
    with open(Path(args.out_root) / MANIFEST_NAME, "a") as mf:
        while done < len(clips):
            try:
                res = res_q.get(timeout=10)
            except queue_mod.Empty:
                if not any(w.is_alive() for w in workers):
                    print("[WARN] all workers exited; unfinished clips will be retried on rerun",
                          file=sys.stderr, flush=True)
                    crashed = True
                    break
                continue
            done += 1
            n_ok += bool(res["ok"])
            mf.write(json.dumps(res) + "\n")
            mf.flush()
            tag = f"scale={res['scale']:.2f}" if res["ok"] else "FAIL"
            print(f"[{done}/{len(clips)}] {tag} gpu{res['gpu']} {res['seconds']:.0f}s {res['clip_id']}", flush=True)
            if not res["ok"]:
                print(res["error"], file=sys.stderr, flush=True)
    for w in workers:
        w.join(timeout=30)
    print(f"Done. {n_ok}/{done} succeeded. Manifest: {Path(args.out_root) / MANIFEST_NAME}")
    if crashed:
        sys.exit(1)  # non-zero so run_metric_scale.sbatch resubmits instead of treating this as finished


if __name__ == "__main__":
    main()
