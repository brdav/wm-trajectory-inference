#!/usr/bin/env python3
"""Re-runs GPS fusion ONLY (not SLAM) for every already-processed clip under --out_root, using
each clip's existing (untouched) result.npz slam_* fields. For every clip that has a GPS CSV
sibling:

  1. Deletes the clip's cached osm_network/ (forces build_network to re-crop from scratch with
     the current, fixed crop-radius logic -- see vggt_slam/gps_fusion.py's own comment on why a
     stale, too-small cached network.osm would otherwise just get silently reused). It's pure
     disposable scratch (re-cropping from the region PBF's own spatial index costs
     milliseconds -- see build_network's docstring), so this never needs restoring afterward.
  2. Re-runs align_trajectory_to_roads (now with the crop-radius fix, the frame_number-based GPS
     timing fix, and the ICP-against-raw-GPS-route placement procedure -- see
     vggt_slam/gps_fusion.py and vggt_slam/gps.py) from the clip's existing result.npz slam_*
     fields. SLAM itself is never re-run. A clip whose raw GPS is too sparse/noisy to trace a
     matchable route at all now fails outright (status "no_raw_gps_route_too_sparse"/
     "no_raw_gps_route_unmatched" below) rather than silently falling back to a weaker fusion
     method.
  3. Rewrites the SAME result.npz with fresh fused_*/matched_route_wkt/snapped/gps_fusion_json
     fields (slam_*/frame_ids/extraction_fps/video carried over unchanged) -- see
     vggt_slam/slam_utils.py's save_result_npz/load_result_npz.
  4. Appends one record per clip to --out_root/gps_refusion_manifest.jsonl -- a SEPARATE file
     from batch_infer.py's own manifest.jsonl (which stays untouched: it's still the record of
     the original SLAM+fusion run), in the same clip_id/ok/gps_fusion schema so existing
     manifest-reading code works unchanged against it. Append-only, same convention as
     manifest.jsonl itself -- a clip's LATEST record in this file is the authoritative one.

CPU-only (no GPU worker, no torch) -- doesn't need vggt_slam's model backends at all, only
map_matching_2 (subprocess) + osmium (local OSM crop), so this can and does use far more
parallelism than the original 16-GPU-worker pipeline.

Usage (inside the vggtslam_env_ext container -- needs osmium + the map_matching_2 binary):
    pip install --quiet osmium   # once, before this script's own workers fork (see main())
    python3 scripts/tools/refuse_gps.py --dataset_root .../datasets/natix --out_root .../natix_traj \\
        [--num_workers 64] [--limit 20] [--countries Belgium Austria]

To pick a walltime-cut-off run back up without redoing already-recorded clips, add --resume
(pointing --manifest_path, if given explicitly, at the SAME file the interrupted run wrote to):
    python3 scripts/tools/refuse_gps.py --dataset_root .../datasets/natix --out_root .../natix_traj \\
        --resume
"""
# Cap each worker's math-library thread pool BEFORE numpy (or anything that pulls it in) is ever
# imported in this process -- multiprocessing's default "fork" start method copies whatever
# thread pool the PARENT already initialized, so this must happen here, at true module-import
# time, not inside the worker function after the pool already forked. Same rationale as
# scripts/slam/batch_infer.py's gpu_worker (see its own docstring) -- 64 workers each independently
# sizing a BLAS thread pool off the node's full core count would oversubscribe just as badly as
# the GPU-worker case did, even though nothing here touches a GPU.
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]  # scripts/tools/ -> repo root
sys.path.insert(0, str(REPO_DIR))


def discover_refusion_candidates(dataset_root, out_root, countries=None, limit=None, clip_ids=None):
    """Streams clips under out_root that already have a result.npz (SLAM done) AND a GPS CSV
    sibling under dataset_root (GPS fusion was, or should have been, attempted for them).

    clip_ids: if given (a collection of absolute video-path strings, same format as
    manifest.jsonl's own "clip_id" field), restricts to exactly those clips instead of walking
    the whole out_root tree -- for targeted re-fusion of a specific subset (e.g. only the clips a
    manifest audit flagged as affected by a fix) without touching the rest of the dataset. Faster
    too, since it looks each one up directly instead of an rglob over the whole tree.
    """
    dataset_root, out_root = Path(dataset_root), Path(out_root)
    count = 0
    if clip_ids is not None:
        for clip_id in sorted(clip_ids):
            video_path = Path(clip_id)
            rel = video_path.relative_to(dataset_root).with_suffix("")
            out_dir = out_root / rel
            gps_csv = video_path.with_suffix(".csv")
            if not (out_dir / "result.npz").exists() or not gps_csv.exists():
                continue
            yield {"out_dir": out_dir, "video_path": video_path, "gps_csv": gps_csv, "rel": rel}
            count += 1
            if limit and count >= limit:
                return
        return
    roots = [out_root / c for c in countries] if countries else [out_root]
    for root in roots:
        if not root.exists():
            continue
        for result_path in sorted(root.rglob("result.npz")):
            out_dir = result_path.parent
            rel = out_dir.relative_to(out_root)
            video_path = dataset_root / rel.with_suffix(".mp4")
            gps_csv = dataset_root / rel.with_suffix(".csv")
            if not gps_csv.exists():
                continue
            yield {"out_dir": out_dir, "video_path": video_path, "gps_csv": gps_csv, "rel": rel}
            count += 1
            if limit and count >= limit:
                return


def _clear_stale_osm_network(out_dir):
    network_dir = out_dir / "osm_network"
    if network_dir.exists():
        shutil.rmtree(network_dir)


def refuse_one(clip):
    """Runs in a worker process. Returns the manifest record dict (never raises -- any failure
    is captured into the record's own gps_fusion.status, same as batch_infer.py's per-clip
    try/except)."""
    from vggt_slam.fusion import run_gps_fusion
    from vggt_slam.map_match import trim_osm_network_cache
    from vggt_slam.regions import region_candidates
    from vggt_slam.slam_utils import save_result_npz, load_result_npz

    out_dir, video_path, gps_csv, rel = clip["out_dir"], clip["video_path"], clip["gps_csv"], clip["rel"]
    osm_extracts_dir = clip["osm_extracts_dir"]
    t0 = time.time()
    result_path = out_dir / "result.npz"
    network_dir = out_dir / "osm_network"
    gps_fusion_info = None
    try:
        _clear_stale_osm_network(out_dir)
        loaded = load_result_npz(result_path)
        region_pbf_path, fallback_paths = region_candidates(rel, video_path, osm_extracts_dir)
        gps_fusion_info, fused = run_gps_fusion(
            loaded["slam_rotations"], loaded["slam_translations"], loaded["frame_ids"], loaded["extraction_fps"],
            gps_csv, network_dir, region_pbf_path, fallback_paths, traceback_chars=2000, label=str(video_path))
        fused = fused or {}
        # keeps result.npz self-describing: its gps_fusion_json reflects this attempt on every outcome
        save_result_npz(result_path, loaded["frame_ids"], loaded["slam_rotations"], loaded["slam_translations"],
                        loaded["extraction_fps"], loaded["video"],
                        fused_rotations=fused.get("rotations"), fused_translations=fused.get("translations"),
                        matched_line=fused.get("matched_line"), snapped=fused.get("snapped"),
                        gps_fusion_info=gps_fusion_info,
                        # the depth-based metric scale (scripts/metric_scale/integrate_metric_scale.py) is independent of
                        # GPS fusion: carry it over, don't drop it on rewrite
                        metric_scale=loaded.get("metric_scale"), metric_poses=loaded.get("metric_poses"),
                        metric_scale_info=loaded.get("metric_scale_info"))
    except Exception:
        # result.npz itself unreadable: nothing to rewrite, just record the outcome
        gps_fusion_info = {"status": "failed", "error": traceback.format_exc()[-2000:]}
        trim_osm_network_cache(network_dir)
    return {"clip_id": str(video_path), "ok": True, "seconds": time.time() - t0, "gps_fusion": gps_fusion_info}


def _init_worker(osm_extracts_dir):
    # Passed via a module-global rather than baked into each clip dict, purely to keep the
    # per-task payload small across the multiprocessing pickling boundary.
    global _OSM_EXTRACTS_DIR
    _OSM_EXTRACTS_DIR = osm_extracts_dir


def _refuse_one_task(clip):
    clip["osm_extracts_dir"] = _OSM_EXTRACTS_DIR
    return refuse_one(clip)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--osm_extracts_dir", default=None, help="Default: {out_root}/_osm_extracts")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--clip_list", default=None,
                     help="Path to a text file of clip_ids (one absolute video path per line, "
                          "same format as manifest.jsonl's own clip_id field) -- restricts "
                          "re-fusion to exactly this set instead of every clip under --out_root. "
                          "Mutually exclusive with --countries.")
    ap.add_argument("--num_workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="For testing: only process the first N candidates")
    ap.add_argument("--manifest_path", default=None,
                     help="Default: {out_root}/gps_refusion_manifest.jsonl -- deliberately NOT "
                          "batch_infer.py's own manifest.jsonl, which stays untouched as the "
                          "record of the original SLAM+fusion run")
    ap.add_argument("--resume", action="store_true",
                     help="Skip any clip that already has a record in --manifest_path (from an "
                          "earlier, e.g. walltime-cut-off, run of this same script against the "
                          "same manifest file) instead of re-fusing it again. Off by default: a "
                          "plain re-run always re-fuses everything, since that's the usual "
                          "reason to run this script at all (e.g. testing a gps_fusion.py "
                          "change) -- --resume is only for picking a genuinely interrupted run "
                          "back up where it left off.")
    args = ap.parse_args()

    osm_extracts_dir = args.osm_extracts_dir or str(Path(args.out_root) / "_osm_extracts")
    manifest_path = Path(args.manifest_path or Path(args.out_root) / "gps_refusion_manifest.jsonl")

    clip_ids = None
    if args.clip_list:
        if args.countries:
            ap.error("--clip_list and --countries are mutually exclusive")
        with open(args.clip_list) as f:
            clip_ids = {line.strip() for line in f if line.strip()}
        print(f"--clip_list: restricting to {len(clip_ids)} clip_ids from {args.clip_list}", flush=True)

    clips = list(discover_refusion_candidates(args.dataset_root, args.out_root, args.countries,
                                               args.limit, clip_ids=clip_ids))
    print(f"{len(clips)} clips to re-fuse", flush=True)

    if args.resume and manifest_path.exists():
        # Only skip clips whose LATEST recorded outcome is deterministic -- "ok",
        # "no_roads_in_any_region", "no_raw_gps_route_too_sparse", "no_raw_gps_route_unmatched",
        # "implausible_gps_scale", "no_real_gps_fixes", and "insufficient_gps_matches" all
        # re-derive identically from the same (untouched) result.npz slam_* fields + GPS csv + OSM
        # region, so re-running them is pure waste.
        # "failed" (the generic catch-all bucket) is deliberately NOT skipped: confirmed directly
        # on this exact resume -- a burst of consecutive "failed" records right before a crash
        # were ALL the identical BrokenPipeError/"transport endpoint shutdown" from a transient
        # filesystem/network blip, not real per-clip problems, so treating "failed" as
        # permanently done would have silently stranded them forever. Retrying a genuinely-broken
        # clip is harmless (it just fails the same way again); skipping a transiently-broken one
        # forever is not.
        DETERMINISTIC_STATUSES = {"ok", "no_roads_in_any_region", "no_raw_gps_route_too_sparse",
                                   "no_raw_gps_route_unmatched", "implausible_gps_scale",
                                   "no_real_gps_fixes", "insufficient_gps_matches",
                                   "gps_log_too_short"}
        latest_status = {}
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                latest_status[d["clip_id"]] = d.get("gps_fusion", {}).get("status")
        already_done = {cid for cid, status in latest_status.items() if status in DETERMINISTIC_STATUSES}
        n_will_retry_failed = sum(1 for s in latest_status.values() if s not in DETERMINISTIC_STATUSES)
        before = len(clips)
        clips = [c for c in clips if str(c["video_path"]) not in already_done]
        print(f"--resume: skipping {before - len(clips)} clips with a deterministic outcome "
              f"already recorded in {manifest_path}; retrying {n_will_retry_failed} previously-"
              f"'failed' clips too (status=failed can be a transient infra hiccup, not "
              f"necessarily a real per-clip problem) -- {len(clips)} left to do", flush=True)

    if not clips:
        return

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    n_ok = n_failed = n_manual_drivable = 0
    with ctx.Pool(args.num_workers, initializer=_init_worker, initargs=(osm_extracts_dir,)) as pool, \
         open(manifest_path, "a") as manifest_f:
        for i, record in enumerate(pool.imap_unordered(_refuse_one_task, clips), 1):
            manifest_f.write(json.dumps(record) + "\n")
            manifest_f.flush()
            status = record["gps_fusion"]["status"]
            n_ok += status == "ok"
            n_failed += status != "ok"
            profile_used = record["gps_fusion"].get("network_profile_used")
            n_manual_drivable += profile_used == "manual_drivable"
            err_m = record["gps_fusion"].get("map_match_error_mean_m")
            err_str = f" map_match_error={err_m:.2f}m" if err_m is not None else f" status={status}"
            profile_str = " [manual_drivable fallback]" if profile_used == "manual_drivable" else ""
            print(f"[{i}/{len(clips)}] {record['clip_id']}{err_str}{profile_str}", flush=True)

    print(f"Done. {n_ok} ok, {n_failed} not-ok, {n_manual_drivable} used the manual_drivable "
          f"network-profile fallback, out of {len(clips)}. See {manifest_path}")


if __name__ == "__main__":
    main()
