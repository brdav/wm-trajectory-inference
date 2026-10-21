#!/usr/bin/env python3
"""One-off repair of trajectories/manifest.jsonl rows written by older versions of the pipeline.

A fresh run of the current pipeline writes rows with `fixed_intrinsics` for every clip and has a row for every
video in the sample manifest; the existing manifest predates that. This script
  1. drops the worker bookkeeping fields `gpu` and `pid` (no longer part of a row),
  2. fills `fixed_intrinsics` where it is missing: True iff batch_infer.py would find usable camera calibration for
     the clip (a FIXED_METADATA.json above the video whose camera_front principal point is plausible for the
     video's resolution, batch_infer._resolve_camera_meta) AND the undistortion maps can be computed from it
     (the two conditions of the original run). Before
     changing anything it re-computes the value for a sample of rows that already have it and requires the two to
     agree (>= 99%); otherwise it stops. The recorded value was False for ~0.6% of clips that do have usable
     calibration (transient fallbacks to self-calibration), so an inferred True can be wrong for a few hundred
     rows; every filled row is marked fixed_intrinsics_inferred = true,
  3. adds a row for every sample-manifest video that has none (ok=false, stage="backfilled": the clip has a result
     in its session zip but the original run left no record, most likely a worker that died after saving the SLAM
     result),
and writes the manifest in place after saving a timestamped backup next to it. Run
scripts/update_manifest_metric_scale.py afterwards (it normalizes every row to the fixed schema and fills qc /
metric_scale / pose_source for the new rows). Idempotent.

Usage:
    python3 scripts/backfill_manifest.py --dry_run
    python3 scripts/backfill_manifest.py [--out /tmp/manifest_backfilled.jsonl]
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import batch_infer  # noqa: E402  (_resolve_camera_meta)
from vggt_slam.sample_manifest import (DEFAULT_DATASET_ROOT, DEFAULT_SAMPLE_MANIFEST,  # noqa: E402
                                       DEFAULT_TRAJECTORY_ROOT, source_videos)
from vggt_slam.undistort import compute_undistort_maps  # noqa: E402


_MAPS_OK = {}


def has_calibration(clip_id):
    """What the original run recorded as fixed_intrinsics: usable calibration was found AND the undistortion
    maps could be computed from it (batch_infer.extract_one falls back to self-calibration otherwise)."""
    meta = batch_infer._resolve_camera_meta(Path(clip_id))
    if meta is None:
        return False
    dims = batch_infer._read_mp4_dims(Path(clip_id))
    if not dims:
        return False
    key = (tuple(sorted(meta.items())), dims)
    if key not in _MAPS_OK:
        try:
            compute_undistort_maps(dims[0], dims[1], fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"],
                                   k1=meta["k1"], k2=meta["k2"], k3=meta["k3"], p1=meta["p1"], p2=meta["p2"])
            _MAPS_OK[key] = True
        except Exception:
            _MAPS_OK[key] = False
    return _MAPS_OK[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=f"{DEFAULT_TRAJECTORY_ROOT}/manifest.jsonl")
    ap.add_argument("--sample_manifest", default=DEFAULT_SAMPLE_MANIFEST)
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=None, help="Write here instead of rewriting --manifest in place")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--check_n", type=int, default=3000, help="Rows with a recorded fixed_intrinsics used to validate the method")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    print(f"{len(rows)} rows", flush=True)
    n_dropped = sum(1 for r in rows if "gpu" in r or "pid" in r)
    rows = [{k: v for k, v in r.items() if k not in ("gpu", "pid")} for r in rows]

    missing = [r for r in rows if r.get("fixed_intrinsics") is None]
    have = [r for r in rows if isinstance(r.get("fixed_intrinsics"), bool)]
    random.seed(0)
    check = random.sample(have, min(args.check_n, len(have)))
    have_ids = {r["clip_id"] for r in rows}
    dataset_root = args.dataset_root.rstrip("/")
    new_videos = [v for v in source_videos(args.sample_manifest) if f"{dataset_root}/{v}" not in have_ids]
    todo = [r["clip_id"] for r in missing] + [r["clip_id"] for r in check] + [f"{dataset_root}/{v}" for v in new_videos]
    print(f"gpu/pid on {n_dropped} rows; fixed_intrinsics missing on {len(missing)}; validating the method on "
          f"{len(check)} rows that have it; {len(new_videos)} sample-manifest videos have no row", flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        result = dict(zip(todo, pool.imap(has_calibration, todo, chunksize=64)))
    print(f"calibration lookups took {time.time() - t0:.0f}s", flush=True)

    agree = sum(result[r["clip_id"]] == r["fixed_intrinsics"] for r in check)
    rate = agree / max(len(check), 1)
    print(f"method check: agrees with the recorded fixed_intrinsics on {agree}/{len(check)} rows ({100 * rate:.2f}%)", flush=True)
    if rate < 0.99:
        sys.exit("agreement below 99% -- not backfilling; investigate _resolve_camera_meta vs the original runs")

    for r in missing:
        r["fixed_intrinsics"] = result[r["clip_id"]]
        r["fixed_intrinsics_inferred"] = True  # not recorded by the original run; see the method check above
    for v in new_videos:
        cid = f"{dataset_root}/{v}"
        rows.append({"clip_id": cid, "ok": False, "seconds": None, "fixed_intrinsics": result[cid],
                     "fixed_intrinsics_inferred": True, "stage": "backfilled",
                     "error": "no record in the original run's manifest (the clip has a result in its session zip; the "
                              "worker most likely died after saving the SLAM result); row added by scripts/backfill_manifest.py"})
    print(f"filled fixed_intrinsics on {len(missing)} rows "
          f"(calibrated: {sum(bool(r['fixed_intrinsics']) for r in missing)}, not: {sum(not r['fixed_intrinsics'] for r in missing)}); "
          f"added {len(new_videos)} rows -> {len(rows)} rows", flush=True)
    if args.dry_run:
        return

    out_path = Path(args.out or args.manifest)
    if args.out is None:
        bak = Path(f"{args.manifest}.bak_{time.strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(args.manifest, bak)
        print(f"backup: {bak}")
    tmp = out_path.with_name(out_path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if args.out is None:
        shutil.copymode(args.manifest, tmp)
    os.replace(tmp, out_path)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
