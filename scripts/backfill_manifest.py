#!/usr/bin/env python3
"""One-off repair of trajectories/manifest.jsonl rows written by older versions of the pipeline.

A fresh run of the current pipeline writes rows with `fixed_intrinsics` for every clip and has a row for every
video in the sample manifest; the existing manifest predates that. This script
  1. drops the worker bookkeeping fields `gpu` and `pid` (no longer part of a row),
  2. removes rows of videos that are not in the sample manifest (a leftover of an earlier dataset listing); the removed
     rows are saved next to the manifest backups so nothing is lost,
  3. adds a row for every sample-manifest video that has none (ok=false, stage="backfilled": the clip has a result
     in its session zip but the original run left no record, most likely a worker that died after saving the SLAM
     result),
  4. sets `fixed_intrinsics` on every row that was not recorded by the original run (missing, or already marked
     fixed_intrinsics_inferred): True iff the clip's session zip contains `undistort_maps.npz`. batch_infer.py writes
     that file into the session folder when it computes the undistortion maps from the camera calibration, so a
     session without it was processed entirely with self-calibration. Rows with a recorded value keep it.
     The rule is checked against all rows that have a recorded value and the script stops below 99% agreement
     (it is 99.2%). The 576 disagreements are single clips recorded False in sessions that do have the maps; all of them
     were processed after the maps existed and they are spread evenly over the run. The cause is not known: the code
     self-calibrates a clip when its camera-metadata lookup (_resolve_camera_meta, which swallows every exception)
     returned nothing, and that lookup succeeds for them today. About 0.8% of the inferred True values (about 375 rows)
     can therefore be wrong and cannot be told apart. Every row touched by this rule is marked
     fixed_intrinsics_inferred = true.
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
import shutil
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vggt_slam.sample_manifest import (DEFAULT_DATASET_ROOT, DEFAULT_SAMPLE_MANIFEST,  # noqa: E402
                                       DEFAULT_TRAJECTORY_ROOT, session_of, source_videos)

MAPS_FILE = "undistort_maps.npz"


def session_has_maps(args):
    """(session, True/False/None): does the session's zip (or, if not packed, its loose folder) hold the
    undistortion maps? None if the session has neither."""
    session, out_root = args
    zp = Path(out_root) / (session + ".zip")
    try:
        if zp.exists():
            with zipfile.ZipFile(zp) as zf:
                return session, MAPS_FILE in set(zf.namelist())
        loose = Path(out_root) / session
        if loose.is_dir():
            return session, (loose / MAPS_FILE).exists()
    except Exception:
        pass
    return session, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=f"{DEFAULT_TRAJECTORY_ROOT}/manifest.jsonl")
    ap.add_argument("--out_root", default=DEFAULT_TRAJECTORY_ROOT, help="Trajectories root holding the session zips")
    ap.add_argument("--sample_manifest", default=DEFAULT_SAMPLE_MANIFEST)
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=None, help="Write here instead of rewriting --manifest in place")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--removed_log", default=f"{DEFAULT_TRAJECTORY_ROOT}/_pipeline/backups/manifest_rows_removed_{time.strftime('%Y%m%d')}.jsonl",
                    help="Where rows removed because their video is not in the sample manifest are saved")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    print(f"{len(rows)} rows", flush=True)
    n_dropped = sum(1 for r in rows if "gpu" in r or "pid" in r)
    rows = [{k: v for k, v in r.items() if k not in ("gpu", "pid")} for r in rows]

    dataset_root = args.dataset_root.rstrip("/")
    wanted = {f"{dataset_root}/{v}" for v in source_videos(args.sample_manifest)}
    extra = [r for r in rows if r["clip_id"] not in wanted]
    rows = [r for r in rows if r["clip_id"] in wanted]
    have_ids = {r["clip_id"] for r in rows}
    new_videos = [v for v in source_videos(args.sample_manifest) if f"{dataset_root}/{v}" not in have_ids]
    for v in new_videos:
        rows.append({"clip_id": f"{dataset_root}/{v}", "ok": False, "seconds": None, "fixed_intrinsics": None,
                     "stage": "backfilled",
                     "error": "no record in the original run's manifest (the clip has a result in its session zip; the "
                              "worker most likely died after saving the SLAM result); row added by scripts/backfill_manifest.py"})

    sess = lambda r: session_of(r["clip_id"][len(dataset_root) + 1:])
    sessions = sorted({sess(r) for r in rows})
    t0 = time.time()
    with Pool(args.workers) as pool:
        maps = dict(pool.map(session_has_maps, [(s, args.out_root) for s in sessions], chunksize=200))
    print(f"gpu/pid on {n_dropped} rows; {len(extra)} rows of videos outside the sample manifest; {len(new_videos)} videos "
          f"without a row; looked into {len(sessions)} session zips in {time.time() - t0:.0f}s "
          f"({sum(v is True for v in maps.values())} have {MAPS_FILE}, {sum(v is False for v in maps.values())} lack it, "
          f"{sum(v is None for v in maps.values())} not found)", flush=True)

    recorded = [r for r in rows if r.get("fixed_intrinsics") is not None and not r.get("fixed_intrinsics_inferred")]
    known = [r for r in recorded if maps[sess(r)] is not None]
    agree = sum(maps[sess(r)] == r["fixed_intrinsics"] for r in known)
    rate = agree / max(len(known), 1)
    print(f"rule check on all {len(known)} rows with a recorded value: agrees on {agree} ({100 * rate:.2f}%); "
          f"{len(known) - agree} disagree (recorded False although the session has maps: "
          f"{sum(1 for r in known if maps[sess(r)] and not r['fixed_intrinsics'])}; recorded True although it has none: "
          f"{sum(1 for r in known if maps[sess(r)] is False and r['fixed_intrinsics'])})", flush=True)
    if rate < 0.99:
        sys.exit("agreement below 99% -- not backfilling; investigate")

    targets = [r for r in rows if r.get("fixed_intrinsics") is None or r.get("fixed_intrinsics_inferred")]
    changed = unresolved = 0
    for r in targets:
        v = maps[sess(r)]
        if v is None:
            unresolved += 1
            continue
        changed += (r.get("fixed_intrinsics") != v)
        r["fixed_intrinsics"] = v
        r["fixed_intrinsics_inferred"] = True
    print(f"rule applied to {len(targets)} rows not recorded by the original run: value changed on {changed}, "
          f"left unresolved (no session zip/folder): {unresolved}; result: "
          f"{sum(1 for r in targets if r['fixed_intrinsics'])} True, {sum(1 for r in targets if r['fixed_intrinsics'] is False)} False; "
          f"{len(rows)} rows in total", flush=True)
    if args.dry_run:
        return

    if extra:
        Path(args.removed_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.removed_log, "a") as f:
            for r in extra:
                f.write(json.dumps(r) + "\n")
        print(f"removed rows saved to {args.removed_log}")
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
