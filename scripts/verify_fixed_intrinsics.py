#!/usr/bin/env python3
"""Make `fixed_intrinsics` in trajectories/manifest.jsonl definitive wherever there is hard evidence, instead of inferred.

The question per clip: did the original run use the camera calibration (fixed intrinsics) or fall back to
self-calibration? batch_infer.py recorded the answer for ~72,800 clips; for the rest it was inferred from the session
zips (scripts/backfill_manifest.py), which misses clips that fell back individually (a race in the undistortion cache,
fixed in b722ac4). This script overrides a value wherever there is hard evidence, in decreasing order of precedence:

  1. rerun_identical / rerun_differs  (--rerun_root, repeatable): a rerun of the clip with the CURRENT code (fixed
     intrinsics) in another output folder, compared with the stored result.npz. Reruns of fixed-intrinsics clips
     reproduce the stored poses exactly (100 of 100 control clips), so identical poses mean the stored run used fixed
     intrinsics and different poses mean it self-calibrated. A rerun whose own recorded fixed_intrinsics is false proves
     nothing and is skipped.
  2. estimator_rerun_identical / estimator_rerun_differs / estimator_rerun_refused: the same comparison done by
     scripts/estimate_metric_scale.py + integrate_metric_scale.py for every clip that needed a depth-based scale (a
     SLAM rerun with fixed intrinsics): integrated without `transferred_from_rerun` = identical poses; integrated with
     it, or refused (poses too different to transfer the scale) = the stored run differs.
  3. session_without_maps: the session's zip has no undistort_maps.npz, so no clip of that session used fixed
     intrinsics.

For every clip with such evidence `fixed_intrinsics` is set to the evidenced value; clips without evidence keep
their value (recorded by the original run, or inferred from the session maps). The evidence per clip is written
to a CSV (--evidence_out) so every verified value stays traceable.
Writes the manifest in place after saving a timestamped backup next to it. Idempotent.

Usage:
    python3 scripts/verify_fixed_intrinsics.py --dry_run [--rerun_root $SCRATCH/rerun576/out]
"""
import argparse
import csv
import io
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vggt_slam.sample_manifest import DEFAULT_DATASET_ROOT, DEFAULT_TRAJECTORY_ROOT, session_of  # noqa: E402
import update_manifest_metric_scale as ums  # noqa: E402  (integration_status)

PRECEDENCE = ("rerun_identical", "rerun_differs", "estimator_rerun_identical", "estimator_rerun_differs",
              "estimator_rerun_refused", "session_without_maps")


def _member(clip_rel):
    rel = clip_rel[:-len(".mp4")].split("/")
    return "/".join(rel[:-3]), "/".join(rel[-3:]) + "/result.npz"


def read_estimator_evidence(job):
    """(session, [(clip_rel, transferred)]) for clips of one session that carry a depth scale in their result.npz."""
    session, clips, root = job
    out = []
    try:
        with zipfile.ZipFile(Path(root) / (session + ".zip")) as zf:
            names = set(zf.namelist())
            for clip in clips:
                inner = _member(clip)[1]
                if inner in names:
                    z = np.load(io.BytesIO(zf.read(inner)))
                    if "metric_scale_json" in z.files:
                        out.append((clip, "transferred_from_rerun" in json.loads(str(z["metric_scale_json"]))))
    except Exception:
        pass
    return session, out


def compare_rerun(job):
    """(clip_rel, 'rerun_identical' | 'rerun_differs' | None): stored poses vs a rerun, for one clip."""
    clip, root_stored, root_rerun = job
    sess, inner = _member(clip)
    try:
        with zipfile.ZipFile(Path(root_stored) / (sess + ".zip")) as a, zipfile.ZipFile(Path(root_rerun) / (sess + ".zip")) as b:
            za, zb = np.load(io.BytesIO(a.read(inner))), np.load(io.BytesIO(b.read(inner)))
        same = (np.array_equal(za["frame_ids"], zb["frame_ids"]) and za["slam_poses"].shape == zb["slam_poses"].shape
                and float(np.abs(za["slam_poses"] - zb["slam_poses"]).max()) <= 1e-6)
        return clip, "rerun_identical" if same else "rerun_differs"
    except Exception:
        return clip, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=f"{DEFAULT_TRAJECTORY_ROOT}/manifest.jsonl")
    ap.add_argument("--out_root", default=DEFAULT_TRAJECTORY_ROOT, help="Trajectories root holding the session zips")
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--rerun_root", action="append", default=[], help="Trajectories-style output of a rerun with the current code (repeatable)")
    ap.add_argument("--integration_log", default=f"{DEFAULT_TRAJECTORY_ROOT}/_pipeline/integration_log.jsonl")
    ap.add_argument("--evidence_out", default=f"{DEFAULT_TRAJECTORY_ROOT}/_pipeline/fixed_intrinsics_evidence.csv")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = args.dataset_root.rstrip("/") + "/"
    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    by_clip = {r["clip_id"][len(root):]: r for r in rows}
    evidence = {}  # clip_rel -> (value, source)

    def add(clip, value, source):
        old = evidence.get(clip)
        if old is None or PRECEDENCE.index(source) < PRECEDENCE.index(old[1]):
            evidence[clip] = (value, source)

    with Pool(args.workers) as pool:
        # 1. reruns with the current code
        for rr in args.rerun_root:
            rerun_rows = [json.loads(l) for l in open(Path(rr) / "manifest.jsonl") if l.strip()]
            todo = [r["clip_id"][len(root):] for r in rerun_rows if r.get("fixed_intrinsics") is True and r["clip_id"][len(root):] in by_clip]
            for clip, res in pool.imap(compare_rerun, [(c, args.out_root, rr) for c in todo], chunksize=16):
                if res:
                    add(clip, res == "rerun_identical", res)
            print(f"rerun {rr}: {len(todo)} clips compared", flush=True)
        # 2. estimator reruns (depth-scale step), read from each clip's result.npz
        cand = [c for c, r in by_clip.items() if r["metric_scale"]["source"] == "metric_depth" or r["scale_check"]["depth_scale"] is not None]
        by_sess = {}
        for c in cand:
            by_sess.setdefault(session_of(c), []).append(c)
        for _, items in pool.imap(read_estimator_evidence, [(s, cs, args.out_root) for s, cs in by_sess.items()], chunksize=50):
            for clip, transferred in items:
                add(clip, not transferred, "estimator_rerun_differs" if transferred else "estimator_rerun_identical")
        integ = ums.integration_status(args.integration_log)
        for cid, v in integ.items():
            if v["state"] == "mismatch" and cid[len(root):] in by_clip:
                add(cid[len(root):], False, "estimator_rerun_refused")
        # 3. sessions whose zip has no maps file
        sessions = sorted({session_of(c) for c in by_clip})
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from backfill_manifest import session_has_maps
        maps = dict(pool.map(session_has_maps, [(s, args.out_root) for s in sessions], chunksize=200))
    for c, r in by_clip.items():
        if maps.get(session_of(c)) is False:
            add(c, False, "session_without_maps")

    # ---- what changes
    stat = Counter()
    changes = []
    for clip, (value, source) in sorted(evidence.items()):
        r = by_clip[clip]
        stat[(source, r["fixed_intrinsics"], value)] += 1
        changes.append((clip, value, source, r["fixed_intrinsics"]))
    print(f"\nevidence for {len(evidence)} of {len(rows)} clips. (source, previous value -> evidenced value): rows")
    for k, v in sorted(stat.items(), key=lambda x: -x[1]):
        print(f"  {v:6d}  {k[0]:28s} {str(k[1]):5s} -> {k[2]}")
    print(f"\nvalue changes: {sum(1 for c, v, s, old in changes if old != v)}")
    if args.dry_run:
        return

    for clip, value, source, old in changes:
        by_clip[clip]["fixed_intrinsics"] = value
    Path(args.evidence_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.evidence_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "fixed_intrinsics", "evidence", "previous_value"])
        for clip, value, source, old in changes:
            w.writerow([root + clip, value, source, old])
    bak = Path(f"{args.manifest}.bak_{time.strftime('%Y%m%dT%H%M%S')}")
    shutil.copy2(args.manifest, bak)
    print(f"backup: {bak}")
    tmp = Path(args.manifest + f".tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    shutil.copymode(args.manifest, tmp)
    os.replace(tmp, args.manifest)
    print(f"wrote {args.manifest}; evidence in {args.evidence_out}")


if __name__ == "__main__":
    main()
