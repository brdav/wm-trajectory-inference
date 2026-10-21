#!/usr/bin/env python3
"""Merge scripts/estimate_metric_scale.py's per-clip metric_scale.npz files into the clips' own
result.npz inside the packed session zips, so each clip stays ONE file.

For every clip with a metric_scale.npz under --work_dir, its result.npz inside
<traj_root>/<Country/.../Session>.zip gets three extra keys (nothing else is touched):
    metric_scale       meters per SLAM unit (scalar)
    metric_poses       slam_poses with translation * metric_scale, (N,4,4)
    metric_scale_json  provenance + quality stats (source, model, p25/p75, rel_iqr, n_frames_used, ...)
see vggt_slam.slam_utils.save_result_npz / load_result_npz. These are kept apart from fused_* on
purpose: fused_poses are GPS/road grounded, these come from monocular metric depth.

Safety (the zips are shared, archival and not otherwise backed up):
  * the scale was computed on a SLAM *rerun*, which is bit-identical to the stored run for ~99.6%
    of clips but not all (GPU nondeterminism). Checked per clip, not assumed:
      - same frame_ids and poses (atol 1e-6): scale applies as is;
      - same frame_ids, poses differ only by an arbitrary SLAM gauge (rerun ~= similarity of stored,
        max residual <= MAX_ALIGN_RESID of the path length): the scale is TRANSFERRED to the stored
        poses via the alignment's scale (scale_stored = scale_rerun * umeyama_scale); the entry is
        marked transferred_from_rerun in metric_scale_json and in the log;
      - otherwise (the reruns really diverged): skipped and logged as "mismatch";
  * the new zip is written next to the old one under a temp name, then re-opened and verified
    (zip CRC test; every untouched member's CRC equals the original's; every updated result.npz
    keeps all its original arrays bit-for-bit and gains the three keys) BEFORE os.replace();
  * --dry_run does all the reading/matching/checking and writes nothing;
  * idempotent: clips whose result.npz already has metric_scale are skipped (--force to redo).

One line per session is appended to <pipeline_dir>/integration_log.jsonl.

Usage:
    python3 scripts/integrate_metric_scale.py --work_dir $SCRATCH/natix_metric_scale_work \\
        --traj_root .../processed_datasets/natix/trajectories --dry_run
    python3 scripts/integrate_metric_scale.py ... [--countries Austria] [--limit_sessions 2] [--workers 16]
"""
import argparse
import copy
import io
import json
import os
import shutil
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_TRAJ_ROOT = "/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories"
DEFAULT_PIPELINE_DIR = f"{DEFAULT_TRAJ_ROOT}/_pipeline"  # integration_log.jsonl goes here
DEFAULT_WORK_DIR = os.path.join(os.environ.get("SCRATCH", "/tmp"), "natix_metric_scale_work")  # per-clip files from estimate_metric_scale.py
NEW_KEYS = ("metric_scale", "metric_poses", "metric_scale_json")
# Fallbacks for stats not stored in metric_scale.npz by the first sweep (its fixed settings).
DEFAULT_ESTIMATOR = {"model": "depth-anything/DA3METRIC-LARGE", "n_frames_requested": 50,
                     "pcd_stride": 3, "process_res": 504}


def split_rel(rel):
    """Country/.../Session/HH-MM-SS/FRONT_FOLDER/FRONT_x  ->  (Country/.../Session, inner zip path)."""
    parts = rel.split("/")
    return "/".join(parts[:-3]), "/".join(parts[-3:]) + "/result.npz"


def find_clips(metric_root, countries):
    metric_root = Path(metric_root)
    if not metric_root.is_dir():  # nothing estimated since the last cleanup
        return defaultdict(list)
    by_session = defaultdict(list)
    for country_dir in sorted(p for p in metric_root.iterdir() if p.is_dir() and p.name != "logs"):
        if countries and country_dir.name not in countries:
            continue
        for f in country_dir.rglob("metric_scale.npz"):
            rel = str(f.parent.relative_to(metric_root))
            session, inner = split_rel(rel)
            by_session[session].append((inner, str(f)))
    return by_session


MAX_ALIGN_RESID = 0.05  # max |rerun - aligned stored| position residual / path length to transfer a scale


def align_scale(stored_poses, rerun_poses):
    """Similarity alignment (Umeyama, positions only) of the stored trajectory onto the rerun's.
    Returns (umeyama_scale, max_residual / path_length) or None if degenerate. positions_rerun ~=
    scale * R @ positions_stored + t, so  meters = scale_rerun * positions_rerun
    = (scale_rerun * scale) * positions_stored."""
    P, Q = stored_poses[:, :3, 3].astype(float), rerun_poses[:, :3, 3].astype(float)
    length = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    if len(P) < 3 or length <= 0 or not (np.isfinite(P).all() and np.isfinite(Q).all()):
        return None
    mu_p, mu_q = P.mean(0), Q.mean(0)
    p0, q0 = P - mu_p, Q - mu_q
    U, S, Vt = np.linalg.svd(q0.T @ p0 / len(P))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var = (p0 ** 2).sum() / len(P)
    if var <= 0:
        return None
    scale = float((S * np.diag(D)).sum() / var)
    resid = np.linalg.norm(scale * (R @ p0.T).T + mu_q - Q, axis=1).max() / length
    return scale, float(resid)


def build_updated_npz(old_bytes, ms, force):
    """(new_npz_bytes | None, reason, detail). Original arrays are copied through untouched.
    reason: updated | transferred | already_integrated | mismatch. detail: dict for the log."""
    old = np.load(io.BytesIO(old_bytes))
    if "metric_scale" in old.files and not force:
        return None, "already_integrated", None
    same_ids = np.array_equal(old["frame_ids"], ms["frame_ids"]) and old["slam_poses"].shape == ms["slam_poses"].shape
    if not same_ids:
        return None, "mismatch", {"why": "frame_ids differ"}
    exact = np.allclose(old["slam_poses"], ms["slam_poses"], rtol=0, atol=1e-6)
    stats = json.loads(str(ms["stats_json"])) if "stats_json" in ms.files else {}
    info = {"source": "metric_depth", **DEFAULT_ESTIMATOR, **stats,
            "integrated_at": datetime.now().isoformat(timespec="seconds")}
    scale, reason, detail = float(ms["scale"]), "updated", None
    if not exact:
        al = align_scale(old["slam_poses"], ms["slam_poses"])
        if al is None or al[1] > MAX_ALIGN_RESID:
            return None, "mismatch", {"why": "trajectories differ", "resid": None if al is None else al[1]}
        mu, resid = al
        for k in ("scale", "scale_p25", "scale_p75"):
            if k in info:
                info[k] = float(info[k]) * mu
        info["scale_rerun"] = scale
        info["transferred_from_rerun"] = {"umeyama_scale": mu, "resid_max_over_pathlen": resid}
        scale *= mu
        reason = "transferred"
        detail = {"scale": scale, "scale_p25": info.get("scale_p25"), "scale_p75": info.get("scale_p75"),
                  "umeyama_scale": mu, "resid_max_over_pathlen": resid}
    poses = np.array(old["slam_poses"], copy=True)
    poses[:, :3, 3] *= scale
    arrays = {k: old[k] for k in old.files if k not in NEW_KEYS}
    arrays.update(metric_scale=np.asarray(scale, dtype=float), metric_poses=poses,
                  metric_scale_json=np.asarray(json.dumps(info)))
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue(), reason, detail


def same_array(a, b):
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)


def verify(tmp_path, orig_infos, updates):
    with zipfile.ZipFile(tmp_path) as z:
        if z.testzip() is not None:
            raise RuntimeError("zip CRC test failed")
        new_infos = {i.filename: i for i in z.infolist()}
        if set(new_infos) != set(orig_infos):
            raise RuntimeError("member list differs from original")
        for name, oi in orig_infos.items():
            if name not in updates and new_infos[name].CRC != oi.CRC:
                raise RuntimeError(f"untouched member changed: {name}")
        for name, (new_bytes, old_bytes) in updates.items():
            new, old = np.load(io.BytesIO(z.read(name))), np.load(io.BytesIO(old_bytes))
            for k in old.files:
                if not same_array(new[k], old[k]):
                    raise RuntimeError(f"{name}: array {k} changed")
            if not all(k in new.files for k in NEW_KEYS):
                raise RuntimeError(f"{name}: new keys missing")


def process_session(session, clips, traj_root, dry_run, force):
    t0 = time.time()
    zip_path = Path(traj_root) / (session + ".zip")
    rec = {"session": session, "n_clips": len(clips), "updated": 0, "skipped": {}, "status": "ok",
           "updated_inner": [], "transferred": {}, "mismatch_detail": {}}
    try:
        if not zip_path.exists():
            rec["status"] = "no_zip"
            return rec
        updates = {}  # inner name -> (new_bytes, old_bytes)
        with zipfile.ZipFile(zip_path) as zin:
            names = set(zin.namelist())
            for inner, ms_path in clips:
                if inner not in names:
                    rec["skipped"][inner] = "no_result_npz_in_zip"
                    continue
                old_bytes = zin.read(inner)
                new_bytes, reason, detail = build_updated_npz(old_bytes, np.load(ms_path), force)
                if new_bytes is None:
                    rec["skipped"][inner] = reason
                    if detail:
                        rec["mismatch_detail"][inner] = detail
                else:
                    updates[inner] = (new_bytes, old_bytes)
                    if reason == "transferred":
                        rec["transferred"][inner] = detail
            rec["updated"] = len(updates)
            rec["updated_inner"] = sorted(updates)
            if dry_run or not updates:
                rec["status"] = "dry_run" if dry_run else "nothing_to_do"
                return rec
            orig_infos = {i.filename: i for i in zin.infolist()}
            tmp_path = zip_path.with_name(zip_path.name + f".tmp-{os.getpid()}")
            try:
                with zipfile.ZipFile(tmp_path, "w") as zout:
                    for info in zin.infolist():
                        if info.filename in updates:
                            zi = zipfile.ZipInfo(info.filename, info.date_time)
                            zi.compress_type, zi.external_attr = info.compress_type, info.external_attr
                            zout.writestr(zi, updates[info.filename][0])
                        else:
                            zout.writestr(copy.copy(info), zin.read(info.filename))
                verify(tmp_path, orig_infos, updates)
                shutil.copymode(zip_path, tmp_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
        os.replace(tmp_path, zip_path)
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default=DEFAULT_WORK_DIR, help="Per-clip metric_scale.npz files made by estimate_metric_scale.py")
    ap.add_argument("--traj_root", default=DEFAULT_TRAJ_ROOT)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--pipeline_dir", default=DEFAULT_PIPELINE_DIR, help="Where integration_log*.jsonl goes")
    ap.add_argument("--limit_sessions", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Redo clips that already have metric_scale")
    args = ap.parse_args()

    by_session = find_clips(args.work_dir, set(args.countries) if args.countries else None)
    sessions = sorted(by_session)[:args.limit_sessions]
    print(f"{sum(len(by_session[s]) for s in sessions)} clips in {len(sessions)} sessions"
          f"{' (DRY RUN)' if args.dry_run else ''}", flush=True)

    counts = defaultdict(int)
    skipped = defaultdict(int)
    n_updated = n_transferred = 0
    log_path = Path(args.pipeline_dir) / ("integration_log_dryrun.jsonl" if args.dry_run else "integration_log.jsonl")
    with ProcessPoolExecutor(args.workers) as pool, open(log_path, "a") as log:
        futs = [pool.submit(process_session, s, by_session[s], args.traj_root, args.dry_run, args.force)
                for s in sessions]
        for i, fut in enumerate(futs, 1):
            rec = fut.result()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            counts[rec["status"]] += 1
            n_updated += rec["updated"]
            n_transferred += len(rec["transferred"])
            for reason in rec["skipped"].values():
                skipped[reason] += 1
            if rec["status"] == "failed":
                print(f"[FAIL] {rec['session']}: {rec['error']}", file=sys.stderr, flush=True)
            if i % 200 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] sessions={dict(counts)} clips_updated={n_updated} (transferred {n_transferred}) "
                      f"clips_skipped={dict(skipped)}", flush=True)
    print(f"Done. Log: {log_path}")
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
