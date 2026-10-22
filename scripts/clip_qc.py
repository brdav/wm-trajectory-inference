#!/usr/bin/env python3
"""
clip_qc.py - per-clip quality control that separates the two ways a trajectory can be bad.

  * ego-trajectory (SLAM) failure: the visual odometry itself diverged. The poses are unusable
    and must not be used at all (not even re-scaled).
  * fusion/scale failure: SLAM is fine but the GPS / map-matching step gave it a wrong metric
    scale (or none). `fused_poses` are unusable, but `slam_poses` can be re-scaled, i.e. the clip
    belongs on the scripts/estimate_metric_scale.py path.

This works because fused_poses are slam_poses under ONE global similarity transform (verified
over the whole dataset: the fused/SLAM step-length ratio is constant within a clip), so the two
failures are independently checkable:

  slam_ok   scale-free test on slam_poses alone -- no GPS needed. A keyframe step more than
            SLAM_JUMP_RATIO x the clip's typical step is a divergence. `slam_valid_until_s` is
            the time of the keyframe before the first jump: the clean prefix of a clip that only
            blows up late. Such a clip is still route=ego_failed (never re-scale a clip that
            contains a divergence); a caller that wants the prefix can truncate at that time.
  scale_ok  only meaningful when slam_ok and fused_poses exist. Fails if the fused trajectory
            implies an impossible ground speed (max over SCALE_WINDOW_S windows > SCALE_MAX_MPS)
            or is much longer than the raw GPS track over the same time span
            (> SCALE_MAX_GPS_RATIO).

`gps_fusion.status == "ok"` is NOT used: the known bad clips all carry it.

route (the column to act on):
  static             fewer than MIN_KEYFRAMES keyframes AND the raw GPS fixes span < STATIC_MAX_GPS_EXTENT_M:
                     the vehicle barely moved, this is not a SLAM failure (gps_extent_m is recorded);
                     with no GPS to verify, or a larger extent, the clip stays ego_failed
  needs_scale_check  slam_ok and scale_ok, but the fused/GPS length ratio is outside
                     [SCALE_CHECK_LOW, SCALE_CHECK_HIGH] or could not be computed, or the map-match
                     error is above MAP_MATCH_CHECK_M -> fused_poses
                     stay, but estimate_metric_scale.py cross-checks them (scale_reason says why)
  use_fused          slam_ok and scale_ok            -> fused_poses  (gps_fusion status no_route_match
                     is never use_fused: its scale is not trusted, it goes to needs_metric_scale)
  needs_metric_scale slam_ok, fusion failed/missing  -> estimate_metric_scale.py path
  ego_failed         not slam_ok                     -> never use this clip's poses
  no_result          result.npz unreadable / empty (counts as ego_failed downstream)

Reads only the existing result.npz files (and the raw GPS CSVs for the length check); CPU only.
Thresholds were set from the distribution over ~120k clips and not checked against ground
truth -- see the constants below and the printed summary.

Usage:  python scripts/clip_qc.py --out clip_qc.csv [--workers 32] [--limit_zips N]
"""
import argparse
import io
import json
import os
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vggt_slam import gps as gps_module
from vggt_slam import paths

TRAJ_ROOT = Path(paths.TRAJ_ROOT)
DATASET_ROOT = Path(paths.DATASET_ROOT)

QC_VERSION = "2026-10-21.1"  # bump whenever a threshold or routing rule below changes
# Evidence behind each threshold: memory/qc-thresholds-rationale.md
MIN_KEYFRAMES = 4
SLAM_JUMP_RATIO = 50.0     # step / typical step; typical = max(median, 0.25 * p90)
SLAM_RUN_RATIO = 10.0      # ...or SLAM_RUN_LEN consecutive steps above this: catches a sustained
SLAM_RUN_LEN = 6           # divergence whose individual steps stay under SLAM_JUMP_RATIO
SCALE_WINDOW_S = 5.0       # window for the implied-speed check (= one 121-frame sample)
SCALE_MAX_MPS = 50.0       # 180 km/h
SCALE_MAX_GPS_RATIO = 1.5  # fused path length / raw GPS path length over the GPS time span (hard limit)
SCALE_CHECK_LOW = 0.8      # passes the hard checks but the ratio is outside [LOW, HIGH] (or could not be
SCALE_CHECK_HIGH = 1.25    # computed): route needs_scale_check -> cross-checked with a depth-based scale
MAP_MATCH_CHECK_M = 10.0   # fused path's mean distance to its map-matched road above this: needs_scale_check
STATIC_MAX_GPS_EXTENT_M = 30.0  # a clip with < MIN_KEYFRAMES keyframes whose raw GPS fixes span less than this
                           # barely moved (SLAM rarely adds keyframes without motion): route static, not ego_failed
SCALE_MIN_GPS_LEN_M = 30.0  # below this the GPS length is jitter-dominated, skip the ratio


def check_slam(t, slam_poses):
    """Returns (slam_ok, reason, slam_valid_until_s, max_step_ratio)."""
    if len(t) < MIN_KEYFRAMES:
        return False, "too_few_keyframes", np.nan, np.nan
    if not np.isfinite(slam_poses).all():
        return False, "nonfinite_poses", np.nan, np.nan
    steps = np.linalg.norm(np.diff(slam_poses[:, :3, 3], axis=0), axis=1)
    typical = max(np.median(steps), 0.25 * np.percentile(steps, 90), 1e-9)
    ratio = steps / typical
    hi = ratio > SLAM_RUN_RATIO
    run = hi[:-SLAM_RUN_LEN + 1] if SLAM_RUN_LEN > 1 else hi
    for k in range(1, SLAM_RUN_LEN):
        run = run & hi[k:len(hi) - SLAM_RUN_LEN + 1 + k]
    jump = np.union1d(np.flatnonzero(ratio > SLAM_JUMP_RATIO), np.flatnonzero(run))
    if len(jump) == 0:
        return True, "", t[-1], float(ratio.max())
    return False, "slam_jump", float(t[jump[0]]), float(ratio.max())


def window_speed_max(t, xz, window_s):
    """Max over keyframes of the displacement over the next >= window_s seconds / elapsed time."""
    j = np.searchsorted(t, t + window_s)
    ok = j < len(t)
    if not ok.any():
        return np.nan
    i = np.flatnonzero(ok)
    d = np.linalg.norm(xz[j[i]] - xz[i], axis=1)
    return float((d / (t[j[i]] - t[i])).max())


def check_scale(t, fused_poses, csv_path):
    """Returns (scale_ok, reason, max_window_speed_mps, fused_over_gps_len)."""
    xz = fused_poses[:, :3, 3][:, [0, 2]]
    if not np.isfinite(xz).all():
        return False, "nonfinite_fused", np.nan, np.nan
    vmax = window_speed_max(t, xz, SCALE_WINDOW_S)
    if np.isfinite(vmax) and vmax > SCALE_MAX_MPS:
        return False, "implausible_speed", vmax, np.nan
    ratio = np.nan
    try:
        g = gps_module.load_gps_fixes(str(csv_path))
        ge = np.stack([g["easting"], g["northing"]], axis=1)
        gl = np.linalg.norm(np.diff(ge, axis=0), axis=1).sum()
        inside = (t >= g["elapsed_s"].min()) & (t <= g["elapsed_s"].max())
        if inside.sum() >= 2 and gl >= SCALE_MIN_GPS_LEN_M:
            fl = np.linalg.norm(np.diff(xz[inside], axis=0), axis=1).sum()
            ratio = float(fl / gl)
    except Exception:
        pass  # no/unreadable GPS log: speed check only
    if np.isfinite(ratio) and ratio > SCALE_MAX_GPS_RATIO:
        return False, "longer_than_gps", vmax, ratio
    return True, "", vmax, ratio


def gps_extent_m(csv_path):
    """Diagonal of the bounding box of the clip's raw GPS fixes (m), nan if no usable GPS."""
    try:
        g = gps_module.load_gps_fixes(str(csv_path))
        e = np.stack([g["easting"], g["northing"]], axis=1)
        return float(np.linalg.norm(e.max(0) - e.min(0)))
    except Exception:
        return float("nan")


def write_config(csv_path, df=None):
    """<csv>.config.json: which QC (version, git commit of this code, every threshold) produced the CSV and
    how many clips ended on each route -- so a manifest row's qc entry can always be traced back."""
    import datetime
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*a):
        try:
            return subprocess.run(["git", "-C", here, *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    if df is None:
        df = pd.read_csv(csv_path)
    consts = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str)) and k != "QC_VERSION"
              and not k.endswith("_ROOT")}
    cfg = dict(qc_version=QC_VERSION, created=datetime.datetime.now().isoformat(timespec="seconds"),
               git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", os.path.abspath(__file__))),
               script="scripts/clip_qc.py", thresholds=consts,
               n_clips=int(len(df)), routes={k: int(v) for k, v in df.route.value_counts().items()})
    with open(str(csv_path) + ".config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def raw_csv_path(zip_path, member):
    prefix = zip_path.relative_to(TRAJ_ROOT).parent
    session = zip_path.stem
    hms, front_folder, stem, _ = member.split("/")
    return DATASET_ROOT / prefix / session / hms / front_folder / f"{stem}.csv"


def process_zip(zip_path):
    rows = []
    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception:
        return rows
    for member in zf.namelist():
        if not member.endswith("/result.npz"):
            continue
        csv_path = raw_csv_path(zip_path, member)
        clip = str(csv_path.relative_to(DATASET_ROOT).with_suffix(".mp4"))
        row = dict(clip=clip, route="no_result")
        try:
            z = np.load(io.BytesIO(zf.read(member)))
            t = (z["frame_ids"] - 1) / float(z["extraction_fps"])  # same time base as gps_fusion
            row["n_keyframes"] = len(t)
            row["has_metric_scale"] = "metric_scale" in z.files  # already integrated, see scripts/integrate_metric_scale.py
            slam_ok, reason, valid_until, max_ratio = check_slam(t, z["slam_poses"])
            row.update(slam_ok=slam_ok, slam_reason=reason, slam_valid_until_s=valid_until,
                       slam_max_step_ratio=max_ratio)
            if "gps_fusion_json" in z.files:
                info = json.loads(str(z["gps_fusion_json"]))
                row["gps_fusion_status"] = info.get("status")
                row["map_match_error_mean_m"] = info.get("map_match_error_mean_m")
            if not slam_ok:
                row["route"] = "ego_failed"
                if reason == "too_few_keyframes":
                    row["gps_extent_m"] = ext = gps_extent_m(csv_path)
                    if np.isfinite(ext) and ext < STATIC_MAX_GPS_EXTENT_M:
                        row["route"] = "static"  # (GPS-verified) barely moved: not a SLAM failure
            elif "fused_poses" not in z.files:
                row.update(scale_ok=False, scale_reason="no_fused_poses", route="needs_metric_scale")
            else:
                scale_ok, reason, vmax, ratio = check_scale(t, z["fused_poses"], csv_path)
                row.update(scale_ok=scale_ok, scale_reason=reason, fused_max_window_mps=vmax,
                           fused_over_gps_len=ratio,
                           route="use_fused" if scale_ok else "needs_metric_scale")
                # fusion placed the trajectory but found no matching road route: the GPS-fit scale
                # is not trusted, get a depth-based one instead
                if row.get("gps_fusion_status") == "no_route_match" and row["route"] == "use_fused":
                    row.update(scale_ok=False, scale_reason="no_route_match", route="needs_metric_scale")
                # passes every hard check, but close enough to a limit (or unverifiable) to be
                # cross-checked against an independent depth-based scale
                if row["route"] == "use_fused":
                    ratio = row.get("fused_over_gps_len")
                    mme = row.get("map_match_error_mean_m")
                    why = ("no_gps_len_ratio" if ratio is None or not np.isfinite(ratio) else
                           "len_ratio_low" if ratio < SCALE_CHECK_LOW else
                           "len_ratio_high" if ratio > SCALE_CHECK_HIGH else
                           "map_match_error_high" if mme is not None and mme > MAP_MATCH_CHECK_M else "")
                    if why:
                        row.update(scale_reason=why, route="needs_scale_check")
        except Exception as e:
            row.update(route="no_result", slam_reason=f"unreadable: {str(e)[:60]}")
        rows.append(row)
    return rows


def main():
    global TRAJ_ROOT  # --traj_root overrides the module default; must be declared before TRAJ_ROOT is used below
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj_root", default=str(TRAJ_ROOT), help="Trajectories root holding the session zips (default: the natix output)")
    ap.add_argument("--out", default=str(TRAJ_ROOT / "_pipeline" / "clip_qc.csv"), help="Output CSV, one row per clip")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit_zips", type=int, default=None, help="testing")
    ap.add_argument("--config_only", action="store_true",
                    help="Do not recompute: only (re)write <out>.config.json for an existing --out CSV")
    args = ap.parse_args()

    if args.config_only:
        print(json.dumps(write_config(args.out), indent=2))
        return
    TRAJ_ROOT = Path(args.traj_root)  # set before the worker pool forks (see the `global` at the top of main)
    zips = sorted(TRAJ_ROOT.rglob("*.zip"))[:args.limit_zips]
    print(f"{len(zips)} zips")
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(process_zip, zips, chunksize=8), 1):
            rows += r
            if k % 5000 == 0:
                print(f"  {k}/{len(zips)} zips", flush=True)
    df = pd.DataFrame(rows).sort_values("clip")
    df.to_csv(args.out, index=False)
    write_config(args.out, df)
    print(f"wrote {args.out} (+ .config.json): {len(df)} clips")
    print(df.route.value_counts().to_string())
    print("\nreasons:")
    for col in ("slam_reason", "scale_reason"):
        print(df[col].replace("", np.nan).value_counts().head(8).to_string())
    if "gps_fusion_status" in df:
        print("\nroute x gps_fusion_status:")
        print(pd.crosstab(df.gps_fusion_status.fillna("(none)"), df.route).to_string())


if __name__ == "__main__":
    main()
