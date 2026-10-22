#!/usr/bin/env python3
"""Add a `metric_scale` entry to every row of trajectories/manifest.jsonl, so the manifest alone
says, per clip, where its metric scale comes from and how good it is:

    "metric_scale": {"source": "gps_fusion" | "metric_depth" | null,
                     "scale": <meters per SLAM unit> | null, ...}

  gps_fusion    clips with fused_poses (gps_fusion status ok / no_route_match): scale = total
                step length of fused_poses / total step length of slam_poses (they differ by one
                global similarity, so this ratio is the similarity's scale). Computed here by
                reading each clip's result.npz from its session zip.
  metric_depth  clips whose scale came from scripts/estimate_metric_scale.py (taken from that
                run's manifest): adds scale_p25 / scale_p75 / scale_rel_iqr (spread of the
                per-frame estimates), n_frames_used (clips with few frames are weak evidence --
                filter on it) and model. The scale was computed on a SLAM rerun; if its gauge
                differed from the stored run, `scale` is the value transferred to the stored poses
                (scale_transferred true, transfer_residual = alignment residual / path length).
                Rule: a clip that went through estimate_metric_scale.py ALWAYS reports metric_depth,
                even if clip_qc.py's route says use_fused (e.g. the no_route_match clips); its GPS
                scale, when one exists, is kept as `gps_fusion_scale` for comparison.
  null          no usable scale. `status`/`error` say why when the metric-depth run failed
                (e.g. no_calibration); clips never attempted just have source null.

Clips clip_qc.py routed to needs_scale_check (fused_poses pass but are near a limit or
unverifiable) additionally get `scale_check` {depth_scale, ratio = depth/GPS, verdict, n_frames_used,
scale_rel_iqr}: confirmed (ratio 0.8-1.25), ambiguous, contradicted (outside 0.67-1.5) or unavailable.
They keep `fused` unless contradicted; then `metric_scale.source` is null and pose_source none. Exception: a
contradiction on a clip flagged only for its map-matching error is overruled when the raw GPS track length
agrees with the fused scale (`scale_check.adjudication` = gps_length_supports_fused; pose_source fused).
`scale_check.confidence` is "low" for noisy depth estimates (scale_rel_iqr > 0.25), which are judged on their
whole interquartile range instead of the median (see scale_check()).

Every row is normalized to a fixed schema (normalize_row): all keys always present, null when not applicable,
no gpu/pid.

With --qc_csv every row also gets `pose_source` (fused / metric / none) and `pose_source_reason`:
the single field downstream consumers should branch on (see pose_source() below).

With --qc_csv (scripts/clip_qc.py's output) every row also gets a `qc` dict -- route
(use_fused / needs_metric_scale / ego_failed / no_result), slam_ok, slam_reason, n_keyframes,
slam_valid_until_s, scale_ok, scale_reason. `qc.route` is the verdict to act on: a clip routed to
needs_metric_scale whose depth-based scale is missing does NOT keep its (rejected) GPS scale as
`scale` -- it gets source null with status "gps_scale_rejected_by_qc" and the rejected value in
`gps_fusion_scale`. For ego_failed clips (diverged SLAM) check qc.route first: a scale entry, if
any, says nothing about whether the poses are usable.

The entry is only derived data: for metric_depth clips the same numbers (plus metric_poses) are
inside the clip's result.npz (see scripts/integrate_metric_scale.py). For gps_fusion clips the
scale lives only here (it can be recomputed from the clip's own fused/slam poses).

Writes --out (default: rewrite --manifest in place, atomically, after saving a timestamped .bak_<time> copy).
Rows keep their order; every other field is unchanged. Idempotent.

Usage:
    python3 scripts/update_manifest_metric_scale.py --dry_run [--limit_sessions 200]
    python3 scripts/update_manifest_metric_scale.py --out /tmp/manifest_new.jsonl     # review first
    python3 scripts/update_manifest_metric_scale.py                                    # in place
"""
import argparse
import io
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (vggt_slam package)
from vggt_slam import paths  # noqa: E402

TRAJ_ROOT = paths.TRAJ_ROOT
DATASET_ROOT = paths.DATASET_ROOT.rstrip("/") + "/"  # clip_ids are DATASET_ROOT + the dataset-relative path
PIPELINE_DIR = paths.PIPELINE_DIR
METRIC_MANIFEST = f"{PIPELINE_DIR}/manifest_metric_scale.jsonl"
INTEGRATION_LOG = f"{PIPELINE_DIR}/integration_log.jsonl"
FUSED_STATUSES = {"ok", "no_route_match"}


def split_clip_id(clip_id):
    rel = clip_id[len(DATASET_ROOT):-len(".mp4")] if clip_id.startswith(DATASET_ROOT) else None
    if rel is None:
        return None, None
    parts = rel.split("/")
    return "/".join(parts[:-3]), "/".join(parts[-3:]) + "/result.npz"


def step_length(poses):
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def gps_scales_for_session(args):
    """{clip_id: scale | None} for one session zip."""
    session, clips, traj_root = args
    out = {}
    try:
        z = zipfile.ZipFile(Path(traj_root) / (session + ".zip"))
    except Exception:
        return {cid: None for cid, _ in clips}
    with z:
        names = set(z.namelist())
        for cid, inner in clips:
            scale = None
            try:
                if inner in names:
                    d = np.load(io.BytesIO(z.read(inner)))
                    if "fused_poses" in d.files:
                        _, first = np.unique(d["frame_ids"], return_index=True)  # drop duplicate ids
                        first = np.sort(first)
                        slam_len, fused_len = step_length(d["slam_poses"][first]), step_length(d["fused_poses"][first])
                        if slam_len > 0:
                            scale = fused_len / slam_len
            except Exception:
                scale = None
            out[cid] = scale
    return out


def latest_by_clip(path):
    latest = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                latest[r["clip_id"]] = r
    return latest


def load_qc(path):
    """{clip_id: qc dict} from clip_qc.py's CSV (clip column = path relative to the dataset root)."""
    import csv
    import math

    def val(k, v):
        if v in ("", None):
            return None
        if k in ("slam_ok", "scale_ok"):
            return v == "True"
        if k == "n_keyframes":
            return int(float(v))
        if k in ("slam_valid_until_s", "gps_extent_m", "fused_over_gps_len", "fused_max_window_mps", "slam_max_step_ratio"):
            f = float(v)
            return None if math.isnan(f) else f
        return v

    keys = ("route", "slam_ok", "slam_reason", "n_keyframes", "slam_valid_until_s", "scale_ok", "scale_reason",
            "gps_extent_m", "fused_over_gps_len", "fused_max_window_mps", "slam_max_step_ratio")
    qc = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            qc[DATASET_ROOT + row["clip"]] = {k: val(k, row.get(k)) for k in keys}
    return qc


def integration_status(log_path):
    """{clip_id: {"state": "mismatch"} | {"state": "integrated", "transferred": {...} | None}} from
    integrate_metric_scale.py's append-only log; the LATEST record per clip wins (so a clip that
    mismatched in an early run but was integrated later is not stuck as mismatch).
    "mismatch" = the rerun used for the scale did not reproduce the stored poses and the
    trajectories were too different to transfer the scale (see integrate_metric_scale.py)."""
    status = {}
    if log_path and Path(log_path).exists():
        for line in open(log_path):
            r = json.loads(line)
            cid = lambda inner: f"{DATASET_ROOT}{r['session']}/{inner[:-len('/result.npz')]}.mp4"
            for inner, why in r.get("skipped", {}).items():
                if why == "mismatch":
                    status[cid(inner)] = {"state": "mismatch"}
            for inner in r.get("updated_inner", []):
                status[cid(inner)] = {"state": "integrated", "transferred": r.get("transferred", {}).get(inner)}
    return status


SCALE_CHECK_LOW, SCALE_CHECK_HIGH = 0.8, 1.25  # clip_qc.py's fused/GPS path-length band (see adjudicate())
AGREE = (0.8, 1.25)        # depth/GPS scale ratio counted as agreement (depth scale is only ~+-10%)
CLEAR = (1 / 1.5, 1.5)     # outside this: the two estimates clearly contradict each other
CONFIDENT_MIN_FRAMES, CONFIDENT_MAX_REL_IQR = 5, 0.25  # a depth scale weaker than this cannot contradict


def scale_check(gps, m, state):
    """Cross-check of a GPS-fused scale (clip_qc route needs_scale_check) against the depth-based one.
    m = that clip's estimate_metric_scale manifest record, state = its integration status.
    verdict: confirmed (ratio in AGREE) | ambiguous | contradicted (ratio outside CLEAR) |
             unavailable (no/failed/mismatched depth scale, or fewer than CONFIDENT_MIN_FRAMES frames).
    confidence: "high" when the per-frame depth estimates are consistent (scale_rel_iqr <=
    CONFIDENT_MAX_REL_IQR): the verdict is judged on the median ratio. "low" (noisy estimates): a clip can
    only be contradicted if the WHOLE interquartile range of its per-frame scales lies outside CLEAR on one
    side; "confirmed" needs the median in AGREE; anything else is ambiguous. So noisy depth can no longer
    block a verdict, but it cannot condemn a clip on a single unlucky median either."""
    out = {"depth_scale": None, "ratio": None, "verdict": "unavailable", "confidence": None,
           "n_frames_used": None, "scale_rel_iqr": None}
    if gps is None or m is None or not m.get("ok") or (state or {}).get("state") == "mismatch":
        return out
    tr = (state or {}).get("transferred") or {}
    depth, p25, p75 = tr.get("scale", m["scale"]), tr.get("scale_p25", m["scale_p25"]), tr.get("scale_p75", m["scale_p75"])
    out.update(depth_scale=depth, ratio=depth / gps, n_frames_used=m["n_frames_used"], scale_rel_iqr=m["scale_rel_iqr"])
    if m["n_frames_used"] < CONFIDENT_MIN_FRAMES:
        return out
    r = out["ratio"]
    if m["scale_rel_iqr"] <= CONFIDENT_MAX_REL_IQR:
        out["confidence"] = "high"
        out["verdict"] = "confirmed" if AGREE[0] <= r <= AGREE[1] else "contradicted" if not (CLEAR[0] <= r <= CLEAR[1]) else "ambiguous"
    else:
        out["confidence"] = "low"
        lo, hi = p25 / gps, p75 / gps
        out["verdict"] = ("contradicted" if hi < CLEAR[0] or lo > CLEAR[1] else
                          "confirmed" if AGREE[0] <= r <= AGREE[1] else "ambiguous")
    return out


def adjudicate(chk, q):
    """A contradicted cross-check on a clip that was only flagged for its map-matching error already passed the
    length-ratio checks: the fused path is 0.8-1.25x the raw GPS track length, an estimate that does not
    depend on depth. GPS fit and GPS track length agree against the depth scale (2 vs 1): keep the GPS scale.
    Returns the adjudication tag or None."""
    ratio = q.get("fused_over_gps_len")
    if (chk["verdict"] == "contradicted" and q.get("scale_reason") == "map_match_error_high"
            and ratio is not None and SCALE_CHECK_LOW <= ratio <= SCALE_CHECK_HIGH):
        return "gps_length_supports_fused"
    return None


# ---- row schema (docs/manifest-schema.md): every key below is ALWAYS present, null when it does not
# apply, so a reader never needs `if key in row`. Unknown extra keys are kept after the known ones.
TOP_KEYS = ("clip_id", "ok", "seconds", "fixed_intrinsics", "stage", "error", "gps_fusion", "qc", "metric_scale",
            "scale_check", "pose_source", "pose_source_reason")
GPS_FUSION_KEYS = ("status", "error", "region_used", "used_fallback", "network_profile_used", "matched_keyframes",
                   "total_keyframes", "map_match_error_mean_m", "map_match_error_max_m", "map_match_error_n_points")
METRIC_SCALE_KEYS = ("source", "scale", "scale_p25", "scale_p75", "scale_rel_iqr", "n_frames_used", "model",
                     "scale_transferred", "transfer_residual", "gps_fusion_scale", "depth_scale", "status", "reason", "error")
SCALE_CHECK_KEYS = ("depth_scale", "ratio", "verdict", "confidence", "n_frames_used", "scale_rel_iqr", "adjudication")
DROPPED_KEYS = ("gpu", "pid", "fixed_intrinsics_inferred")  # worker bookkeeping of older batch_infer versions / a retired provenance flag, not part of the row


def _fill(d, keys):
    d = dict(d or {})
    out = {k: d.pop(k, None) for k in keys}
    out.update(d)
    return out


def normalize_row(r):
    """The row with the uniform schema above (and without gpu / pid)."""
    r = {k: v for k, v in r.items() if k not in DROPPED_KEYS}
    r["gps_fusion"] = _fill(r.get("gps_fusion"), GPS_FUSION_KEYS)
    ms = _fill(r.get("metric_scale"), METRIC_SCALE_KEYS)
    if ms["source"] == "metric_depth" and ms["scale_transferred"] is None:
        ms["scale_transferred"] = False
    r["metric_scale"] = ms
    r["scale_check"] = _fill(r.get("scale_check"), SCALE_CHECK_KEYS)  # all null = not cross-checked
    return _fill(r, TOP_KEYS)


def pose_source(r):
    """The one field downstream should branch on: which poses of this clip's result.npz to use.
        fused   fused_poses   (clip_qc route use_fused: GPS/road-fused, metric; also route
                              needs_scale_check unless the depth scale clearly contradicts it)
        metric  metric_poses  (route needs_metric_scale and a depth-based scale exists)
        static  the vehicle barely moved (clip_qc route static: < 4 keyframes and a raw GPS extent
                < 30 m); poses are ~stationary, there is no meaningful trajectory or scale
        none    no usable poses (SLAM failed, no scale could be obtained, or a needs_scale_check
                clip whose GPS scale is clearly contradicted by the depth scale -- we cannot tell
                which of the two is wrong, so neither is trusted)
    plus a short reason (the clip_qc reason, or why no scale)."""
    q, m = r["qc"], r["metric_scale"]
    if q["route"] == "use_fused":
        return "fused", "gps_fusion"
    if q["route"] == "static":
        return "static", "stationary"
    if q["route"] == "needs_scale_check":
        v = (r.get("scale_check") or {}).get("verdict")
        if v == "contradicted":
            if (r.get("scale_check") or {}).get("adjudication") == "gps_length_supports_fused":
                return "fused", "gps_length_supports_fused"
            return "none", "scale_contradicted"
        return "fused", {"confirmed": "gps_confirmed_by_depth", "ambiguous": "gps_ambiguous_vs_depth"}.get(v, "gps_unchecked")
    if q["route"] == "needs_metric_scale":
        if m.get("source") == "metric_depth":
            return "metric", q.get("scale_reason") or "needs_metric_scale"
        return "none", m.get("reason") or m.get("status") or "no_scale"
    return "none", q.get("slam_reason") or q["route"]


def metric_depth_entry(r):
    if r.get("ok"):
        return {"source": "metric_depth", "scale": r["scale"], "scale_p25": r["scale_p25"],
                "scale_p75": r["scale_p75"], "scale_rel_iqr": r["scale_rel_iqr"],
                "n_frames_used": r["n_frames_used"], "model": "depth-anything/DA3METRIC-LARGE"}
    err = (r.get("error") or "").strip().splitlines()
    text = err[-1][:200] if err else ""
    reason = ("no_calibration" if "no_calibration" in text else
              "too_few_usable_frames" if "usable frames" in text else "estimation_failed")
    return {"source": None, "scale": None, "status": "failed", "reason": reason, "error": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=f"{TRAJ_ROOT}/manifest.jsonl")
    ap.add_argument("--traj_root", default=TRAJ_ROOT)
    ap.add_argument("--metric_manifest", default=METRIC_MANIFEST)
    ap.add_argument("--integration_log", default=INTEGRATION_LOG,
                    help="integrate_metric_scale.py's log; clips it skipped as 'mismatch' get no metric scale")
    ap.add_argument("--qc_csv", default=f"{PIPELINE_DIR}/clip_qc.csv", help="clip_qc.py output: adds a `qc` entry to each row")
    ap.add_argument("--out", default=None, help="Write here instead of rewriting --manifest in place")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit_sessions", type=int, default=None, help="Testing: only compute GPS scales for the first N sessions (others get scale null)")
    ap.add_argument("--dry_run", action="store_true", help="Compute and print a summary; write nothing")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    metric = latest_by_clip(args.metric_manifest)
    qc = load_qc(args.qc_csv) if args.qc_csv else {}
    qc_cfg = {}
    if args.qc_csv and Path(str(args.qc_csv) + ".config.json").exists():
        qc_cfg = json.load(open(str(args.qc_csv) + ".config.json"))
    integ = integration_status(args.integration_log)
    mismatched = {c for c, v in integ.items() if v["state"] == "mismatch"}
    print(f"{len(mismatched)} clips without a usable depth scale (rerun trajectory differs from the stored one), "
          f"{sum(1 for v in integ.values() if v.get('transferred'))} clips with a scale transferred from the rerun")

    by_session = defaultdict(list)
    for r in rows:
        status = (r.get("gps_fusion") or {}).get("status")
        if r.get("ok") and status in FUSED_STATUSES:
            session, inner = split_clip_id(r["clip_id"])
            if session:
                by_session[session].append((r["clip_id"], inner))
    sessions = sorted(by_session)[:args.limit_sessions]
    print(f"{len(rows)} rows; computing GPS scale for {sum(len(by_session[s]) for s in sessions)} clips "
          f"in {len(sessions)} session zips...", flush=True)
    gps_scale = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(gps_scales_for_session,
                                         [(s, by_session[s], args.traj_root) for s in sessions], chunksize=8), 1):
            gps_scale.update(res)
            if i % 2000 == 0:
                print(f"  {i}/{len(sessions)} sessions", flush=True)

    summary = Counter()
    for r in rows:
        cid = r["clip_id"]
        r.pop("scale_check", None)  # recomputed below for needs_scale_check rows; never keep a stale one
        route = (qc.get(cid) or {}).get("route")
        if route == "needs_scale_check":
            gps = gps_scale.get(cid)
            r["scale_check"] = chk = scale_check(gps, metric.get(cid), integ.get(cid))
            adj = adjudicate(chk, qc.get(cid) or {})
            if adj:
                chk["adjudication"] = adj
            entry = ({"source": "gps_fusion", "scale": gps} if gps is not None else {"source": None, "scale": None})
            if chk["verdict"] == "contradicted" and not adj:
                entry = {"source": None, "scale": None, "status": "scale_contradicted", "reason": "scale_contradicted",
                         "gps_fusion_scale": gps, "depth_scale": chk["depth_scale"]}
        elif cid in mismatched:
            entry = {"source": None, "scale": None, "status": "failed", "reason": "rerun_poses_mismatch",
                     "error": "rerun_poses_mismatch: SLAM rerun for the depth-based scale did not reproduce the stored poses"}
            if gps_scale.get(cid) is not None:
                entry.update(source="gps_fusion", scale=gps_scale[cid], status=None, error=None)
                entry = {k: v for k, v in entry.items() if v is not None or k in ("source", "scale")}
        elif cid in metric:
            # a clip deliberately routed through estimate_metric_scale.py: its metric-depth scale
            # wins; the GPS-fit scale (only exists for clips that still have fused_poses, e.g.
            # no_route_match) is kept next to it for comparison
            entry = metric_depth_entry(metric[cid])
            tr = (integ.get(cid) or {}).get("transferred")
            if tr and entry["source"] == "metric_depth":
                # scale was computed on a SLAM rerun whose gauge differed from the stored run: use
                # the value transferred to the stored poses (what is in result.npz)
                entry.update(scale=tr["scale"], scale_p25=tr["scale_p25"], scale_p75=tr["scale_p75"],
                             scale_transferred=True, transfer_residual=tr["resid_max_over_pathlen"])
            if gps_scale.get(cid) is not None:
                entry["gps_fusion_scale"] = gps_scale[cid]
        elif cid in gps_scale and gps_scale[cid] is not None:
            entry = {"source": "gps_fusion", "scale": gps_scale[cid]}
        else:
            # no GPS-fused scale and never sent to the depth step (e.g. SLAM failed): say so explicitly
            entry = {"source": None, "scale": None, "status": "not_attempted",
                     "reason": (qc.get(cid) or {}).get("route") or "no_scale_available"}
        if qc.get(cid) is not None:
            r["qc"] = dict(qc[cid], version=qc_cfg.get("qc_version"))
            if qc[cid]["route"] == "needs_metric_scale" and entry["source"] == "gps_fusion":
                entry = {"source": None, "scale": None, "status": "gps_scale_rejected_by_qc",
                         "reason": "gps_scale_rejected_by_qc", "gps_fusion_scale": entry["scale"]}
        r["metric_scale"] = entry
        if "qc" in r:
            r["pose_source"], r["pose_source_reason"] = pose_source(r)
        summary[entry["source"] or ("not_attempted" if entry.get("status") == "not_attempted" else
                                    "failed" if entry.get("status") else "none")] += 1
    rows = [normalize_row(r) for r in rows]
    print("metric_scale source counts:", dict(summary))
    if args.dry_run:
        return

    out_path = Path(args.out or args.manifest)
    if args.out is None:
        # a timestamped backup of whatever is being replaced, every time (never overwritten)
        import datetime
        bak = Path(f"{args.manifest}.bak_{datetime.datetime.now():%Y%m%dT%H%M%S}")
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
    if args.out is None and qc_cfg:
        # one provenance file next to the manifest: how its qc / metric_scale / pose_source fields were made
        import datetime
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        git = lambda *a: subprocess.run(["git", "-C", here, *a], capture_output=True, text=True).stdout.strip()
        cfg = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), manifest="manifest.jsonl",
                   rows=len(rows), qc=qc_cfg,
                   depth_scale=dict(script="scripts/estimate_metric_scale.py", model="depth-anything/DA3METRIC-LARGE",
                                    n_frames_requested=50, pcd_stride=3, process_res=504,
                                    agree_band=AGREE, contradicted_outside=CLEAR,
                                    confident_if=dict(min_frames=CONFIDENT_MIN_FRAMES, max_rel_iqr=CONFIDENT_MAX_REL_IQR),
                                    verdict_rules=dict(
                                        high_confidence="judged on the median depth/GPS ratio",
                                        low_confidence="contradicted only if the whole IQR of per-frame scales is outside the clear band; confirmed if the median is in the agree band; else ambiguous",
                                        adjudication="a contradiction on a map_match_error_high clip is overruled when fused/GPS path length is within "
                                                     f"[{SCALE_CHECK_LOW}, {SCALE_CHECK_HIGH}] (pose_source fused, reason gps_length_supports_fused)")),
                   pose_source_counts=dict(Counter(r["pose_source"] for r in rows if "pose_source" in r)),
                   metric_scale_source_counts=dict(summary),
                   manifest_script_git_commit=git("rev-parse", "HEAD"),
                   notes="see README.md, section 'Which poses to use (downstream)', and docs/manifest-schema.md")
        with open(out_path.with_name("qc_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"wrote {out_path.with_name('qc_config.json')}")


if __name__ == "__main__":
    main()
