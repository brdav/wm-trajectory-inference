#!/usr/bin/env python3
"""Does clip_qc.py's scale check miss more (false negatives) or reject more (false positives)?

Reference = the depth-based scale (DA3METRIC-LARGE vs the SLAM point cloud, ~+-8-10% in the 5 GPS
clips it was validated on). Per clip: r = depth_scale / gps_scale (gps_scale = fused/slam path-length
ratio). r ~ 1 means the two independent estimates agree. Neither estimate is ground truth, so a
disagreement says "one of them is wrong", not which.

  passed  = clips QC routed to use_fused: a sample stratified towards near-threshold cases
            (depth scale computed just for this check, outputs under qc_edge/metric/)
  rejected = clips QC sent to needs_metric_scale because of the scale check (implausible_speed,
            longer_than_gps, no_route_match): production depth scales are already in the manifest

Inputs (in $QC_VALIDATION_DIR): clip_list.txt (dataset-relative .mp4 + TAB + stratum label) and metric/<clip>/metric_scale.npz,
made with:  scripts/estimate_metric_scale.py --clip_list clip_list.txt --out_root $QC_VALIDATION_DIR/metric
Outputs (same dir): summary.csv, summary.txt, fig_strip.png, fig_gallery_*.png
"""
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/users/dbrggema/wm_trajectory_inference")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from integrate_metric_scale import align_scale  # noqa: E402
from vggt_slam import gps as gps_module  # noqa: E402

E = Path(os.environ.get("QC_VALIDATION_DIR", Path(__file__).parent))  # holds clip_list.txt, metric/ (scores), outputs
TRAJ = Path("/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories")
DATA = Path("/capstor/store/cscs/swissai/ab036/datasets/natix")
AGREE = (0.8, 1.25)   # depth/gps ratio counted as agreement (depth is only ~+-10%)
CLEAR = (1 / 1.5, 1.5)  # outside this: clear disagreement

# palette slots 1-3 (validated all-pairs) + muted ink, light surface
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8985"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9, "axes.titlesize": 9})


def load_clip(clip):
    rel = clip[:-4].split("/")
    session, inner = "/".join(rel[:-3]), "/".join(rel[-3:]) + "/result.npz"
    z = np.load(io.BytesIO(zipfile.ZipFile(TRAJ / (session + ".zip")).read(inner)))
    return z, rel


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    sc = (S * np.diag(D)).sum() / ((s0 ** 2).sum() / len(src))
    return sc, R, mu_d - sc * R @ mu_s


def analyse(clip, group, label, depth_scale=None, depth_info=None, ms_npz=None):
    """One row of facts + the paths to plot. depth scale comes from the standalone rerun output
    (ms_npz, transferred to the stored poses like integrate_metric_scale does) or is given."""
    z, rel = load_clip(clip)
    _, first = np.unique(z["frame_ids"], return_index=True)
    first = np.sort(first)
    slam, fused = z["slam_poses"][first][:, :3, 3], z["fused_poses"][first][:, :3, 3]
    ln = lambda P: np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    s_gps = ln(fused) / ln(slam)
    row = dict(clip=clip, group=group, stratum=label, s_gps=s_gps, n_keyframes=len(first))
    if ms_npz is not None:
        ms = np.load(ms_npz)
        mu, resid = 1.0, 0.0
        if not (np.array_equal(z["frame_ids"], ms["frame_ids"]) and
                np.allclose(z["slam_poses"], ms["slam_poses"], rtol=0, atol=1e-6)):
            al = align_scale(z["slam_poses"], ms["slam_poses"]) if np.array_equal(z["frame_ids"], ms["frame_ids"]) else None
            mu, resid = (al if al and al[1] <= 0.05 else (np.nan, np.nan))
        depth_scale = float(ms["scale"]) * mu
        st = json.loads(str(ms["stats_json"])) if "stats_json" in ms.files else {}
        depth_info = dict(n_frames_used=st.get("n_frames_used"), scale_rel_iqr=st.get("scale_rel_iqr"))
    row.update(s_depth=depth_scale, r=(depth_scale / s_gps) if depth_scale else np.nan, **(depth_info or {}))
    return row, dict(slam=slam, fused=fused, clip=clip, csv=DATA / ("/".join(rel) + ".csv"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def trajectory_panel(ax, row, path):
    slam, fused = path["slam"], path["fused"]
    sc, R, t = umeyama(slam, fused)
    # depth-scaled SLAM path, rotated like the fused one and started at the same point
    depth = row["s_depth"] * (R @ (slam - slam[0]).T).T + fused[0]
    o = fused[0, [0, 2]]  # plot in meters east/north of the clip's start
    try:
        g = gps_module.load_gps_fixes(str(path["csv"]))
        ax.plot(g["easting"] - o[0], g["northing"] - o[1], ".", color=MUTED, ms=3, label="raw GPS fixes", zorder=1)
    except Exception:
        pass
    ax.plot(fused[:, 0] - o[0], fused[:, 2] - o[1], "-", color=BLUE, lw=2, label="GPS-fused path", zorder=3)
    ax.plot(depth[:, 0] - o[0], depth[:, 2] - o[1], "-", color=ORANGE, lw=2, label="SLAM x depth scale", zorder=2)
    ax.plot(0, 0, "o", color=SURF, mec=INK, ms=7, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("east (m)", fontsize=7); ax.set_ylabel("north (m)", fontsize=7)
    q = QC.get(row["clip"], {})
    ax.set_title(f"{row['stratum']}  |  {Path(row['clip']).stem[6:]}\n"
                 f"r = {row['r']:.2f}   (depth {row['s_depth']:.1f} / GPS {row['s_gps']:.1f})\n"
                 f"QC: len ratio {q.get('fused_over_gps_len', float('nan')):.2f}, max {q.get('fused_max_window_mps', float('nan')):.0f} m/s",
                 fontsize=7, color=INK, loc="left")


def gallery(rows_paths, title, out):
    n = len(rows_paths)
    cols = 4
    nr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nr, cols, figsize=(14, 3.9 * nr), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (row, path) in zip(axes.ravel(), rows_paths):
        ax.axis("on")
        trajectory_panel(ax, row, path)
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.5)
    fig.savefig(out, dpi=130)
    plt.close(fig)


QC = {}


def main():
    import pandas as pd
    qc = pd.read_csv("/capstor/store/cscs/swissai/ab036/dbrggema/natix_metric_scale/clip_qc.csv")
    QC.update({r["clip"]: r for r in qc.to_dict("records")})
    # ---- passed sample (standalone depth scales) ----
    lab = dict(line.rstrip("\n").split("\t") for line in open(E / "clip_list.txt"))
    rows, paths = [], {}
    for clip, stratum in lab.items():
        ms = E / "metric" / clip[:-4] / "metric_scale.npz"
        if not ms.exists():
            continue
        try:
            row, p = analyse(clip, "passed", stratum, ms_npz=ms)
        except Exception as e:  # keep going
            print("skip", clip, repr(e)[:80]); continue
        rows.append(row); paths[clip] = p
    # ---- rejected by the scale check (production depth scales, already integrated) ----
    man = [json.loads(l) for l in open(TRAJ / "manifest.jsonl")]
    for r in man:
        q, m = r["qc"], r["metric_scale"]
        rej = q["route"] == "needs_metric_scale" and (q["scale_reason"] in ("implausible_speed", "longer_than_gps", "no_route_match"))
        if not rej or m.get("source") != "metric_depth" or not m.get("gps_fusion_scale"):
            continue
        clip = r["clip_id"][len(str(DATA)) + 1:]
        try:
            row, p = analyse(clip, "rejected", q["scale_reason"], depth_scale=m["scale"],
                             depth_info=dict(n_frames_used=m["n_frames_used"], scale_rel_iqr=m["scale_rel_iqr"]))
        except Exception as e:
            print("skip", clip, repr(e)[:80]); continue
        rows.append(row); paths[clip] = p
    df = pd.DataFrame(rows)
    df["verdict"] = np.where(df.r.isna(), "no depth scale",
                     np.where(df.r.between(*AGREE), "agree",
                     np.where(df.r.between(*CLEAR), "ambiguous", "clear disagreement")))
    df.to_csv(E / "summary.csv", index=False)

    # ---- numbers ----
    out = []
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    ok = df[df.r.notna()]
    P(f"clips analysed: {len(df)} (with a usable depth scale: {len(ok)})  agree band {AGREE}, clear disagreement outside {tuple(round(x,2) for x in CLEAR)}\n")
    P("per group / stratum:  n | median r | agree | ambiguous | clear disagreement")
    for (g, s), d in ok.groupby(["group", "stratum"], sort=False):
        c = d.verdict.value_counts()
        P(f"  {g:8s} {s:20s} n={len(d):3d} median r={d.r.median():.2f}  agree {c.get('agree',0):3d}  ambiguous {c.get('ambiguous',0):3d}  clear {c.get('clear disagreement',0):3d}")
    conf = ok[(ok.n_frames_used.fillna(0) >= 5) & (ok.scale_rel_iqr.fillna(9) <= 0.25)]
    P(f"\nconfident depth scales only (>=5 frames, rel_iqr<=0.25): {len(conf)} of {len(ok)}")
    pas = conf[conf.group == "passed"]; rej = conf[conf.group == "rejected"]
    rnd = pas[pas.stratum == ("random2" if (pas.stratum == "random2").any() else "random")]
    k, n = int((rnd.verdict == "clear disagreement").sum()), len(rnd)
    lo, hi = wilson(k, n)
    # false-negative extrapolation uses the sample drawn from the CURRENT unflagged use_fused pool ("random2" if
    # present, else the legacy "random" stratum, drawn before the needs_scale_check tier existed)
    qc = pd.read_csv("/capstor/store/cscs/swissai/ab036/dbrggema/natix_metric_scale/clip_qc.csv", usecols=["route"])
    NP = int((qc.route == "use_fused").sum())
    P(f"   robust sigma of log(depth/GPS) in that sample: {np.median(np.abs(np.log(rnd.r) - np.median(np.log(rnd.r)))) * 1.4826:.3f}; median r {rnd.r.median():.3f}")
    P(f"\nFALSE NEGATIVES (passed, depth clearly disagrees) -- random sample of the unflagged use_fused pool: {k}/{n} = {100*k/max(n,1):.1f}% "
      f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)  -> ~{k/max(n,1)*NP:,.0f} of {NP:,} passed clips (CI {lo*NP:,.0f}-{hi*NP:,.0f})")
    kk = int((rnd.verdict == "ambiguous").sum())
    P(f"   plus ambiguous (r outside 0.8-1.25 but inside 0.67-1.5): {kk}/{n}")
    edge = pas[~pas.stratum.isin(["random", "random2"])]
    P(f"   edge-case strata: {int((edge.verdict=='clear disagreement').sum())}/{len(edge)} clear disagreement")
    k2, n2 = int((rej.verdict == 'agree').sum()), len(rej)
    lo2, hi2 = wilson(k2, n2)
    P(f"\nFALSE POSITIVES (rejected, depth agrees with the GPS scale): {k2}/{n2} = {100*k2/max(n2,1):.1f}% "
      f"(95% CI {100*lo2:.1f}-{100*hi2:.1f}%)  -> ~{k2} clips of the {n2} rejected-with-confident-scale")
    P(f"   rejected & depth clearly disagrees (QC right): {int((rej.verdict=='clear disagreement').sum())}/{n2}")
    P("\ncaveat: depth scale is itself +-10% and not ground truth; 'clear disagreement' = factor > 1.5, well outside that.")
    (E / "summary.txt").write_text("\n".join(out))
    print("\n".join(out))

    # ---- figure 1: strip chart of r per group/stratum ----
    order = [("passed", "random2"), ("passed", "random"), ("passed", "low_len_ratio<0.8"), ("passed", "len_ratio_1.5-2"), ("passed", "speed_30-50mps"),
             ("passed", "no_gps_len_ratio"), ("passed", "slam_step_25-50"),
             ("rejected", "implausible_speed"), ("rejected", "longer_than_gps"), ("rejected", "no_route_match")]
    order = [o for o in order if ((ok.group == o[0]) & (ok.stratum == o[1])).any()]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.8))
    ax.axvspan(*AGREE, color=AQUA, alpha=0.13, lw=0)
    rng = np.random.default_rng(1)
    for i, (g, s) in enumerate(order):
        d = ok[(ok.group == g) & (ok.stratum == s)]
        ax.plot(d.r, i + rng.uniform(-0.22, 0.22, len(d)), "o", ms=4.5, color=BLUE if g == "passed" else ORANGE,
                mec=SURF, mew=0.6, alpha=0.9)
        ax.text(1.02, i, f"n={len(d)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.axvline(1, color=INK2, lw=0.8)
    ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{g}: {s}" for g, s in order]); ax.invert_yaxis()
    ax.set_xlabel("depth-based scale / GPS-fused scale\n1 = agree;  shaded band = 0.8 to 1.25;  < 0.67 or > 1.5 = clear disagreement")
    ax.grid(True, axis="x", lw=0.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title("QC verdict vs. an independent scale estimate  (blue = QC passed, orange = QC rejected)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(E / "fig_strip.png", dpi=140); plt.close(fig)

    # ---- galleries ----
    def pick(d, n, key, asc=True):
        d = d.sort_values(key, ascending=asc).head(n)
        return [(r, paths[r["clip"]]) for _, r in d.iterrows()]
    ok = ok.assign(logr=np.abs(np.log(ok.r)))
    pas_ok = ok[ok.group == "passed"]
    gallery(pick(pas_ok, 12, "logr", False), "QC PASSED, depth scale disagrees most (candidate false negatives)", E / "fig_gallery_passed_disagree.png")
    ag = pas_ok[(pas_ok.verdict == "agree") & (pas_ok.stratum != "random")]
    gallery([(r, paths[r["clip"]]) for _, r in ag.sample(min(8, len(ag)), random_state=0).iterrows()],
            "QC PASSED edge cases where depth agrees (QC was right to pass)", E / "fig_gallery_passed_agree.png")
    rj = ok[ok.group == "rejected"]
    rja = rj[rj.verdict == "agree"]
    gallery([(r, paths[r["clip"]]) for _, r in rja.sample(min(8, len(rja)), random_state=0).iterrows()],
            "QC REJECTED but depth agrees with the GPS scale (candidate false positives)", E / "fig_gallery_rejected_agree.png")
    gallery(pick(rj[rj.verdict == "clear disagreement"], 8, "logr", False),
            "QC REJECTED and depth clearly disagrees (QC right)", E / "fig_gallery_rejected_disagree.png")


if __name__ == "__main__":
    main()
