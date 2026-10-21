#!/usr/bin/env python3
"""Placement check, independent of scale (also reports the cross-track distance, which ignores GPS-vs-video timing): how far is the fused path from the raw GPS fixes at the same
times, for clips QC passed (route use_fused)? Large residuals are either a badly placed trajectory or
bad GPS; the logged horizontal accuracy tells the two apart. Writes placement_residuals.csv,
placement_summary.txt and fig_placement_outliers.png next to this file.
"""
import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path("/users/dbrggema/wm_trajectory_inference")
sys.path.insert(0, str(REPO))
from vggt_slam import gps as G  # noqa: E402

E = Path(os.environ.get("QC_VALIDATION_DIR", Path(__file__).parent))
TRAJ = Path("/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories")
DATA = "/capstor/store/cscs/swissai/ab036/datasets/natix/"
BLUE, GREY, INK, INK2, GRID, SURF = "#2a78d6", "#8a8985", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "grid.color": GRID, "font.size": 9})


def clip_data(clip_id):
    rel = clip_id[len(DATA):-4].split("/")
    sess, inner = "/".join(rel[:-3]), "/".join(rel[-3:]) + "/result.npz"
    z = np.load(io.BytesIO(zipfile.ZipFile(TRAJ / (sess + ".zip")).read(inner)))
    t = (z["frame_ids"] - 1) / float(z["extraction_fps"])
    _, i = np.unique(t, return_index=True)
    i = np.sort(i)
    t, F = t[i], z["fused_poses"][i][:, :3, 3][:, [0, 2]]
    g = G.load_gps_fixes(clip_id[:-4] + ".csv")
    return t, F, g


def main():
    rows = [json.loads(l) for l in open(TRAJ / "manifest.jsonl")]
    pool = [x for x in rows if x["qc"]["route"] == "use_fused"]
    random.seed(11)
    out = []
    for x in random.sample(pool, 1500):
        try:
            t, F, g = clip_data(x["clip_id"])
            ins = (g["elapsed_s"] >= t[0]) & (g["elapsed_s"] <= t[-1])
            if ins.sum() < 5:
                continue
            fx = np.stack([np.interp(g["elapsed_s"][ins], t, F[:, 0]), np.interp(g["elapsed_s"][ins], t, F[:, 1])], 1)
            gp = np.stack([g["easting"][ins], g["northing"][ins]], 1)
            d = np.linalg.norm(fx - gp, axis=1)  # along + cross track (depends on time sync)
            seg = np.linspace(0, 1, 20)[None, :, None]
            dense = (F[:-1, None, :] * (1 - seg) + F[1:, None, :] * seg).reshape(-1, 2)
            ct = cKDTree(dense).query(gp)[0]  # distance to the fused polyline: ignores time offsets
            out.append(dict(clip=x["clip_id"][len(DATA):], resid_med_m=float(np.median(d)), resid_p90_m=float(np.percentile(d, 90)),
                            cross_med_m=float(np.median(ct)), cross_p90_m=float(np.percentile(ct, 90)),
                            speed_mps=float(np.linalg.norm(np.diff(F, axis=0), axis=1).sum() / max(t[-1] - t[0], 1e-6)),
                            gps_acc_med_m=float(np.median(g["accuracy_m"][ins])), n_fix=int(ins.sum()),
                            map_match_err_m=(x["gps_fusion"] or {}).get("map_match_error_mean_m")))
        except Exception:
            pass
    df = pd.DataFrame(out)
    df.to_csv(E / "placement_residuals.csv", index=False)
    big = df[df.resid_med_m > 50]
    lines = [f"clips analysed: {len(df)} (QC-passed, >=5 GPS fixes in range)",
             f"median residual p50/p90/p99: {np.round(df.resid_med_m.quantile([.5, .9, .99]).values, 1)} m",
             f"> 20 m: {int((df.resid_med_m > 20).sum())}   > 50 m: {len(big)}   > 100 m: {int((df.resid_med_m > 100).sum())}",
             f"median logged GPS accuracy (all clips): {df.gps_acc_med_m.median():.1f} m",
             f"outliers (>50 m): logged accuracy median {big.gps_acc_med_m.median():.1f} m; with accuracy > 25 m (GPS itself poor): "
             f"{int((big.gps_acc_med_m > 25).sum())} of {len(big)}; with accuracy <= 25 m: {int((big.gps_acc_med_m <= 25).sum())}",
             f"cross-track (time-offset-free) per-clip median p50/p90/p99: {np.round(df.cross_med_m.quantile([.5, .9, .99]).values, 1)} m; "
             f"> 20 m: {int((df.cross_med_m > 20).sum())}  > 50 m: {int((df.cross_med_m > 50).sum())}",
             f"implied GPS-vs-video time offset of the >50 m along-track outliers (residual / speed) p50/p90: "
             f"{np.round((big.resid_med_m / big.speed_mps.clip(lower=1)).quantile([.5, .9]).values, 1)} s",
             f"outliers with map-match error > 10 m: {int((big.map_match_err_m.fillna(0) > 10).sum())} of {len(big)}"]
    (E / "placement_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    sel = big.sample(min(12, len(big)), random_state=0)
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), sel.iterrows()):
        ax.axis("on")
        t, F, g = clip_data(DATA + r["clip"])
        o = F[0]
        ax.plot(g["easting"] - o[0], g["northing"] - o[1], ".", color=GREY, ms=3, label="raw GPS fixes")
        ax.plot(F[:, 0] - o[0], F[:, 1] - o[1], "-", color=BLUE, lw=2, label="GPS-fused path")
        ax.plot(0, 0, "o", color=SURF, mec=INK, ms=7)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, lw=0.5)
        ax.tick_params(labelsize=7); ax.set_xlabel("east (m)", fontsize=7); ax.set_ylabel("north (m)", fontsize=7)
        mm = r["map_match_err_m"]
        ax.set_title(f"{Path(r['clip']).stem[6:]}\nresidual {r['resid_med_m']:.0f} m (median) | GPS accuracy {r['gps_acc_med_m']:.0f} m\n"
                     f"map-match error {mm:.0f} m" if mm == mm else f"{Path(r['clip']).stem[6:]}\nresidual {r['resid_med_m']:.0f} m | acc {r['gps_acc_med_m']:.0f} m",
                     fontsize=7, loc="left")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("QC-passed clips whose fused path is > 50 m from the raw GPS fixes (random 12 of the outliers)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    fig.savefig(E / "fig_placement_outliers.png", dpi=130)


if __name__ == "__main__":
    main()
