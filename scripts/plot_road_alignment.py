#!/usr/bin/env python3
"""Plot the raw GPS track, the final map-matched OSM route, and the final
vggt_slam.gps_fusion.align_trajectory_to_roads-aligned predicted trajectory, all on top of the
OSM road network.

Reads main.py's own saved GPS-fusion result (a clip's consolidated result.npz -- see
vggt_slam.slam_utils.save_result_npz/load_result_npz) rather than re-running
align_trajectory_to_roads -- so, unlike before, this script never needs to shell out to
`map_matching_2` and only needs `shapely` (to parse the saved route, and to parse OSM ways for
the gray road backdrop below via vggt_slam.map_match.build_network, same as main.py's own
alignment run and reusing its cached --network_dir if pointed at the same one) and `pyproj`
(already required by vggt_slam.gps for the raw GPS log's UTM projection, and reused here for the
network fetch's lat/lon center).

Usage:
    scripts/plot_road_alignment.py --result_npz <dir>/result.npz \\
        --gps_csv <FRONT_*.csv> --network_dir <dir>/osm_network --out <dir>/road_alignment.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely import wkt as shapely_wkt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vggt_slam import gps as gps_module
from vggt_slam.map_match import build_network
from vggt_slam.slam_utils import load_result_npz

# natix raw sensor logs for the clips this script has been used on so far -- not all publicly
# discoverable from the processed run directories under /capstor/scratch, so recorded here once
# rather than re-tracked down each time.
NATIX_GPS_CSV = {
    "france": "/capstor/store/cscs/swissai/ab036/datasets/natix/France/2025-09-10/"
              "2025-09-10_40ba3633-6c77-4073-b8e4-6150a08f60f9/17-04-42/"
              "FRONT_FOLDER/FRONT_2025-09-10_17-04-42.csv",
    "latvia": "/capstor/store/cscs/swissai/ab036/datasets/natix/Latvia/2025-08-22/"
              "2025-08-22_76009c1b-fdc3-4006-8f53-369c7fff7ead_1/13-35-00/"
              "FRONT_FOLDER/FRONT_2025-08-22_13-35-00.csv",
    "nyc": "/capstor/store/cscs/swissai/ab036/datasets/natix/United States/New York/2025-08-28/"
           "2025-08-28_6b1f6e7a-bad1-49d0-a008-bcd044d94833_seq02/01-06-18/"
           "FRONT_FOLDER/FRONT_2025-08-28_01-06-18.csv",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result_npz", required=True, help="main.py's result.npz")
    ap.add_argument("--gps_csv", help="Raw natix FRONT_*.csv GPS log. If omitted, --clip must "
                                       "name one of the clips recorded in NATIX_GPS_CSV above")
    ap.add_argument("--clip", choices=sorted(NATIX_GPS_CSV), help="Shortcut for --gps_csv")
    ap.add_argument("--network_dir", required=True, help="OSM network cache dir for the plotted "
                                                           "gray road backdrop -- reused if it "
                                                           "already holds a network from main.py's "
                                                           "own align_trajectory_to_roads run")
    ap.add_argument("--osm_region_pbf_path", required=True,
                     help="Pre-downloaded whole-region OSM extract for this clip's country/state "
                          "(see scripts/download_osm_extracts.py and batch_infer.py's "
                          "--osm_extracts_dir) -- build_network requires this now (no more live "
                          "OSM fetch), but --network_dir being already populated by main.py's own "
                          "run means it's normally just cache-hit here, not actually re-cropped")
    ap.add_argument("--map_buffer_margin_m", type=float, default=150.0)
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--pairs_out", help="Output PNG path for the per-point residual-pairing plot "
                                         "(final predicted point <-> its snap onto the matched "
                                         "route). Defaults to '<out>_pairs.png'")
    ap.add_argument("--zoom_margin_m", type=float, default=15.0)
    args = ap.parse_args()

    gps_csv = args.gps_csv or NATIX_GPS_CSV.get(args.clip)
    if gps_csv is None:
        ap.error("pass --gps_csv, or --clip <" + "|".join(sorted(NATIX_GPS_CSV)) + ">")

    result = load_result_npz(args.result_npz)
    if result["fused_translations"] is None or result["matched_route_wkt"] is None:
        ap.error(f"{args.result_npz} has no fused trajectory / matched route -- this script "
                 "only plots a clip whose GPS fusion actually succeeded")
    traj_en = result["fused_translations"][:, [0, 2]]
    gps = gps_module.load_gps_fixes(gps_csv)

    matched_line = shapely_wkt.loads(result["matched_route_wkt"])
    snapped = result["snapped"]

    ok = ~np.isnan(snapped[:, 0])
    if ok.any():
        dev = np.linalg.norm(traj_en[ok] - snapped[ok], axis=1)
        print(f"final map-matched route: {int(ok.sum())}/{len(ok)} points snapped, "
              f"mean deviation={dev.mean():.2f}m max={dev.max():.2f}m")

    # OSM network fetch for the gray backdrop only (see module docstring) -- same buffer
    # computation as align_trajectory_to_roads's own loop (vggt_slam/gps_fusion.py), so a
    # --network_dir already populated by main.py's run is reused as-is with no re-fetch.
    center_en = traj_en.mean(axis=0)
    to_wgs84 = Transformer.from_crs(gps["utm_crs"], "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", gps["utm_crs"], always_xy=True)
    center_lon, center_lat = to_wgs84.transform(*center_en)
    map_buffer_m = float(np.linalg.norm(traj_en - center_en, axis=1).max()) + args.map_buffer_margin_m
    _, edges = build_network(center_lat, center_lon, map_buffer_m, to_utm, args.network_dir,
                              args.osm_region_pbf_path)

    # --- Plot: OSM network (gray) + raw GPS (green) + map-matched route (blue) +
    # final predicted/aligned trajectory (red dashed) ---
    gps_en = np.stack([gps["easting"], gps["northing"]], axis=1)
    all_pts = np.concatenate([gps_en, traj_en], axis=0)
    (xmin, ymin), (xmax, ymax) = all_pts.min(axis=0), all_pts.max(axis=0)
    m = args.zoom_margin_m
    xmin, xmax, ymin, ymax = xmin - m, xmax + m, ymin - m, ymax + m
    w, h = xmax - xmin, ymax - ymin
    max_dim = 9.0
    figsize = (max_dim, max_dim * h / w) if w >= h else (max_dim * w / h, max_dim)
    figsize = (max(figsize[0], 5.0), max(figsize[1], 5.0) + 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    for edge in edges:
        ex, ey = edge.xy
        ax.plot(ex, ey, color="#999999", linewidth=0.8, zorder=1)
    ax.plot(gps_en[:, 0], gps_en[:, 1], "-o", color="#33aa33", linewidth=1.5, markersize=3,
            label="raw GPS", zorder=3)
    if matched_line is not None:
        mx, my = matched_line.xy
        ax.plot(mx, my, "-", color="#3366cc", linewidth=2.5, label="map-matched route", zorder=4)
    ax.plot(traj_en[:, 0], traj_en[:, 1], "--", color="#cc3333", linewidth=2.0,
            label="final predicted trajectory", zorder=5)
    ax.scatter(*traj_en[0], color="black", s=60, zorder=6, label="start")
    ax.set_title("align_trajectory_to_roads: raw GPS, map-matched route, predicted trajectory")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")

    # --- Second plot: per-point residual pairing -- each final predicted trajectory point
    # connected to its own snapped position on the matched route, exactly the geometry dev above
    # is measured against. This is what actually shows whether a stretch of points is well-matched
    # (short, near-perpendicular pairing lines) or not (long/fanning lines, e.g. the
    # wrong-street-at-a-junction case from align_trajectory_to_roads's own docstring caveat).
    root, ext = os.path.splitext(args.out)
    pairs_out = args.pairs_out or f"{root}_pairs{ext}"
    fig2, ax2 = plt.subplots(figsize=figsize)
    if matched_line is not None:
        mx, my = matched_line.xy
        ax2.plot(mx, my, "-", color="#3366cc", linewidth=2.0, label="map-matched route", zorder=2)
    if ok.any():
        for (px, py), (sx, sy) in zip(traj_en[ok], snapped[ok]):
            ax2.plot([px, sx], [py, sy], "-", color="#ff8800", linewidth=0.8, alpha=0.6, zorder=1)
        ax2.scatter(snapped[ok, 0], snapped[ok, 1], s=10, color="#ff8800",
                    label="each point's own snap onto the matched route", zorder=4)
    ax2.plot(traj_en[:, 0], traj_en[:, 1], "-", color="#cc3333", linewidth=1.2, marker="o",
              markersize=3, label="final predicted trajectory", zorder=3)
    ax2.set_title("align_trajectory_to_roads: final residual pairing")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_aspect("equal")
    fig2.tight_layout()
    fig2.savefig(pairs_out, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {pairs_out}")


if __name__ == "__main__":
    main()
