#!/usr/bin/env python3
"""Reads a clip's consolidated result.npz (see vggt_slam/slam_utils.py's save_result_npz).

Single clip:
    scripts/plot_road_alignment_standalone.py --result_npz <dir>/result.npz \\
        --gps_csv <FRONT_*.csv> --network_dir <dir>/osm_network --out <dir>/road_alignment.png

Random sample of already-completed clips, for spot-checking fit quality at dataset scale --
reads batch_infer.py's manifest.jsonl to find clips whose GPS fusion actually succeeded (skips
ones that failed or never matched a route), so every sampled clip is guaranteed to have a
result.npz with a fused trajectory + matched route:
    scripts/plot_road_alignment_standalone.py --sample 5 \\
        --manifest natix_traj/manifest.jsonl --dataset_root .../datasets/natix \\
        --out_root natix_traj --out_dir /path/to/qa_plots [--seed 0]
"""
import argparse
import csv
import json
import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString


def _utm_crs_for(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def load_gps_fixes(csv_path):
    """Parse a natix FRONT_*.csv sensor log into real (non-'na') GPS fixes, projected to a local
    UTM CRS. Returns a dict with elapsed_s, easting, northing, accuracy_m, utm_crs.

    Mirrors vggt_slam/gps.py's own load_gps_fixes (kept in sync by hand -- this script is
    deliberately standalone/no vggt_slam import, see module docstring) -- in particular, elapsed
    time is measured from each row's own `frame_number` column and a native-fps estimate, NOT
    from the CSV's own first-row timestamp: those two only agree if sensor logging happened to
    start at the video's own first frame, which isn't always true (see vggt_slam/gps.py's
    docstring for the concrete clip that motivated this fix). Falls back to the old
    first-row-timestamp behavior if `frame_number` isn't present (older/other log format).
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    t0 = datetime.fromisoformat(rows[0]["timestamp"])
    has_frame_number = "frame_number" in rows[0]

    if has_frame_number:
        wall_s = np.array([(datetime.fromisoformat(r["timestamp"]) - t0).total_seconds() for r in rows])
        frame_number = np.array([int(r["frame_number"]) for r in rows], dtype=float)
        native_fps, _ = np.polyfit(wall_s, frame_number, deg=1)

    elapsed, lats, lons, accs = [], [], [], []
    for row in rows:
        if row["GPS_latitude_deg"] == "na":
            continue
        if has_frame_number:
            t_s = (int(row["frame_number"]) - 1) / native_fps
        else:
            t_s = (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()
        elapsed.append(t_s)
        lats.append(float(row["GPS_latitude_deg"]))
        lons.append(float(row["GPS_longitude_deg"]))
        accs.append(float(row["horizontal_accuracy_m"]) if row["horizontal_accuracy_m"] != "na" else 10.0)

    if not elapsed:
        raise ValueError(f"No real GPS fixes (all 'na') in {csv_path}")

    lats, lons = np.array(lats), np.array(lons)
    utm_crs = _utm_crs_for(float(lons.mean()), float(lats.mean()))
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    easting, northing = to_utm.transform(lons, lats)

    return dict(
        elapsed_s=np.asarray(elapsed, dtype=float),
        easting=np.asarray(easting, dtype=float),
        northing=np.asarray(northing, dtype=float),
        accuracy_m=np.asarray(accs, dtype=float),
        utm_crs=utm_crs,
    )


def load_osm_edges(osm_path, to_utm):
    """Parses network_dir/network.osm (OSM XML, written by main.py's own GPS-fusion run) into a
    list of shapely LineStrings (UTM meters) for the plotted gray road-network backdrop -- one
    per `highway` way."""
    tree = ET.parse(osm_path)
    root = tree.getroot()
    nodes = {}
    for node in root.iter("node"):
        nodes[node.get("id")] = (float(node.get("lon")), float(node.get("lat")))

    edges = []
    for way in root.iter("way"):
        if not any(tag.get("k") == "highway" for tag in way.iter("tag")):
            continue
        refs = [nd.get("ref") for nd in way.iter("nd")]
        lonlat = [nodes[r] for r in refs if r in nodes]
        if len(lonlat) < 2:
            continue
        lons, lats = zip(*lonlat)
        eastings, northings = to_utm.transform(lons, lats)
        edges.append(LineString(zip(eastings, northings)))
    return edges


def load_result_npz(path):
    """Standalone equivalent of vggt_slam.slam_utils.load_result_npz -- duplicated by hand
    (like load_gps_fixes above) rather than imported, since importing vggt_slam.slam_utils would
    pull in torch/torchvision/scipy at module level just for this, defeating the point of this
    script being usable without the full ML stack. Only the fields this script needs."""
    data = np.load(path)
    out = {"fused_translations": None, "matched_route_wkt": None, "snapped": None}
    if "fused_poses" in data:
        out["fused_translations"] = data["fused_poses"][:, :3, 3]
    if "matched_route_wkt" in data:
        out["matched_route_wkt"] = str(data["matched_route_wkt"])
    if "snapped" in data:
        out["snapped"] = data["snapped"]
    return out


def find_completed_clips(manifest_path):
    """Streams manifest.jsonl (written by scripts/batch_infer.py, one JSON object per processed
    clip) and yields each clip_id (source video path) whose GPS fusion actually produced a route
    match -- i.e. its result.npz has fused_poses/matched_route_wkt/snapped fields.
    Excludes clips that failed outright, that had no GPS fusion requested, or whose fusion ran
    but found no road-matched route (gps_fusion.status in {"failed", "no_roads_in_any_region",
    "no_route_match"} -- see main.py's own gps_fusion_info handling) -- this script would have
    nothing to plot for those anyway.
    """
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            gps_fusion = rec.get("gps_fusion")
            if rec.get("ok") and isinstance(gps_fusion, dict) and gps_fusion.get("status") == "ok":
                yield rec["clip_id"]


def clip_paths(video_path, dataset_root, out_root):
    """Derives every artifact path this script needs for one clip, mirroring
    scripts/batch_infer.py's own out_dir layout (out_root/<video's path relative to
    dataset_root, minus .mp4>) and main.py's result.npz/osm_network naming.
    """
    video_path = Path(video_path)
    rel = video_path.relative_to(Path(dataset_root))
    out_dir = Path(out_root) / rel.with_suffix("")
    return dict(
        result_npz=out_dir / "result.npz",
        network_dir=out_dir / "osm_network",
        gps_csv=video_path.with_suffix(".csv"),
        name="_".join(rel.with_suffix("").parts),
    )


def plot_clip(result_npz, gps_csv, network_dir, out,
              pairs_out=None, map_buffer_margin_m=150.0, zoom_margin_m=15.0):
    """Core single-clip plotting logic: writes `out` (raw GPS / map-matched route / final fused
    trajectory over the OSM road backdrop) and `pairs_out` (per-point residual pairing). Returns
    the (mean_deviation_m, max_deviation_m) tuple over snapped points, or None if none snapped."""
    osm_path = os.path.join(network_dir, "network.osm")
    if not os.path.exists(osm_path):
        raise FileNotFoundError(
            f"{osm_path} doesn't exist -- this script only plots a clip whose GPS fusion has "
            f"already run (main.py writes network.osm as part of that, and keeps it even after "
            f"trimming the rest of osm_network/ -- see vggt_slam/map_match.py's "
            f"trim_osm_network_cache); it can't fetch or build an OSM network from scratch (see "
            f"module docstring)")

    result = load_result_npz(result_npz)
    if result["fused_translations"] is None or result["matched_route_wkt"] is None:
        raise ValueError(
            f"{result_npz} has no fused trajectory / matched route -- this script only plots a "
            f"clip whose GPS fusion actually succeeded (gps_fusion.status == 'ok')")
    traj_en = result["fused_translations"][:, [0, 2]]
    gps = load_gps_fixes(gps_csv)

    matched_line = shapely_wkt.loads(result["matched_route_wkt"])
    snapped = result["snapped"]

    ok = ~np.isnan(snapped[:, 0])
    dev_stats = None
    if ok.any():
        dev = np.linalg.norm(traj_en[ok] - snapped[ok], axis=1)
        dev_stats = (float(dev.mean()), float(dev.max()))
        print(f"final map-matched route: {int(ok.sum())}/{len(ok)} points snapped, "
              f"mean deviation={dev.mean():.2f}m max={dev.max():.2f}m")

    # Gray road-network backdrop only -- see load_osm_edges. No live fetch, no re-crop: just
    # parse whatever main.py's own run already cached at network_dir/network.osm.
    to_utm = Transformer.from_crs("EPSG:4326", gps["utm_crs"], always_xy=True)
    edges = load_osm_edges(osm_path, to_utm)

    # --- Plot: OSM network (gray) + raw GPS (green) + map-matched route (blue) +
    # final predicted/aligned trajectory (red dashed) ---
    gps_en = np.stack([gps["easting"], gps["northing"]], axis=1)
    all_pts = np.concatenate([gps_en, traj_en], axis=0)
    (xmin, ymin), (xmax, ymax) = all_pts.min(axis=0), all_pts.max(axis=0)
    m = zoom_margin_m
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
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # --- Second plot: per-point residual pairing -- each final predicted trajectory point
    # connected to its own snapped position on the matched route, exactly the geometry dev above
    # is measured against. This is what actually shows whether a stretch of points is well-matched
    # (short, near-perpendicular pairing lines) or not (long/fanning lines, e.g. the
    # wrong-street-at-a-junction case from align_trajectory_to_roads's own docstring caveat).
    root, ext = os.path.splitext(out)
    pairs_out = pairs_out or f"{root}_pairs{ext}"
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
    return dev_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int,
                     help="Instead of plotting one explicitly-named clip, randomly sample this "
                          "many already-completed clips (successful GPS fusion, per --manifest) "
                          "and plot each into --out_dir. Requires --manifest, --dataset_root, "
                          "--out_root, and --out_dir; mutually exclusive with --fused_poses_npz "
                          "et al below.")
    ap.add_argument("--manifest", help="--sample mode: scripts/batch_infer.py's manifest.jsonl")
    ap.add_argument("--dataset_root", help="--sample mode: dataset root manifest.jsonl's "
                                            "clip_ids are absolute paths under")
    ap.add_argument("--out_root", help="--sample mode: batch_infer.py's --out_root (where each "
                                        "clip's result.npz lives)")
    ap.add_argument("--out_dir", help="--sample mode: directory to write the sampled clips' "
                                       "PNGs into, one pair per clip")
    ap.add_argument("--seed", type=int, help="--sample mode: random seed, for a reproducible "
                                              "sample. Default: unseeded (a fresh sample "
                                              "each run)")

    ap.add_argument("--result_npz", help="Single-clip mode: main.py's result.npz")
    ap.add_argument("--gps_csv", help="Single-clip mode: raw natix FRONT_*.csv GPS log")
    ap.add_argument("--network_dir", help="Single-clip mode: dir holding network.osm, already "
                                           "written by main.py's own align_trajectory_to_roads "
                                           "run for this clip (this script does not know how to "
                                           "build one from scratch -- see module docstring)")
    ap.add_argument("--out", help="Single-clip mode: output PNG path")
    ap.add_argument("--pairs_out", help="Single-clip mode: output PNG path for the per-point "
                                         "residual-pairing plot. Defaults to '<out>_pairs.png'")
    ap.add_argument("--map_buffer_margin_m", type=float, default=150.0)
    ap.add_argument("--zoom_margin_m", type=float, default=15.0)
    args = ap.parse_args()

    if args.sample:
        missing = [name for name in ("manifest", "dataset_root", "out_root", "out_dir")
                   if getattr(args, name) is None]
        if missing:
            ap.error(f"--sample requires {', '.join('--' + n for n in missing)}")

        completed = list(find_completed_clips(args.manifest))
        if not completed:
            ap.error(f"no clips with a successful GPS-fusion route match found in {args.manifest}")
        rng = random.Random(args.seed)
        chosen = rng.sample(completed, min(args.sample, len(completed)))
        print(f"Sampling {len(chosen)}/{len(completed)} completed clips...")

        os.makedirs(args.out_dir, exist_ok=True)
        results = []
        for i, video_path in enumerate(chosen, 1):
            paths = clip_paths(video_path, args.dataset_root, args.out_root)
            print(f"=== [{i}/{len(chosen)}] {video_path} ===")
            try:
                dev_stats = plot_clip(
                    result_npz=paths["result_npz"], gps_csv=paths["gps_csv"],
                    network_dir=paths["network_dir"],
                    out=os.path.join(args.out_dir, f"{i}_{paths['name']}.png"),
                    map_buffer_margin_m=args.map_buffer_margin_m,
                    zoom_margin_m=args.zoom_margin_m,
                )
            except Exception as e:
                print(f"[WARN] skipping {video_path}: {e}")
                continue
            results.append((video_path, dev_stats))

        print(f"\nDone: {len(results)}/{len(chosen)} sampled clips plotted into {args.out_dir}")
        return

    required = ("result_npz", "gps_csv", "network_dir", "out")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        ap.error(f"pass --sample N, or all of {', '.join('--' + n for n in required)}")

    plot_clip(
        result_npz=args.result_npz, gps_csv=args.gps_csv, network_dir=args.network_dir,
        out=args.out, pairs_out=args.pairs_out,
        map_buffer_margin_m=args.map_buffer_margin_m, zoom_margin_m=args.zoom_margin_m,
    )


if __name__ == "__main__":
    main()
