#!/usr/bin/env python3
"""Samples a few successfully-fused clips from manifest.jsonl and renders their trajectory plots
(raw GPS / map-matched route / final fused trajectory over the OSM road backdrop) into a single
self-contained local HTML report -- each plot is embedded as a base64 PNG, so the resulting file
needs no external images/paths to view: copy it anywhere and open it in a browser.

Reuses plot_road_alignment_standalone.py's own clip-discovery and plotting logic unchanged
(imported, not duplicated) -- see that script for the single-clip/PNG-output equivalent of this.

Usage:
    scripts/plot_trajectory_report.py --sample 8 \\
        --manifest .../natix_traj/manifest.jsonl --dataset_root .../datasets/natix \\
        --out_root .../natix_traj --out /path/to/report.html [--seed 0]
"""
import argparse
import base64
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_road_alignment_standalone import clip_paths, find_completed_clips, plot_clip

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>GPS fusion trajectory QA report</title>
<style>
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; background: #fafafa; color: #222; }
h1 { font-size: 1.4rem; }
.clip { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; }
.clip h2 { font-size: 1.0rem; margin: 0 0 0.5rem 0; word-break: break-all; font-weight: 600; }
.meta { font-size: 0.85rem; color: #555; margin-bottom: 0.75rem; }
.meta code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
img { max-width: 100%; height: auto; border: 1px solid #eee; }
</style></head><body>
"""


def _load_manifest_records(manifest_path):
    """clip_id -> latest JSON record in manifest_path (append-only: a later line for the same
    clip_id overwrites an earlier one, same convention as the manifest itself)."""
    records = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "clip_id" in rec:
                records[rec["clip_id"]] = rec
    return records


def _clip_meta_html(rec):
    gf = rec.get("gps_fusion", {})
    bits = []
    if "map_match_error_mean_m" in gf:
        bits.append(f"map-match deviation: mean={gf['map_match_error_mean_m']:.2f}m "
                     f"max={gf.get('map_match_error_max_m', float('nan')):.2f}m")
    if gf.get("region_used"):
        region_bit = f"region: <code>{Path(gf['region_used']).name}</code>"
        if gf.get("used_fallback"):
            region_bit += " (fallback)"
        bits.append(region_bit)
    if "matched_keyframes" in gf:
        bits.append(f"matched keyframes: {gf['matched_keyframes']}/{gf.get('total_keyframes', '?')}")
    return " &middot; ".join(bits)


def build_report(clips, dataset_root, out_root, manifest_records, out_html,
                  map_buffer_margin_m=150.0, zoom_margin_m=15.0):
    sections = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, video_path in enumerate(clips, 1):
            print(f"=== [{i}/{len(clips)}] {video_path} ===")
            paths = clip_paths(video_path, dataset_root, out_root)
            tmp_png = os.path.join(tmp, f"{i}.png")
            try:
                plot_clip(
                    result_npz=paths["result_npz"], gps_csv=paths["gps_csv"],
                    network_dir=paths["network_dir"], out=tmp_png,
                    pairs_out=os.path.join(tmp, f"{i}_pairs.png"),
                    map_buffer_margin_m=map_buffer_margin_m, zoom_margin_m=zoom_margin_m,
                )
            except Exception as e:
                print(f"[WARN] skipping {video_path}: {e}")
                continue

            with open(tmp_png, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")
            meta_html = _clip_meta_html(manifest_records.get(video_path, {}))
            sections.append(
                f'<div class="clip">\n'
                f'  <h2>[{i}] {video_path}</h2>\n'
                f'  <div class="meta">{meta_html}</div>\n'
                f'  <img src="data:image/png;base64,{img_b64}" alt="trajectory plot {i}">\n'
                f'</div>\n'
            )

    html = (HTML_HEAD + f"<h1>GPS fusion trajectory QA report ({len(sections)} clips)</h1>\n"
            + "".join(sections) + "</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)) or ".", exist_ok=True)
    with open(out_html, "w") as f:
        f.write(html)
    print(f"\nWrote {out_html} ({len(sections)}/{len(clips)} clips)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, required=True, help="Number of clips to sample")
    ap.add_argument("--manifest", required=True, help="scripts/batch_infer.py's manifest.jsonl")
    ap.add_argument("--dataset_root", required=True,
                     help="Dataset root manifest.jsonl's clip_ids are absolute paths under")
    ap.add_argument("--out_root", required=True,
                     help="batch_infer.py's --out_root (where each clip's result.npz lives)")
    ap.add_argument("--out", required=True, help="Output .html report path")
    ap.add_argument("--seed", type=int, help="Random seed, for a reproducible sample")
    ap.add_argument("--map_buffer_margin_m", type=float, default=150.0)
    ap.add_argument("--zoom_margin_m", type=float, default=15.0)
    args = ap.parse_args()

    completed = list(find_completed_clips(args.manifest))
    if not completed:
        ap.error(f"no clips with a successful GPS-fusion route match found in {args.manifest}")
    rng = random.Random(args.seed)
    chosen = rng.sample(completed, min(args.sample, len(completed)))
    print(f"Sampling {len(chosen)}/{len(completed)} completed clips...")

    manifest_records = _load_manifest_records(args.manifest)
    build_report(chosen, args.dataset_root, args.out_root, manifest_records, args.out,
                 map_buffer_margin_m=args.map_buffer_margin_m, zoom_margin_m=args.zoom_margin_m)


if __name__ == "__main__":
    main()
