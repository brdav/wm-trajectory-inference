#!/usr/bin/env python3
"""
summarize_images.py - totals of the rendered aerial images and a consistency check against the OSM maps: every OSM map record must be
accounted for exactly once, either as an image or as an unavailable sample, and every image must point to an existing OSM record.

  python3 summarize_images.py --images <satellite/images> --maps <osm_bev/maps_predicted> [--out summary_stats.json]
"""
import argparse
import collections
import glob
import json
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    osm = {}
    for f in sorted(glob.glob(f"{a.maps}/index*.jsonl")):
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                osm[r["image"]] = r["region"]
    imgs, unav = {}, {}
    for zp in sorted(glob.glob(f"{a.images}/**/*.zip", recursive=True)):
        with zipfile.ZipFile(zp) as zf:
            for ln in zf.read("index.jsonl").decode().splitlines():
                r = json.loads(ln)
                imgs[r["osm_image"]] = r
            for ln in zf.read("unavailable.jsonl").decode().splitlines():
                r = json.loads(ln)
                unav[r["osm_image"]] = r
    both, none = set(imgs) & set(unav), set(osm) - set(imgs) - set(unav)
    stray = (set(imgs) | set(unav)) - set(osm)
    print(f"OSM records {len(osm)} | images {len(imgs)} | unavailable {len(unav)}")
    print(f"consistency: in both {len(both)}, in neither {len(none)}, not in the OSM index {len(stray)}  ->",
          "OK" if not (both or none or stray) else "PROBLEM")
    reasons = collections.Counter(u["reason"] for u in unav.values())
    print("unavailable by reason:", dict(reasons))
    by_region = collections.defaultdict(lambda: collections.Counter())
    for k, reg in osm.items():
        by_region[reg]["total"] += 1
        by_region[reg]["image" if k in imgs else unav[k]["reason"] if k in unav else "missing"] += 1
    print(f"{'region':30s} {'total':>8s} {'images':>8s} {'%':>5s} {'no_source':>10s} {'no_imagery':>11s}")
    for reg, c in sorted(by_region.items(), key=lambda kv: -kv[1]["total"]):
        print(f"{reg:30s} {c['total']:8d} {c['image']:8d} {100 * c['image'] / c['total']:5.1f} {c['no_source']:10d} {c['no_imagery']:11d}")
    srcs = collections.Counter(s for r in imgs.values() for s in r["sources"])
    print("images per source (a crop with two sources counts for both):", dict(srcs.most_common()))
    miss = [r["missing"] for r in imgs.values()]
    print(f"images with some uncovered pixels: {sum(m > 0 for m in miss)} (mean uncovered of those {sum(m for m in miss if m > 0) / max(1, sum(m > 0 for m in miss)):.2f})")
    if a.out:
        json.dump(dict(osm=len(osm), images=len(imgs), unavailable=dict(reasons), by_region={k: dict(v) for k, v in by_region.items()},
                       sources=dict(srcs)), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
