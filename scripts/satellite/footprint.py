#!/usr/bin/env python3
"""
footprint.py - how much ground do the satellite crops cover? Reads the OSM map index (one record per image, with lat/lon) and
counts the unique web-mercator tiles within the axis-aligned window every heading-up 384 m crop needs (384*sqrt(2) = 543 m, plus
a margin). Per region: samples, unique tiles at zoom Z, area, and the redundancy (samples per unique tile).

  python3 scripts/satellite/footprint.py --maps <osm_bev/maps_predicted> [--zoom 17] [--half-m 285] [--out footprint.json]
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from tiles import window_tile_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--zoom", type=int, default=17)
    ap.add_argument("--half-m", type=float, default=285.0, help="half side of the window around each sample [m]")
    ap.add_argument("--out")
    a = ap.parse_args()

    by = defaultdict(lambda: ([], []))
    for f in sorted(glob.glob(f"{a.maps}/index*.jsonl")):
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                by[r["region"]][0].append(r["lat"])
                by[r["region"]][1].append(r["lon"])

    n = 2 ** a.zoom
    out, tot_s, tot_t, tot_km2 = {}, 0, 0, 0.0
    for reg, (la, lo) in sorted(by.items()):
        la, lo = np.array(la), np.array(lo)
        u = window_tile_ids(la, lo, a.zoom, a.half_m)
        tile_m = 40_075_017.0 * np.cos(np.radians(la.mean())) / n          # tile side at the region's mean latitude
        km2 = len(u) * (tile_m / 1000.0) ** 2
        out[reg] = dict(samples=len(la), tiles=int(len(u)), km2=round(km2, 1), samples_per_tile=round(len(la) / len(u), 1),
                        tile_m=round(tile_m, 1), lat=round(float(la.mean()), 2), lon=round(float(lo.mean()), 2))
        tot_s += len(la); tot_t += len(u); tot_km2 += km2
    for reg, d in sorted(out.items(), key=lambda kv: -kv[1]["km2"]):
        print(f"{reg:32s} samples={d['samples']:8d} tiles={d['tiles']:8d} area={d['km2']:9.1f} km2  samples/tile={d['samples_per_tile']}")
    print(f"TOTAL z{a.zoom}: samples={tot_s} tiles={tot_t} area={tot_km2:.0f} km2 (tile ~{np.mean([d['tile_m'] for d in out.values()]):.0f} m)")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
