#!/usr/bin/env python3
"""Download whole-region OSM extracts (Geofabrik) for every country/region this dataset spans,
and build each one's spatial index.

Why: vggt_slam/map_match.py originally fetched small per-tile bounding boxes from the public
Overpass API on demand, lazily, per clip. That API enforces fairly aggressive rate limiting on
repeated/concurrent queries from the same client, which made GPS fusion unreliable at batch
scale. Geofabrik's extracts are plain HTTPS file downloads from a CDN, not a rate-limited query
API, and are downloaded once, up front, instead -- vggt_slam/map_match.py no longer queries
Overpass at all.

A raw .osm.pbf has no spatial index, though -- finding "which roads are near this GPS point"
means linearly scanning the whole file regardless of how small the area of interest is (slow for
country-sized files, see memory/performance-investigations.md). Doing that lazily, once per
clip's first cache miss, would stall that clip's GPU worker for however long its country's file
takes to scan. So this script also builds a SQLite R-tree spatial index over each region's
highway ways right after downloading it (see vggt_slam.map_match._build_spatial_index) --
another one-time, per-region cost, after which any future bbox query (any tile, any clip, even
ones not known about yet) is an indexed lookup taking on the order of milliseconds instead.

Usage:
    python scripts/setup/download_osm_extracts.py --out_dir DIR [--workers 6] [--index_workers 4] \\
        [--dry_run] [--skip_index]

--dry_run: just HEAD-request every URL and report total size, without downloading or indexing.
--skip_index: download only, skip spatial-index building (e.g. to re-run indexing separately).

Output layout: {out_dir}/{region_key}.osm.pbf (+ {region_key}.osm.pbf.spatial_index.sqlite),
where region_key matches what vggt_slam/geofabrik.py's region_key_for_clip(country, state)
returns (see that function -- kept in sync with REGION_SLUGS below).
"""
import argparse
import concurrent.futures
import os
import sys
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))
from vggt_slam.geofabrik import BASE, all_targets  # noqa: E402


def _head_size(path):
    req = urllib.request.Request(f"{BASE}/{path}-latest.osm.pbf", method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0))


def _download(path, dest):
    tmp = dest + ".part"
    req = urllib.request.Request(f"{BASE}/{path}-latest.osm.pbf")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def _index_one(dest):
    """Runs in its own process (ProcessPoolExecutor) -- building a spatial index is CPU-bound
    Python-level work (iterating pyosmium objects, SQLite inserts), so real parallelism across
    regions needs separate processes, not threads (which would just serialize on the GIL)."""
    from vggt_slam.map_match import _build_spatial_index, _spatial_index_path

    index_path = _spatial_index_path(dest)
    if os.path.exists(index_path):
        return "already indexed"
    _build_spatial_index(dest, index_path)
    return f"indexed ({os.path.getsize(index_path) / 1e6:.0f} MB)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=6, help="Concurrent downloads -- Geofabrik is a plain CDN, not rate-limited like Overpass, so a handful in parallel is fine")
    p.add_argument("--index_workers", type=int, default=4, help="Concurrent spatial-index builds (separate processes, CPU-bound) -- each can use several GB RAM for a large country, so don't set this too high relative to available memory")
    p.add_argument("--dry_run", action="store_true", help="Only HEAD-check sizes, don't download or index")
    p.add_argument("--skip_index", action="store_true", help="Download only, skip spatial-index building")
    args = p.parse_args()

    targets = list(all_targets())
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dry_run:
        total = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_head_size, path): (key, path) for key, path in targets}
            for fut in concurrent.futures.as_completed(futs):
                key, path = futs[fut]
                try:
                    size = fut.result()
                    total += size
                    print(f"{key}: {size / 1e6:.0f} MB")
                except Exception as e:
                    print(f"{key}: FAILED ({e})", file=sys.stderr)
        print(f"Total: {total / 1e9:.1f} GB across {len(targets)} regions")
        return

    def work(item):
        key, path = item
        dest = os.path.join(args.out_dir, key.replace("/", "_") + ".osm.pbf")
        if os.path.exists(dest):
            return key, dest, "already downloaded"
        try:
            _download(path, dest)
            return key, dest, f"downloaded ({os.path.getsize(dest) / 1e6:.0f} MB)"
        except Exception as e:
            return key, dest, f"FAILED: {e}"

    done = 0
    dests = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for key, dest, status in pool.map(work, targets):
            done += 1
            print(f"[{done}/{len(targets)}] {key}: {status}", flush=True)
            if not status.startswith("FAILED"):
                dests.append((key, dest))

    if args.skip_index:
        return
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.index_workers) as pool:
        futs = {pool.submit(_index_one, dest): key for key, dest in dests}
        for fut in concurrent.futures.as_completed(futs):
            key = futs[fut]
            done += 1
            try:
                status = fut.result()
            except Exception as e:
                status = f"FAILED: {e}"
            print(f"[index {done}/{len(dests)}] {key}: {status}", flush=True)


if __name__ == "__main__":
    main()
