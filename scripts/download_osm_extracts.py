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
means linearly scanning the whole file regardless of how small the area of interest is (measured
directly: ~30s for a 139MB country file, minutes for a multi-GB one). Doing that lazily, once per
clip's first cache miss, would stall that clip's GPU worker for however long its country's file
takes to scan. So this script also builds a SQLite R-tree spatial index over each region's
highway ways right after downloading it (see vggt_slam.map_match._build_spatial_index) --
another one-time, per-region cost, after which any future bbox query (any tile, any clip, even
ones not known about yet) is an indexed lookup taking on the order of milliseconds instead.

Usage:
    python scripts/download_osm_extracts.py --out_dir DIR [--workers 6] [--index_workers 4] \\
        [--dry_run] [--skip_index]

--dry_run: just HEAD-request every URL and report total size, without downloading or indexing.
--skip_index: download only, skip spatial-index building (e.g. to re-run indexing separately).

Output layout: {out_dir}/{region_key}.osm.pbf (+ {region_key}.osm.pbf.spatial_index.sqlite),
where region_key matches what vggt_slam/map_match.py's region_key_for_clip(country, state)
returns (see that function -- kept in sync with REGION_SLUGS below).
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

BASE = "https://download.geofabrik.de"

# natix dataset country-folder name -> Geofabrik path (relative to BASE, without -latest.osm.pbf).
# Ireland has no Republic-of-Ireland-only extract on Geofabrik; "ireland-and-northern-ireland" is
# the only/correct option (also covers Northern Ireland/UK territory).
#
# Czech Republic and Liechtenstein aren't natix dataset folders -- added once real clips filed
# under other countries turned out to actually be located there (confirmed: a "Spain"-labeled
# clip actually near Brno, Czech Republic; several "Austria"-labeled clips actually in
# Liechtenstein, near Vaduz) -- same mislabeling pattern as vggt_slam/country_regions.py's other
# entries, just two more neighbors that happened not to have any of their OWN natix folder to
# already guarantee coverage.
EUROPE_COUNTRIES = {
    "Austria": "austria", "Belgium": "belgium", "Croatia": "croatia",
    "Czech Republic": "czech-republic", "Finland": "finland", "France": "france",
    "Germany": "germany", "Greece": "greece", "Hungary": "hungary", "Iceland": "iceland",
    "Ireland": "ireland-and-northern-ireland", "Italy": "italy",
    "Liechtenstein": "liechtenstein", "Latvia": "latvia", "Netherlands": "netherlands",
    "Poland": "poland", "Portugal": "portugal", "Romania": "romania", "Slovenia": "slovenia",
    "Spain": "spain", "Sweden": "sweden", "Switzerland": "switzerland",
}
EUROPE_PREFIX = "europe"

OTHER_REGIONS = {
    "Canada": "north-america/canada",
    "Puerto Rico": "north-america/us/puerto-rico",
    "Japan": "asia/japan",
    # Not a natix dataset country folder -- a fallback-only region for clips mislabeled as
    # Austria (confirmed: GPS fixes ~300km south of Austria, in Bosnia and Herzegovina). See
    # scripts/batch_infer.py's FALLBACK_REGIONS.
    "Bosnia and Herzegovina": "europe/bosnia-herzegovina",
}

# "United States" clips are nested one level deeper (natix's own layout: United States/<State>/...,
# unlike every other country here) -- each state maps to its own Geofabrik us/<slug> extract.
#
# Includes every US state + DC, not just the ones natix's own folder layout happens to have --
# clips are sometimes filed under the wrong state (confirmed on real clips, see
# vggt_slam/us_regions.py's docstring), so a clip's *true* location can land in a state that has
# no folder of its own in the dataset at all (e.g. a clip filed under "Rhode Island" but actually
# in New Hampshire, which otherwise wouldn't have an extract downloaded here). GPS-centroid-based
# region selection (scripts/batch_infer.py's `_region_candidates`) needs the full set to have
# anywhere to actually resolve a mislabeled clip to.
US_STATE_SLUGS = {
    "Alabama": "alabama", "Alaska": "alaska", "Arizona": "arizona", "Arkansas": "arkansas",
    "California": "california", "Colorado": "colorado", "Connecticut": "connecticut",
    "Delaware": "delaware", "District of Columbia": "district-of-columbia", "Florida": "florida",
    "Georgia": "georgia", "Hawaii": "hawaii", "Idaho": "idaho", "Illinois": "illinois",
    "Indiana": "indiana", "Iowa": "iowa", "Kansas": "kansas", "Kentucky": "kentucky",
    "Louisiana": "louisiana", "Maine": "maine", "Maryland": "maryland",
    "Massachusetts": "massachusetts", "Michigan": "michigan", "Minnesota": "minnesota",
    "Mississippi": "mississippi", "Missouri": "missouri", "Montana": "montana",
    "Nebraska": "nebraska", "Nevada": "nevada", "New Hampshire": "new-hampshire",
    "New Jersey": "new-jersey", "New Mexico": "new-mexico", "New York": "new-york",
    "North Carolina": "north-carolina", "North Dakota": "north-dakota", "Ohio": "ohio",
    "Oklahoma": "oklahoma", "Oregon": "oregon", "Pennsylvania": "pennsylvania",
    "Rhode Island": "rhode-island", "South Carolina": "south-carolina",
    "South Dakota": "south-dakota", "Tennessee": "tennessee", "Texas": "texas", "Utah": "utah",
    "Vermont": "vermont", "Virginia": "virginia", "Washington": "washington",
    "West Virginia": "west-virginia", "Wisconsin": "wisconsin", "Wyoming": "wyoming",
}


def region_key_for_clip(country, state=None):
    """country/state as they appear in the dataset's own folder names. Returns the same
    region_key build_network's region-extract lookup uses -- keep any change here in sync with
    vggt_slam/map_match.py.
    """
    if country == "United States":
        return f"United States/{state}"
    return country


def all_targets():
    """Yields (region_key, geofabrik_path) for every extract this dataset needs."""
    for name, slug in EUROPE_COUNTRIES.items():
        yield name, f"{EUROPE_PREFIX}/{slug}"
    for name, path in OTHER_REGIONS.items():
        yield name, path
    for state, slug in US_STATE_SLUGS.items():
        yield region_key_for_clip("United States", state), f"north-america/us/{slug}"


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

    # TODO: add this to image
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--index-url", "https://pypi.org/simple", "--no-deps", "osmium"], check=True)

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
