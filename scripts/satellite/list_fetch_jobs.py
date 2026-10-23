#!/usr/bin/env python3
"""
list_fetch_jobs.py - the work list of run_fetch_tiles.sbatch: one line "<name> <region>|<region>|..." per fetch_tiles.py process
(spaces in region names written as ~). Regions with at least one open source; the US states are split over --us-groups processes.

  python3 list_fetch_jobs.py --osm-data <osm_bev/data> [--regions Switzerland Austria] [--us-groups 3]
"""
import argparse
import json
from pathlib import Path

from sources import sources_of


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm-data", required=True, help="dir with _regions.json (the region list of the OSM data)")
    ap.add_argument("--regions", nargs="*", help="only these (default: all)")
    ap.add_argument("--us-groups", type=int, default=3)
    a = ap.parse_args()
    # _regions.json keys are "United States/Nevada"; the sample index (and sources.py) use the directory name "United States_Nevada"
    regs = a.regions or [k.replace("/", "_") for k in json.loads((Path(a.osm_data) / "_regions.json").read_text())]
    regs = [r for r in regs if sources_of(r)]
    us = [r for r in regs if r.startswith("United States_")]
    for r in regs:
        if r not in us:
            print(r.replace(" ", "_"), r.replace(" ", "~"))
    for i in range(a.us_groups):
        if us[i::a.us_groups]:
            print(f"US{i}", "|".join(r.replace(" ", "~") for r in us[i::a.us_groups]))


if __name__ == "__main__":
    main()
