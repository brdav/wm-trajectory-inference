#!/usr/bin/env python3
"""
export_sources_meta.py - write sources_meta.json: every source with its provider kind, licence, attribution and (where known) year, so the
`sources` field of an image index can be joined to its licence and credit line.

  python3 export_sources_meta.py --out <images dir>/sources_meta.json
"""
import argparse
import json

from sources import ATTRIBUTION, SOURCES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = {n: dict(regions=s["regions"] + [p + "*" for p in s.get("prefixes", ())], kind=s["kind"], license=s.get("license"),
                    attribution=ATTRIBUTION.get(n), year=s.get("year"), gsd_m=s.get("gsd_m"), status=s.get("status", "open"),
                    note=s.get("note")) for n, s in SOURCES.items()}
    json.dump(meta, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"{len(meta)} sources -> {a.out}")


if __name__ == "__main__":
    main()
