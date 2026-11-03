#!/usr/bin/env python3
"""Cities, towns and villages of the OSM extracts -> <osm-data>/_places.parquet, for the "Location: ..." line of the
route text (scripts/osm_maps/osm_route_text.py). Only the place nodes are read (name, kind, position, population), so a
country takes a few minutes and the file is a few MB. Independent of `osm_bev.py prepare`.

  python scripts/osm_maps/extract_places.py --extracts <dir with *.osm.pbf> --out <osm-data>/_places.parquet \\
         [--regions Japan "United Kingdom"]     # file stems; default: every extract in the folder

The name is the first of name:en, int_name, name:ja-Latn, name:ja_rm, name (so Japanese places carry a Latin spelling when
OSM has one). Other place kinds (hamlet, suburb, ...) are not kept: the point is "which town is this", not a gazetteer.
"""
import argparse
from pathlib import Path

KINDS = ("city", "town", "village")
NAME_KEYS = ("name:en", "int_name", "name:ja-Latn", "name:ja_rm", "name")


def read_places(pbf):
    import osmium

    rows = []
    fp = osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_filter(osmium.filter.KeyFilter("place"))
    for o in fp:
        kind = o.tags.get("place")
        if kind not in KINDS or not o.location.valid():
            continue
        name = next((o.tags[k] for k in NAME_KEYS if k in o.tags), None)
        if not name:
            continue
        pop = o.tags.get("population", "")
        rows.append(dict(name=name, kind=kind, lat=o.location.lat, lon=o.location.lon,
                         population=int(pop) if pop.isdigit() else 0))
    return rows


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracts", required=True, help="folder with <region>.osm.pbf files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", nargs="*", help="file stems to read (default: all)")
    a = ap.parse_args()

    pbfs = sorted(Path(a.extracts).glob("*.osm.pbf"))
    if a.regions:
        pbfs = [p for p in pbfs if p.name.removesuffix(".osm.pbf") in set(a.regions)]
    rows = []
    for p in pbfs:
        got = read_places(p)
        print(f"{p.name}: {len(got)} places", flush=True)
        rows += got
    cols = ("name", "kind", "lat", "lon", "population")
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in cols}), a.out)
    print(f"{len(rows)} places -> {a.out}")


if __name__ == "__main__":
    main()
