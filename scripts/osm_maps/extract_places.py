#!/usr/bin/env python3
"""Two small lookups from the OSM extracts for the route text (scripts/osm_maps/osm_route_text.py), independent of
`osm_bev.py prepare`:
  _places.parquet           cities, towns and villages (name, kind, position, population): the "Location" line
  _roundabout_ways.parquet  ids of the ways tagged junction=roundabout (the prepared tiles do not keep that tag)
Only place nodes and ways with a junction tag are read, so a country takes a few minutes and the files are a few MB.

  python scripts/osm_maps/extract_places.py --extracts <dir with *.osm.pbf> --out-dir <osm-data> [--regions Japan "United Kingdom"]

--regions are file stems; the default is every extract in the folder. make_osm_maps.py reads both files from its --osm-data.

The place name is the first of name:en, int_name, name:ja-Latn, name:ja_rm, name (so Japanese places carry a Latin spelling
when OSM has one). Other place kinds (hamlet, suburb, ...) are not kept: the point is "which town is this", not a gazetteer.
"""
import argparse
from pathlib import Path

KINDS = ("city", "town", "village")
NAME_KEYS = ("name:en", "int_name", "name:ja-Latn", "name:ja_rm", "name")


def read_pbf(pbf):
    """-> (place rows, roundabout way ids)."""
    import osmium

    places, roundabouts = [], []
    fp = osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY).with_filter(osmium.filter.KeyFilter("place", "junction"))
    for o in fp:
        if o.is_way():
            if o.tags.get("junction") == "roundabout":
                roundabouts.append(o.id)
            continue
        kind = o.tags.get("place")
        if kind not in KINDS or not o.location.valid():
            continue
        name = next((o.tags[k] for k in NAME_KEYS if k in o.tags), None)
        if not name:
            continue
        pop = o.tags.get("population", "")
        places.append(dict(name=name, kind=kind, lat=o.location.lat, lon=o.location.lon,
                           population=int(pop) if pop.isdigit() else 0))
    return places, roundabouts


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracts", required=True, help="folder with <region>.osm.pbf files")
    ap.add_argument("--out-dir", required=True, help="where _places.parquet and _roundabout_ways.parquet are written")
    ap.add_argument("--regions", nargs="*", help="file stems to read (default: all)")
    a = ap.parse_args()

    pbfs = sorted(Path(a.extracts).glob("*.osm.pbf"))
    if a.regions:
        pbfs = [p for p in pbfs if p.name.removesuffix(".osm.pbf") in set(a.regions)]
    places, roundabouts = [], []
    for p in pbfs:
        pl, ra = read_pbf(p)
        print(f"{p.name}: {len(pl)} places, {len(ra)} roundabout ways", flush=True)
        places += pl
        roundabouts += ra
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ("name", "kind", "lat", "lon", "population")
    pq.write_table(pa.table({c: [r[c] for r in places] for c in cols}), out / "_places.parquet")
    pq.write_table(pa.table({"osm_id": pa.array(sorted(set(roundabouts)), pa.int64())}), out / "_roundabout_ways.parquet")
    print(f"{len(places)} places, {len(set(roundabouts))} roundabout ways -> {out}")


if __name__ == "__main__":
    main()
