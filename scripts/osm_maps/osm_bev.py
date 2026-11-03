#!/usr/bin/env python3
"""
osm_bev.py - OSM extracts -> ego-centred top-down RGB map crops ("everything is an image").

Two stages
----------
1. prepare : country PBF  ->  grid-partitioned GeoParquet   (run once per country)
2. render  : (lat, lon, heading[, country][, future ego path]) -> RGB uint8 image

Design choices
--------------
* Crop size comes from the prediction horizon:  v_max * T * safety
      120 km/h = 33.3 m/s  x 5 s = 166.7 m   x 1.15 safety  ->  ~192 m half-extent
  The crop is a square centred on the car, so the car's whole 5 s reach (in ANY direction,
  also after rotating to heading-up) is inside the image. Pixels are rounded up to a multiple
  of STRIDE (32 = spatial stride of the LTX video VAE; check this for your checkpoint).
  Default: 0.5 m/px -> 768 px -> 384 m x 384 m -> 24x24 latent tokens per image.
* A fixed semantic palette (below) and fixed metres-per-pixel make images comparable across
  all countries. Projection is a local azimuthal-equidistant frame centred on the car, so no
  per-country CRS is needed.
* Everything is drawn into the one RGB image: speed limits as numbers on the roads, lanes as dividers and per-lane
  arrows, one-way as arrows, a small set of sign pictograms, and optionally the ego's future path (see
  scripts/osm_maps/make_osm_maps.py). There is no separate attribute image.

Dependencies:  pip install osmium shapely pyproj pyarrow numpy opencv-python
Optional    :  osmium-tool (CLI) - strongly recommended for big countries (pre-filtering).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
V_MAX_KMH = 120.0      # realistic max ego speed
HORIZON_S = 5.0        # how far ahead the model must be able to see the car
SAFETY = 1.15          # margin for acceleration / localisation / heading error
RES_M_PER_PX = 0.5     # metres per output pixel
STRIDE = 32            # output side is rounded up to a multiple of this
SUPERSAMPLE = 2        # render at 2x and downsample (anti-aliasing for thin features)
CELL_DEG = 0.05        # spatial grid used for on-disk partitioning (~5 km)

# Geofabrik paths (https://download.geofabrik.de/<path>-latest.osm.pbf). Verify before use.
# NB: Geofabrik's 'us' file already contains Puerto Rico - you can skip one of the two.
COUNTRIES = {
    "Austria": "europe/austria", "Belgium": "europe/belgium", "Canada": "north-america/canada",
    "Croatia": "europe/croatia", "Finland": "europe/finland", "France": "europe/france",
    "Germany": "europe/germany", "Greece": "europe/greece", "Hungary": "europe/hungary",
    "Iceland": "europe/iceland", "Ireland": "europe/ireland-and-northern-ireland",
    "Italy": "europe/italy", "Japan": "asia/japan", "Latvia": "europe/latvia",
    "Netherlands": "europe/netherlands", "Poland": "europe/poland", "Portugal": "europe/portugal",
    "Puerto Rico": "north-america/us/puerto-rico", "Romania": "europe/romania",
    "Slovenia": "europe/slovenia", "Spain": "europe/spain", "Sweden": "europe/sweden",
    "Switzerland": "europe/switzerland", "United States": "north-america/us",
}
LEFT_HAND_TRAFFIC = {"Ireland", "Japan"}   # everyone else in the list drives on the right


def region_dir(key: str) -> str:
    """Region key ('Japan', 'United States/Nevada') -> directory name under the data root."""
    return key.replace("/", "_")


def country_of(region: str | None) -> str | None:
    """'United States_Nevada' / 'United States/Nevada' -> 'United States'."""
    return re.split(r"[/_]", region)[0] if region else None


def drive_side(country: str | None) -> str:
    return "left" if country_of(country) in LEFT_HAND_TRAFFIC else "right"



def crop_geometry(res=RES_M_PER_PX, v_kmh=V_MAX_KMH, t_s=HORIZON_S, safety=SAFETY, stride=STRIDE):
    """-> (image side in px, half extent in m)."""
    reach = v_kmh / 3.6 * t_s * safety
    px = int(math.ceil(2 * reach / res / stride) * stride)
    return px, px * res / 2.0


# --------------------------------------------------------------------------------------
# Styling (RGB). Semantic, high-contrast colours; identical for every country.
# --------------------------------------------------------------------------------------
BG = (242, 239, 233)
COL = dict(
    green=(196, 224, 170), water=(140, 190, 235), parking=(222, 222, 236), pedestrian=(236, 214, 226),
    island=(200, 200, 190), building=(196, 189, 182), rail=(70, 70, 80), casing=(105, 105, 105),
    arrow=(60, 60, 60), ego=(150, 0, 255), ego_ring=(255, 255, 255), outline=(25, 25, 25),
    cross_signals=(0, 80, 220), cross_marked=(90, 160, 255), cross_other=(175, 205, 255),
    stop=(210, 0, 0), give_way=(255, 140, 0), signals=(0, 155, 0), calming=(255, 0, 255),
    bus_stop=(255, 215, 0), level_crossing=(20, 20, 20), mini_roundabout=(90, 90, 90),
    sign_speed=(255, 255, 255), sign_speed_ring=(215, 0, 0),
    speed_map=(20, 20, 20), speed_inferred=(20, 20, 20),   # same black: the mapped/inferred difference is too subtle to show
)
# class: (draw priority, default two-way width [m], RGB fill)
ROADS = {
    "motorway": (90, 10.0, (214, 69, 65)), "motorway_link": (85, 5.0, (226, 100, 95)),
    "trunk": (80, 9.0, (232, 130, 60)), "trunk_link": (75, 5.0, (240, 160, 100)),
    "primary": (70, 8.0, (245, 175, 50)), "primary_link": (65, 5.0, (248, 200, 100)),
    "secondary": (60, 7.0, (247, 220, 90)), "secondary_link": (55, 4.5, (250, 232, 140)),
    "tertiary": (50, 6.5, (255, 250, 190)), "tertiary_link": (45, 4.5, (255, 252, 215)),
    "unclassified": (40, 5.5, (250, 250, 250)), "residential": (35, 5.5, (255, 255, 255)),
    "living_street": (30, 4.5, (225, 240, 255)), "service": (25, 3.5, (205, 205, 205)),
    "pedestrian": (20, 4.0, (232, 190, 215)), "track": (15, 3.0, (190, 160, 110)),
    "cycleway": (12, 2.0, (70, 110, 230)), "footway": (10, 1.6, (235, 140, 140)),
    "path": (9, 1.2, (200, 150, 110)), "steps": (8, 1.5, (210, 110, 110)),
    "bridleway": (7, 1.5, (170, 130, 90)),
}
NON_VEHICLE = {"pedestrian", "track", "cycleway", "footway", "path", "steps", "bridleway"}
IMPLICIT_ONEWAY = {"motorway", "motorway_link"}
GREEN = {"natural": {"wood", "scrub", "grassland", "heath"},
         "landuse": {"forest", "grass", "meadow", "village_green", "recreation_ground"},
         "leisure": {"park", "garden", "pitch", "playground", "golf_course", "nature_reserve"}}
RAIL = {"rail", "light_rail", "tram", "narrow_gauge"}
AREA_PRIORITY = ["green", "water", "parking", "pedestrian", "island", "roadarea", "building"]

_END = r"(\[|[;,\s]|$)"
# traffic_sign is free text with country prefixes, so only the two classes whose national codes could be validated are
# mapped; everything else is deliberately NOT drawn (validation: memory/osm-bev-tag-reliability.md).
SIGN_PATTERNS = [
    (r"(^|[;,\s])(stop|DE:206|US:R1-1|FR:AB4|CH:3\.01|SE:B2|PL:B-20|IT:II\.37|ES:R-?2|JP:330[-\w.]*)" + _END, "stop"),
    (r"(^|[;,\s])(give_way|DE:205|US:R1-2|FR:AB3a|CH:3\.02|SE:B1|PL:A-7|IT:II\.36|ES:R-?1)" + _END, "give_way"),
]


# --------------------------------------------------------------------------------------
# Tag parsing
# --------------------------------------------------------------------------------------
IMPLICIT_SPEED = {   # country-prefixed implicit values -> km/h (statutory defaults; only where unambiguous)
    "CH:urban": 50, "CH:rural": 80, "CH:motorway": 120, "CH:trunk": 100, "CH:living_street": 20,
    "AT:urban": 50, "AT:rural": 100, "AT:motorway": 130, "DE:urban": 50, "DE:rural": 100,
    "FR:urban": 50, "FR:rural": 80, "FR:motorway": 130,
    "IT:urban": 50, "IT:rural": 90, "IT:trunk": 110, "IT:motorway": 130,
    "ES:urban": 50, "ES:rural": 90, "ES:trunk": 100, "ES:motorway": 120,
    "PL:urban": 50, "PL:rural": 90, "PL:expressway": 120, "PL:motorway": 140, "PL:living_street": 20,
    "CZ:urban": 50, "CZ:rural": 90, "CZ:motorway": 130,
    "JP:nsl": 60, "JP:expressway": 100,
}
_IMPLICIT_LC = {k.lower(): float(x) for k, x in IMPLICIT_SPEED.items()}


def _parse_one_speed(v: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?", v)
    if m:
        val = float(m.group(1))
        return val * 1.609344 if m.group(2) == "mph" else val
    if v == "walk":
        return 7.0
    if v == "none":
        return 140.0
    m = re.fullmatch(r"[a-z]{2}(?:-[a-z]{2,3})?:(?:zone:?)?(\d{2,3})", v)
    if m:
        return float(m.group(1))
    return _IMPLICIT_LC.get(v, 0.0)


def parse_speed_kmh(v: str | None) -> float:
    """OSM maxspeed -> km/h. Plain numbers are km/h; mph needs an explicit suffix. 0 = unknown.
    'none' -> 140. Implicit: 'XX:zoneNN' -> NN, and the IMPLICIT_SPEED table ('CH:urban', ...).
    Lists ('40;50', '50|30', '35 mph;30 mph') -> the lowest value if every item parses."""
    if not v:
        return 0.0
    items = [i.strip() for i in re.split(r"[;|]", v.strip().lower()) if i.strip()]
    if len(items) > 1 and "," not in v:
        vals = [_parse_one_speed(i) for i in items]
        return min(vals) if all(vals) else 0.0
    return _parse_one_speed(items[0]) if items else 0.0


def way_speed_kmh(tags: dict) -> float:
    """Best speed limit for a way: explicit maxspeed, zone / type tags, then directional variants."""
    for k in ("maxspeed", "zone:maxspeed", "maxspeed:type", "source:maxspeed"):
        sp = parse_speed_kmh(tags.get(k))
        if sp:
            return sp
    return max(parse_speed_kmh(tags.get("maxspeed:forward")), parse_speed_kmh(tags.get("maxspeed:backward")))


def parse_width_m(v: str | None) -> float:
    if not v:
        return 0.0
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(?:(')|m)?(?:\s*(\d+)\s*\")?", v.replace(",", "."))
    if not m:
        return 0.0
    w = float(m.group(1))
    if m.group(2):  # feet (and optional inches)
        w = w * 0.3048 + (float(m.group(3)) * 0.0254 if m.group(3) else 0.0)
    return w if 0.5 <= w <= 60 else 0.0


def parse_int(v: str | None) -> int:
    try:
        return max(0, int(float(v))) if v else 0
    except ValueError:
        return 0


def oneway_of(tags: dict, cls: str) -> int:
    ow = tags.get("oneway")
    if ow in ("yes", "true", "1"):
        return 1
    if ow == "-1":
        return -1
    if ow == "no":
        return 0
    if tags.get("junction") == "roundabout" or cls in IMPLICIT_ONEWAY:
        return 1
    return 0


def direction_of(tags: dict) -> int:
    """0 none, 1 forward (along way), -1 backward, 2 both."""
    for k in ("direction", "traffic_signals:direction", "traffic_sign:direction"):
        v = tags.get(k)
        if v in ("forward",):
            return 1
        if v in ("backward",):
            return -1
        if v in ("both", "all"):
            return 2
    if tags.get("stop") == "all":
        return 2
    return 0


def classify_sign(tags: dict) -> str | None:
    """'stop' / 'give_way' from the free-text traffic_sign (validated codes only), else None."""
    ts = tags.get("traffic_sign") or tags.get("traffic_sign:forward") or tags.get("traffic_sign:backward")
    if not ts:
        return None
    for pat, cat in SIGN_PATTERNS:
        if re.search(pat, ts, re.I):
            return cat
    return None


def crossing_subtype(tags: dict) -> str:
    c = tags.get("crossing") or tags.get("bicycle_crossing") or ""
    if c == "traffic_signals":
        return "signals"
    if c in ("marked", "zebra", "uncontrolled") or tags.get("crossing:markings") not in (None, "no"):
        return "marked"
    return "other"


def classify_way(tags: dict):
    """-> (kind, cls, sub) or None."""
    h = tags.get("highway")
    if h and tags.get("area") != "yes" and tags.get("indoor") is None:
        if h in ROADS:
            if tags.get("footway") == "crossing" or tags.get("cycleway") == "crossing":
                return "crossing_way", "crossing", crossing_subtype(tags)
            return "road", h, ""
    r = tags.get("railway")
    if r in RAIL and tags.get("tunnel") not in ("yes", "building_passage"):
        return "rail", r, ""
    return None


def classify_area(tags: dict):
    """-> (kind='area', cls, sub) or None."""
    if tags.get("building") not in (None, "no"):
        return "area", "building", ""
    if tags.get("natural") == "water" or tags.get("landuse") in ("reservoir", "basin") \
            or tags.get("waterway") == "riverbank":
        return "area", "water", ""
    if tags.get("amenity") == "parking":
        return "area", "parking", ""
    ah = tags.get("area:highway")
    if ah == "traffic_island":
        return "area", "island", ""
    if ah in ("footway", "pedestrian", "cycleway") or (tags.get("area") == "yes"
                                                        and tags.get("highway") in ("pedestrian", "footway")):
        return "area", "pedestrian", ""
    if ah in ROADS or (tags.get("area") == "yes" and tags.get("highway") in ROADS):
        return "area", "roadarea", ah if ah in ROADS else tags.get("highway")   # area:highway=bus_stop etc. is no road class
    for key, vals in GREEN.items():
        if tags.get(key) in vals:
            return "area", "green", ""
    return None


BUMP_TYPES = {"bump", "hump", "table", "cushion"}


def classify_node(tags: dict):
    """-> (kind='node', cls, sub) or None."""
    h = tags.get("highway")
    if h == "stop":
        return "node", "stop", ""
    if h == "give_way":
        return "node", "give_way", ""
    if h == "traffic_signals":
        return "node", "signals", ""
    if h == "crossing":
        return "node", "crossing", crossing_subtype(tags)
    if h == "bus_stop":
        return "node", "bus_stop", ""
    if h == "mini_roundabout":
        return "node", "mini_roundabout", ""
    if "traffic_calming" in tags:
        # only vertical-deflection bumps get the speed-bump symbol; islands / chokers / chicanes are no bumps,
        # and 'yes' / 'no' say nothing about the type
        hit = sorted({v.strip() for v in tags["traffic_calming"].split(";")} & BUMP_TYPES)
        if hit:
            return "node", "calming", hit[0]
    if tags.get("railway") == "level_crossing":
        return "node", "level_crossing", ""
    sign = classify_sign(tags)
    if sign:
        return "node", sign, ""
    return None


# --------------------------------------------------------------------------------------
# Regions: which extract covers a GPS point (Geofabrik .poly boundaries; independent of folder labels)
# --------------------------------------------------------------------------------------
def _parse_poly(txt: str):
    import shapely
    lines = [ln.strip() for ln in txt.splitlines()]
    outers, holes, i = [], [], 1
    while i < len(lines):
        head = lines[i]
        i += 1
        if not head or head == "END":
            continue
        ring = []
        while i < len(lines) and lines[i] != "END":
            if lines[i]:
                x, y = lines[i].split()[:2]
                ring.append((float(x), float(y)))
            i += 1
        i += 1
        if len(ring) >= 4:
            (holes if head.startswith("!") else outers).append(shapely.Polygon(ring))
    geom = shapely.unary_union(outers)
    return geom.difference(shapely.unary_union(holes)) if holes else geom


def fetch_regions(root: str, workers: int = 8):
    """Download the Geofabrik boundary polygon of every extract this dataset uses -> <root>/_regions.json."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import shapely
    from vggt_slam import geofabrik

    targets = dict(geofabrik.all_targets())

    def get(item):
        key, path = item
        txt = urllib.request.urlopen(f"{geofabrik.BASE}/{path}.poly", timeout=120).read().decode()
        return key, shapely.simplify(_parse_poly(txt), 0.003).wkt

    with ThreadPoolExecutor(workers) as ex:
        out = dict(ex.map(get, targets.items()))
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "_regions.json").write_text(json.dumps(out))
    print(f"{len(out)} region polygons -> {Path(root) / '_regions.json'}")


class Regions:
    def __init__(self, root: Path):
        import shapely
        from shapely.strtree import STRtree
        d = json.loads((Path(root) / "_regions.json").read_text())
        self.keys = list(d)
        self.geoms = [shapely.from_wkt(w) for w in d.values()]
        self.tree = STRtree(self.geoms)

    def near(self, lat, lon, radius_m=0.0):
        """Region keys whose extract covers the point (or comes within radius_m of it); nearest one within
        ~5 km if the point itself is outside every polygon (coast / GPS error). Centre-containing first."""
        import shapely
        pt = shapely.Point(lon, lat)
        hit = [int(i) for i in self.tree.query(pt, predicate="intersects")]
        if radius_m:
            ring = pt.buffer(radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.05)))
            hit += [int(i) for i in self.tree.query(ring, predicate="intersects") if int(i) not in hit]
        if not hit:
            idx = self.tree.query_nearest(pt, max_distance=0.05)
            hit = [int(i) for i in idx]
        return [self.keys[i] for i in hit]


# --------------------------------------------------------------------------------------
# Stage 1: prepare  (PBF -> hive-partitioned GeoParquet, one directory per country)
# --------------------------------------------------------------------------------------
TAG_FILTERS = [
    "w/highway", "w/railway=rail,light_rail,tram,narrow_gauge",
    "wr/building", "wr/natural=water,wood,scrub,grassland,heath", "wr/waterway=riverbank",
    "wr/landuse=forest,grass,meadow,reservoir,basin,village_green,recreation_ground",
    "wr/leisure=park,garden,pitch,playground,golf_course,nature_reserve", "wr/amenity=parking",
    "wr/area:highway",
    "n/highway=stop,give_way,traffic_signals,crossing,bus_stop,mini_roundabout",
    "n/traffic_calming", "n/railway=level_crossing", "n/traffic_sign", "n/maxspeed",
]


# Road tags stored in the `extra` column of prepared data (read by the lane-marking rule). Everything else the renderer needs
# has its own column. (An earlier version stored the ~45 tags that the legal-default speed rules read; those rules are not used.)
EXTRA_ROAD_TAGS = ("lane_markings",)

TILE_CELLS = 10          # a storage tile is TILE_CELLS x TILE_CELLS grid cells (0.5 deg): one parquet file per tile


def tile_of_cell(ix: int, iy: int):
    return ix // TILE_CELLS, iy // TILE_CELLS


def cell_of(lon: float, lat: float):
    return int(math.floor((lon + 180) / CELL_DEG)), int(math.floor((lat + 90) / CELL_DEG))


class Sink:
    SCHEMA_COLS = ("cell", "tile", "kind", "cls", "sub", "osm_id", "layer", "tunnel", "oneway", "lanes",
                   "width", "maxspeed", "direction", "extra", "minx", "miny", "maxx", "maxy", "wkb")

    def __init__(self, parts_dir: Path, flush_rows=400_000):
        self.dir, self.flush_rows, self.n_part, self.skipped = parts_dir, flush_rows, 0, 0
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buf = {c: [] for c in self.SCHEMA_COLS}

    def add(self, geom, kind, cls, sub, osm_id, layer=0, tunnel=False, oneway=0, lanes=0, width=0.0,
            maxspeed=0.0, direction=0, extra=""):
        import shapely
        if geom.is_empty:
            return
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        x0, y0, x1, y1 = geom.bounds
        ix0, iy0 = cell_of(x0, y0)
        ix1, iy1 = cell_of(x1, y1)
        if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > 2500:      # country-sized polygon: skip
            self.skipped += 1
            return
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                g = geom
                if ix0 != ix1 or iy0 != iy1:
                    box = shapely.box(ix * CELL_DEG - 180, iy * CELL_DEG - 90,
                                      (ix + 1) * CELL_DEG - 180, (iy + 1) * CELL_DEG - 90)
                    g = geom.intersection(box)
                    if g.is_empty:
                        continue
                b = g.bounds
                row = (f"{ix}_{iy}", "%d_%d" % tile_of_cell(ix, iy), kind, cls, sub, osm_id, layer, tunnel, oneway, lanes, width,
                       maxspeed, direction, extra, b[0], b[1], b[2], b[3], g.wkb)
                for c, v in zip(self.SCHEMA_COLS, row):
                    self.buf[c].append(v)
        if len(self.buf["cell"]) >= self.flush_rows:
            self.flush()

    def flush(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if not self.buf["cell"]:
            return
        t = pa.table({
            "cell": pa.array(self.buf["cell"], pa.string()), "tile": pa.array(self.buf["tile"], pa.string()),
            "kind": pa.array(self.buf["kind"], pa.string()),
            "cls": pa.array(self.buf["cls"], pa.string()), "sub": pa.array(self.buf["sub"], pa.string()),
            "osm_id": pa.array(self.buf["osm_id"], pa.int64()), "layer": pa.array(self.buf["layer"], pa.int8()),
            "tunnel": pa.array(self.buf["tunnel"], pa.bool_()), "oneway": pa.array(self.buf["oneway"], pa.int8()),
            "lanes": pa.array(self.buf["lanes"], pa.int8()), "width": pa.array(self.buf["width"], pa.float32()),
            "maxspeed": pa.array(self.buf["maxspeed"], pa.float32()),
            "direction": pa.array(self.buf["direction"], pa.int8()),
            "extra": pa.array(self.buf["extra"], pa.string()),
            **{k: pa.array(self.buf[k], pa.float64()) for k in ("minx", "miny", "maxx", "maxy")},
            "wkb": pa.array(self.buf["wkb"], pa.binary()),
        })
        pq.write_table(t, self.dir / f"part-{self.n_part:05d}.parquet", compression="zstd")
        self.n_part += 1
        self.buf = {c: [] for c in self.SCHEMA_COLS}


def way_layer(tags: dict) -> tuple[int, bool]:
    tunnel = tags.get("tunnel") in ("yes", "culvert", "building_passage")
    try:
        layer = int(tags.get("layer", "0"))
    except ValueError:
        layer = 0
    if tunnel:
        layer = -1
    elif layer == 0 and tags.get("bridge") not in (None, "no"):
        layer = 1
    return max(-1, min(layer, 3)), tunnel


def _prefilter_pbf(pbf, work, bbox, prefilter):
    """Shrink the extract with osmium-tool (clip to `bbox`, keep only the features the renderer draws). Returns the
    path to read: a file in `work`, or `pbf` itself when osmium-tool is unavailable or disabled."""
    if not (shutil.which("osmium") and prefilter):
        print("WARNING: osmium-tool not used - processing the raw file (slow / memory hungry "
              "for big countries).", file=sys.stderr)
        return pbf
    src = pbf
    if bbox:
        clipped = str(work / "clip.osm.pbf")
        subprocess.run(["osmium", "extract", "-b", ",".join(map(str, bbox)), "-s", "smart",
                        src, "-o", clipped, "--overwrite"], check=True)
        src = clipped
    filtered = str(work / "filtered.osm.pbf")
    print("osmium tags-filter ...", flush=True)
    subprocess.run(["osmium", "tags-filter", src, *TAG_FILTERS, "-o", filtered, "--overwrite"], check=True)
    return filtered


def _extract_features(src, sink, py_bbox, node_cache):
    """Classify every node, way and area of `src` and add the drawn ones to `sink`. py_bbox: clip in Python
    (only when osmium-tool could not). Returns the counts per kind."""
    import osmium
    import shapely

    def road_extra(tags):
        """JSON of the few extra road tags the renderer reads (EXTRA_ROAD_TAGS), '' if none are present."""
        e = {k: tags[k] for k in EXTRA_ROAD_TAGS if k in tags}
        return json.dumps(e, ensure_ascii=False, separators=(",", ":")) if e else ""

    def outside(x0, y0, x1, y1):
        return py_bbox is not None and (x1 < py_bbox[0] or x0 > py_bbox[2] or y1 < py_bbox[1] or y0 > py_bbox[3])

    fac = osmium.geom.WKBFactory()
    fp = osmium.FileProcessor(src)
    fp = fp.with_locations(f"sparse_file_array,{node_cache}") if node_cache else fp.with_locations()
    fp = fp.with_areas()
    n = dict(node=0, way=0, area=0, bad=0)
    for o in fp:
        try:
            if o.is_node():
                if not o.tags:
                    continue
                tags = {t.k: t.v for t in o.tags}
                c = classify_node(tags)
                if c and not outside(o.location.lon, o.location.lat, o.location.lon, o.location.lat):
                    sink.add(shapely.Point(o.location.lon, o.location.lat), *c, o.id,
                             direction=direction_of(tags))
                    n["node"] += 1
            elif o.is_way():
                tags = {t.k: t.v for t in o.tags}
                c = classify_way(tags)
                if not c:
                    continue
                geom = shapely.from_wkb(fac.create_linestring(o))
                if outside(*geom.bounds):
                    continue
                layer, tunnel = way_layer(tags)
                kind, cls, sub = c
                lanes = parse_int(tags.get("lanes")) or (
                    parse_int(tags.get("lanes:forward")) + parse_int(tags.get("lanes:backward")))
                speed = way_speed_kmh(tags)
                if not speed and cls == "living_street":
                    speed = 20.0
                sink.add(geom, kind, cls, sub, o.id, layer=layer, tunnel=tunnel,
                         oneway=oneway_of(tags, cls) if kind == "road" else 0, lanes=min(lanes, 12),
                         width=parse_width_m(tags.get("width")), maxspeed=speed,
                         extra=road_extra(tags) if kind == "road" and cls not in NON_VEHICLE else "")
                n["way"] += 1
            elif o.is_area():
                tags = {t.k: t.v for t in o.tags}
                c = classify_area(tags)
                if not c:
                    continue
                geom = shapely.from_wkb(fac.create_multipolygon(o))
                if outside(*geom.bounds):
                    continue
                sink.add(geom, *c, o.orig_id())
                n["area"] += 1
        except (RuntimeError, ValueError, osmium.InvalidLocationError):
            n["bad"] += 1
    return n


def _write_tiles(parts, out, tmp):
    """Storage layout: ONE parquet file per 0.5-degree tile (<region>/tile_<tx>_<ty>.parquet), rows sorted by cell so
    each 5 km cell is its own row group (the reader pushes a `cell == ...` filter down to row-group statistics).
    (The first version wrote one directory + one file per 5 km cell: ~10^5 inodes for a big country.)
    Reads the sink's `parts`, returns the number of tile files written to `out`; `tmp` is scratch."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    dset = ds.dataset(str(parts), format="parquet")
    if tmp.exists():
        shutil.rmtree(tmp)
    part = ds.partitioning(pa.schema([("tile", pa.string())]), flavor="hive")
    ds.write_dataset(dset, str(tmp), format="parquet", partitioning=part, max_partitions=1 << 20,
                     max_open_files=256, basename_template="part-{i}.parquet")
    out.mkdir(parents=True, exist_ok=True)
    n_tiles = 0
    for d in sorted(tmp.iterdir()):
        t = pq.read_table(str(d)).sort_by("cell")               # write_dataset leaves the partition column out
        writer = None
        cells = t["cell"].to_pylist()
        start = 0
        for i in range(1, len(cells) + 1):
            if i == len(cells) or cells[i] != cells[start]:
                chunk = t.slice(start, i - start)
                if writer is None:
                    writer = pq.ParquetWriter(str(out / f"{d.name.replace('tile=', 'tile_')}.parquet"), chunk.schema,
                                              compression="zstd")
                writer.write_table(chunk, row_group_size=len(chunk))     # exactly one row group per cell
                start = i
        writer.close()
        n_tiles += 1
    shutil.rmtree(tmp)
    try:
        tmp.parent.rmdir()
    except OSError:
        pass
    return n_tiles


def prepare(pbf: str, root: str, country: str, bbox=None, node_cache: str | None = None, prefilter=True):
    """PBF extract -> the renderer's tile files under <root>/<region>/ (one region = one country / US state)."""
    root_p = Path(root)
    out = root_p / region_dir(country)
    parts = root_p / "_parts" / region_dir(country)
    if parts.exists():
        shutil.rmtree(parts)
    if out.exists():
        shutil.rmtree(out)
    work = Path(tempfile.mkdtemp(prefix="osm_bev_"))
    try:
        src = _prefilter_pbf(pbf, work, bbox, prefilter)
        sink = Sink(parts)
        n = _extract_features(src, sink, bbox if (bbox and src == pbf) else None, node_cache)  # python clip if no osmium-tool
        sink.flush()
        print(f"{country}: {n}, skipped oversize={sink.skipped}, parts={sink.n_part}", flush=True)
        if sink.n_part == 0:
            raise SystemExit("no features extracted")

        n_tiles = _write_tiles(parts, out, root_p / "_tiles" / region_dir(country))
        print(f"{country}: {n_tiles} tile files", flush=True)
        shutil.rmtree(parts)
        try:
            parts.parent.rmdir()          # remove empty _parts/ (ignored if other countries still use it)
        except OSError:
            pass
        (out / "_info.json").write_text(json.dumps(
            dict(country=country, source=Path(pbf).name, cell_deg=CELL_DEG, tile_cells=TILE_CELLS, layout="tile_files", counts=n)))
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Stage 2: render
# --------------------------------------------------------------------------------------
class Canvas:
    """Supersampled RGB canvas in ego metres: x = right, y = up, car at the centre."""

    def __init__(self, px: int, res: float, ss: int, fill):
        import cv2
        self.cv2, self.px, self.ss = cv2, px, ss
        self.n = px * ss
        self.scale = ss / res
        self.img = np.empty((self.n, self.n, 3), np.uint8)
        self.img[:] = fill

    def _pts(self, xy):
        xy = np.asarray(xy, float)
        cr = np.column_stack([self.n / 2 + xy[:, 0] * self.scale, self.n / 2 - xy[:, 1] * self.scale])
        return np.round(cr * 16).astype(np.int32)

    def fill_xy(self, xy, color):
        self.cv2.fillPoly(self.img, [self._pts(xy)], tuple(int(c) for c in color), self.cv2.LINE_AA, 4)

    def fill_geom(self, geom, color):
        import shapely
        for g in shapely.get_parts(geom):
            if g.geom_type != "Polygon" or g.is_empty:
                continue
            rings = [np.asarray(g.exterior.coords)] + [np.asarray(r.coords) for r in g.interiors]
            self.cv2.fillPoly(self.img, [self._pts(r) for r in rings],
                              tuple(int(c) for c in color), self.cv2.LINE_AA, 4)

    def text(self, c, txt, height_m, color, halo=False, anchor="c"):
        """Upright text centred (anchor 'c') or left-aligned (anchor 'l') at ego-metre point c; height_m = cap height."""
        cv2 = self.cv2
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 1.0
        th = max(self.ss, int(round(height_m * self.scale * 0.16)))
        (w, h), _ = cv2.getTextSize(txt, font, scale, th)
        k = height_m * self.scale / h
        (w, h), _ = cv2.getTextSize(txt, font, scale * k, th)
        x = int(round(self.n / 2 + c[0] * self.scale - (w / 2 if anchor == "c" else 0)))
        y = int(round(self.n / 2 - c[1] * self.scale + h / 2))
        if halo:
            cv2.putText(self.img, txt, (x, y), font, scale * k, (255, 255, 255), th + 3 * self.ss, cv2.LINE_AA)
        cv2.putText(self.img, txt, (x, y), font, scale * k, tuple(int(v) for v in color), th, cv2.LINE_AA)

    def result(self):
        if self.ss == 1:
            return self.img
        return self.cv2.resize(self.img, (self.px, self.px), interpolation=self.cv2.INTER_AREA)


def _poly_pts(kind: str, c, u, r):
    """Vertices (ego metres) of a symbol of 'radius' r centred at c; u = unit vector it points along."""
    c, u = np.asarray(c, float), np.asarray(u, float)
    if kind == "octagon":
        a = np.radians(22.5 + 45 * np.arange(8))
        return c + r * np.column_stack([np.cos(a), np.sin(a)])
    if kind == "circle":
        a = np.radians(np.arange(0, 360, 20))
        return c + r * np.column_stack([np.cos(a), np.sin(a)])
    if kind == "triangle":   # tip along u
        n = np.array([-u[1], u[0]])
        return np.array([c + u * r, c - u * r * 0.7 + n * r * 0.9, c - u * r * 0.7 - n * r * 0.9])
    if kind == "square":
        n = np.array([-u[1], u[0]])
        return np.array([c + (u + n) * r * 0.7, c + (u - n) * r * 0.7, c + (-u - n) * r * 0.7, c + (-u + n) * r * 0.7])
    raise ValueError(kind)


def _unit(v):
    v = np.asarray(v, float)
    nrm = np.hypot(*v)
    return v / nrm if nrm > 1e-9 else np.array([0.0, 1.0])


def _ngon(c, r, n, rot_deg=0.0):
    a = np.radians(rot_deg + 360.0 / n * np.arange(n))
    return np.asarray(c, float) + r * np.column_stack([np.cos(a), np.sin(a)])


def _tri(c, r, down):
    """Equilateral triangle of circum-radius r; point down (give way) or up (warning). Always upright."""
    k = -1.0 if down else 1.0
    return np.asarray(c, float) + r * np.array([[0.0, k], [0.87, -0.5 * k], [-0.87, -0.5 * k]])


def _rect(c, w, h):
    c = np.asarray(c, float)
    return c + np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])


def _rrect(c, w, h, deg):
    a = math.radians(deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.asarray(c, float) + _rect((0, 0), w, h) @ rot.T


SIGN_R = 7.0     # nominal sign radius in metres (~14 px at 0.5 m/px): big enough to read, small enough not to clutter


TRAJ_COL = (150, 0, 255)       # the ego chevron's violet: the path IS the ego's future; no map feature uses this colour
TRAJ_WIDTH_M = 1.9             # about one car width


def draw_trajectory(cv, xy, ticks=()):
    """Future ego path (ego-metre points, first = car): white halo + violet core one car wide, a dot every `ticks`
    index (one per second), and an arrowhead at the far end. Drawn above roads and symbols, under the speed labels and ego."""
    import shapely
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return
    line = shapely.LineString(xy)
    cv.fill_geom(line.buffer(TRAJ_WIDTH_M / 2 + 0.8, cap_style="round"), (255, 255, 255))
    cv.fill_geom(line.buffer(TRAJ_WIDTH_M / 2, cap_style="round"), TRAJ_COL)
    for k in ticks:
        cv.fill_xy(_ngon(xy[k], 2.7, 16), (255, 255, 255))
        cv.fill_xy(_ngon(xy[k], 1.9, 16), TRAJ_COL)
    t = _unit(xy[-1] - xy[max(0, len(xy) - 6)])
    n = np.array([-t[1], t[0]])
    for L, W, col in ((5.2, 3.6, (255, 255, 255)), (4.2, 2.7, TRAJ_COL)):
        cv.fill_xy([xy[-1] + t * L, xy[-1] - t * 1.2 + n * W, xy[-1] - t * 1.2 - n * W], col)


DRAWN = {"crossing", "bus_stop", "level_crossing", "mini_roundabout", "calming", "give_way", "stop", "signals"}


def draw_symbol(cv, cls, c, speed=0.0, sub=""):
    """Upright pictogram at ego-metre point c (always screen-upright, never rotated). One visual vocabulary for all
    countries, Swiss/Vienna-convention look: stop = red octagon + STOP, give way = inverted red-bordered triangle,
    signals = 3-lamp traffic light, bus stop = yellow disc + H, level crossing = red St Andrew's cross, mini roundabout =
    blue disc with arrows, calming = speed-bump triangle. Only symbols derived from core OSM tags (plus the stop / give-way
    sign codes validated against them) are drawn."""
    R, o, w = SIGN_R, COL["outline"], (255, 255, 255)
    if cls == "stop":
        Rs = R * 1.3      # a bit larger than the round signs so the octagon + STOP survive downsampling
        cv.fill_xy(_ngon(c, Rs * 1.08, 8, 22.5), o)
        cv.fill_xy(_ngon(c, Rs, 8, 22.5), w)
        cv.fill_xy(_ngon(c, Rs * 0.88, 8, 22.5), COL["stop"])
        cv.text(c, "STOP", Rs * 0.34, w)
    elif cls == "give_way":
        cv.fill_xy(_tri(c, R * 1.25, True), o)
        cv.fill_xy(_tri(c, R * 1.17, True), COL["sign_speed_ring"])
        cv.fill_xy(_tri(c, R * 0.68, True), w)
    elif cls == "signals":
        cv.fill_xy(_rect(c, R * 0.95, R * 2.5), o)
        for k, col in enumerate(((220, 30, 30), (250, 190, 0), (0, 170, 60))):
            cv.fill_xy(_ngon(np.asarray(c) + [0, (1 - k) * R * 0.78], R * 0.3, 14), col)
    elif cls == "bus_stop":
        cv.fill_xy(_ngon(c, R * 0.8, 20), o)
        cv.fill_xy(_ngon(c, R * 0.72, 20), COL["bus_stop"])
        cv.text(c, "H", R * 0.55, o)
    elif cls == "level_crossing":    # St Andrew's cross
        cv.fill_xy(_rect(c, R * 1.7, R * 1.7), o)
        cv.fill_xy(_rect(c, R * 1.55, R * 1.55), w)
        for ang in (45, -45):
            cv.fill_xy(_rrect(c, R * 1.7, R * 0.36, ang), COL["sign_speed_ring"])
    elif cls == "mini_roundabout":   # blue disc with three chasing arrows
        blue = (0, 90, 170)
        cv.fill_xy(_ngon(c, R * 0.9, 24), o)
        cv.fill_xy(_ngon(c, R * 0.82, 24), blue)
        for ang in (90, 210, 330):
            a_ = math.radians(ang)
            pos = np.asarray(c) + 0.5 * R * np.array([math.cos(a_), math.sin(a_)])
            t_ = np.array([-math.sin(a_), math.cos(a_)])
            n_ = np.array([math.cos(a_), math.sin(a_)])
            cv.fill_xy([pos + t_ * R * 0.42, pos - t_ * R * 0.2 + n_ * R * 0.3, pos - t_ * R * 0.2 - n_ * R * 0.3], w)
    elif cls == "calming":           # speed-bump warning: triangle with a black hump
        cv.fill_xy(_tri(c, R * 1.3, False), o)
        cv.fill_xy(_tri(c, R * 1.22, False), COL["sign_speed_ring"])
        cv.fill_xy(_tri(c, R * 0.72, False), w)
        th = np.radians(np.linspace(0, 180, 12))
        hump = np.asarray(c, float) + np.column_stack([R * 0.5 * np.cos(th), R * 0.42 * np.sin(th) - R * 0.32])
        cv.fill_xy(hump, o)


def link_roads(roads):
    """Endpoint adjacency: sets r['_ends'] and returns {endpoint key: [roads touching there]} (0.5 m grid)."""
    touch = {}
    for r in roads:
        out = []
        for ln in shapely_parts(r["g"]):
            if ln.geom_type == "LineString" and len(ln.coords) >= 2:
                out += [ln.coords[0], ln.coords[-1]]
        r["_ends"] = [(round(x * 2), round(y * 2)) for x, y in out]
        for k in r["_ends"]:
            touch.setdefault(k, []).append(r)
    return touch


def infer_road_speeds(roads):
    """Sets r['speed'] (km/h) and r['src'] in {'map', 'inferred', None} on every road dict.

    1. 'map'      : the limit is tagged in OSM.
    2. 'inferred' : a same-class road touching this one at an endpoint has a limit (limits of a road continue; lowest
                    neighbour wins; repeated so a chain of untagged ways gets filled).
    Everything else stays without a number: statutory defaults are not used (memory/osm-bev-tag-reliability.md)."""
    for r in roads:
        r["speed"], r["src"] = float(r["maxspeed"]), ("map" if r["maxspeed"] > 0 else None)
    touch = link_roads(roads)
    for _ in range(4):
        changed = False
        for r in roads:
            if r["src"] or r["cls"] in NON_VEHICLE or r["cls"] == "service":
                continue
            vals = [o["speed"] for k in r["_ends"] for o in touch[k] if o is not r and o["speed"] > 0
                    and o["cls"] == r["cls"]]
            if vals:
                r["speed"], r["src"], changed = min(vals), "inferred", True
        if not changed:
            break


MARKED_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                  "secondary_link", "tertiary", "tertiary_link"}
CENTRE_LINE_CLASSES = {"trunk", "primary", "secondary"}     # two-way roads of these classes normally have one


def infer_road_lanes(roads, agree=True):
    """Sets r['nl'] (total lanes, both directions; 0 = unknown) and r['lsrc'] in {'map', 'neighbour', None}.
    1. 'map'       : tagged (`lanes`, or lanes:forward + lanes:backward).
    2. 'neighbour' : touching roads of the same class and same one-way-ness that have a lane count (counts continue
                     along a road; repeated so chains get filled). With agree=True the count is only taken when ALL such
                     neighbours report the same number; agree=False takes the most common one.
    Everything else stays unknown (nl = 0): no default count (memory/osm-bev-tag-reliability.md)."""
    touch = link_roads(roads)
    for r in roads:
        r["nl"] = int(r["lanes"]) if r["lanes"] > 0 else 0
        r["lsrc"] = "map" if r["nl"] else None
        r["_x"] = json.loads(r["extra"]) if r.get("extra") else {}
    for _ in range(3):
        changed = False
        new = {}
        for r in roads:
            if r["nl"] or r["cls"] in NON_VEHICLE or r["cls"] == "service":
                continue
            vals = [o["nl"] for k in r["_ends"] for o in touch[k] if o is not r and o["nl"] > 0
                    and o["cls"] == r["cls"] and bool(o["oneway"]) == bool(r["oneway"])]
            if not vals:
                continue
            cnt = collections.Counter(vals).most_common()
            if agree and len(cnt) > 1:
                continue
            new[id(r)] = (r, min(c for c, n in cnt if n == cnt[0][1]))
        for r, nl in new.values():                     # apply after the pass: no order dependence within a pass
            r["nl"], r["lsrc"], changed = nl, "neighbour", True
        if not changed:
            break


def lane_marks(r):
    """Markings to draw inside one road: [(lateral offset from the centre line in m, solid?)].
    Only what such roads normally have: dashed lane dividers on multi-lane roads, a solid centre line on two-way roads
    of trunk/primary/secondary class (or where OSM says lane_markings=yes). Unmarked streets stay unmarked."""
    cls, nl, w = r["cls"], r.get("nl", 0), r["w"]
    if cls in NON_VEHICLE or cls in ("service", "living_street") or nl < 2 or w / nl < 2.6:
        return []
    lm = r["_x"].get("lane_markings")
    if lm == "no":
        return []
    marked = lm == "yes" or cls in MARKED_CLASSES
    lw = w / nl
    offs = [-w / 2 + k * lw for k in range(1, nl)]
    if r["oneway"]:
        return [(o, False) for o in offs] if marked else []
    out = []
    for o in offs:
        if abs(o) < 1e-6:                                        # even count, two-way: the direction separator
            if cls in CENTRE_LINE_CLASSES or lm == "yes" or (nl >= 4 and r["lsrc"] == "map"):
                out.append((0.0, True))
        elif marked and nl >= 3:
            out.append((o, False))
    return out


def draw_lane_marks(cv, r, fill, half):
    import shapely
    from shapely.ops import substring
    marks = lane_marks(r)
    if not marks:
        return
    box = shapely.box(-half - 30, -half - 30, half + 30, half + 30)
    col = tuple(int(v * 0.62) for v in fill)                     # darker than the road fill: visible on every colour
    for part in shapely_parts(r["g"].intersection(box)):
        if part.geom_type != "LineString" or part.length < 6:
            continue
        for off, solid in marks:
            try:
                oc = part.offset_curve(off) if off else part
            except Exception:
                continue
            for seg in shapely_parts(oc):
                if seg.geom_type != "LineString" or seg.length < 1:
                    continue
                if solid:
                    cv.fill_geom(seg.buffer(0.3, cap_style="flat"), col)
                else:
                    for s0 in np.arange(2.0, seg.length - 1.0, 9.0):
                        cv.fill_geom(substring(seg, s0, min(s0 + 3.0, seg.length)).buffer(0.28, cap_style="flat"), col)


def shapely_parts(g):
    import shapely
    return shapely.get_parts(g)


class _Legend:
    """Legend sheet: three columns of labelled samples, drawn with the renderer's own palette and drawing functions."""
    PX = 2560
    COLS = [-122.0, -52.0, 18.0]      # left edge (m) of the 3 columns (70 m apart)
    TOP = 122.0

    def __init__(self, res):
        self.res = res
        self.cv = Canvas(self.PX, res, 2, BG)
        self.ink = COL["outline"]
        self.cursor = [self.TOP] * 3

    @staticmethod
    def P(x, y):
        return np.array([x, y], float)

    def head(self, ci, title):
        import shapely
        cv, cols, cursor = self.cv, self.COLS, self.cursor
        cursor[ci] -= 2.0
        cv.text(self.P(cols[ci], cursor[ci]), title, 3.4, self.ink, anchor="l")
        cv.fill_geom(shapely.box(cols[ci], cursor[ci] - 3.4, cols[ci] + 62, cursor[ci] - 3.0), (150, 150, 150))
        cursor[ci] -= 8.0

    def row(self, ci, label, drawer, pitch=9.0, sw=12.0):
        y = self.cursor[ci]
        drawer(self.COLS[ci], y, sw)
        self.cv.text(self.P(self.COLS[ci] + sw + 3.0, y), label, 2.4, self.ink, anchor="l")
        self.cursor[ci] -= pitch

    def note(self, ci, lines, size=1.8, colour=(90, 90, 90), dy=0.0):
        for ln in lines:
            self.cv.text(self.P(self.COLS[ci], self.cursor[ci] + dy), ln, size, colour, anchor="l")
            self.cursor[ci] -= 3.6

    # drawers: functions (x, y, sample_width) that draw one sample
    def road_drawer(self, cls, tunnel=False):
        import shapely
        cv = self.cv
        w = min(ROADS[cls][1], 7.0)
        col = np.array(ROADS[cls][2], float)
        if tunnel:
            col = 0.5 * col + 0.5 * np.array(BG)

        def d(x, y, sw):
            ln = shapely.LineString([(x, y), (x + sw, y)])
            cv.fill_geom(ln.buffer(w / 2 + 0.45, cap_style="flat"), COL["casing"])
            cv.fill_geom(ln.buffer(w / 2, cap_style="flat"), col)
        return d

    def area_drawer(self, col, outline=None):
        import shapely
        cv = self.cv

        def d(x, y, sw):
            b = shapely.box(x, y - 2.8, x + sw, y + 2.8)
            if outline:
                cv.fill_geom(b.buffer(0.4), outline)
            cv.fill_geom(b, col)
        return d

    def stripes_drawer(self, col):
        import shapely
        cv = self.cv

        def d(x, y, sw):
            for k in range(4):
                cv.fill_geom(shapely.box(x + 1.0 + 3.0 * k, y - 3.0, x + 2.6 + 3.0 * k, y + 3.0), col)
        return d

    def roads(self):
        """Column 0: road classes."""
        self.head(0, "ROADS (fill colour = class)")
        for lab, cls in [("Motorway", "motorway"), ("Motorway ramp", "motorway_link"), ("Trunk road", "trunk"),
                         ("Primary road", "primary"), ("Secondary road", "secondary"), ("Tertiary road", "tertiary"),
                         ("Residential / unclassified", "residential"), ("Living street", "living_street"), ("Service road", "service"),
                         ("Pedestrian street", "pedestrian"), ("Track", "track"), ("Cycleway", "cycleway"), ("Footway", "footway"),
                         ("Path", "path"), ("Steps", "steps"), ("Bridleway", "bridleway")]:
            self.row(0, lab, self.road_drawer(cls))
        self.row(0, "Tunnel (class colour, faded)", self.road_drawer("primary", tunnel=True))
        self.cursor[0] -= 1.0
        self.note(0, ("Width: tagged width, else lanes x 3 m,", "else the class default. A dark grey", "outline (casing) separates roads."))

    def areas_and_markings(self):
        """Column 1: areas, rail, road markings and labels."""
        import shapely
        cv, P = self.cv, self.P
        self.head(1, "AREAS AND LINES")
        for lab, col, ol in [("Background / unmapped", BG, (170, 170, 170)), ("Park, forest, grass", COL["green"], None),
                             ("Water", COL["water"], None), ("Parking area", COL["parking"], None),
                             ("Pedestrian area", COL["pedestrian"], None), ("Traffic island", COL["island"], None),
                             ("Building", COL["building"], None)]:
            self.row(1, lab, self.area_drawer(col, ol))

        def rail(x, y, sw):
            cv.fill_geom(shapely.LineString([(x, y), (x + sw, y)]).buffer(0.8, cap_style="flat"), COL["rail"])
        self.row(1, "Railway / tram line", rail)
        self.cursor[1] -= 2.0
        self.head(1, "ROAD MARKINGS AND LABELS")

        def oneway(x, y, sw):
            for k in range(3):
                q = P(x + 2.5 + 4.0 * k, y)
                cv.fill_xy([q + P(2.6, 0), q + P(-1.8, 1.3), q + P(-1.8, -1.3)], COL["arrow"])
        self.row(1, "One-way arrow", oneway)

        def dashed(x, y, sw):
            cv.fill_geom(shapely.box(x, y - 3.4, x + sw, y + 3.4), ROADS["secondary"][2])
            for k in range(2):
                col = tuple(int(v * 0.62) for v in ROADS["secondary"][2])
                cv.fill_geom(shapely.box(x + 1.0 + 6.0 * k, y - 0.28, x + 4.0 + 6.0 * k, y + 0.28), col)
        self.row(1, "Lane divider (dashed)", dashed)

        def centre(x, y, sw):
            cv.fill_geom(shapely.box(x, y - 3.4, x + sw, y + 3.4), ROADS["secondary"][2])
            cv.fill_geom(shapely.box(x, y - 0.3, x + sw, y + 0.3), tuple(int(v * 0.62) for v in ROADS["secondary"][2]))
        self.row(1, "Centre line, two-way road", centre)
        self.row(1, "Speed limit, km/h (on road)", lambda x, y, sw: cv.text(P(x + sw / 2, y), "50", 3.4, COL["speed_map"], halo=True))
        self.row(1, "Crossing, with traffic signals", self.stripes_drawer(COL["cross_signals"]))
        self.row(1, "Crossing, marked / zebra", self.stripes_drawer(COL["cross_marked"]))
        self.row(1, "Crossing, other / unmarked", self.stripes_drawer(COL["cross_other"]))
        self.cursor[1] -= 1.0
        self.note(1, ("No number on a road = limit not", "mapped (nor copied from a touching", "road of the same class)."))

    def symbols_and_ego(self):
        """Column 2: point symbols, the ego marker and path, notes."""
        cv, P, cols, cursor = self.cv, self.P, self.COLS, self.cursor
        self.head(2, "SYMBOLS (always upright)")
        cursor[2] -= 6.0
        for lab, cls in [("Stop sign", "stop"), ("Give way", "give_way"), ("Traffic signals", "signals"), ("Bus stop", "bus_stop"),
                         ("Level / tram crossing", "level_crossing"), ("Mini roundabout", "mini_roundabout"),
                         ("Speed bump, hump, table", "calming")]:
            self.row(2, lab, lambda x, y, sw, c=cls: draw_symbol(cv, c, P(x + sw / 2, y), 0.0), pitch=18.0, sw=16.0)
        cursor[2] -= 1.0
        self.head(2, "EGO VEHICLE")
        cursor[2] -= 5.0

        def chev(x, y, sw):
            h, n = P(0, 1), P(-1, 0)
            c0 = P(x + sw / 2, y)
            for L, W, col in ((13.0, 8.4, COL["outline"]), (11.6, 7.2, COL["ego_ring"]), (9.8, 5.6, COL["ego"])):
                L, W = L * 0.5, W * 0.5
                cv.fill_xy([c0 + h * L, c0 - h * L * 0.7 + n * W, c0 - h * L * 0.25, c0 - h * L * 0.7 - n * W], col)
        self.row(2, "Ego position and heading", chev, pitch=12.0)
        y0 = cursor[2] - 2.0
        t = np.linspace(0, 1, 51)
        path = np.column_stack([cols[2] + 3.0 + 36.0 * t, y0 - 10.0 * np.sin(t * np.pi * 0.9)])
        draw_trajectory(cv, path, ticks=(10, 20, 30, 40, 50))
        cv.text(P(cols[2], y0 - 15.0), "Ego path of the next 5 s (drawn above", 1.8, (90, 90, 90), anchor="l")
        cv.text(P(cols[2], y0 - 18.5), "roads and symbols): one dot per second,", 1.8, (90, 90, 90), anchor="l")
        cv.text(P(cols[2], y0 - 22.0), "arrow = where the car is after 5 s.", 1.8, (90, 90, 90), anchor="l")
        cursor[2] = y0 - 26.0
        self.head(2, "NOTES")
        self.note(2, ("Symbols and text here are 5x larger", "than in the images (0.5 m/px, 768 px).",
                      "Map orientation: heading up, ego at the", "image centre; right-hand traffic unless",
                      "the region is Ireland or Japan (left).", "Absence of a sign / limit / lane marking",
                      "can mean 'not mapped', not 'none'."), size=1.9, colour=(60, 60, 60), dy=2.0)

    def image(self):
        """RGB image, cropped to the content."""
        img = self.cv.result()
        low = min(self.cursor) - 6.0
        rows = int(min(self.PX, (self.TOP + 6.0 - low) / self.res))
        return img[: max(rows, 512)]


def draw_legend(res=0.1):
    """Legend of everything the map images show, drawn with the renderer's own palette and drawing functions so it cannot
    drift from the maps. Enlarged 5x vs the 0.5 m/px of the maps (res 0.1 m/px). Returns an RGB uint8 image."""
    legend = _Legend(res)
    legend.roads()
    legend.areas_and_markings()
    legend.symbols_and_ego()
    return legend.image()


class MapRenderer:
    def __init__(self, root, res=RES_M_PER_PX, v_kmh=V_MAX_KMH, t_s=HORIZON_S, safety=SAFETY,
                 stride=STRIDE, supersample=SUPERSAMPLE, heading_up=True):
        self.root = Path(root)
        self.res, self.ss, self.heading_up = res, supersample, heading_up
        self.px, self.half = crop_geometry(res, v_kmh, t_s, safety, stride)
        self.countries = sorted(d.name for d in self.root.iterdir()
                                if d.is_dir() and not d.name.startswith("_"))
        self.regions = Regions(self.root) if (self.root / "_regions.json").exists() else None

    # ---- data access -------------------------------------------------------------------
    @lru_cache(maxsize=256)
    def _cell(self, country: str, cell: str):
        import pyarrow.parquet as pq
        ix, iy = (int(v) for v in cell.split("_"))
        tf = self.root / country / ("tile_%d_%d.parquet" % tile_of_cell(ix, iy))
        if tf.exists():                                              # current layout: row group per cell inside a tile file
            t = pq.read_table(str(tf), filters=[("cell", "=", cell)])
            return t if t.num_rows else None
        d = self.root / country / f"cell={cell}"                     # legacy layout: one directory per cell
        return pq.read_table(str(d)) if d.exists() else None

    def _query(self, lat, lon, countries):
        r = self.half * math.sqrt(2) + 25.0           # a rotated square fits in this disc
        dlat = r / 111_320.0
        dlon = r / (111_320.0 * max(math.cos(math.radians(lat)), 0.05))
        x0, x1, y0, y1 = lon - dlon, lon + dlon, lat - dlat, lat + dlat
        (ix0, iy0), (ix1, iy1) = cell_of(x0, y0), cell_of(x1, y1)
        rows, found = [], []
        for ctry in countries:
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    t = self._cell(ctry, f"{ix}_{iy}")
                    if t is None:
                        continue
                    if ctry not in found:
                        found.append(ctry)
                    m = ((t["maxx"].to_numpy() >= x0) & (t["minx"].to_numpy() <= x1) &
                         (t["maxy"].to_numpy() >= y0) & (t["miny"].to_numpy() <= y1))
                    if m.any():
                        rows.extend(t.filter(m).to_pylist())
        return rows, found

    # ---- rendering ---------------------------------------------------------------------
    def _regions_near(self, lat, lon, country):
        """(region dirs to read, the region of the crop centre or None)."""
        if country:
            return [country], None
        if self.regions is None:
            return self.countries, None
        near = self.regions.near(lat, lon, radius_m=self.half * math.sqrt(2) + 25.0)
        countries = [region_dir(k) for k in near if region_dir(k) in self.countries]
        centre_region = next((region_dir(k) for k in self.regions.near(lat, lon) if region_dir(k) in self.countries), None)
        return countries, centre_region

    @staticmethod
    def _local_frame(lat, lon, heading_rad):
        """Function mapping (lon, lat) arrays to metres in the frame centred on the car and rotated so the heading
        points up (x right, y up): aeqd projection, then rotation."""
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs",
                                  always_xy=True)
        cb, sb = math.cos(heading_rad), math.sin(heading_rad)

        def to_local(xy):
            e, n = tr.transform(xy[:, 0], xy[:, 1])
            return np.column_stack([e * cb - n * sb, e * sb + n * cb])
        return to_local

    @staticmethod
    def _group_rows(rows):
        """Rows by kind: (areas by class, roads, crossing ways, rails, nodes)."""
        areas = {k: [] for k in AREA_PRIORITY}
        roads, crossings_way, rails, nodes = [], [], [], []
        for r in rows:
            k = r["kind"]
            if k == "area":
                areas.setdefault(r["cls"], []).append(r)
            elif k == "road":
                roads.append(r)
            elif k == "crossing_way":
                crossings_way.append(r)
            elif k == "rail":
                rails.append(r)
            else:
                nodes.append(r)
        return areas, roads, crossings_way, rails, nodes

    @staticmethod
    def _road_width(r):
        prio, dflt, _ = ROADS[r["cls"]]
        if r["width"] > 0:
            return float(r["width"])
        if r["lanes"] > 0 and r["cls"] not in NON_VEHICLE:
            return max(2.5, r["lanes"] * (3.25 if r["oneway"] else 3.0))
        return max(1.0, dflt * 0.62) if (r["oneway"] and r["cls"] not in NON_VEHICLE) else dflt

    def _prepare_roads(self, roads):
        """Width, polygon and drawing order of every road, lanes inferred. Returns the sorted layer levels."""
        for r in roads:
            r["w"] = self._road_width(r)
            r["poly"] = r["g"].buffer(r["w"] / 2, cap_style="round", quad_segs=3)
            r["level"] = r["layer"]
        roads.sort(key=lambda r: (r["level"], ROADS[r["cls"]][0]))
        levels = sorted({r["level"] for r in roads})
        infer_road_lanes(roads)
        return levels

    def _draw_base(self, cv, areas, roads, rails, levels):
        """Areas, then roads layer by layer (casings first -> clean junctions, lane marks right after their fill), rails."""
        for k in AREA_PRIORITY:
            for r in areas.get(k, []):
                if k == "roadarea" and r["sub"] not in ROADS:
                    continue                            # rows from older prepared data with a non-road subtype
                color = COL[k] if k != "roadarea" else ROADS[r["sub"]][2]
                cv.fill_geom(r["g"], color)
        for lv in levels:
            grp = [r for r in roads if r["level"] == lv]
            for r in grp:
                if r["cls"] in NON_VEHICLE and r["w"] < 2.5:
                    continue
                cv.fill_geom(r["g"].buffer(r["w"] / 2 + 0.45, cap_style="round", quad_segs=3), COL["casing"])
            for r in grp:
                col = np.array(ROADS[r["cls"]][2], float)
                if r["tunnel"]:
                    col = 0.5 * col + 0.5 * np.array(BG)
                cv.fill_geom(r["poly"], col)
                draw_lane_marks(cv, r, col, self.half)
        for r in rails:
            cv.fill_geom(r["g"].buffer(0.8, cap_style="flat"), COL["rail"])

    def _draw_oneway_arrows(self, cv, roads):
        import shapely
        for r in roads:
            if not r["oneway"] or r["cls"] in NON_VEHICLE or r["w"] < 3.0:
                continue
            for line in shapely.get_parts(r["g"]):
                if line.geom_type != "LineString" or line.length < 8:
                    continue
                for d in np.arange(10.0, line.length - 4.0, 22.0):
                    p = np.array(line.interpolate(d).coords[0])
                    if np.abs(p).max() > self.half + 5:
                        continue
                    p2 = np.array(line.interpolate(min(d + 1.0, line.length)).coords[0])
                    t = _unit(p2 - p) * (1 if r["oneway"] > 0 else -1)
                    n = np.array([-t[1], t[0]])
                    # multi-lane one-way road with lane dividers: one arrow per lane, like painted arrows
                    lw = r["w"] / r["nl"] if r["nl"] >= 2 and lane_marks(r) else 0.0
                    for off in ([(k + 0.5) * lw - r["w"] / 2 for k in range(r["nl"])] if lw else [0.0]):
                        q = p + n * off
                        hw = min(1.3, 0.36 * lw) if lw else 1.3          # arrow narrower than its lane
                        L = 2.6 if not lw else 2.2
                        cv.fill_xy([q + t * L, q - t * (L * 0.7) + n * hw, q - t * (L * 0.7) - n * hw], COL["arrow"])

    def _draw_nodes(self, cv, nodes, roads, crossings_way, side):
        """Crossing stripes and the point symbols; symbols need the tangent of the nearest vehicle road."""
        import shapely
        from shapely.strtree import STRtree

        veh_parts = [(p, r) for r in roads if r["cls"] not in NON_VEHICLE
                     for p in shapely.get_parts(r["g"]) if p.geom_type == "LineString"]
        tree = STRtree([p for p, _ in veh_parts]) if veh_parts else None

        def road_at(pt, max_d=10.0):
            if tree is None:
                return None
            idx = tree.query_nearest(pt, max_distance=max_d, all_matches=False)
            if len(idx) == 0:
                return None
            line, r = veh_parts[int(idx[0])]
            d = line.project(pt)
            a = np.array(line.interpolate(max(d - 1.0, 0)).coords[0])
            c = np.array(line.interpolate(min(d + 1.0, line.length)).coords[0])
            return _unit(c - a), r["w"]

        for r in crossings_way:
            cv.fill_geom(r["g"].buffer(1.6, cap_style="flat"), COL[f"cross_{r['sub']}"])

        ORDER = ["crossing", "bus_stop", "level_crossing", "mini_roundabout", "calming", "give_way", "stop", "signals"]
        for r in sorted(nodes, key=lambda r: ORDER.index(r["cls"]) if r["cls"] in ORDER else 0):
            pt = r["g"]
            if abs(pt.x) > self.half + 10 or abs(pt.y) > self.half + 10:
                continue
            c = np.array([pt.x, pt.y])
            cls = r["cls"]
            ra = road_at(pt)
            t, hw = (ra[0], ra[1] / 2) if ra else (np.array([0.0, 1.0]), 2.5)
            if cls == "crossing":
                if ra is None:
                    continue
                n = np.array([-t[1], t[0]])
                L = hw + 0.6
                cv.fill_xy([c + n * L + t * 1.5, c + n * L - t * 1.5, c - n * L - t * 1.5, c - n * L + t * 1.5],
                           COL[f"cross_{r['sub']}"])
                continue
            # travel direction + roadside placement
            d = r["direction"]
            u = t if d in (0, 1, 2) else -t
            if d in (1, -1) and ra is not None:
                n_right = np.array([u[1], -u[0]])
                c = c + n_right * (hw + SIGN_R) * (1 if side == "right" else -1)
            if cls not in DRAWN or (cls == "calming" and not r["sub"]):
                continue                         # legacy rows (older prepared data): classes that are no longer drawn
            draw_symbol(cv, cls, c, r["maxspeed"], r["sub"])

    def _draw_speed_labels(self, cv, roads, nodes):
        """Speed-limit numbers along the roads, sliding along a road to dodge symbols and other labels."""
        import shapely
        infer_road_speeds(roads)
        placed = []
        sym_pts = np.array([[n["g"].x, n["g"].y] for n in nodes if n["cls"] in DRAWN and n["cls"] not in ("crossing", "calming")
                            and abs(n["g"].x) < self.half + 10 and abs(n["g"].y) < self.half + 10]).reshape(-1, 2)
        for r in sorted(roads, key=lambda r: -ROADS[r["cls"]][0]):
            if r["speed"] <= 0 or r["cls"] in NON_VEHICLE or r["w"] < 3.4:
                continue
            col = COL["speed_map"] if r["src"] == "map" else COL["speed_inferred"]
            for line in shapely.get_parts(r["g"]):
                if line.geom_type != "LineString" or line.length < 24:
                    continue
                dists = [line.length / 2] if line.length < 100 else list(np.arange(30.0, line.length - 20.0, 90.0))
                for d in dists:
                    for shift in (0.0, 14.0, -14.0, 28.0, -28.0):
                        dd = d + shift
                        if not 4.0 < dd < line.length - 4.0:
                            continue
                        pt = np.array(line.interpolate(dd).coords[0])
                        if np.abs(pt).max() > self.half - 6 or any(np.hypot(*(pt - q)) < 42 for q in placed):
                            continue
                        if len(sym_pts) and np.min(np.hypot(*(sym_pts - pt).T)) < 11.0:
                            continue
                        placed.append(pt)
                        cv.text(pt, str(int(round(r["speed"]))), 5.4, col, halo=True)
                        break

    def _draw_ego(self, cv, heading_deg, has_path, ego_scale):
        """Heading chevron: ~26 m long on its own so the ego pose is obvious; half that when a future path is drawn
        (a full-size chevron would hide the first second or two of a short path)."""
        h = np.array([0.0, 1.0]) if self.heading_up else np.array([math.sin(math.radians(heading_deg)),
                                                                   math.cos(math.radians(heading_deg))])
        n = np.array([-h[1], h[0]])
        k = ego_scale if ego_scale is not None else (0.5 if has_path else 1.0)
        for L, W, col in ((13.0, 8.4, COL["outline"]), (11.6, 7.2, COL["ego_ring"]), (9.8, 5.6, COL["ego"])):
            L, W = L * k, W * k
            cv.fill_xy([h * L, -h * L * 0.7 + n * W, -h * L * 0.25, -h * L * 0.7 - n * W], col)

    def render(self, lat, lon, heading_deg=0.0, country=None, ego=False, oneway_arrows=True, trajectory=None,
               ego_scale=None):
        """Render the ego-centred crop at (lat, lon) heading `heading_deg` (compass, clockwise from north).
        trajectory: dict(lonlat=(N, 2) array, ticks=[indices]) of the future ego path to draw on top.
        Returns dict(image=RGB array, meta=..., scene=...); `scene` is the classified, local-frame feature set the image was
        drawn from (roads with inferred lanes / speeds, nodes, crossing ways), for osm_route_text.py."""
        import shapely

        countries, centre_region = self._regions_near(lat, lon, country)
        rows, found = self._query(lat, lon, countries)
        country = country or centre_region or (found[0] if len(found) == 1 else None)
        side = drive_side(country)

        to_local = self._local_frame(lat, lon, math.radians(heading_deg) if self.heading_up else 0.0)
        geoms = shapely.transform(shapely.from_wkb(np.array([r["wkb"] for r in rows], dtype=object)), to_local) \
            if rows else []
        for r, g in zip(rows, geoms):
            r["g"] = g
        areas, roads, crossings_way, rails, nodes = self._group_rows(rows)
        levels = self._prepare_roads(roads)

        cv = Canvas(self.px, self.res, self.ss, BG)
        self._draw_base(cv, areas, roads, rails, levels)
        if oneway_arrows:
            self._draw_oneway_arrows(cv, roads)
        self._draw_nodes(cv, nodes, roads, crossings_way, side)
        # the future ego path is drawn above roads, lane marks, arrows AND the symbols (at dense junctions it would
        # otherwise vanish under traffic lights), below the labels and the ego marker
        if trajectory is not None:
            draw_trajectory(cv, to_local(np.asarray(trajectory["lonlat"], float)), trajectory.get("ticks", ()))
        self._draw_speed_labels(cv, roads, nodes)  # above the path so the limit along the route stays readable
        if ego:
            self._draw_ego(cv, heading_deg, trajectory is not None, ego_scale)

        return dict(image=cv.result(), meta=dict(
            lat=lat, lon=lon, heading_deg=heading_deg, heading_up=self.heading_up, country=country,
            countries_with_data=found, drive_side=side, meters_per_px=self.res, px=self.px,
            half_extent_m=self.half, n_features=len(rows)),
            scene=dict(roads=roads, nodes=nodes, crossings_way=crossings_way, to_local=to_local, side=side,
                       region=country))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _save(path, rgb):
    import cv2
    cv2.imwrite(str(path), rgb[:, :, ::-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="print crop geometry and country list")
    p.add_argument("--res", type=float, default=RES_M_PER_PX)

    p = sub.add_parser("prepare", help="PBF -> partitioned GeoParquet")
    p.add_argument("--country", required=True, help="region key, e.g. Japan or \"United States/Nevada\"")
    p.add_argument("--pbf", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat (needs osmium-tool)")
    p.add_argument("--node-cache", help="file for pyosmium's sparse_file_array (large countries)")
    p.add_argument("--no-prefilter", action="store_true")

    p = sub.add_parser("regions", help="download Geofabrik boundary polygons -> <root>/_regions.json")
    p.add_argument("--root", required=True)

    p = sub.add_parser("legend", help="render the legend of all map features (colours, symbols, markings)")
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=float, default=0.25, help="0.5 = true size in the model input")

    p = sub.add_parser("render", help="render one crop")
    p.add_argument("--root", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--heading", type=float, default=0.0, help="compass bearing, degrees clockwise from north")
    p.add_argument("--country", help="force a region dir; default: looked up from lat/lon via _regions.json")
    p.add_argument("--out", required=True)
    p.add_argument("--ego", action="store_true")
    p.add_argument("--north-up", action="store_true")
    p.add_argument("--res", type=float, default=RES_M_PER_PX)

    p = sub.add_parser("render-csv", help="render many crops; CSV columns: id,lat,lon,heading[,country]")
    p.add_argument("--root", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--north-up", action="store_true")
    p.add_argument("--res", type=float, default=RES_M_PER_PX)

    a = ap.parse_args(argv)
    if a.cmd == "info":
        px, half = crop_geometry(a.res)
        print(json.dumps(dict(px=px, half_extent_m=half, side_m=2 * half, res=a.res, tokens_32x32=(px // 32) ** 2,
                              reach_m=V_MAX_KMH / 3.6 * HORIZON_S, countries=sorted(COUNTRIES),
                              left_hand=sorted(LEFT_HAND_TRAFFIC)), indent=1))
    elif a.cmd == "regions":
        fetch_regions(a.root)
    elif a.cmd == "legend":
        _save(a.out, draw_legend())
    elif a.cmd == "prepare":
        bbox = [float(x) for x in a.bbox.split(",")] if a.bbox else None
        prepare(a.pbf, a.root, a.country, bbox, a.node_cache, not a.no_prefilter)
    elif a.cmd == "render":
        mr = MapRenderer(a.root, res=a.res, heading_up=not a.north_up)
        o = mr.render(a.lat, a.lon, a.heading, a.country, ego=a.ego)
        _save(a.out, o["image"])
        print(json.dumps(o["meta"], indent=1))
    elif a.cmd == "render-csv":
        mr = MapRenderer(a.root, res=a.res, heading_up=not a.north_up)
        od = Path(a.outdir)
        od.mkdir(parents=True, exist_ok=True)
        with open(a.csv, newline="") as f:
            for row in csv.DictReader(f):
                o = mr.render(float(row["lat"]), float(row["lon"]), float(row["heading"]),
                              row.get("country") or None)
                _save(od / f"{row['id']}.png", o["image"])


if __name__ == "__main__":
    main()