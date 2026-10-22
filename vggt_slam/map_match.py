"""OSM road-network and map-matching-2 helpers used internally by
vggt_slam.gps_fusion.align_trajectory_to_roads to map-match a trajectory's own shape against real
roads -- see that function's docstring for how these two pieces (build_network, run_map_match)
fit into its iterative refinement.

Uses map-matching-2 (https://github.com/addy90/map-matching-2), an MDP (value-iteration, by
default) / HMM map-matcher whose reward already accounts for route length, azimuth, turn, and
direction consistency against the input track -- unlike a plain distance-only HMM (e.g. fmm's
STMATCH, used here previously), it doesn't need a raw candidate edge to be force-connected to the
next one through whatever the network graph's shortest path happens to be: a route that would
require a long unrelated detour is simply reward-penalized (and by default, `--skip-errors`
leaves a gap there instead of forcing a bad connection).

CLI-only (no Python bindings), so this module shells out to the `map_matching_2` binary (built
from source in the Docker image, see Dockerfile) rather than calling into it in-process.
"""
import csv
import datetime
import os
import shutil
import sqlite3
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

MM2_BINARY = os.environ.get("MM2_BINARY", "map_matching_2")

# Fallback query for build_network's profile="manual_drivable" -- see its own docstring for why
# this exists: --osm-profile car applies its own undocumented access/oneway filtering (not
# controllable via --osm-query/--osm-filter/--osm-oneway) that can exclude the correct road
# entirely on real access-restricted streets (tram/PSV-designated lanes, access:lanes=no, etc.),
# forcing a detour through whatever remains unrestricted. This query keeps the same drivable
# highway *types* car would use, but skips that implicit restriction -- appropriate as a fallback
# since matching what a GPS trace shows a vehicle actually did matters more here than legal
# access semantics, but weaker as a default (it also accepts genuinely inaccessible roads).
DRIVABLE_OSM_QUERY = (
    "false"
    "|true,highway=motorway|true,highway=trunk|true,highway=primary"
    "|true,highway=secondary|true,highway=tertiary|true,highway=unclassified"
    "|true,highway=residential|true,highway=service|true,highway=living_street"
    "|true,highway=motorway_link|true,highway=trunk_link|true,highway=primary_link"
    "|true,highway=secondary_link|true,highway=tertiary_link"
)

# Degrees-per-meter at the equator; longitude is additionally scaled by cos(latitude) below.
# Not precise, and doesn't need to be, since map_buffer_m already has its own margin (see
# align_trajectory_to_roads).
_METERS_PER_DEGREE = 111_320.0


def _parse_osm_edges(osm_path, to_utm):
    """Parses the raw OSM XML at osm_path into a list of shapely LineStrings (projected to UTM
    meters via to_utm) -- one per `highway` way -- for a plotted gray road-network backdrop
    (the plotting script that used it was removed). Not used by matching itself (map-matching-2's own --prepare
    reads the OSM XML directly); this is a lightweight, dependency-free (stdlib ElementTree)
    stand-in for what osmnx/geopandas used to hand back from this module's previous fmm-based
    implementation.
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()
    nodes = {}
    for node in root.iter("node"):
        nodes[node.get("id")] = (float(node.get("lon")), float(node.get("lat")))

    edges = []
    for way in root.iter("way"):
        if not any(tag.get("k") == "highway" for tag in way.iter("tag")):
            continue
        refs = [nd.get("ref") for nd in way.iter("nd")]
        lonlat = [nodes[r] for r in refs if r in nodes]
        if len(lonlat) < 2:
            continue
        lons, lats = zip(*lonlat)
        eastings, northings = to_utm.transform(lons, lats)
        edges.append(LineString(zip(eastings, northings)))
    return edges


def _spatial_index_path(region_pbf_path):
    return region_pbf_path + ".spatial_index.sqlite"


def _build_spatial_index(region_pbf_path, index_path):
    """One-time (per region, ever) pass over region_pbf_path's highway ways, building a SQLite
    R-tree spatial index of each way's own bounding box alongside its node ids/locations/tags.
    Costs about as much as a single linear scan of the file (unavoidable: PBF itself has no
    spatial index) -- but unlike re-scanning per tile, this is paid exactly once per region,
    after which _query_bbox's lookups are effectively instant regardless of how many different
    tiles are ever queried, including ones not known about yet. Measured directly: 51s to build
    for a 139MB country file, vs. a 0.002s indexed query afterward (~15,000x faster than the
    equivalent full re-scan).

    Requires the `osmium` package (pip install osmium) to be importable -- not a hard dependency
    of the rest of this module, only needed for the local-extraction path (region_pbf_path).
    """
    import json

    import osmium

    tmp_path = index_path + ".building"
    con = sqlite3.connect(tmp_path)
    con.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, lat REAL, lon REAL)")
    con.execute("CREATE TABLE ways (id INTEGER PRIMARY KEY, node_ids TEXT, tags TEXT)")
    con.execute("CREATE VIRTUAL TABLE rtree_idx USING rtree(id, minLat, maxLat, minLon, maxLon)")

    seen_nodes = set()
    node_rows, way_rows, rtree_rows = [], [], []
    for obj in osmium.FileProcessor(region_pbf_path).with_locations():
        if not (obj.is_way() and "highway" in obj.tags):
            continue
        locs = [(n.ref, n.location) for n in obj.nodes if n.location.valid()]
        if not locs:
            continue
        lats = [loc.lat for _, loc in locs]
        lons = [loc.lon for _, loc in locs]
        way_rows.append((obj.id, json.dumps([ref for ref, _ in locs]), json.dumps(dict(obj.tags))))
        rtree_rows.append((obj.id, min(lats), max(lats), min(lons), max(lons)))
        for ref, loc in locs:
            if ref not in seen_nodes:
                seen_nodes.add(ref)
                node_rows.append((ref, loc.lat, loc.lon))

    con.executemany("INSERT INTO nodes VALUES (?, ?, ?)", node_rows)
    con.executemany("INSERT INTO ways VALUES (?, ?, ?)", way_rows)
    con.executemany("INSERT INTO rtree_idx VALUES (?, ?, ?, ?, ?)", rtree_rows)
    con.commit()
    con.close()
    os.replace(tmp_path, index_path)



def _extract_local_bbox(region_pbf_path, south, west, north, east, osm_path):
    """Crop the given WGS84 bbox's highway ways (+ their nodes) out of an already-downloaded
    whole-region OSM extract (a Geofabrik country/state .osm.pbf -- see
    scripts/setup/download_osm_extracts.py), via a SQLite R-tree spatial index built once per region
    (see _build_spatial_index) and cached alongside the .osm.pbf itself, and save as OSM XML at
    osm_path. No network access at all.
    """
    import json

    import osmium

    index_path = _spatial_index_path(region_pbf_path)
    if not os.path.exists(index_path):
        print(f"_extract_local_bbox: building spatial index for {region_pbf_path} "
              f"(one-time, reused by every future tile in this region)...", flush=True)
        _build_spatial_index(region_pbf_path, index_path)

    con = sqlite3.connect(index_path)
    way_ids = [row[0] for row in con.execute(
        "SELECT id FROM rtree_idx WHERE maxLat >= ? AND minLat <= ? AND maxLon >= ? AND minLon <= ?",
        (south, north, west, east),
    )]

    tmp_path = osm_path[:-len(".osm")] + ".building.osm" if osm_path.endswith(".osm") else osm_path + ".building"
    if os.path.exists(tmp_path):
        # A previous attempt at this exact clip crashed (whole-process kill, e.g. a transient
        # Lustre-triggered SIGBUS/SIGKILL -- confirmed on real clips) between creating this temp
        # file and the os.replace below, leaving it behind. osmium.SimpleWriter refuses to open
        # an already-existing path, so without this a stale leftover here would permanently
        # block every future retry of this clip with an opaque "RuntimeError: Open failed ...
        # File exists" -- confirmed: this was happening on 4 real clips. Safe to remove: it's
        # always mid-write scratch, never the finished osm_path itself (only reached via
        # os.replace at the end of this function).
        os.remove(tmp_path)
    written_nodes = {}
    with osmium.SimpleWriter(tmp_path) as writer:
        for way_id in way_ids:
            node_ids_json, tags_json = con.execute(
                "SELECT node_ids, tags FROM ways WHERE id = ?", (way_id,)).fetchone()
            node_ids = json.loads(node_ids_json)
            for nid in node_ids:
                if nid not in written_nodes:
                    lat, lon = con.execute(
                        "SELECT lat, lon FROM nodes WHERE id = ?", (nid,)).fetchone()
                    loc = osmium.osm.Location(lon, lat)
                    written_nodes[nid] = loc
                    writer.add_node(osmium.osm.mutable.Node(id=nid, location=loc))
            writer.add_way(osmium.osm.mutable.Way(id=way_id, nodes=node_ids, tags=json.loads(tags_json)))
    con.close()
    os.replace(tmp_path, osm_path)


def trim_osm_network_cache(network_dir):
    """Call once GPS fusion for a clip has finished (success or failure): deletes everything
    under network_dir EXCEPT network.osm itself. The prepared*/ graphs, tracks.csv, and
    results/ are pure map_matching_2 working files with no lasting value once fusion is done --
    `--prepare` costs milliseconds to rebuild from network.osm, and network.osm itself is
    trivial to re-crop from the region PBF's own spatial index (see build_network's docstring)
    -- but network.osm is still kept (a plotting script that has since been removed read it;
    nothing in the pipeline reads it now), and it's the cheapest piece to keep around
    (a few hundred KB vs. the prepared binary graphs, which are usually the bulk of this
    directory). No-op if network_dir doesn't exist (e.g. GPS fusion was never attempted).
    """
    network_dir = str(network_dir)
    if not os.path.isdir(network_dir):
        return
    for name in os.listdir(network_dir):
        if name == "network.osm":
            continue
        path = os.path.join(network_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def build_network(center_lat, center_lon, map_buffer_m, to_utm, network_dir, region_pbf_path,
                   profile="car"):
    """Fetch OSM ways around the clip (if not already cached from a previous call within THIS
    clip's own network_dir -- e.g. this function's own iterative refinement loop, or a later
    plotting script pointed at the same output) and prepare them into map-matching-2's own binary
    network format.

    No cross-clip cache sharing: earlier versions of this function shared network.osm/prepared
    across clips via a coarse lat/lon tile grid, to amortize --prepare's cost over more clips per
    fetch. That doesn't hold once region_pbf_path's spatial index exists (see
    _extract_local_bbox) -- an indexed crop costs milliseconds regardless of whether it's shared,
    and sharing a tile sized generously enough for one clip risked silently under-covering a
    different, bigger-need clip mapped to the same grid cell. Every clip's own precise
    map_buffer_m is used directly instead.

    region_pbf_path: pre-downloaded whole-region extract (see scripts/setup/download_osm_extracts.py)
    to crop locally -- see _extract_local_bbox. Required: this project's whole dataset has one
    downloaded+indexed for every country/state it spans (see scripts/setup/download_osm_extracts.py),
    so there's no other source to fall back to.

    profile: "car" (default) imports via map-matching-2's own --osm-profile car, which also
    applies its own undocumented access/oneway filtering -- usually correct (real access
    restrictions genuinely should exclude a road from car routing), but confirmed on two real
    clips to sometimes exclude the *correct* road entirely (tram/PSV-designated lanes,
    access:lanes=no, etc.), forcing a detour through whatever remains unrestricted. "manual_drivable"
    imports the same drivable highway types via --osm-profile manual with an explicit query
    (DRIVABLE_OSM_QUERY), skipping that implicit filtering -- see align_trajectory_to_roads for
    how it's used strictly as a fallback, not a default (it's weaker: no access-restriction
    awareness at all). Cached separately from the "car" prepared network (different subdirectory)
    so both can coexist for the same clip without re-preparing.

    Returns (prepared_dir, edges): prepared_dir is what --match's own --input expects; edges is a
    list of shapely LineStrings (UTM meters) for the plotted road-network backdrop only (see
    _parse_osm_edges) -- not used for matching itself.
    """
    os.makedirs(network_dir, exist_ok=True)
    osm_path = os.path.join(network_dir, "network.osm")
    prepared_dir = os.path.join(network_dir, "prepared" if profile == "car" else f"prepared_{profile}")
    graph_path = os.path.join(prepared_dir, "graph.bin")

    if not os.path.exists(osm_path):
        lat_buffer_deg = map_buffer_m / _METERS_PER_DEGREE
        lon_buffer_deg = map_buffer_m / (_METERS_PER_DEGREE * np.cos(np.radians(center_lat)))
        south, west = center_lat - lat_buffer_deg, center_lon - lon_buffer_deg
        north, east = center_lat + lat_buffer_deg, center_lon + lon_buffer_deg
        print(f"build_network: cropping OSM ways around ({center_lat:.5f},{center_lon:.5f}) "
              f"+/-{map_buffer_m:.0f}m locally from {region_pbf_path}...")
        _extract_local_bbox(region_pbf_path, south, west, north, east, osm_path)
    else:
        print(f"build_network: reusing cached OSM data at {osm_path}")

    if not os.path.exists(graph_path):
        print(f"build_network: preparing {profile!r}-profile network into {prepared_dir}")
        os.makedirs(prepared_dir, exist_ok=True)
        if profile == "car":
            prepare_args = ["--osm-profile", "car"]
        elif profile == "manual_drivable":
            prepare_args = ["--osm-profile", "manual", "--osm-query", DRIVABLE_OSM_QUERY]
        else:
            raise ValueError(f"unknown build_network profile {profile!r}")
        subprocess.run(
            [MM2_BINARY, "--prepare", "--file", osm_path, "--output", prepared_dir] + prepare_args,
            check=True, capture_output=True, text=True,
        )
    else:
        print(f"build_network: reusing prepared {profile!r}-profile network at {prepared_dir}")

    edges = _parse_osm_edges(osm_path, to_utm)
    return prepared_dir, edges


def _snap_points_to_line(points_en, line):
    """Each input point's own position on `line` (a shapely LineString), found by projecting the
    point onto it and clamping the arc-length offset to be non-decreasing from the previous
    point's -- so a point can never snap backwards relative to the direction of travel already
    established by earlier points. Returns (snapped, dev): dev is the distance (m) that snap
    moved it. Mirrors this module's previous fmm-based per-point snapping (see git history) --
    map-matching-2's own per-point output isn't used here since it targets the *matched/simplified*
    track, not every raw input point.
    """
    snapped = np.full((len(points_en), 2), np.nan)
    dev = np.full(len(points_en), np.nan)
    offset = 0.0
    for i, (px, py) in enumerate(points_en):
        p = Point(px, py)
        offset = max(line.project(p), offset)
        snapped_pt = line.interpolate(offset)
        snapped[i] = [snapped_pt.x, snapped_pt.y]
        dev[i] = snapped_pt.distance(p)
    return snapped, dev


def run_map_match(points_en, timestamps_s, prepared_dir, to_wgs84, to_utm, network_dir):
    """Run map-matching-2 and derive each point's on-road position from its own matched route.

    Returns (snapped, dev, matched_line): matched_line is the complete matched route (a shapely
    LineString, in the same UTM meters frame as points_en). snapped is each input point's own
    position on that route (see _snap_points_to_line). dev is the distance (m) that snap moved
    it. All three are None if map-matching-2 aborted the match or returned no geometry at all
    (e.g. no candidate edge found anywhere near the track).

    `--skip-errors` (map-matching-2's default) can leave gaps in the matched route rather than
    forcing a bad connection through them (unlike fmm's STMATCH, which always force-connects the
    complete route -- the actual cause of the spurious-detour issue this replaces); a gappy
    result comes back as a WKT MULTILINESTRING, which is merged back into one LineString via
    shapely.ops.linemerge, falling back to just its longest component if segments remain
    genuinely disconnected (e.g. the track crossed an isolated part of the network) -- an
    imperfect stand-in for a real gap, but consistent with every caller here expecting one
    continuous line.
    """
    tracks_path = os.path.join(network_dir, "tracks.csv")
    results_dir = os.path.join(network_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    lons, lats = to_wgs84.transform(points_en[:, 0], points_en[:, 1])
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    with open(tracks_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "latitude", "longitude", "timestamp"])
        for lat, lon, t in zip(lats, lons, timestamps_s):
            ts = (epoch + datetime.timedelta(seconds=float(t))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            writer.writerow([1, f"{lat:.8f}", f"{lon:.8f}", ts])

    result = subprocess.run(
        [MM2_BINARY, "--match", "--input", prepared_dir, "--tracks", tracks_path,
         "--output", results_dir, "--filename", "matches.csv",
         "--x", "longitude", "--y", "latitude",
         "--columns", "id,aborted,match,track_length,match_length",
         "--time-format", "%FT%T%OzZ", "--quiet"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"run_map_match: map_matching_2 --match failed (exit {result.returncode}): "
              f"{result.stderr.strip()}")
        return None, None, None

    with open(os.path.join(results_dir, "matches.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or rows[0]["aborted"] == "yes" or not rows[0]["match"]:
        return None, None, None

    try:
        match_geom = shapely_wkt.loads(rows[0]["match"])
    except Exception:
        # map_matching_2 represents "no usable match" (e.g. match_length=0) as the malformed
        # "LINESTRING()" rather than valid-WKT "LINESTRING EMPTY" -- same outcome as the aborted/
        # empty-match checks above, just caught later once GEOS actually rejects the syntax.
        return None, None, None
    if isinstance(match_geom, MultiLineString):
        merged = linemerge(match_geom)
        match_geom = max(merged.geoms, key=lambda g: g.length) if isinstance(merged, MultiLineString) else merged
    if match_geom.is_empty or match_geom.length == 0:
        return None, None, None

    mlons, mlats = zip(*match_geom.coords)
    meastings, mnorthings = to_utm.transform(mlons, mlats)
    matched_line = LineString(zip(meastings, mnorthings))

    snapped, dev = _snap_points_to_line(points_en, matched_line)
    return snapped, dev, matched_line
