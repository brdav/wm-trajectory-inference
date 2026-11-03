"""Text description of what a forward-facing camera on the ego car would see along its next 5 s: the road it drives on
(class, one-way, lanes, speed limit), the traffic controls and crossings it passes, and the manoeuvre. Built from the
same rendered scene as the map image (`MapRenderer.render(...)["scene"]`), so the text and the image never disagree.

Two products per sample: `describe_route` -> a JSON-serialisable dict (authoritative, stored in the session zip's
text.jsonl) and `caption` -> a sentence generated from that dict only (so captions can be remade from the stored JSON).

What counts as visible (the constants below): the road the path is matched to; controls at or just past the path that face
the direction of travel (stop, give way, traffic lights, crossings, bumps, level crossings, mini roundabouts, bus stops);
nothing behind the start, nothing far to the side, nothing beyond the path end plus LOOKAHEAD_M. Values that OSM does not
carry are left out, never guessed; speed limits and lane counts that were taken from a touching road, not tagged, carry
source "inferred" (memory/osm-bev-tag-reliability.md).
"""
import math

import numpy as np
import shapely
from shapely.strtree import STRtree

from osm_bev import NON_VEHICLE

VERSION = 1
ROAD_MATCH_M = 8.0        # a path point is matched to roads within this distance
HYSTERESIS_M = 2.0        # keep the previous road unless another one scores better by more than this
LOOKAHEAD_M = 40.0        # events up to this far beyond the end of the path are reported (about one sight distance)
NO_LIMIT_KMH = 140.0      # osm_bev.parse_speed_kmh turns maxspeed=none (no limit, e.g. German Autobahn) into this value
MIN_MOVE_M = 2.0          # shorter paths = standing still
# largest lateral distance (m) from the path at which an event is reported
LATERAL_M = dict(stop=12.0, give_way=12.0, signals=10.0, crossing=3.0, calming=3.5, level_crossing=3.5,
                 mini_roundabout=3.5, bus_stop=12.0)
UNKNOWN_FACING_LATERAL_M = 3.0   # stop / give way without a mapped direction: only if the node lies on the driven road
DEDUPE_M = 8.0                   # events of one type closer than this along the path are one event


def _unit(v):
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.array([0.0, 1.0])


def _tangent(line, s, half=1.0):
    a = np.array(line.interpolate(max(s - half, 0.0)).coords[0])
    b = np.array(line.interpolate(min(s + half, line.length)).coords[0])
    return _unit(b - a)


def _directions(xy):
    """Unit direction of travel at every path point (central differences); up = the heading when standing still."""
    d = np.gradient(xy, axis=0) if len(xy) > 1 else np.zeros((1, 2))
    return np.array([_unit(v) for v in d])


def _match_roads(roads, xy, dirs):
    """Per path point the (osm_id, road dict) of the road it is on, or None. Direction-aware (a crossing street at a
    junction is not the driven road) with hysteresis (a parallel street does not flip the match)."""
    parts = [(p, r) for r in roads if r["cls"] not in NON_VEHICLE
             for p in shapely.get_parts(r["g"]) if p.geom_type == "LineString"]
    if not parts:
        return [None] * len(xy)
    tree = STRtree([p for p, _ in parts])
    out, prev = [], None
    for pt_xy, d in zip(xy, dirs):
        pt = shapely.Point(*pt_xy)
        best, scores = None, {}
        for j in tree.query(pt, predicate="dwithin", distance=ROAD_MATCH_M):
            line, r = parts[int(j)]
            s = line.project(pt)
            cos = float(np.dot(_tangent(line, s), d))
            if r["oneway"] and cos * r["oneway"] < 0:
                score = line.distance(pt) + 8.0 + 6.0          # against the one-way direction: very unlikely
            else:
                score = line.distance(pt) + 8.0 * (1.0 - abs(cos))
            if r["osm_id"] not in scores or score < scores[r["osm_id"]][0]:
                scores[r["osm_id"]] = (score, r)
            if best is None or score < best[0]:
                best = (score, r)
        if best is None:
            prev = None
            out.append(None)
            continue
        if prev is not None and prev["osm_id"] in scores and scores[prev["osm_id"]][0] <= best[0] + HYSTERESIS_M:
            best = scores[prev["osm_id"]]
        prev = best[1]
        out.append(best[1])
    return out


def _road_record(r, t0, t1):
    lanes = int(r["nl"]) or None
    speed = float(r["speed"]) if r["speed"] > 0 else None
    no_limit = speed == NO_LIMIT_KMH
    return dict(t_start_s=t0, t_end_s=t1, osm_id=int(r["osm_id"]), road_class=r["cls"], one_way=bool(r["oneway"]),
                tunnel=bool(r["tunnel"]), lanes=lanes, lanes_source=r["lsrc"] if lanes else None,
                speed_limit_kmh=None if no_limit or not speed else int(round(speed)), no_explicit_speed_limit=no_limit,
                speed_limit_source=r["src"] if speed else None)


def _road_segments(matched, times):
    segs = []
    for r, t in zip(matched, times):
        t = round(float(t), 2)
        if r is None:
            continue
        if segs and segs[-1]["osm_id"] == int(r["osm_id"]) and segs[-1]["t_end_s"] >= t - 0.25:
            segs[-1]["t_end_s"] = t
        else:
            segs.append(_road_record(r, t, t))
    return segs


def _maneuver(xy, times, length):
    if length < MIN_MOVE_M:
        return dict(kind="stationary", turn_deg=None)
    dirs = _directions(xy)
    a, b = dirs[times <= times[0] + 1.0].mean(axis=0), dirs[times >= times[-1] - 1.0].mean(axis=0)
    turn = math.degrees(math.atan2(a[1] * b[0] - a[0] * b[1], float(np.dot(a, b))))   # + = right (x right, y up)
    t = abs(turn)
    kind = "straight" if t < 15 else "slight" if t < 45 else "turn" if t < 135 else "u_turn"
    return dict(kind=kind, turn_deg=round(turn, 1),
                direction=None if kind == "straight" else ("right" if turn > 0 else "left"))


def _tangents_at(tree, parts, pt, near_m=1.5, max_d=10.0):
    """Tangents of the vehicle roads through `pt` (a node at a junction belongs to several); if none is that close, of
    the nearest road within max_d."""
    if tree is None:
        return []
    idx = tree.query(pt, predicate="dwithin", distance=near_m)
    if len(idx) == 0:
        idx = tree.query_nearest(pt, max_distance=max_d, all_matches=False)
    return [_tangent(parts[int(j)][0], parts[int(j)][0].project(pt)) for j in idx]


def describe_route(scene, lonlat, times):
    """scene: the `scene` of MapRenderer.render (roads after inference, nodes, crossing ways, local frame).
    lonlat: (N, 2) path in lon/lat, times: (N,) seconds since the sample start. -> JSON-serialisable dict."""
    to_local = scene["to_local"]
    xy = to_local(np.asarray(lonlat, float))
    times = np.asarray(times, float)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    length = float(cum[-1])
    moving = length >= MIN_MOVE_M
    dirs = _directions(xy) if moving else np.tile([0.0, 1.0], (len(xy), 1))
    roads = scene["roads"]

    matched = _match_roads(roads, xy, dirs)
    segs = _road_segments(matched, times - times[0])

    # path, extended straight ahead by LOOKAHEAD_M so controls just past its end are found
    ext = shapely.LineString(np.vstack([xy, xy[-1] + dirs[-1] * LOOKAHEAD_M])) if moving else None
    events = []
    if moving:
        veh = [(p, r) for r in roads if r["cls"] not in NON_VEHICLE
               for p in shapely.get_parts(r["g"]) if p.geom_type == "LineString"]
        tree = STRtree([p for p, _ in veh]) if veh else None
        half = max(abs(xy).max() + LOOKAHEAD_M, 1.0)

        def add(kind, sub, pt, direction_known, node_id):
            s = float(ext.project(pt))
            q = np.array(ext.interpolate(s).coords[0])
            tan = _tangent(ext, s)
            v = np.array([pt.x, pt.y]) - q
            lat = float(np.hypot(*v))
            if s < 1e-6 and float(np.dot(np.array([pt.x, pt.y]) - xy[0], dirs[0])) < -1.0:
                return                                                   # behind the start
            if lat > LATERAL_M[kind]:
                return
            beyond = s > length
            events.append(dict(
                type=kind, subtype=sub or None, t_s=None if beyond else round(float(np.interp(s, cum, times - times[0])), 2),
                dist_from_start_m=round(s, 1), beyond_path=beyond, ahead_of_end_m=round(s - length, 1) if beyond else None,
                side="left" if tan[0] * v[1] - tan[1] * v[0] > 0 else "right", lateral_m=round(lat, 1),
                direction_known=direction_known, osm_id=node_id))

        for n in scene["nodes"]:
            kind, pt = n["cls"], n["g"]
            if kind not in LATERAL_M or abs(pt.x) > half or abs(pt.y) > half:
                continue
            c = np.array([pt.x, pt.y])
            if kind in ("stop", "give_way", "signals"):
                tans = _tangents_at(tree, veh, pt)
                fwd = _tangent(ext, float(ext.project(pt)))
                if n["direction"] in (1, -1) and tans:
                    sign = 1.0 if n["direction"] == 1 else -1.0         # travel direction the sign applies to
                    if any(sign * float(np.dot(t, fwd)) >= 0.5 for t in tans):
                        add(kind, "", pt, True, int(n["osm_id"]))        # (else it faces another approach)
                elif kind == "signals":
                    add(kind, "", pt, False, int(n["osm_id"]))
                elif ext.distance(pt) <= UNKNOWN_FACING_LATERAL_M and any(float(np.dot(t, fwd)) ** 2 > 0.5 for t in tans):
                    add(kind, "", pt, False, int(n["osm_id"]))
            else:
                add(kind, n["sub"], pt, None, int(n["osm_id"]))
        for r in scene["crossings_way"]:
            hit = ext.intersection(r["g"].buffer(1.0))
            if not hit.is_empty:
                add("crossing", r["sub"], shapely.Point(*np.array(hit.representative_point().coords[0])), None,
                    int(r["osm_id"]))
        events.sort(key=lambda e: e["dist_from_start_m"])
        kept = []
        for e in events:                                                  # one crossing drawn as node + way = one event
            if not any(k["type"] == e["type"] and abs(k["dist_from_start_m"] - e["dist_from_start_m"]) < DEDUPE_M
                       for k in kept):
                kept.append(e)
        events = kept

    return dict(version=VERSION, horizon_s=round(float(times[-1] - times[0]), 2), moving=moving,
                path_length_m=round(length, 1), drive_side=scene["side"], maneuver=_maneuver(xy, times, length),
                roads=segs, events=events)


# --------------------------------------------------------------------------------------
# Caption: a pure function of the stored JSON
# --------------------------------------------------------------------------------------
ROAD_NAMES = {
    "motorway": "motorway", "motorway_link": "motorway slip road", "trunk": "trunk road", "trunk_link": "trunk slip road",
    "primary": "primary road", "primary_link": "primary slip road", "secondary": "secondary road",
    "secondary_link": "secondary slip road", "tertiary": "tertiary road", "tertiary_link": "tertiary slip road",
    "unclassified": "minor road", "residential": "residential street", "living_street": "living street",
    "service": "service road",
}
EVENT_NAMES = {
    "stop": "a stop sign", "give_way": "a give-way sign", "signals": "traffic lights", "bus_stop": "a bus stop",
    "level_crossing": "a level crossing", "mini_roundabout": "a mini roundabout", "calming": "a speed bump",
}
CROSSING_NAMES = {"signals": "a signalised pedestrian crossing", "marked": "a marked pedestrian crossing",
                  "other": "a pedestrian crossing"}


def _road_phrase(s):
    parts = [("one-way " if s["one_way"] else "") + ROAD_NAMES.get(s["road_class"], "road")]
    if s["lanes"]:
        parts.append(f"{'probably ' if s['lanes_source'] == 'neighbour' else ''}{s['lanes']} "
                     f"lane{'s' if s['lanes'] > 1 else ''}")
    if s["no_explicit_speed_limit"]:
        parts.append("no explicit speed limit" + (" (also on the adjoining road)" if s["speed_limit_source"] == "inferred" else ""))
    elif s["speed_limit_kmh"]:
        parts.append(f"{'likely ' if s['speed_limit_source'] == 'inferred' else ''}{s['speed_limit_kmh']} km/h limit")
    if s["tunnel"]:
        parts.append("in a tunnel")
    return ", ".join(parts)


def _event_phrase(e):
    if e["type"] == "crossing":
        name = CROSSING_NAMES.get(e["subtype"], "a pedestrian crossing")
    else:
        name = EVENT_NAMES[e["type"]]
        if e["type"] == "calming" and e["subtype"]:
            name = f"a speed {e['subtype']}"
    if e["direction_known"] is False:
        name = "possibly " + name
    if e["beyond_path"]:
        return f"just beyond the end of the path (about {e['ahead_of_end_m']:.0f} m): {name}"
    return f"at about {e['t_s']:.1f} s: {name}"


def caption(d):
    """One short paragraph from a describe_route dict."""
    m, out = d["maneuver"], []
    if not d["roads"]:
        out.append("No mapped road near the path.")
    else:
        first = d["roads"][0]
        out.append(("The car is on a " if d["moving"] else "The car is standing on a ") + _road_phrase(first) + ".")
        for s in d["roads"][1:]:
            if s["t_end_s"] - s["t_start_s"] >= 0.5 or s is d["roads"][-1]:
                out.append(f"From about {s['t_start_s']:.1f} s it is on a {_road_phrase(s)}.")
    if m["kind"] == "straight":
        out.append("The path goes straight.")
    elif m["kind"] == "slight":
        out.append(f"The path bears slightly {m['direction']}.")
    elif m["kind"] == "turn":
        out.append(f"The path turns {m['direction']} (about {abs(m['turn_deg']):.0f} degrees).")
    elif m["kind"] == "u_turn":
        out.append(f"The path makes a U-turn to the {m['direction']}.")
    for e in d["events"]:
        phrase = _event_phrase(e)
        out.append(f"{phrase[0].upper()}{phrase[1:]} ({e['side']}).")
    return " ".join(out)
