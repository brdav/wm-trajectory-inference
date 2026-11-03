"""scripts/osm_maps/osm_route_text.py on a synthetic scene (no map data): ego drives 40 m north along a road."""
import pytest

pytest.importorskip("shapely")
import numpy as np
import shapely

import osm_route_text as rt


def road(coords, osm_id=1, cls="residential", oneway=0, nl=0, speed=0.0, src=None):
    return dict(g=shapely.LineString(coords), osm_id=osm_id, cls=cls, oneway=oneway, tunnel=False, nl=nl,
                lsrc="map" if nl else None, speed=speed, src=src)


def node(cls, x, y, direction=0, sub="", osm_id=100):
    return dict(g=shapely.Point(x, y), cls=cls, direction=direction, sub=sub, osm_id=osm_id)


def scene(roads, nodes=(), crossings=()):
    return dict(roads=list(roads), nodes=list(nodes), crossings_way=list(crossings), to_local=lambda a: a, side="right")


TIMES = np.linspace(0.0, 5.0, 51)


def north(length=40.0):
    return np.column_stack([np.zeros(51), np.linspace(0.0, length, 51)])


def test_road_attributes_and_sources():
    main = road([(0, -30), (0, 100)], cls="primary", nl=2, speed=50.0, src="inferred")
    d = rt.describe_route(scene([main]), north(), TIMES)
    seg, = d["roads"]
    assert (seg["road_class"], seg["lanes"], seg["lanes_source"], seg["speed_limit_kmh"],
            seg["speed_limit_source"]) == ("primary", 2, "map", 50, "inferred")
    assert d["maneuver"]["kind"] == "straight" and d["moving"]
    cap = rt.caption(d)
    assert "primary road, 2 lanes, likely 50 km/h limit" in cap and "goes straight" in cap


def test_maxspeed_none_is_no_explicit_speed_limit():
    d = rt.describe_route(scene([road([(0, -30), (0, 100)], cls="motorway", oneway=1, speed=140.0, src="map")]),
                          north(), TIMES)
    seg, = d["roads"]
    assert seg["speed_limit_kmh"] is None and seg["no_explicit_speed_limit"] and seg["speed_limit_source"] == "map"
    assert "no explicit speed limit" in rt.caption(d) and "140" not in rt.caption(d)


def test_events_only_those_facing_travel_ahead_and_beside_the_path():
    main = road([(0, -30), (0, 100)])
    nodes = [
        node("stop", 3, 20, direction=1, osm_id=1),       # on the main road (tangent north), facing north: seen
        node("stop", 3, 25, direction=-1, osm_id=2),      # faces the opposite way: not for us
        node("give_way", 3, -10, direction=1, osm_id=3),  # behind the start
        node("signals", 6, 60, osm_id=4),                 # 20 m beyond the end: reported as beyond the path
        node("signals", 6, 200, osm_id=5),                # far beyond the look-ahead
        node("signals", 40, 20, osm_id=6),                # far to the side
        node("calming", 0.5, 30, sub="hump", osm_id=7),
    ]
    d = rt.describe_route(scene([main], nodes), north(), TIMES)
    got = {e["osm_id"]: e for e in d["events"]}
    assert set(got) == {1, 4, 7}
    assert got[1]["side"] == "right" and got[1]["direction_known"] is True and abs(got[1]["t_s"] - 2.5) < 0.2
    assert got[4]["beyond_path"] and got[4]["t_s"] is None and abs(got[4]["ahead_of_end_m"] - 20.0) < 1.0
    assert [e["osm_id"] for e in d["events"]] == [1, 7, 4]            # ordered along the path
    cap = rt.caption(d)
    assert "The path is about 40 m long." in cap
    assert "At about 20 m: a stop sign (right)." in cap and "At about 30 m: a speed hump" in cap
    assert "At about 60 m: possibly traffic lights" in cap and " s:" not in cap                 # distances, no times


def test_stop_sign_without_direction_is_flagged_and_needs_to_be_on_the_driven_road():
    cross = road([(-50, 20), (50, 20)], osm_id=2)
    main = road([(0, -30), (0, 100)], osm_id=1)
    d = rt.describe_route(scene([main, cross], [node("stop", 0, 20, osm_id=1), node("stop", 30, 20, osm_id=2)]),
                          north(), TIMES)
    assert [(e["osm_id"], e["direction_known"]) for e in d["events"]] == [(1, False)]
    assert "possibly a stop sign" in rt.caption(d)


def test_crossing_way_and_node_are_one_event():
    main = road([(0, -30), (0, 100)])
    cw = dict(g=shapely.LineString([(-8, 20), (8, 20)]), sub="signals", osm_id=9)
    d = rt.describe_route(scene([main], [node("crossing", 0, 20, sub="signals")], [cw]), north(), TIMES)
    assert [e["type"] for e in d["events"]] == ["crossing"]


def test_oneway_road_against_travel_is_not_the_driven_road():
    wrong = road([(0, 100), (0, -30)], oneway=1, osm_id=1)          # digitised southwards, one-way: forbids going north
    right = road([(2, -30), (2, 100)], oneway=1, osm_id=2)
    d = rt.describe_route(scene([wrong, right]), north(), TIMES)
    assert [s["osm_id"] for s in d["roads"]] == [2]


def test_turn_and_standing_still():
    th = np.linspace(0.0, np.pi / 2, 51)                              # quarter circle, bending right
    xy = np.column_stack([20 * (1 - np.cos(th)), 20 * np.sin(th)])
    m = rt.describe_route(scene([]), xy, TIMES)["maneuver"]
    assert m["kind"] == "turn" and m["direction"] == "right" and 60 < m["turn_deg"] < 95
    still = rt.describe_route(scene([road([(-20, 0), (20, 0)])]), np.zeros((51, 2)), TIMES)
    assert not still["moving"] and still["events"] == [] and "standing" in rt.caption(still)
    assert "No mapped road" in rt.caption(rt.describe_route(scene([]), north(), TIMES))


def test_road_change_is_described_by_distance_and_a_pass_at_the_start_is_at_the_start():
    a, b = road([(0, -30), (0, 20)], osm_id=1, cls="residential"), road([(0, 20), (0, 100)], osm_id=2, cls="primary")
    d = rt.describe_route(scene([a, b], [node("give_way", 0.5, 0.5, direction=1, osm_id=5)]), north(), TIMES)
    assert [(s["osm_id"], s["dist_start_m"]) for s in d["roads"]][1][0] == 2
    cap = rt.caption(d)
    assert "After about 20 m it is on a primary road." in cap and "At the start: a give-way sign" in cap
