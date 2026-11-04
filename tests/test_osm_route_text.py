"""scripts/osm_maps/osm_route_text.py on a synthetic scene (no map data): ego drives 40 m north along a road."""
import pytest

pytest.importorskip("shapely")
import numpy as np
import shapely

import osm_route_text as rt


def road(coords, osm_id=1, cls="residential", oneway=0, nl=0, speed=0.0, src=None):
    return dict(g=shapely.LineString(coords), osm_id=osm_id, cls=cls, oneway=oneway, tunnel=False, layer=0, nl=nl,
                lsrc="map" if nl else None, speed=speed, src=src)


def node(cls, x, y, direction=0, sub="", osm_id=100):
    return dict(g=shapely.Point(x, y), cls=cls, direction=direction, sub=sub, osm_id=osm_id)


def scene(roads, nodes=(), crossings=(), region="United States_Nevada"):
    return dict(roads=list(roads), nodes=list(nodes), crossings_way=list(crossings), to_local=lambda a: a, side="right",
                region=region)


TIMES = np.linspace(0.0, 5.0, 51)


def north(length=40.0):
    return np.column_stack([np.zeros(51), np.linspace(0.0, length, 51)])


def test_road_attributes_and_sources():
    main = road([(0, -30), (0, 100)], cls="primary", nl=2, speed=50.0, src="inferred")
    d = rt.describe_route(scene([main]), north(), TIMES)
    seg, = d["roads"]
    assert (seg["road_class"], seg["lanes"], seg["lanes_source"], seg["speed_limit"],
            seg["speed_limit_source"]) == ("primary", 2, "map", 31, "inferred")   # 50 km/h stored as mph in Nevada
    assert d["speed_unit"] == "mph"
    assert d["maneuver"]["kind"] == "straight" and d["moving"]
    cap = rt.caption(d)
    assert "primary road, 2 lanes, likely 31 mph limit" in cap and "goes straight" in cap


def test_maxspeed_none_is_no_explicit_speed_limit():
    d = rt.describe_route(scene([road([(0, -30), (0, 100)], cls="motorway", oneway=1, speed=140.0, src="map")]),
                          north(), TIMES)
    seg, = d["roads"]
    assert seg["speed_limit"] is None and seg["no_explicit_speed_limit"] and seg["speed_limit_source"] == "map"
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
    assert "It drives at about 20 mph, at a steady speed." in cap                    # 8 m/s in Nevada: signs read mph
    assert "At about 20 m: a stop sign (right)." in cap and "At about 30 m: a speed hump" in cap
    assert "At about 60 m: possibly traffic lights" in cap and " s:" not in cap                 # distances, no times


def test_stop_sign_without_direction_is_flagged_and_needs_to_be_on_the_driven_road():
    cross = road([(-50, 20), (50, 20)], osm_id=2)
    main = road([(0, -30), (0, 100)], osm_id=1)
    d = rt.describe_route(scene([main, cross], [node("stop", 0, 20, osm_id=1), node("stop", 30, 20, osm_id=2)]),
                          north(), TIMES)
    assert [(e["osm_id"], e["direction_known"]) for e in d["events"] if e["type"] == "stop"] == [(1, False)]
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


def test_location_line_and_ego_vehicle_wording(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    f = tmp_path / "places.parquet"
    pq.write_table(pa.table(dict(name=["Reno", "Tiny"], kind=["city", "village"], lat=[0.0, 0.0], lon=[0.0, 0.5],
                                 population=[0, 0])), f)
    places = rt.PlaceIndex(f)
    main = road([(0, -30), (0, 100)])
    # to_local is the identity here, so "lon/lat" of the path start (0, 0) is also the position of Reno
    d = rt.describe_route(scene([main]), north(), TIMES, places)
    assert d["location"] == dict(country="United States", state="Nevada", place="Reno", place_kind="city",
                                 place_distance_km=0.0, within_place=True)
    cap = rt.caption(d)
    assert cap.startswith("Location: Reno, Nevada, United States. Traffic drives on the right. The ego vehicle is on a residential street")
    assert "The car" not in cap
    assert places.nearest(0.0, 0.02)["within"] is True and places.nearest(0.0, 0.2)["within"] is False   # 22 km from Reno: "near"
    assert places.nearest(40.0, 40.0) is None                                                           # nothing within 3 radii
    far = dict(d["location"], place=None, within_place=None)
    assert rt._location_phrase(far) == "Nevada, United States"
    assert "Location" not in rt.caption(dict(d, location=dict(far, country=None)))


def _way(coords, osm_id, cls="residential", layer=0):
    r = road(coords, osm_id=osm_id, cls=cls)
    r["layer"] = layer
    return r


JUNCTIONS = ("intersection", "side_road", "t_junction", "fork", "roundabout")


def _events(roads, ids=None, kinds=JUNCTIONS):
    d = rt.describe_route(scene(roads), north(), TIMES, None, ids)
    return [e for e in d["events"] if e["type"] in kinds]


def test_crossroads_is_an_intersection_but_one_joining_side_road_is_not():
    main = _way([(0, -30), (0, 20), (0, 100)], 1)                  # passes through the shared node at (0, 20)
    cross = _way([(-50, 20), (0, 20), (50, 20)], 2)
    (e,) = _events([main, cross])
    assert e["type"] == "intersection" and e["arms"] == 4 and abs(e["dist_from_start_m"] - 20.0) < 1.0
    assert "a 4-way intersection" in rt._event_phrase(e)
    (e,) = _events([main, _way([(0, 20), (40, 20)], 3)])           # one side road ends on the road: it joins from the right
    assert (e["type"], e["side"], e["arms"]) == ("side_road", "right", 3)
    assert rt._event_phrase(e).endswith("a side road joining on the right")
    assert "a side road joining on the right." in rt.caption(rt.describe_route(scene([main, _way([(0, 20), (40, 20)], 3)]), north(), TIMES))
    (e,) = _events([main, _way([(0, 20), (-40, 20)], 3)])
    assert (e["type"], e["side"]) == ("side_road", "left")
    two = _events([main, _way([(0, 20), (40, 20)], 3), _way([(0, 20), (-40, 20)], 4)])      # one each side = crossroads
    assert [(e["type"], e["arms"]) for e in two] == [("intersection", 4)]
    assert "a 4-way intersection." in rt.caption(rt.describe_route(scene([main, cross]), north(), TIMES))     # no side


def test_t_junction_where_the_driven_road_ends_and_a_fork():
    ends = _way([(0, -30), (0, 20)], 1)
    top = _way([(-50, 20), (0, 20), (50, 20)], 2)
    (e,) = _events([ends, top])
    assert (e["type"], e["arms"]) == ("t_junction", 3) and "the road ends" in rt._event_phrase(e)
    (e,) = _events([_way([(0, -30), (0, 20)], 1), _way([(0, 20), (-12, 60)], 2), _way([(0, 20), (12, 60)], 3)])
    assert e["type"] == "fork"
    assert _events([_way([(0, -30), (0, 20)], 1), _way([(0, 20), (-40, 20)], 2)]) == []    # a bend: two arms


def test_no_junction_for_continuations_service_roads_bridges_or_junctions_behind_the_start():
    a, b = _way([(0, -30), (0, 20)], 1), _way([(0, 20), (0, 100)], 2)             # two ways end to end: 2 arms
    assert _events([a, b]) == []
    main = _way([(0, -30), (0, 100)], 1)
    assert _events([main, _way([(-30, 20), (0, 20), (30, 20)], 2, cls="service")]) == []        # driveway / parking aisle
    assert _events([main, _way([(-30, 20), (30, 20)], 2, layer=1)]) == []                       # bridge over the road
    assert _events([main, _way([(-30, -10), (30, -10)], 2)]) == []                             # behind the start


def test_roundabout_needs_the_tag_and_counts_its_exits():
    th = np.linspace(0, 2 * np.pi, 33)
    pts = np.column_stack([12 * np.cos(th), 30 + 12 * np.sin(th)])
    pts[[0, 8, 24]] = [(12, 30), (0, 42), (0, 18)]                                         # vertices the arms attach to
    pts[-1] = pts[0]                                                                       # exactly closed, like an OSM ring
    ring = _way(pts.tolist(), 10)                                                          # ahead, centre 30 m
    ring["oneway"] = 1
    arms = [_way([(0, -30), (0, 18)], 11), _way([(0, 42), (0, 100)], 12), _way([(12, 30), (60, 30)], 13)]
    assert all(e["type"] != "roundabout" for e in _events([ring] + arms, set()))          # untagged: not a roundabout
    ev = _events([ring] + arms, {10})
    assert [e["type"] for e in ev] == ["roundabout"] and ev[0]["connected_ways"] == 3
    assert abs(ev[0]["dist_from_start_m"] - 18.0) < 1.5
    assert rt._event_phrase(ev[0]).endswith("a roundabout")


def test_speed_unit_follows_the_signs_of_the_country():
    main = road([(0, -30), (0, 100)], cls="primary", speed=112.65, src="map")           # OSM "70 mph", held as km/h by osm_bev
    d = rt.describe_route(scene([main], region="United Kingdom"), north(), TIMES)
    assert d["speed_unit"] == "mph" and d["roads"][0]["speed_limit"] == 70 and "70 mph limit" in rt.caption(d)
    d = rt.describe_route(scene([main], region="Switzerland"), north(), TIMES)
    assert d["speed_unit"] == "km/h" and d["roads"][0]["speed_limit"] == 113 and "113 km/h limit" in rt.caption(d)



def test_consecutive_ways_with_the_same_description_are_mentioned_once():
    a = road([(0, -30), (0, 20)], osm_id=1, cls="primary", nl=2, speed=50.0, src="map")
    b = road([(0, 20), (0, 100)], osm_id=2, cls="primary", nl=2, speed=50.0, src="map")
    d = rt.describe_route(scene([a, b]), north(), TIMES)
    assert len(d["roads"]) == 2 and "After about" not in rt.caption(d)


def test_junction_is_judged_by_the_approach_even_when_the_path_turns_at_it():
    main = _way([(0, -30), (0, 20), (0, 100)], 1)
    cross = _way([(-50, 20), (0, 20), (50, 20)], 2)
    th = np.linspace(0, 1, 51)                                    # drives up the main road, then turns right through the junction
    xy = np.column_stack([np.where(th < 0.5, 0.0, (th - 0.5) * 80), np.where(th < 0.5, th * 40, 20.0)])
    d = rt.describe_route(scene([main, cross]), xy, TIMES)
    assert [e["type"] for e in d["events"] if e["type"] in JUNCTIONS] == ["intersection"]


def _along(speed_of_t):
    """Straight path north whose speed (m/s) at time t is speed_of_t(t)."""
    v = np.array([speed_of_t(t) for t in TIMES])
    return np.column_stack([np.zeros(51), np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(TIMES))])])


def test_speed_and_trend_from_the_path():
    main = road([(0, -30), (0, 400)])
    cases = [(lambda t: 10.0, "steady"), (lambda t: 5.0 + 2.0 * t, "accelerating"), (lambda t: 15.0 - 2.0 * t, "braking"),
             (lambda t: 8.0 * max(0.0, 1.0 - t / 3.0), "stopping"), (lambda t: 8.0 * min(1.0, max(0.0, t - 1.0) / 2.0), "pulling_away"),
             (lambda t: 0.0, "stationary")]
    for fn, trend in cases:
        d = rt.describe_route(scene([main]), _along(fn), TIMES)
        assert d["motion"]["trend"] == trend, (trend, d["motion"])
    d = rt.describe_route(scene([main]), _along(cases[0][0]), TIMES)
    assert d["motion"]["mean_speed"] == 22.4 and d["speed_unit"] == "mph"          # 10 m/s in Nevada
    assert "It drives at about 20 mph, at a steady speed." in rt.caption(d)
    ch = rt.describe_route(scene([main], region="Switzerland"), _along(cases[0][0]), TIMES)
    assert ch["motion"]["mean_speed"] == 36.0 and "It drives at about 35 km/h, at a steady speed." in rt.caption(ch)
    d = rt.describe_route(scene([main]), _along(cases[2][0]), TIMES)
    assert "and is braking." in rt.caption(d)
    d = rt.describe_route(scene([main]), _along(cases[5][0]), TIMES)
    assert "It drives" not in rt.caption(d) and not d["moving"]


def test_side_is_only_given_for_controls_clearly_beside_the_path():
    main = road([(0, -30), (0, 100)])
    nodes = [node("signals", 0.8, 20, osm_id=1), node("signals", 6.0, 40, osm_id=2), node("crossing", 0, 60, sub="marked", osm_id=3),
             node("stop", -4.0, 70, direction=1, osm_id=4), node("level_crossing", 0.2, 80, osm_id=5)]
    d = rt.describe_route(scene([main], nodes), north(), TIMES)
    side = {e["osm_id"]: e["side"] for e in d["events"] if e["type"] != "intersection"}
    assert side == {1: None, 2: "right", 3: None, 4: "left", 5: None}
    cap = rt.caption(d)
    assert "At about 20 m: possibly traffic lights." in cap and "At about 40 m: possibly traffic lights (right)." in cap
    assert "pedestrian crossing." in cap and "level crossing." in cap


def test_drive_side_is_stated_and_the_uk_drives_on_the_left():
    import osm_bev
    assert [osm_bev.drive_side(c) for c in ("United Kingdom", "Japan", "Ireland", "Switzerland", "United States_Nevada")] == \
        ["left", "left", "left", "right", "right"]
    main = road([(0, -30), (0, 100)])
    d = rt.describe_route(dict(scene([main]), side="left"), north(), TIMES)
    assert d["drive_side"] == "left" and "Traffic drives on the left." in rt.caption(d)
