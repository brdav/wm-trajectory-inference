"""Pure helpers of scripts/osm_maps/osm_bev.py. CPU only; pixel-exact regression of the renderer is a GPU-node job
(see tests/compare_results.py for the pipeline equivalent)."""
import pytest

pytest.importorskip("shapely")
pytest.importorskip("cv2")
import numpy as np

import osm_bev as ob


def road(cls="residential", width=0, lanes=0, oneway=0):
    return dict(cls=cls, width=width, lanes=lanes, oneway=oneway)


def test_road_width_prefers_tagged_width_then_lanes_then_class_default():
    assert ob.MapRenderer._road_width(road(width=6.5, lanes=4)) == 6.5
    assert ob.MapRenderer._road_width(road(lanes=2)) == 6.0
    assert ob.MapRenderer._road_width(road(lanes=2, oneway=1)) == 6.5
    assert ob.MapRenderer._road_width(road("footway", lanes=3)) == ob.ROADS["footway"][1]  # non-vehicle: lanes ignored
    assert ob.MapRenderer._road_width(road()) == ob.ROADS["residential"][1]


def test_group_rows_by_kind():
    rows = [dict(kind="area", cls="water"), dict(kind="road"), dict(kind="crossing_way"), dict(kind="rail"),
            dict(kind="node"), dict(kind="area", cls="odd_class")]
    areas, roads, crossings, rails, nodes = ob.MapRenderer._group_rows(rows)
    assert len(areas["water"]) == 1 and len(areas["odd_class"]) == 1
    assert (len(roads), len(crossings), len(rails), len(nodes)) == (1, 1, 1, 1)
    assert set(ob.AREA_PRIORITY) <= set(areas)


def test_local_frame_puts_the_heading_up():
    # driving east (heading 90 deg): a point 100 m east of the car must end up straight ahead (+y) of it
    to_local = ob.MapRenderer._local_frame(47.0, 8.0, np.radians(90.0))
    dlon = 100.0 / (111_320.0 * np.cos(np.radians(47.0)))
    x, y = to_local(np.array([[8.0 + dlon, 47.0]]))[0]
    assert abs(x) < 0.5 and abs(y - 100.0) < 0.5


def test_legend_is_an_rgb_image():
    img = ob.draw_legend(res=0.5)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
