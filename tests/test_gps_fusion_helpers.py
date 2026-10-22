"""Stage helpers of vggt_slam/gps_fusion.py: the 2D placement transform and region selection. CPU only."""
import numpy as np
import pytest

pytest.importorskip("shapely")
from vggt_slam import gps_fusion, map_match


def test_apply_2d_transform_scales_then_rotates_and_translates_in_x_z():
    rot = np.tile(np.eye(3), (2, 1, 1))
    trans = np.array([[1.0, 5.0, 0.0], [2.0, 5.0, 1.0]])  # (x, y, z): y is the untouched vertical axis
    R2 = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90 degrees in the (x, z) plane
    out_rot, out_trans = gps_fusion._apply_2d_transform(R2, np.array([10.0, 20.0]), 2.0, rot, trans)
    # (x, z) = 2 * (1, 0) -> rotated (0, 2) -> + (10, 20)
    np.testing.assert_allclose(out_trans[0], [10.0, 10.0, 22.0])
    np.testing.assert_allclose(out_trans[:, 1], [10.0, 10.0])  # 2 * y
    assert out_rot.shape == (2, 3, 3) and abs(np.linalg.det(out_rot[0]) - 1) < 1e-12


class FakeBuilder:
    """build_network stand-in: roads exist only in the listed region extracts."""
    def __init__(self, with_roads):
        self.with_roads, self.calls = set(with_roads), []

    def __call__(self, lat, lon, buffer_m, to_utm, network_dir, region_pbf_path, profile="car"):
        self.calls.append((str(region_pbf_path), profile))
        return f"{network_dir}/prepared", ([1] if str(region_pbf_path) in self.with_roads else [])


def select(monkeypatch, tmp_path, with_roads, region, fallback):
    builder = FakeBuilder(with_roads)
    monkeypatch.setattr(map_match, "build_network", builder)
    net, used = gps_fusion._select_region(47.0, 8.0, 500.0, None, str(tmp_path), region, fallback)
    return net, used, builder


def test_labeled_region_with_roads_is_used(monkeypatch, tmp_path):
    net, used, builder = select(monkeypatch, tmp_path, {"A"}, "A", ["B"])
    assert used == "A" and net.region_pbf_path == "A" and builder.calls == [("A", "car")]


def test_first_fallback_with_roads_wins_and_gets_its_own_dir(monkeypatch, tmp_path):
    net, used, builder = select(monkeypatch, tmp_path, {"C"}, "A", ["B", "C"])
    assert used == "C" and net.region_pbf_path == "C"
    assert net.network_dir.endswith("fallback_region_1")
    assert [c[0] for c in builder.calls] == ["A", "B", "C"]


def test_single_fallback_path_string(monkeypatch, tmp_path):
    _, used, _ = select(monkeypatch, tmp_path, {"B"}, "A", "B")
    assert used == "B"


def test_no_roads_anywhere(monkeypatch, tmp_path):
    net, used, _ = select(monkeypatch, tmp_path, set(), "A", ["B"])
    assert used is None and net.region_pbf_path == "A"
    _, used, _ = select(monkeypatch, tmp_path, set(), "A", None)
    assert used is None
