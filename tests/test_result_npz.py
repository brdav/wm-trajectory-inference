"""result.npz round trip (vggt_slam/slam_utils.py). CPU only."""
import numpy as np
import pytest

from vggt_slam.slam_utils import load_result_npz, save_result_npz


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_slam_only_round_trip(tmp_path):
    n = 5
    R = np.stack([rot_z(0.1 * i) for i in range(n)])
    T = np.stack([[i, 0.5 * i, 0.0] for i in range(n)])
    save_result_npz(tmp_path / "result.npz", np.arange(1, n + 1), R, T, 10.0, "v.mp4")
    r = load_result_npz(tmp_path / "result.npz")
    np.testing.assert_allclose(r["slam_rotations"], R, atol=1e-12)
    np.testing.assert_allclose(r["slam_translations"], T)
    assert r["extraction_fps"] == 10.0 and r["video"] == "v.mp4"
    assert r["fused_rotations"] is None and r["metric_scale"] is None and r["gps_fusion_info"] is None


def test_fusion_and_metric_fields(tmp_path):
    n = 4
    R = np.stack([np.eye(3)] * n)
    T = np.arange(n * 3, dtype=float).reshape(n, 3)
    info = {"status": "ok", "map_match_error_mean_m": 2.5}
    poses = np.tile(np.eye(4), (n, 1, 1))
    save_result_npz(tmp_path / "r.npz", np.arange(1, n + 1), R, T, 10.0, "v.mp4",
                    fused_rotations=R, fused_translations=2 * T, gps_fusion_info=info,
                    metric_scale=1.7, metric_poses=poses, metric_scale_info={"n_frames_used": 9})
    r = load_result_npz(tmp_path / "r.npz")
    np.testing.assert_allclose(r["fused_translations"], 2 * T)
    assert r["gps_fusion_info"] == info
    assert r["metric_scale"] == 1.7 and r["metric_scale_info"] == {"n_frames_used": 9}
    np.testing.assert_allclose(r["metric_poses"], poses)
