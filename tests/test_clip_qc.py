"""QC routing (scripts/clip_qc.py) on synthetic poses, GPS logs and session zips. CPU only."""
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import clip_qc

FPS = 10.0


def poses_along_x(steps, start=0.0):
    """(N,4,4) identity-rotation poses with x translation = cumulative sum of steps."""
    x = start + np.concatenate([[0.0], np.cumsum(steps)])
    P = np.tile(np.eye(4), (len(x), 1, 1))
    P[:, 0, 3] = x
    return P


def write_gps_csv(path, speed_mps, n_rows=7, native_fps=30, lat=47.0, lon=8.0):
    """1 Hz sensor log of a vehicle driving north at speed_mps (frame_number from a 30 fps camera)."""
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    lines = ["timestamp,frame_number,GPS_latitude_deg,GPS_longitude_deg,horizontal_accuracy_m"]
    for i in range(n_rows):
        dlat = speed_mps * i / 111_000.0
        lines.append(f"{(t0 + timedelta(seconds=i)).isoformat()},{1 + i * native_fps},{lat + dlat:.7f},{lon:.7f},5")
    Path(path).write_text("\n".join(lines) + "\n")


def result_npz_bytes(n, step, fused_scale=None, fps=FPS):
    """result.npz of n keyframes (every frame of the extraction), moving `step` per keyframe along x."""
    slam = poses_along_x(np.full(n - 1, step))
    arrs = dict(frame_ids=np.arange(1, n + 1), extraction_fps=fps, slam_poses=slam)
    if fused_scale is not None:
        arrs["fused_poses"] = slam * np.array([1, 1, 1, 1.0])  # copy
        arrs["fused_poses"][:, :3, 3] *= fused_scale
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    return buf.getvalue()


class TestCheckSlam:
    def test_too_few_keyframes(self):
        ok, reason, *_ = clip_qc.check_slam(np.arange(3.0), poses_along_x([1, 1]))
        assert not ok and reason == "too_few_keyframes"

    def test_smooth_trajectory_is_ok(self):
        t = np.arange(20) / FPS
        ok, reason, valid_until, ratio = clip_qc.check_slam(t, poses_along_x(np.ones(19)))
        assert ok and reason == "" and valid_until == t[-1] and ratio == pytest.approx(1.0)

    def test_single_jump_reports_time_of_last_clean_keyframe(self):
        steps = np.ones(19)
        steps[10] = 500  # step 10 -> between keyframe 10 and 11
        t = np.arange(20) / FPS
        ok, reason, valid_until, _ = clip_qc.check_slam(t, poses_along_x(steps))
        assert not ok and reason == "slam_jump" and valid_until == pytest.approx(t[10])

    def test_sustained_divergence_below_jump_ratio(self):
        # the run must stay below 10% of the steps, else the p90-based "typical step" absorbs it
        steps = np.ones(100)
        steps[-clip_qc.SLAM_RUN_LEN:] = clip_qc.SLAM_RUN_RATIO * 2  # > run ratio, < jump ratio
        ok, reason, *_ = clip_qc.check_slam(np.arange(101) / FPS, poses_along_x(steps))
        assert not ok and reason == "slam_jump"

    def test_short_spike_below_run_length_is_ok(self):
        steps = np.ones(100)
        steps[-(clip_qc.SLAM_RUN_LEN - 1):] = clip_qc.SLAM_RUN_RATIO * 2
        ok, *_ = clip_qc.check_slam(np.arange(101) / FPS, poses_along_x(steps))
        assert ok

    def test_nonfinite(self):
        P = poses_along_x(np.ones(9))
        P[4, 0, 3] = np.nan
        ok, reason, *_ = clip_qc.check_slam(np.arange(10) / FPS, P)
        assert not ok and reason == "nonfinite_poses"


class TestCheckScale:
    def test_window_speed(self):
        t = np.arange(0, 20, 1.0)
        xz = np.stack([t * 10.0, np.zeros_like(t)], axis=1)  # 10 m/s
        assert clip_qc.window_speed_max(t, xz, 5.0) == pytest.approx(10.0)

    def test_window_longer_than_clip_is_nan(self):
        t = np.arange(0, 3, 1.0)
        assert np.isnan(clip_qc.window_speed_max(t, np.zeros((3, 2)), 5.0))

    def test_implausible_speed(self, tmp_path):
        t = np.arange(0, 20, 1.0)
        ok, reason, vmax, _ = clip_qc.check_scale(t, poses_along_x(np.full(19, 80.0)), tmp_path / "missing.csv")
        assert not ok and reason == "implausible_speed" and vmax == pytest.approx(80.0)

    def test_no_gps_log_only_speed_check(self, tmp_path):
        t = np.arange(0, 20, 1.0)
        ok, reason, _, ratio = clip_qc.check_scale(t, poses_along_x(np.full(19, 10.0)), tmp_path / "missing.csv")
        assert ok and np.isnan(ratio)

    @pytest.mark.parametrize("fused_speed,expect_ok", [(10.0, True), (18.0, False)])
    def test_length_ratio_against_gps(self, tmp_path, fused_speed, expect_ok):
        csv = tmp_path / "gps.csv"
        write_gps_csv(csv, speed_mps=10.0)  # 6 s of driving at 10 m/s
        t = np.arange(0, 7, 1.0)
        ok, reason, _, ratio = clip_qc.check_scale(t, poses_along_x(np.full(6, fused_speed)), csv)
        assert ok == expect_ok
        assert ratio == pytest.approx(fused_speed / 10.0, rel=0.05)
        if not expect_ok:
            assert reason == "longer_than_gps"


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A fake trajectories/ + dataset/ tree with one session zip; returns add(clip, npz_bytes, gps_speed)."""
    traj, data = tmp_path / "traj", tmp_path / "data"
    sess = Path("Switzerland/2025-01-01/2025-01-01_abc_1")
    monkeypatch.setattr(clip_qc, "TRAJ_ROOT", traj)
    monkeypatch.setattr(clip_qc, "DATASET_ROOT", data)
    zip_path = traj / sess.parent / (sess.name + ".zip")
    zip_path.parent.mkdir(parents=True)
    members = {}

    def add(hms, npz_bytes, gps_speed=None):
        stem = f"FRONT_2025-01-01_{hms}"
        members[f"{hms}/FRONT_FOLDER/{stem}/result.npz"] = npz_bytes
        if gps_speed is not None:
            csv = data / sess / hms / "FRONT_FOLDER" / f"{stem}.csv"
            csv.parent.mkdir(parents=True, exist_ok=True)
            write_gps_csv(csv, gps_speed)
        return str(sess / hms / "FRONT_FOLDER" / f"{stem}.mp4")

    def build():
        with zipfile.ZipFile(zip_path, "w") as zf:
            for k, v in members.items():
                zf.writestr(k, v)
        return zip_path

    return add, build


def routes_of(rows):
    return {r["clip"]: r["route"] for r in rows}


class TestProcessZip:
    def test_routes(self, session):
        add, build = session
        n = 70  # 7 s at 10 fps; GPS covers 6 s at 10 m/s -> 1.0 m per keyframe step = 10 m/s
        step = 1.0
        clips = {
            "use_fused": add("10-00-00", result_npz_bytes(n, step, fused_scale=1.0), gps_speed=10.0),
            "no_fused": add("10-00-01", result_npz_bytes(n, step), gps_speed=10.0),
            "too_long": add("10-00-02", result_npz_bytes(n, step, fused_scale=1.8), gps_speed=10.0),
            "ratio_low": add("10-00-03", result_npz_bytes(n, step, fused_scale=0.6), gps_speed=10.0),
            "static": add("10-00-04", result_npz_bytes(3, 0.1), gps_speed=0.0),
            "few_kf_moved": add("10-00-05", result_npz_bytes(3, 0.1), gps_speed=10.0),
            "few_kf_no_gps": add("10-00-06", result_npz_bytes(3, 0.1)),
        }
        got = routes_of(clip_qc.process_zip(build()))
        assert got[clips["use_fused"]] == "use_fused"
        assert got[clips["no_fused"]] == "needs_metric_scale"
        assert got[clips["too_long"]] == "needs_metric_scale"
        assert got[clips["ratio_low"]] == "needs_scale_check"
        assert got[clips["static"]] == "static"
        assert got[clips["few_kf_moved"]] == "ego_failed"
        assert got[clips["few_kf_no_gps"]] == "ego_failed"

    def test_unreadable_member_is_no_result(self, session):
        add, build = session
        clip = add("10-00-00", b"not an npz")
        rows = clip_qc.process_zip(build())
        assert routes_of(rows) == {clip: "no_result"}

    def test_unreadable_zip(self, tmp_path):
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"nope")
        assert clip_qc.process_zip(bad) == []
