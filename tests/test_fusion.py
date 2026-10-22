"""Outcome classification of GPS fusion failures (vggt_slam/fusion.py). CPU only."""
import pytest

pytest.importorskip("shapely")
from vggt_slam import fusion


@pytest.mark.parametrize("message,status", [
    ("no roads found near the GPS fixes in any region", "no_roads_in_any_region"),
    ("raw-GPS map-match unavailable (matched_line=found, 2 keyframes in window)", "no_raw_gps_route_too_sparse"),
    ("raw-GPS map-match unavailable (matched_line=none)", "no_raw_gps_route_unmatched"),
    ("implausible GPS-derived scale 120 m/s", "implausible_gps_scale"),
    ("No real GPS fixes (all 'na') in x.csv", "no_real_gps_fixes"),
    ("only 1 keyframe(s) matched to a GPS fix", "insufficient_gps_matches"),
    ("Only 1 sensor log row(s) (1 distinct timestamp(s)) in x.csv", "gps_log_too_short"),
    ("something else entirely", "failed"),
])
def test_value_error_statuses(message, status):
    try:
        raise ValueError(message)
    except ValueError as e:
        info = fusion.classify_error(e)
    assert info["status"] == status
    assert (message in info["error"]) if status != "failed" else ("something else" in info["error"])


def test_non_value_error_is_failed_with_traceback_tail():
    try:
        raise RuntimeError("boom " * 1000)
    except RuntimeError as e:
        info = fusion.classify_error(e, traceback_chars=100)
    assert info["status"] == "failed" and len(info["error"]) == 100


def test_first_matching_status_wins():
    try:
        raise ValueError("no roads found near the GPS fixes; No real GPS fixes")
    except ValueError as e:
        assert fusion.classify_error(e)["status"] == "no_roads_in_any_region"


def test_missing_region_extract_is_reported_not_raised(tmp_path):
    info, fused = fusion.run_gps_fusion(None, None, None, 10.0, tmp_path / "x.csv", tmp_path / "net", None)
    assert fused is None and info["status"] == "failed" and "OSM region extract" in info["error"]
