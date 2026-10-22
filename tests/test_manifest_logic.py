"""Cross-check, adjudication, pose_source and row normalization (scripts/update_manifest_metric_scale.py). CPU only."""
import pytest

import update_manifest_metric_scale as U


def depth_record(scale, iqr=0.1, n=50, p25=None, p75=None):
    return dict(ok=True, scale=scale, scale_p25=p25 if p25 is not None else scale * 0.95,
                scale_p75=p75 if p75 is not None else scale * 1.05, scale_rel_iqr=iqr, n_frames_used=n)


class TestScaleCheck:
    @pytest.mark.parametrize("ratio,verdict", [
        (1.0, "confirmed"), (0.8, "confirmed"), (1.25, "confirmed"),
        (0.75, "ambiguous"), (1.4, "ambiguous"),
        (0.5, "contradicted"), (2.0, "contradicted"),
    ])
    def test_confident_verdicts(self, ratio, verdict):
        chk = U.scale_check(1.0, depth_record(ratio), None)
        assert chk["verdict"] == verdict and chk["confidence"] == "high"
        assert chk["ratio"] == pytest.approx(ratio)

    @pytest.mark.parametrize("gps,m,state", [
        (None, depth_record(1.0), None),                      # no GPS scale
        (1.0, None, None),                                    # no depth record
        (1.0, dict(ok=False), None),                          # failed depth run
        (1.0, depth_record(1.0), {"state": "mismatch"}),      # rerun poses mismatch
        (1.0, depth_record(1.0, n=4), None),                  # too few frames
    ])
    def test_unavailable(self, gps, m, state):
        assert U.scale_check(gps, m, state)["verdict"] == "unavailable"

    def test_noisy_depth_needs_whole_iqr_outside_to_contradict(self):
        # median ratio 0.5 (clearly outside) but IQR straddles the band -> not contradicted
        straddle = U.scale_check(1.0, depth_record(0.5, iqr=0.6, p25=0.3, p75=0.9), None)
        assert straddle["confidence"] == "low" and straddle["verdict"] == "ambiguous"
        outside = U.scale_check(1.0, depth_record(0.4, iqr=0.6, p25=0.3, p75=0.6), None)
        assert outside["verdict"] == "contradicted"
        agree = U.scale_check(1.0, depth_record(1.0, iqr=0.6, p25=0.7, p75=1.4), None)
        assert agree["verdict"] == "confirmed"

    def test_transferred_scale_is_used(self):
        state = {"transferred": {"scale": 2.0, "scale_p25": 1.9, "scale_p75": 2.1}}
        chk = U.scale_check(1.0, depth_record(1.0), state)
        assert chk["depth_scale"] == 2.0 and chk["verdict"] == "contradicted"


class TestAdjudicate:
    def q(self, reason="map_match_error_high", ratio=1.0):
        return dict(scale_reason=reason, fused_over_gps_len=ratio)

    def test_only_map_match_flag_with_agreeing_length(self):
        assert U.adjudicate({"verdict": "contradicted"}, self.q()) == "gps_length_supports_fused"

    @pytest.mark.parametrize("chk,q", [
        ({"verdict": "confirmed"}, dict(scale_reason="map_match_error_high", fused_over_gps_len=1.0)),
        ({"verdict": "contradicted"}, dict(scale_reason="len_ratio_low", fused_over_gps_len=1.0)),
        ({"verdict": "contradicted"}, dict(scale_reason="map_match_error_high", fused_over_gps_len=0.7)),
        ({"verdict": "contradicted"}, dict(scale_reason="map_match_error_high", fused_over_gps_len=None)),
    ])
    def test_no_adjudication(self, chk, q):
        assert U.adjudicate(chk, q) is None


def row(route, scale_reason=None, verdict=None, adjudication=None, source=None, reason=None, status=None, slam_reason=None):
    return U.normalize_row({
        "clip_id": "x.mp4", "ok": True,
        "qc": dict(route=route, scale_reason=scale_reason, slam_reason=slam_reason),
        "metric_scale": dict(source=source, reason=reason, status=status),
        "scale_check": dict(verdict=verdict, adjudication=adjudication),
    })


class TestPoseSource:
    @pytest.mark.parametrize("r,expected", [
        (row("use_fused"), ("fused", "gps_fusion")),
        (row("static"), ("static", "stationary")),
        (row("needs_scale_check", verdict="confirmed"), ("fused", "gps_confirmed_by_depth")),
        (row("needs_scale_check", verdict="ambiguous"), ("fused", "gps_ambiguous_vs_depth")),
        (row("needs_scale_check", verdict="unavailable"), ("fused", "gps_unchecked")),
        (row("needs_scale_check", verdict="contradicted"), ("none", "scale_contradicted")),
        (row("needs_scale_check", verdict="contradicted", adjudication="gps_length_supports_fused"),
         ("fused", "gps_length_supports_fused")),
        (row("needs_metric_scale", scale_reason="no_fused_poses", source="metric_depth"), ("metric", "no_fused_poses")),
        (row("needs_metric_scale", source=None, reason="no_calibration", status="failed"), ("none", "no_calibration")),
        (row("needs_metric_scale"), ("none", "no_scale")),
        (row("ego_failed", slam_reason="slam_jump"), ("none", "slam_jump")),
    ])
    def test_pose_source(self, r, expected):
        assert U.pose_source(r) == expected


class TestNormalizeRow:
    def test_all_schema_keys_always_present(self):
        r = U.normalize_row({"clip_id": "x.mp4", "ok": True})
        assert list(r)[:len(U.TOP_KEYS)] == list(U.TOP_KEYS)
        assert set(r["gps_fusion"]) == set(U.GPS_FUSION_KEYS)
        assert set(r["metric_scale"]) == set(U.METRIC_SCALE_KEYS)
        assert set(r["scale_check"]) == set(U.SCALE_CHECK_KEYS)
        assert all(v is None for v in r["scale_check"].values())

    def test_dropped_keys_removed_and_unknown_kept(self):
        r = U.normalize_row({"clip_id": "x", "gpu": 1, "pid": 2, "fixed_intrinsics_inferred": True, "extra": 5})
        assert "gpu" not in r and "pid" not in r and "fixed_intrinsics_inferred" not in r
        assert r["extra"] == 5 and list(r)[-1] == "extra"

    def test_depth_scale_gets_transferred_false(self):
        r = U.normalize_row({"clip_id": "x", "metric_scale": {"source": "metric_depth", "scale": 1.0}})
        assert r["metric_scale"]["scale_transferred"] is False

    def test_idempotent(self):
        r = row("needs_scale_check", verdict="confirmed")
        assert U.normalize_row(r) == r
