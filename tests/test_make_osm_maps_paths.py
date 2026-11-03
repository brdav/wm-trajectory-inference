"""Clip paths and UTM zone of scripts/osm_maps/make_osm_maps.py for real (absolute ids, GPS csv) and synthetic clips."""
import json

import pytest

try:
    import make_osm_maps as mm
except Exception as e:                      # heavy deps (cv2, scipy, pyarrow) missing or broken in this env
    pytest.skip(f"make_osm_maps not importable: {e}", allow_module_level=True)

ROOT = "/capstor/x/natix-jp-uk/6adb62fa"


def test_real_and_synthetic_clip_ids_resolve_to_the_same_layout():
    real = f"{ROOT}/Japan/abc_3/12-17-14/FRONT_FOLDER/FRONT_2026-06-24_12-17-14.mp4"
    syn = "United Kingdom/syn-v2_crawley_7_5_ego253_0/08-24-53/FRONT_FOLDER/FRONT_2000-01-01_08-24-53"
    assert mm.clip_files(real, ROOT) == ("Japan/abc_3", "12-17-14",
                                         "12-17-14/FRONT_FOLDER/FRONT_2026-06-24_12-17-14/result.npz", real[:-4] + ".csv")
    assert mm.clip_files(syn, ROOT)[:3] == ("United Kingdom/syn-v2_crawley_7_5_ego253_0", "08-24-53",
                                            "08-24-53/FRONT_FOLDER/FRONT_2000-01-01_08-24-53/result.npz")
    assert mm.rel_clip(real, ROOT) == "Japan/abc_3/12-17-14/FRONT_FOLDER/FRONT_2026-06-24_12-17-14.mp4"
    assert mm.rel_clip(syn, ROOT) == syn


def test_utm_zone_of_a_synthetic_clip_comes_from_its_result(tmp_path):
    npz = {"gps_fusion_json": json.dumps({"utm_epsg": 32631})}
    assert mm.clip_utm_crs(npz, str(tmp_path / "missing.csv")).to_epsg() == 32631
    with pytest.raises(ValueError):
        mm.clip_utm_crs({}, str(tmp_path / "missing.csv"))
