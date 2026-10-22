"""OSM region selection (vggt_slam/regions.py). CPU only."""
from pathlib import Path

from vggt_slam import regions


def write_csv(path, lat, lon):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("timestamp,GPS_latitude_deg,GPS_longitude_deg\n"
                    f"2025-01-01T00:00:00,{lat},{lon}\n2025-01-01T00:00:01,{lat},{lon}\n")


def touch_extracts(d, names):
    d.mkdir(exist_ok=True)
    for n in names:
        (d / f"{n}.osm.pbf").touch()
    return d


def test_folder_region():
    assert regions.folder_region(Path("Switzerland/a/b")) == "Switzerland"
    assert regions.folder_region(Path("United States/Ohio/a")) == "United States_Ohio"


def test_gps_beats_folder_label(tmp_path):
    ext = touch_extracts(tmp_path / "ext", ["Switzerland", "Germany", "Austria"])
    video = tmp_path / "data" / "Switzerland" / "s" / "FRONT_FOLDER" / "v.mp4"
    write_csv(video.with_suffix(".csv"), 53.5, 10.0)  # Hamburg, filed under Switzerland
    primary, fallback = regions.region_candidates(Path("Switzerland/s/FRONT_FOLDER/v.mp4"), video, ext)
    assert primary.name == "Germany.osm.pbf"
    assert {p.name for p in fallback} == {"Switzerland.osm.pbf", "Austria.osm.pbf"}


def test_only_downloaded_extracts_are_candidates(tmp_path):
    ext = touch_extracts(tmp_path / "ext", ["Austria"])
    video = tmp_path / "v.mp4"
    write_csv(video.with_suffix(".csv"), 48.1, 11.6)
    primary, fallback = regions.region_candidates(Path("Germany/s/FRONT_FOLDER/v.mp4"), video, ext)
    assert primary.name == "Austria.osm.pbf" and fallback == []


def test_no_gps_falls_back_to_folder_label_and_configured_fallbacks(tmp_path):
    ext = touch_extracts(tmp_path / "ext", ["Austria", "Croatia", "Bosnia and Herzegovina"])
    video = tmp_path / "v.mp4"  # no csv next to it
    primary, fallback = regions.region_candidates(Path("Austria/s/FRONT_FOLDER/v.mp4"), video, ext)
    assert primary.name == "Austria.osm.pbf"
    assert [p.name for p in fallback] == ["Croatia.osm.pbf", "Bosnia and Herzegovina.osm.pbf"]


def test_no_extracts_dir():
    assert regions.region_candidates(Path("Austria/s/f/v.mp4"), Path("v.mp4"), None) == (None, [])


def test_missing_folder_extract_gives_none(tmp_path):
    ext = touch_extracts(tmp_path / "ext", [])
    primary, fallback = regions.region_candidates(Path("Austria/s/f/v.mp4"), tmp_path / "v.mp4", ext)
    assert primary is None and fallback == []
