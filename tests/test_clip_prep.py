"""Camera metadata lookup of a clip (vggt_slam/clip_prep.py). CPU only."""
import json
import struct

import pytest

pytest.importorskip("cv2")
from vggt_slam import clip_prep


def box(btype, payload):
    return struct.pack(">I", 8 + len(payload)) + btype + payload


def fake_mp4(width, height):
    """Smallest MP4-like box tree the parser walks: moov/trak/mdia/minf/stbl/stsd/avc1 with a width/height."""
    entry = b"\0" * 6 + b"\0\1" + b"\0" * 2 + b"\0" * 2 + b"\0" * 12 + struct.pack(">HH", width, height)
    stsd = box(b"stsd", b"\0" * 8 + box(b"avc1", entry))
    return box(b"ftyp", b"isom") + box(b"moov", box(b"trak", box(b"mdia", box(b"minf", box(b"stbl", stsd)))))


def make_clip(tmp_path, cam, width=2896, height=1876):
    """<session>/<hms>/FRONT_FOLDER/video.mp4 with FIXED_METADATA.json in the session dir (two levels above)."""
    session = tmp_path / "session"
    video = session / "10-00-00" / "FRONT_FOLDER" / "v.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(fake_mp4(width, height))
    (session / "FIXED_METADATA.json").write_text(json.dumps(
        {"device_extrinsics": [dict(device_name="camera_front", **cam)]}))
    return video


CAM = dict(fx=1500.0, fy=1500.0, cx=1448.0, cy=938.0, k1=0.1)


def test_reads_dimensions_from_box_tree(tmp_path):
    video = make_clip(tmp_path, CAM)
    assert clip_prep.read_mp4_dims(video) == (2896, 1876)


def test_camera_meta_filled_with_zero_distortion_defaults(tmp_path):
    meta = clip_prep.resolve_camera_meta(make_clip(tmp_path, CAM))
    assert meta["fx"] == 1500.0 and meta["k1"] == 0.1 and meta["k3"] == 0.0 and meta["p2"] == 0.0


def test_off_centre_principal_point_is_rejected(tmp_path):
    # the declared resolution in FIXED_METADATA.json can be wrong; a principal point far from the real centre is not trusted
    assert clip_prep.resolve_camera_meta(make_clip(tmp_path, dict(CAM, cx=400.0))) is None


def test_missing_metadata_or_unreadable_video(tmp_path):
    video = make_clip(tmp_path, CAM)
    (tmp_path / "session" / "FIXED_METADATA.json").unlink()
    assert clip_prep.resolve_camera_meta(video) is None
    assert clip_prep.read_mp4_dims(tmp_path / "nope.mp4") is None


def test_lower_case_metadata_file_name(tmp_path):
    video = make_clip(tmp_path, CAM)
    (tmp_path / "session" / "FIXED_METADATA.json").rename(tmp_path / "session" / "fixed_metadata.json")
    assert clip_prep.resolve_camera_meta(video)["fx"] == 1500.0
