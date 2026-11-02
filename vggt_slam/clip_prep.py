"""Preparing one clip for SLAM: camera calibration from the clip's FIXED_METADATA.json, frame extraction with
ffmpeg, and the (session-cached) undistortion maps. Used by scripts/slam/batch_infer.py and
scripts/metric_scale/estimate_metric_scale.py."""
import json
import os
import struct
import subprocess
import threading
import time
import traceback

import cv2

from vggt_slam.undistort import compute_undistort_maps, save_undistort_maps, session_cache_dir

_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd", b"edts"}
_VIDEO_SAMPLE_ENTRY_TYPES = {b"avc1", b"hvc1", b"hev1", b"mp4v", b"vp09", b"av01"}


def _walk_mp4_boxes(f, start, end, in_stsd=False):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            return None
        size = struct.unpack(">I", header[:4])[0]
        btype = header[4:8]
        header_len = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header_len = 16
        if size == 0:
            size = end - pos
        box_end = pos + size

        if in_stsd and btype in _VIDEO_SAMPLE_ENTRY_TYPES:
            # VisualSampleEntry: box header(8) + reserved(6) + data_ref_idx(2) + pre_defined(2)
            # + reserved(2) + pre_defined[3](12) = 24 bytes in, then width/height (2 uint16 BE).
            f.seek(pos + 8 + 6 + 2 + 2 + 2 + 12)
            dims = f.read(4)
            if len(dims) == 4:
                return struct.unpack(">HH", dims)
            return None
        if btype == b"stsd":
            result = _walk_mp4_boxes(f, pos + header_len + 8, box_end, in_stsd=True)
            if result:
                return result
        elif btype in _CONTAINER_BOXES:
            result = _walk_mp4_boxes(f, pos + header_len, box_end, in_stsd=in_stsd)
            if result:
                return result
        pos = box_end
    return None


def read_mp4_dims(video_path):
    """Reads the encoded video stream's (width, height) straight from the MP4 box structure,
    without needing ffprobe. Only seeks/reads box headers (not frame data), so this stays cheap
    regardless of file size. Returns None if unparseable."""
    try:
        with open(video_path, "rb") as f:
            return _walk_mp4_boxes(f, 0, os.path.getsize(video_path))
    except Exception:
        return None


def resolve_camera_meta(video_path):
    """Looks up this clip's camera_front intrinsics + distortion from a sibling
    FIXED_METADATA.json. Returns {"fx","fy","cx","cy","k1","k2","k3","p1","p2"} (used as-is, NOT
    rescaled) or None if metadata is missing, unparseable, or fails a sanity check against the
    actual video.

    FIXED_METADATA.json's own declared frame_width/height field cannot be trusted: for at least
    one clip it reported exactly half the true encoded resolution. But cam["fx"/"cx"/etc] were
    STILL correct as-is for the actual (not the declared) resolution in that case -- e.g. a
    clip with declared frame_width=1448 but actual encoded width 2896 had cx=1448, which is
    actual_width/2 exactly, not declared_width/2. So the fix is not to rescale fx/fy/cx/cy at
    all -- it's to validate them directly against the actual video (principal point should sit
    near the actual frame's center) instead of trusting the declared reference resolution.
    """
    try:
        # the natix-jp-uk delivery spells it in lower case
        meta_path = next((p for p in (video_path.parents[2] / n for n in ("FIXED_METADATA.json", "fixed_metadata.json"))
                          if p.exists()), None)
        if meta_path is None:
            return None
        meta = json.loads(meta_path.read_text())
        cam = next((d for d in meta.get("device_extrinsics", []) if d.get("device_name") == "camera_front"), None)
        if cam is None:
            return None
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        actual = read_mp4_dims(video_path)
        if not actual:
            return None
        aw, ah = actual
        if not (0.3 * aw < cx < 0.7 * aw and 0.3 * ah < cy < 0.7 * ah):
            return None  # principal point isn't near-centered on the actual frame -- don't trust it
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "k1": cam.get("k1", 0.0), "k2": cam.get("k2", 0.0), "k3": cam.get("k3", 0.0),
            "p1": cam.get("p1", 0.0), "p2": cam.get("p2", 0.0),
        }
    except Exception:
        return None


def extract_one(clip, fps, ffmpeg_threads=0):
    frames_dir = clip["frames_dir"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not any(frames_dir.iterdir()):
        _t0 = time.time()
        try:
            subprocess.run(
                # -threads before -i sets ffmpeg's own decode parallelism (0 = let it pick;
                # decode is what actually scales with source resolution, e.g. France's
                # 2896x1876 vs 1280x960 for the other natix clips -- this was the single
                # biggest per-clip CPU cost observed for that clip).
                ["ffmpeg", "-y", "-threads", str(ffmpeg_threads), "-i", str(clip["video_path"]),
                 "-vf", f"fps={fps}", str(frames_dir / "frame_%05d.jpg")],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            return clip, e.stderr.decode(errors="replace")[-2000:]
        print(f"extract_one DIAG: ffmpeg extraction for {clip['video_path'].name} took "
              f"{time.time()-_t0:.2f}s", flush=True)

    cam_meta = clip["camera_meta"]
    if cam_meta is None:
        clip["image_folder"] = str(frames_dir)
        clip["fixed_intrinsics_str"] = None
        return clip, None

    # No second (undistorted) copy of the frames is ever written to disk: the remap tables only
    # depend on resolution + known distortion, not frame content, so they're computed once here
    # from a single frame's shape and cached; vggt_slam/pipeline.py loads them and undistorts in-memory, only
    # for the frames it actually keeps as keyframes (see vggt_slam/undistort.py's docstring).
    #
    # Cached at the SESSION level (session_cache_dir), not per-clip: every clip sharing one
    # FIXED_METADATA.json has identical camera_meta and (in practice) identical resolution, so
    # this ~10MB file was previously being redundantly recomputed and re-saved once per clip
    # (confirmed: avg 6.0 clips/session in Croatia's data) -- a major contributor to this
    # pipeline's inode/storage footprint at ~120k-clip scale. Concurrent workers processing
    # different clips of the same session may race here; save_undistort_maps writes atomically
    # so that's harmless (redundant compute, not corruption).
    session_dir = session_cache_dir(clip["out_dir"])
    session_dir.mkdir(parents=True, exist_ok=True)
    new_k_path = session_dir / "undistorted_intrinsics.json"
    maps_path = session_dir / "undistort_maps.npz"
    clip["image_folder"] = str(frames_dir)
    try:
        if not new_k_path.exists() or not maps_path.exists():
            first = cv2.imread(sorted(frames_dir.iterdir())[0].as_posix())
            h, w = first.shape[:2]
            map1, map2, roi, result = compute_undistort_maps(
                w, h,
                fx=cam_meta["fx"], fy=cam_meta["fy"], cx=cam_meta["cx"], cy=cam_meta["cy"],
                k1=cam_meta["k1"], k2=cam_meta["k2"], k3=cam_meta["k3"],
                p1=cam_meta["p1"], p2=cam_meta["p2"],
            )
            save_undistort_maps(str(maps_path), map1, map2, roi)
            # Same atomic-write rationale as save_undistort_maps -- a plain write_text here
            # would be racy now that this cache is shared across a session's concurrent workers.
            tmp_k_path = new_k_path.with_name(new_k_path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
            tmp_k_path.write_text(json.dumps(result))
            tmp_k_path.replace(new_k_path)
        else:
            result = json.loads(new_k_path.read_text())
        clip["undistort_maps_path"] = str(maps_path)
        clip["fixed_intrinsics_str"] = (
            f"{result['fx']},{result['fy']},{result['cx']},{result['cy']},"
            f"{result['width']},{result['height']}"
        )
    except Exception:
        # Undistortion failed (bad coefficients, corrupt frame, degenerate crop, ...) -- fall
        # back to raw frames + self-calibration rather than losing the clip entirely.
        clip["undistort_maps_path"] = None
        clip["fixed_intrinsics_str"] = None
        clip["undistort_warning"] = traceback.format_exc()[-2000:]
    return clip, None
