"""Undistort frames using a known pinhole K + lens distortion, without ever writing a
second (undistorted) copy of the frames to disk.

Standard OpenCV Brown-Conrady model: distCoeffs = (k1, k2, p1, p2, k3). The remap tables only
depend on a clip's resolution and known camera intrinsics -- not on any frame's pixel content --
so they're computed once from just (width, height) via compute_undistort_maps(), and applied
in-memory, per frame, via undistort_frame(). This lets a caller (vggt_slam/pipeline.py's keyframe-selection
loop) undistort only the frames it actually keeps as keyframes, instead of every extracted frame.

Used by scripts/slam/batch_infer.py (computes+caches the maps once per recording session -- see
session_cache_dir -- and vggt_slam/pipeline.py (loads the cached maps and applies them per selected keyframe).
"""
import os
import threading

import cv2
import numpy as np


def session_cache_dir(out_dir):
    """Where the undistort-map cache for out_dir's recording session lives, given a clip's own
    out_dir (out_root / rel_video_path_without_suffix).

    Every clip (FRONT_*.mp4) under the same session directory (two levels above the clip, same
    place _resolve_camera_meta's FIXED_METADATA.json lookup points at) shares identical camera
    calibration, so the remap tables/intrinsics are identical too -- confirmed directly (Croatia:
    957 sessions, avg 6.0 clips/session, up to 73). Caching this once per session instead of once
    per clip avoids redundantly recomputing and storing the same ~10MB file for every sub-clip of
    one drive. out_dir's own parent hierarchy mirrors the dataset video_path's exactly (only the
    final path component's .mp4 suffix is ever dropped), so out_dir.parents[2] lands on the same
    session directory, just rooted under out_root instead of the read-only dataset_root.
    """
    return out_dir.parents[2]


def compute_undistort_maps(width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, alpha=0.0):
    """Computes the undistortion remap tables once for a clip's resolution + known distortion.
    alpha=0 crops to the largest all-valid-pixel rectangle (no black border pixels reach the SLAM
    backend). Returns (map1, map2, roi, new_intrinsics), where roi=(x, y, w, h) is the alpha crop
    and new_intrinsics = {"fx", "fy", "cx", "cy", "width", "height"} already accounts for it (so
    it's directly usable as e.g. vggt_slam/pipeline.py's --fixed_intrinsics).
    """
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), alpha)
    x, y, roi_w, roi_h = roi
    if roi_w <= 0 or roi_h <= 0:
        raise ValueError(
            f"getOptimalNewCameraMatrix returned an empty valid region (roi={roi}) -- "
            "distortion coefficients or alpha are likely wrong for this image size"
        )
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (width, height), cv2.CV_32FC1)

    new_intrinsics = {
        "fx": float(new_K[0, 0]), "fy": float(new_K[1, 1]),
        "cx": float(new_K[0, 2] - x), "cy": float(new_K[1, 2] - y),
        "width": int(roi_w), "height": int(roi_h),
    }
    return map1, map2, (int(x), int(y), int(roi_w), int(roi_h)), new_intrinsics


def undistort_frame(img, map1, map2, roi):
    """Applies a precomputed remap + alpha crop to one already-decoded (in-memory) frame."""
    x, y, roi_w, roi_h = roi
    undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undist[y:y + roi_h, x:x + roi_w]


def save_undistort_maps(path, map1, map2, roi):
    """Write-then-atomic-rename (temp file in the same directory, then os.replace()) rather
    than a direct np.savez to path: this cache is now shared across every clip in a recording
    session (see session_cache_dir), so multiple workers can race to populate it for different
    clips at once. A direct write is racy -- np.savez writes several arrays non-atomically, so a
    concurrent reader could observe a torn file. Redundant computation in that race is harmless
    (deterministic output; whichever write finishes last just wins), this only prevents a
    corrupted partial read.
    """
    path = str(path)
    # unique per thread as well as per process: scripts/slam/batch_infer.py extracts clips in a thread pool inside one
    # process, and two threads of the same session used to share one temp name (the second os.replace then found
    # no file and that clip silently fell back to self-calibration)
    tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp_path, "wb") as f:
        np.savez(f, map1=map1, map2=map2, roi=np.array(roi, dtype=np.int64))
    os.replace(tmp_path, path)


def load_undistort_maps(path):
    data = np.load(path)
    return data["map1"], data["map2"], tuple(int(v) for v in data["roi"])
