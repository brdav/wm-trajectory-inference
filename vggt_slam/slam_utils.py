import json
import os
import re
import numpy as np
import matplotlib
import scipy
import time
from PIL import Image
from torchvision import transforms as TF
from scipy.spatial.transform import Rotation
import torch


def _poses_and_quats(rotations, translations):
    translations = np.asarray(translations)
    quaternions = Rotation.from_matrix(rotations).as_quat()  # x, y, z, w
    poses = np.tile(np.eye(4), (len(translations), 1, 1))
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    return poses, quaternions


def _atomic_savez(path, **fields):
    """np.savez to a temp file in the same directory, then os.replace() into path -- so a
    process killed mid-write (this session hit real SIGKILLs from cluster instability) never
    leaves a truncated/corrupt result file behind, and two workers racing to populate the same
    shared cache path (see vggt_slam/undistort.py's session-level cache) can't produce a torn
    read either. Redundant recomputation in that race is harmless since the output is
    deterministic; this only prevents a corrupted partial read."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "wb") as f:
        np.savez(f, **fields)
    os.replace(tmp_path, path)


def save_result_npz(path, frame_ids, slam_rotations, slam_translations, extraction_fps, video,
                     fused_rotations=None, fused_translations=None, matched_line=None,
                     snapped=None, gps_fusion_info=None, metric_scale=None, metric_poses=None,
                     metric_scale_info=None):
    """Writes one consolidated per-clip result file: the raw SLAM trajectory plus (once
    available) the GPS-fused trajectory, matched route, and per-point snap -- replaces the old
    poses.npz + poses_gps_fused.npz + poses_gps_route.wkt + poses_gps_snapped.npy, whose file
    count (plus a since-removed per-clip osm_network/ scratch subtree) was adding up to ~1-2M
    inodes across the ~120k-clip dataset and visibly straining the shared filesystem.

    Always writes frame_ids/slam_poses/slam_quaternions/extraction_fps/video. The
    fused_*/matched_line/snapped/gps_fusion_info arguments are optional and simply omitted from
    the file when None -- callers write this file TWICE per clip: once right after SLAM with
    only the always-present fields (so the SLAM result survives even if GPS fusion later crashes
    the whole process), and again after GPS fusion finishes (success or failure) with the
    slam_* arguments unchanged and whichever fusion fields are now available. See
    load_result_npz for the read side and exact optional-field semantics.

    metric_scale / metric_poses / metric_scale_info: the depth-based metric scale from
    scripts/estimate_metric_scale.py (for clips without usable GPS fusion): meters per SLAM unit,
    the slam poses with translation * metric_scale (N,4,4), and a dict of provenance/quality
    stats (stored as JSON). Kept separate from fused_* on purpose: fused_poses are GPS/road
    grounded, these are an estimate from monocular metric depth. Anything that rewrites an
    existing result.npz (e.g. scripts/tools/refuse_gps.py) must pass these through or they are lost.

    gps_fusion_info: the same dict already returned to callers and written to manifest.jsonl
    (status, region_used, network_profile_used, matched_keyframes, total_keyframes,
    map_match_error_mean_m/max_m/n_points, error) -- stored as JSON so the file is
    self-describing without cross-referencing a manifest.
    """
    frame_ids = np.asarray(frame_ids, dtype=float)
    slam_poses, slam_quaternions = _poses_and_quats(slam_rotations, slam_translations)

    fields = dict(
        frame_ids=frame_ids, slam_poses=slam_poses, slam_quaternions=slam_quaternions,
        extraction_fps=np.asarray(extraction_fps, dtype=float), video=np.asarray(str(video)),
    )
    if fused_rotations is not None:
        fused_poses, fused_quaternions = _poses_and_quats(fused_rotations, fused_translations)
        fields["fused_poses"] = fused_poses
        fields["fused_quaternions"] = fused_quaternions
    if matched_line is not None:
        fields["matched_route_wkt"] = np.asarray(matched_line.wkt)
    if snapped is not None:
        fields["snapped"] = np.asarray(snapped)
    if gps_fusion_info is not None:
        fields["gps_fusion_json"] = np.asarray(json.dumps(gps_fusion_info))
    if metric_scale is not None:
        fields["metric_scale"] = np.asarray(metric_scale, dtype=float)
    if metric_poses is not None:
        fields["metric_poses"] = np.asarray(metric_poses)
    if metric_scale_info is not None:
        fields["metric_scale_json"] = np.asarray(json.dumps(metric_scale_info))

    _atomic_savez(path, **fields)


def load_result_npz(path):
    """Reads a consolidated result.npz (see save_result_npz). Always-present keys:
    frame_ids/slam_rotations/slam_translations/extraction_fps/video. The GPS-fusion keys
    (fused_rotations/fused_translations/snapped/matched_route_wkt/gps_fusion_info) are None
    when that clip's fusion wasn't attempted or didn't reach that stage -- mirrors the exact
    three-tier semantics the old separate-files layout expressed via file existence (SLAM-only /
    SLAM+placement-no-route / full success), just via dict values instead of sibling files.
    metric_scale/metric_poses/metric_scale_info are None unless scripts/estimate_metric_scale.py's
    result was integrated for this clip (see scripts/integrate_metric_scale.py)."""
    data = np.load(path)
    out = {
        "frame_ids": data["frame_ids"],
        "slam_rotations": data["slam_poses"][:, :3, :3],
        "slam_translations": data["slam_poses"][:, :3, 3],
        "extraction_fps": float(data["extraction_fps"]),
        "video": str(data["video"]),
        "fused_rotations": None, "fused_translations": None,
        "snapped": None, "matched_route_wkt": None, "gps_fusion_info": None,
        "metric_scale": None, "metric_poses": None, "metric_scale_info": None,
    }
    if "fused_poses" in data:
        out["fused_rotations"] = data["fused_poses"][:, :3, :3]
        out["fused_translations"] = data["fused_poses"][:, :3, 3]
    if "snapped" in data:
        out["snapped"] = data["snapped"]
    if "matched_route_wkt" in data:
        out["matched_route_wkt"] = str(data["matched_route_wkt"])
    if "gps_fusion_json" in data:
        out["gps_fusion_info"] = json.loads(str(data["gps_fusion_json"]))
    if "metric_scale" in data:
        out["metric_scale"] = float(data["metric_scale"])
    if "metric_poses" in data:
        out["metric_poses"] = data["metric_poses"]
    if "metric_scale_json" in data:
        out["metric_scale_info"] = json.loads(str(data["metric_scale_json"]))
    return out


def slice_with_overlap(lst, n, k):
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)  # Ensure progress even if k >= n
    return result


def sort_images_by_number(image_paths):
    def extract_number(path):
        filename = os.path.basename(path)
        # Look for digits (with optional decimal), followed by any non-digit characters, then the extension
        match = re.search(r'\d+(?:\.\d+)?(?=[^\d]*\.[^.]+$)', filename)
        return float(match.group()) if match else float('inf')
        
    return sorted(image_paths, key=extract_number)

def downsample_images(image_names, downsample_factor):
    """
    Downsamples a list of image names by keeping every `downsample_factor`-th image.
    
    Args:
        image_names (list of str): List of image filenames.
        downsample_factor (int): Factor to downsample the list. E.g., 2 keeps every other image.

    Returns:
        list of str: Downsampled list of image filenames.
    """
    return image_names[::downsample_factor]

def decompose_camera(P, no_inverse=False):
    """
    Decompose a 3x4 or 4x4 camera projection matrix P into intrinsics K,
    rotation R, and translation t.
    """
    if P.shape[0] != 3:
        P = P / P[-1,-1]
        P = P[0:3, :]

    # Ensure P is (3,4)
    assert P.shape == (3, 4)

    # Left 3x3 part
    M = P[:, :3]

    # RQ decomposition
    K, R = scipy.linalg.rq(M)

    # Make sure intrinsics have positive diagonal
    if K[0,0] < 0:
        K[:,0] *= -1
        R[0,:] *= -1
    if K[1,1] < 0:
        K[:,1] *= -1
        R[1,:] *= -1
    if K[2,2] < 0:
        K[:,2] *= -1
        R[2,:] *= -1

    scale = K[2,2]
    # print("Scale factor from K[2,2]:", scale)
    if not no_inverse:
        R = np.linalg.inv(R)
        t = -R @ np.linalg.inv(K) @ P[:, 3]
    else:
        t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale
    
    return K, R, t, scale

def compute_image_embeddings(model, preprocess, image_paths, batch_size=64, device="cuda"):
    all_embs = []

    # Load all images into memory (PIL -> tensor)
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))

    # Stack into a single tensor
    imgs = torch.stack(imgs).to(device)

    # Loop over batches
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())

    # Combine into one (N, D) array
    return torch.cat(all_embs, dim=0).numpy()

def compute_text_embeddings(clip_model, tokenizer, text, device="cuda"):
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.cpu().numpy()

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors a and b.
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T

def normalize_to_sl4(H):
    """
    Normalize a 4x4 homography matrix H to be in SL(4).
    """
    det = np.linalg.det(H)
    if det == 0:
        raise ValueError("Homography matrix is singular and cannot be normalized.")
    scale = det ** (1/4)
    H_normalized = H / scale
    return H_normalized

def compute_obb_from_points(points: np.ndarray):
    """
    Compute an oriented bounding box (OBB) for a Nx3 point cloud.

    Returns:
        center      : (3,) world-space center of OBB
        extent      : (3,) lengths of OBB along its principal axes
        rotation    : (3,3) rotation matrix (columns = principal axes)
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be Nx3 points"

    # Remove NaN/inf if any
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty or invalid")

    # 1. Compute centroid
    centroid = points.mean(axis=0)

    # 2. PCA on centered points
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    # Eigen decomposition (sorted by eigenvalue descending)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    rotation = eigvecs  # columns = principal axes (R)

    # 3. Project points to PCA frame
    points_local = centered @ rotation

    # 4. Compute min/max in PCA frame → box extents
    min_corner = points_local.min(axis=0)
    max_corner = points_local.max(axis=0)
    extent = max_corner - min_corner

    # 5. Compute box center (local), then world
    center_local = 0.5 * (min_corner + max_corner)
    center_world = centroid + center_local @ rotation.T

    return center_world, extent, rotation

def overlay_masks(image, masks):
    image = image.convert("RGBA")

    # masks: (N, 1, H, W) or (N, H, W)
    masks = (255 * masks.cpu().numpy()).astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        # Ensure mask is 2D
        mask = np.squeeze(mask)
        # Now mask is shape (H, W)

        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image

class Accumulator:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.total_time += (time.perf_counter() - self.start)
