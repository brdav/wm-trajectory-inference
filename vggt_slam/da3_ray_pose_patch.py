"""Monkeypatch for DA3's ray-pose (--use_ray_pose) camera estimation.

DA3's ray-pose path (depth_anything_3.model.da3.DepthAnything3Net._process_ray_pose_estimation,
-> depth_anything_3.utils.ray_utils.get_extrinsic_from_camray -> camray_to_caminfo ->
compute_optimal_rotation_intrinsics_batch) solves a *free* per-frame homography H=Q@L via
RANSAC-weighted DLT, then QL-decomposes H into a rotation Q and a triangular L encoding
(fx, fy, cx, cy). Because H is solved with no constraint tying its K and R apart across frames,
ordinary per-frame ray-prediction noise can land in either factor -- this is the mechanism
behind frame-to-frame K jitter for a single physical camera. Solver's IntrinsicsLock deals with
this today by overriding the RETURNED K after the fact, but that leaves K and R geometrically
inconsistent (R was fit assuming the model's own, different, per-frame K).

This module patches the estimation itself so K and R come out consistent, in two ways:

  - fixed mode (Solver.intrinsics_lock has a fixed_intrinsics K, known from frame 1): skip the
    free homography solve entirely and directly solve for R via a closed-form Kabsch/Wahba fit
    against the known K -- R is no longer free to reinterpret K's error as its own. Single pass,
    since K never changes.

  - two-phase joint calibration (--shared_intrinsics: K unknown but constant across the video):
    main.py runs every submap's DA3 forward pass first (exactly one call per submap, same cost
    as today) with capture enabled -- each call's raw ray-pose material (rays_origin/target/
    weights/z_mask) is stashed instead of discarded, and DA3's own free per-frame K estimate is
    used for that submap in the meantime (nothing is locked yet). Once every submap has been
    seen, `joint_refit` pools every submap's free K estimate into a single median (far better
    conditioned than an online running median that gives the first submap the least-refined
    estimate) and refits every frame's rotation against it via the same closed-form Kabsch solve
    -- cheap batched linear algebra reusing the already-captured data, no second network forward
    pass. Camera center T is estimated independently of K/R (see camray_to_caminfo) and is
    reused as-is; only K and R are replaced.

Math: H = Q @ L (Q rotation, L the internal triangular "K"). Given L fixed/locked, define
s_i = normalize(L @ p_origin_i) (canonical ray transformed by the known camera) and
t_i = normalize(p_target_i) (network's own predicted ray, unchanged). The best-fit Q mapping
s_i -> t_i (maximizing sum of confidence-weighted dot products) is the standard closed-form
Kabsch/Wahba solution via SVD of the weighted cross-covariance matrix.

Fragility note: this reaches into pinned internals of the `depth-anything-3` pip package
(function names, argument order, and the exact L/Q matrix convention were all confirmed against
the actual installed 0.1.1 wheel, not just upstream source) -- an upgrade of that package can
silently break this file. Everything here is monkeypatched at import time; nothing is vendored.
"""
import numpy as np
import torch

from depth_anything_3.model import da3 as _da3
from depth_anything_3.utils import ray_utils as _ray_utils

_ORIGINAL_COMPUTE = _ray_utils.compute_optimal_rotation_intrinsics_batch
_ORIGINAL_PROCESS = _da3.DepthAnything3Net._process_ray_pose_estimation

# Set by DA3Backend immediately before model.inference(...) and cleared immediately after; the
# capture fields are toggled separately by main.py around its whole first pass. Not thread-safe
# -- relies on the pipeline processing one submap's forward pass at a time (true today).
_state = {
    "mode": None, "K_pixel": None, "ref_wh": None, "width": None, "height": None,
    "capture": False, "captured": [],
}


def set_lock(K_pixel, ref_wh, mode="fixed"):
    """K_pixel: (3,3) array, the fixed K, calibrated for ref_wh=(width, height). Call before
    backend.model.inference(...)."""
    assert mode == "fixed"
    _state["mode"] = mode
    _state["K_pixel"] = np.asarray(K_pixel, dtype=np.float64)
    _state["ref_wh"] = (float(ref_wh[0]), float(ref_wh[1]))


def clear_lock():
    _state["mode"] = None
    _state["K_pixel"] = None
    _state["ref_wh"] = None
    _state["width"] = None
    _state["height"] = None


def set_capture(enabled):
    """Toggles raw ray-pose material capture for the two-phase --shared_intrinsics joint fit.
    Call with True before main.py's first pass over all submaps, False after -- this only stops
    NEW entries from being captured; the buffer itself is untouched (joint_refit still needs to
    read it after this call) and is only ever cleared by joint_refit itself once it's done, or
    explicitly via clear_captured() if abandoning a capture pass early."""
    _state["capture"] = enabled


def clear_captured():
    """Explicitly drops anything captured so far, e.g. on an early exit before joint_refit runs."""
    _state["captured"] = []


def discard_last_capture():
    """Pops the most recently captured entry -- for a forward pass that WAS captured (see
    set_capture) but whose result is then rejected before ever being attached to a submap's
    staged predictions, e.g. solver.py's loop-closure match_score check: infer_loop_closure_pair
    runs the model (captured, since it goes through the same patched
    compute_optimal_rotation_intrinsics_batch as any other call) before the caller knows whether
    the match will pass the score threshold. Without this, a rejected loop closure leaves a
    phantom entry in the capture buffer that no staged_predictions dict accounts for, and
    joint_refit's captured-vs-expected count permanently desyncs by one (confirmed: this is the
    root cause of "joint_refit: captured N entries but staged_predictions implies N-1").

    Guarded on _state["capture"] so this is a safe no-op when capture isn't active (the common
    non-two-phase case, or --two_phase's second/third pass) -- the caller doesn't need to know
    whether the just-made forward pass was actually captured. Relies on the pipeline processing
    one forward pass at a time (same non-thread-safety assumption as the rest of this module):
    the loop-closure pair's own capture is always the module's only new entry since the
    submap's own (already-kept) capture, so popping the last entry is always correct here.
    """
    if _state["capture"] and _state["captured"]:
        _state["captured"].pop()


def joint_refit(staged_predictions):
    """Call once, after every submap (and any loop-closure pair) has been through
    backend.infer_submap/infer_loop_closure_pair with capture enabled and NO lock active (so
    each one used DA3's own free per-frame estimate). Mutates each dict in staged_predictions
    (as returned by Solver.run_predictions -- already squeezed to plain numpy) in place:

      1. Pools every submap's own free per-frame K estimate (predictions["intrinsic"], already
         computed during the capture pass -- no need to recompute) into a single median. Loop-
         closure frames are excluded from this median: their 2-view estimate is noisier than a
         full submap's and shouldn't skew the consensus used for every frame.
      2. Refits every frame's rotation (submap and loop-closure alike) against that one
         consensus K, via the closed-form Kabsch solve, using the raw ray-pose data captured
         during the (already-run) forward passes -- no second network forward pass.
      3. Each frame's own camera center T is preserved exactly (estimated independently of K/R,
         see camray_to_caminfo, so refitting R doesn't invalidate it).

    Returns the final (3, 3) consensus K. Clears the capture buffer.
    """
    captured = _state["captured"]

    rows = []
    for pred in staged_predictions:
        for K in pred["intrinsic"]:
            rows.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    if not rows:
        raise ValueError("joint_refit: no staged predictions to pool a consensus K from")
    fx, fy, cx, cy = np.median(np.asarray(rows), axis=0)
    K_final = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    expected = sum(1 + (1 if "intrinsic_lc" in p else 0) for p in staged_predictions)
    if len(captured) != expected:
        counts = [(p["intrinsic"].shape[0], "intrinsic_lc" in p) for p in staged_predictions]
        cap_sizes = [e["rays_origin"].shape[0] for e in captured]
        raise ValueError(
            f"joint_refit: captured {len(captured)} entries but staged_predictions implies "
            f"{expected} (len(staged_predictions)={len(staged_predictions)}, "
            f"per-submap (S, has_lc)={counts}, captured entry sizes={cap_sizes})"
        )

    idx = 0
    for pred in staged_predictions:
        entry = captured[idx]
        idx += 1
        pred["intrinsic"], pred["extrinsic"] = _refit_one(pred["intrinsic"], pred["extrinsic"], entry, K_final)
        if "intrinsic_lc" in pred:
            entry_lc = captured[idx]
            idx += 1
            pred["intrinsic_lc"], pred["extrinsic_lc"] = _refit_one(
                pred["intrinsic_lc"], pred["extrinsic_lc"], entry_lc, K_final)

    _state["captured"] = []
    return K_final


def _refit_one(intrinsic, extrinsic, entry, K_final):
    """intrinsic: (S,3,3), extrinsic: (S,3,4) c2w numpy arrays (Solver.run_predictions'
    convention). entry: this call's captured raw ray-pose material. Returns (new_intrinsic,
    new_extrinsic) with K_final broadcast and R refit via Kabsch; T (translation) is untouched."""
    width, height = entry["width"], entry["height"]
    f0, f1, pp0, pp1 = _build_L_scalars(K_final, (width, height), width, height)
    R_final = _solve_locked(entry["rays_origin"], entry["rays_target"], entry["weights"], entry["z_mask"],
                             f0, f1, pp0, pp1)
    R_final = R_final.detach().cpu().numpy()

    S = intrinsic.shape[0]
    new_intrinsic = np.tile(K_final[None, :, :], (S, 1, 1)).astype(intrinsic.dtype)
    new_extrinsic = extrinsic.copy()
    for i in range(S):
        c2w = np.eye(4)
        c2w[:3, :4] = extrinsic[i]
        w2c = np.linalg.inv(c2w)
        w2c[:3, :3] = R_final[i]
        new_extrinsic[i] = np.linalg.inv(w2c)[:3, :4]
    return new_intrinsic, new_extrinsic


def _build_L_scalars(K_pixel, ref_wh, width, height):
    """Converts a pixel-space K (calibrated for ref_wh) into the (f0, f1, pp0, pp1) scalars
    DA3's internal "imw=imh=2.0" normalized-plane convention uses, for a call whose actual
    resized resolution is (width, height). Inverse of da3.py's
    `_process_ray_pose_estimation`'s own pred_intrinsic[...] = .../2*width etc conversion."""
    ref_w, ref_h = ref_wh
    sx, sy = width / ref_w, height / ref_h
    fx, fy = K_pixel[0, 0] * sx, K_pixel[1, 1] * sy
    cx, cy = K_pixel[0, 2] * sx, K_pixel[1, 2] * sy
    f0 = width / (2.0 * fx)
    f1 = height / (2.0 * fy)
    pp0 = 2.0 * cx / width - 1.0
    pp1 = 2.0 * cy / height - 1.0
    return f0, f1, pp0, pp1


def _kabsch_batch(s, t, w):
    """s, t: (B, N, 3) unit vectors; w: (B, N) non-negative weights. Returns R: (B, 3, 3)
    maximizing sum_n w[b,n] * (R[b] @ s[b,n]) . t[b,n], the standard closed-form Kabsch/Wahba
    rotation-alignment solution (computed in float32 regardless of input dtype -- SVD support/
    stability for fp16/bf16 is inconsistent, and this is cheap relative to the network forward)."""
    orig_dtype = s.dtype
    s32, t32, w32 = s.float(), t.float(), w.float()
    wn = w32 / w32.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    N_mat = torch.einsum("bn,bni,bnj->bij", wn, t32, s32)  # sum wn * outer(t, s)
    U, _, Vh = torch.linalg.svd(N_mat)
    V = Vh.transpose(-1, -2)
    d = torch.det(torch.matmul(U, V.transpose(-1, -2)))
    D = torch.eye(3, device=s.device, dtype=torch.float32).unsqueeze(0).repeat(s.shape[0], 1, 1)
    D[:, 2, 2] = d
    R = torch.matmul(torch.matmul(U, D), V.transpose(-1, -2))
    return R.to(orig_dtype)


def _solve_locked(rays_origin, rays_target, weights, z_mask, f0, f1, pp0, pp1):
    """Mirrors the preprocessing in the original compute_optimal_rotation_intrinsics_batch
    (z-divide by the 3rd/depth component, keep only the first 2 dims) but replaces the free
    RANSAC homography + QL decomposition with the closed-form fit against a KNOWN L."""
    device, dtype = rays_origin.device, rays_origin.dtype
    B, N, _ = rays_origin.shape
    ro, rt = rays_origin.clone(), rays_target.clone()
    ro[:, :, 0][z_mask] /= ro[:, :, 2][z_mask]
    ro[:, :, 1][z_mask] /= ro[:, :, 2][z_mask]
    rt[:, :, 0][z_mask] /= rt[:, :, 2][z_mask]
    rt[:, :, 1][z_mask] /= rt[:, :, 2][z_mask]

    ones = torch.ones(B, N, 1, device=device, dtype=dtype)
    p_origin_h = torch.cat([ro[:, :, :2], ones], dim=-1)  # (B, N, 3)
    p_target_h = torch.cat([rt[:, :, :2], ones], dim=-1)  # (B, N, 3)

    L = torch.tensor([[f0, 0.0, 0.0], [0.0, f1, 0.0], [pp0, pp1, 1.0]], device=device, dtype=dtype)
    s = torch.einsum("ij,bnj->bni", L, p_origin_h)
    s = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t = p_target_h / p_target_h.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return _kabsch_batch(s, t, weights)


def _patched_compute(
    rays_origin, rays_target, z_threshold=1e-4, reproj_threshold=0.2, weights=None,
    n_sample=None, n_iter=100, num_sample_for_ransac=8, rand_sample_iters_idx=None,
):
    mode = _state["mode"]
    capture = _state["capture"]

    if mode is None and not capture:
        return _ORIGINAL_COMPUTE(
            rays_origin, rays_target, z_threshold=z_threshold, reproj_threshold=reproj_threshold,
            weights=weights, n_sample=n_sample, n_iter=n_iter,
            num_sample_for_ransac=num_sample_for_ransac, rand_sample_iters_idx=rand_sample_iters_idx,
        )

    assert weights is not None, "weights must be provided"
    # Replicate the original function's z-mask + IN-PLACE weight zeroing exactly: the caller
    # (camray_to_caminfo) reuses this same `weights`/`confidence` tensor afterward for its
    # camera-center (T) estimate and relies on this mutation regardless of which path runs here.
    z_mask = torch.logical_and(
        torch.abs(rays_target[:, :, 2]) > z_threshold, torch.abs(rays_origin[:, :, 2]) > z_threshold
    )
    weights[~z_mask] = 0

    if capture:
        _state["captured"].append({
            "rays_origin": rays_origin.detach().clone(),
            "rays_target": rays_target.detach().clone(),
            "weights": weights.detach().clone(),
            "z_mask": z_mask.detach().clone(),
            "width": _state["width"],
            "height": _state["height"],
        })

    if mode is None:
        # Capture-only: still use DA3's own free estimate for this (soon to be refit) call.
        return _ORIGINAL_COMPUTE(
            rays_origin, rays_target, z_threshold=z_threshold, reproj_threshold=reproj_threshold,
            weights=weights, n_sample=n_sample, n_iter=n_iter,
            num_sample_for_ransac=num_sample_for_ransac, rand_sample_iters_idx=rand_sample_iters_idx,
        )

    width, height = _state["width"], _state["height"]
    f0, f1, pp0, pp1 = _build_L_scalars(_state["K_pixel"], _state["ref_wh"], width, height)
    R_locked = _solve_locked(rays_origin, rays_target, weights, z_mask, f0, f1, pp0, pp1)
    f_locked = torch.tensor([f0, f1], device=rays_origin.device, dtype=rays_origin.dtype)
    f_locked = f_locked.unsqueeze(0).expand(R_locked.shape[0], 2).clone()
    pp_locked = torch.tensor([pp0, pp1], device=rays_origin.device, dtype=rays_origin.dtype)
    pp_locked = pp_locked.unsqueeze(0).expand(R_locked.shape[0], 2).clone()
    return R_locked, f_locked, pp_locked


def _patched_process(self, output, height, width):
    if _state["mode"] is not None or _state["capture"]:
        _state["width"], _state["height"] = width, height
    return _ORIGINAL_PROCESS(self, output, height, width)


def install():
    """Idempotent -- safe to call more than once (always reassigns to the same wrappers, which
    close over the ORIGINAL functions captured once at module import time)."""
    _ray_utils.compute_optimal_rotation_intrinsics_batch = _patched_compute
    _da3.DepthAnything3Net._process_ray_pose_estimation = _patched_process
