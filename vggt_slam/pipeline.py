"""SLAM stage for one clip: keyframe selection, submap SLAM, GPS fusion. Used by scripts/slam/batch_infer.py
(and scripts/metric_scale/estimate_metric_scale.py), which load the model once per GPU process and call run_video per clip."""
import glob
import os
import queue
import sys
import threading
import time
import traceback

import cv2
import torch
from tqdm.auto import tqdm

import vggt_slam.slam_utils as utils
from vggt_slam.fusion import run_gps_fusion
from vggt_slam.intrinsics_lock import parse_fixed_intrinsics
from vggt_slam.map import GraphMap
from vggt_slam.model_backends import DA3Backend, VGGTBackend
from vggt_slam.pipeline_params import SlamParams  # noqa: F401  (re-exported)
from vggt_slam.solver import Solver
from vggt_slam.undistort import load_undistort_maps, undistort_frame


def build_backend(params, device):
    """Load the feed-forward geometry model once, so a batch driver can reuse it across videos."""
    print(f"Initializing and loading {params.model_backend} model...")
    if params.model_backend == "vggt":
        from vggt.models.vggt import VGGT

        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
        model.eval()
        model = model.to(torch.bfloat16).to(device)
        return VGGTBackend(model)
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(params.da3_checkpoint)
    model.eval()
    model = model.to(device=device)
    return DA3Backend(model, use_ray_pose=params.use_ray_pose, ref_view_strategy=params.ref_view_strategy)


def _produce_submap_batches(out_queue, solver, image_names, params, undistort_maps):
    """Producer thread: read frames, select keyframes by optical-flow disparity, and hand each
    submap's worth of keyframes (with their decoded RGB arrays) to the main thread over `out_queue`.

    Reading + disparity checks are CPU work, running a submap is GPU work; overlapping them keeps the
    GPU from idling while the next batch is prepared (both sides spend most of their time in code that
    releases the GIL: cv2 and CUDA calls). The queue's maxsize bounds how far this runs ahead.
    Puts ("batch", names, frame_cache, count) per submap, then ("done", image_count, keyframe_time),
    or ("error", traceback) if anything raises."""
    map1, map2, roi = undistort_maps if undistort_maps is not None else (None, None, None)
    image_names_subset = []
    image_count = 0
    keyframe_time = utils.Accumulator()
    count = 0
    # image_name -> decoded RGB array, handed to DA3 so it does not re-read and re-decode the frame
    # (cv2 reads BGR; DA3's array path assumes RGB, so the cached copy is converted once here)
    frame_cache = {}
    try:
        for image_name in tqdm(image_names):
            with keyframe_time:
                img = cv2.imread(image_name)
                # disparity is a motion heuristic, checked on the frame as read (possibly still distorted)
                if solver.flow_tracker.compute_disparity(img, params.min_disparity):
                    image_names_subset.append(image_name)
                    if map1 is not None:
                        img = undistort_frame(img, map1, map2, roi)
                    frame_cache[image_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image_count += 1

            if len(image_names_subset) == params.submap_size + params.overlapping_window_size or image_name == image_names[-1]:
                count += 1
                batch_cache = {k: frame_cache[k] for k in image_names_subset if k in frame_cache}
                out_queue.put(("batch", list(image_names_subset), batch_cache, count))
                # keep the overlap frames for the next submap
                image_names_subset = image_names_subset[-params.overlapping_window_size:]
                frame_cache = {k: frame_cache[k] for k in image_names_subset if k in frame_cache}
        out_queue.put(("done", image_count, keyframe_time))
    except Exception:
        out_queue.put(("error", traceback.format_exc()))


def run_slam(params, backend, image_folder, fixed_intrinsics=None, shared_intrinsics=False, undistort_maps_path=None):
    """SLAM over one clip's extracted frames in `image_folder`. Returns the Solver holding the map and pose graph.

    fixed_intrinsics: "fx,fy,cx,cy,width,height" known calibration, used for every frame.
    shared_intrinsics: K unknown but constant over the clip: jointly refit one whole-video K (DA3 ray-pose
        only). Mutually exclusive with fixed_intrinsics.
    undistort_maps_path: .npz of remap tables; image_folder then holds the RAW frames and only the
        selected keyframes are undistorted, in memory (fixed_intrinsics must describe the rectified geometry)."""
    if fixed_intrinsics is not None and shared_intrinsics:
        raise ValueError("fixed_intrinsics and shared_intrinsics are mutually exclusive")
    # shared_intrinsics is implemented only as DA3's two-phase ray-pose joint calibration (see
    # vggt_slam/da3_ray_pose_patch.py): the whole video is available upfront, so every submap's
    # estimate is pooled into one whole-video median
    if shared_intrinsics and not (params.model_backend == "da3" and params.use_ray_pose):
        raise ValueError("shared_intrinsics requires model_backend da3 with use_ray_pose "
                         "(the only path with a joint two-phase K+R refit; see vggt_slam/da3_ray_pose_patch.py)")
    # VGGTBackend reads frames from disk by path and never sees the in-memory undistorted frame_cache
    if undistort_maps_path is not None and params.model_backend != "da3":
        raise ValueError("undistort_maps_path requires model_backend da3 (VGGTBackend would silently "
                         "read the still-distorted frame from disk)")
    two_phase = shared_intrinsics

    fixed_K, fixed_K_size = parse_fixed_intrinsics(fixed_intrinsics)
    solver = Solver(
        init_conf_threshold=params.conf_threshold,
        lc_thres=params.lc_thres,
        overlapping_window_size=params.overlapping_window_size,
        fixed_intrinsics=fixed_K,
        fixed_intrinsics_size=fixed_K_size,
    )

    if two_phase:
        from vggt_slam import da3_ray_pose_patch
        da3_ray_pose_patch.set_capture(True)
        # solver.map must stay empty until every submap has been refit (add_edge's "previous submap"
        # logic needs it to hold only committed submaps), but loop-closure retrieval still has to see
        # every submap processed so far, so phase 1 fills a separate map for that
        solver.retrieval_map = GraphMap()

    print(f"Loading images from {image_folder}...")
    image_names = utils.sort_images_by_number(glob.glob(os.path.join(image_folder, "*.jpg")))
    print(f"Found {len(image_names)} images")

    undistort_maps = None
    if undistort_maps_path is not None:
        undistort_maps = load_undistort_maps(undistort_maps_path)
        # so the backend's own path-based re-read of loop-closure frames rectifies too
        backend.set_undistort_maps(*undistort_maps)

    total_time_start = time.time()
    backend_time = utils.Accumulator()
    staged_predictions = []  # two_phase only: submaps run through DA3 but not yet added to the map

    batch_queue = queue.Queue(maxsize=2)
    producer_thread = threading.Thread(
        target=_produce_submap_batches, args=(batch_queue, solver, image_names, params, undistort_maps), daemon=True)
    producer_thread.start()

    while True:
        item = batch_queue.get()
        if item[0] == "error":
            producer_thread.join()
            raise RuntimeError(f"Frame reading/keyframe-selection thread failed:\n{item[1]}")
        if item[0] == "done":
            image_count, keyframe_time = item[1], item[2]
            break
        _, image_names_subset, frame_cache, count = item

        print(image_names_subset)
        t1 = time.time()
        predictions = solver.run_predictions(image_names_subset, backend, params.max_loops, frame_cache=frame_cache)
        print("Solver total time", time.time() - t1)
        print(count, "submaps processed")

        if two_phase:
            # defer add_points / graph.optimize() until joint_refit has replaced the per-frame K and R;
            # register the submap for loop-closure retrieval right away
            staged_predictions.append(predictions)
            solver.retrieval_map.add_submap(predictions["_submap"])
        else:
            solver.add_points(predictions)
            with backend_time:
                solver.graph.optimize()

    producer_thread.join()

    if two_phase:
        da3_ray_pose_patch.set_capture(False)
        print(f"Two-phase shared intrinsics: jointly refitting K, R across {len(staged_predictions)} submaps "
              f"({sum(p['intrinsic'].shape[0] for p in staged_predictions)} frames)...")
        t1 = time.time()
        try:
            K_final = da3_ray_pose_patch.joint_refit(staged_predictions)
        finally:
            # GPU workers are long-lived: a capture buffer left dirty by a failed refit would poison
            # every later shared-intrinsics clip this worker picks up
            da3_ray_pose_patch.clear_captured()
        print(f"Joint refit done in {time.time() - t1:.2f}s. Consensus K:\n{K_final}")

        for predictions in staged_predictions:
            solver.add_points(predictions)
            with backend_time:
                solver.graph.optimize()

    total_time = time.time() - total_time_start
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for backend model calls: {solver.backend_timer.total_time:.4f}s")
    print("Average backend model time per frame:", solver.backend_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", image_count / total_time)
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())
    return solver


def run_video(params, backend, image_folder, log_path, video, extraction_fps, fixed_intrinsics=None,
              shared_intrinsics=False, undistort_maps_path=None, dense_log=False, gps_csv=None,
              osm_region_pbf_path=None, osm_fallback_region_pbf_paths=None, gps_max_dt=0.5):
    """Process one clip: run_slam on its frames, save the trajectory to `log_path` (result.npz), and,
    with `gps_csv`, fuse it with GPS and the road network and save again.

    The SLAM result is written first, so it survives even if fusion kills the process. Fusion is
    enrichment: its failure is recorded, not raised. Returns the gps_fusion_info dict (None when
    no `gps_csv`), which lands in manifest.jsonl.

    video: source video path stored in result.npz. extraction_fps: fps the frames were extracted at (aligns
    keyframes to GPS fixes by elapsed time). dense_log: also write the point cloud next to `log_path`.
    The other arguments are described at run_slam and vggt_slam.fusion.run_gps_fusion."""
    solver = run_slam(params, backend, image_folder, fixed_intrinsics, shared_intrinsics, undistort_maps_path)

    frame_ids, rotations, translations = solver.map.get_all_poses_and_frame_ids(solver.graph)
    utils.save_result_npz(log_path, frame_ids, rotations, translations, extraction_fps, video)
    if dense_log:
        solver.map.write_points_to_file(solver.graph, os.path.splitext(log_path)[0] + "_points.pcd")
    if gps_csv is None:
        return None

    network_dir = os.path.join(os.path.dirname(log_path) or ".", "osm_network")
    gps_fusion_info, fused = run_gps_fusion(
        rotations, translations, frame_ids, extraction_fps, gps_csv, network_dir, osm_region_pbf_path,
        osm_fallback_region_pbf_paths, max_dt=gps_max_dt, label=str(video))
    fused = fused or {}
    # rewrites the same result.npz: slam_* fields unchanged, plus whatever fusion produced
    utils.save_result_npz(
        log_path, frame_ids, rotations, translations, extraction_fps, video,
        fused_rotations=fused.get("rotations"), fused_translations=fused.get("translations"),
        matched_line=fused.get("matched_line"), snapped=fused.get("snapped"), gps_fusion_info=gps_fusion_info)
    return gps_fusion_info
