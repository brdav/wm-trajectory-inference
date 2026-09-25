import os
import glob
import queue
import sys
import threading
import time
import traceback
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap
from vggt_slam.map import GraphMap
from vggt_slam.model_backends import VGGTBackend, DA3Backend
from vggt_slam.intrinsics_lock import parse_fixed_intrinsics
from vggt_slam.undistort import load_undistort_maps, undistort_frame
from vggt_slam import gps as gps_module
from vggt_slam.gps_fusion import align_trajectory_to_roads
from vggt_slam.map_match import trim_osm_network_cache

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--model_backend", type=str, default="da3", choices=["vggt", "da3"], help="Feed-forward geometry model to use: VGGT (original) or Depth Anything 3. Default da3, since DA3-GIANT-1.1 outperformed VGGT on GPS-fit trajectory error across all three natix clips tested (see conversation notes / gps_fit_*.png)")
parser.add_argument("--da3_checkpoint", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 checkpoint: a local directory or a Hugging Face repo id. Only used when --model_backend da3. Default DA3-GIANT-1.1 (plain any-view, no metric claim) -- DA3NESTED-GIANT-LARGE-1.1's 'metric' scale was empirically unreliable (1.3x-4x off) and DA3-GIANT-1.1's trajectory shape was more accurate anyway once fairly scale-fit")
parser.add_argument("--use_ray_pose", action=argparse.BooleanOptionalAction, default=True, help="Passed through to DepthAnything3.inference: use ray-based pose estimation instead of the camera decoder. Default on, since it's a prerequisite for vggt_slam/da3_ray_pose_patch.py's --fixed_intrinsics/--shared_intrinsics support (fitting rotation directly against the known/consensus K instead of DA3's own free per-frame estimate); pass --no-use_ray_pose to fall back to the camera decoder. Only used when --model_backend da3")
parser.add_argument("--ref_view_strategy", type=str, default="saddle_balanced", choices=["first", "middle", "saddle_balanced", "saddle_sim_range"], help="Passed through to DepthAnything3.inference: strategy for selecting the reference view from multiple views. Only used when --model_backend da3")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--headless", action="store_true", help="Skip the viser viewer entirely -- no server, no point-cloud/pose pushes at all. For batch runs where nobody watches it; overrides --vis_map.")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="result.npz", help="Path to save the log file")
parser.add_argument("--video", type=str, default=None, help="Source video path, saved into --log_path's npz (if --log_path ends in .npz) for later GPS/frame-number alignment. Only used with a .npz --log_path")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="Number of overlapping frames shared between consecutive submaps, used in SL(4) scale/pose estimation between them -- higher values average the scale estimate over more shared frames and add one between-factor per extra shared frame to the pose graph")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=25, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--fixed_intrinsics", type=str, default=None, help="Comma-separated known camera intrinsics \"fx,fy,cx,cy,width,height\" (zero skew), where width,height is the resolution this K was calibrated for (e.g. the original video's) -- it gets rescaled to whatever resolution the backend actually processes each call at (backends resize internally, so this is NOT the same as the resolution in --image_folder). Use this K for every frame instead of the backend's own per-frame estimate. Mutually exclusive with --shared_intrinsics")
parser.add_argument("--shared_intrinsics", action="store_true", help="Assume all frames share one (unknown) camera intrinsics matrix: jointly refit a single whole-video consensus K (median of fx/fy/cx/cy across every submap) and refit every frame's rotation against it. Requires --model_backend da3 with --use_ray_pose. Mutually exclusive with --fixed_intrinsics")
parser.add_argument("--gps_csv", type=str, default=None, help="Path to a natix-style FRONT_*.csv sensor log (alongside the source video) to fuse with the trajectory via GPS (see vggt_slam/gps_fusion.py). Requires --log_results")
parser.add_argument("--extraction_fps", type=float, default=10.0, help="fps used when extracting --image_folder's frames from the source video. Only used to align keyframes to GPS fixes by elapsed time when --gps_csv is set")
parser.add_argument("--gps_max_dt", type=float, default=0.5, help="Max seconds between a keyframe and a GPS fix to consider them matched. Only used when --gps_csv is set")
parser.add_argument("--osm_region_pbf_path", type=str, default=None, help="Path to a pre-downloaded whole-region OSM extract (a Geofabrik country/state .osm.pbf -- see scripts/download_osm_extracts.py) to crop locally for GPS fusion's road network. Required when --gps_csv is set.")
parser.add_argument("--osm_fallback_region_pbf_path", type=str, default=None, help="Tried once instead of --osm_region_pbf_path if the GPS fixes' own crop finds zero roads there -- some clips are filed under the wrong country folder in this dataset.")
parser.add_argument("--undistort_maps_path", type=str, default=None, help="Path to a .npz (vggt_slam.undistort.save_undistort_maps) with the remap tables for this clip's known distortion. When set, --image_folder is assumed to hold the RAW (distorted) frames, and undistortion is applied in-memory to only the frames selected as keyframes -- not written to disk, not applied to frames that get filtered out. --fixed_intrinsics should already describe the post-undistortion (rectified) geometry (batch_infer.py's undistort_maps_path/fixed_intrinsics pair are computed together).")


def build_backend(args, device):
    """Load the feed-forward geometry model (+ optional open-set semantic models) once.
    Split out of main() so a batch driver can load weights a single time per GPU process
    and reuse them across many videos instead of paying model-load cost per video."""
    print(f"Initializing and loading {args.model_backend} model...")

    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        processor = None
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    if args.model_backend == "vggt":
        from vggt.models.vggt import VGGT

        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

        model.eval()
        model = model.to(torch.bfloat16)  # use half precision
        model = model.to(device)
        backend = VGGTBackend(model)
    else:
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(args.da3_checkpoint)
        model.eval()
        model = model.to(device=device)
        backend = DA3Backend(model, use_ray_pose=args.use_ray_pose, ref_view_strategy=args.ref_view_strategy)

    return backend, clip_model, clip_preprocess, clip_tokenizer, processor


def run_video(args, backend, device, clip_model=None, clip_preprocess=None, clip_tokenizer=None, processor=None):
    """Run the VGGT-SLAM pipeline over a single video's already-extracted frames
    (args.image_folder), using an already-loaded backend/model. Everything that's
    video-specific (image folder, log path, GPS csv, ...) is read off `args`, so a batch
    driver can call this repeatedly with a fresh Solver per video while reusing one loaded
    model per GPU process."""
    use_optical_flow_downsample = True

    if args.fixed_intrinsics is not None and args.shared_intrinsics:
        raise ValueError("--fixed_intrinsics and --shared_intrinsics are mutually exclusive")

    # --shared_intrinsics (K unknown but constant across the video) only has an implementation
    # path via DA3's ray-pose two-phase joint calibration below (see
    # vggt_slam/da3_ray_pose_patch.py's module docstring) -- the whole video is available upfront
    # (this is a batch driver, not a live stream), so every submap's estimate is pooled into one
    # whole-video median instead of an online running one. There is no other backend for it.
    if args.shared_intrinsics and not (args.model_backend == "da3" and args.use_ray_pose):
        raise ValueError(
            "--shared_intrinsics requires --model_backend da3 with --use_ray_pose "
            "(the only path with a joint two-phase K+R refit; see vggt_slam/da3_ray_pose_patch.py)"
        )
    two_phase = args.shared_intrinsics

    fixed_K, fixed_K_size = parse_fixed_intrinsics(args.fixed_intrinsics)
    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
        overlapping_window_size=args.overlapping_window_size,
        fixed_intrinsics=fixed_K,
        fixed_intrinsics_size=fixed_K_size,
        headless=args.headless,
    )

    if two_phase:
        from vggt_slam import da3_ray_pose_patch
        da3_ray_pose_patch.set_capture(True)
        # solver.map itself must stay empty until phase 3 (add_edge's "find the previous
        # submap" logic depends on it holding ONLY the submaps already committed, in order), but
        # retrieval-based loop-closure search still needs to see every submap processed so far
        # during phase 1 -- so give it a separate map to search that phase 1 populates as it goes.
        solver.retrieval_map = GraphMap()

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    # VGGTBackend.infer_submap reads straight from image_names by path, ignoring frame_cache
    # entirely (see model_backends.py) -- it has no way to see the in-memory undistorted array,
    # so it would silently process the still-distorted frame on disk while --fixed_intrinsics
    # claims rectified geometry. Only DA3Backend consumes frame_cache.
    if args.undistort_maps_path is not None and args.model_backend != "da3":
        raise ValueError(
            "--undistort_maps_path requires --model_backend da3 (VGGTBackend doesn't use "
            "frame_cache, so it would silently read the still-distorted frame from disk)"
        )

    # When set, --image_folder holds RAW (distorted) frames -- undistort_frame is applied
    # in-memory below, only to frames that survive keyframe selection, instead of writing a
    # second (undistorted) copy of every frame to disk and reading it back (see
    # vggt_slam/undistort.py's docstring for why this is worth doing).
    undistort_map1 = undistort_map2 = undistort_roi = None
    if args.undistort_maps_path is not None:
        undistort_map1, undistort_map2, undistort_roi = load_undistort_maps(args.undistort_maps_path)
        # So infer_loop_closure_pair's own path-based re-read (see DA3Backend._run) rectifies too.
        backend.set_undistort_maps(undistort_map1, undistort_map2, undistort_roi)

    total_time_start = time.time()
    backend_time = utils.Accumulator()
    staged_predictions = []  # two_phase only: submaps run through DA3 but not yet added to the map

    # Reading+filtering upcoming frames (CPU: cv2.imread + FrameTracker's optical-flow disparity
    # check) and running the current submap (GPU: solver.run_predictions -> DA3 forward pass) were
    # previously one strictly sequential loop -- confirmed via live GPU-utilization sampling (1s
    # resolution) that this leaves the GPU idle for multi-second stretches while each frame's own
    # CPU-side read+check runs, and that all 4 GPU workers' idle stretches aren't staggered, so
    # they frequently land on 0% utilization simultaneously. Neither CPU nor GPU was ever close to
    # saturated (peak node load ~103/288 cores, GPU util averaging ~20%) -- the bottleneck is this
    # serialization, not raw capacity on either side.
    #
    # Fix: run the read+filter loop on a background thread, one submap-batch ahead, handed to the
    # main thread over a small bounded queue; the main thread's own loop below is otherwise
    # unchanged (same solver calls, same ordering, same two_phase deferral). This overlaps cleanly
    # despite Python's GIL because both sides spend most of their time inside code that releases
    # it: cv2's core (imread/goodFeaturesToTrack/calcOpticalFlowPyrLK) is GIL-releasing C++, and
    # PyTorch releases the GIL while a CUDA call is in flight -- so while the main thread awaits
    # the current submap's forward pass on the GPU, the producer thread is free to actually run
    # the next batch's CPU-bound read+filter work, not just be scheduled and blocked behind it.
    # queue.Queue(maxsize=2) caps how far the producer can run ahead (~2 submaps' worth of decoded
    # frame_cache arrays, a few tens of MB) -- enough to always have the next batch ready without
    # unbounded memory growth for a long clip.
    def _produce_submap_batches(out_queue):
        image_names_subset = []
        image_count = 0
        keyframe_time = utils.Accumulator()
        count = 0
        # image_name (str) -> already-decoded RGB array, for frames the disparity check just read
        # via cv2.imread below. DA3's own .inference() otherwise re-reads+re-decodes every
        # included frame from disk a second time (confirmed: its `image` param natively accepts
        # np.ndarray, not just paths -- see depth_anything_3's InputProcessor._load_image) --
        # passing the array we already have skips that redundant read+decode entirely for every
        # frame that makes it into a submap. cv2.imread is BGR; DA3's array path assumes RGB
        # ("# Assume HxWxC uint8/RGB" in its source), so the cached copy is converted once here --
        # the disparity check itself still runs on the original BGR array unchanged
        # (cv2.cvtColor(..., COLOR_BGR2GRAY) doesn't care about the R/B swap for keyframe-selection
        # purposes, but DA3's model absolutely does).
        frame_cache = {}
        try:
            for image_name in tqdm(image_names):
                if use_optical_flow_downsample:
                    with keyframe_time:
                        img = cv2.imread(image_name)
                        # Disparity is checked on the (possibly still-distorted) frame as read --
                        # this is a motion/coverage heuristic, not a geometric measurement, so the
                        # lens distortion (mild, uncorrected here on purpose) doesn't affect the
                        # keyframe decision enough to matter.
                        enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                        if enough_disparity:
                            image_names_subset.append(image_name)
                            if undistort_map1 is not None:
                                img = undistort_frame(img, undistort_map1, undistort_map2, undistort_roi)
                            frame_cache[image_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            image_count += 1
                else:
                    image_names_subset.append(image_name)
                    if undistort_map1 is not None:
                        img = undistort_frame(cv2.imread(image_name), undistort_map1, undistort_map2, undistort_roi)
                        frame_cache[image_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
                    count += 1
                    batch_cache = {k: frame_cache[k] for k in image_names_subset if k in frame_cache}
                    out_queue.put(("batch", list(image_names_subset), batch_cache, count))
                    # Reset for next submap -- same tail-keeping logic as before, just applied
                    # here instead of after the (now-remote) solver.run_predictions call.
                    image_names_subset = image_names_subset[-args.overlapping_window_size:]
                    frame_cache = {k: frame_cache[k] for k in image_names_subset if k in frame_cache}
            out_queue.put(("done", image_count, keyframe_time))
        except Exception:
            out_queue.put(("error", traceback.format_exc()))

    batch_queue = queue.Queue(maxsize=2)
    producer_thread = threading.Thread(target=_produce_submap_batches, args=(batch_queue,), daemon=True)
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
        predictions = solver.run_predictions(image_names_subset, backend, args.max_loops, clip_model, clip_preprocess,
                                              frame_cache=frame_cache)
        print("Solver total time", time.time()-t1)
        print(count, "submaps processed")

        if two_phase:
            # Defer add_points/graph.optimize() (and therefore solver.map itself) until every
            # submap's forward pass is done and joint_refit has replaced this K (DA3's own
            # free per-frame estimate for now) and R with the whole-video consensus fit.
            # Register the submap into solver.retrieval_map right away regardless, so
            # run_predictions's own find_loop_closures call (for the NEXT submap, before this
            # block executes again) can still search against every submap seen so far,
            # exactly as it would against solver.map outside two-phase mode.
            staged_predictions.append(predictions)
            solver.retrieval_map.add_submap(predictions["_submap"])
        else:
            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

    producer_thread.join()

    if two_phase:
        da3_ray_pose_patch.set_capture(False)
        print(f"Two-phase shared intrinsics: jointly refitting K, R across {len(staged_predictions)} submaps "
              f"({sum(p['intrinsic'].shape[0] for p in staged_predictions)} frames)...")
        t1 = time.time()
        try:
            K_final = da3_ray_pose_patch.joint_refit(staged_predictions)
        finally:
            # joint_refit clears its own capture buffer on success (a no-op here then), but on
            # failure (e.g. a submap/loop-closure capture-count mismatch) nothing else would --
            # and since GPU workers are long-lived, reusing the same process across many clips,
            # a left-dirty buffer here silently poisons every later --shared_intrinsics clip this
            # worker picks up, cascading into repeated failures/slowdowns for the rest of the run.
            da3_ray_pose_patch.clear_captured()
        print(f"Joint refit done in {time.time() - t1:.2f}s. Consensus K:\n{K_final}")

        for predictions in staged_predictions:
            solver.add_points(predictions)
            with backend_time:
                solver.graph.optimize()

        if args.vis_map:
            # Nothing was incrementally visualized during the (deferred) first pass -- show
            # whatever the final map looks like now that every submap has been added.
            solver.update_all_submap_vis()

    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for backend model calls: {solver.backend_timer.total_time:.4f}s")
    print("Average backend model time per frame:", solver.backend_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())


    if args.run_os:
        # Register the viser object-query panel so the user can search for
        # objects in the viewer in addition to the terminal prompt below.
        data_lock = threading.Lock()
        solver.viewer.add_object_query_gui(solver, clip_model, clip_tokenizer, processor, data_lock)

        while True:
            # Prompt user for text input
            query = input("\nEnter text query or q to quit: ").strip()
            if len(query) == 0:
                print("Empty query. Exiting.")
                return
            
            if query == "q":
                print("Exiting.")
                return
            
            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            overall_best_score, overall_best_submap_id, overall_best_frame_index = solver.map.retrieve_best_semantic_frame(text_emb)

            found_submap = solver.map.get_submap(overall_best_submap_id)

            # Display image
            best_img = found_submap.get_frame_at_index(overall_best_frame_index)
            print("Score:", overall_best_score)
            with torch.no_grad():
                # convert torch image to PIL
                best_img = to_pil_image(best_img)
                inference_state = processor.set_image(best_img)
                output = processor.set_text_prompt(state=inference_state, prompt=query)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                print("Scores:", scores.cpu().numpy())


            masked_img = utils.overlay_masks(best_img, masks)
            masked_img.show()

            for i in range(masks.shape[0]):
                mask = masks[i].cpu().numpy()
                obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                solver.viewer.visualize_obb(
                    center=obb_center,
                    extent=obb_extent,
                    rotation=obb_rotation,
                    color=(255, 0, 0),
                    line_width=8.0,
                )

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    gps_fusion_info = None  # stays None when --gps_csv isn't set at all (no GPS data for this clip)
    if args.log_results:
        frame_ids, rotations, translations = solver.map.get_all_poses_and_frame_ids(solver.graph)
        # Written now with only the SLAM fields, and again below (same path) once GPS fusion
        # finishes -- so the SLAM result survives on disk even if fusion crashes the whole
        # process outright (e.g. a SIGKILL from cluster instability, confirmed to happen in
        # practice), not just a caught Python exception. See save_result_npz's own docstring.
        utils.save_result_npz(args.log_path, frame_ids, rotations, translations,
                               args.extraction_fps, args.video)

        if not args.skip_dense_log:
            # Log the full point cloud as one file
            solver.map.write_points_to_file(solver.graph, os.path.splitext(args.log_path)[0] + "_points.pcd")

        if args.gps_csv is not None:
            # GPS fusion is enrichment on top of an already-saved, valid SLAM trajectory (written
            # just above) -- a failure here (bad/missing GPS data, no region extract for this
            # clip's country, no map-match found, ...) must not throw away that trajectory or make
            # the whole clip look failed. Caught and logged instead of propagated. gps_fusion_info
            # is returned to the caller (batch_infer.py's gpu_worker) and lands in manifest.jsonl,
            # so a clip's actual GPS-fusion outcome -- and whether/which region fallback was
            # needed -- is queryable at dataset scale without grepping every clip's log.
            network_dir = os.path.join(os.path.dirname(args.log_path) or ".", "osm_network")
            try:
                if not args.osm_region_pbf_path:
                    raise ValueError("--gps_csv requires --osm_region_pbf_path (no region extract given)")
                gps = gps_module.load_gps_fixes(args.gps_csv)
                (fused_rot, fused_trans, scale, matches, matched_line, snapped, region_used,
                 network_profile) = align_trajectory_to_roads(
                    rotations, translations, frame_ids, args.extraction_fps, gps, network_dir,
                    args.osm_region_pbf_path, max_dt=args.gps_max_dt,
                    fallback_region_pbf_path=args.osm_fallback_region_pbf_path,
                )
                used_fallback = region_used is not None and region_used != args.osm_region_pbf_path
                gps_fusion_info = {
                    "region_used": region_used, "used_fallback": used_fallback,
                    "network_profile_used": network_profile,
                    "matched_keyframes": len(matches), "total_keyframes": len(frame_ids),
                }
                print(f"GPS fusion: matched {len(matches)}/{len(frame_ids)} keyframes, "
                      f"initial visual-gauge -> meters scale = {scale:.4f}, "
                      f"region_used={region_used}, used_fallback={used_fallback}, "
                      f"network_profile={network_profile}")

                if matched_line is not None:
                    gps_fusion_info["status"] = "ok"

                    # Single-number reliability signal for this clip's fusion -- mean/max
                    # distance between the final predicted trajectory and its own snap onto the
                    # map-matched route, at every point map-matching-2 actually snapped (NaN
                    # rows excluded). Logged into manifest.jsonl so a bad fit (wrong-junction
                    # match, a fit anchored to too few/poorly-conditioned GPS correspondences,
                    # ...) is queryable/sortable across the whole dataset without re-opening
                    # every clip's own artifacts -- see scripts/plot_road_alignment*.py, which
                    # compute + print the same number for one clip at a time.
                    traj_en = fused_trans[:, [0, 2]]
                    snapped_ok = ~np.isnan(snapped[:, 0])
                    if snapped_ok.any():
                        dev = np.linalg.norm(traj_en[snapped_ok] - snapped[snapped_ok], axis=1)
                        gps_fusion_info["map_match_error_mean_m"] = float(dev.mean())
                        gps_fusion_info["map_match_error_max_m"] = float(dev.max())
                        gps_fusion_info["map_match_error_n_points"] = int(snapped_ok.sum())
                        err_str = f", mean map-match error {dev.mean():.2f}m (max {dev.max():.2f}m)"
                    else:
                        err_str = ""
                    print(f"Final map-matched route + per-point snaps written into {args.log_path}"
                          f"{err_str}")
                else:
                    gps_fusion_info["status"] = "no_route_match"
                    print(f"[WARN] GPS fusion: map-matching-2 found no route match for the final "
                          f"trajectory (region_used={region_used} did have nearby roads) -- no "
                          f"route/snapped written for {args.video}.", file=sys.stderr, flush=True)

                # Rewrites the SAME result.npz from above, slam_* fields unchanged, now with
                # whatever GPS-fusion fields this outcome produced (fused_rot/fused_trans are
                # always valid here -- only matched_line/snapped are conditionally None).
                utils.save_result_npz(args.log_path, frame_ids, rotations, translations,
                                       args.extraction_fps, args.video,
                                       fused_rotations=fused_rot, fused_translations=fused_trans,
                                       matched_line=matched_line, snapped=snapped,
                                       gps_fusion_info=gps_fusion_info)
            except ValueError as e:
                # align_trajectory_to_roads raises these specific, expected precondition
                # failures rather than ever returning normally with region_used=None (an earlier
                # version of the code above had a now-removed elif branch for that case, which
                # was actually unreachable) -- give each its own structured status, same as
                # scripts/refuse_gps.py's matching except-ValueError branch, instead of burying
                # them in the generic "failed" bucket below where only a traceback grep
                # distinguishes them. Confirmed this was previously happening at real scale:
                # ~72% of all "failed" gps_fusion records in one manifest.jsonl audit were
                # actually one of the first two messages below.
                if "no roads found near the GPS fixes" in str(e):
                    gps_fusion_info = {"status": "no_roads_in_any_region", "error": str(e)}
                elif "raw-GPS map-match unavailable" in str(e):
                    # align_trajectory_to_roads's own message already distinguishes WHY via
                    # "matched_line=found"/"matched_line=none" -- split into two statuses instead
                    # of lumping both under one, since they're different root causes: "found" means
                    # a route WAS matched but too few keyframes fall in that time window (a data-
                    # sparsity issue, same family as no_real_gps_fixes/insufficient_gps_matches --
                    # confirmed on real clips: typically only 2-16 real GPS fixes), while "none"
                    # means no plausible route through the road network was found at all for the
                    # raw GPS trace (a mix of also-sparse clips and a smaller tail of clips with
                    # PLENTY of good GPS data that still can't be routed -- confirmed on one real
                    # clip: 53/53 real fixes, realistic speed, 13m accuracy, still no match -- a
                    # genuine map-matching/road-network limitation worth its own bucket to find).
                    if "matched_line=found" in str(e):
                        gps_fusion_info = {"status": "no_raw_gps_route_too_sparse", "error": str(e)}
                    else:
                        gps_fusion_info = {"status": "no_raw_gps_route_unmatched", "error": str(e)}
                elif "implausible GPS-derived scale" in str(e):
                    gps_fusion_info = {"status": "implausible_gps_scale", "error": str(e)}
                elif "No real GPS fixes" in str(e):
                    # vggt_slam.gps.load_gps_fixes: every row in this clip's sensor log is 'na' --
                    # a genuine data-quality issue with that clip (nothing to fix in code), not a
                    # bug. Confirmed at dataset scale: this alone was 70.9% of all "failed"
                    # gps_fusion records in one manifest.jsonl audit.
                    gps_fusion_info = {"status": "no_real_gps_fixes", "error": str(e)}
                elif "keyframe(s) matched to a GPS fix" in str(e):
                    # align_trajectory_to_roads's own len(matches) < 2 precondition: real GPS
                    # fixes exist, but fewer than 2 landed within max_dt of an actual keyframe --
                    # too sparse to fit a scale at all, distinct from "no_raw_gps_route_*" (which
                    # has >=2 matches but no matchable route) or "implausible_gps_scale" (which
                    # has >=2 matches but a degenerate/wrong fit). 20.2% of the same audit above.
                    gps_fusion_info = {"status": "insufficient_gps_matches", "error": str(e)}
                elif "distinct timestamp(s)" in str(e):
                    # vggt_slam.gps.load_gps_fixes: fewer than 2 distinct wall-clock timestamps
                    # in the sensor log -- can't fit a native-fps line at all. Confirmed on 268
                    # real clips, every single one with exactly 1 sensor-log row for the whole
                    # clip (a natix logging gap, not a code bug) -- previously surfaced as an
                    # opaque numpy.linalg.LinAlgError ("SVD did not converge") before this
                    # precondition check existed, all lumped into the generic "failed" bucket.
                    gps_fusion_info = {"status": "gps_log_too_short", "error": str(e)}
                else:
                    gps_fusion_info = {"status": "failed", "error": traceback.format_exc()}
                utils.save_result_npz(args.log_path, frame_ids, rotations, translations,
                                       args.extraction_fps, args.video,
                                       gps_fusion_info=gps_fusion_info)
                print(f"[WARN] GPS fusion failed for {args.video} (status="
                      f"{gps_fusion_info['status']}); SLAM trajectory at {args.log_path} is "
                      f"still valid and was already saved:\n{gps_fusion_info['error']}",
                      file=sys.stderr, flush=True)
            except Exception:
                gps_fusion_info = {"status": "failed", "error": traceback.format_exc()}
                utils.save_result_npz(args.log_path, frame_ids, rotations, translations,
                                       args.extraction_fps, args.video,
                                       gps_fusion_info=gps_fusion_info)
                print(f"[WARN] GPS fusion failed for {args.video}; SLAM trajectory at "
                      f"{args.log_path} is still valid and was already saved:\n"
                      f"{gps_fusion_info['error']}", file=sys.stderr, flush=True)
            finally:
                # Trims osm_network/ down to just network.osm regardless of outcome -- see
                # trim_osm_network_cache's own docstring for why the rest (prepared*/ graphs,
                # tracks.csv, results/) has no lasting value once fusion is done.
                trim_osm_network_cache(network_dir)
    return gps_fusion_info


def main():
    """CLI entrypoint: load the model once and run it on a single video's frames."""
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    backend, clip_model, clip_preprocess, clip_tokenizer, processor = build_backend(args, device)
    run_video(args, backend, device, clip_model, clip_preprocess, clip_tokenizer, processor)


if __name__ == "__main__":
    main()
