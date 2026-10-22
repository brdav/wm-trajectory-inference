import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import time
import open3d as o3d
from termcolor import colored
from scipy.linalg import rq

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map

from vggt_slam.slam_utils import compute_image_embeddings, Accumulator
from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.graph import PoseGraph
from vggt_slam.scale_solver import estimate_scale_pairwise
from vggt_slam.viewer import Viewer
from vggt_slam.intrinsics_lock import IntrinsicsLock
from vggt_slam import da3_ray_pose_patch

DEBUG = False

def debug_visualize(pcd1_points, pcd2_points):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        lc_thres: float = 0.80,
        vis_voxel_size: float = None,
        vis_imgs: bool = False,
        overlapping_window_size: int = 1,
        fixed_intrinsics: np.ndarray = None,
        fixed_intrinsics_size: tuple = None,
        headless: bool = False):

        self.init_conf_threshold = init_conf_threshold
        self.vis_voxel_size = vis_voxel_size
        self.vis_imgs = vis_imgs
        self.overlapping_window_size = overlapping_window_size
        self.intrinsics_lock = IntrinsicsLock(fixed_intrinsics, fixed_intrinsics_size)

        # A real viser.ViserServer (network server + full-resolution point-cloud serialization
        # per submap) is expensive and pointless when nothing ever views it, e.g. one persistent
        # server per clip across a many-clip batch run -- skip creating it entirely.
        self.viewer = None if headless else Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.graph = PoseGraph()
        # Where find_loop_closures searches -- self.map by default (matching every submap ever
        # actually committed, exactly as before). main.py's two-phase --shared_intrinsics path
        # points this at a separate, phase-1-only GraphMap instead: self.map must stay empty
        # until phase 3 (add_edge's "find the previous submap" logic depends on self.map holding
        # ONLY the submaps already committed, in order), but retrieval-based loop-closure search
        # still needs to see every submap processed so far during phase 1.
        self.retrieval_map = self.map

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None
        # Tracked independently of self.map's contents (rather than derived from
        # self.map.get_largest_key()) so submap IDs come out correctly even when add_points is
        # deferred well past run_predictions -- as main.py's two-phase --shared_intrinsics path
        # does, running every submap's forward pass before adding any of them to the map.
        self._next_submap_id = 0

        self.lc_thres = lc_thres

        self.temp_count = 0
        self.backend_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.viewer is None:
            return
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        self.viewer.server.scene.add_point_cloud(
            name="pcd_"+name,
            points=points_in_world_frame,
            colors=points_colors,
            point_size=point_size,
            point_shape="circle",
        )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        if self.viewer is None:
            return
        points_in_world_frame = submap.get_points_in_world_frame(self.graph)
        points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        if self.viewer is None:
            return
        extrinsics = submap.get_all_poses_world(self.graph)
        images = submap.get_all_frames() if self.vis_imgs else None
        self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def tranform_submap_to_canonical(self, proj_mat_world_to_cam, world_points):
        P_first_cam = proj_mat_world_to_cam[0].copy()

        # Apply transformation to camera matrices such that the first camera matrix of the submap is [I | 0]
        proj_mat_world_to_cam = proj_mat_world_to_cam @ np.linalg.inv(P_first_cam)

        # Apply transformation to points such that the first camera matrix of the submap is [I | 0]
        h, w = world_points.shape[1:3]
        for i in range(len(proj_mat_world_to_cam)):
            points_in_cam = world_points[i,...]
            points_in_cam_h = np.hstack([points_in_cam.reshape(-1, 3), np.ones((points_in_cam.shape[0] * points_in_cam.shape[1], 1))])
            points_in_cam_h = (P_first_cam @ points_in_cam_h.T).T
            points_in_cam = points_in_cam_h[:, :3] / points_in_cam_h[:, 3:]
            world_points[i] = points_in_cam.reshape(h, w, 3)
        
        return proj_mat_world_to_cam, world_points

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False, overlap=1):
        """
        overlap: number of consecutive frames physically shared between the previous and current
        submaps (frame_id_prev - overlap + 1 + k <-> frame_id_curr + k for k in range(overlap)).
        Always 1 for loop-closure edges, regardless of --overlapping_window_size.
        """
        assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
        scale_factor = 1.0
        current_submap = self.map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)
        if submap_id_prev is not None:
            # frame_id_prev/frame_id_curr denote the LAST shared pair; the anchor pair used to
            # seed the new submap's global pose is the FIRST shared pair (k=0).
            anchor_frame_prev = frame_id_prev - overlap + 1
            overlapping_node_id_prev = submap_id_prev + anchor_frame_prev

            # Estimate scale factor between submaps, aggregating over every shared frame so the
            # estimate isn't just a single frame's noise.
            prior_submap = self.map.get_submap(submap_id_prev)

            t1_parts, t2_parts = [], []
            for k in range(overlap):
                curr_idx = frame_id_curr + k
                prev_idx = anchor_frame_prev + k

                current_conf = current_submap.get_conf_masks_frame(curr_idx)
                prior_conf = prior_submap.get_conf_masks_frame(prev_idx)
                good_mask = (prior_conf > prior_submap.get_conf_threshold()) * (current_conf > prior_submap.get_conf_threshold())
                good_mask = good_mask.reshape(-1)

                if np.sum(good_mask) < 100:
                    print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
                    good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
                    if np.sum(good_mask) < 100: # Handle the case where loop closure frames do not have enough points.
                        good_mask = (prior_conf > 0).reshape(-1)

                P_temp = np.linalg.inv(prior_submap.proj_mats[prev_idx]) @ current_submap.proj_mats[curr_idx]
                t1_parts.append((P_temp[0:3,0:3] @ current_submap.get_frame_pointcloud(curr_idx).reshape(-1, 3)[good_mask].T).T)
                t2_parts.append(prior_submap.get_frame_pointcloud(prev_idx).reshape(-1, 3)[good_mask])

            t1 = np.concatenate(t1_parts, axis=0)
            t2 = np.concatenate(t2_parts, axis=0)
            scale_factor_est_output = estimate_scale_pairwise(t1, t2)
            print(colored("scale factor", 'green'), scale_factor_est_output)
            scale_factor = scale_factor_est_output[0]
            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            if DEBUG:
                print("Estimated scale factor between submaps:", scale_factor)
                debug_visualize(scale_factor*t1, t2)

            # Compute the first camera matrix of the new submap in world frame.
            H_overlap_prior_overlap_current = np.linalg.inv(prior_submap.proj_mats[anchor_frame_prev]) @ current_submap.proj_mats[frame_id_curr] @ H_scale
            H_w_submap = self.graph.get_homography(overlapping_node_id_prev) @ H_overlap_prior_overlap_current

            # Add first node of the new submap to the graph.
            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            # Add between factor for intra submaps constraint.
            self.graph.add_between_factor(overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_overlap_prior_overlap_current, self.graph.intra_submap_noise)

            if DEBUG:
                print("Adding first homography of submap: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])
                print("Adding between factor: \n", overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_scale)

        else:
            assert (submap_id_curr == 0 and frame_id_curr == 0), "First added node must be submap 0 frame 0"
            self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            self.graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)
            if DEBUG:
                print("Adding first homography of graph: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])

        # Loop closure only gets intra submap constraints.
        if is_loop_closure:
            return

        # Add nodes and edges for the inner submap constraints.
        world_to_cam = current_submap.get_all_poses()
        for index, pose in enumerate(world_to_cam):
            if index == 0:
                continue

            H_inner = world_to_cam[index-1] @ np.linalg.inv(pose)
            current_node = self.graph.get_homography(submap_id_curr + index - 1) @ H_inner

            # Add node to graph.
            self.graph.add_homography(submap_id_curr + index, current_node)

            # Add between factor for inner submap constraint.
            self.graph.add_between_factor(submap_id_curr + index - 1, submap_id_curr + index, H_inner, self.graph.inner_submap_noise)

            if DEBUG:
                print("Adding homography: \n", submap_id_curr + index, current_node / current_node[-1,-1])
                print("Adding between factor: \n", submap_id_curr + index - 1, submap_id_curr + index, H_inner)

        # With overlap > 1, frames 1..overlap-1 of the current submap are ALSO physically the
        # same frames as prior_submap's anchor_frame_prev+1..frame_id_prev -- each was just added
        # above as an ordinary inner-submap node, with no knowledge of its prior-submap twin. Tie
        # each such pair together with an extra between factor (same construction as the anchor
        # pair above), giving bundle adjustment more than one measurement of the drift between
        # the two submaps' independent solves instead of relying on the single anchor frame.
        if submap_id_prev is not None and overlap > 1:
            for k in range(1, overlap):
                curr_idx = frame_id_curr + k
                prev_idx = anchor_frame_prev + k
                H_overlap_k = np.linalg.inv(prior_submap.proj_mats[prev_idx]) @ current_submap.proj_mats[curr_idx] @ H_scale
                self.graph.add_between_factor(submap_id_prev + prev_idx, submap_id_curr + curr_idx, H_overlap_k, self.graph.intra_submap_noise)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        # Explicitly re-bind to the submap object this pred_dict was actually created for, rather
        # than trusting self.current_working_submap to still point at it: main.py's two-phase
        # --shared_intrinsics path runs every submap's run_predictions() call (each of which
        # reassigns self.current_working_submap) well before any of their add_points() calls, so
        # by the time this runs, self.current_working_submap would otherwise just be whichever
        # submap happened to be created last.
        self.current_working_submap = pred_dict["_submap"]
        t1 = time.time()
        images = pred_dict["images"]  # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        detected_loops = pred_dict["detected_loops"]

        depth_map = pred_dict["depth"]  # (S, H, W, 1)
        conf = pred_dict["depth_conf"]  # (S, H, W)

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        h, w = world_points.shape[1:3]
        
        # Create projection matrices
        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)


        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0
        frame_id_prev = None

        first_edge = submap_id_prev is None

        if not first_edge:
            frame_id_prev = self.map.get_latest_submap(ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()

        # Add attributes to submap and add submap to map.
        self.current_working_submap.add_all_poses(world_to_cam)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, K_4x4)
        self.current_working_submap.set_conf_masks(conf)
        self.map.add_submap(self.current_working_submap)

        # Add all constraints for the new submap.
        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure=False, overlap=self.overlapping_window_size)

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            cam_to_world_lc = closed_form_inverse_se3(pred_dict["extrinsic_lc"]) 
            K_4x4_lc = np.tile(np.eye(4), (2, 1, 1))
            K_4x4_lc[:, :3, :3] = pred_dict["intrinsic_lc"]
            world_to_cam_lc = np.linalg.inv(cam_to_world_lc)
            depth_map_lc = pred_dict["depth_lc"]  # (S, H, W, 1)
            conf_lc = pred_dict["depth_conf_lc"]  # (S, H, W)

            intrinsics_cam = pred_dict["intrinsic_lc"]
            

            world_points_lc = unproject_depth_map_to_point_map(depth_map_lc, pred_dict["extrinsic_lc"], intrinsics_cam)

            lc_submap_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1
            print(f"Creating new Loop closure submap with id {lc_submap_num}")
            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)
            lc_submap.add_all_frames(pred_dict["frames_lc"])
            lc_submap.set_frame_ids(pred_dict["frames_lc_names"])
            lc_submap.set_last_non_loop_frame_index(1)

            lc_submap.add_all_poses(world_to_cam_lc)
            lc_colors = (np.transpose(pred_dict["frames_lc"].cpu().numpy(), (0, 2, 3, 1)) * 255).astype(np.uint8)
            lc_submap.add_all_points(world_points_lc, lc_colors, conf_lc, self.init_conf_threshold, K_4x4_lc)
            print("Loop closure conf", conf_lc.shape)
            print(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame)
            lc_submap.set_conf_masks(conf_lc)
            self.map.add_submap(lc_submap)

            self.add_edge(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame, is_loop_closure=False)
            self.add_edge(loop.detected_submap_id, loop.detected_submap_frame, lc_submap_num, 1, is_loop_closure=True)

    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, backend, max_loops, clip_model, clip_preprocess, frame_cache=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        known_intrinsics = self.intrinsics_lock.get_backend_lock()
        t1 = time.time()
        with self.backend_timer:
            predictions = backend.infer_submap(image_names, device, known_intrinsics=known_intrinsics, frame_cache=frame_cache)
        # predictions["images"] keeps its leading dummy batch-of-1 dim (needed
        # further down so the generic numpy .squeeze(0) conversion loop applies
        # uniformly to every key); torch's squeeze is a safe no-op here even if
        # that dim isn't actually 1, unlike numpy's.
        images = predictions["images"].squeeze(0)
        print(f"Loaded, preprocessed, and ran backend inference on {len(image_names)} images in {time.time() - t1:.2f} seconds")
        print(f"Preprocessed images shape: {images.shape}")

        new_pcd_num = self._next_submap_id

        print(f"Creating new submap with id {new_pcd_num}")
        t1 = time.time()
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))
        new_submap.set_img_names(image_names)

        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_names)
                new_submap.set_all_semantic_vectors(image_embs)

        self.current_working_submap = new_submap
        predictions["_submap"] = new_submap
        print(f"Created new submap in {time.time() - t1:.2f} seconds")

        # Check for loop closures and add retrieval vectors from new submap to the database
        predictions_lc = None
        with self.loop_closure_timer:
            detected_loops = self.image_retrieval.find_loop_closures(self.retrieval_map, new_submap, max_loop_closures=max_loops, max_similarity_thres=self.lc_thres)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            retrieved_frames = self.retrieval_map.get_frames_from_loops(detected_loops)
            retrieved_submap = self.retrieval_map.get_submap(detected_loops[0].detected_submap_id)
            loop_closure_frame_names = [
                new_submap.get_img_names_at_index(detected_loops[0].query_submap_frame),
                retrieved_submap.get_img_names_at_index(detected_loops[0].detected_submap_frame),
            ]
            with torch.no_grad():
                predictions_lc = backend.infer_loop_closure_pair(
                    query_frame=new_submap.get_frame_at_index(detected_loops[0].query_submap_frame),
                    query_frame_path=loop_closure_frame_names[0],
                    retrieved_frame=retrieved_frames[0],
                    retrieved_frame_path=loop_closure_frame_names[1],
                    device=device,
                    known_intrinsics=known_intrinsics,
                )

            # Visualize loop closure frames
            if DEBUG:
                imgs = predictions_lc["frames_lc"].permute(0, 2, 3, 1).cpu().numpy()  # shape -> (2, H, W, C)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                for i in range(2):
                    axes[i].imshow(imgs[i])
                    axes[i].axis('off')
                plt.tight_layout()
                plt.title("Loop Closure Frames. Left: Query Frame, Right: Retrieved Frame")
                plt.show()

        predictions["detected_loops"] = detected_loops

        if predictions_lc is not None:
            match_score = predictions_lc["match_score"]
            if match_score < 0.95:
                print(colored("Loop closure match score too low, skipping loop closure", "red"))
                predictions_lc = None # We set to None to ignore the loop closure
                predictions["detected_loops"] = []
                # infer_loop_closure_pair's forward pass above already ran through
                # da3_ray_pose_patch's patched compute_optimal_rotation_intrinsics_batch, which
                # captures unconditionally (it has no way to know yet that this call will end up
                # rejected) -- undo that capture now that we know this LC pair isn't being kept,
                # or joint_refit's captured-vs-staged_predictions accounting desyncs by one. See
                # discard_last_capture's docstring.
                da3_ray_pose_patch.discard_last_capture()
            else:
                self.graph.increment_loop_closure()
                predictions["extrinsic_lc"] = predictions_lc["extrinsic_lc"]
                predictions["intrinsic_lc"] = predictions_lc["intrinsic_lc"]
                predictions["depth_lc"] = predictions_lc["depth_lc"]
                predictions["depth_conf_lc"] = predictions_lc["depth_conf_lc"]
                predictions["intrinsic_lc_locked_internally"] = predictions_lc["intrinsic_lc_locked_internally"]

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor) and key != "target_tokens":
                predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        if predictions_lc is not None:
            predictions["frames_lc"] = predictions_lc["frames_lc"][0:2, ...]
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names

        if self.intrinsics_lock.is_active:
            if not predictions.get("intrinsics_locked_internally", False):
                # The backend already fit R directly against the fixed K (see
                # da3_ray_pose_patch.py) when locked internally, so
                # predictions["intrinsic"]/["extrinsic"] are already a self-consistent pair --
                # no override needed. Otherwise, apply the fixed K post-hoc.
                predictions["intrinsic"] = self.intrinsics_lock.apply(
                    predictions["intrinsic"], actual_hw=predictions["depth"].shape[1:3])

            if "intrinsic_lc" in predictions and not predictions.get("intrinsic_lc_locked_internally", False):
                predictions["intrinsic_lc"] = self.intrinsics_lock.apply(
                    predictions["intrinsic_lc"], actual_hw=predictions["depth_lc"].shape[1:3])

        # Reserve this submap's id range (and, if a loop-closure submap will also be created by
        # add_points below -- always exactly 2 frames -- its range too) for whenever add_points
        # actually runs, which may be well after this call returns (see self._next_submap_id).
        self._next_submap_id = new_pcd_num + images.shape[0] + (2 if predictions_lc is not None else 0)

        return predictions