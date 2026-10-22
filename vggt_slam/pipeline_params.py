"""SLAM settings. Kept apart from pipeline.py because that module imports torch, and the batch driver
must set its thread-count environment variables before torch is first imported."""
from dataclasses import dataclass


@dataclass
class SlamParams:
    """SLAM settings shared by every clip of a run. Defaults are the production settings."""
    model_backend: str = "da3"                           # "vggt" or "da3" (rationale: memory/backend-choice.md)
    da3_checkpoint: str = "depth-anything/DA3-GIANT-1.1"  # local dir or Hugging Face repo id; da3 only
    use_ray_pose: bool = True                            # da3 only; required by fixed/shared intrinsics
    ref_view_strategy: str = "middle"                    # da3 only: how DA3 picks the reference view of a submap
    submap_size: int = 16                                # new frames per submap, excluding overlap/loop-closure frames
    overlapping_window_size: int = 2                     # frames shared between consecutive submaps
    max_loops: int = 1                                   # only 1 (loop closures on) or 0 (off) is supported
    min_disparity: float = 25.0                          # minimum mean optical-flow disparity to take a keyframe
    conf_threshold: float = 25.0                         # percentage of lowest-confidence points to filter out
    lc_thres: float = 0.95                               # image-retrieval threshold for loop closures, [0, 1]
