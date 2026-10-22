# Backend and model choices

- **Default `--model_backend da3`:** DA3-GIANT-1.1 outperformed VGGT on GPS-fit trajectory error on
  all three natix clips tested (comparison figures `gps_fit_*.png` were not kept).
- **Default `--da3_checkpoint depth-anything/DA3-GIANT-1.1`** (plain any-view, no metric claim):
  DA3NESTED-GIANT-LARGE-1.1's "metric" scale was empirically unreliable (1.3x-4x off) and
  DA3-GIANT-1.1's trajectory shape was more accurate anyway once fairly scale-fit. Metric scale comes
  from GPS fusion or from DA3METRIC-LARGE depth (stage 3).
- **`--use_ray_pose` default on:** prerequisite for `da3_ray_pose_patch.py`'s fixed/shared
  intrinsics support (rotation fit against the known/consensus K instead of DA3's free per-frame estimate).
- **Loop closure with DA3:** DA3 has no trained match-quality head; the acceptance test checks
  whether the pair's own 2-view reconstruction is geometrically self-consistent. Not calibrated
  against the VGGT fork's learned ratio, `--lc_thres` may need separate tuning.
- **Why per-clip intrinsics are locked:** each backend estimates K independently per submap and
  loop-closure pair, so ordinary model noise looks like K drift and leaks into point-cloud
  unprojection, inter-submap scale estimation and GPS fusion. FIXED_METADATA.json's own
  `frame_width/height` was observed to be exactly half the true encoded resolution for at least one
  clip, so it is cross-checked against the video's resolution and never trusted alone.
