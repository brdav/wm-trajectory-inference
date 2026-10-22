"""GPS/road fusion of one clip's SLAM trajectory, reported as a structured outcome.

Wraps gps_fusion.align_trajectory_to_roads for the two callers that need the same outcome record
(vggt_slam/pipeline.py for a fresh clip, scripts/tools/refuse_gps.py for a rerun): the expected
failure modes each get their own status instead of a generic "failed" (why:
memory/gps-fusion-notes.md), and the record is what lands in manifest.jsonl and result.npz.
"""
import sys
import traceback

import numpy as np

from vggt_slam import gps as gps_module
from vggt_slam.gps_fusion import align_trajectory_to_roads
from vggt_slam.map_match import trim_osm_network_cache

def _raw_gps_route_status(msg):
    # "matched_line=found": a route matched but too few keyframes fall in its time window;
    # "none": no plausible route through the road network
    return "no_raw_gps_route_too_sparse" if "matched_line=found" in msg else "no_raw_gps_route_unmatched"


# (substring of the ValueError align_trajectory_to_roads / load_gps_fixes raises, status or a function of the
# message), first match wins
_ERROR_STATUSES = (
    ("no roads found near the GPS fixes", "no_roads_in_any_region"),
    ("raw-GPS map-match unavailable", _raw_gps_route_status),
    ("implausible GPS-derived scale", "implausible_gps_scale"),
    ("No real GPS fixes", "no_real_gps_fixes"),              # every row of the sensor log is 'na'
    ("keyframe(s) matched to a GPS fix", "insufficient_gps_matches"),  # < 2 keyframes within max_dt of a fix
    ("distinct timestamp(s)", "gps_log_too_short"),           # < 2 distinct timestamps in the sensor log
)


def classify_error(e, traceback_chars=None):
    """gps_fusion_info for an exception raised during fusion."""
    msg = str(e)
    if isinstance(e, ValueError):
        for needle, status in _ERROR_STATUSES:
            if needle in msg:
                return {"status": status(msg) if callable(status) else status, "error": msg}
    tb = traceback.format_exc()
    return {"status": "failed", "error": tb[-traceback_chars:] if traceback_chars else tb}


def run_gps_fusion(rotations, translations, frame_ids, extraction_fps, gps_csv, network_dir,
                   region_pbf_path, fallback_region_pbf_paths=None, max_dt=0.5, traceback_chars=None, label=""):
    """Fuse a SLAM trajectory with the clip's GPS log and the road network of its OSM region extract.

    Never raises. Returns (gps_fusion_info, fused): fused is None unless fusion placed the
    trajectory, else a dict with rotations, translations, matched_line (None if no route matched)
    and snapped. `label` only names the clip in log messages."""
    try:
        if not region_pbf_path:
            raise ValueError("GPS fusion needs an OSM region extract (none found for this clip's region)")
        gps = gps_module.load_gps_fixes(str(gps_csv))
        (fused_rot, fused_trans, scale, matches, matched_line, snapped, region_used,
         network_profile) = align_trajectory_to_roads(
            rotations, translations, frame_ids, extraction_fps, gps, str(network_dir), str(region_pbf_path),
            max_dt=max_dt, fallback_region_pbf_path=[str(p) for p in fallback_region_pbf_paths or []] or None)
        used_fallback = region_used is not None and region_used != str(region_pbf_path)
        info = {"region_used": region_used, "used_fallback": used_fallback,
                "network_profile_used": network_profile,
                "matched_keyframes": len(matches), "total_keyframes": len(frame_ids)}
        print(f"GPS fusion: matched {len(matches)}/{len(frame_ids)} keyframes, "
              f"initial visual-gauge -> meters scale = {scale:.4f}, "
              f"region_used={region_used}, used_fallback={used_fallback}, network_profile={network_profile}")
        if matched_line is not None:
            info["status"] = "ok"
            # reliability signal: distance between the final trajectory and its snap onto the matched route
            snapped_ok = ~np.isnan(snapped[:, 0])
            if snapped_ok.any():
                dev = np.linalg.norm(fused_trans[:, [0, 2]][snapped_ok] - snapped[snapped_ok], axis=1)
                info["map_match_error_mean_m"] = float(dev.mean())
                info["map_match_error_max_m"] = float(dev.max())
                info["map_match_error_n_points"] = int(snapped_ok.sum())
                print(f"Final map-matched route + per-point snaps: mean map-match error {dev.mean():.2f}m "
                      f"(max {dev.max():.2f}m)")
        elif region_used is None:
            info["status"] = "no_roads_in_any_region"
        else:
            info["status"] = "no_route_match"
            print(f"[WARN] GPS fusion: map-matching-2 found no route match for the final trajectory "
                  f"(region_used={region_used} did have nearby roads) -- no route/snapped written for {label}.",
                  file=sys.stderr, flush=True)
        return info, dict(rotations=fused_rot, translations=fused_trans, matched_line=matched_line, snapped=snapped)
    except Exception as e:
        info = classify_error(e, traceback_chars)
        print(f"[WARN] GPS fusion failed for {label} (status={info['status']}); the SLAM trajectory "
              f"is still valid:\n{info['error']}", file=sys.stderr, flush=True)
        return info, None
    finally:
        # trims osm_network/ to just network.osm whatever the outcome (see trim_osm_network_cache)
        trim_osm_network_cache(network_dir)
