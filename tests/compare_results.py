"""Regression check for refactors that touch the SLAM / GPS-fusion path (needs a GPU run, so it is not a pytest).

Process the same clip(s) before and after a change into two output folders, then:

    python tests/compare_results.py <old_out_dir> <new_out_dir> [--atol 1e-4]

Every result.npz found under <old_out_dir> is compared with the file at the same relative path under
<new_out_dir>: same keyframes, and slam_poses / fused_poses / metric_poses equal within --atol, same
gps_fusion status. Exits non-zero on any difference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

POSE_KEYS = ("slam_poses", "fused_poses", "metric_poses")


def compare(a_path, b_path, atol):
    a, b = np.load(a_path), np.load(b_path)
    problems = []
    if set(a.files) != set(b.files):
        problems.append(f"fields differ: {sorted(set(a.files) ^ set(b.files))}")
    if "frame_ids" in a.files and "frame_ids" in b.files and not np.array_equal(a["frame_ids"], b["frame_ids"]):
        problems.append("frame_ids differ")
        return problems
    for k in POSE_KEYS:
        if k in a.files and k in b.files:
            d = np.abs(a[k] - b[k]).max() if a[k].shape == b[k].shape else np.inf
            if not d <= atol:
                problems.append(f"{k}: max abs diff {d:.3g}")
    if "gps_fusion_json" in a.files and "gps_fusion_json" in b.files:
        sa, sb = (json.loads(str(x["gps_fusion_json"])).get("status") for x in (a, b))
        if sa != sb:
            problems.append(f"gps_fusion status {sa} -> {sb}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_dir", type=Path)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()
    old = sorted(args.old_dir.rglob("result.npz"))
    if not old:
        sys.exit(f"no result.npz under {args.old_dir}")
    bad = 0
    for p in old:
        rel = p.relative_to(args.old_dir)
        q = args.new_dir / rel
        problems = ["missing in new_dir"] if not q.exists() else compare(p, q, args.atol)
        bad += bool(problems)
        print(("DIFF " if problems else "ok   ") + str(rel) + ("  " + "; ".join(problems) if problems else ""))
    print(f"{len(old) - bad}/{len(old)} identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
