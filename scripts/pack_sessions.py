#!/usr/bin/env python3
"""Packs finalized recording sessions in --out_root into single .zip archives, replacing each
Country/Date/Session/ directory tree with a sibling Country/Date/Session.zip -- collapses the
per-clip leaf nesting (HH-MM-SS/FRONT_FOLDER/FRONT_<timestamp>/ -- ~3 directories/clip of pure
input-mirroring with no information beyond what the zip's internal paths already carry, once
that clip's own file count was already cut to 1-2 by save_result_npz) down to one inode per
session, on top of the per-clip consolidation already in place.

Archival, not transparent -- see vggt_slam/session_pack.py's docstring for why. A packed session
is meant to be done for good: scripts/batch_infer.py's discover_clips skip-check already knows a
packed session counts as complete (vggt_slam.session_pack.is_session_packed), so the live
pipeline won't reprocess it, but anything that reads/rewrites one clip's result.npz by path
(plot/diagnose scripts, a future scripts/refuse_gps.py sweep) just gets FileNotFoundError against
a packed session until it's restored with scripts/unpack_sessions.py.

A session is packed only once ALL of:
  1. Complete: every FRONT_*.mp4 under the session's mirrored --dataset_root directory has a
     result.npz in the output tree (i.e. discover_clips would yield nothing for it).
  2. Quiet: nothing under the session's output directory has been modified in the last
     --min_age_days days -- a cheap guard against packing a session a currently-running or
     just-resumed batch_infer.py job is still mid-write on (no cross-process lock exists for
     this today, same as the rest of the pipeline).
  3. Not already packed (no sibling Session.zip yet).

Sessions are independent, so it's safe to run several invocations concurrently against disjoint
--countries if a full sweep needs to go faster.

Usage (dry run first -- packing deletes the original directory once its zip is verified):
    python3 scripts/pack_sessions.py --dataset_root .../datasets/natix \\
        --out_root .../natix_traj/trajectories --dry_run
    python3 scripts/pack_sessions.py --dataset_root .../datasets/natix \\
        --out_root .../natix_traj/trajectories [--countries Switzerland Croatia] \\
        [--min_age_days 7]

Appends one record per packed session to --out_root/packed_sessions.jsonl (same append-only
convention as batch_infer.py's own manifest.jsonl).
"""
import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_infer import _discover_all_video_paths  # noqa: E402


def _iter_sessions(dataset_root, out_root, countries):
    """Groups _discover_all_video_paths's clips by session (two parents above each clip's
    FRONT_FOLDER video, mirrored into out_root -- same convention as
    vggt_slam.undistort.session_cache_dir), returning {session_out_dir: [clip video paths]}.
    Reuses the all_video_paths.txt cache via _discover_all_video_paths instead of re-walking
    dataset_root, for the same reason batch_infer.py itself does: packing runs repeatedly over
    an effectively-unchanging dataset_root, so re-paying the walk on every invocation would be
    wasteful at ~120k-clip scale."""
    dataset_root = Path(dataset_root)
    country_set = set(countries) if countries else None
    sessions = {}
    for video_path in _discover_all_video_paths(dataset_root, out_root):
        rel = video_path.relative_to(dataset_root)
        if country_set is not None and rel.parts[0] not in country_set:
            continue
        session_in_dir = video_path.parents[2]
        session_out_dir = Path(out_root) / session_in_dir.relative_to(dataset_root)
        sessions.setdefault(session_out_dir, []).append(video_path)
    return sessions


def _is_complete(video_paths, dataset_root, out_root):
    for video_path in video_paths:
        rel = video_path.relative_to(dataset_root)
        clip_out_dir = Path(out_root) / rel.with_suffix("")
        if not (clip_out_dir / "result.npz").exists():
            return False
    return True


def _is_quiet(session_dir, min_age_days):
    cutoff = time.time() - min_age_days * 86400
    for dirpath, _, filenames in os.walk(session_dir):
        for fname in filenames:
            if os.path.getmtime(os.path.join(dirpath, fname)) > cutoff:
                return False
    return True


def _pack_one(session_dir):
    """Zips session_dir's whole subtree (arcnames relative to session_dir, so the archive's
    internal layout matches what it replaces) to a sibling Session.zip, verifies it, then
    removes the original directory. Returns (zip_path, n_files)."""
    files = sorted(p for p in session_dir.rglob("*") if p.is_file())
    zip_path = session_dir.parent / (session_dir.name + ".zip")
    tmp_path = zip_path.with_name(zip_path.name + ".tmp")

    # ZIP_STORED, not DEFLATE: result.npz and undistort_maps.npz are themselves zip/npz
    # containers already, and network.osm is small relative to those -- recompressing buys
    # nothing here and only costs CPU across thousands of sessions.
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in files:
            zf.write(p, arcname=str(p.relative_to(session_dir)))

    with zipfile.ZipFile(tmp_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad!r} in freshly-written {tmp_path}")
        if len(zf.namelist()) != len(files):
            raise RuntimeError(
                f"{tmp_path}: wrote {len(zf.namelist())} members, expected {len(files)}")

    # Atomic: a crash between here and rmtree leaves a verified Session.zip sitting next to the
    # still-intact Session/ dir -- next run just sees zip_path.exists() and skips it as
    # already-packed, then a human can rmtree the leftover directory by hand. Never a state
    # where data was deleted without a verified archive to replace it.
    os.replace(tmp_path, zip_path)
    shutil.rmtree(session_dir)
    return zip_path, len(files)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", required=True, help="Read-only source dataset root (same tree discover_clips walks)")
    p.add_argument("--out_root", required=True, help="Trajectories output root (e.g. .../natix_traj/trajectories)")
    p.add_argument("--countries", nargs="+", default=None, help="Restrict to these country subdirs (default: all)")
    p.add_argument("--min_age_days", type=float, default=7.0,
                   help="Skip a session if anything under it was modified more recently than this many days ago (default: 7)")
    p.add_argument("--limit", type=int, default=None, help="Stop after packing this many sessions (for testing)")
    p.add_argument("--dry_run", action="store_true", help="Report what would be packed without writing or deleting anything")
    args = p.parse_args()

    out_root = Path(args.out_root)
    manifest_f = None if args.dry_run else open(out_root / "packed_sessions.jsonl", "a")

    sessions = _iter_sessions(args.dataset_root, out_root, args.countries)
    print(f"{len(sessions)} sessions found under --countries filter", flush=True)

    n_packed = n_incomplete = n_busy = n_already = 0
    for session_out_dir, video_paths in sessions.items():
        zip_path = session_out_dir.parent / (session_out_dir.name + ".zip")
        if zip_path.exists():
            n_already += 1
            continue
        if not _is_complete(video_paths, args.dataset_root, out_root):
            n_incomplete += 1
            continue
        if not session_out_dir.is_dir() or not _is_quiet(session_out_dir, args.min_age_days):
            n_busy += 1
            continue

        session_rel = str(session_out_dir.relative_to(out_root))
        if args.dry_run:
            n_files = sum(1 for p in session_out_dir.rglob("*") if p.is_file())
            print(f"[dry_run] would pack {session_rel} ({n_files} files, {len(video_paths)} clips)", flush=True)
            n_packed += 1
        else:
            zp, n_files = _pack_one(session_out_dir)
            n_packed += 1
            print(f"[{n_packed}] packed {session_rel} -> {zp.name} ({n_files} files)", flush=True)
            manifest_f.write(json.dumps({
                "session": session_rel, "zip_path": str(zp), "n_files": n_files,
                "n_clips": len(video_paths), "packed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }) + "\n")
            manifest_f.flush()

        if args.limit and n_packed >= args.limit:
            break

    if manifest_f is not None:
        manifest_f.close()
    print(f"done: packed {n_packed}, already-packed {n_already}, "
          f"incomplete {n_incomplete}, too-recent/missing {n_busy}", flush=True)


if __name__ == "__main__":
    main()
