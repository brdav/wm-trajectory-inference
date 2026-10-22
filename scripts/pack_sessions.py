#!/usr/bin/env python3
"""Packs finalized recording sessions in --out_root into single .zip archives, replacing each
Country/Date/Session/ directory tree with a sibling Country/Date/Session.zip -- collapses the
per-clip leaf nesting (HH-MM-SS/FRONT_FOLDER/FRONT_<timestamp>/ -- ~3 directories/clip of pure
input-mirroring with no information beyond what the zip's internal paths already carry, once
that clip's own file count was already cut to 1-2 by save_result_npz) down to one inode per
session, on top of the per-clip consolidation already in place.

Archival, not transparent -- see vggt_slam/session_pack.py's docstring for why. A packed session
is meant to be done for good: scripts/batch_infer.py's discover_clips finds a clip's result inside the
session zip and does not reprocess it, but anything that reads/rewrites one clip's result.npz by path
(plot/diagnose scripts, a future scripts/tools/refuse_gps.py sweep) just gets FileNotFoundError against
a packed session until it's restored with scripts/tools/unpack_sessions.py (the metric-scale pipeline reads
and updates the zips directly).

Which sessions exist and which clips belong to them comes from the dataset's sample manifest (the same
list scripts/batch_infer.py processes, via vggt_slam/sample_manifest.py); the dataset tree is not walked.

A session directory is packed only once ALL of:
  1. Complete: every source video of that session listed in the sample manifest has a result.npz,
     either loose in the session directory or already inside the session's zip.
  2. Quiet: nothing under the session's output directory has been modified in the last
     --min_age_days days -- a cheap guard against packing a session a currently-running or
     just-resumed batch_infer.py job is still mid-write on (no cross-process lock exists for
     this today, same as the rest of the pipeline). run_full_dataset.sbatch packs right after its
     batch_infer.py run has finished, so it passes 0.

If the session already has a zip (it was packed before) and a loose session directory has appeared next
to it -- new clips of that session were processed later -- the loose files are MERGED into the zip
(a verified new zip replaces the old one; same-named members such as a re-run result.npz are taken from
the loose directory). A session with a zip and no loose directory is simply skipped.

Sessions are independent, so it is safe to run several invocations concurrently against disjoint
--countries if a full sweep needs to go faster.

Usage (dry run first -- packing deletes the original directory once its zip is verified; the dataset root,
output root and manifest default to the natix locations):
    python3 scripts/pack_sessions.py --dry_run
    python3 scripts/pack_sessions.py --min_age_days 0 [--countries Switzerland Croatia]

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (vggt_slam package)
from vggt_slam.paths import DATASET_ROOT, SAMPLE_MANIFEST, TRAJ_ROOT  # noqa: E402
from vggt_slam.sample_manifest import source_videos  # noqa: E402


def _iter_sessions(dataset_root, out_root, countries, sample_manifest):
    """Groups the sample manifest's source videos by session (two parents above each clip's
    FRONT_FOLDER video, mirrored into out_root -- same convention as
    vggt_slam.undistort.session_cache_dir), returning {session_out_dir: [clip video paths]}."""
    dataset_root = Path(dataset_root)
    sessions = {}
    for rel in source_videos(sample_manifest, countries):
        video_path = dataset_root / rel
        session_out_dir = Path(out_root) / video_path.parents[2].relative_to(dataset_root)
        sessions.setdefault(session_out_dir, []).append(video_path)
    return sessions


def _is_complete(video_paths, dataset_root, out_root, zip_path=None):
    """Every video has a result.npz, loose in out_root or (if the session already has a zip) inside it."""
    zip_names = None
    if zip_path is not None and zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zip_names = set(zf.namelist())
    for video_path in video_paths:
        rel = video_path.relative_to(dataset_root)
        clip_out_dir = Path(out_root) / rel.with_suffix("")
        if (clip_out_dir / "result.npz").exists():
            continue
        if zip_names is not None and (Path(*rel.parts[-3:]).with_suffix("") / "result.npz").as_posix() in zip_names:
            continue
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
    removes the original directory. If Session.zip already exists the loose files are merged into it
    (members with the same arcname are taken from the loose directory, all other old members are kept).
    Returns (zip_path, n_files_in_zip, merged)."""
    files = sorted(p for p in session_dir.rglob("*") if p.is_file())
    loose = {str(p.relative_to(session_dir)): p for p in files}
    zip_path = session_dir.parent / (session_dir.name + ".zip")
    tmp_path = zip_path.with_name(zip_path.name + ".tmp")
    merged = zip_path.exists()

    # ZIP_STORED, not DEFLATE: result.npz and undistort_maps.npz are themselves zip/npz
    # containers already, and network.osm is small relative to those -- recompressing buys
    # nothing here and only costs CPU across thousands of sessions.
    n_expected = len(loose)
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        if merged:
            with zipfile.ZipFile(zip_path) as old:
                for info in old.infolist():
                    if info.filename not in loose:
                        zf.writestr(info.filename, old.read(info.filename))
                        n_expected += 1
        for arc, p in loose.items():
            zf.write(p, arcname=arc)

    with zipfile.ZipFile(tmp_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad!r} in freshly-written {tmp_path}")
        if len(zf.namelist()) != n_expected:
            raise RuntimeError(
                f"{tmp_path}: wrote {len(zf.namelist())} members, expected {n_expected}")

    # Atomic: a crash between here and rmtree leaves a verified Session.zip sitting next to the
    # still-intact Session/ dir -- next run just sees both, merges again (idempotent) and removes
    # the loose directory. Never a state where data was deleted without a verified archive to replace it.
    os.replace(tmp_path, zip_path)
    shutil.rmtree(session_dir)
    return zip_path, n_expected, merged


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", default=DATASET_ROOT, help="Source dataset root that the sample manifest's relative video paths are resolved against")
    p.add_argument("--out_root", default=TRAJ_ROOT, help="Trajectories output root (batch_infer.py's --out_root)")
    p.add_argument("--sample_manifest", default=SAMPLE_MANIFEST, help="Sample manifest (parquet) listing the source videos, i.e. which clips each session should contain")
    p.add_argument("--countries", nargs="+", default=None, help="Restrict to these country subdirs (default: all)")
    p.add_argument("--min_age_days", type=float, default=7.0,
                   help="Skip a session if anything under it was modified more recently than this many days ago (default: 7)")
    p.add_argument("--limit", type=int, default=None, help="Stop after packing this many sessions (for testing)")
    p.add_argument("--dry_run", action="store_true", help="Report what would be packed without writing or deleting anything")
    args = p.parse_args()

    out_root = Path(args.out_root)
    manifest_f = None if args.dry_run else open(out_root / "packed_sessions.jsonl", "a")

    sessions = _iter_sessions(args.dataset_root, out_root, args.countries, args.sample_manifest)
    print(f"{len(sessions)} sessions found under --countries filter", flush=True)

    n_packed = n_merged = n_incomplete = n_busy = n_already = 0
    for session_out_dir, video_paths in sessions.items():
        zip_path = session_out_dir.parent / (session_out_dir.name + ".zip")
        if zip_path.exists() and not session_out_dir.is_dir():
            n_already += 1
            continue
        if not _is_complete(video_paths, args.dataset_root, out_root, zip_path):
            n_incomplete += 1
            continue
        if not session_out_dir.is_dir() or not _is_quiet(session_out_dir, args.min_age_days):
            n_busy += 1
            continue

        session_rel = str(session_out_dir.relative_to(out_root))
        will_merge = zip_path.exists()
        if args.dry_run:
            n_files = sum(1 for p in session_out_dir.rglob("*") if p.is_file())
            print(f"[dry_run] would {'merge into the existing zip of' if will_merge else 'pack'} {session_rel} "
                  f"({n_files} loose files, {len(video_paths)} clips)", flush=True)
            n_packed += 1
            n_merged += will_merge
        else:
            zp, n_files, merged = _pack_one(session_out_dir)
            n_packed += 1
            n_merged += merged
            print(f"[{n_packed}] {'merged into' if merged else 'packed'} {session_rel} -> {zp.name} ({n_files} files)", flush=True)
            manifest_f.write(json.dumps({
                "session": session_rel, "zip_path": str(zp), "n_files": n_files, "merged": merged,
                "n_clips": len(video_paths), "packed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }) + "\n")
            manifest_f.flush()

        if args.limit and n_packed >= args.limit:
            break

    if manifest_f is not None:
        manifest_f.close()
    print(f"done: packed {n_packed} (of which merged into an existing zip: {n_merged}), already-packed {n_already}, "
          f"incomplete {n_incomplete}, too-recent/missing {n_busy}", flush=True)


if __name__ == "__main__":
    main()
