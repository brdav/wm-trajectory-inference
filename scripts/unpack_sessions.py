#!/usr/bin/env python3
"""Reverses scripts/pack_sessions.py: extracts a packed Country/Date/Session.zip back to a
loose Country/Date/Session/ directory and removes the zip. Needed before anything that reads or
rewrites an individual clip's result.npz by path (a plot/diagnose script, a future
scripts/refuse_gps.py sweep) can touch a packed session again -- see
vggt_slam/session_pack.py's docstring for why packing doesn't try to stay transparently
read/write-able instead.

Usage:
    # a single session, path relative to --out_root (.zip suffix optional):
    python3 scripts/unpack_sessions.py --out_root .../natix_traj/trajectories \\
        --session "Switzerland/2025-10-06/2025-10-06_1affc082-e589-4605-a6d2-89404f7241bc_3"

    # every packed session under given countries (all countries if --countries is omitted):
    python3 scripts/unpack_sessions.py --out_root .../natix_traj/trajectories --countries Croatia
"""
import argparse
import zipfile
from pathlib import Path


def _unpack_one(zip_path):
    session_dir = zip_path.with_name(zip_path.stem)  # Session.zip -> Session/
    # Extract to a working name first and rename into place once complete, so a process killed
    # mid-extract never leaves a half-populated Session/ dir sitting where pack_sessions.py's own
    # (out_dir / "result.npz").exists() skip-check -- and everything else -- expects a finished one.
    tmp_dir = session_dir.with_name(session_dir.name + ".unpacking")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    tmp_dir.rename(session_dir)
    zip_path.unlink()
    return session_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", required=True)
    p.add_argument("--session", default=None,
                   help="A single session's path relative to --out_root (.zip suffix optional), "
                        "e.g. 'Switzerland/2025-10-06/2025-10-06_<uuid>_3'")
    p.add_argument("--countries", nargs="+", default=None,
                   help="Unpack every packed (.zip) session under these countries "
                        "(default with no --session given: every country)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    out_root = Path(args.out_root)

    if args.session:
        session_rel = args.session[:-4] if args.session.endswith(".zip") else args.session
        targets = [out_root / (session_rel + ".zip")]
    else:
        country_set = set(args.countries) if args.countries else None
        targets = [zp for zp in sorted(out_root.glob("*/*/*.zip"))
                   if country_set is None or zp.relative_to(out_root).parts[0] in country_set]

    for zip_path in targets:
        if not zip_path.exists():
            print(f"skip (not found): {zip_path}")
            continue
        if args.dry_run:
            print(f"[dry_run] would unpack {zip_path}")
            continue
        session_dir = _unpack_one(zip_path)
        print(f"unpacked {zip_path} -> {session_dir}")


if __name__ == "__main__":
    main()
