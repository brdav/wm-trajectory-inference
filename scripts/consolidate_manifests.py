#!/usr/bin/env python3
"""One-time consolidation of manifest.jsonl + the gps_refusion_*_manifest.jsonl overlay files
into a single, deduplicated manifest.jsonl -- one row per clip, the latest/authoritative outcome
only.

Both manifest.jsonl (batch_infer.py's own, append-only across every run_full_dataset.sbatch
resubmission) and the various gps_refusion_*_manifest.jsonl overlays (scripts/refuse_gps.py's
own, one row appended per re-fusion attempt) had accumulated many historical rows per clip_id --
"a clip's LATEST record is authoritative" has been this project's convention all along, but that
requires every reader to know to dedupe by clip_id and to know an overlay file even exists and
takes priority. This script does that dedup+merge once, so manifest.jsonl alone is the single,
current source of truth going forward -- readers no longer need to reconstruct the merge logic
themselves.

Merge rule: for each clip_id, take the overlay's record if it appears there (a refuse_gps.py
re-fusion is always a deliberate, later, targeted re-processing of that specific clip's GPS
fusion -- authoritative over whatever batch_infer.py originally recorded), else the base
manifest's own (already deduped-to-latest) record.
"""
import argparse
import json
import shutil
from datetime import date


def dedup_latest(path):
    latest = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            latest[r["clip_id"]] = r
    return latest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest_path", help="batch_infer.py's own manifest.jsonl (rewritten in place)")
    ap.add_argument("overlay_paths", nargs="+", help="gps_refusion_*_manifest.jsonl overlay(s), applied in order given (later ones win on conflict)")
    ap.add_argument("--dry_run", action="store_true", help="report counts only, don't write anything")
    args = ap.parse_args()

    base = dedup_latest(args.manifest_path)
    print(f"{args.manifest_path}: {len(base)} unique clips")

    merged = dict(base)
    for overlay_path in args.overlay_paths:
        overlay = dedup_latest(overlay_path)
        print(f"{overlay_path}: {len(overlay)} unique clips ({sum(1 for c in overlay if c not in base)} not in base manifest)")
        merged.update(overlay)

    print(f"\nconsolidated: {len(merged)} unique clips (was {sum(1 for _ in open(args.manifest_path))} raw lines in base manifest alone)")

    if args.dry_run:
        print("--dry_run: not writing anything")
        return

    backup_path = f"{args.manifest_path}.pre_consolidation_{date.today().isoformat()}"
    shutil.copy2(args.manifest_path, backup_path)
    print(f"backed up original to {backup_path}")

    with open(args.manifest_path, "w") as f:
        for clip_id in sorted(merged):
            f.write(json.dumps(merged[clip_id]) + "\n")
    print(f"rewrote {args.manifest_path} ({len(merged)} records, one per clip, sorted by clip_id)")


if __name__ == "__main__":
    main()
