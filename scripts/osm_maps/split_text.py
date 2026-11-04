#!/usr/bin/env python3
"""Take the route text out of a finished OSM map folder (one-off for runs made before make_osm_maps.py kept the text apart).

  python scripts/osm_maps/split_text.py --maps <dir>/maps_predicted --text <dir>/text_predicted            # check only
  python scripts/osm_maps/split_text.py --maps <dir>/maps_predicted --text <dir>/text_predicted --apply    # strip

--text is the folder written by `make_osm_maps.py --text-only` (or by a normal run) for the same samples. A session zip of --maps
is only rewritten when its text.jsonl is identical to the one in the same zip of --text; it then loses text.jsonl and the caption
in its index.jsonl. The top-level index*.jsonl lose the `caption` field. Sessions without text.jsonl are left alone, so the script
can be run again after an interruption. Do not run it while a job writes to either folder.
"""
import argparse
import json
import multiprocessing as mp
import zipfile
from pathlib import Path


def lines(zf, name):
    return [json.loads(x) for x in zf.read(name).decode().splitlines() if x]


def strip_zip(job):
    """-> (relative zip, status): 'stripped' | 'would strip' | 'no text' | 'no text copy' | 'differs'."""
    zp, tp, apply = job
    rel = zp.relative_to(job_root[0]).as_posix()
    with zipfile.ZipFile(zp) as zf:
        if "text.jsonl" not in zf.namelist():
            return rel, "no text"
        if not tp.exists():
            return rel, "no text copy"
        with zipfile.ZipFile(tp) as tf:
            if "text.jsonl" not in tf.namelist() or lines(zf, "text.jsonl") != lines(tf, "text.jsonl"):
                return rel, "differs"
        if not apply:
            return rel, "would strip"
        part = zp.with_name(zp.name + ".part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as out:
            for info in zf.infolist():
                if info.filename == "text.jsonl":
                    continue
                data = zf.read(info.filename)
                if info.filename == "index.jsonl":
                    data = "".join(json.dumps({k: v for k, v in r.items() if k != "caption"}) + "\n"
                                   for r in lines(zf, "index.jsonl")).encode()
                out.writestr(info.filename, data)
    part.replace(zp)
    return rel, "stripped"


def init(root):
    job_root[:] = [root]


job_root = [None]


def strip_index(path):
    tmp = path.with_name(path.name + ".part")
    n = 0
    with open(path) as f, open(tmp, "w") as out:
        for line in f:
            if not line.strip():
                continue
            out.write(json.dumps({k: v for k, v in json.loads(line).items() if k != "caption"}) + "\n")
            n += 1
    tmp.replace(path)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--apply", action="store_true", help="rewrite the files (default: only check)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    maps, text = Path(a.maps), Path(a.text)

    zips = sorted(maps.rglob("*.zip"))
    jobs = [(z, text / z.relative_to(maps), a.apply) for z in zips]
    counts = {}
    with mp.Pool(a.workers, initializer=init, initargs=(maps,)) as pool:
        for i, (rel, status) in enumerate(pool.imap_unordered(strip_zip, jobs, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if status in ("differs", "no text copy"):
                print(f"  {status}: {rel}", flush=True)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} {counts}", flush=True)
    print(f"{len(zips)} zips: {counts}")
    blocked = counts.get("differs", 0) + counts.get("no text copy", 0)
    if blocked:
        print(f"{blocked} zips were NOT changed (their text is not in --text or differs): the index files are left as they are")
        return 1
    if a.apply:
        for p in sorted(maps.glob("index*.jsonl")):
            print(f"{p.name}: {strip_index(p)} records without caption")
    else:
        print("check only: rerun with --apply to strip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
