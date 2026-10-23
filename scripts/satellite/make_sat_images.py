#!/usr/bin/env python3
"""
make_sat_images.py - the aerial image for every sample that has an OSM map: same 768 px / 0.5 m/px heading-up crop (render_sat.py),
cut from the tile cache of fetch_tiles.py. One record per OSM map image (<osm maps>/index*.jsonl) is rendered at that record's
lat/lon/heading, so image <i> of the OSM maps and of the aerial images show exactly the same ground.

Output (to keep the file count down, ~1.1M images go into one stored zip per country folder and month, not one per session):
  <out>/<folder>/<YYYY-MM>.zip        entries <date>/<session>/<time>_s<idx>.<ext>  (same stem as the OSM map entry),
                                      index.jsonl (one record per image) and unavailable.jsonl (samples without an image)
  <out>/summary<shard>.jsonl          one line per finished zip (counts, reasons, sources used)
After adding sources for a region, `--redo-regions R ...` re-renders only the zips that list unavailable samples of R (atomically replaced;
summary.jsonl then has a second line for those zips, the later one counts).
A sample gets NO image when its region has no imagery source ("no_source") or more than --max-missing of the crop is not covered
("no_imagery"); partly covered crops are kept, the uncovered part is black and `missing` says how much. Each index record lists the
`sources` that contributed pixels (needed for the attribution, see LICENSES.md).
Zips are written to .part and renamed, so a finished zip is final and a rerun skips it (a killed job loses at most the zips in progress).
Bundles with more than --chunk samples are rendered in chunks by several workers (<zip>.chunk<k>-<hash> files, kept if the job is killed and
reused on the rerun) and merged into the final zip, so one big (folder, month) does not occupy a single core for an hour.

  python3 make_sat_images.py --maps <osm_bev/maps_predicted> --cache <tiles dir> --out <dir> [--countries Switzerland] [--shard i/N]
"""
import argparse
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

from render_sat import SatRenderer
from sources import sources_of

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_state = {}


def split_entry(image):
    """'<folder>/<date>/<session>.zip::<time>_s0000.webp' -> (folder, 'YYYY-MM', '<date>/<session>/<time>_s0000')."""
    zname, name = image.split("::")
    parts = zname[:-4].split("/")
    k = next(i for i, p in enumerate(parts) if DATE.match(p))
    return "/".join(parts[:k]), parts[k][:7], "/".join(parts[k:]) + "/" + name.rsplit(".", 1)[0]


def _init(cache, res, ext, quality, max_missing):
    cv2.setNumThreads(1)
    _state.update(cache=cache, res=res, ext=ext, quality=quality, max_missing=max_missing, renderers={})


def _renderer(region):
    r = _state["renderers"]
    if region not in r:
        srcs = sources_of(region)
        r[region] = SatRenderer(_state["cache"], srcs, res=_state["res"]) if srcs else None
    return r[region]


def _encode(rgb):
    bgr = rgb[:, :, ::-1]
    if _state["ext"] == ".webp":
        return cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, _state["quality"]])[1].tobytes()
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _state["quality"]])[1].tobytes()


def _render_into(zp_part, recs):
    """Render `recs` into the stored zip `zp_part` (images + index.jsonl + unavailable.jsonl). Returns (index, unavailable, sources)."""
    index, unavailable, srcs = [], [], collections.Counter()
    with zipfile.ZipFile(zp_part, "w", zipfile.ZIP_STORED) as zf:
        for rec in recs:
            _, _, stem = split_entry(rec["image"])
            r = _renderer(rec["region"])
            if r is None:
                unavailable.append(dict(osm_image=rec["image"], sample_id=rec["sample_id"], region=rec["region"], reason="no_source"))
                continue
            img, missing, names = r.render(rec["lat"], rec["lon"], rec["heading_deg"])
            if missing > _state["max_missing"]:
                unavailable.append(dict(osm_image=rec["image"], sample_id=rec["sample_id"], region=rec["region"],
                                        reason="no_imagery", missing=round(missing, 3)))
                continue
            name = stem + _state["ext"]
            zf.writestr(name, _encode(img))
            index.append(dict(image=name, osm_image=rec["image"], sample_id=rec["sample_id"], sample_index=rec["sample_index"],
                              clip_id=rec["clip_id"], lat=rec["lat"], lon=rec["lon"], heading_deg=rec["heading_deg"],
                              region=rec["region"], sources=names, missing=round(missing, 3)))
            srcs.update(names)
        zf.writestr("index.jsonl", "".join(json.dumps(x) + "\n" for x in index))
        zf.writestr("unavailable.jsonl", "".join(json.dumps(x) + "\n" for x in unavailable))
    return index, unavailable, srcs


def _summary(out, zp, index, unavailable, srcs):
    return dict(zip=str(zp.relative_to(out)), images=len(index), unavailable=dict(collections.Counter(u["reason"] for u in unavailable)),
                sources=dict(srcs))


def process_bundle(job):
    """A whole (folder, month) -> <out>/<folder>/<month>.zip, or only one chunk of a big one -> <zip>.chunk<k>-<hash>
    (merged into the final zip by merge_bundle once all its chunks are there). Returns a summary dict."""
    out, folder, month, recs = job["out"], job["folder"], job["month"], job["records"]
    zp = Path(out) / folder / f"{month}.zip"
    if zp.exists() and not job.get("redo"):
        return dict(zip=str(zp.relative_to(out)), skipped_existing=True)
    zp.parent.mkdir(parents=True, exist_ok=True)
    recs.sort(key=lambda r: r["image"])                                      # consecutive samples share tiles (decode cache)
    if "chunk" in job:
        tag = hashlib.sha1(f"{len(recs)}:{recs[0]['image']}:{recs[-1]['image']}".encode()).hexdigest()[:8]
        cz = zp.with_name(f"{zp.name}.chunk{job['chunk']}-{tag}")
        if not cz.exists():                                                   # a finished chunk of an earlier (killed) run is reused
            part = cz.with_name(cz.name + ".part")
            _render_into(part, recs)
            part.replace(cz)
        return dict(zip=str(zp.relative_to(out)), chunk=job["chunk"], chunk_file=str(cz))
    part = zp.with_name(zp.name + ".part")
    index, unavailable, srcs = _render_into(part, recs)
    part.replace(zp)
    return _summary(out, zp, index, unavailable, srcs)


def merge_bundle(job):
    """Concatenate the chunk zips of one big (folder, month) into the final zip (images are copied as they are), delete the chunks."""
    out, zp, chunks = job["out"], Path(job["out"]) / job["zip"], [Path(c) for c in job["chunks"]]
    part = zp.with_name(zp.name + ".part")
    index, unavailable, srcs = [], [], collections.Counter()
    with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as zf:
        for c in chunks:
            with zipfile.ZipFile(c) as cf:
                for name in cf.namelist():
                    if name == "index.jsonl":
                        index += [json.loads(l) for l in cf.read(name).decode().splitlines()]
                    elif name == "unavailable.jsonl":
                        unavailable += [json.loads(l) for l in cf.read(name).decode().splitlines()]
                    else:
                        zf.writestr(name, cf.read(name))
        zf.writestr("index.jsonl", "".join(json.dumps(x) + "\n" for x in index))
        zf.writestr("unavailable.jsonl", "".join(json.dumps(x) + "\n" for x in unavailable))
    for x in index:
        srcs.update(x["sources"])
    part.replace(zp)
    for c in chunks:
        c.unlink()
    return _summary(out, zp, index, unavailable, srcs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True, help="OSM map output dir (index*.jsonl, one record per map image)")
    ap.add_argument("--cache", required=True, help="tile cache written by fetch_tiles.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--countries", nargs="*", help="only these country folders of the dataset, e.g. Switzerland 'United States/Nevada'")
    ap.add_argument("--limit", type=int, help="random subset of this many records per bundle (test)")
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--format", choices=["webp", "jpg"], default="webp")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--max-missing", type=float, default=0.2, help="largest uncovered fraction of a crop that is still written")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--redo-regions", nargs="*", default=[],
                    help="re-render the existing zips that list samples of these regions as unavailable (after adding sources for them)")
    ap.add_argument("--chunk", type=int, default=6000, help="bundles with more samples are rendered in chunks of this size in parallel")
    ap.add_argument("--shard", default="0/1", help="i/N: only zips with hash(folder, month) %% N == i (Slurm array)")
    a = ap.parse_args()

    si, sn = (int(v) for v in a.shard.split("/"))
    tag = "" if sn == 1 else f".shard{si}of{sn}"
    bundles = collections.defaultdict(list)
    n_all = 0
    for f in sorted(glob.glob(f"{a.maps}/index*.jsonl")):
        with open(f) as fh:
            for ln in fh:
                rec = json.loads(ln)
                folder, month, _ = split_entry(rec["image"])
                if a.countries and not any(folder == c or folder.startswith(c + "/") for c in a.countries):
                    continue
                n_all += 1
                if int(hashlib.sha1(f"{folder}:{month}".encode()).hexdigest(), 16) % sn == si:
                    bundles[(folder, month)].append({k: rec[k] for k in ("image", "sample_id", "sample_index", "clip_id", "lat", "lon",
                                                                           "heading_deg", "region")})
    if a.limit:
        import random
        for k, v in bundles.items():
            bundles[k] = random.Random(0).sample(v, min(a.limit, len(v)))
    redo = set()
    if a.redo_regions:                                                      # existing zips with unavailable samples in these regions
        for (f, m) in bundles:
            zp = Path(a.out) / f / f"{m}.zip"
            if zp.exists():
                with zipfile.ZipFile(zp) as zf:
                    if any(json.loads(l)["region"] in a.redo_regions for l in zf.read("unavailable.jsonl").decode().splitlines()):
                        redo.add((f, m))
        print(f"redo: {len(redo)} existing zips list unavailable samples in {a.redo_regions}", flush=True)
    jobs, need = [], {}
    for (f, m), r in bundles.items():
        if len(r) <= a.chunk:
            jobs.append(dict(out=a.out, folder=f, month=m, records=r, redo=(f, m) in redo))
            continue
        r.sort(key=lambda x: x["image"])
        parts = [r[i:i + a.chunk] for i in range(0, len(r), a.chunk)]
        need[(f, m)] = dict(n=len(parts), files=[], zip=str((Path(f) / f"{m}.zip")))
        jobs += [dict(out=a.out, folder=f, month=m, records=p, chunk=k, redo=(f, m) in redo) for k, p in enumerate(parts)]
    jobs.sort(key=lambda j: -len(j["records"]))                             # biggest first: the long ones start early
    print(f"{n_all} records, shard {si}/{sn}: {len(bundles)} zips ({len(need)} rendered in chunks), {len(jobs)} jobs, "
          f"{sum(len(j['records']) for j in jobs)} samples", flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    tot, srcs, un = 0, collections.Counter(), collections.Counter()
    pending = []
    with open(Path(a.out) / f"summary{tag}.jsonl", "a") as summ, \
            mp.Pool(a.workers, initializer=_init, initargs=(a.cache, a.res, ".webp" if a.format == "webp" else ".jpg",
                                                            a.quality, a.max_missing)) as pool:
        def record(s):
            nonlocal tot
            summ.write(json.dumps(s) + "\n")
            summ.flush()
            tot += s.get("images", 0)
            srcs.update(s.get("sources", {}))
            un.update(s.get("unavailable", {}))
            print(f"  zip {s['zip']}  images so far {tot}  unavailable {dict(un)}", flush=True)

        for i, s in enumerate(pool.imap_unordered(process_bundle, jobs, chunksize=1), 1):
            if "chunk" in s:
                key = next(k for k in need if str(Path(k[0]) / f"{k[1]}.zip") == s["zip"])
                need[key]["files"].append(s["chunk_file"])
                if len(need[key]["files"]) == need[key]["n"]:                # all chunks there: merge in the pool
                    pending.append(pool.apply_async(merge_bundle, (dict(out=a.out, zip=s["zip"], chunks=sorted(need[key]["files"])),)))
            elif not s.get("skipped_existing"):
                record(s)
            if i % 25 == 0:
                print(f"  jobs {i}/{len(jobs)}", flush=True)
        for p in pending:
            record(p.get())
    print(f"done: {tot} images in {a.out}; unavailable {dict(un)}; images per source {dict(srcs)}")


if __name__ == "__main__":
    main()
