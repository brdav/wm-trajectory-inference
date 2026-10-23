#!/usr/bin/env python3
"""
fetch_tiles.py - download the aerial tiles that the satellite crops of a region need, into the tile cache (see tiles.py).

The tiles are taken from the OSM map index (one record per image, lat/lon of the crop centre): every tile touched by the
axis-aligned window around any sample is fetched ONCE, however many samples/clips share it. Work is split per parent tile
(one zip), so it is resumable: finished parents are skipped. Requests share one global rate limit.

  python3 fetch_tiles.py --maps <osm_bev/maps_predicted> --cache <dir> --regions Switzerland Austria [--limit-samples N]
"""
import argparse
import glob
import json
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

import cog
import sheets
from sources import SOURCES, ZOOM, sources_of, tile_url, tiles_in_source
from tiles import PARENT_DZ, TileCache, parent_tiles, parent_zip, read_parent, tile_centre_lonlat, window_tile_ids, write_parent

UA = "sdsc-world-model-research/0.1 (david.bruggemann@datascience.ch)"


class Limiter:
    def __init__(self, per_s):
        self.dt, self.t, self.lock = 1.0 / per_s, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            self.t = max(self.t + self.dt, now)
            d = self.t - now
        if d > 0:
            time.sleep(d)


def is_blank(b):
    """WMS/tile servers answer outside their coverage with a flat white/black/transparent image instead of an error."""
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return im is None or im.std() < 1.5


def crop_left(b, n):
    """Cut the n extra left columns requested from services that burn a text strip into the left edge."""
    if not n or b is None:
        return b
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, n:]
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def get_tile(sess, lim, url, tries=10):
    """jpeg bytes, None if the source has no (non-blank) tile there; raises after `tries` failed attempts (backoff up to 60 s, ~6 min:
    servers have had outages of a minute or more)."""
    for k in range(tries):
        lim.wait()
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") and len(r.content) > 0:
                return None if is_blank(r.content) else r.content
            if r.status_code in (404, 204, 410) or (r.status_code == 200 and len(r.content) == 0):
                return None
            if r.status_code == 400 and b"OutOfRange" in r.content[:2000]:    # WMTS TileOutOfRange: outside the service's tile grid
                return None
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"giving up: {url}")


def read_samples(maps, regions, limit, seed):
    by = defaultdict(lambda: ([], []))
    for f in sorted(glob.glob(f"{maps}/index*.jsonl")):
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                if r["region"] in regions:
                    by[r["region"]][0].append(r["lat"])
                    by[r["region"]][1].append(r["lon"])
    out = {}
    for reg, (la, lo) in by.items():
        idx = list(range(len(la)))
        if limit and len(idx) > limit:
            idx = random.Random(seed).sample(idx, limit)
        out[reg] = (np.array(la)[idx], np.array(lo)[idx])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--half-m", type=float, default=285.0)
    ap.add_argument("--limit-samples", type=int, help="random subset of samples per region (pilot)")
    ap.add_argument("--rate", type=float, default=10.0, help="max requests per second (all workers)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only count the tiles per source (and those no source serves)")
    a = ap.parse_args()

    n = 2 ** ZOOM
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    lim = Limiter(a.rate)
    for reg, (la, lo) in read_samples(a.maps, set(a.regions), a.limit_samples, a.seed).items():
        ids = window_tile_ids(la, lo, ZOOM, a.half_m)
        xs, ys = ids % n, ids // n
        srcs = sources_of(reg)
        print(f"{reg}: {len(la)} samples, {len(ids)} tiles, sources {srcs or 'NONE'}", flush=True)
        remaining = np.ones(len(ids), bool)                       # tiles no source has yet
        for src in srcs:
            sel = np.flatnonzero(remaining & tiles_in_source(a.cache, src, *tile_centre_lonlat(xs, ys, ZOOM)))
            if a.dry_run and SOURCES[src]["kind"] != "imported":
                print(f"  [{src}] {len(sel)} tiles")
                remaining[sel] = False
                continue
            if SOURCES[src]["kind"] == "imported":                       # nothing to download: it covers what import_tiles.py brought in
                tc = TileCache(a.cache, src, ZOOM)
                got = np.array([tc.get(x, y) is not None for x, y in zip(xs[sel].tolist(), ys[sel].tolist())], bool)
                remaining[sel[got]] = False
                print(f"  [{src}] imported: has imagery for {int(got.sum())} of {len(sel)} tiles", flush=True)
                continue
            parents = defaultdict(list)
            for x, y in zip(xs[sel].tolist(), ys[sel].tolist()):
                parents[(x >> PARENT_DZ, y >> PARENT_DZ)].append((x, y))
            have = {p: parent_tiles(parent_zip(a.cache, src, ZOOM, *p)) for p in parents}        # earlier runs may have fetched fewer tiles
            todo = {p: [t for t in tl if t not in have[p]] for p, tl in parents.items() if any(t not in have[p] for t in tl)}
            print(f"  [{src}] {len(sel)} tiles in {len(parents)} parents ({len(todo)} to fetch)", flush=True)
            t0, done, miss = time.time(), 0, 0

            def do_parent(item, src=src):
                p, tl = item                                           # tl: tiles not in the parent zip yet
                kind = SOURCES[src]["kind"]
                if kind in ("cog", "sheets"):                          # raster sources are painted in blocks of 16 x 16 tiles
                    blocks = defaultdict(list)
                    for x, y in tl:
                        blocks[(x >> 4, y >> 4)].append((x, y))
                    tiles = {}
                    for bl in blocks.values():
                        tiles.update(cog.render_tiles(SOURCES[src], ZOOM, bl) if kind == "cog"
                                     else sheets.render_tiles(a.cache, src, ZOOM, bl, sess))
                else:
                    tiles = {(x, y): crop_left(get_tile(sess, lim, tile_url(src, ZOOM, x, y)), SOURCES[src].get("crop_left", 0))
                             for x, y in tl}
                path = parent_zip(a.cache, src, ZOOM, *p)
                write_parent(path, {**read_parent(path), **tiles})
                return len(tl), sum(b is None for b in tiles.values())

            def safe_parent(item, f=do_parent):
                try:
                    return f(item)
                except Exception as e:                                  # one dead server must not stop the other blocks: rerun later
                    print(f"    FAILED block {item[0]}: {e}", flush=True)
                    return 0, 0

            with ThreadPoolExecutor(a.workers) as ex:
                for k, (nt, nm) in enumerate(ex.map(safe_parent, todo.items()), 1):
                    done, miss = done + nt, miss + nm
                    if k % 20 == 0 or k == len(todo):
                        print(f"    {k}/{len(todo)} parents, {done} tiles ({miss} missing), {done / (time.time() - t0):.1f} tiles/s", flush=True)
            tc = TileCache(a.cache, src, ZOOM)
            got = np.array([tc.get(x, y) is not None for x, y in zip(xs[sel].tolist(), ys[sel].tolist())], bool)
            remaining[sel[got]] = False
            print(f"  [{src}] has imagery for {int(got.sum())} of {len(sel)} tiles", flush=True)
        print(f"{reg}: {'no source for' if a.dry_run else 'no imagery for'} {int(remaining.sum())} of {len(ids)} tiles ({100 * remaining.mean():.1f} %)", flush=True)


if __name__== "__main__":
    main()
