#!/usr/bin/env python3
"""
render_sat.py - ego-centred, heading-up aerial crop on exactly the same pixel grid as osm_bev.MapRenderer (768 px, 0.5 m/px,
aeqd frame centred on the car, heading up), sampled from the tile cache written by fetch_tiles.py.

  python3 render_sat.py pilot --maps <osm_bev/maps_predicted> --cache <dir> --regions Switzerland --n 6 --out <dir>
    (writes OSM | satellite | 50 % overlay side-by-sides to check the alignment)
"""
import argparse
import glob
import json
import math
import random
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer
from sources import ZOOM, sources_of
from tiles import TILE_PX, TileCache, tile_xy


class SatRenderer:
    def __init__(self, cache, sources, px=768, res=0.5, z=ZOOM, grid=49):
        """sources: the region's source names in priority order; a tile comes from the first one that has it."""
        self.sources = list(sources)
        self.tcs, self.px, self.res, self.z, self.grid = [TileCache(cache, s, z) for s in self.sources], px, res, z, grid

    @lru_cache(maxsize=384)                 # consecutive samples of a clip share most tiles
    def _tile(self, x, y):
        """(decoded BGR tile, index of the source it came from) or (None, -1)."""
        for k, tc in enumerate(self.tcs):
            b = tc.get(x, y)
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
            if im is not None and im.shape[:2] == (TILE_PX, TILE_PX):
                return im, k
        return None, -1

    def render(self, lat, lon, heading_deg):
        """-> (RGB uint8 (px, px, 3), fraction of pixels without imagery, names of the sources that contributed pixels)."""
        px, g = self.px, self.grid
        th = math.radians(heading_deg)
        # output pixel centres on a coarse grid (the mapping is smooth), in the heading-up frame: x right, y up [m]
        u = np.linspace(0.5, px - 0.5, g)
        x, y = np.meshgrid((u - px / 2) * self.res, (px / 2 - u) * self.res)
        e, n = x * math.cos(th) + y * math.sin(th), -x * math.sin(th) + y * math.cos(th)    # inverse of osm_bev._local_frame
        aeqd = Transformer.from_crs(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs", "EPSG:4326", always_xy=True)
        lo, la = aeqd.transform(e, n)
        tx, ty = tile_xy(lo, la, self.z)
        x0, x1, y0, y1 = int(tx.min()), int(tx.max()), int(ty.min()), int(ty.max())
        mos = np.zeros(((y1 - y0 + 1) * TILE_PX, (x1 - x0 + 1) * TILE_PX, 3), np.uint8)
        src = np.zeros(mos.shape[:2], np.uint8)                       # 0 = no imagery, k + 1 = tile from source k
        for j in range(y0, y1 + 1):
            for i in range(x0, x1 + 1):
                im, k = self._tile(i, j)
                if im is not None:
                    sl = (slice((j - y0) * TILE_PX, (j - y0 + 1) * TILE_PX), slice((i - x0) * TILE_PX, (i - x0 + 1) * TILE_PX))
                    mos[sl], src[sl] = im, k + 1
        mx = cv2.resize(((tx - x0) * TILE_PX).astype(np.float32), (px, px), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(((ty - y0) * TILE_PX).astype(np.float32), (px, px), interpolation=cv2.INTER_LINEAR)
        out = cv2.remap(mos, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        used = cv2.remap(src, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        missing = float((used == 0).mean())
        out[used == 0] = 0                                            # no imagery: black
        names = [self.sources[k - 1] for k in np.unique(used) if k > 0]
        return out[:, :, ::-1], missing, names          # tiles are BGR in cv2; return RGB


def _osm_image(maps, rec):
    zname, name = rec["image"].split("::")
    with zipfile.ZipFile(Path(maps) / zname) as zf:
        return cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]


def pilot(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = {r: [] for r in a.regions}
    for f in sorted(glob.glob(f"{a.maps}/index*.jsonl")):
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                if r["region"] in recs:
                    recs[r["region"]].append(r)
    for reg, rs in recs.items():
        sr = SatRenderer(a.cache, sources_of(reg))
        for k, r in enumerate(random.Random(a.seed).sample(rs, min(a.n, len(rs)))):
            sat, miss, _ = sr.render(r["lat"], r["lon"], r["heading_deg"])
            osm = _osm_image(a.maps, r)
            ov = (0.5 * osm + 0.5 * sat).astype(np.uint8)
            cv2.imwrite(str(out / f"{reg.replace(' ', '_')}_{k}.jpg"), np.hstack([osm, sat, ov])[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(reg, k, f"missing={miss:.2f}", r["image"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--maps", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    pilot(a)


if __name__ == "__main__":
    main()
