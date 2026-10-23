#!/usr/bin/env python3
"""
import_tiles.py - bring tile caches that someone else fetched (same layout as fetch_tiles.py: <root>/<id>/z<Z>_d6/<px>_<py>.zip with
members <x>_<y>.jpg, 256 px, web mercator) into our tile cache as z18 caches under the same source id. z18 caches are copied as they
are; a coarser one (e.g. z17) is upsampled 2x per level (bicubic) and cut into z18 tiles. Zero-byte members (= no tile) are kept.

  python3 import_tiles.py --src /iopsstor/scratch/cscs/qzou/natix_aerial/tiles --cache <our tiles dir> --ids hr_dgu2024 ro_mapn2020
"""
import argparse
import glob
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from sources import ZOOM
from tiles import PARENT_DZ, parent_zip, read_parent, write_parent


def import_one(src_root, cache, sid):
    n_files = n_tiles = 0
    for d in sorted(glob.glob(f"{src_root}/{sid}/z*_d{PARENT_DZ}")):
        z = int(os.path.basename(d).split("_")[0][1:])
        for zp in sorted(glob.glob(d + "/*.zip")):
            px, py = (int(v) for v in Path(zp).stem.split("_"))
            if z == ZOOM:
                dst = parent_zip(cache, sid, ZOOM, px, py)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".part")
                shutil.copyfile(zp, tmp)
                tmp.replace(dst)
                n_files += 1
                n_tiles += len(read_parent(dst))
                continue
            k = ZOOM - z                                                        # levels to go up: each tile becomes 4**k tiles
            out = defaultdict(dict)
            for (x, y), b in read_parent(zp).items():
                if b is None:
                    for i in range(2 ** k):
                        for j in range(2 ** k):
                            X, Y = x * 2 ** k + i, y * 2 ** k + j
                            out[(X >> PARENT_DZ, Y >> PARENT_DZ)][(X, Y)] = None
                    continue
                im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                big = cv2.resize(im, (256 * 2 ** k, 256 * 2 ** k), interpolation=cv2.INTER_CUBIC)
                for i in range(2 ** k):
                    for j in range(2 ** k):
                        X, Y = x * 2 ** k + i, y * 2 ** k + j
                        out[(X >> PARENT_DZ, Y >> PARENT_DZ)][(X, Y)] = cv2.imencode(
                            ".jpg", big[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            for (qx, qy), tiles in out.items():
                dst = parent_zip(cache, sid, ZOOM, qx, qy)
                write_parent(dst, {**read_parent(dst), **tiles})                # a z18 parent can get tiles from several coarse zips
                n_files += 1
            n_tiles += sum(len(t) for t in out.values())
    return n_files, n_tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="their tile root (one dir per source id)")
    ap.add_argument("--cache", required=True, help="our tile cache")
    ap.add_argument("--ids", nargs="+", required=True)
    a = ap.parse_args()
    for sid in a.ids:
        f, t = import_one(a.src, a.cache, sid)
        print(f"{sid}: {t} z{ZOOM} tiles, {f} zip writes", flush=True)


if __name__ == "__main__":
    main()
