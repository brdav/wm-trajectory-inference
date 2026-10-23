"""Web-mercator tile maths and the on-disk tile cache shared by fetch_tiles.py and render_sat.py.

Cache layout: <root>/<source>/z<Z>_d<PARENT_DZ>/<px>_<py>.zip, one stored zip per z(Z-PARENT_DZ) parent tile holding the child tiles
<x>_<y>.jpg (PARENT_DZ = 6: up to 64 x 64 tiles per zip, ~4,000 files for the whole dataset: the file count is what counts against
the inode quota). A zero-byte entry means the source has no tile there (so it is not asked again). A parent zip is written once, atomically.
"""
import math
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

WORLD_M = 40_075_016.6856      # web-mercator world width [m]
TILE_PX = 256
PARENT_DZ = 6                  # a zip holds up to 64 x 64 child tiles


def tile_xy(lon, lat, z):
    """Fractional tile coordinates (x east, y south) of lon/lat arrays at zoom z."""
    n = 2 ** z
    return (lon + 180.0) / 360.0 * n, (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n


def window_tile_ids(lat, lon, z, half_m):
    """Unique ids (y * 2^z + x) of the tiles touched by the axis-aligned window of half side half_m around each lat/lon."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * np.cos(np.radians(lat)))
    x0, y1 = tile_xy(lon - dlon, lat - dlat, z)         # y grows southwards
    x1, y0 = tile_xy(lon + dlon, lat + dlat, z)
    x0, x1, y0, y1 = (np.floor(v).astype(np.int64) for v in (x0, x1, y0, y1))
    n, ids = 2 ** z, []
    for dx in range(int((x1 - x0).max()) + 1):
        for dy in range(int((y1 - y0).max()) + 1):
            m = (x0 + dx <= x1) & (y0 + dy <= y1)
            ids.append((y0[m] + dy) * n + (x0[m] + dx))
    return np.unique(np.concatenate(ids))


def tile_centre_lonlat(x, y, z):
    """lon, lat of the centre of tiles (x, y) at zoom z."""
    n = 2 ** z
    lon = (np.asarray(x) + 0.5) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * (np.asarray(y) + 0.5) / n))))
    return lon, lat


def parent_of(x, y):
    return x >> PARENT_DZ, y >> PARENT_DZ


def parent_zip(root, source, z, px, py):
    return Path(root) / source / f"z{z}_d{PARENT_DZ}" / f"{px}_{py}.zip"


def write_parent(path, tiles):
    """tiles: {(x, y): jpeg bytes or None (= no tile)} -> stored zip, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for (x, y), b in tiles.items():
            zf.writestr(f"{x}_{y}.jpg", b or b"")
    os.replace(tmp, path)


def read_parent(path):
    """{(x, y): jpeg bytes or None} of an existing parent zip ({} if there is none)."""
    if not Path(path).exists():
        return {}
    with zipfile.ZipFile(path) as zf:
        return {tuple(int(v) for v in n[:-4].split("_")): (zf.read(n) or None) for n in zf.namelist()}


def parent_tiles(path):
    """Set of (x, y) already in a parent zip (names only, nothing is read)."""
    if not Path(path).exists():
        return set()
    with zipfile.ZipFile(path) as zf:
        return {tuple(int(v) for v in n[:-4].split("_")) for n in zf.namelist()}


class TileCache:
    """Reader of one source's cache."""

    def __init__(self, root, source, z):
        self.root, self.source, self.z = root, source, z

    @lru_cache(maxsize=64)
    def _zip(self, px, py):
        p = parent_zip(self.root, self.source, self.z, px, py)
        return zipfile.ZipFile(p) if p.exists() else None

    def get(self, x, y):
        """jpeg bytes, or None if the tile is missing / was never fetched."""
        zf = self._zip(*parent_of(x, y))
        if zf is None:
            return None
        try:
            return zf.read(f"{x}_{y}.jpg") or None
        except KeyError:
            return None
