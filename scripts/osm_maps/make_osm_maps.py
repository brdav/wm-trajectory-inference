#!/usr/bin/env python3
"""Write ego-centred OSM map images with the ego's next-5-seconds path, for clips with a fused trajectory.

For every clip with `pose_source == "fused"` in trajectories/manifest.jsonl (GPS/road-fused, QC-passed) and for every
row of the sample manifest of that clip (processed_datasets/natix/manifest/samples_24fps.parquet; one row = one 5 s
training sample), let t0 be the row's start frame (`start_s`, on the video's clock). The map (scripts/osm_maps/osm_bev.py)
is rendered centred on the car at t0, rotated heading-up, with the car's path over [t0, t0 + horizon] drawn on top.
The path comes from one of three sources (--path):

  predicted   : the SLAM + GPS-fused poses (`fused_poses`) interpolated over video time: the estimated driven path,
                including lateral position, but with the fusion's placement error (~1-2 m) on top. Default.
  map_matched : the trajectory snapped onto the map-matched OSM route (`snapped`, `matched_route_wkt` in result.npz):
                points ON the road centre line, smooth, no lane-level information. Built by interpolating the arclength
                along the route over video time and sampling the route geometry.
  none        : the map only (ego marker at full size).

Both path options use the same drawing (violet, one car wide, white halo, a dot every second, arrowhead), so a model
sees one visual vocabulary. The map is rotated to the fused motion direction for both and centred on the path's own
start, so the ego marker is always at the image centre.

Consistency gate (--max-snippet-error-m / --max-snippet-max-error-m): over exactly the 5 s window being drawn, the
predicted and the map-matched positions are compared at the same times. If the mean (or max) distance is above the
threshold, the fused trajectory and the map disagree there and the sample is labelled unplottable on OSM: no image is
written, and the moment is recorded in unplottable.jsonl with its errors. A clip whose route cannot be used for the
check (e.g. a reversal) counts as unplottable too.

Output: ONE ZIP PER SESSION, --out/<Country>/[<State>/]<date>/<session>.zip (the layout of the trajectories' packed
sessions: ~45k files for the whole dataset instead of ~1.3M). A zip is stored (images are not recompressed) and holds
<HH-MM-SS>_s<sample_index>.<png|webp> plus index.jsonl, unplottable.jsonl and text.jsonl for that session; optional --save-mask
(path layer, *_path.png) and --save-map-only (view without the path, *_map.png). The image of a sample row is therefore
entry <time>_s<sample_index>.<ext> of <session>.zip (both from source_video_path / sample_index).
At the top of --out: index*.jsonl (one line per image: sample_id, start_frame, t0, lat/lon, heading, region, path
length, snippet errors, ...), unplottable*.jsonl and done_sessions*.txt. Re-running skips sessions listed in
done_sessions (--overwrite redoes them); for Slurm arrays use --shard i/N (every shard writes its own
*.shard<i>of<N> files).

Route text (scripts/osm_maps/osm_route_text.py; --no-text skips it): for every image also the location (country, nearest
city / town / village from extract_places.py) and what a forward camera would see along the 5 s path (the road's class /
one-way / lanes / speed limit, signs, lights, crossings and bumps facing the direction of travel, the manoeuvre). The structured record is entry text.jsonl of the session zip (one line per image, keyed by `name` =
the image entry), the sentence made from it (`caption`) is also in the index. Lane counts / limits taken from a neighbouring
road carry source "inferred".

Map data must be prepared for the area (osm_bev.py prepare ...; _regions.json from osm_bev.py regions). Samples whose
position has no prepared data are counted and skipped.

Usage:
  python scripts/osm_maps/make_osm_maps.py --osm-data <prepared root> --out <dir> --path predicted \\
      --countries Switzerland --limit-clips 50 --workers 16
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import multiprocessing as mp
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from scipy.interpolate import PchipInterpolator
from shapely import wkt

REPO = Path(__file__).resolve().parents[2]  # scripts/osm_maps/ -> repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "osm_maps"))  # osm_bev

import osm_bev as ob  # noqa: E402
import osm_route_text as rt  # noqa: E402
from vggt_slam.gps import load_gps_fixes  # noqa: E402
from vggt_slam.paths import DATASET_ROOT, SAMPLE_MANIFEST as SAMPLES_PARQUET, TRAJ_ROOT  # noqa: E402

TICKS_PER_S = 10  # path samples per second; one marker dot per second


# --------------------------------------------------------------------------------------
# Locating and loading a clip
# --------------------------------------------------------------------------------------
def clip_files(clip_id: str, dataset_root: str):
    """-> (session_rel, time, zip/dir-relative inner path of result.npz, gps csv path)."""
    rel = Path(clip_id).relative_to(dataset_root).parts  # [Country, (State,) date, session, time, FRONT_FOLDER, f.mp4]
    session_rel = "/".join(rel[:-3])
    time_dir, stem = rel[-3], Path(rel[-1]).stem
    return session_rel, time_dir, f"{time_dir}/FRONT_FOLDER/{stem}/result.npz", str(Path(clip_id).with_suffix(".csv"))


def load_result(traj_root: str, session_rel: str, inner: str):
    """result.npz of a clip from a loose session directory or its packed .zip; None if neither exists."""
    loose = Path(traj_root) / session_rel / inner
    if loose.exists():
        return np.load(loose)
    z = Path(traj_root) / (session_rel + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            try:
                return np.load(io.BytesIO(zf.read(inner)))
            except KeyError:
                return None
    return None


# --------------------------------------------------------------------------------------
# The path options
# --------------------------------------------------------------------------------------
def _bearing(a, b, min_dist):
    """Compass bearing a -> b in degrees [0, 360) (0 = north, E/N coordinates); None if they are closer than min_dist."""
    d = np.asarray(b) - np.asarray(a)
    return float(np.degrees(np.arctan2(d[0], d[1])) % 360.0) if np.hypot(*d) > min_dist else None


class ClipPath:
    """Interpolated trajectory of one fused clip, in UTM metres (E, N), as functions of video time."""

    def __init__(self, npz, utm_crs):
        fid, fps = np.asarray(npz["frame_ids"], float), float(npz["extraction_fps"])
        t = (fid - 1.0) / fps                                    # keyframes are irregular in time (0.1-1.1 s apart)
        P = npz["fused_poses"]
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        self.t, self.t0_clip, self.t1_clip = t[keep], t[keep][0], t[keep][-1]
        self.fE = PchipInterpolator(self.t, P[keep, 0, 3])
        self.fN = PchipInterpolator(self.t, P[keep, 2, 3])
        self.route = wkt.loads(str(npz["matched_route_wkt"])) if "matched_route_wkt" in npz else None
        self.route_ok = False
        if self.route is not None and "snapped" in npz and not self.route.is_empty:
            s = np.array([self.route.project(shapely.Point(x, y)) for x, y in np.asarray(npz["snapped"])[keep]])
            mono = np.maximum.accumulate(s)
            self.route_ok = float((mono - s).max()) < 5.0       # a U-turn / reversal breaks "arclength grows with time"
            self.fS = PchipInterpolator(self.t, mono)
        self.to_ll = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    def predicted(self, tau):
        return np.column_stack([self.fE(tau), self.fN(tau)])

    def map_matched(self, tau):
        s = np.clip(self.fS(tau), 0.0, self.route.length)
        return np.array([self.route.interpolate(v).coords[0] for v in s])

    def route_heading(self, t0, half_window_m=4.0):
        s = float(np.clip(self.fS(t0), 0.0, self.route.length))
        a = self.route.interpolate(max(0.0, s - half_window_m)).coords[0]
        b = self.route.interpolate(min(self.route.length, s + half_window_m)).coords[0]
        return _bearing(a, b, 0.5)

    def map_matched_heading(self, t0):
        """Direction of travel of the map-matched path itself over t0 +- 0.5 s. (The route tangent at s(t0) is wrong at
        corners: a +-4 m window straddling a sharp vertex points between the old and the new road direction.)"""
        ta, tb = max(self.t0_clip, t0 - 0.5), min(self.t1_clip, t0 + 0.5)
        a, b = self.map_matched(np.array([ta, tb]))
        return _bearing(a, b, 1.0)

    def predicted_heading(self, t0):
        a, b = self.predicted(np.array([max(self.t0_clip, t0 - 0.5), min(self.t1_clip, t0 + 0.5)]))
        return _bearing(a, b, 1.0)


def _turn_deg(path):
    """Signed change of direction over the path (last second vs first second), degrees; + = right turn."""
    if len(path) < 2 * TICKS_PER_S + 1:
        return None
    b0, b1 = _bearing(path[0], path[TICKS_PER_S], 0.5), _bearing(path[-TICKS_PER_S - 1], path[-1], 0.5)
    return None if b0 is None or b1 is None else round(float((b1 - b0 + 180.0) % 360.0 - 180.0), 1)


# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------
_R = {}


def _init(osm_data, res, places, roundabouts):
    _R["mr"] = ob.MapRenderer(osm_data, res=res)
    _R["places"] = rt.PlaceIndex(places) if places else None
    _R["roundabouts"] = rt.load_roundabout_ids(roundabouts) if roundabouts else None


def _has_map_data(mr, lat, lon):
    """Is there any prepared map feature in the crop around (lat, lon)? Same lookup as MapRenderer.render."""
    if mr.regions is None:
        countries = mr.countries
    else:
        near = mr.regions.near(lat, lon, radius_m=mr.half * math.sqrt(2) + 25.0)
        countries = [ob.region_dir(k) for k in near if ob.region_dir(k) in mr.countries]
    rows, _ = mr._query(lat, lon, countries)
    return len(rows) > 0


def process_clip(row, samples, a, files, texts):
    """Renders every sample of one clip. -> (index records, skip Counter, unplottable records); encoded images are
    appended to `files` as (zip entry name, bytes), the route descriptions to `texts` (osm_route_text.describe_route)."""
    mr = _R["mr"]
    skipped, records, unplottable = collections.Counter(), [], []
    session_rel, time_dir, inner, csv_path = clip_files(row["clip_id"], a["dataset_root"])
    npz = load_result(a["traj_root"], session_rel, inner)
    if npz is None or "fused_poses" not in npz:
        skipped["no_result"] += 1
        return records, skipped, unplottable
    try:
        cp = ClipPath(npz, load_gps_fixes(csv_path)["utm_crs"])
    except Exception:                                            # no GPS csv / no real fixes / degenerate route
        skipped["no_gps_or_bad_result"] += 1
        return records, skipped, unplottable
    option = a["path"]
    if option == "map_matched" and not cp.route_ok:
        skipped["no_usable_route"] += 1
        return records, skipped, unplottable

    n_made = 0
    for smp in samples:                                          # the clip's rows of the sample manifest
        if n_made >= a["max_per_clip"]:
            break
        t0 = float(smp["start_s"])                               # start frame of the sample, on the video's own clock
        if t0 < cp.t0_clip - 1e-6 or t0 + a["horizon"] > cp.t1_clip + 1e-6:
            skipped["outside_pose_range"] += 1                   # poses do not cover this sample's full 5 s
            continue
        tau = t0 + np.linspace(0.0, a["horizon"], int(round(a["horizon"] * TICKS_PER_S)) + 1)
        err_mean = err_max = None
        lon_s, lat_s = cp.to_ll.transform(*cp.predicted(tau[:1])[0])
        if not _has_map_data(mr, lat_s, lon_s):               # before the gate: not-prepared is not a bad fit
            skipped["no_map_data"] += 1
            continue
        if option != "none" and a["gate_mean"] > 0:
            if not cp.route_ok:
                skipped["unplottable_no_route"] += 1
                unplottable.append(dict(clip_id=row["clip_id"], sample_id=smp["sample_id"], t0_s=round(t0, 2),
                                        reason="no_usable_route"))
                continue
            d = np.hypot(*(cp.predicted(tau) - cp.map_matched(tau)).T)
            err_mean, err_max = float(d.mean()), float(d.max())
            if err_mean > a["gate_mean"] or err_max > a["gate_max"]:
                skipped["unplottable_snippet_error"] += 1
                unplottable.append(dict(clip_id=row["clip_id"], sample_id=smp["sample_id"], t0_s=round(t0, 2),
                                        reason="snippet_error", err_mean_m=round(err_mean, 2),
                                        err_max_m=round(err_max, 2)))
                continue
        if option == "map_matched":
            path = cp.map_matched(tau)
            # ONE heading source for both options (the fused motion = where the camera actually points), so the two
            # maps are identical and only the path differs; the map-matched path / route tangent are fallbacks
            heading = cp.predicted_heading(t0) or cp.map_matched_heading(t0) or cp.route_heading(t0)
        else:                                                    # predicted, or none (only the start position is used)
            path = cp.predicted(tau if option == "predicted" else tau[:1])
            heading = cp.predicted_heading(t0)                   # None when (almost) standing still ...
            if heading is None and cp.route_ok:
                heading = cp.route_heading(t0)                   # ... then the road direction is the best guess
        if heading is None:
            skipped["no_heading"] += 1
            continue
        name = f"{time_dir}_s{int(smp['sample_index']):04d}"     # entry names inside the session zip
        lon0, lat0 = cp.to_ll.transform(*path[0])
        length = float(np.hypot(*np.diff(path, axis=0).T).sum())
        traj = None
        ll = np.column_stack(cp.to_ll.transform(path[:, 0], path[:, 1]))
        if option != "none" and length >= 2.0:                   # a standing car has no direction to draw
            traj = dict(lonlat=ll, ticks=[k for k in range(TICKS_PER_S, len(path), TICKS_PER_S)])
        ego_scale = 1.0 if option == "none" else 0.5
        o = mr.render(lat0, lon0, heading, None, ego=True, trajectory=traj, ego_scale=ego_scale)
        if o["meta"]["n_features"] == 0:
            skipped["no_map_data"] += 1
            continue
        img = np.ascontiguousarray(o["image"][:, :, ::-1])
        files.append((name + a["ext"], _encode(img, a)))
        if (a["save_mask"] or a["save_map_only"]) and traj is not None:
            base = np.ascontiguousarray(
                mr.render(lat0, lon0, heading, None, ego=True, trajectory=None, ego_scale=ego_scale)["image"][:, :, ::-1])
            if a["save_map_only"]:
                files.append((name + "_map" + a["ext"], _encode(base, a)))
            if a["save_mask"]:
                files.append((name + "_path.png", cv2.imencode(".png", ((img != base).any(axis=2) * 255).astype(np.uint8),
                                                              [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes()))
        route = rt.describe_route(o["scene"], ll, tau - t0, _R["places"], _R["roundabouts"]) if a["text"] and option != "none" else None
        if route is not None:
            texts.append(dict(name=name + a["ext"], sample_id=smp["sample_id"], caption=rt.caption(route), **route))
        records.append(dict(
            image=f"{session_rel}.zip::{name}{a['ext']}", clip_id=row["clip_id"], sample_id=smp["sample_id"],
            sample_index=int(smp["sample_index"]), start_frame=int(smp["start_frame"]), t0_s=round(t0, 2), path=option,
            lat=lat0, lon=lon0, heading_deg=round(heading, 2), region=o["meta"]["country"],
            drive_side=o["meta"]["drive_side"], path_length_m=round(length, 1),
            mean_speed_kmh=round(length / a["horizon"] * 3.6, 1) if option != "none" else None,
            path_drawn=traj is not None, n_features=o["meta"]["n_features"],
            path_turn_deg=_turn_deg(path),
            map_match_error_mean_m=(row.get("gps_fusion") or {}).get("map_match_error_mean_m"),
            snippet_err_mean_m=None if err_mean is None else round(err_mean, 2),
            snippet_err_max_m=None if err_max is None else round(err_max, 2),
            caption=None if route is None else texts[-1]["caption"]))
        n_made += 1
    return records, skipped, unplottable


def _encode(img_bgr, a):
    """Lossless encoding of one map image (bytes). webp is ~2x smaller than png for these flat-colour maps."""
    if a["ext"] == ".webp":
        return cv2.imencode(".webp", img_bgr, [cv2.IMWRITE_WEBP_QUALITY, 101])[1].tobytes()     # 101 = lossless
    return cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, a["png_level"]])[1].tobytes()


def process_session(job):
    """One whole session (all its fused clips) -> one zip <out>/<session>.zip (stored, no recompression) holding the images,
    index.jsonl and unplottable.jsonl. Written to a .part file and renamed, so a killed job never leaves a half zip.
    Returns (session_rel, n images, skip Counter, index records, unplottable records)."""
    a, session_rel = job["args"], job["session"]
    files, texts, records, unplottable, skipped = [], [], [], [], collections.Counter()
    for row, samples in job["clips"]:
        r, sk, u = process_clip(row, samples, a, files, texts)
        records += r
        unplottable += u
        skipped.update(sk)
    if files:
        zp = Path(a["out"]) / (session_rel + ".zip")
        zp.parent.mkdir(parents=True, exist_ok=True)
        part = zp.with_name(zp.name + ".part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as zf:
            for name, data in files:
                zf.writestr(name, data)
            zf.writestr("index.jsonl", "".join(json.dumps(x) + "\n" for x in records))
            zf.writestr("unplottable.jsonl", "".join(json.dumps(x) + "\n" for x in unplottable))
            if a["text"]:
                zf.writestr("text.jsonl", "".join(json.dumps(x) + "\n" for x in texts))
        part.replace(zp)
    return session_rel, len(records), skipped, records, unplottable


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm-data", required=True, help="prepared map root (osm_bev.py prepare; needs _regions.json)")
    ap.add_argument("--out", required=True, help="output root (one zip per session plus index/unplottable/done files)")
    ap.add_argument("--path", choices=["map_matched", "predicted", "none"], default="predicted")
    ap.add_argument("--manifest", default=f"{TRAJ_ROOT}/manifest.jsonl", help="trajectory manifest (pose_source per clip)")
    ap.add_argument("--samples", default=SAMPLES_PARQUET,
                    help="sample manifest: one row per 5 s training sample; the map is drawn at each row's start frame")
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--traj-root", default=TRAJ_ROOT)
    ap.add_argument("--countries", nargs="*", help="e.g. Switzerland Japan 'United States'")
    ap.add_argument("--limit-clips", type=int, help="process only the first N clips (in seeded hash order)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the stable clip ordering used by --limit-clips")
    ap.add_argument("--max-map-match-error-m", type=float,
                    help="skip clips whose mean map-match error (manifest) is larger; ~2.0 is sensible for --path predicted")
    ap.add_argument("--max-snippet-error-m", type=float, default=6.0,
                    help="mean predicted-vs-map-matched distance over the drawn 5 s above which a moment is unplottable "
                         "(<= 0 disables the gate)")
    ap.add_argument("--max-snippet-max-error-m", type=float, default=12.0, help="same, for the maximum distance")
    ap.add_argument("--horizon", type=float, default=5.0, help="seconds of future path")
    ap.add_argument("--max-per-clip", type=int, default=100, help="maximum number of images per clip")
    ap.add_argument("--res", type=float, default=0.5, help="map resolution in m/pixel")
    ap.add_argument("--workers", type=int, default=8, help="worker processes (each loads its own MapRenderer)")
    ap.add_argument("--places", help="places parquet of extract_places.py for the city in the route text "
                                     "(default: <osm-data>/_places.parquet if it exists)")
    ap.add_argument("--roundabouts", help="roundabout way ids of extract_places.py for the route text "
                                          "(default: <osm-data>/_roundabout_ways.parquet if it exists)")
    ap.add_argument("--no-text", action="store_true", help="skip the route descriptions (text.jsonl in every zip, caption in the index)")
    ap.add_argument("--save-mask", action="store_true", help="also write the path layer as *_path.png (binary mask)")
    ap.add_argument("--save-map-only", action="store_true", help="also write the view without the path as *_map.png")
    ap.add_argument("--format", choices=["png", "webp"], default="png", help="lossless image format inside the zips")
    ap.add_argument("--png-level", type=int, default=9, help="PNG compression level (0-9)")
    ap.add_argument("--shard", default="0/1", help="i/N: process only sessions with hash(session) %% N == i (Slurm array)")
    ap.add_argument("--overwrite", action="store_true", help="redo sessions that are already in done_sessions*.txt")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [r for r in rows if r.get("pose_source") == "fused"]
    if a.countries:
        keep = tuple(f"{a.dataset_root}/{c}/" for c in a.countries)
        rows = [r for r in rows if r["clip_id"].startswith(keep)]
    if a.max_map_match_error_m is not None:
        rows = [r for r in rows
                if ((r.get("gps_fusion") or {}).get("map_match_error_mean_m") or 1e9) <= a.max_map_match_error_m]
    # stable sample: order clips by a hash of (seed, clip_id), not a shuffle of the manifest, so the selection does not
    # change when other jobs add/modify manifest rows (the manifest is rewritten by the QC / metric-scale steps)
    rows.sort(key=lambda r: hashlib.sha1(f"{a.seed}:{r['clip_id']}".encode()).hexdigest())
    if a.limit_clips:
        rows = rows[:a.limit_clips]
    keep = {r["clip_id"].replace(a.dataset_root.rstrip("/") + "/", "", 1) for r in rows}
    t = pq.read_table(a.samples, columns=["sample_id", "source_video_path", "sample_index", "start_frame", "start_s"])
    by = collections.defaultdict(list)
    for sid, vp, si, sf, ss in zip(*(t[c].to_pylist() for c in ("sample_id", "source_video_path", "sample_index",
                                                               "start_frame", "start_s"))):
        if vp in keep:
            by[vp].append(dict(sample_id=sid, sample_index=si, start_frame=sf, start_s=ss))
    n_samples = sum(len(v) for v in by.values())
    print(f"{len(rows)} fused clips selected -> {n_samples} samples in {a.samples}; path={a.path}; horizon={a.horizon}s",
          flush=True)

    def lookup(flag, name):
        return flag or (str(Path(a.osm_data) / name) if (Path(a.osm_data) / name).exists() else None)
    places, roundabouts = lookup(a.places, "_places.parquet"), lookup(a.roundabouts, "_roundabout_ways.parquet")
    print(f"route text: places {places or 'none (country only)'}; roundabouts {roundabouts or 'none (not recognised)'}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    args = dict(path=a.path, out=a.out, dataset_root=a.dataset_root, traj_root=a.traj_root, horizon=a.horizon,
                ext=".webp" if a.format == "webp" else ".png", png_level=a.png_level,
                gate_mean=a.max_snippet_error_m, gate_max=a.max_snippet_max_error_m, max_per_clip=a.max_per_clip,
                save_mask=a.save_mask, save_map_only=a.save_map_only, text=not a.no_text)
    # session jobs: all selected clips of one session together (one zip per session)
    by_session = collections.defaultdict(list)
    for r in rows:
        key = r["clip_id"].replace(a.dataset_root.rstrip("/") + "/", "", 1)
        smp = sorted(by.get(key, []), key=lambda x: x["sample_index"])
        if smp:
            by_session[clip_files(r["clip_id"], a.dataset_root)[0]].append((r, smp))
    si, sn = (int(v) for v in a.shard.split("/"))
    tag = "" if sn == 1 else f".shard{si}of{sn}"
    sessions = sorted(k for k in by_session if int(hashlib.sha1(k.encode()).hexdigest(), 16) % sn == si)
    done_file = Path(a.out) / f"done_sessions{tag}.txt"
    done = set(done_file.read_text().split("\n")) if done_file.exists() and not a.overwrite else set()
    jobs = [dict(session=k, clips=by_session[k], args=args) for k in sessions if k not in done]
    print(f"shard {si}/{sn}: {len(sessions)} sessions, {len(done & set(sessions))} already done, {len(jobs)} to do", flush=True)

    skipped, n_img = collections.Counter(), 0
    with open(Path(a.out) / f"index{tag}.jsonl", "a") as index, open(Path(a.out) / f"unplottable{tag}.jsonl", "a") as unplot, \
            open(done_file, "a") as donef, mp.Pool(a.workers, initializer=_init, initargs=(a.osm_data, a.res, places, roundabouts)) as pool:
        for i, (sess, n, sk, recs, unp) in enumerate(pool.imap_unordered(process_session, jobs, chunksize=1), 1):
            skipped.update(sk)
            for u in unp:
                unplot.write(json.dumps(u) + "\n")
            for rec in recs:
                index.write(json.dumps(rec) + "\n")
            donef.write(sess + "\n")                         # after the zip is in place: safe to resume from this file
            for f in (index, unplot, donef):
                f.flush()
            n_img += n
            if i % 25 == 0 or i == len(jobs):
                print(f"  sessions {i}/{len(jobs)}  images {n_img}  skipped {dict(skipped)}", flush=True)
    print(f"done: {n_img} images in {a.out}; skip reasons: {dict(skipped)}")


if __name__ == "__main__":
    main()
