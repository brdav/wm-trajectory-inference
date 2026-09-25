"""Loading real GPS fixes from natix-style sensor logs (FRONT_*.csv, alongside
the source video) and projecting them to a local metric (UTM) frame.
"""
import csv
from datetime import datetime

import numpy as np
from pyproj import CRS, Transformer


def utm_crs_for(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def load_gps_fixes(csv_path):
    """Parse a natix FRONT_*.csv sensor log into real (non-'na') GPS fixes,
    projected to a local UTM CRS.

    Elapsed time is measured relative to the VIDEO's own start (native camera frame 1), derived
    from each row's own `frame_number` column (the native camera's frame index, NOT the
    extracted/keyframe index) and a native fps estimated from the log itself -- NOT from the
    CSV's own first row's timestamp. Those two are only the same reference point if sensor
    logging happened to start at (or very near) the video's first frame -- confirmed FALSE for
    at least one real clip, where the CSV's first row was already at frame_number=1045
    (~29s into the video at its ~36fps native rate): every GPS fix in that log was silently
    matched against the wrong keyframe by a ~29s constant offset (match_gps_to_frames matches by
    elapsed-time proximity, and (frame_id-1)/extraction_fps -- used for the keyframe side, see
    vggt_slam.gps_fusion -- IS anchored to the true video start). That corrupted the whole
    Umeyama fit (fit to keyframe/GPS pairs describing DIFFERENT real moments) while leaving each
    individual pairing line short/plausible-looking, which is why it isn't obvious from a
    residual-pairing plot alone. Confirmed fix on that clip: matched keyframes' own residual at
    the fit went from mean 19.5m/max 34.0m (wrong reference) to mean 3.3m/max 7.6m (frame_number-
    based), matching the healthy range seen on clips where the two references happened to agree.

    Native fps is estimated once via a linear fit of frame_number against wall-clock seconds
    across the WHOLE log (robust to the log's own start offset, and to logging at a lower/
    irregular rate than the native camera, since only the slope -- frames per wall-clock second
    -- is used, not any particular row's absolute value).

    Returns a dict with:
        elapsed_s: (N,) float, seconds since the video's own first frame
        easting, northing: (N,) float, local UTM meters
        accuracy_m: (N,) float, each fix's own reported horizontal_accuracy_m
        utm_crs: pyproj CRS used
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    t0 = datetime.fromisoformat(rows[0]["timestamp"])
    has_frame_number = "frame_number" in rows[0]

    if has_frame_number:
        wall_s = np.array([(datetime.fromisoformat(r["timestamp"]) - t0).total_seconds() for r in rows])
        frame_number = np.array([int(r["frame_number"]) for r in rows], dtype=float)
        if len(np.unique(wall_s)) < 2:
            # np.polyfit needs >= 2 distinct wall-clock timestamps to fit a line's slope --
            # with fewer, the least-squares design matrix is singular and numpy's LAPACK
            # backend fails with an opaque "SVD did not converge" instead of a clear precondition
            # error. Confirmed on 268 real clips, ALL with exactly 1 sensor-log row for the whole
            # ~60s clip (a natix logging gap, not a code bug) -- raising here gives that a clear,
            # categorizable status instead of an unrelated-looking numpy crash.
            raise ValueError(
                f"Only {len(rows)} sensor log row(s) ({len(np.unique(wall_s))} distinct "
                f"timestamp(s)) in {csv_path} -- need >= 2 distinct timestamps to estimate "
                "native fps."
            )
        # slope of frame_number vs. wall_s == native camera fps; intercept is discarded (we only
        # need frames-per-second, not this particular log's own start offset).
        native_fps, _ = np.polyfit(wall_s, frame_number, deg=1)

    elapsed, lats, lons, accs = [], [], [], []
    for row in rows:
        if row["GPS_latitude_deg"] == "na":
            continue
        if has_frame_number:
            t_s = (int(row["frame_number"]) - 1) / native_fps
        else:
            # No frame_number column (older/other log format) -- fall back to the CSV's own
            # first-row timestamp, which is only correct if logging happened to start at the
            # video's own first frame.
            t_s = (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()
        elapsed.append(t_s)
        lats.append(float(row["GPS_latitude_deg"]))
        lons.append(float(row["GPS_longitude_deg"]))
        accs.append(float(row["horizontal_accuracy_m"]) if row["horizontal_accuracy_m"] != "na" else 10.0)

    if not elapsed:
        raise ValueError(f"No real GPS fixes (all 'na') in {csv_path}")

    lats, lons = np.array(lats), np.array(lons)
    utm_crs = utm_crs_for(float(lons.mean()), float(lats.mean()))
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    easting, northing = to_utm.transform(lons, lats)

    return dict(
        elapsed_s=np.asarray(elapsed, dtype=float),
        easting=np.asarray(easting, dtype=float),
        northing=np.asarray(northing, dtype=float),
        accuracy_m=np.asarray(accs, dtype=float),
        utm_crs=utm_crs,
    )
