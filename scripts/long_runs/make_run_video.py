#!/usr/bin/env python3
"""Renders the runs of find_long_runs.py as one continuous video each, to inspect the seams and the ego-trajectory.

Per run the front videos of its chunks are played back to back (nothing is inserted at a seam, so it looks as it would in a
long-form clip) next to a bird's-eye view of the fused ego-trajectory:
  top right     the whole run, north up: driven part highlighted, tick at every chunk start, dot = now
  bottom right  heading-up window around the car: past path grey, coming path blue (+-~150 m)
and on the video the run position, chunk number, speed and, for 1.5 s after each chunk start, the gap to the previous chunk.

The poses are result.npz's fused_poses of every chunk (UTM metres: x = east, z = north, camera z = driving direction), read
from the session zip (or the loose session dir) under --traj_root. Between the last keyframe of a chunk and its end the
position is held, so the dot freezes for a moment before the next chunk (the keyframes stop ~0.5 s before the end).

Usage (needs numpy, pandas, av, opencv-python-headless):
    python3 scripts/long_runs/make_run_video.py --chunks_csv run_chunks.csv --out_dir videos [--runs 1580 4960] [--n 3]
"""
import argparse
import io
import os
import zipfile

import av
import cv2
import numpy as np
import pandas as pd

TRAJ = {"natix": "/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories",
        "natix-jp-uk": "/capstor/store/cscs/swissai/ab036/processed_datasets/natix-jp-uk/trajectories"}
VIDEO = {"natix": "/capstor/store/cscs/swissai/ab036/datasets/natix",
         "natix-jp-uk": "/capstor/store/cscs/swissai/ab036/shared/datasets/natix-jp-uk/6adb62fa"}
VW, VH, RW, PH = 960, 540, 480, 270          # video, right column width, height of each of the two BEV panels
GREY, BLUE, ORANGE, WHITE = (150, 150, 150), (255, 150, 40), (40, 140, 255), (255, 255, 255)   # BGR
AHEAD_M, SPAN_M = 150.0, 300.0                  # local panel: metres shown up the panel / lateral window


def load_chunk_poses(dataset, session, source_video_path):
    """(t_s, east, north, fwd_east, fwd_north) of the fused poses of one chunk; t_s is video time."""
    stem = os.path.basename(source_video_path)[:-4].replace("FRONT_", "")           # <date>_<HH-MM-SS>
    inner = f"{stem.split('_')[1]}/FRONT_FOLDER/FRONT_{stem}/result.npz"
    zip_path = f"{TRAJ[dataset]}/{session}.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            data = np.load(io.BytesIO(z.read(inner)))
    else:
        data = np.load(f"{TRAJ[dataset]}/{session}/{inner}")
    f, t = data["fused_poses"], (data["frame_ids"] - 1.0) / float(data["extraction_fps"])
    return t, f[:, 0, 3], f[:, 2, 3], f[:, 0, 2], f[:, 2, 2]


class Route:
    """All chunks of a run: the poses on one time axis (run seconds), the chunk starts, interpolation at a time."""

    def __init__(self, chunks):
        t0, parts = 0.0, []
        self.starts = []
        for _, c in chunks.iterrows():
            t, e, n, fe, fn = load_chunk_poses(c.dataset, c.session, c.source_video_path)
            self.starts.append(t0)
            parts.append(np.stack([t + t0, e, n, fe, fn], 1))
            t0 += c.dur
        a = np.concatenate(parts)
        _, keep = np.unique(a[:, 0], return_index=True)
        self.t, self.e, self.n, self.fe, self.fn = a[keep].T
        self.total = t0
        self.xy = np.stack([self.e, self.n], 1)

    def at(self, t):
        e, n = np.interp(t, self.t, self.e), np.interp(t, self.t, self.n)
        h = np.array([np.interp(t, self.t, self.fe), np.interp(t, self.t, self.fn)])
        return np.array([e, n]), h / (np.linalg.norm(h) + 1e-9)

    def speed_kmh(self, t):
        p0, _ = self.at(t - 0.5)
        p1, _ = self.at(t + 0.5)
        return 3.6 * float(np.linalg.norm(p1 - p0))


def put(img, text, org, scale=0.6, color=WHITE):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    box = img[max(y - h - 4, 0):y + base + 2, max(x - 4, 0):x + w + 4]
    box[:] = box // 3                                   # darken the text's background for legibility
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class Overview:
    """North-up view of the whole route; the base image is drawn once, the driven part and the dot per frame."""

    def __init__(self, route, starts_idx):
        lo, hi = route.xy.min(0), route.xy.max(0)
        s = min((RW - 40) / max(hi[0] - lo[0], 1.0), (PH - 40) / max(hi[1] - lo[1], 1.0))
        self.px = np.stack([20 + (route.e - lo[0]) * s + (RW - 40 - (hi[0] - lo[0]) * s) / 2,
                            PH - 20 - (route.n - lo[1]) * s - (PH - 40 - (hi[1] - lo[1]) * s) / 2], 1)
        self.route, self.s, self.lo = route, s, lo
        self.base = np.full((PH, RW, 3), 30, np.uint8)
        cv2.polylines(self.base, [self.px.astype(np.int32)], False, GREY, 2, cv2.LINE_AA)
        for i in starts_idx:
            cv2.circle(self.base, tuple(self.px[i].astype(int)), 4, WHITE, 1, cv2.LINE_AA)
        put(self.base, "N", (RW - 22, 22), 0.5)
        self.scale_m = 10 ** np.floor(np.log10(120 / s))   # a round length that is about 60-120 px
        x1 = 20 + int(self.scale_m * s)
        cv2.line(self.base, (20, PH - 8), (x1, PH - 8), WHITE, 2)
        put(self.base, f"{self.scale_m:g} m" if self.scale_m < 1000 else f"{self.scale_m / 1000:g} km", (x1 + 6, PH - 3), 0.4)

    def draw(self, t):
        img = self.base.copy()
        k = int(np.searchsorted(self.route.t, t))
        if k > 1:
            cv2.polylines(img, [self.px[:k].astype(np.int32)], False, ORANGE, 2, cv2.LINE_AA)
        pos = np.array([np.interp(t, self.route.t, self.px[:, 0]), np.interp(t, self.route.t, self.px[:, 1])])
        cv2.circle(img, tuple(pos.astype(int)), 6, (0, 0, 255), -1, cv2.LINE_AA)
        return img


def local_view(route, t):
    """Heading-up window: the car at the bottom centre, the route within +-SPAN_M/2 around it."""
    img = np.full((PH, RW, 3), 30, np.uint8)
    p, h = route.at(t)
    right = np.array([h[1], -h[0]])
    s = (PH - 50) / AHEAD_M
    d = route.xy - p
    fwd, lat = d @ h, d @ right
    px = np.stack([RW / 2 + lat * s, PH - 30 - fwd * s], 1)
    near = np.hypot(fwd, lat) < 2 * AHEAD_M
    past = near & (route.t <= t)
    for sel, col in ((past, GREY), (near & ~past, BLUE)):
        idx = np.flatnonzero(sel)
        if len(idx) > 1:
            # break the polyline where indices jump (the far parts of a looping route)
            for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if len(seg) > 1:
                    cv2.polylines(img, [px[seg].astype(np.int32)], False, col, 3, cv2.LINE_AA)
    c = (int(RW / 2), PH - 30)
    cv2.fillPoly(img, [np.array([[c[0], c[1] - 14], [c[0] - 8, c[1] + 8], [c[0] + 8, c[1] + 8]])], (0, 0, 255), cv2.LINE_AA)
    for m in (50, 100, 150):                       # range rings
        cv2.ellipse(img, c, (int(m * s), int(m * s)), 0, 180, 360, (70, 70, 70), 1, cv2.LINE_AA)
        put(img, f"{m} m", (c[0] + 4, c[1] - int(m * s) + 14), 0.35)
    put(img, "heading up", (8, 18), 0.45)
    return img


def render(chunks, out_path, step):
    route = Route(chunks)
    starts_idx = [int(np.searchsorted(route.t, s)) for s in route.starts]
    starts_idx = [min(i, len(route.t) - 1) for i in starts_idx]
    over = Overview(route, starts_idx)
    gaps = chunks.gap.fillna(0.0).to_numpy()
    out = av.open(out_path, "w")
    fps = None
    t_run = 0.0
    for ci, (_, c) in enumerate(chunks.iterrows()):
        with av.open(f"{VIDEO[c.dataset]}/{c.source_video_path}") as src:
            s = src.streams.video[0]
            s.thread_type = "AUTO"
            if fps is None:
                fps = float(s.average_rate) / step
                v = out.add_stream("libx264", rate=int(round(fps)))
                v.width, v.height, v.pix_fmt = VW + RW, VH, "yuv420p"
                v.options = {"crf": "26", "preset": "veryfast"}
            first = None
            for i, fr in enumerate(src.decode(s)):
                if i % step:
                    continue
                first = fr.pts if first is None else first
                tc = float(fr.pts - first) * float(s.time_base)
                t = route.starts[ci] + tc
                img = np.zeros((VH, VW + RW, 3), np.uint8)
                img[:, :VW] = fr.reformat(width=VW, height=VH, format="bgr24").to_ndarray()
                img[:PH, VW:] = over.draw(t)
                img[PH:PH * 2, VW:] = local_view(route, t)
                m, sec = divmod(int(t), 60)
                put(img, f"{m:02d}:{sec:02d} / {int(route.total) // 60:02d}:{int(route.total) % 60:02d}   chunk {ci + 1}/{len(chunks)}"
                         f"   {route.speed_kmh(t):3.0f} km/h", (12, 28), 0.7)
                if ci and tc < 1.5:
                    put(img, f"chunk start: gap to previous {gaps[ci]:+.2f} s", (12, 60), 0.7, (0, 255, 255))
                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                for pkt in v.encode(frame):
                    out.mux(pkt)
    for pkt in v.encode():
        out.mux(pkt)
    out.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks_csv", required=True, help="--out_chunks_csv of find_long_runs.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--runs", nargs="+", type=int, default=None, help="run ids to render (default: the first --n)")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--step", type=int, default=2, help="use every N-th video frame (the source is ~36 fps)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.chunks_csv, parse_dates=["start"]).sort_values(["run", "start"])
    runs = args.runs or list(dict.fromkeys(df.run))[:args.n]
    for r in runs:
        ch = df[df.run == r].reset_index(drop=True)
        name = f"run{r}_{ch.country[0].replace(' ', '')}_{len(ch)}chunks.mp4"
        print("rendering", name, flush=True)
        render(ch, os.path.join(args.out_dir, name), args.step)


if __name__ == "__main__":
    main()
