#!/usr/bin/env python3
"""Finds long continuous runs of 60 s chunks (several minutes of one drive) that are not in a training set and have a
usable ego-trajectory for every chunk.

A run is consecutive chunks of one session where
  - no chunk is in the training parquet (a training chunk ends the run on both sides),
  - every chunk has the wanted `pose_source` in trajectories/manifest.jsonl (default: fused) and its poses reach the end
    of the chunk (`qc.slam_valid_until_s` within --max_tail_s of the chunk's end),
  - the gap between neighbours is at most --max_gap seconds either way (negative = overlap).
The gap is next start - (previous start + previous duration). Chunk folders are named by their exact start second; the
duration is the real mp4 duration, read from the container (the manifest's resampled duration is ~0.3 s short, which would
shift every gap). Real gaps of one session are spread over about -0.4 .. +1.5 s, so a strict --max_gap leaves few runs
(5+ min: 9 at 0.5 s, ~100 at 0.75 s, ~520 at 1.0 s for the natix + natix-jp-uk data of 2026-11).

With --top_n N (default 10) only the N longest runs are kept, at most one per session (a session is one drive; two runs of it
are not independent samples). With --top_n 0 every run is written, longest first.

The mp4 durations are only read for chunks that already form a run of >= --min_minutes under a loose gap
(--max_gap + --prefilter_slack, by manifest duration), which keeps it to ~10k files.

Usage (on the cluster; needs pandas, pyarrow, av):
    python3 scripts/long_runs/find_long_runs.py --out_csv runs.csv --out_chunks_csv run_chunks.csv     # the 10 longest runs, gap <= 0.6 s
    python3 scripts/long_runs/find_long_runs.py --max_gap 1.0 --top_n 0 --pose_sources fused metric --min_minutes 10 ...  # all runs
"""
import argparse
import math
import re
from concurrent.futures import ThreadPoolExecutor

import av
import pandas as pd

PROCESSED = "/capstor/store/cscs/swissai/ab036/processed_datasets"
# dataset name -> (processed_datasets subdir, root of the source videos)
DATASETS = {
    "natix": ("natix", "/capstor/store/cscs/swissai/ab036/datasets/natix"),
    "natix-jp-uk": ("natix-jp-uk", "/capstor/store/cscs/swissai/ab036/shared/datasets/natix-jp-uk/6adb62fa"),
}
# <session>/<HH-MM-SS>/<VIEW>_FOLDER/<VIEW>_<date>_<HH-MM-SS>.mp4
PATH_RE = re.compile(r"^(?P<session>.*)/\d\d-\d\d-\d\d/[^/]+/[A-Z]+_(?P<date>\d{4}-\d\d-\d\d)_(?P<time>\d\d-\d\d-\d\d)\.mp4$")


def load_chunks(train_parquet, processed, datasets):
    """One row per source video: dataset, country, session, start, manifest duration, in_train, pose fields."""
    train = set(pd.read_parquet(train_parquet, columns=["source_video_path"]).source_video_path)
    frames = []
    for ds in datasets:
        sub, _ = DATASETS[ds]
        m = pd.read_parquet(f"{processed}/{sub}/manifest/samples_24fps.parquet",
                            columns=["source_video_path", "country", "source_num_resampled_frames"])
        m = m.drop_duplicates("source_video_path")
        m["dataset"] = ds
        m["dur"] = m.source_num_resampled_frames / 24.0
        frames.append(m.drop(columns="source_num_resampled_frames"))
    m = pd.concat(frames, ignore_index=True)
    parts = m.source_video_path.str.extract(PATH_RE)
    assert parts.session.notna().all(), "unparsed source_video_path"
    m["session"] = parts.session
    m["start"] = pd.to_datetime(parts.date + " " + parts.time.str.replace("-", ":"))
    m["in_train"] = m.source_video_path.isin(train)
    return pd.concat([m, load_poses(processed, datasets, m)], axis=1)


def load_poses(processed, datasets, m):
    """pose_source / pose_source_reason / slam_valid_until_s from each trajectories/manifest.jsonl, aligned to m's rows."""
    rows = []
    for ds in datasets:
        sub, _ = DATASETS[ds]
        manifest = pd.read_json(f"{processed}/{sub}/trajectories/manifest.jsonl", lines=True)
        # clip_id is the full video path; the part after the dataset root is source_video_path
        rel = manifest.clip_id.str.replace(r"^.*?/(?:natix|natix-jp-uk/6adb62fa)/", "", regex=True)
        rows.append(pd.DataFrame({"source_video_path": rel, "pose_source": manifest.pose_source,
                                  "reason": manifest.pose_source_reason,
                                  "valid_until": manifest.qc.map(lambda q: (q or {}).get("slam_valid_until_s"))}))
    poses = pd.concat(rows, ignore_index=True)
    return m[["source_video_path"]].merge(poses, on="source_video_path", how="left").drop(columns="source_video_path")


def split_runs(chunks, max_gap):
    """Adds `gap` (s to the previous chunk of the session) and `run` (a new run starts at a missing or too large gap)."""
    c = chunks.sort_values(["session", "start"]).reset_index(drop=True)
    g = c.groupby("session")
    prev_end = g.start.shift() + pd.to_timedelta(g.dur.shift(), unit="s")
    c["gap"] = (c.start - prev_end).dt.total_seconds()
    c["run"] = (c.gap.isna() | (c.gap.abs() > max_gap)).cumsum()
    c.loc[c.run != c.run.shift(), "gap"] = float("nan")   # a run's first chunk has no gap inside the run
    return c


def summarize(c, all_chunks):
    r = c.groupby("run").agg(dataset=("dataset", "first"), country=("country", "first"), session=("session", "first"),
                             n_chunks=("dur", "size"), minutes=("dur", lambda s: s.sum() / 60), start=("start", "min"),
                             max_gap_s=("gap", lambda s: s.abs().max() if len(s) > 1 else 0.0),
                             n_fused=("pose_source", lambda s: (s == "fused").sum()))
    untouched = all_chunks.groupby("session").in_train.sum() == 0   # no chunk of the session is a training chunk
    r["whole_session_untouched"] = r.session.map(untouched)
    return r


def mp4_duration(path):
    for _ in range(3):  # the shared file system sometimes fails a read
        try:
            with av.open(path) as k:
                return k.duration / 1e6
        except Exception:
            pass
    return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_parquet", default=f"{PROCESSED}/natix/clean_data_train.parquet",
                   help="chunks listed here (source_video_path) are excluded and break runs")
    p.add_argument("--processed_root", default=PROCESSED)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--pose_sources", nargs="+", default=["fused"], choices=["fused", "metric"])
    p.add_argument("--max_gap", type=float, default=0.6, help="max |gap| between neighbouring chunks, seconds")
    p.add_argument("--top_n", type=int, default=10, help="keep the N longest runs, one per session (0: all runs)")
    p.add_argument("--min_minutes", type=float, default=5.0)
    p.add_argument("--max_tail_s", type=float, default=2.0,
                   help="poses must reach within this many seconds of the chunk's end (qc.slam_valid_until_s)")
    p.add_argument("--prefilter_slack", type=float, default=1.5,
                   help="extra gap allowed in the prefilter that uses the manifest duration")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out_csv", required=True, help="one row per run")
    p.add_argument("--out_chunks_csv", required=True, help="the ordered chunks of every run")
    args = p.parse_args()

    allc = load_chunks(args.train_parquet, args.processed_root, args.datasets)
    print(allc.groupby("dataset").agg(chunks=("dur", "size"), in_train=("in_train", "sum")).to_string())
    ok = allc[~allc.in_train & allc.pose_source.isin(args.pose_sources) & ((allc.dur - allc.valid_until) <= args.max_tail_s)]
    print(f"chunks with {'/'.join(args.pose_sources)} poses to the end, not in training: {len(ok)}")

    # prefilter on the manifest duration, so mp4s are only opened for chunks that can still form a long enough run
    loose = split_runs(ok, args.max_gap + args.prefilter_slack)
    n_runs = loose.groupby("run").dur.transform("size")
    cand = loose[n_runs >= math.floor(args.min_minutes)].drop(columns=["gap", "run"]).copy()
    print(f"opening {len(cand)} mp4s")
    roots = {ds: root for ds, (_, root) in DATASETS.items()}
    with ThreadPoolExecutor(args.threads) as ex:
        cand["dur"] = list(ex.map(mp4_duration, (f"{roots[d]}/{v}" for d, v in zip(cand.dataset, cand.source_video_path))))
    print("unreadable mp4s:", int(cand.dur.isna().sum()))
    cand = cand.dropna(subset=["dur"])

    c = split_runs(cand, args.max_gap)
    r = summarize(c, allc)
    r = r[r.minutes >= args.min_minutes].sort_values("minutes", ascending=False)
    if args.top_n:
        r = r.drop_duplicates("session").head(args.top_n)
    r.round({"minutes": 2, "max_gap_s": 3}).to_csv(args.out_csv)
    c[c.run.isin(r.index)][["run", "dataset", "country", "session", "start", "dur", "gap", "pose_source", "reason",
                            "valid_until", "source_video_path"]].to_csv(args.out_chunks_csv, index=False)
    print(f"{len(r)} runs >= {args.min_minutes} min; by dataset: {r.groupby('dataset').size().to_dict()}; "
          f"whole session untouched by training: {int(r.whole_session_untouched.sum())}")


if __name__ == "__main__":
    main()
