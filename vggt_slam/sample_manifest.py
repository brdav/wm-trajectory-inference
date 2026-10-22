"""Reader for the dataset's sample manifest -- the list of clips the pipeline should process.

The manifest (parquet, one row per 5 s sample/chunk) lives at vggt_slam.paths.SAMPLE_MANIFEST. Its
`source_video_path` column names the full source video each sample was cut from, as a path relative to
the dataset root (Country/.../<session>/<HH-MM-SS>/FRONT_FOLDER/FRONT_<timestamp>.mp4). The pipeline
processes whole source videos, so the work list is the set of unique source videos in the manifest,
not the 5 s chunks. Nothing walks the dataset tree.

Both scripts/batch_infer.py (what to process) and scripts/pack_sessions.py (which sessions are
complete) read the manifest through this module, so they always agree on the clip list.

To process more data: add its videos to the sample manifest (same columns), then run
scripts/run_full_dataset.sbatch.
"""
from pathlib import Path

from vggt_slam.paths import SAMPLE_MANIFEST

VIDEO_COLUMN = "source_video_path"


def source_videos(manifest_path=SAMPLE_MANIFEST, countries=None):
    """Unique source videos of the manifest as dataset-relative path strings, in order of first
    appearance. manifest_path: a parquet file, or a hive-partitioned parquet directory
    (.../samples_24fps/country=<name>/part.parquet); both hold the same rows.
    countries: optional iterable of country names (first path component) to keep."""
    import pyarrow.dataset as ds

    path = Path(manifest_path)
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive" if path.is_dir() else None)
    if VIDEO_COLUMN not in dataset.schema.names:
        raise ValueError(f"{manifest_path} has no '{VIDEO_COLUMN}' column (columns: {dataset.schema.names})")
    seen = {}
    for batch in dataset.to_batches(columns=[VIDEO_COLUMN]):
        for v in batch.column(0).to_pylist():
            if v is not None and v not in seen:
                seen[v] = None
    keep = set(countries) if countries else None
    return [v for v in seen if keep is None or v.split("/", 1)[0] in keep]


def session_of(rel_video_path):
    """Session directory (dataset-relative) of a source video: two levels above its FRONT_FOLDER,
    i.e. Country/.../<session>. Same convention as vggt_slam.undistort.session_cache_dir."""
    return "/".join(str(rel_video_path).split("/")[:-3])
