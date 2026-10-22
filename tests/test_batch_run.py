"""BatchRun's clip feeder (scripts/slam/batch_infer.py): a failing clip selection must end the feed, not hang the run."""
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cv2")
import batch_infer


def make_run(clips):
    run = batch_infer.BatchRun(SimpleNamespace(fps=10, ffmpeg_threads=0), clips, {}, Path("."))
    run.extract_slot = threading.Semaphore(100)
    return run


class InlinePool:
    """Stands in for the extraction thread pool: runs submitted work immediately."""
    def submit(self, fn, *a):
        from concurrent.futures import Future
        f = Future()
        f.set_result(("clip", None))
        return f


def drain(q):
    items = []
    while not q.empty():
        items.append(q.get())
    return items


def test_feeder_failure_still_sends_sentinel(monkeypatch):
    def clips():
        raise FileNotFoundError("sample manifest missing")
        yield  # noqa: unreachable, makes this a generator like clips_to_process

    run = make_run(clips())
    run._feed(InlinePool())
    assert drain(run.extract_done_q) == [None]
    assert "sample manifest missing" in run.feeder_error


def test_feeder_submits_every_clip_then_sentinel(monkeypatch):
    monkeypatch.setattr(batch_infer, "extract_one", lambda c, fps, threads: (c, None))
    clips = [{"camera_meta": {"fx": 1}}, {"camera_meta": None}, {"camera_meta": {"fx": 1}}]
    run = make_run(iter(clips))
    run._feed(InlinePool())
    items = drain(run.extract_done_q)
    assert items[-1] is None and len(items) == 4
    assert run.submitted == 3 and (run.fixed_count, run.shared_count) == (2, 1)
    assert run.feeder_error is None
