"""Shared helpers for the session-level zip archive format that scripts/pack_sessions.py /
scripts/unpack_sessions.py use to cut the output tree's inode count further, on top of the
per-clip result.npz consolidation (see save_result_npz's docstring). Packing replaces a whole
Country/Date/Session/ directory (session_cache_dir's return value -- everything sharing one
undistort cache, avg ~6 clips/session in Croatia's data) with a single sibling
Country/Date/Session.zip containing the same files at the same relative paths.

Archival, not transparent: a packed session is meant to be done for good. Anything that needs
to read or rewrite a packed clip (a plot script, a future scripts/refuse_gps.py sweep) has to
scripts/unpack_sessions.py it back to a loose directory first -- see that script's docstring for
why (refuse_gps.py already re-touches old, already-"finished" clips, so packing can't assume a
session will never need writing to again; it can only make that the exception, not something
every reader has to handle transparently).

Packing-awareness in the live pipeline: scripts/batch_infer.py's discover_clips treats a clip as done if
its result.npz is loose, has an ok row in manifest.jsonl, or is a member of the session zip
(session_zip_path). A packed session is therefore NOT assumed complete: a clip added to it later is processed,
and scripts/pack_sessions.py merges it into the existing zip.
"""
from vggt_slam.undistort import session_cache_dir


def session_zip_path(out_dir):
    """The path a clip's session is (or would be) packed to, given the clip's own out_dir.
    session_cache_dir(out_dir) is the session directory itself; this is its zip sibling --
    same parent, ".zip" appended to the directory name rather than replacing any part of it
    (session dirnames are "<date>_<uuid>_<n>", no extension of their own to collide with)."""
    session_dir = session_cache_dir(out_dir)
    return session_dir.parent / (session_dir.name + ".zip")


def is_session_packed(out_dir):
    """True if out_dir's session has already been packed (its directory replaced by a .zip)."""
    return session_zip_path(out_dir).exists()
