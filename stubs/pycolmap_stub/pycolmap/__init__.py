"""No-op stand-in for the real `pycolmap` (github.com/colmap/pycolmap), which
has no aarch64 PyPI wheel (GH200 target hardware). `depth_anything_3.utils.
export.colmap` imports `pycolmap` eagerly at module load time, even though this
pipeline never calls its COLMAP-format export -- so without this stub,
`from depth_anything_3.api import DepthAnything3` fails outright on aarch64.
Anything beyond the bare `import pycolmap` (i.e. actually exporting to COLMAP
format) will raise AttributeError; that's expected, since that path is unused
here.
"""
