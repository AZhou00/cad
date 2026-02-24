"""
Parallel solve: layout + per-scan reconstruction + exact synthesis.

Shared kernels: cad (src/cad). This package: layout, run_one_scan, synthesis.
"""

from .layout import (
    GlobalLayout,
    build_layout,
    load_layout,
    save_layout,
    discover_fields,
    discover_scan_paths,
    load_scan_for_layout,
)
from .reconstruct_scan import load_scan_artifact, run_one_scan
from .synthesize_scan import run_synthesis

__all__ = [
    "GlobalLayout",
    "build_layout",
    "load_layout",
    "save_layout",
    "discover_fields",
    "discover_scan_paths",
    "load_scan_for_layout",
    "run_one_scan",
    "run_synthesis",
    "load_scan_artifact",
]
