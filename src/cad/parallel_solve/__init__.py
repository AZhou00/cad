"""
Parallel solve: layout + per-scan reconstruction + exact synthesis.

Shared kernels: cad (src/cad). This package: layout, run_one_scan, synthesis.
"""

from .artifact_io import load_scan_artifact
from .layout import (
    GlobalLayout,
    build_layout,
    load_layout,
    save_layout,
    discover_fields,
    discover_scan_paths,
    load_scan_for_layout,
)
from .synthesize_scan import run_synthesis, run_synthesis_multi_obs


def run_one_scan(*args, **kwargs):
    """Lazy import to avoid pulling solver deps unless reconstruction is used."""
    from .reconstruct_scan import run_one_scan as _run_one_scan

    return _run_one_scan(*args, **kwargs)

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
    "run_synthesis_multi_obs",
    "load_scan_artifact",
]
