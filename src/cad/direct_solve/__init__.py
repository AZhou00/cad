"""
Direct solve: sequential per-scan reconstruction + exact multi-scan joint synthesis.

Shared kernels: cad (src/cad). This package: direct workflow + synthesize_scans kernel.
"""

from .synthesize_scan import MultiScanSolve, synthesize_scans
from .workflow_joint import (
    merge_preps_for_all_observations,
    prepare_synthesis_inputs,
    run_synthesis_group,
)
from .workflow_single import DirectConfig, discover_recon_paths, run_one_scan

__all__ = [
    "DirectConfig",
    "MultiScanSolve",
    "synthesize_scans",
    "discover_recon_paths",
    "merge_preps_for_all_observations",
    "prepare_synthesis_inputs",
    "run_one_scan",
    "run_synthesis_group",
]
