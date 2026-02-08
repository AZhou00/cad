"""
cad: CMB atmosphere decontamination utilities.

The main public entry points are:
  - `solve_single_scan` (single scan)
  - `synthesize_scans` (multi-scan)
"""

from .reconstruct_scan import ScanSolve, solve_single_scan
from .synthesize_scan import MultiScanSolve, synthesize_scans
from .prior import SpectralPriorFFT
from .wind import estimate_wind_deg_per_s

__all__ = [
    "SpectralPriorFFT",
    "estimate_wind_deg_per_s",
    "ScanSolve",
    "solve_single_scan",
    "MultiScanSolve",
    "synthesize_scans",
]

