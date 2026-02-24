#!/usr/bin/env python3
"""
Exact global synthesis from per-scan cov_inv and Pt_Ninv_d. Thin CLI around parallel_solve.

Loads per-scan npzs (cov_inv, Pt_Ninv_d, obs_pix_global_scan), accumulates cov_inv_tot and Pt_Ninv_d_tot,
solves cov_inv_tot @ c_hat = Pt_Ninv_d_tot, writes combined map. Paths configurable at top of file.
Run after all per-scan reconstructions complete.
"""

from __future__ import annotations

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

# Hardcoded paths (edit for another observation)
DATASET_NAME = "ra0hdec-59.75"
FIELD_ID = "101706388"
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
LAYOUT_NPZ = OUT_BASE / DATASET_NAME / FIELD_ID / "layout.npz"
SCAN_NPZ_DIR = OUT_BASE / DATASET_NAME / FIELD_ID / "scans"
OUT_NPZ = OUT_BASE / DATASET_NAME / FIELD_ID / "recon_combined_ml.npz"

from cad.parallel_solve import load_layout, run_synthesis


def main() -> None:
    layout = load_layout(LAYOUT_NPZ)
    run_synthesis(layout, SCAN_NPZ_DIR, OUT_NPZ)


if __name__ == "__main__":
    main()
