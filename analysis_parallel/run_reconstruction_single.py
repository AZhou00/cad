#!/usr/bin/env python3
"""
Single-scan ML reconstruction for one layout task. Thin CLI around parallel_solve.
Usage: run_reconstruction_single.py <layout.npz> <out_dir> [scan_index] [n_ell_bins] ...
"""

from __future__ import annotations

import os
import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

from cad.parallel_solve import load_layout, run_one_scan


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(
            "Usage: run_reconstruction_single.py <layout.npz> <out_dir> [scan_index]\n"
            "  or set SLURM_ARRAY_TASK_ID and omit scan_index.",
            file=sys.stderr,
        )
        sys.exit(1)
    layout_path = pathlib.Path(argv[0])
    out_dir = pathlib.Path(argv[1])
    scan_index = int(argv[2]) if len(argv) > 2 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    n_ell_bins = int(argv[3]) if len(argv) > 3 else 128
    cl_floor = float(argv[4]) if len(argv) > 4 else 1e-12
    noise_mk = float(argv[5]) if len(argv) > 5 else 10.0
    cg_tol = float(argv[6]) if len(argv) > 6 else 5e-4
    cg_maxiter = int(argv[7]) if len(argv) > 7 else 1200

    layout = load_layout(layout_path)
    if scan_index < 0 or scan_index >= layout.n_scans:
        raise RuntimeError(f"scan_index {scan_index} out of range [0, {layout.n_scans})")
    run_one_scan(
        layout,
        scan_index,
        out_dir,
        n_ell_bins=n_ell_bins,
        cl_floor_mk2=cl_floor,
        noise_per_raw_detector_per_153hz_sample_mk=noise_mk,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
    )


if __name__ == "__main__":
    main()
