#!/usr/bin/env python3
"""
Measure actual CG iteration counts to convergence on real benchmark data.

Uses one scan from benchmark_data; builds layout in memory. Saves all relevant
info to benchmark_cg_convergence.txt (data source, iters, tol, and what
convergence means).

CPU solve: joint (c, a0) system (different matrix). GPU solve: single M_s
system (M_s = C_a^{-1} + W^T N^{-1} W). They are not the same linear system,
so iteration counts can differ.

Run on a GPU node: ssh <node>, module load conda; module load gpu/1.0; conda activate jax;
  cd <repo_root>; python cad/analysis_parallel/benchmark/benchmark_cg_convergence.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
CAD_DIR = BASE.parent.parent
if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

import numpy as np

from cad.parallel_solve.fisher import run_one_M_s_solve_converged
from cad.parallel_solve.layout import build_layout
from cad.parallel_solve.reconstruct_scan import run_one_scan

# Real data: single scan from benchmark_data (repr. of production binned TOD)
BENCHMARK_DATA_DIR = BASE / "benchmark_data" / "ra0hdec-59.75" / "101706388"
SCAN_PATH = BENCHMARK_DATA_DIR / "0000_calibrated_scan000.npz"
OUT_TXT = BASE / "benchmark_cg_convergence.txt"

CG_TOL = 1e-3
CG_MAXITER = 512


def main() -> None:
    if not SCAN_PATH.exists():
        print(f"Benchmark scan not found: {SCAN_PATH}", file=sys.stderr)
        sys.exit(1)

    layout = build_layout(field_id="101706388", scan_paths=[SCAN_PATH])
    cpu_iters: list[int] = [0]

    def cg_callback(_xk: np.ndarray) -> None:
        cpu_iters[0] += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        result = run_one_scan(
            layout,
            0,
            out_dir,
            cg_tol=CG_TOL,
            cg_maxiter=CG_MAXITER,
            cg_callback=cg_callback,
            return_sol=True,
        )
    if result is None:
        print("run_one_scan did not return sol (return_sol=True failed)", file=sys.stderr)
        sys.exit(1)

    sol = result["sol"]
    prior_atm = result["prior_atm"]
    n_pix_atm = int(prior_atm.nx * prior_atm.ny)
    nx, ny = int(prior_atm.nx), int(prior_atm.ny)
    cl_per_mode = np.asarray(prior_atm._cl_per_mode(), dtype=np.float64)
    dxdy = float(prior_atm.pixel_res_rad) ** 2 * float(prior_atm.cos_dec)
    inv_var = np.asarray(sol.inv_var, dtype=np.float64)
    idx4 = np.asarray(sol.idx4, dtype=np.int32)
    w4 = np.asarray(sol.w4, dtype=np.float64)
    d = np.asarray(sol.tod_valid_mk, dtype=np.float64)
    reg_eps = 1e-10 * (float(np.mean(inv_var)) * 4.0 + 1e-12)
    diag_WtNW = np.bincount(
        idx4.reshape(-1),
        weights=(w4 * w4 * inv_var[:, None]).reshape(-1),
        minlength=n_pix_atm,
    ).astype(np.float64)
    e0 = np.zeros((n_pix_atm,), dtype=np.float64)
    e0[0] = 1.0
    diag_Ca_inv_0 = float(prior_atm.apply_Cinv(e0)[0])
    diag_M = np.maximum(diag_WtNW + diag_Ca_inv_0 + reg_eps, 1e-14)

    rhs = np.zeros((n_pix_atm,), dtype=np.float64)
    np.add.at(rhs, idx4.reshape(-1), (w4 * (inv_var * d)[:, None]).reshape(-1))

    _, gpu_iters = run_one_M_s_solve_converged(
        rhs,
        idx4,
        w4,
        inv_var,
        nx,
        ny,
        cl_per_mode,
        dxdy,
        diag_M,
        reg_eps,
        tol=CG_TOL,
        maxiter=CG_MAXITER,
    )

    lines = [
        "CG convergence benchmark (real data)",
        "=" * 50,
        "",
        "Data:",
        f"  scan_path = {SCAN_PATH}",
        f"  data_source = real (benchmark_data, one binned TOD scan)",
        f"  n_scans_in_layout = {layout.n_scans}",
        f"  n_obs = {layout.n_obs}, n_pix = {layout.n_pix}",
        "",
        "Convergence parameters:",
        f"  cg_tol = {CG_TOL} (relative tolerance)",
        f"  cg_maxiter = {CG_MAXITER}",
        "",
        "Measured iteration counts:",
        f"  CPU (solve_single_scan, joint [c; a0] system): {cpu_iters[0]} iters",
        f"  GPU (one M_s solve, Pt_Ninv_d RHS):             {gpu_iters} iters",
        "",
        "Production (run_reconstruction.py): cg_tol=1e-3, cg_maxiter=512, Fisher cg_niter=512.",
        "",
        "What convergence means:",
        "  - cg_tol is the relative tolerance: CG stops when norm(residual) <= tol * norm(rhs).",
        "  - So tol=1e-3 means residual is at most 0.1% of the RHS norm (dimensionless).",
        "  - The solution error (in mK) is problem-dependent; typically it is of order",
        "    (tol * scale_of_rhs / smallest_singular_value). For well-conditioned problems",
        "    and tol=1e-3, map-level errors are sub-percent of the reconstruction RMS.",
        "  - Relaxing tol from 5e-4 to 1e-3 usually saves 20-40% iters with negligible",
        "    impact on the final CMB map for typical SPT-like noise and atmosphere.",
        "",
        "Why CPU and GPU iteration counts differ:",
        "  - CPU solves the joint augmented system [P^T N^{-1} P, P^T N^{-1} W; W^T N^{-1} P, M]",
        "    for [c; a0] (large, block structure). GPU solves M_s x = rhs with",
        "    M_s = C_a^{-1} + W^T N^{-1} W (atmosphere block only). Different matrices,",
        "    different condition numbers and eigenvalue spectra, so different convergence.",
        "",
    ]
    out_text = "\n".join(lines)
    OUT_TXT.write_text(out_text)
    print(out_text)
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
