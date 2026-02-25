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
FIELD_ID = "ra0hdec-59.75"
OBSERVATION_ID = "101706388"
BENCHMARK_DATA_DIR = BASE / "benchmark_data" / FIELD_ID / OBSERVATION_ID
SCAN_PATH = BENCHMARK_DATA_DIR / "0000_calibrated_scan000.npz"
OUT_TXT = BASE / "benchmark_cg_convergence.txt"

CG_TOL = 1e-3
CG_MAXITER = 512


def main() -> None:
    if not SCAN_PATH.exists():
        print(f"Benchmark scan not found: {SCAN_PATH}", file=sys.stderr)
        sys.exit(1)

    layout = build_layout(field_id=OBSERVATION_ID, scan_paths=[SCAN_PATH])
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
        "CG convergence (real data, benchmark_data)",
        "=" * 60,
        "",
        "Data",
        f"   scan_path = {SCAN_PATH}",
        f"   n_scans = {layout.n_scans}, n_obs = {layout.n_obs}, n_pix = {layout.n_pix}",
        "",
        "Joint solve (CPU): [P' N^{-1} P, P' N^{-1} W; W' N^{-1} P, M] [c; a0] = rhs",
        "   Implemented by: cad.direct_solve.reconstruct_scan.solve_single_scan (CG)",
        f"   ___ Benchmark: {cpu_iters[0]} iters to reach norm(r) <= {CG_TOL} * norm(rhs), maxiter={CG_MAXITER}.",
        "   CG stops when relative residual <= cg_tol (dimensionless).",
        "",
        "M_s solve (GPU): M_s x = W' N^{-1} d, M_s = C_a^{-1} + W' N^{-1} W",
        "   Implemented by: cad.parallel_solve.fisher.run_one_M_s_solve_converged",
        f"   ___ Benchmark: {gpu_iters} iters to reach norm(r) <= {CG_TOL} * norm(rhs), maxiter={CG_MAXITER}.",
        "   One RHS for Pt_Ninv_d; Fisher builds many such solves per scan.",
        "",
        "Production: cg_tol=1e-3, cg_maxiter=512 (run_reconstruction.py and Fisher cg_niter).",
        "CPU and GPU solve different matrices (joint vs atmosphere block), so iters differ.",
        "",
    ]
    out_text = "\n".join(lines)
    OUT_TXT.write_text(out_text)
    print(out_text)
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
