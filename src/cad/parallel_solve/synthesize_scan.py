"""
Parallel solve: exact global synthesis from per-scan cov_inv and Pt_Ninv_d.

Sum [Cov(hat c_s)]^{-1} and P^T tilde N^{-1} d at global observed indices; solve for hat c.
We must form cov_inv_tot to save it; the solve (cov_inv_tot @ c_hat = RHS) is O(n_obs^3).
Solve runs on GPU via JAX (fail if solve errors). CG is an option for the solve
(matrix-free CG would avoid storing cov_inv_tot during solve but we still form it once to save).
Most efficient with dense precision required: form once, solve on GPU (Cholesky/solve).
The dense precision matrix cov_inv_tot (n_obs x n_obs) is always saved in the output NPZ.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .layout import GlobalLayout
from .reconstruct_scan import load_scan_artifact


def _solve_synthesis_gpu(cov_inv_good: np.ndarray, Pt_Ninv_d_good: np.ndarray) -> np.ndarray:
    """Solve cov_inv_good @ x = Pt_Ninv_d_good on GPU with JAX. Returns c_hat_good (n_good,)."""
    import jax.numpy as jnp
    A = jnp.asarray(cov_inv_good)
    b = jnp.asarray(Pt_Ninv_d_good)
    x = jnp.linalg.solve(A, b)
    return np.asarray(x, dtype=np.float64)


def run_synthesis(
    layout: GlobalLayout,
    scan_npz_dir: Path,
    out_path: Path,
    timings: dict | None = None,
) -> None:
    """
    Exact marginalized synthesis: accumulate cov_inv_tot and Pt_Ninv_d_tot from per-scan npzs,
    solve cov_inv_tot @ c_hat_obs = Pt_Ninv_d_tot, embed to full CMB grid.
    Always saves the dense precision matrix cov_inv_tot (n_obs x n_obs). If timings, keys: load_s, accumulate_s, solve_s.
    """
    n_obs = layout.n_obs
    global_to_obs = layout.global_to_obs
    cov_inv_tot = np.zeros((n_obs, n_obs), dtype=np.float64)
    Pt_Ninv_d_tot = np.zeros((n_obs,), dtype=np.float64)

    existing = [i for i in range(layout.n_scans) if (scan_npz_dir / f"scan_{i:04d}_ml.npz").exists()]
    if not existing:
        raise FileNotFoundError(f"No scan_*_ml.npz found in {scan_npz_dir}")

    t0 = time.perf_counter()
    artifacts = []
    for scan_index in tqdm(existing, desc="load scans", leave=True):
        npz_path = scan_npz_dir / f"scan_{scan_index:04d}_ml.npz"
        art = load_scan_artifact(npz_path)
        artifacts.append((scan_index, art))
    if timings is not None:
        timings["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for scan_index, art in tqdm(artifacts, desc="accumulate", leave=True):
        obs_pix_global_scan = art["obs_pix_global_scan"]
        cov_inv_s = art["cov_inv"]
        Pt_Ninv_d_s = art["Pt_Ninv_d"]
        obs_idx = np.asarray(global_to_obs[obs_pix_global_scan], dtype=np.int64)
        valid = obs_idx >= 0
        obs_idx_valid = obs_idx[valid]
        if obs_idx_valid.size == 0:
            continue
        cov_inv_s_valid = cov_inv_s[np.ix_(valid, valid)]
        Pt_Ninv_d_s_valid = Pt_Ninv_d_s[valid]
        cov_inv_tot[np.ix_(obs_idx_valid, obs_idx_valid)] += cov_inv_s_valid
        Pt_Ninv_d_tot[obs_idx_valid] += Pt_Ninv_d_s_valid
    if timings is not None:
        timings["accumulate_s"] = time.perf_counter() - t0

    zero_precision_mask = np.all(cov_inv_tot == 0.0, axis=1)
    good = ~zero_precision_mask
    c_hat_obs = np.zeros((n_obs,), dtype=np.float64)
    t0 = time.perf_counter()
    if np.any(good):
        cov_inv_good = cov_inv_tot[np.ix_(good, good)]
        Pt_Ninv_d_good = Pt_Ninv_d_tot[good]
        c_hat_good = _solve_synthesis_gpu(cov_inv_good, Pt_Ninv_d_good)
        c_hat_good = np.asarray(c_hat_good, dtype=np.float64)
        c_hat_obs[good] = c_hat_good
        c_hat_obs -= float(np.mean(c_hat_obs[good]))
    if timings is not None:
        timings["solve_s"] = time.perf_counter() - t0

    precision_diag_total = np.diag(cov_inv_tot).copy()
    var_diag_total = np.full((n_obs,), np.inf, dtype=np.float64)
    var_diag_total[good] = np.where(
        precision_diag_total[good] > 0.0,
        1.0 / precision_diag_total[good],
        np.inf,
    )

    n_pix = layout.n_pix
    c_hat_full_mk = np.zeros((n_pix,), dtype=np.float64)
    c_hat_full_mk[layout.obs_pix_global] = c_hat_obs  # (n_pix_cmb,)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _out = dict(
        estimator_mode=np.array("ML", dtype=object),
        bbox_ix0=np.int64(layout.bbox_ix0),
        bbox_iy0=np.int64(layout.bbox_iy0),
        nx=np.int64(layout.nx),
        ny=np.int64(layout.ny),
        pixel_size_deg=np.float64(layout.pixel_size_deg),
        obs_pix_global=layout.obs_pix_global,
        c_hat_full_mk=c_hat_full_mk,
        c_hat_obs=c_hat_obs,
        precision_diag_total=precision_diag_total,
        var_diag_total=var_diag_total,
        zero_precision_mask=zero_precision_mask,
        n_scans=np.int64(layout.n_scans),
        n_scans_used=np.int64(len(existing)),
        cov_inv_tot=cov_inv_tot,
    )
    np.savez_compressed(out_path, **_out)
    print(f"[write] {out_path} n_obs={n_obs} n_scans_used={len(existing)}/{layout.n_scans}", flush=True)
