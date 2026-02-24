"""
Parallel solve: exact global synthesis from per-scan cov_inv and Pt_Ninv_d.

Sum [Cov(hat c_s)]^{-1} and P^T tilde N^{-1} d at global observed indices; solve for hat c.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg as la

from .layout import GlobalLayout
from .reconstruct_scan import load_scan_artifact


def run_synthesis(
    layout: GlobalLayout,
    scan_npz_dir: Path,
    out_path: Path,
) -> None:
    """
    Exact marginalized synthesis: accumulate cov_inv_tot and Pt_Ninv_d_tot from per-scan npzs,
    solve cov_inv_tot @ c_hat_obs = Pt_Ninv_d_tot, embed to full CMB grid.
    """
    n_obs = layout.n_obs
    global_to_obs = layout.global_to_obs
    cov_inv_tot = np.zeros((n_obs, n_obs), dtype=np.float64)
    Pt_Ninv_d_tot = np.zeros((n_obs,), dtype=np.float64)

    for scan_index in range(layout.n_scans):
        npz_path = scan_npz_dir / f"scan_{scan_index:04d}_ml.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing scan artifact: {npz_path}")
        art = load_scan_artifact(npz_path)
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

    zero_precision_mask = np.all(cov_inv_tot == 0.0, axis=1)
    good = ~zero_precision_mask
    c_hat_obs = np.zeros((n_obs,), dtype=np.float64)
    if np.any(good):
        cov_inv_good = cov_inv_tot[np.ix_(good, good)]
        Pt_Ninv_d_good = Pt_Ninv_d_tot[good]
        try:
            c_hat_good = la.solve(cov_inv_good, Pt_Ninv_d_good, assume_a="sym", check_finite=False)
        except la.LinAlgError as err:
            raise RuntimeError(
                "Global synthesis solve failed (singular/ill-conditioned cov_inv_tot)."
            ) from err
        c_hat_obs[good] = np.asarray(c_hat_good, dtype=np.float64)
        c_hat_obs -= float(np.mean(c_hat_obs[good]))

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
    np.savez_compressed(
        out_path,
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
    )
    print(f"[write] {out_path} n_obs={n_obs}", flush=True)
