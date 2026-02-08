#!/usr/bin/env python3
"""
Toy single-scan validation for `cad.reconstruct_scan`.

Checks:
  - Adjoint tests for P/P^T and W/W^T.
  - End-to-end ML solve on a small synthetic scan.
"""

from __future__ import annotations

import numpy as np

import cad
from cad.map import BBox


def main() -> None:
    rng = np.random.default_rng(0)

    pixel_size_deg = 1.0
    pixel_res_rad = float(pixel_size_deg) * np.pi / 180.0

    bbox_cmb = BBox(ix0=0, ix1=7, iy0=0, iy1=7)  # 8x8
    bbox_atm = BBox(ix0=-3, ix1=10, iy0=-3, iy1=10)  # 14x14 (padded)

    nx_c, ny_c = int(bbox_cmb.nx), int(bbox_cmb.ny)
    nx_a, ny_a = int(bbox_atm.nx), int(bbox_atm.ny)
    n_pix_cmb = int(nx_c * ny_c)
    n_pix_atm = int(nx_a * ny_a)

    n_t = 40
    n_det = 6
    t_s = np.linspace(0.0, 20.0, n_t).astype(np.float64)

    # Simple scan: boresight sweeps in +ix; detectors offset in iy.
    ix_base = np.linspace(1, 6, n_t).round().astype(np.int64)
    iy_base = np.full((n_t,), 3, dtype=np.int64)
    det_dy = np.arange(n_det, dtype=np.int64) - (n_det // 2)

    pix_index = np.zeros((n_t, n_det, 2), dtype=np.int64)
    pix_index[..., 0] = ix_base[:, None]
    pix_index[..., 1] = iy_base[:, None] + det_dy[None, :]
    pix_index[..., 0] = np.clip(pix_index[..., 0], bbox_cmb.ix0, bbox_cmb.ix1)
    pix_index[..., 1] = np.clip(pix_index[..., 1], bbox_cmb.iy0, bbox_cmb.iy1)

    wind_deg_per_s = (0.25, -0.05)

    tod_dummy = np.zeros((n_t, n_det), dtype=np.float64)
    pm, vm = cad.util.pointing_from_pix_index(pix_index=pix_index, tod_mk=tod_dummy, bbox=bbox_cmb)
    pix_obs_local = pm[vm].astype(np.int64, copy=False)

    idx4, w4 = cad.util.frozen_screen_bilinear_weights(
        pointing_matrix=pm,
        valid_mask=vm,
        bbox_cmb=bbox_cmb,
        bbox_atm=bbox_atm,
        wind_deg_per_s=wind_deg_per_s,
        t_s=t_s,
        pixel_size_deg=float(pixel_size_deg),
        strict=True,
    )

    def P_apply(c_full: np.ndarray) -> np.ndarray:
        return np.asarray(c_full, dtype=np.float64)[pix_obs_local]

    def PT_apply(y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        return np.bincount(pix_obs_local, weights=yy, minlength=n_pix_cmb).astype(np.float64, copy=False)

    def W_apply(a0: np.ndarray) -> np.ndarray:
        a = np.asarray(a0, dtype=np.float64).reshape(-1)
        return np.sum(w4 * a[idx4], axis=1)

    def WT_apply(y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        out = np.zeros((n_pix_atm,), dtype=np.float64)
        np.add.at(out, idx4.reshape(-1), (w4 * yy[:, None]).reshape(-1))
        return out

    # Adjoint tests.
    c = rng.normal(size=(n_pix_cmb,))
    y = rng.normal(size=(pix_obs_local.size,))
    print(f"[adjoint] P mismatch = {float(np.dot(P_apply(c), y) - np.dot(c, PT_apply(y))):.3e}")

    a = rng.normal(size=(n_pix_atm,))
    y2 = rng.normal(size=(pix_obs_local.size,))
    print(f"[adjoint] W mismatch = {float(np.dot(W_apply(a), y2) - np.dot(a, WT_apply(y2))):.3e}")

    # Synthetic data.
    c_true = rng.normal(size=(n_pix_cmb,))
    a_true = rng.normal(size=(n_pix_atm,))
    d_valid = P_apply(c_true) + W_apply(a_true)

    sigma_det = 0.1 * np.ones((n_det,), dtype=np.float64)  # mK
    tod = np.full((n_t, n_det), np.nan, dtype=np.float64)
    tod[vm] = d_valid + rng.normal(scale=sigma_det[np.where(vm)[1]])

    n_ell_bins = 32
    cl_atm = np.full((n_ell_bins,), 10.0, dtype=np.float64)
    cl_cmb = np.full((n_ell_bins,), 1.0, dtype=np.float64)
    prior_atm = cad.SpectralPriorFFT(nx=nx_a, ny=ny_a, pixel_res_rad=pixel_res_rad, cl_bins_mk2=cl_atm)
    prior_cmb = cad.SpectralPriorFFT(nx=nx_c, ny=ny_c, pixel_res_rad=pixel_res_rad, cl_bins_mk2=cl_cmb)

    obs_pix = np.arange(n_pix_cmb, dtype=np.int64)
    global_to_obs = np.arange(n_pix_cmb, dtype=np.int64)

    sol = cad.reconstruct_scan.solve_single_scan(
        tod_mk=tod,
        pix_index=pix_index,
        t_s=t_s,
        pixel_size_deg=pixel_size_deg,
        wind_deg_per_s=wind_deg_per_s,
        noise_std_det_mk=sigma_det,
        prior_atm=prior_atm,
        prior_cmb=prior_cmb,
        bbox_cmb=bbox_cmb,
        bbox_atm=bbox_atm,
        obs_pix_cmb=obs_pix,
        global_to_obs_cmb=global_to_obs,
        estimator_mode="ML",
        n_scans=1,
        cg_tol=1e-10,
        cg_maxiter=400,
    )

    c_hat = np.asarray(sol.c_hat_full_mk, dtype=np.float64)
    hits = np.bincount(pix_obs_local, minlength=n_pix_cmb)
    m = hits > 0
    ct = c_true[m] - float(np.mean(c_true[m]))
    ch = c_hat[m] - float(np.mean(c_hat[m]))
    corr = float(np.corrcoef(ct, ch)[0, 1])
    rel = float(np.linalg.norm(ch - ct) / np.linalg.norm(ct))
    print(f"[solve] hit_pix={int(np.sum(m))}/{n_pix_cmb} corr={corr:.3f} rel_err={rel:.3e}")


if __name__ == "__main__":
    main()

