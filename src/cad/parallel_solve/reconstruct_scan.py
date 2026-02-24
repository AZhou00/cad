"""
Parallel solve: single-scan ML reconstruction (run_one_scan) and scan artifact loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import cad
from cad import dataset_io
from cad import map as map_util
from cad.map import BBox
from cad import power
from cad import reconstruct_scan as cad_reconstruct_scan
from cad import util

from .layout import GlobalLayout


def _atm_var_per_sample_from_w4(
    *,
    prior_atm: cad.FourierGaussianPrior,
    w4: np.ndarray,
) -> np.ndarray:
    """Approximate diag(W C_a W^T) for bilinear W using stationary C_a."""
    w4 = np.asarray(w4, dtype=np.float64)
    if w4.ndim != 2 or w4.shape[1] != 4:
        raise ValueError("w4 must have shape (n_valid, 4).")
    nx = int(prior_atm.nx)
    ny = int(prior_atm.ny)
    n_pix = nx * ny
    e0 = np.zeros((n_pix,), dtype=np.float64)
    e0[0] = 1.0
    cov0 = np.asarray(prior_atm.apply_C(e0), dtype=np.float64).reshape(nx, ny)
    offsets = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    k4 = np.empty((4, 4), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            dx = int((offsets[b, 0] - offsets[a, 0]) % nx)
            dy = int((offsets[b, 1] - offsets[a, 1]) % ny)
            k4[a, b] = float(cov0[dx, dy])
    atm_var = np.einsum("ni,ij,nj->n", w4, k4, w4, optimize=True)
    return np.maximum(atm_var, 0.0)


def run_one_scan(
    layout: GlobalLayout,
    scan_index: int,
    out_dir: Path,
    *,
    n_ell_bins: int = 128,
    cl_floor_mk2: float = 1e-12,
    noise_per_raw_detector_per_153hz_sample_mk: float = 10.0,
    cg_tol: float = 5e-4,
    cg_maxiter: int = 1200,
) -> None:
    scan_path = layout.scan_paths[scan_index]
    s0 = dataset_io.load_scan(scan_path)
    pixel_size_deg = float(s0["pixel_size_deg"])
    pixel_res_rad = pixel_size_deg * np.pi / 180.0

    w_x, w_y, _wind_diag, mask_good = cad.estimate_wind_deg_per_s(
        eff_tod_mk=s0["eff_tod_mk"],
        eff_pos_deg=s0["eff_pos_deg"],
        boresight_pos_deg=s0["boresight_pos_deg"],
        t_s=s0["t_s"],
        eff_counts=s0["eff_counts"],
        physical_degree=False,
    )
    wind_deg_per_s = (float(w_x), float(w_y))

    s = dict(s0)
    msk = np.asarray(mask_good, dtype=bool)
    s["eff_tod_mk"] = np.asarray(s0["eff_tod_mk"])[:, msk]
    s["pix_index"] = np.asarray(s0["pix_index"], dtype=np.int64)[:, msk, :]
    s["eff_counts"] = np.asarray(s0["eff_counts"], dtype=np.float64)[msk]

    noise_std = util.noise_std_eff_mk_from_counts(
        eff_counts=np.asarray(s["eff_counts"], dtype=np.float64),
        bin_sec=float(s["bin_sec"]),
        sample_rate_hz=float(s["sample_rate_hz"]),
        noise_per_raw_detector_per_153hz_sample_mk=noise_per_raw_detector_per_153hz_sample_mk,
    )

    valid_i = np.isfinite(s["eff_tod_mk"])
    bbox_obs_i = map_util.scan_bbox_from_pix_index(
        pix_index=np.asarray(s["pix_index"], dtype=np.int64),
        valid_mask=valid_i,
    )
    coadd_mk, hit_2d = map_util.coadd_map_global(
        scans_eff_tod_mk=[np.asarray(s["eff_tod_mk"], dtype=np.float64)],
        scans_pix_index=[np.asarray(s["pix_index"], dtype=np.int64)],
        bbox=bbox_obs_i,
    )
    hit_i = hit_2d > 0
    coadd_mk = np.asarray(coadd_mk, dtype=np.float64)
    hm = hit_i & np.isfinite(coadd_mk)
    if np.any(hm):
        coadd_mk = coadd_mk.copy()
        coadd_mk[hm] -= float(np.mean(coadd_mk[hm]))
    ell_i, cl_i = power.radial_cl_1d_from_map(
        map_2d_mk=coadd_mk,
        pixel_res_rad=pixel_res_rad,
        hit_mask=hit_i,
        n_ell_bins=n_ell_bins,
    )
    cl_i = np.where(np.isfinite(cl_i), cl_i, 0.0)
    cl_i = np.maximum(np.asarray(cl_i, dtype=np.float64), cl_floor_mk2)

    bbox_cmb = BBox(
        ix0=layout.bbox_ix0,
        ix1=layout.bbox_ix0 + layout.nx - 1,
        iy0=layout.bbox_iy0,
        iy1=layout.bbox_iy0 + layout.ny - 1,
    )
    bbox_atm = util.bbox_pad_for_open_boundary(
        bbox_obs=bbox_obs_i,
        scans_pix_index=[np.asarray(s["pix_index"], dtype=np.int64)],
        scans_tod_mk=[np.asarray(s["eff_tod_mk"], dtype=np.float64)],
        scans_t_s=[np.asarray(s["t_s"], dtype=np.float64)],
        winds_deg_per_s=[wind_deg_per_s],
        pixel_size_deg=pixel_size_deg,
    )
    iy_mid = (float(bbox_cmb.iy0) + float(bbox_cmb.iy1)) / 2.0
    cos_dec = float(np.cos(np.deg2rad(iy_mid * pixel_size_deg)))

    prior_atm = cad.FourierGaussianPrior(
        nx=int(bbox_atm.nx),
        ny=int(bbox_atm.ny),
        pixel_res_rad=pixel_res_rad,
        cl_bins_mk2=cl_i,
        cos_dec=cos_dec,
        cl_floor_mk2=cl_floor_mk2,
    )
    cl_cmb_bins_mk2 = np.full_like(np.asarray(cl_i, dtype=np.float64), cl_floor_mk2, dtype=np.float64)
    prior_cmb = cad.FourierGaussianPrior(
        nx=int(bbox_cmb.nx),
        ny=int(bbox_cmb.ny),
        pixel_res_rad=pixel_res_rad,
        cl_bins_mk2=cl_cmb_bins_mk2,
        cos_dec=cos_dec,
        cl_floor_mk2=cl_floor_mk2,
    )

    sol = cad_reconstruct_scan.solve_single_scan(
        tod_mk=np.asarray(s["eff_tod_mk"], dtype=np.float64),
        pix_index=np.asarray(s["pix_index"], dtype=np.int64),
        t_s=np.asarray(s["t_s"], dtype=np.float64),
        pixel_size_deg=float(pixel_size_deg),
        wind_deg_per_s=wind_deg_per_s,
        noise_std_det_mk=np.asarray(noise_std, dtype=np.float64),
        prior_atm=prior_atm,
        prior_cmb=prior_cmb,
        bbox_cmb=bbox_cmb,
        bbox_atm=bbox_atm,
        obs_pix_cmb=np.asarray(layout.obs_pix_global, dtype=np.int64),
        global_to_obs_cmb=np.asarray(layout.global_to_obs, dtype=np.int64),
        estimator_mode="ML",
        n_scans=int(layout.n_scans),
        cl_floor_mk2=float(cl_floor_mk2),
        cg_tol=float(cg_tol),
        cg_maxiter=int(cg_maxiter),
    )
    c_hat_full = np.asarray(sol.c_hat_full_mk, dtype=np.float64)

    obs_all = np.asarray(sol.pix_obs_local, dtype=np.int64)
    obs_idx_scan = np.unique(obs_all)
    n_obs_scan = int(obs_idx_scan.size)
    obs_pix_global_scan = np.asarray(layout.obs_pix_global, dtype=np.int64)[obs_idx_scan]
    pix_obs_local_scan = np.searchsorted(obs_idx_scan, obs_all)
    if not np.all(obs_idx_scan[pix_obs_local_scan] == obs_all):
        raise RuntimeError("Internal error: scan-local observed index mapping failed.")

    c_hat_scan_obs = c_hat_full[obs_pix_global_scan]

    data_var = 1.0 / np.asarray(sol.inv_var, dtype=np.float64)
    atm_var = _atm_var_per_sample_from_w4(prior_atm=prior_atm, w4=np.asarray(sol.w4, dtype=np.float64))
    var_eff = data_var + atm_var
    if not np.all(np.isfinite(var_eff)) or np.any(var_eff <= 0.0):
        raise RuntimeError("Invalid effective sample variances in diagonal covariance approximation.")
    inv_var_eff = 1.0 / var_eff
    precision_diag_scan_obs = np.bincount(
        pix_obs_local_scan,
        weights=inv_var_eff,
        minlength=n_obs_scan,
    ).astype(np.float64)
    precision_diag_scan_obs = np.maximum(precision_diag_scan_obs, 1e-30)
    var_diag_scan_obs = 1.0 / precision_diag_scan_obs

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scan_{scan_index:04d}_ml.npz"
    np.savez_compressed(
        out_path,
        scan_index=np.int64(scan_index),
        source_scan_path=np.array(str(scan_path), dtype=object),
        bbox_ix0=np.int64(layout.bbox_ix0),
        bbox_iy0=np.int64(layout.bbox_iy0),
        nx=np.int64(layout.nx),
        ny=np.int64(layout.ny),
        obs_pix_global_scan=obs_pix_global_scan,
        c_hat_scan_obs=c_hat_scan_obs,
        precision_diag_scan_obs=precision_diag_scan_obs,
        var_diag_scan_obs=var_diag_scan_obs,
        pixel_size_deg=np.float64(pixel_size_deg),
        wind_deg_per_s=np.array(wind_deg_per_s, dtype=np.float64),
        n_obs_scan=np.int64(n_obs_scan),
        estimator_mode=np.array("ML", dtype=object),
    )
    print(f"[write] {out_path} n_obs_scan={n_obs_scan}", flush=True)


def load_scan_artifact(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as z:
        return dict(
            obs_pix_global_scan=np.asarray(z["obs_pix_global_scan"], dtype=np.int64).copy(),
            c_hat_scan_obs=np.asarray(z["c_hat_scan_obs"], dtype=np.float64).copy(),
            var_diag_scan_obs=np.asarray(z["var_diag_scan_obs"], dtype=np.float64).copy(),
        )
