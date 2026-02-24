"""
Parallel solve: single-scan ML reconstruction (run_one_scan) and scan artifact loading.

Per-scan: CPU solve_single_scan then build of cov_inv, Pt_Ninv_d; c_hat_scan_obs from the normal equation.
Single npz per scan with point estimate, cov_inv, Pt_Ninv_d, and metadata. Requires JAX/GPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import cad
from cad import dataset_io
from cad import map as map_util
from cad.map import BBox
from cad import power
from cad import reconstruct_scan as cad_reconstruct_scan
from cad import util

from .fisher import build_scan_information
from .layout import GlobalLayout


def run_one_scan(
    layout: GlobalLayout,
    scan_index: int,
    out_dir: Path,
    *,
    n_ell_bins: int = 128,
    cl_floor_mk2: float = 1e-12,
    noise_per_raw_detector_per_153hz_sample_mk: float = 10.0,
    cg_tol: float = 1e-3,
    cg_maxiter: int = 512,
    timings: dict | None = None,
    cg_callback: Optional[Callable[[np.ndarray], None]] = None,
    return_sol: bool = False,
) -> None | dict:
    """If timings is not None, fill with stage keys. If return_sol is True, return dict(sol, prior_atm, ...) after CPU solve (no Fisher/write)."""
    t0 = time.perf_counter()
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
    if timings is not None:
        timings["setup"] = time.perf_counter() - t0

    t_solve = time.perf_counter()
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
        cg_callback=cg_callback,
    )
    if timings is not None:
        timings["solve_single_scan"] = time.perf_counter() - t_solve
    if return_sol:
        return dict(sol=sol, prior_atm=prior_atm, layout=layout, s=s, pixel_size_deg=pixel_size_deg)

    obs_all = np.asarray(sol.pix_obs_local, dtype=np.int64)
    obs_idx_scan = np.unique(obs_all)
    n_obs_scan = int(obs_idx_scan.size)
    obs_pix_global_scan = np.asarray(layout.obs_pix_global, dtype=np.int64)[obs_idx_scan]
    pix_obs_local_scan = np.searchsorted(obs_idx_scan, obs_all)
    if not np.all(obs_idx_scan[pix_obs_local_scan] == obs_all):
        raise RuntimeError("Internal error: scan-local observed index mapping failed.")

    n_pix_atm = int(prior_atm.nx * prior_atm.ny)
    nx, ny = int(prior_atm.nx), int(prior_atm.ny)
    cl_per_mode = np.asarray(prior_atm._cl_per_mode(), dtype=np.float64)
    dxdy = float(prior_atm.pixel_res_rad) ** 2 * float(prior_atm.cos_dec)
    inv_var = np.asarray(sol.inv_var, dtype=np.float64)
    reg_eps = 1e-10 * (float(np.mean(inv_var)) * 4.0 + 1e-12)
    idx4 = np.asarray(sol.idx4, dtype=np.int32)
    w4 = np.asarray(sol.w4, dtype=np.float64)
    diag_WtNW = np.bincount(
        idx4.reshape(-1),
        weights=(w4 * w4 * inv_var[:, None]).reshape(-1),
        minlength=n_pix_atm,
    ).astype(np.float64)
    e0 = np.zeros((n_pix_atm,), dtype=np.float64)
    e0[0] = 1.0
    diag_Ca_inv_0 = float(prior_atm.apply_Cinv(e0)[0])
    diag_M = np.maximum(diag_WtNW + diag_Ca_inv_0 + reg_eps, 1e-14)

    t_fisher = time.perf_counter()
    cov_inv_s, Pt_Ninv_d_s, c_hat_scan_obs = build_scan_information(
        d=np.asarray(sol.tod_valid_mk, dtype=np.float64),
        inv_var=inv_var,
        pix_obs_local=pix_obs_local_scan,
        idx4=idx4,
        w4=w4,
        nx=nx,
        ny=ny,
        cl_per_mode=cl_per_mode,
        dxdy=dxdy,
        diag_M=diag_M,
        n_obs_scan=n_obs_scan,
        n_pix_atm=n_pix_atm,
        reg_eps=reg_eps,
        cg_niter=cg_maxiter,
    )
    if timings is not None:
        timings["build_scan_information"] = time.perf_counter() - t_fisher

    out_dir.mkdir(parents=True, exist_ok=True)
    t_write = time.perf_counter()
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
        cov_inv=cov_inv_s,
        Pt_Ninv_d=Pt_Ninv_d_s,
        pixel_size_deg=np.float64(pixel_size_deg),
        wind_deg_per_s=np.array(wind_deg_per_s, dtype=np.float64),
        n_obs_scan=np.int64(n_obs_scan),
        estimator_mode=np.array("ML", dtype=object),
    )
    if timings is not None:
        timings["write"] = time.perf_counter() - t_write
    print(f"[write] {out_path} n_obs_scan={n_obs_scan}", flush=True)


def load_scan_artifact(npz_path: Path) -> dict:
    """Load per-scan npz: obs_pix_global_scan, c_hat_scan_obs, cov_inv, Pt_Ninv_d (and optional metadata)."""
    with np.load(npz_path, allow_pickle=True) as z:
        out = dict(
            obs_pix_global_scan=np.asarray(z["obs_pix_global_scan"], dtype=np.int64).copy(),
            c_hat_scan_obs=np.asarray(z["c_hat_scan_obs"], dtype=np.float64).copy(),
        )
        if "cov_inv" in z:
            out["cov_inv"] = np.asarray(z["cov_inv"], dtype=np.float64).copy()
            out["Pt_Ninv_d"] = np.asarray(z["Pt_Ninv_d"], dtype=np.float64).copy()
        else:
            out["cov_inv"] = np.asarray(z["F_s"], dtype=np.float64).copy()
            out["Pt_Ninv_d"] = np.asarray(z["b_s"], dtype=np.float64).copy()
        if "var_diag_scan_obs" in z:
            out["var_diag_scan_obs"] = np.asarray(z["var_diag_scan_obs"], dtype=np.float64).copy()
        return out
