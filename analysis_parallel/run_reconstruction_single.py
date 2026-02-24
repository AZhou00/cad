#!/usr/bin/env python3
"""
Single-scan ML reconstruction: exact point estimate plus diagonal covariance approximation.

No multiprocessing: run one process per scan (e.g. SLURM array task).

Usage:
  python run_reconstruction_single.py <layout.npz> <out_dir> [scan_index] [n_ell_bins] [cl_floor_mk2] [noise_mk] [cg_tol] [cg_maxiter]

Math (ML per scan s):
  d_s = P_s c + W_s a_s^0 + n_s
  c_hat_s is solved exactly from the joint normal equations with CG in
  cad.reconstruct_scan.solve_single_scan.

Diagonal covariance approximation on scan-observed pixels:
  F_s = P_s^T Ntilde_s^{-1} P_s
  Approximate:
    diag(F_s)_j ~= sum_{i in samples hitting pixel j} 1/sigma_i^2
  therefore:
    var_diag_s[j] ~= 1 / diag(F_s)_j

Array shapes:
  - obs_pix_global_scan: (n_obs_scan,)
  - c_hat_scan_obs: (n_obs_scan,)
  - precision_diag_scan_obs: (n_obs_scan,)
  - var_diag_scan_obs: (n_obs_scan,)
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

# cad and analysis_parallel
BASE = pathlib.Path(__file__).resolve().parent
CAD_SRC = BASE.parent / "src"
if str(CAD_SRC) not in sys.path:
    sys.path.insert(0, str(CAD_SRC))

import cad
from cad import map as map_util
from cad.map import BBox
from cad import power
from cad import reconstruct_scan
from cad import util

from global_layout import load_layout, GlobalLayout


def _load_scan(npz_path: pathlib.Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        return dict(
            eff_tod_mk=np.asarray(z["eff_tod_mk"], dtype=np.float32),
            pix_index=np.asarray(z["pix_index"], dtype=np.int64),
            t_s=np.asarray(z["t_bin_center_s"], dtype=np.float64),
            pixel_size_deg=float(z["pixel_size_deg"]),
            eff_pos_deg=np.asarray(z["eff_pos_deg"], dtype=np.float32),
            boresight_pos_deg=np.asarray(z["boresight_pos_deg"], dtype=np.float32),
            eff_counts=np.asarray(z["eff_counts"], dtype=np.float64),
            bin_sec=float(z["bin_sec"]),
            sample_rate_hz=float(z["sample_rate_hz"]),
        )


def run_one_scan(
    layout: GlobalLayout,
    scan_index: int,
    out_dir: pathlib.Path,
    *,
    n_ell_bins: int = 128,
    cl_floor_mk2: float = 1e-12,
    noise_per_raw_detector_per_153hz_sample_mk: float = 10.0,
    cg_tol: float = 5e-4,
    cg_maxiter: int = 1200,
) -> None:
    scan_path = layout.scan_paths[scan_index]
    s0 = _load_scan(scan_path)
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

    pm, vm = util.pointing_from_pix_index(
        pix_index=np.asarray(s["pix_index"], dtype=np.int64),
        tod_mk=np.asarray(s["eff_tod_mk"], dtype=np.float64),
        bbox=bbox_cmb,
    )
    valid_pix = pm[vm].astype(np.int64)
    global_obs_all = np.asarray(layout.global_to_obs, dtype=np.int64)[valid_pix]
    keep = global_obs_all >= 0
    if not np.all(keep):
        vm2 = vm.copy()
        vm2[vm] = keep
        vm = vm2
        valid_pix = pm[vm].astype(np.int64)
        global_obs_all = np.asarray(layout.global_to_obs, dtype=np.int64)[valid_pix]

    # Scan-local observed subset as ordered subset of the global observed set.
    obs_idx_scan = np.unique(global_obs_all)
    n_obs_scan = int(obs_idx_scan.size)
    obs_pix_global_scan = np.asarray(layout.obs_pix_global, dtype=np.int64)[obs_idx_scan]
    _, det_idx = np.where(vm)
    sigma_samp = np.asarray(noise_std, dtype=np.float64)[det_idx]
    inv_var = 1.0 / (sigma_samp * sigma_samp)
    pix_obs_local_scan = np.searchsorted(obs_idx_scan, global_obs_all)
    if not np.all(obs_idx_scan[pix_obs_local_scan] == global_obs_all):
        raise RuntimeError("Internal error: scan-local observed index mapping failed.")

    # Exact ML point estimate from the established joint solver.
    sol = reconstruct_scan.solve_single_scan(
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
    c_hat_scan_obs = c_hat_full[obs_pix_global_scan]

    # Diagonal precision approximation from weighted sample hits per observed pixel.
    precision_diag_scan_obs = np.bincount(
        pix_obs_local_scan,
        weights=inv_var,
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
