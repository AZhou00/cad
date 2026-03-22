"""
I/O helpers for per-scan artifacts written by parallel reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_scan_artifact(npz_path: Path) -> dict:
    """
    Load per-scan npz written by run_one_scan.

    Returns keys: obs_pix_global_scan, c_hat_scan_obs, cov_inv, Pt_Ninv_d,
    wind_deg_per_s, wind_sigma_x_deg_per_s, wind_sigma_y_deg_per_s, ell_atm, cl_atm_mk2.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        out = dict(
            obs_pix_global_scan=np.asarray(z["obs_pix_global_scan"], dtype=np.int64).copy(),
            c_hat_scan_obs=np.asarray(z["c_hat_scan_obs"], dtype=np.float64).copy(),
            cov_inv=np.asarray(z["cov_inv"], dtype=np.float64).copy(),
            Pt_Ninv_d=np.asarray(z["Pt_Ninv_d"], dtype=np.float64).copy(),
            wind_deg_per_s=np.asarray(z["wind_deg_per_s"], dtype=np.float64).copy(),
            wind_sigma_x_deg_per_s=float(z["wind_sigma_x_deg_per_s"]),
            wind_sigma_y_deg_per_s=float(z["wind_sigma_y_deg_per_s"]),
            ell_atm=np.asarray(z["ell_atm"], dtype=np.float64).copy(),
            cl_atm_mk2=np.asarray(z["cl_atm_mk2"], dtype=np.float64).copy(),
        )
        return out
