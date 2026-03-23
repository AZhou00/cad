"""
I/O helpers for per-scan artifacts written by parallel reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Required keys in synthesis NPZ written by synthesize_scan.run_synthesis / run_synthesis_multi_obs.
SYNTHESIS_NPZ_REQUIRED_KEYS: tuple[str, ...] = (
    "bbox_ix0",
    "bbox_iy0",
    "nx",
    "ny",
    "pixel_size_deg",
    "c_hat_full_mk",
    "c_hat_obs",
    "obs_pix_global",
    "cov_inv_tot",
    "good_mask",
    "uncertain_mode_vectors",
    "uncertain_mode_variances",
    "n_uncertain_modes_stored",
    "lanczos_n_modes",
    "scan_metadata",
)


def assert_synthesis_npz_keys(z: Any) -> None:
    """Raise KeyError if z is missing any synthesis artifact key (fail loud for API drift)."""
    missing = [k for k in SYNTHESIS_NPZ_REQUIRED_KEYS if k not in z.files]
    if missing:
        raise KeyError(
            "Synthesis NPZ missing required keys: "
            f"{missing}. Expected a single-file output from run_synthesis / run_synthesis_multi_obs."
        )


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
