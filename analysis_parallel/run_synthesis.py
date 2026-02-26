#!/usr/bin/env python3
"""
Exact global synthesis from per-scan cov_inv and Pt_Ninv_d. Thin CLI around parallel_solve.

Accumulates cov_inv_tot and Pt_Ninv_d_tot from per-scan npzs, solves cov_inv_tot @ c_hat = Pt_Ninv_d_tot,
embeds to full CMB grid, writes combined map and scan_metadata. Run after run_reconstruction.py.

CLI
---
  python run_synthesis.py [observation_ids]

  observation_ids: optional, comma-separated list (no spaces). If omitted, uses OBSERVATION_IDS below.
  Examples:
    python run_synthesis.py
    python run_synthesis.py 101706388
    python run_synthesis.py 101706388,101715260

Input (per observation)
-----------------------
  OUT_BASE / FIELD_ID / <obs_id> / layout.npz
  OUT_BASE / FIELD_ID / <obs_id> / scans / scan_0000_ml.npz, scan_0001_ml.npz, ...

Output tree
-----------
  Single observation (one obs_id):
    OUT_BASE / FIELD_ID / <obs_id> / recon_combined_ml.npz

  Multiple observations (comma-separated obs_ids):
    OUT_BASE / FIELD_ID / synthesized / recon_combined_ml.npz

  Both paths write a single npz with the same structure (c_hat_*, cov_inv_tot, scan_metadata, etc.).
  scan_metadata is a list of per-scan dicts (observation_id, scan_index, wind_deg_per_s, wind_sigma_*, ell_atm, cl_atm_mk2).
"""

from __future__ import annotations

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

from cad.parallel_solve import load_layout, run_synthesis, run_synthesis_multi_obs

FIELD_ID = "ra0hdec-59.75"
OBSERVATION_IDS = ["101724132"]
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")


def main() -> None:
    observation_ids = OBSERVATION_IDS if len(sys.argv) < 2 else [s.strip() for s in sys.argv[1].split(",") if s.strip()]
    if not observation_ids:
        observation_ids = OBSERVATION_IDS
    if len(observation_ids) == 1:
        obs_id = observation_ids[0]
        layout = load_layout(OUT_BASE / FIELD_ID / obs_id / "layout.npz")
        run_synthesis(
            layout,
            OUT_BASE / FIELD_ID / obs_id / "scans",
            OUT_BASE / FIELD_ID / obs_id / "recon_combined_ml.npz",
            observation_id=obs_id,
        )
    else:
        run_synthesis_multi_obs(OUT_BASE, FIELD_ID, observation_ids, out_subdir="synthesized")


if __name__ == "__main__":
    main()
