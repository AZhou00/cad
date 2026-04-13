#!/usr/bin/env python3
"""
Exact global synthesis from per-scan cov_inv and Pt_Ninv_d. Thin CLI around parallel_solve.

Accumulates cov_inv_tot and Pt_Ninv_d_tot from per-scan npzs, solves cov_inv_tot @ c_hat = Pt_Ninv_d_tot,
embeds to full CMB grid, writes combined map and scan_metadata. Run after run_reconstruction.py.

CLI
---
  python run_synthesis_full.py [observation_ids]

  observation_ids: optional, comma-separated list (no spaces). If omitted, discovers every
  reconstruction-complete observation (binned npz count == scan_*_ml count, non-empty binned, layout.npz).
  Multi-observation synthesis remaps each layout.npz into a union plate grid (common pixel_size_deg
  required); identical layouts use the same arithmetic as the pre-unification code path.

  Examples:
    python run_synthesis_full.py
    python run_synthesis_full.py 101706388
    python run_synthesis_full.py 101706388,101715260

Input (per observation)
-----------------------
  OUT_BASE / FIELD_ID / <obs_id> / layout.npz
  OUT_BASE / FIELD_ID / <obs_id> / scans / scan_0000_ml.npz, scan_0001_ml.npz, ...

Output tree
-----------
  Single NPZ at OUT_FILENAME (e.g. recon_combined_ml_full.npz) under OUT_BASE/FIELD_ID/<obs_id>/ or
  .../synthesized/ for multi-obs. See cad.parallel_solve.synthesize_scan module docstring for NPZ schema.
  scan_metadata: list of per-scan dicts (observation_id, scan_index, wind_deg_per_s, wind_sigma_*, ell_atm, cl_atm_mk2).
"""

from __future__ import annotations

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

from cad.parallel_solve import (
    discover_synthesis_ready_observation_ids,
    load_layout,
    run_synthesis,
    run_synthesis_multi_obs,
)

FIELD_ID = "ra0hdec-59.75"
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
OUT_SUBDIR_MULTI = "synthesized"
OUT_FILENAME = "recon_combined_ml_full.npz"
MARGIN_FRAC = 0.0
N_UNCERTAIN_MODES = 4096
LANCZOS_OVERSAMPLE = 256
LANCZOS_MAXITER = 8192
# Heuristic: lanczos_maxiter >= 2 * N_UNCERTAIN_MODES; oversample ~ N_UNCERTAIN_MODES/16..N_UNCERTAIN_MODES/4.


def main() -> None:
    if len(sys.argv) >= 2:
        observation_ids = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
    else:
        observation_ids = discover_synthesis_ready_observation_ids(OUT_BASE / FIELD_ID)
    if not observation_ids:
        print(
            f"No synthesis-ready observations under {OUT_BASE / FIELD_ID}: need reconstruction-complete "
            "directories (non-empty binned_tod_10arcmin, matching scan_*_ml count) with layout.npz.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(observation_ids) == 1:
        obs_id = observation_ids[0]
        layout_path = OUT_BASE / FIELD_ID / obs_id / "layout.npz"
        scan_dir = OUT_BASE / FIELD_ID / obs_id / "scans"
        out_path = OUT_BASE / FIELD_ID / obs_id / OUT_FILENAME
        layout = load_layout(layout_path)
        run_synthesis(
            layout,
            scan_dir,
            out_path,
            n_uncertain_modes=N_UNCERTAIN_MODES,
            lanczos_oversample=LANCZOS_OVERSAMPLE,
            lanczos_maxiter=LANCZOS_MAXITER,
            observation_id=obs_id,
            margin_frac=MARGIN_FRAC,
        )
    else:
        run_synthesis_multi_obs(
            OUT_BASE,
            FIELD_ID,
            observation_ids,
            n_uncertain_modes=N_UNCERTAIN_MODES,
            lanczos_oversample=LANCZOS_OVERSAMPLE,
            lanczos_maxiter=LANCZOS_MAXITER,
            out_subdir=OUT_SUBDIR_MULTI,
            out_filename=OUT_FILENAME,
            margin_frac=MARGIN_FRAC,
        )


if __name__ == "__main__":
    main()
