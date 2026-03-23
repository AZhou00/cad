#!/usr/bin/env python3
"""
Exact global synthesis on an inner footprint (10% margin removed on each side).

Same accumulation/solve as run_synthesis.py, but all computations are performed on a reduced
bbox where edge pixels are dropped from every scan before summation:
  - map solve
  - precision/covariance diagnostics
  - uncertain eigenmodes

Writes recon_combined_ml_margined_<k>modes.npz per k in UNCERTAIN_MODE_VARIANTS (same schema; plot_reconstruction.py).

CLI
---
  python run_synthesis_margined.py [observation_ids]

  observation_ids: optional, comma-separated list (no spaces). If omitted, uses OBSERVATION_IDS below.
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
OBSERVATION_IDS = ["101706388", "101715260", "101724132", "101931619"]
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
OUT_SUBDIR_MULTI = "synthesized"
OUT_FILENAME = "recon_combined_ml_margined.npz"
MARGIN_FRAC = 0.10
# Lanczos rank = max(UNCERTAIN_MODE_VARIANTS); one npz per k with *_k*modes.npz.
UNCERTAIN_MODE_VARIANTS = [10, 50, 100, 200, 300, 400]
LANCZOS_OVERSAMPLE = 128
LANCZOS_MAXITER = 2048
# Heuristic: Lanczos rank = max(UNCERTAIN_MODE_VARIANTS); lanczos_maxiter >= 2*max(k); oversample ~ max(k)/16..max(k)/4.


def main() -> None:
    observation_ids = OBSERVATION_IDS if len(sys.argv) < 2 else [s.strip() for s in sys.argv[1].split(",") if s.strip()]
    if not observation_ids:
        observation_ids = OBSERVATION_IDS
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
            uncertain_mode_variants=UNCERTAIN_MODE_VARIANTS,
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
            uncertain_mode_variants=UNCERTAIN_MODE_VARIANTS,
            lanczos_oversample=LANCZOS_OVERSAMPLE,
            lanczos_maxiter=LANCZOS_MAXITER,
            out_subdir=OUT_SUBDIR_MULTI,
            out_filename=OUT_FILENAME,
            margin_frac=MARGIN_FRAC,
        )


if __name__ == "__main__":
    main()
