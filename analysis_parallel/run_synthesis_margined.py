#!/usr/bin/env python3
"""
Exact global synthesis on an inner footprint (10% margin removed on each side).

Same accumulation/solve as run_synthesis.py, but all computations are performed on a reduced
bbox where edge pixels are dropped from every scan before summation:
  - map solve
  - precision/covariance diagnostics
  - uncertain eigenmodes

Writes a single recon_combined_ml_margined.npz (see synthesize_scan module docstring for NPZ schema).

CLI
---
  python run_synthesis_margined.py [observation_ids]

  observation_ids: optional, comma-separated list (no spaces). If omitted, uses every numeric
  observation directory under OUT_BASE/FIELD_ID/ that is reconstruction-complete:
  len(binned_tod_10arcmin/*.npz) == len(scans/scan_*_ml.npz) with non-empty binned (same rule as
  run_reconstruction_field.py "ok" rows).
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
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
OUT_SUBDIR_MULTI = "synthesized"
OUT_FILENAME = "recon_combined_ml_margined.npz"
MARGIN_FRAC = 0.10
N_UNCERTAIN_MODES = 400
LANCZOS_OVERSAMPLE = 128
LANCZOS_MAXITER = 2048
# Heuristic: lanczos_maxiter >= 2 * N_UNCERTAIN_MODES; oversample ~ N_UNCERTAIN_MODES/16..N_UNCERTAIN_MODES/4.


def _count_binned_npz(obs_dir: pathlib.Path) -> int | None:
    binned = obs_dir / "binned_tod_10arcmin"
    if not binned.is_dir():
        return None
    return sum(1 for _ in binned.glob("*.npz"))


def _count_scan_ml_npz(obs_dir: pathlib.Path) -> int:
    scans = obs_dir / "scans"
    if not scans.is_dir():
        return 0
    return sum(1 for _ in scans.glob("scan_*_ml.npz"))


def discover_processed_observation_ids(field_root: pathlib.Path) -> list[str]:
    """Obs ids under field_root with non-empty binned TOD and matching scan_*_ml count."""
    if not field_root.is_dir():
        return []
    out: list[str] = []
    subdirs = [p for p in field_root.iterdir() if p.is_dir() and p.name.isdigit()]
    subdirs.sort(key=lambda p: int(p.name))
    for obs_dir in subdirs:
        n_b = _count_binned_npz(obs_dir)
        n_s = _count_scan_ml_npz(obs_dir)
        if n_b is None or n_b == 0:
            continue
        if n_s == n_b:
            out.append(obs_dir.name)
    return out


def main() -> None:
    if len(sys.argv) >= 2:
        observation_ids = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
    else:
        observation_ids = discover_processed_observation_ids(OUT_BASE / FIELD_ID)
    if not observation_ids:
        print(
            f"No processed observations found under {OUT_BASE / FIELD_ID} "
            "(need binned_tod_10arcmin/*.npz count == scans/scan_*_ml.npz, non-empty binned).",
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
