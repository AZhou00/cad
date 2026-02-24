#!/usr/bin/env python3
"""
Synthesis from per-scan exact point estimates + diagonal covariance approximations.

Input per scan:
  - obs_pix_global_scan: (n_obs_scan,)
  - c_hat_scan_obs: (n_obs_scan,)
  - var_diag_scan_obs: (n_obs_scan,)

Diagonal inverse-variance weighting (pixelwise):
  precision_tot[p] = sum_s 1 / var_s[p]
  rhs[p] = sum_s c_hat_s[p] / var_s[p]
  c_hat[p] = rhs[p] / precision_tot[p]

Array shapes:
  - precision_tot: (n_obs_global,)
  - rhs: (n_obs_global,)
  - c_hat_obs: (n_obs_global,)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
CAD_SRC = BASE.parent / "src"
if str(CAD_SRC) not in sys.path:
    sys.path.insert(0, str(CAD_SRC))

from global_layout import load_layout, GlobalLayout


def load_scan_artifact(npz_path: pathlib.Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as z:
        return dict(
            obs_pix_global_scan=np.asarray(z["obs_pix_global_scan"], dtype=np.int64).copy(),
            c_hat_scan_obs=np.asarray(z["c_hat_scan_obs"], dtype=np.float64).copy(),
            var_diag_scan_obs=np.asarray(z["var_diag_scan_obs"], dtype=np.float64).copy(),
        )


def run_synthesis(
    layout: GlobalLayout,
    scan_npz_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """
    Pixelwise diagonal inverse-variance synthesis on the global observed basis.
    """
    n_obs = layout.n_obs
    global_to_obs = layout.global_to_obs
    precision_tot = np.zeros((n_obs,), dtype=np.float64)
    rhs = np.zeros((n_obs,), dtype=np.float64)

    for scan_index in range(layout.n_scans):
        npz_path = scan_npz_dir / f"scan_{scan_index:04d}_ml.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing scan artifact: {npz_path}")
        art = load_scan_artifact(npz_path)
        obs_s = art["obs_pix_global_scan"]
        c_s = art["c_hat_scan_obs"]
        var_s = art["var_diag_scan_obs"]
        obs_idx = np.asarray(global_to_obs[obs_s], dtype=np.int64)
        valid = obs_idx >= 0
        obs_idx = obs_idx[valid]
        if obs_idx.size == 0:
            continue
        var_valid = np.asarray(var_s[valid], dtype=np.float64)
        if not np.all(np.isfinite(var_valid)) or np.any(var_valid <= 0.0):
            raise RuntimeError(f"Invalid diagonal variances in {npz_path}")
        prec = 1.0 / var_valid
        precision_tot[obs_idx] += prec
        rhs[obs_idx] += prec * np.asarray(c_s[valid], dtype=np.float64)

    bad = precision_tot <= 0.0
    c_hat_obs = np.zeros((n_obs,), dtype=np.float64)
    good = ~bad
    c_hat_obs[good] = rhs[good] / precision_tot[good]
    c_hat_obs = np.asarray(c_hat_obs, dtype=np.float64)
    c_hat_obs -= float(np.mean(c_hat_obs))
    var_diag_total = np.full((n_obs,), np.inf, dtype=np.float64)
    var_diag_total[good] = 1.0 / precision_tot[good]

    n_pix = layout.n_pix
    c_hat_full = np.zeros((n_pix,), dtype=np.float64)
    c_hat_full[layout.obs_pix_global] = c_hat_obs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        estimator_mode=np.array("ML", dtype=object),
        bbox_ix0=np.int64(layout.bbox_ix0),
        bbox_iy0=np.int64(layout.bbox_iy0),
        nx=np.int64(layout.nx),
        ny=np.int64(layout.ny),
        obs_pix_global=layout.obs_pix_global,
        c_hat_full_mk=c_hat_full,
        c_hat_obs=c_hat_obs,
        precision_diag_total=precision_tot,
        var_diag_total=var_diag_total,
        zero_precision_mask=bad,
        n_scans=np.int64(layout.n_scans),
    )
    print(f"[write] {out_path} n_obs={n_obs}", flush=True)


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 3:
        print(
            "Usage: run_synthesis.py <layout.npz> <scan_npz_dir> <out_combined.npz>",
            file=sys.stderr,
        )
        sys.exit(1)
    layout_path = pathlib.Path(argv[0])
    scan_npz_dir = pathlib.Path(argv[1])
    out_path = pathlib.Path(argv[2])

    layout = load_layout(layout_path)
    run_synthesis(layout, scan_npz_dir, out_path)


if __name__ == "__main__":
    main()
