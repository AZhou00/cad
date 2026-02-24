#!/usr/bin/env python3
"""
Single-field direct reconstruction (max 8 scans per field). Orchestrates discovery and run_one_scan.
Input: cad/data/<dataset_dir>/  Output: /pscratch/.../<dataset>_recon_direct/
"""

from __future__ import annotations

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent
DATA_DIR = CAD_DIR / "data"
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
MAX_SCANS_DIRECT = 8

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

from cad import dataset_io
from cad.direct_solve import DirectConfig, run_one_scan


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) >= 2 else "ra0hdec-59.75"
    max_scans = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    if max_scans is not None and max_scans > MAX_SCANS_DIRECT:
        max_scans = MAX_SCANS_DIRECT
    dataset_dir = DATA_DIR / dataset
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory does not exist: {dataset_dir}")
    out_root = OUT_BASE / f"{dataset}_recon_direct"
    out_root.mkdir(parents=True, exist_ok=True)

    fields = dataset_io.discover_fields(dataset_dir)
    if len(fields) == 0:
        raise RuntimeError(
            f"No obs-id subdirectories found under {dataset_dir}. "
            "Expected layout: <dataset>/<obs_id>/binned_tod_*/<scan>.npz"
        )

    for estimator_mode in ("ML", "MAP"):
        cfg = DirectConfig(
            dataset_dir=dataset,
            estimator_mode=estimator_mode,
            max_scans=max_scans,
        )
        mode = cfg.estimator_mode.upper()
        for field_id, field_in_dir in fields:
            scan_paths = dataset_io.discover_scan_paths(
                field_in_dir,
                prefer_binned_subdir=cfg.prefer_binned_subdir,
                max_scans=cfg.max_scans,
            )
            if len(scan_paths) == 0:
                print(f"[skip] field {field_id}: no scan NPZs found under {field_in_dir}", flush=True)
                continue
            out_dir = out_root / str(field_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            run_one_scan(
                field_id=str(field_id),
                scan_paths=scan_paths,
                out_dir=out_dir,
                mode=mode,
                cfg=cfg,
            )


if __name__ == "__main__":
    main()
