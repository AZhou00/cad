#!/usr/bin/env python3
"""
Direct multi-scan synthesis (max 8 scans per group). Orchestrates discovery, per-field and combined synthesis.
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
from cad.direct_solve import (
    DirectConfig,
    discover_recon_paths,
    merge_preps_for_all_observations,
    prepare_synthesis_inputs,
    run_synthesis_group,
)


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) >= 2 else "ra0hdec-59.75"
    scope = sys.argv[2] if len(sys.argv) >= 3 else "both"
    max_scans = int(sys.argv[3]) if len(sys.argv) >= 4 else None
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
            synthesis_scope=scope,
            max_scans=max_scans,
        )
        mode = cfg.estimator_mode.upper()
        scope_l = cfg.synthesis_scope.lower()

        per_field_preps = []
        for field_id, field_in_dir in fields:
            scan_paths = dataset_io.discover_scan_paths(
                field_in_dir,
                prefer_binned_subdir=cfg.prefer_binned_subdir,
                max_scans=cfg.max_scans,
            )
            if len(scan_paths) == 0:
                print(f"[skip] field {field_id}: no scan NPZs found under {field_in_dir}", flush=True)
                continue
            if len(scan_paths) > MAX_SCANS_DIRECT:
                raise RuntimeError(
                    f"Direct solve allows at most {MAX_SCANS_DIRECT} scans per field; got {len(scan_paths)} for {field_id}."
                )
            recon_dir = out_root / str(field_id)
            recon_paths = discover_recon_paths(recon_dir=recon_dir, n_scans=len(scan_paths), mode=mode)
            prep = prepare_synthesis_inputs(
                label=str(field_id),
                scan_paths=scan_paths,
                recon_paths=recon_paths,
                mode=mode,
                cfg=cfg,
            )
            if scope_l == "both":
                out_path = out_root / str(field_id) / f"recon_combined_{mode.lower()}.npz"
                run_synthesis_group(label=str(field_id), prep=prep, out_path=out_path, mode=mode, cfg=cfg)
            per_field_preps.append(prep)

        if len(per_field_preps) == 0:
            raise RuntimeError(f"No scan NPZ files discovered under dataset: {dataset_dir}")
        prep_all = merge_preps_for_all_observations(preps=per_field_preps, mode=mode)
        out_all = out_root / f"recon_{mode.lower()}.npz"
        run_synthesis_group(label="all_observations", prep=prep_all, out_path=out_all, mode=mode, cfg=cfg)


if __name__ == "__main__":
    main()
