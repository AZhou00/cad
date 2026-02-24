#!/usr/bin/env python3
"""
Build and save the global scan layout for a field.

Usage:
  python build_layout.py <data_dir> <dataset_name> <field_id> <layout_out.npz> [max_scans] [min_hits_per_pix]
"""

from __future__ import annotations

import pathlib
import sys

from global_layout import (
    build_layout,
    discover_fields,
    discover_scan_paths,
    save_layout,
)


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 4:
        print(
            "Usage: build_layout.py <data_dir> <dataset_name> <field_id> <layout_out.npz> [max_scans] [min_hits]",
            file=sys.stderr,
        )
        sys.exit(1)
    data_dir = pathlib.Path(argv[0])
    dataset_name = str(argv[1])
    field_id = str(argv[2])
    layout_out = pathlib.Path(argv[3])
    max_scans = int(argv[4]) if len(argv) > 4 else None
    min_hits_per_pix = int(argv[5]) if len(argv) > 5 else 1

    dataset_dir = data_dir / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    fields = discover_fields(dataset_dir)
    field_dir = next((d for fid, d in fields if fid == field_id), None)
    if field_dir is None:
        raise ValueError(f"Field {field_id} not in {[f[0] for f in fields]}")
    scan_paths = discover_scan_paths(
        field_dir,
        prefer_binned_subdir="binned_tod_10arcmin",
        max_scans=max_scans,
    )
    if not scan_paths:
        raise RuntimeError(f"No scan NPZs under {field_dir}")
    layout = build_layout(
        field_id=field_id,
        scan_paths=scan_paths,
        min_hits_per_pix=min_hits_per_pix,
    )
    save_layout(layout, layout_out)
    print(f"[write] {layout_out} n_scans={layout.n_scans} n_obs={layout.n_obs}", flush=True)


if __name__ == "__main__":
    main()
