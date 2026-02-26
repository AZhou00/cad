#!/usr/bin/env python3
"""
Build and save global scan layout for a field. Thin CLI around parallel_solve.
Output layout path: OUT_BASE / field_id / observation_id / "layout.npz"
(OUT_BASE = /pscratch/sd/j/junzhez/cmb-atmosphere-data).

How to run (repeat per observation for "all" data):

  python build_layout.py <field_id> <observation_id> [max_scans] [min_hits]
  Example: python build_layout.py ra0hdec-59.75 101706388

Then run_reconstruction.py (builds layout if needed, splits scans across GPUs), then run_synthesis.py.
Observation IDs = numeric subdirs under cad/data/<field_id>/ (e.g. 101706388, 101715260, ...).
"""

from __future__ import annotations

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent
DATA_DIR = CAD_DIR / "data"
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

from cad.parallel_solve import build_layout, discover_fields, discover_scan_paths, save_layout


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 2:
        print("Usage: build_layout.py <field_id> <observation_id> [max_scans] [min_hits]", file=sys.stderr)
        sys.exit(1)
    field_id, observation_id = str(argv[0]), str(argv[1])
    layout_out = OUT_BASE / field_id / observation_id / "layout.npz"
    max_scans = int(argv[2]) if len(argv) > 2 else None
    min_hits_per_pix = int(argv[3]) if len(argv) > 3 else 1

    field_dir = DATA_DIR / field_id
    if not field_dir.exists():
        raise FileNotFoundError(f"Field dir not found: {field_dir}")
    observations = discover_fields(field_dir)
    obs_dir = next((d for oid, d in observations if oid == observation_id), None)
    if obs_dir is None:
        raise ValueError(f"Observation {observation_id} not in {[o[0] for o in observations]}")
    scan_paths = discover_scan_paths(
        obs_dir,
        prefer_binned_subdir="binned_tod_10arcmin",
        max_scans=max_scans,
    )
    if not scan_paths:
        raise RuntimeError(f"No scan NPZs under {obs_dir}")
    layout = build_layout(
        field_id=field_id,
        scan_paths=scan_paths,
        min_hits_per_pix=min_hits_per_pix,
    )
    layout_out.parent.mkdir(parents=True, exist_ok=True)
    save_layout(layout, layout_out)
    print(f"[write] {layout_out} n_scans={layout.n_scans} n_obs={layout.n_obs}", flush=True)


if __name__ == "__main__":
    main()
