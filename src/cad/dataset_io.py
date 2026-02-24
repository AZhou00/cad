"""
Shared dataset discovery and scan NPZ loading.

Used by both direct_solve and parallel_solve. Input layout:
  <dataset_dir>/<field_id>/binned_tod_*/<scan>.npz
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

def discover_fields(dataset_dir: Path) -> list[tuple[str, Path]]:
    """
    Return [(field_id, field_input_dir)].

    Multi-field convention: subdirectories named with digits are treated as obs ids.
    Otherwise, treat the dataset directory itself as a single field.
    """
    dataset_dir = Path(dataset_dir)
    subdirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    obs = sorted([p for p in subdirs if re.fullmatch(r"\d+", p.name)], key=lambda p: p.name)
    if obs:
        return [(p.name, p) for p in obs]
    return [(dataset_dir.name, dataset_dir)]


def discover_scan_paths(
    field_dir: Path,
    *,
    prefer_binned_subdir: str = "binned_tod_10arcmin",
    max_scans: int | None = None,
) -> list[Path]:
    """
    Return sorted scan NPZ paths for a field directory.

    Looks for binned_tod_*/ subdirectories under field_dir, picks the preferred
    one (or the first), then lists .npz files. If max_scans is set, truncates.
    """
    field_dir = Path(field_dir)
    binned_dirs = sorted(
        [p for p in field_dir.iterdir() if p.is_dir() and p.name.startswith("binned_tod_")]
    )
    if not binned_dirs:
        return []
    chosen = next(
        (p for p in binned_dirs if p.name == prefer_binned_subdir),
        binned_dirs[0],
    )
    scan_paths = sorted(
        [
            p
            for p in chosen.iterdir()
            if p.is_file() and p.suffix == ".npz" and not p.name.startswith(".")
        ]
    )
    if max_scans is not None and max_scans > 0:
        scan_paths = scan_paths[:max_scans]
    return scan_paths


def load_scan(npz_path: Path) -> dict:
    """
    Load full scan NPZ. Single source for direct_solve and parallel_solve.

    Returns dict with: eff_tod_mk (n_time, n_det), pix_index (n_time, n_det, 2),
    t_s (n_time,), pixel_size_deg, eff_pos_deg, boresight_pos_deg, eff_counts,
    eff_offsets_arcmin, bin_sec, sample_rate_hz, focal_*_arcmin, effective_box_arcmin.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        return dict(
            eff_tod_mk=np.asarray(z["eff_tod_mk"], dtype=np.float32),
            pix_index=np.asarray(z["pix_index"], dtype=np.int64),
            t_s=np.asarray(z["t_bin_center_s"], dtype=np.float64),
            pixel_size_deg=float(z["pixel_size_deg"]),
            eff_pos_deg=np.asarray(z["eff_pos_deg"], dtype=np.float32),
            boresight_pos_deg=np.asarray(z["boresight_pos_deg"], dtype=np.float32),
            eff_counts=np.asarray(z["eff_counts"], dtype=np.float64),
            eff_offsets_arcmin=np.asarray(z["eff_offsets_arcmin"], dtype=np.float64),
            bin_sec=float(z["bin_sec"]),
            sample_rate_hz=float(z["sample_rate_hz"]),
            focal_x_min_arcmin=float(z["focal_x_min_arcmin"]),
            focal_x_max_arcmin=float(z["focal_x_max_arcmin"]),
            focal_y_min_arcmin=float(z["focal_y_min_arcmin"]),
            focal_y_max_arcmin=float(z["focal_y_max_arcmin"]),
            effective_box_arcmin=float(z["effective_box_arcmin"]),
        )
