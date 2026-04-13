"""
Deterministic scan layout and global coordinate mapping for parallel solve.

Each scan can have a slightly different sky footprint. The layout uses:
- bbox_cmb = union of per-scan bounding boxes (scan_bbox_from_pix_index then bbox_union),
- obs_pix_global = union of hit pixels across all scans above min_hits_per_pix.
So the global grid (nx, ny) and observed-pixel set accommodate all scans; synthesis
accumulates at global indices and scans with smaller footprints contribute only on
their observed subset. Pix convention: pix = iy + ix*ny. global_to_obs[pix] in [0, n_obs) or -1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cad import dataset_io
from cad import map as map_util
from cad import util

discover_fields = dataset_io.discover_fields
discover_scan_paths = dataset_io.discover_scan_paths


def load_scan_for_layout(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pix_index and finite mask only for bbox + hit counting."""
    with np.load(npz_path, allow_pickle=False) as z:
        pix_index = np.asarray(z["pix_index"], dtype=np.int64)
        tod = np.asarray(z["eff_tod_mk"])
    valid = np.isfinite(tod)
    return pix_index, valid


@dataclass(frozen=True)
class GlobalLayout:
    """Single source of truth for global CMB grid and observed-pixel set."""

    bbox_ix0: int
    bbox_iy0: int
    nx: int
    ny: int
    obs_pix_global: np.ndarray
    global_to_obs: np.ndarray
    scan_paths: tuple[Path, ...]
    pixel_size_deg: float
    field_id: str

    @property
    def n_pix(self) -> int:
        return int(self.nx * self.ny)

    @property
    def n_obs(self) -> int:
        return int(self.obs_pix_global.size)

    @property
    def n_scans(self) -> int:
        return len(self.scan_paths)


def build_layout(
    *,
    field_id: str,
    scan_paths: list[Path],
    min_hits_per_pix: int = 1,
) -> GlobalLayout:
    """Build deterministic global layout from scan paths."""
    if not scan_paths:
        raise ValueError("scan_paths must be non-empty")

    pixel_size_deg = None
    boxes = []
    for p in scan_paths:
        pix_index, valid = load_scan_for_layout(p)
        if pixel_size_deg is None:
            with np.load(p, allow_pickle=False) as z:
                pixel_size_deg = float(z["pixel_size_deg"])
        boxes.append(map_util.scan_bbox_from_pix_index(pix_index=pix_index, valid_mask=valid))
    bbox_cmb = map_util.bbox_union(boxes)
    n_pix = int(bbox_cmb.nx * bbox_cmb.ny)

    pointing_mats = []
    valid_masks = []
    for p in scan_paths:
        pix_index, valid = load_scan_for_layout(p)
        pm, vm = util.pointing_from_pix_index(
            pix_index=pix_index,
            tod_mk=np.where(valid, 0.0, np.nan),
            bbox=bbox_cmb,
        )
        pointing_mats.append(pm)
        valid_masks.append(vm)

    obs_pix_global, global_to_obs = util.observed_pixel_index_set(
        pointing_matrices=pointing_mats,
        valid_masks=valid_masks,
        n_pix=n_pix,
        min_hits_per_pix=min_hits_per_pix,
    )
    if obs_pix_global.size == 0:
        raise RuntimeError(f"No observed pixels with min_hits_per_pix={min_hits_per_pix}")

    return GlobalLayout(
        bbox_ix0=int(bbox_cmb.ix0),
        bbox_iy0=int(bbox_cmb.iy0),
        nx=int(bbox_cmb.nx),
        ny=int(bbox_cmb.ny),
        obs_pix_global=np.asarray(obs_pix_global, dtype=np.int64),
        global_to_obs=np.asarray(global_to_obs, dtype=np.int64),
        scan_paths=tuple(scan_paths),
        pixel_size_deg=float(pixel_size_deg),
        field_id=field_id,
    )


def save_layout(layout: GlobalLayout, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        bbox_ix0=np.int64(layout.bbox_ix0),
        bbox_iy0=np.int64(layout.bbox_iy0),
        nx=np.int64(layout.nx),
        ny=np.int64(layout.ny),
        obs_pix_global=layout.obs_pix_global,
        global_to_obs=layout.global_to_obs,
        scan_paths=np.array([str(p) for p in layout.scan_paths], dtype=object),
        pixel_size_deg=np.float64(layout.pixel_size_deg),
        field_id=np.array(layout.field_id, dtype=object),
    )


def load_layout(npz_path: Path) -> GlobalLayout:
    with np.load(npz_path, allow_pickle=True) as z:
        scan_paths = tuple(Path(str(p)) for p in z["scan_paths"])
        field_id = str(z["field_id"].item())
        obs_pix_global = np.asarray(z["obs_pix_global"], dtype=np.int64).copy()
        global_to_obs = np.asarray(z["global_to_obs"], dtype=np.int64).copy()
        bbox_ix0 = int(z["bbox_ix0"])
        bbox_iy0 = int(z["bbox_iy0"])
        nx = int(z["nx"])
        ny = int(z["ny"])
        pixel_size_deg = float(z["pixel_size_deg"])
    return GlobalLayout(
        bbox_ix0=bbox_ix0,
        bbox_iy0=bbox_iy0,
        nx=nx,
        ny=ny,
        obs_pix_global=obs_pix_global,
        global_to_obs=global_to_obs,
        scan_paths=scan_paths,
        pixel_size_deg=pixel_size_deg,
        field_id=field_id,
    )


def cmb_grid_signature(layout: GlobalLayout) -> tuple[int, int, int, int, float]:
    """
    CMB plate grid identity for multi-observation synthesis.

    Flat indices pix = iy + ix*ny are only meaningful when (bbox_ix0, bbox_iy0, nx, ny) and
    pixel_size_deg match across observations.
    """
    return (
        int(layout.bbox_ix0),
        int(layout.bbox_iy0),
        int(layout.nx),
        int(layout.ny),
        float(layout.pixel_size_deg),
    )


def discover_synthesis_ready_observation_ids(field_root: Path) -> list[str]:
    """
    Numeric observation directories under field_root suitable for multi-obs synthesis.

    Requires: non-empty binned_tod_10arcmin/*.npz, same count as scans/scan_*_ml.npz, and
    layout.npz present. Sorted by integer obs id.

    run_synthesis_multi_obs builds a union plate bbox and remaps each observation's local flat
    indices (pix = iy + ix*ny) into that grid; pixel_size_deg must match across observations.
    """
    if not field_root.is_dir():
        return []
    subdirs = sorted(
        (p for p in field_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    out: list[str] = []
    for obs_dir in subdirs:
        binned = obs_dir / "binned_tod_10arcmin"
        scans = obs_dir / "scans"
        layout_path = obs_dir / "layout.npz"
        if not binned.is_dir() or not scans.is_dir() or not layout_path.exists():
            continue
        n_b = sum(1 for _ in binned.glob("*.npz"))
        n_s = sum(1 for _ in scans.glob("scan_*_ml.npz"))
        if n_b == 0 or n_s != n_b:
            continue
        out.append(obs_dir.name)
    return out
