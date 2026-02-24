"""
Deterministic scan layout and global coordinate mapping for analysis_parallel.

Strict global coordinate: single CMB bbox (union over all scans) with pixel index
pix = iy + ix*ny in [0, nx*ny). Observed set is the union of hit pixels across scans
above min_hits_per_pix. global_to_obs[pix] = obs_index in [0, n_obs) or -1.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cad import map as map_util
from cad import util


def discover_fields(dataset_dir: Path) -> list[tuple[str, Path]]:
    """Return [(field_id, field_input_dir)]. Subdirs named with digits are obs ids."""
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
    """Return sorted scan NPZ paths for a field directory."""
    binned_dirs = sorted(
        [p for p in field_dir.iterdir() if p.is_dir() and p.name.startswith("binned_tod_")]
    )
    if not binned_dirs:
        return []
    chosen = next((p for p in binned_dirs if p.name == prefer_binned_subdir), binned_dirs[0])
    scan_paths = sorted(
        [p for p in chosen.iterdir() if p.is_file() and p.suffix == ".npz" and not p.name.startswith(".")]
    )
    if max_scans is not None and max_scans > 0:
        scan_paths = scan_paths[:max_scans]
    return scan_paths


def load_scan_for_layout(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pix_index and finite mask only for bbox + hit counting."""
    with np.load(npz_path, allow_pickle=False) as z:
        pix_index = np.asarray(z["pix_index"], dtype=np.int64)
        tod = np.asarray(z["eff_tod_mk"])
    valid = np.isfinite(tod)
    return pix_index, valid


@dataclass(frozen=True)
class GlobalLayout:
    """
    Single source of truth for global CMB grid and observed-pixel set.
    """

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
    """
    Build deterministic global layout from scan paths.
    """
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
