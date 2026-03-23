"""
Shared plotting utilities used by multiple analysis drivers.

Map geometry (extent, naive coadd, imshow) lives here so drivers do not import each other.
Depends on numpy, matplotlib; optional cad.map.BBox for typed bbox helpers.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from cad.map import BBox

# Matches dataset_io / build_layout discovery of binned TOD under each observation directory.
BINNED_TOD_SUBDIR = "binned_tod_10arcmin"


def img_from_vec(vec: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    """
    Rasterize a flattened map for imshow.

    Convention: vec[pix] with pix = iy + ix * ny (ix axis-0, iy axis-1 in (nx, ny)).

    Args:
      vec: (n_pix,) with n_pix = nx * ny.
      nx, ny: grid size.

    Returns:
      (ny, nx) float64, row iy, column ix.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(int(nx), int(ny))
    return v.T


def deproject_uncertain_modes(
    vec_obs: np.ndarray,
    good_mask: np.ndarray,
    uncertain_vectors: np.ndarray,
) -> np.ndarray:
    """
    Remove components along columns of uncertain_vectors on good pixels (orthogonal projection off).

    uncertain_vectors columns should be orthonormal on the good-pixel subspace (e.g. Lanczos + QR).

    Args:
      vec_obs: (n_obs,) map coefficients in observed-pixel ordering.
      good_mask: (n_obs,) bool, True where precision is trusted / modes defined.
      uncertain_vectors: (n_good, k) with n_good = sum(good_mask); lives only on good rows.

    Returns:
      (n_obs,) copy of vec_obs with good-pixel part projected off span(uncertain_vectors).
    """
    out = np.asarray(vec_obs, dtype=np.float64).copy()
    good = np.asarray(good_mask, dtype=bool)
    v = np.asarray(uncertain_vectors, dtype=np.float64)
    if v.size == 0:
        return out
    vec_good = out[good]
    vec_good -= v @ (v.T @ vec_good)
    out[good] = vec_good
    return out


def extent_deg_from_bbox(*, bbox: BBox, pixel_size_deg: float) -> list[float]:
    """RA/Dec extent in degrees for imshow: [x0, x1, y0, y1] with pixel centers convention."""
    x0 = float(bbox.ix0) * float(pixel_size_deg)
    x1 = float(bbox.ix0 + bbox.nx) * float(pixel_size_deg)
    y0 = float(bbox.iy0) * float(pixel_size_deg)
    y1 = float(bbox.iy0 + bbox.ny) * float(pixel_size_deg)
    return [x0, x1, y0, y1]


def robust_vmin_vmax(
    x: np.ndarray, *, p_lo: float = 2.0, p_hi: float = 98.0, default: tuple[float, float] = (-1.0, 1.0)
) -> tuple[float, float]:
    """Percentile-based symmetric color scale from finite values."""
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(default[0]), float(default[1])
    lo, hi = np.percentile(v, [float(p_lo), float(p_hi)])
    return float(lo), float(hi)


def imshow_ra_dec_map(ax, img, *, extent, title: str, vmin=None, vmax=None, cmap: str = "RdBu_r"):
    """Plate-carree map with RA/Dec labels; invalid masked white."""
    img = np.ma.masked_invalid(np.asarray(img))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=(1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor("white")
    im = ax.imshow(img, origin="lower", extent=extent, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax, interpolation="none")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    return im


def add_shared_colorbar(
    fig,
    axes,
    im,
    *,
    label: str,
    pad: float = 0.012,
    width: float = 0.018,
) -> None:
    """One vertical colorbar aligned to span all provided axes."""
    axs = [ax for ax in np.asarray(axes).reshape(-1) if ax.get_visible()]
    if not axs:
        return
    boxes = [ax.get_position() for ax in axs]
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    cax = fig.add_axes([x1 + pad, y0, width, y1 - y0])
    fig.colorbar(im, cax=cax, orientation="vertical").set_label(label)


def binned_tod_paths(obs_data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Sorted list of binned scan NPZ paths under obs_data_dir/binned_tod_10arcmin/."""
    chosen = obs_data_dir / BINNED_TOD_SUBDIR
    if not chosen.is_dir():
        return []
    return sorted([p for p in chosen.iterdir() if p.is_file() and p.suffix == ".npz" and not p.name.startswith(".")])


def naive_coadd(scan_paths: list[pathlib.Path], bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixel-wise mean coadd of binned TOD maps in bbox (naive, no deprojection).

    Returns:
      naive: (ny, nx) float32
      hit: (ny, nx) bool
    """
    s = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.float64)
    c = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.int64)
    for p in scan_paths:
        with np.load(p, allow_pickle=False) as z:
            eff_tod_mk = np.asarray(z["eff_tod_mk"])
            pix_index = np.asarray(z["pix_index"], dtype=np.int64)
        ok = np.isfinite(eff_tod_mk)
        if not np.any(ok):
            continue
        ij = pix_index[ok]
        ixg = ij[:, 0] - int(bbox.ix0)
        iyg = ij[:, 1] - int(bbox.iy0)
        in_box = (ixg >= 0) & (ixg < int(bbox.nx)) & (iyg >= 0) & (iyg < int(bbox.ny))
        if not np.any(in_box):
            continue
        v = eff_tod_mk[ok].astype(np.float64, copy=False)[in_box]
        np.add.at(s, (iyg[in_box], ixg[in_box]), v)
        np.add.at(c, (iyg[in_box], ixg[in_box]), 1)
    naive = np.full((int(bbox.ny), int(bbox.nx)), np.nan, dtype=np.float32)
    hit = c > 0
    naive[hit] = (s[hit] / c[hit]).astype(np.float32)
    return naive, hit
