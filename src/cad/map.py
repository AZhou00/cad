"""
Naive mapmaking utilities for binned TOD.

We make a simple coadded map:
  sum[p] += tod[t,e]
  cnt[p] += 1
  map[p] = sum[p]/cnt[p] for cnt>0, else NaN.

Units:
  - input TOD is in mK, and output maps are in mK.

Shapes:
  - eff_tod_mk: (n_t, n_eff)
  - pix_index: (n_t, n_eff, 2) int, columns (ix, iy) in global pixel indices
  - output map_2d_mk: (ny, nx) where nx = ix1-ix0+1, ny = iy1-iy0+1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Inclusive bbox in global pixel indices."""

    ix0: int
    ix1: int
    iy0: int
    iy1: int

    @property
    def nx(self) -> int:
        return int(self.ix1 - self.ix0 + 1)

    @property
    def ny(self) -> int:
        return int(self.iy1 - self.iy0 + 1)


def scan_bbox_from_pix_index(*, pix_index: np.ndarray, valid_mask: np.ndarray) -> BBox:
    """
    Compute bbox from a single scan's pix_index and a validity mask.

    Args:
      pix_index: (n_t, n_eff, 2) int64.
      valid_mask: (n_t, n_eff) bool.
    """
    ij = np.asarray(pix_index, dtype=np.int64)[np.asarray(valid_mask, dtype=bool)]
    if ij.size == 0:
        raise ValueError("No valid samples for bbox.")
    return BBox(
        ix0=int(np.min(ij[:, 0])),
        ix1=int(np.max(ij[:, 0])),
        iy0=int(np.min(ij[:, 1])),
        iy1=int(np.max(ij[:, 1])),
    )


def bbox_union(boxes: list[BBox]) -> BBox:
    if not boxes:
        raise ValueError("boxes must be non-empty.")
    return BBox(
        ix0=min(b.ix0 for b in boxes),
        ix1=max(b.ix1 for b in boxes),
        iy0=min(b.iy0 for b in boxes),
        iy1=max(b.iy1 for b in boxes),
    )


def coadd_map(
    *,
    eff_tod_mk: np.ndarray,
    pix_index: np.ndarray,
    bbox: BBox,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coadd one scan into a (ny,nx) map over a given bbox.

    Args:
      eff_tod_mk: (n_t, n_eff) float, mK.
      pix_index: (n_t, n_eff, 2) int, global (ix,iy).
      bbox: map bbox (global indices).

    Returns:
      map_2d_mk: (ny,nx) float32, mK, NaN for unhit pixels.
      hit_2d: (ny,nx) int64, hit counts.
    """
    eff_tod_mk = np.asarray(eff_tod_mk)
    pix_index = np.asarray(pix_index, dtype=np.int64)
    if pix_index.shape[:2] != eff_tod_mk.shape or pix_index.shape[-1] != 2:
        raise ValueError("pix_index must have shape (n_t,n_eff,2) matching eff_tod_mk.")

    ok = np.isfinite(eff_tod_mk)
    s = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.float64)
    c = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.int64)

    if bool(np.any(ok)):
        ij = pix_index[ok]  # (n_hit, 2)
        ix = ij[:, 0] - int(bbox.ix0)
        iy = ij[:, 1] - int(bbox.iy0)
        v = eff_tod_mk[ok].astype(np.float64, copy=False)
        np.add.at(s, (iy, ix), v)
        np.add.at(c, (iy, ix), 1)

    m = np.full((int(bbox.ny), int(bbox.nx)), np.nan, dtype=np.float32)
    hit = c > 0
    m[hit] = (s[hit] / c[hit]).astype(np.float32)
    return m, c


def coadd_map_global(
    *,
    scans_eff_tod_mk: list[np.ndarray],
    scans_pix_index: list[np.ndarray],
    bbox: BBox,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coadd multiple scans into a single (ny,nx) map over a common bbox.

    Returns:
      map_2d_mk: (ny,nx) float32
      hit_2d: (ny,nx) int64
    """
    if len(scans_eff_tod_mk) != len(scans_pix_index):
        raise ValueError("scans_eff_tod_mk and scans_pix_index must have same length.")
    if not scans_eff_tod_mk:
        raise ValueError("Must provide at least one scan.")

    s = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.float64)
    c = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.int64)

    for eff_tod_mk, pix_index in zip(scans_eff_tod_mk, scans_pix_index, strict=True):
        eff_tod_mk = np.asarray(eff_tod_mk)
        pix_index = np.asarray(pix_index, dtype=np.int64)
        ok = np.isfinite(eff_tod_mk)
        if not bool(np.any(ok)):
            continue
        ij = pix_index[ok]
        ix = ij[:, 0] - int(bbox.ix0)
        iy = ij[:, 1] - int(bbox.iy0)
        v = eff_tod_mk[ok].astype(np.float64, copy=False)
        np.add.at(s, (iy, ix), v)
        np.add.at(c, (iy, ix), 1)

    m = np.full((int(bbox.ny), int(bbox.nx)), np.nan, dtype=np.float32)
    hit = c > 0
    m[hit] = (s[hit] / c[hit]).astype(np.float32)
    return m, c

