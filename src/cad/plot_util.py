"""
Shared plotting utilities used by multiple analysis drivers.

Only non-trivial helpers that are duplicated across scripts belong here; keep
figure-specific layout in each driver.
"""

from __future__ import annotations

import numpy as np


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
