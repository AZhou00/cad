"""
FFT-diagonal stationary Gaussian priors on a regular pixel grid.

This file contains the prior/operator pieces only. Inference code should treat
these as linear operators via `apply_Cinv` (and optionally `apply_C`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ell_bin_indices_rfft2(*, nx: int, ny: int, pixel_res_rad: float, cos_dec: float, n_ell_bins: int) -> np.ndarray:
    """
    Build ell-bin indices for rfft2 coefficients.

    Conventions match the CMBAtmo pixel basis:
      - pixel vectors are (n_pix,) with pixel_index = iy + ix * ny
      - we reshape to (nx, ny) (ix as axis 0, iy as axis 1)
      - rfft2 returns shape (nx, ny//2+1); we fftshift only along axis 0

    If cos_dec != 1.0, the x-axis resolution is physically smaller by cos_dec.
    To maintain isotropic ell on the sky, we scale the x wavenumbers:
      kx_phys = kx_grid / cos_dec
    """
    nx = int(nx)
    ny = int(ny)
    dx = float(pixel_res_rad)
    cos_dec = float(cos_dec)
    n_ell_bins = int(n_ell_bins)

    ell_x = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))  # (nx,)
    ell_y = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dx)  # (ny//2+1,)
    
    # Scale x wavenumber to physical units
    ell_x = ell_x / cos_dec

    KX, KY = np.meshgrid(ell_x, ell_y, indexing="ij")  # (nx, ny//2+1)
    ell = np.sqrt(KX * KX + KY * KY)

    ell_max = float(np.max(ell))
    edges = np.linspace(0.0, ell_max, n_ell_bins + 1)
    idx = np.digitize(ell.reshape(-1), edges) - 1
    idx = np.clip(idx, 0, n_ell_bins - 1).astype(np.int32, copy=False)
    return idx.reshape(nx, ny // 2 + 1)


@dataclass(frozen=True)
class FourierGaussianPrior:
    """
    Stationary Gaussian prior on a regular grid, diagonal in Fourier space.

    Args:
      nx, ny: grid size.
      pixel_res_rad: pixel size in radians (assumed square pixels in grid coordinates).
      cl_bins_mk2: (n_ell_bins,) binned C_ell in mK^2.
      cos_dec: cosine of the reference declination (scales dx -> dx*cos_dec).
      cl_floor_mk2: floor to keep C_ell positive.

    Shapes:
      - pixel vectors are (n_pix,) with pixel_index = iy + ix * ny.
      - rfft2 coefficient arrays are (nx, ny//2+1) with fftshift on axis 0 only.
    """

    nx: int
    ny: int
    pixel_res_rad: float
    cl_bins_mk2: np.ndarray  # (n_ell_bins,)
    cos_dec: float = 1.0
    cl_floor_mk2: float = 1e-12

    def __post_init__(self) -> None:
        cl = np.asarray(self.cl_bins_mk2, dtype=np.float64)
        cl = np.where(np.isfinite(cl), cl, 0.0)
        cl = np.maximum(cl, float(self.cl_floor_mk2))
        object.__setattr__(self, "cl_bins_mk2", cl)

        bin_idx = _ell_bin_indices_rfft2(
            nx=int(self.nx),
            ny=int(self.ny),
            pixel_res_rad=float(self.pixel_res_rad),
            cos_dec=float(self.cos_dec),
            n_ell_bins=int(cl.size),
        )
        object.__setattr__(self, "_ell_bin_idx", bin_idx)

    def _rfft2_shifted(self, x_pix: np.ndarray) -> np.ndarray:
        x2 = np.asarray(x_pix, dtype=np.float64).reshape(int(self.nx), int(self.ny))
        X_unshift = np.fft.rfft2(x2)
        return np.fft.fftshift(X_unshift, axes=0)

    def _irfft2_from_shifted(self, X_shifted: np.ndarray) -> np.ndarray:
        X_unshift = np.fft.ifftshift(X_shifted, axes=0)
        x = np.fft.irfft2(X_unshift, s=(int(self.nx), int(self.ny)))
        return x.reshape(int(self.nx) * int(self.ny))

    def _cl_per_mode(self) -> np.ndarray:
        idx = np.asarray(self._ell_bin_idx, dtype=np.int32)
        return np.asarray(self.cl_bins_mk2, dtype=np.float64)[idx]  # (nx, ny//2+1)

    def apply_Cinv(self, x_pix: np.ndarray) -> np.ndarray:
        """
        Apply C^{-1} in pixel space.

        With numpy FFT conventions used here:
          - dy = pixel_res_rad
          - dx = pixel_res_rad * cos_dec (physical size of x-pixel)
          - the pixel-space covariance operator has Fourier eigenvalues
              λ(k) = Cl(k) / (dx*dy)
            so the inverse has eigenvalues
              λ⁻¹(k) = (dx*dy) / Cl(k).
        """
        X = self._rfft2_shifted(x_pix)
        cl_mode = self._cl_per_mode()
        dxdy = float(self.pixel_res_rad) * float(self.pixel_res_rad) * float(self.cos_dec)
        factor = dxdy / cl_mode
        return self._irfft2_from_shifted(X * factor).real

    def apply_C(self, x_pix: np.ndarray) -> np.ndarray:
        """
        Apply C in pixel space.

        This is the inverse of `apply_Cinv` (up to numerical roundoff), with
        Fourier eigenvalues:
          λ(k) = Cl(k) / (dx*dy).
        """
        X = self._rfft2_shifted(x_pix)
        cl_mode = self._cl_per_mode()
        dxdy = float(self.pixel_res_rad) * float(self.pixel_res_rad) * float(self.cos_dec)
        factor = cl_mode / dxdy
        return self._irfft2_from_shifted(X * factor).real
