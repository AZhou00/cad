#!/usr/bin/env python3
"""
Sanity checks for the FFT-diagonal prior normalization used in `cad.prior`.

Checks:
  - Constant C_ell: C^{-1} = (dx dy / C_ell) I in pixel space.
  - MC: sample maps with target C_ell, recover mean C_ell.
"""

from __future__ import annotations

import numpy as np

import cad
from cad import power


def _ell_edges_like_prior(*, nx: int, ny: int, pixel_res_rad: float, n_ell_bins: int) -> np.ndarray:
    """Reproduce ell bin edges used by the FFT prior."""
    nx = int(nx)
    ny = int(ny)
    dx = float(pixel_res_rad)
    ell_x = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))  # (nx,)
    ell_y = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dx)  # (ny//2+1,)
    KX, KY = np.meshgrid(ell_x, ell_y, indexing="ij")
    ell_max = float(np.max(np.sqrt(KX * KX + KY * KY)))
    return np.linspace(0.0, ell_max, int(n_ell_bins) + 1)


def _bin_idx_full_fft2(*, nx: int, ny: int, pixel_res_rad: float, edges: np.ndarray) -> np.ndarray:
    """Bin indices for the full fft2 grid (nx, ny), using `edges`."""
    nx = int(nx)
    ny = int(ny)
    dx = float(pixel_res_rad)
    edges = np.asarray(edges, dtype=np.float64)

    ell_x = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)  # (nx,)
    ell_y = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)  # (ny,)
    KX, KY = np.meshgrid(ell_x, ell_y, indexing="ij")  # (nx, ny)
    ell = np.sqrt(KX * KX + KY * KY)

    idx = np.digitize(ell.reshape(-1), edges) - 1
    idx = np.clip(idx, 0, int(edges.size) - 2).astype(np.int32, copy=False)
    return idx.reshape(nx, ny)


def _sample_real_field_fft2(*, nx: int, ny: int, var_fk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a real periodic field via full FFT2 coefficients."""
    nx = int(nx)
    ny = int(ny)
    var_fk = np.asarray(var_fk, dtype=np.float64)
    if var_fk.shape != (nx, ny):
        raise ValueError("var_fk must have shape (nx, ny).")

    F = np.zeros((nx, ny), dtype=np.complex128)
    kx = np.arange(nx, dtype=np.int64)
    kx_conj = (-kx) % nx

    def _complex_gaussian(var: np.ndarray) -> np.ndarray:
        s = np.sqrt(np.maximum(var, 0.0) / 2.0)
        return rng.normal(scale=s) + 1j * rng.normal(scale=s)

    # ky slices paired with distinct conjugate slices.
    ky_max = ny // 2
    ky_mid_end = ky_max - 1 if (ny % 2 == 0) else ky_max
    for ky in range(1, int(ky_mid_end) + 1):
        z = _complex_gaussian(var_fk[:, ky])
        F[:, ky] = z
        F[kx_conj, (ny - ky) % ny] = np.conj(z[kx])

    # ky = 0 slice (self-conjugate).
    F[0, 0] = float(rng.normal(scale=np.sqrt(var_fk[0, 0])))
    kx_max = nx // 2
    kx_mid_end = kx_max - 1 if (nx % 2 == 0) else kx_max
    for kxi in range(1, int(kx_mid_end) + 1):
        z = _complex_gaussian(np.array(var_fk[kxi, 0]))
        F[kxi, 0] = z
        F[(nx - kxi) % nx, 0] = np.conj(z)
    if nx % 2 == 0:
        F[kx_max, 0] = float(rng.normal(scale=np.sqrt(var_fk[kx_max, 0])))

    # ky = ny/2 slice when ny even (self-conjugate).
    if ny % 2 == 0:
        ky_nyq = ky_max
        F[0, ky_nyq] = float(rng.normal(scale=np.sqrt(var_fk[0, ky_nyq])))
        for kxi in range(1, int(kx_mid_end) + 1):
            z = _complex_gaussian(np.array(var_fk[kxi, ky_nyq]))
            F[kxi, ky_nyq] = z
            F[(nx - kxi) % nx, ky_nyq] = np.conj(z)
        if nx % 2 == 0:
            F[kx_max, ky_nyq] = float(rng.normal(scale=np.sqrt(var_fk[kx_max, ky_nyq])))

    return np.fft.ifft2(F).real.astype(np.float64, copy=False)


def main() -> None:
    rng = np.random.default_rng(0)

    nx, ny = 32, 48
    pixel_size_deg = 0.5
    pixel_res_rad = float(pixel_size_deg) * np.pi / 180.0
    dxdy = float(pixel_res_rad) * float(pixel_res_rad)
    n_pix = int(nx * ny)

    n_ell_bins = 64
    cl_floor_mk2 = 1e-12

    # (1) Constant C_ell: C^{-1} = (dx dy / C_ell) I.
    cl0 = 7.0  # mK^2
    prior_const = cad.FourierGaussianPrior(
        nx=nx,
        ny=ny,
        pixel_res_rad=pixel_res_rad,
        cl_bins_mk2=np.full((n_ell_bins,), float(cl0), dtype=np.float64),
        cl_floor_mk2=float(cl_floor_mk2),
    )
    x = rng.normal(size=(n_pix,)).astype(np.float64)
    y = prior_const.apply_Cinv(x)
    y_ref = (dxdy / float(cl0)) * x
    rel_err = float(np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref))
    print(f"[const] rel_err(apply_Cinv vs scalar*I) = {rel_err:.3e}")

    # (2) Varying spectrum: sample Gaussian maps and compare mean C_ell.
    edges = _ell_edges_like_prior(nx=nx, ny=ny, pixel_res_rad=pixel_res_rad, n_ell_bins=n_ell_bins)
    ell_centers = 0.5 * (edges[:-1] + edges[1:])
    cl_bins = 50.0 * (1.0 + (ell_centers / 600.0) ** 2) ** (-1.5)
    cl_bins = np.maximum(cl_bins, float(cl_floor_mk2))

    prior = cad.FourierGaussianPrior(
        nx=nx,
        ny=ny,
        pixel_res_rad=pixel_res_rad,
        cl_bins_mk2=np.asarray(cl_bins, dtype=np.float64),
        cl_floor_mk2=float(cl_floor_mk2),
    )

    # Sample with E|F(k)|^2 = n_pix * C_ell / (dxdy).
    bin_idx_full = _bin_idx_full_fft2(nx=nx, ny=ny, pixel_res_rad=pixel_res_rad, edges=edges)
    cl_mode_full = np.asarray(cl_bins, dtype=np.float64)[bin_idx_full]  # (nx, ny)
    var_fk = float(n_pix) * cl_mode_full / dxdy  # E|F(k)|^2

    n_mc = 80
    cl_acc = np.zeros((n_ell_bins,), dtype=np.float64)
    used = 0
    quad = []
    for _ in range(n_mc):
        x2 = _sample_real_field_fft2(nx=nx, ny=ny, var_fk=var_fk, rng=rng)
        x_pix = x2.reshape(n_pix)
        quad.append(float(np.dot(x_pix, prior.apply_Cinv(x_pix))))
        ell_est, cl_est = power.radial_cl_1d_from_map(map_2d_mk=x2.T, pixel_res_rad=pixel_res_rad, n_ell_bins=n_ell_bins)
        m = np.isfinite(cl_est)
        if bool(np.any(m)):
            cl_acc += np.where(m, cl_est, 0.0)
            used += 1

    print(f"[mc] E[x^T Cinv x] ≈ {float(np.mean(quad)):.2f} (target n_pix={n_pix})")
    cl_mean = cl_acc / max(used, 1)
    cl_tgt = np.interp(np.asarray(ell_est, dtype=np.float64), ell_centers, cl_bins)
    m = np.isfinite(cl_mean) & np.isfinite(cl_tgt) & (cl_tgt > 0)
    rel_rms = float(np.sqrt(np.mean(((cl_mean[m] - cl_tgt[m]) / cl_tgt[m]) ** 2))) if bool(np.any(m)) else float("nan")
    print(f"[mc] radial C_ell mean rel_RMS ≈ {rel_rms:.3e} (bins_used={int(np.sum(m))}/{n_ell_bins})")


if __name__ == "__main__":
    main()

