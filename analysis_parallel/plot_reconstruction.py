#!/usr/bin/env python3
"""
Plot combined and per-scan reconstructions (parallel path output).

Input: path to synthesis npz (recon_combined_ml.npz). Path is normally under OUT_BASE, e.g.
  OUT_BASE/field_id/observation_id/recon_combined_ml.npz or OUT_BASE/field_id/synthesized/recon_combined_ml.npz.
  field_id is inferred as path.parent.parent.name for binned TOD lookup (DATA_DIR/field_id/<obs_id>/).
  Optional: out_dir/scans/scan_*_ml.npz for per-scan map/precision plots; binned TOD from scan_metadata obs ids.
Output: out_dir/plots/ with out_dir = path.parent.

Synthesis npz (recon_combined_ml.npz) structure:
  - bbox_ix0, bbox_iy0, nx, ny, pixel_size_deg: scalars
  - c_hat_full_mk: (n_pix,) ML map on full CMB grid [mK]
  - c_hat_obs: (n_obs,) ML map on observed pixels
  - obs_pix_global: (n_obs,) global pixel indices
  - cov_inv_tot: (n_obs, n_obs) global precision
  - good_mask: (n_obs,) bool
  - uncertain_mode_vectors: (n_good, k), uncertain_mode_variances: (k,)
  - scan_metadata: object array of length n_scans; each element is a dict with
    observation_id (str), scan_index (int), wind_deg_per_s (2,), wind_sigma_x_deg_per_s (float),
    wind_sigma_y_deg_per_s (float), ell_atm (n_ell,), cl_atm_mk2 (n_ell,)

Output plots (each function docstring lists I/O shapes):
  maps_naive_vs_combined_ml, power2d_naive_vs_combined_ml, cl_naive_vs_combined_ml,
  pixel_precision_synthesized_ml, cl_atm_distribution_ml, wind_scatter_ml,
  maps_single_scan_naive_vs_point_ml, pixel_precision_scan0_scan1_ml,
  uncertain_eigenvalues_ml, maps_eigenmode_removed_ml, cl_eigenmode_removed_ml.

Usage:
  python plot_reconstruction.py <path_to_recon_combined_ml.npz>
  Path should be under OUT_BASE (e.g. .../field_id/observation_id/recon_combined_ml.npz). Observation ids
  for binned TOD are read from scan_metadata; per-scan plots use out_dir/scans/ when that dir exists.
"""

from __future__ import annotations

import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

THIS_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = THIS_DIR.parent
DATA_DIR = CAD_DIR / "data"
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

from cad import map as map_util
from cad import power
from cad.parallel_solve.reconstruct_scan import load_scan_artifact

DEFAULT_SYNTHESIS_NPZ = OUT_BASE / "ra0hdec-59.75" / "101706388" / "recon_combined_ml.npz"
N_ELL_BINS = 128
BINNED_TOD_SUBDIR = "binned_tod_10arcmin"
TOP_N_SCANS = 5
PRECISION_CBAR_MIN = 1e-4


def _img_from_vec(vec: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    """vec: pix = iy + ix*ny. Return (ny, nx) with iy as rows."""
    v = np.asarray(vec, dtype=np.float64).reshape(int(nx), int(ny))
    return v.T


def _extent_deg(*, bbox: map_util.BBox, pixel_size_deg: float) -> list[float]:
    x0 = float(bbox.ix0) * float(pixel_size_deg)
    x1 = float(bbox.ix0 + bbox.nx) * float(pixel_size_deg)
    y0 = float(bbox.iy0) * float(pixel_size_deg)
    y1 = float(bbox.iy0 + bbox.ny) * float(pixel_size_deg)
    return [x0, x1, y0, y1]


def _robust_vmin_vmax(x: np.ndarray, *, p_lo: float = 2.0, p_hi: float = 98.0, default=(-1.0, 1.0)) -> tuple[float, float]:
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(default[0]), float(default[1])
    lo, hi = np.percentile(v, [float(p_lo), float(p_hi)])
    return float(lo), float(hi)


def _imshow(ax, img, *, extent, title: str, vmin=None, vmax=None, cmap="RdBu_r"):
    img = np.ma.masked_invalid(np.asarray(img))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=(1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor("white")
    im = ax.imshow(img, origin="lower", extent=extent, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax, interpolation="none")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    return im


def _binned_tod_paths(obs_data_dir: pathlib.Path) -> list[pathlib.Path]:
    chosen = obs_data_dir / BINNED_TOD_SUBDIR
    if not chosen.is_dir():
        return []
    return sorted([p for p in chosen.iterdir() if p.is_file() and p.suffix == ".npz" and not p.name.startswith(".")])


def _naive_coadd(scan_paths: list[pathlib.Path], bbox: map_util.BBox) -> tuple[np.ndarray, np.ndarray]:
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


def _naive_coadd_one_scan(scan_path: pathlib.Path, bbox: map_util.BBox) -> np.ndarray:
    """Naive coadd from a single scan; return (ny, nx)."""
    with np.load(scan_path, allow_pickle=False) as z:
        eff_tod_mk = np.asarray(z["eff_tod_mk"])
        pix_index = np.asarray(z["pix_index"], dtype=np.int64)
    ok = np.isfinite(eff_tod_mk)
    s = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.float64)
    c = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.int64)
    if np.any(ok):
        ij = pix_index[ok]
        ixg = ij[:, 0] - int(bbox.ix0)
        iyg = ij[:, 1] - int(bbox.iy0)
        in_box = (ixg >= 0) & (ixg < int(bbox.nx)) & (iyg >= 0) & (iyg < int(bbox.ny))
        if np.any(in_box):
            v = eff_tod_mk[ok].astype(np.float64, copy=False)[in_box]
            np.add.at(s, (iyg[in_box], ixg[in_box]), v)
            np.add.at(c, (iyg[in_box], ixg[in_box]), 1)
    out = np.full((int(bbox.ny), int(bbox.nx)), np.nan, dtype=np.float32)
    hit = c > 0
    out[hit] = (s[hit] / c[hit]).astype(np.float32)
    return out


def _load_scan_recon_map(npz_path: pathlib.Path) -> np.ndarray | None:
    """Load single-scan npz; return (ny, nx) map (c_hat on full grid)."""
    with np.load(npz_path, allow_pickle=True) as z:
        nx = int(z["nx"])
        ny = int(z["ny"])
        obs_pix = np.asarray(z["obs_pix_global_scan"], dtype=np.int64)
        c_obs = np.asarray(z["c_hat_scan_obs"], dtype=np.float64)
    n_pix = nx * ny
    c_full = np.zeros((n_pix,), dtype=np.float64)
    c_full[obs_pix] = c_obs
    return _img_from_vec(c_full, nx=nx, ny=ny)


def _scan_npz_sort_key(p: pathlib.Path) -> int:
    m = re.search(r"scan_(\d+)_", p.name)
    return int(m.group(1)) if m else 0


# --- Plot functions (called by main with preprocessed data) ---


def plot_naive_vs_combined_maps(
    out_path: pathlib.Path,
    naive: np.ndarray,
    rec_masked: np.ndarray,
    extent: list[float],
    vmin: float,
    vmax: float,
) -> None:
    """
    Side-by-side maps: naive coadd vs ML combined reconstruction.

    Naive coadd = pixel-wise mean of binned TOD (no atmosphere removal). ML combined =
    solution of (sum_s Cov(c_hat_s)^{-1}) c_hat = sum_s P' N^{-1} d_s after marginalizing
    atmosphere. Same color scale; highlights residual atmosphere in naive and cleaning in recon.

    I/O: naive (ny, nx), rec_masked (ny, nx), extent [4], vmin/vmax scalars. Output: PNG to out_path.
    """
    n = 2
    fig, axs = plt.subplots(n, 1, figsize=(9.0, 2.3 * n), dpi=150, sharex=True, sharey=True)
    ims = [_imshow(axs[0], naive, extent=extent, title="Naive coadd [mK]", vmin=vmin, vmax=vmax),
           _imshow(axs[1], rec_masked, extent=extent, title="Recon combined (ML) [mK]", vmin=vmin, vmax=vmax)]
    fig.subplots_adjust(right=0.86, hspace=0.25)
    pos = axs[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    fig.colorbar(ims[-1], cax=cax, orientation="vertical").set_label("mK")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_naive_vs_combined_power2d(
    out_path: pathlib.Path,
    naive: np.ndarray,
    rec_masked: np.ndarray,
    pixel_res_rad: float,
    hit_mask: np.ndarray,
) -> None:
    """
    2D power spectrum (flat-sky C_ell in (lx, ly)): naive vs ML combined.

    Uses same hit mask for both; shows where power is suppressed by the ML filter
    (e.g. along atmospheric dispersion). Units muK_CMB^2; log scale.

    I/O: naive, rec_masked (ny, nx), pixel_res_rad scalar, hit_mask (ny, nx). Output: PNG to out_path.
    """
    maps_2d_mk = [("Naive coadd", naive), ("Recon combined (ML)", rec_masked)]
    n_pan = len(maps_2d_mk)
    fig, axes = plt.subplots(1, n_pan, figsize=(4.8 * n_pan, 4.2), sharex=True, sharey=True, dpi=150)
    vmin, vmax = None, None
    for _name, m2d in maps_2d_mk:
        KX, KY, ps2d = power.power2d_from_map(map_2d_mk=m2d, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask)
        ps2d = np.ma.masked_invalid(ps2d * 1e6)
        disp = (np.abs(KX) <= 500.0) & (np.abs(KY) <= 500.0)
        vals = ps2d[disp].compressed()
        if vals.size > 0:
            vmax_i = float(np.max(vals))
            vmin_i = max(vmax_i * 1e-5, 1e-30)
            vmax = vmax_i if vmax is None else max(vmax, vmax_i)
            vmin = vmin_i if vmin is None else min(vmin, vmin_i)
    if vmin is None or vmax is None:
        plt.close(fig)
        return
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 30)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))
    cmap.set_under(cmap(0.0))
    cf = None
    for ax, (name, m2d) in zip(axes, maps_2d_mk, strict=True):
        KX, KY, ps2d = power.power2d_from_map(map_2d_mk=m2d, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask)
        ps2d = np.ma.masked_invalid(ps2d * 1e6)
        cf = ax.contourf(KX, KY, ps2d, levels=levels, cmap=cmap, norm=norm, extend="min")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$\ell_x$ [rad$^{-1}$]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
    axes[0].set_ylabel(r"$\ell_y$ [rad$^{-1}$]")
    fig.subplots_adjust(right=0.86, wspace=0.22)
    pos = axes[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    cb = fig.colorbar(cf, cax=cax, orientation="vertical")
    cb.set_label(r"$C_\ell$ [$\mu K_{\rm CMB}^2$]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_naive_vs_combined_cl(
    out_path: pathlib.Path,
    ell: np.ndarray,
    cl_naive: np.ndarray,
    cl_rec: np.ndarray,
) -> None:
    """
    1D radial C_ell (azimuthal average): naive coadd vs ML combined.

    Quantifies total power per ell; ML should show lower power on large scales
    where atmosphere dominates. Both use same hit mask. Units mK^2.

    I/O: ell (n_ell_bins,), cl_naive (n_ell_bins,), cl_rec (n_ell_bins,). Output: PNG to out_path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ax.plot(ell, cl_naive, color="k", lw=2.0, label="naive coadd")
    ax.plot(ell, cl_rec, color="C3", lw=2.5, label="recon combined (ML)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mK$^2$]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_single_scan_naive_vs_point(
    out_path: pathlib.Path,
    naive_per_scan: list[np.ndarray],
    rec_per_scan: list[np.ndarray],
    extent: list[float],
    vmin: float,
    vmax: float,
    wind_per_scan: list[tuple[float, float]] | None = None,
) -> None:
    """
    Per-scan rows: left = naive coadd of that scan's TOD, right = per-scan ML point estimate.

    Each row is one scan. Point estimate is c_hat_s from (P' tilde N_s^{-1} P) c_hat_s = P' tilde N_s^{-1} d_s.
    Wind (wx, wy) deg/s in title. Shows scan-to-scan variation and single-scan recon quality.

    I/O: naive_per_scan, rec_per_scan list of (ny, nx), extent [4], vmin/vmax scalars,
    wind_per_scan optional list of (wx, wy). Output: PNG to out_path.
    """
    n_rows = len(naive_per_scan)
    if n_rows == 0:
        return
    fig, axs = plt.subplots(n_rows, 2, figsize=(10.0, 2.3 * n_rows), dpi=150, sharex=True, sharey=True)
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    im_last = None
    for i in range(n_rows):
        _imshow(axs[i, 0], naive_per_scan[i], extent=extent, title=f"Scan {i}: naive coadd [mK]", vmin=vmin, vmax=vmax)
        right_title = f"Scan {i}: point estimate [mK]"
        if wind_per_scan is not None and i < len(wind_per_scan):
            right_title += _wind_title(wind_per_scan[i][0], wind_per_scan[i][1])
        im_last = _imshow(axs[i, 1], rec_per_scan[i], extent=extent, title=right_title, vmin=vmin, vmax=vmax)
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.15)
    pos = axs[-1, -1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.015, pos.height])
    fig.colorbar(im_last, cax=cax, orientation="vertical").set_label("mK")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _precision_vmin_vmax(v: np.ndarray) -> tuple[float, float]:
    v_flat = np.abs(np.asarray(v, dtype=np.float64)).ravel()
    v_pos = v_flat[(v_flat > 0) & np.isfinite(v_flat)]
    vmax = float(np.max(v_pos)) if v_pos.size > 0 else 1.0
    vmin = max(PRECISION_CBAR_MIN, float(np.min(v_pos)) if v_pos.size > 0 else PRECISION_CBAR_MIN)
    vmax = max(vmax, vmin)
    return vmin, vmax


def _wind_title(wx: float, wy: float) -> str:
    return f" w=({wx:.2f},{wy:.2f}) deg/s"


def plot_pixel_precision_first_two_scans(
    out_path: pathlib.Path,
    cov_inv_0: np.ndarray,
    cov_inv_1: np.ndarray,
    wind_0: tuple[float, float] | None = None,
    wind_1: tuple[float, float] | None = None,
) -> None:
    """
    Diagonal and off-diagonal of per-scan precision [Cov(c_hat_s)]^{-1} = P' tilde N_s^{-1} P for scans 0 and 1.

    Log scale |precision|; structure reflects scan geometry and wind (degenerate directions have
    low precision). Same color range for comparison.

    I/O: cov_inv_0, cov_inv_1 (n_obs_scan, n_obs_scan), wind_0/wind_1 optional (wx, wy). Output: PNG to out_path.
    """
    titles = ["Scan 0: precision [Cov(hat c)]^{-1}", "Scan 1: precision [Cov(hat c)]^{-1}"]
    if wind_0 is not None:
        titles[0] += _wind_title(wind_0[0], wind_0[1])
    if wind_1 is not None:
        titles[1] += _wind_title(wind_1[0], wind_1[1])
    vmin0, vmax0 = _precision_vmin_vmax(cov_inv_0)
    vmin1, vmax1 = _precision_vmin_vmax(cov_inv_1)
    vmin = min(vmin0, vmin1)
    vmax = max(vmax0, vmax1)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), dpi=150)
    for ax, cov_inv, title in zip(axes, [cov_inv_0, cov_inv_1], titles, strict=True):
        v = np.abs(np.asarray(cov_inv, dtype=np.float64))
        im = ax.imshow(v, aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("pixel index")
        ax.set_ylabel("pixel index")
        plt.colorbar(im, ax=ax, label="|precision|")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_synthesized_precision(out_path: pathlib.Path, cov_inv_tot: np.ndarray) -> None:
    """
    Global precision (sum over scans) [Cov(c_hat)]^{-1} = sum_s P_s' tilde N_s^{-1} P_s.

    Log scale |precision|; higher where multiple scans or wind diversity add information.

    I/O: cov_inv_tot (n_obs, n_obs). Output: PNG to out_path.
    """
    v = np.abs(np.asarray(cov_inv_tot, dtype=np.float64))
    vmin, vmax = _precision_vmin_vmax(v)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0), dpi=150)
    im = ax.imshow(v, aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    ax.set_title("Synthesized precision [Cov(hat c)]^{-1}", fontsize=10)
    ax.set_xlabel("pixel index")
    ax.set_ylabel("pixel index")
    plt.colorbar(im, ax=ax, label="|precision|")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _deproject_uncertain_modes(
    vec_obs: np.ndarray,
    good_mask: np.ndarray,
    uncertain_vectors: np.ndarray,
) -> np.ndarray:
    """Set amplitudes of unconstrained modes to zero (deproject). vec_obs (n_obs,), good (n_obs,), uncertain_vectors (n_good, k). Returns vec_obs with those modes zeroed."""
    out = np.asarray(vec_obs, dtype=np.float64).copy()
    good = np.asarray(good_mask, dtype=bool)
    V_good = np.asarray(uncertain_vectors, dtype=np.float64)
    if V_good.size == 0:
        return out
    vec_good = out[good]
    vec_good -= V_good @ (V_good.T @ vec_good)
    out[good] = vec_good
    return out


def plot_uncertain_eigenvalues(
    out_path: pathlib.Path,
    uncertain_variances: np.ndarray,
) -> None:
    """
    Posterior variance (eigenvalues of Cov(c_hat)) for the top unconstrained modes.

    uncertain_variances (k,) = 1 / lambda_min from precision; mode index 0 = most uncertain.
    I/O: uncertain_variances (k,). Output: PNG to out_path.
    """
    if uncertain_variances.size == 0:
        return
    v = np.asarray(uncertain_variances, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5), dpi=150)
    ax.plot(np.arange(v.size), v, "o-", ms=2, lw=0.8, color="C0")
    ax.set_xlabel("mode index (most uncertain first)")
    ax.set_ylabel(r"posterior variance [mK$^2$]")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_eigenmode_removed_maps(
    out_path: pathlib.Path,
    naive_2d: np.ndarray,
    c_hat_obs: np.ndarray,
    obs_pix_global: np.ndarray,
    good_mask: np.ndarray,
    uncertain_vectors: np.ndarray,
    nx: int,
    ny: int,
    extent: list[float],
    hit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maps after deprojecting top unconstrained eigenmodes (set their amplitudes to zero).

    Unconstrained modes = eigenvectors of precision [Cov(c_hat)]^{-1} with smallest eigenvalues.
    Deprojecting zeros those mode amplitudes; C_ell and maps show only well-constrained modes.
    Returns (coadd_2d_filtered, rec_2d_filtered) for CL plot.

    I/O: naive_2d (ny, nx), c_hat_obs (n_obs,), obs_pix_global (n_obs,), good_mask (n_obs,),
    uncertain_vectors (n_obs, k), nx/ny scalars, extent [4], hit_mask (ny, nx). Output: PNG to out_path;
    returns (coadd_2d (ny, nx), rec_2d (ny, nx)).
    """
    n_pix = nx * ny
    naive_vec = np.asarray(naive_2d, dtype=np.float64)
    naive_vec = np.where(np.isfinite(naive_vec), naive_vec, 0.0)
    naive_vec = naive_vec.T.ravel()
    coadd_obs = np.asarray(naive_vec[obs_pix_global], dtype=np.float64)
    coadd_filt = _deproject_uncertain_modes(coadd_obs, good_mask, uncertain_vectors)
    rec_filt = _deproject_uncertain_modes(c_hat_obs, good_mask, uncertain_vectors)
    coadd_full = np.full(n_pix, np.nan, dtype=np.float64)
    rec_full = np.full(n_pix, np.nan, dtype=np.float64)
    coadd_full[obs_pix_global] = coadd_filt
    rec_full[obs_pix_global] = rec_filt
    coadd_2d = _img_from_vec(coadd_full, nx=nx, ny=ny)
    rec_2d_filt = _img_from_vec(rec_full, nx=nx, ny=ny)
    vmin, vmax = _robust_vmin_vmax(np.concatenate([coadd_2d.ravel(), rec_2d_filt.ravel()]))
    rec_masked = np.where(hit_mask & np.isfinite(rec_2d_filt), rec_2d_filt, np.nan).astype(np.float32)
    coadd_masked = np.where(hit_mask & np.isfinite(coadd_2d), coadd_2d, np.nan).astype(np.float32)
    fig, axs = plt.subplots(2, 1, figsize=(9.0, 4.6), dpi=150, sharex=True, sharey=True)
    _imshow(axs[0], coadd_masked, extent=extent, title="Coadd (unconstrained modes set to 0) [mK]", vmin=vmin, vmax=vmax)
    im = _imshow(axs[1], rec_masked, extent=extent, title="Synthesized (unconstrained modes set to 0) [mK]", vmin=vmin, vmax=vmax)
    fig.subplots_adjust(right=0.86, hspace=0.25)
    pos = axs[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    fig.colorbar(im, cax=cax, orientation="vertical").set_label("mK")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return coadd_2d, rec_2d_filt


def plot_eigenmode_removed_cl(
    out_path: pathlib.Path,
    coadd_2d_filtered: np.ndarray,
    rec_2d_filtered: np.ndarray,
    pixel_res_rad: float,
    hit_mask: np.ndarray,
) -> None:
    """
    1D C_ell of maps with unconstrained modes set to zero (naive and synthesized).

    Same as cl_naive_vs_combined but for deprojected maps; compares residual power.

    I/O: coadd_2d_filtered, rec_2d_filtered (ny, nx), pixel_res_rad scalar, hit_mask (ny, nx). Output: PNG to out_path.
    """
    ell, cl_coadd = power.radial_cl_1d_from_map(
        map_2d_mk=coadd_2d_filtered, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=N_ELL_BINS
    )
    _, cl_rec = power.radial_cl_1d_from_map(
        map_2d_mk=rec_2d_filtered, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=N_ELL_BINS
    )
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ax.plot(ell, cl_coadd, color="k", lw=2.0, label="coadd (unconstrained modes = 0)")
    ax.plot(ell, cl_rec, color="C3", lw=2.5, label="synthesized (unconstrained modes = 0)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mK$^2$]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cl_distribution(
    out_path: pathlib.Path,
    scan_metadata: list[dict],
) -> None:
    """
    Per-scan atmospheric C_ell: mean and +/- 1 std across scans.

    Each scan contributes ell_atm and cl_atm_mk2 (atmosphere prior from coadd power).
    Plots mean(cl_atm_mk2) with shaded band [mean - std, mean + std] to show scan-to-scan
    variation in estimated atmospheric power. Requires common ell grid (same n_ell_bins).

    I/O: scan_metadata list of dicts (each: ell_atm (n_ell,), cl_atm_mk2 (n_ell,), plus wind/observation_id/scan_index).
    Output: PNG to out_path.
    """
    if not scan_metadata:
        return
    ell_ref = np.asarray(scan_metadata[0]["ell_atm"], dtype=np.float64)
    cl_list = [np.asarray(m["cl_atm_mk2"], dtype=np.float64) for m in scan_metadata]
    n_ell = len(ell_ref)
    if not all(len(c) == n_ell for c in cl_list):
        return
    cl_stack = np.stack(cl_list, axis=0)
    cl_mean = np.mean(cl_stack, axis=0)
    cl_std = np.std(cl_stack, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ax.fill_between(ell_ref, cl_mean - cl_std, cl_mean + cl_std, alpha=0.3, color="C0")
    ax.plot(ell_ref, cl_mean, color="C0", lw=2.0, label=r"mean $\pm$ std across scans")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mK$^2$] (atm prior)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_wind_scatter(
    out_path: pathlib.Path,
    scan_metadata: list[dict],
) -> None:
    """
    Wind velocity (wx, wy) deg/s per scan with error bars; one color per observation_id.

    Scatter (wx, wy); xerr = wind_sigma_x_deg_per_s, yerr = wind_sigma_y_deg_per_s.
    Smaller markers; color by observation_id.

    I/O: scan_metadata list of dicts (each: observation_id, wind_deg_per_s (2,), wind_sigma_* scalars).
    Output: PNG to out_path.
    """
    if not scan_metadata:
        return
    obs_ids = list(dict.fromkeys(m["observation_id"] for m in scan_metadata))
    color_map = {oid: f"C{i % 10}" for i, oid in enumerate(obs_ids)}
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0), dpi=150)
    for obs_id in obs_ids:
        subset = [m for m in scan_metadata if m["observation_id"] == obs_id]
        wx = np.array([float(m["wind_deg_per_s"][0]) for m in subset])
        wy = np.array([float(m["wind_deg_per_s"][1]) for m in subset])
        sx = np.array([float(m["wind_sigma_x_deg_per_s"]) for m in subset])
        sy = np.array([float(m["wind_sigma_y_deg_per_s"]) for m in subset])
        ax.errorbar(wx, wy, xerr=sx, yerr=sy, fmt="o", ms=3, capsize=1.5, capthick=0.8, color=color_map[obs_id], label=str(obs_id))
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"wind $v_x$ [deg/s]")
    ax.set_ylabel(r"wind $v_y$ [deg/s]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="best", ncol=2 if len(obs_ids) > 5 else 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(synthesis_npz: pathlib.Path) -> None:
    """
    Load synthesis npz, optional scan npzs and binned TOD; write all plots.

    Reads from synthesis_npz: bbox_*, nx, ny, pixel_size_deg, c_hat_full_mk (n_pix,), c_hat_obs (n_obs,),
    obs_pix_global (n_obs,), cov_inv_tot (n_obs, n_obs), good_mask (n_obs,), uncertain_mode_vectors (n_good, k),
    scan_metadata (list of dicts). out_dir = synthesis_npz.parent; field_id = synthesis_npz.parent.parent.name.
    Observation ids for binned TOD from scan_metadata; scan_dir = out_dir/scans if present. Writes to out_dir/plots/.
    """
    combined_npz = pathlib.Path(synthesis_npz).resolve()
    if not combined_npz.exists():
        raise FileNotFoundError(f"Combined recon not found: {combined_npz}. Run run_synthesis.py first.")
    out_dir = combined_npz.parent
    field_id = combined_npz.parent.parent.name
    scan_dir = out_dir / "scans"
    scan_dir = scan_dir if scan_dir.is_dir() else None

    plots_dir = out_dir / "plots"
    with np.load(combined_npz, allow_pickle=True) as rc:
        bbox_ix0 = int(rc["bbox_ix0"])
        bbox_iy0 = int(rc["bbox_iy0"])
        nx = int(rc["nx"])
        ny = int(rc["ny"])
        pixel_size_deg = float(rc["pixel_size_deg"])
        rec_full = np.asarray(rc["c_hat_full_mk"], dtype=np.float64)
        c_hat_obs = np.asarray(rc["c_hat_obs"], dtype=np.float64)
        obs_pix_global = np.asarray(rc["obs_pix_global"], dtype=np.int64)
        cov_inv_tot = np.asarray(rc["cov_inv_tot"], dtype=np.float64)
        good_mask = np.asarray(rc["good_mask"], dtype=bool)
        uncertain_vectors = np.asarray(rc["uncertain_mode_vectors"], dtype=np.float64)
        uncertain_variances = np.asarray(rc["uncertain_mode_variances"], dtype=np.float64) if "uncertain_mode_variances" in rc else np.empty((0,), dtype=np.float64)
        scan_metadata = rc["scan_metadata"].tolist() if "scan_metadata" in rc else []

    obs_ids_from_meta = list(dict.fromkeys(m["observation_id"] for m in scan_metadata))
    obs_data_dirs = [DATA_DIR / field_id / oid for oid in obs_ids_from_meta]

    bbox = map_util.BBox(ix0=bbox_ix0, ix1=bbox_ix0 + nx - 1, iy0=bbox_iy0, iy1=bbox_iy0 + ny - 1)
    extent = _extent_deg(bbox=bbox, pixel_size_deg=pixel_size_deg)
    pixel_res_rad = pixel_size_deg * np.pi / 180.0
    rec = _img_from_vec(rec_full, nx=nx, ny=ny)

    tod_paths: list[pathlib.Path] = []
    for d in obs_data_dirs:
        tod_paths.extend(_binned_tod_paths(d))
    naive, hit_mask = _naive_coadd(tod_paths, bbox) if tod_paths else (np.full((int(bbox.ny), int(bbox.nx)), np.nan, dtype=np.float32), np.zeros((int(bbox.ny), int(bbox.nx)), dtype=bool))
    if not np.any(hit_mask) and tod_paths:
        hit_mask = np.isfinite(naive)
    rec_masked = np.where(hit_mask & np.isfinite(rec), rec, np.nan).astype(np.float32, copy=False)

    vmin, vmax = _robust_vmin_vmax(np.concatenate([naive.ravel(), rec_masked.ravel()]))
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_naive_vs_combined_maps(plots_dir / "maps_naive_vs_combined_ml.png", naive, rec_masked, extent, vmin, vmax)
    plot_naive_vs_combined_power2d(plots_dir / "power2d_naive_vs_combined_ml.png", naive, rec_masked, pixel_res_rad, hit_mask)
    ell, cl_naive = power.radial_cl_1d_from_map(map_2d_mk=naive, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=N_ELL_BINS)
    _, cl_rec = power.radial_cl_1d_from_map(map_2d_mk=rec_masked, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=N_ELL_BINS)
    plot_naive_vs_combined_cl(plots_dir / "cl_naive_vs_combined_ml.png", ell, cl_naive, cl_rec)

    plot_synthesized_precision(plots_dir / "pixel_precision_synthesized_ml.png", cov_inv_tot)

    if scan_metadata:
        plot_cl_distribution(plots_dir / "cl_atm_distribution_ml.png", scan_metadata)
        plot_wind_scatter(plots_dir / "wind_scatter_ml.png", scan_metadata)

    if scan_dir is not None:
        scan_npzs = sorted(scan_dir.glob("scan_*_ml.npz"), key=_scan_npz_sort_key)[:TOP_N_SCANS]
        if scan_npzs and len(tod_paths) >= len(scan_npzs):
            naive_per_scan = [_naive_coadd_one_scan(tod_paths[i], bbox) for i in range(len(scan_npzs))]
            rec_per_scan = []
            wind_per_scan = []
            for p in scan_npzs:
                m = _load_scan_recon_map(p)
                rec_per_scan.append(np.where(hit_mask & np.isfinite(m), m, np.nan).astype(np.float32) if m is not None else np.full((int(bbox.ny), int(bbox.nx)), np.nan, dtype=np.float32))
                art = load_scan_artifact(p)
                w = art["wind_deg_per_s"]
                wind_per_scan.append((float(w[0]), float(w[1])))
            vmin_s, vmax_s = _robust_vmin_vmax(np.concatenate([x.ravel() for x in naive_per_scan + rec_per_scan]))
            plot_single_scan_naive_vs_point(plots_dir / "maps_single_scan_naive_vs_point_ml.png", naive_per_scan, rec_per_scan, extent, vmin_s, vmax_s, wind_per_scan=wind_per_scan)

        cov_npzs = sorted(scan_dir.glob("scan_*_ml.npz"), key=_scan_npz_sort_key)[:2]
        if len(cov_npzs) >= 2:
            art0 = load_scan_artifact(cov_npzs[0])
            art1 = load_scan_artifact(cov_npzs[1])
            wind_0 = (float(art0["wind_deg_per_s"][0]), float(art0["wind_deg_per_s"][1]))
            wind_1 = (float(art1["wind_deg_per_s"][0]), float(art1["wind_deg_per_s"][1]))
            plot_pixel_precision_first_two_scans(
                plots_dir / "pixel_precision_scan0_scan1_ml.png",
                art0["cov_inv"],
                art1["cov_inv"],
                wind_0=wind_0,
                wind_1=wind_1,
            )

    if uncertain_variances.size > 0:
        plot_uncertain_eigenvalues(plots_dir / "uncertain_eigenvalues_ml.png", uncertain_variances)
    if uncertain_vectors.shape[1] > 0 and np.any(tod_paths):
        coadd_2d_f, rec_2d_f = plot_eigenmode_removed_maps(
            plots_dir / "maps_eigenmode_removed_ml.png",
            naive,
            c_hat_obs,
            obs_pix_global,
            good_mask,
            uncertain_vectors,
            nx,
            ny,
            extent,
            hit_mask,
        )
        plot_eigenmode_removed_cl(
            plots_dir / "cl_eigenmode_removed_ml.png",
            coadd_2d_f,
            rec_2d_f,
            pixel_res_rad,
            hit_mask,
        )

    print(f"Wrote plots to {plots_dir}", flush=True)


if __name__ == "__main__":
    synthesis_npz = pathlib.Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_SYNTHESIS_NPZ
    main(synthesis_npz=synthesis_npz)
