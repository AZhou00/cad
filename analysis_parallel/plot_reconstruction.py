#!/usr/bin/env python3
"""
Plot combined and per-scan reconstructions (parallel path output).

Reads: OUT_BASE/field_id/observation_id/recon_combined_ml.npz and scans/scan_*_ml.npz.
Builds naive coadd from DATA_DIR/field_id/observation_id/binned_tod_10arcmin/*.npz.
Writes: OUT_BASE/field_id/observation_id/plots/

Plots produced:
  maps_naive_vs_combined_ml.png  — naive coadd vs recon combined (map stack)
  power2d_naive_vs_combined_ml.png — 2D power spectrum naive vs combined
  cl_naive_vs_combined_ml.png   — radial C_ell naive vs combined
  maps_single_scan_naive_vs_point_ml.png — top N scans: left col naive per scan, right col point estimate
  pixel_precision_scan0_scan1_ml.png — precision [Cov(hat c)]^{-1} for first two scans

Usage:
  python plot_reconstruction.py [field_id] [observation_id]
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

FIELD_ID = "ra0hdec-59.75"
OBSERVATION_ID = "101706388"
N_ELL_BINS = 128
PREFER_BINNED = "binned_tod_10arcmin"
TOP_N_SCANS = 5


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
    binned_dirs = sorted([p for p in obs_data_dir.iterdir() if p.is_dir() and p.name.startswith("binned_tod_")])
    if not binned_dirs:
        return []
    chosen = next((p for p in binned_dirs if p.name == PREFER_BINNED), binned_dirs[0])
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
) -> None:
    n_rows = len(naive_per_scan)
    if n_rows == 0:
        return
    fig, axs = plt.subplots(n_rows, 2, figsize=(10.0, 2.3 * n_rows), dpi=150, sharex=True, sharey=True)
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    im_last = None
    for i in range(n_rows):
        _imshow(axs[i, 0], naive_per_scan[i], extent=extent, title=f"Scan {i}: naive coadd [mK]", vmin=vmin, vmax=vmax)
        im_last = _imshow(axs[i, 1], rec_per_scan[i], extent=extent, title=f"Scan {i}: point estimate [mK]", vmin=vmin, vmax=vmax)
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.15)
    pos = axs[-1, -1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.015, pos.height])
    fig.colorbar(im_last, cax=cax, orientation="vertical").set_label("mK")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pixel_precision_first_two_scans(
    out_path: pathlib.Path,
    cov_inv_0: np.ndarray,
    cov_inv_1: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), dpi=150)
    for ax, cov_inv, title in zip(axes, [cov_inv_0, cov_inv_1], ["Scan 0: precision [Cov(hat c)]^{-1}", "Scan 1: precision [Cov(hat c)]^{-1}"], strict=True):
        v = np.abs(np.asarray(cov_inv, dtype=np.float64))
        v_flat = v.ravel()
        v_pos = v_flat[(v_flat > 0) & np.isfinite(v_flat)]
        vmin = float(np.min(v_pos)) if v_pos.size > 0 else 1e-30
        vmax = float(np.max(v_pos)) if v_pos.size > 0 else 1.0
        im = ax.imshow(v, aspect="auto", norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("pixel index")
        ax.set_ylabel("pixel index")
        plt.colorbar(im, ax=ax, label="|precision|")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(field_id: str, observation_id: str) -> None:
    out_dir = OUT_BASE / field_id / observation_id
    obs_data_dir = DATA_DIR / field_id / observation_id
    combined_npz = out_dir / "recon_combined_ml.npz"
    scan_dir = out_dir / "scans"
    plots_dir = out_dir / "plots"

    if not combined_npz.exists():
        raise FileNotFoundError(f"Combined recon not found: {combined_npz}. Run run_synthesis.py first.")

    with np.load(combined_npz, allow_pickle=True) as rc:
        bbox_ix0 = int(rc["bbox_ix0"])
        bbox_iy0 = int(rc["bbox_iy0"])
        nx = int(rc["nx"])
        ny = int(rc["ny"])
        pixel_size_deg = float(rc["pixel_size_deg"])
        rec_full = np.asarray(rc["c_hat_full_mk"], dtype=np.float64)

    bbox = map_util.BBox(ix0=bbox_ix0, ix1=bbox_ix0 + nx - 1, iy0=bbox_iy0, iy1=bbox_iy0 + ny - 1)
    extent = _extent_deg(bbox=bbox, pixel_size_deg=pixel_size_deg)
    pixel_res_rad = pixel_size_deg * np.pi / 180.0
    rec = _img_from_vec(rec_full, nx=nx, ny=ny)

    tod_paths = _binned_tod_paths(obs_data_dir)
    if not tod_paths:
        raise FileNotFoundError(f"No binned TOD under {obs_data_dir}")
    naive, hit_mask = _naive_coadd(tod_paths, bbox)
    rec_masked = np.where(hit_mask & np.isfinite(rec), rec, np.nan).astype(np.float32, copy=False)

    vmin, vmax = _robust_vmin_vmax(np.concatenate([naive.ravel(), rec_masked.ravel()]))
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_naive_vs_combined_maps(plots_dir / "maps_naive_vs_combined_ml.png", naive, rec_masked, extent, vmin, vmax)
    plot_naive_vs_combined_power2d(plots_dir / "power2d_naive_vs_combined_ml.png", naive, rec_masked, pixel_res_rad, hit_mask)

    ell, cl_naive = power.radial_cl_1d_from_map(map_2d_mk=naive, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=N_ELL_BINS)
    _, cl_rec = power.radial_cl_1d_from_map(map_2d_mk=rec_masked, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=N_ELL_BINS)
    plot_naive_vs_combined_cl(plots_dir / "cl_naive_vs_combined_ml.png", ell, cl_naive, cl_rec)

    scan_npzs = sorted(scan_dir.glob("scan_*_ml.npz"), key=_scan_npz_sort_key)[:TOP_N_SCANS]
    if scan_npzs and len(tod_paths) >= len(scan_npzs):
        naive_per_scan = [_naive_coadd_one_scan(tod_paths[i], bbox) for i in range(len(scan_npzs))]
        rec_per_scan = []
        for p in scan_npzs:
            m = _load_scan_recon_map(p)
            rec_per_scan.append(np.where(hit_mask & np.isfinite(m), m, np.nan).astype(np.float32) if m is not None else np.full((int(bbox.ny), int(bbox.nx)), np.nan, dtype=np.float32))
        vmin_s, vmax_s = _robust_vmin_vmax(np.concatenate([x.ravel() for x in naive_per_scan + rec_per_scan]))
        plot_single_scan_naive_vs_point(plots_dir / "maps_single_scan_naive_vs_point_ml.png", naive_per_scan, rec_per_scan, extent, vmin_s, vmax_s)

    cov_npzs = sorted(scan_dir.glob("scan_*_ml.npz"), key=_scan_npz_sort_key)[:2]
    if len(cov_npzs) >= 2:
        with np.load(cov_npzs[0], allow_pickle=True) as z:
            cov_inv_0 = np.asarray(z["cov_inv"], dtype=np.float64)
        with np.load(cov_npzs[1], allow_pickle=True) as z:
            cov_inv_1 = np.asarray(z["cov_inv"], dtype=np.float64)
        plot_pixel_precision_first_two_scans(plots_dir / "pixel_precision_scan0_scan1_ml.png", cov_inv_0, cov_inv_1)

    print(f"Wrote plots to {plots_dir}", flush=True)


if __name__ == "__main__":
    field_id = sys.argv[1] if len(sys.argv) >= 2 else FIELD_ID
    observation_id = sys.argv[2] if len(sys.argv) >= 3 else OBSERVATION_ID
    main(field_id=str(field_id), observation_id=str(observation_id))
