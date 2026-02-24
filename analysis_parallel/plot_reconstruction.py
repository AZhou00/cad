#!/usr/bin/env python3
"""
Plot combined ML reconstruction from the parallel pipeline (run_synthesis output).
Uses hardcoded paths matching run_synthesis. Outputs to analysis_parallel/plots.
"""

from __future__ import annotations

import pathlib
import sys

THIS_DIR = pathlib.Path(__file__).resolve().parent
CAD_SRC = THIS_DIR.parent / "src"
if str(CAD_SRC) not in sys.path:
    sys.path.insert(0, str(CAD_SRC))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from cad import map as map_util
from cad import power

# Hardcoded paths (edit for another observation)
DATASET_NAME = "ra0hdec-59.75"
FIELD_ID = "101706388"
OUT_BASE = pathlib.Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
LAYOUT_NPZ = OUT_BASE / DATASET_NAME / FIELD_ID / "layout.npz"
COMBINED_NPZ = OUT_BASE / DATASET_NAME / FIELD_ID / "recon_combined_ml.npz"
SCAN_NPZ_DIR = OUT_BASE / DATASET_NAME / FIELD_ID / "scans"

PLOTS_DIR = THIS_DIR / "plots"

N_ELL_BINS = 128
KDOTW_EXCLUDE_COS = 0.5


def _img_from_vec(vec: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
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
    im = ax.imshow(
        img,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    return im


def _plot_map_stack(
    *,
    out_path: pathlib.Path,
    imgs: list[np.ndarray],
    titles: list[str],
    extent: list[float],
    vmin: float,
    vmax: float,
    cbar_label: str = "mK",
) -> None:
    n = len(imgs)
    if n == 0:
        return
    fig, axs = plt.subplots(n, 1, figsize=(9.0, 2.3 * n), dpi=150, sharex=True, sharey=True)
    if n == 1:
        axs = [axs]
    ims = []
    for i in range(n):
        im = _imshow(axs[i], imgs[i], extent=extent, title=titles[i], vmin=vmin, vmax=vmax)
        ims.append(im)
    fig.subplots_adjust(right=0.86, hspace=0.25)
    pos = axs[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    fig.colorbar(ims[-1], cax=cax, orientation="vertical").set_label(cbar_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_power2d_comparison(
    *,
    out_path: pathlib.Path,
    maps_2d_mk: list[tuple[str, np.ndarray]],
    pixel_res_rad: float,
    hit_mask_2d: np.ndarray,
) -> None:
    if len(maps_2d_mk) == 0:
        return
    n_pan = len(maps_2d_mk)
    fig, axes = plt.subplots(1, n_pan, figsize=(4.8 * n_pan, 4.2), sharex=True, sharey=True, dpi=150)
    if n_pan == 1:
        axes = [axes]
    vmin = vmax = None
    for _name, m2d in maps_2d_mk:
        KX, KY, ps2d = power.power2d_from_map(map_2d_mk=m2d, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)
        ps2d = np.ma.masked_invalid(ps2d * 1e6)
        disp = (np.abs(KX) <= 500.0) & (np.abs(KY) <= 500.0)
        vals = ps2d[disp].compressed()
        if vals.size == 0:
            continue
        vmax_i = float(np.max(vals))
        vmin_i = max(vmax_i * 1e-5, 1e-30)
        vmax = vmax_i if vmax is None else max(vmax, vmax_i)
        vmin = vmin_i if vmin is None else min(vmin, vmin_i)
    if vmin is None or vmax is None:
        return
    norm = colors.LogNorm(vmin=float(vmin), vmax=float(vmax))
    levels = np.logspace(np.log10(float(vmin)), np.log10(float(vmax)), 30)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))
    cmap.set_under(cmap(0.0))
    last_cf = None
    for ax, (name, m2d) in zip(axes, maps_2d_mk, strict=True):
        KX, KY, ps2d = power.power2d_from_map(map_2d_mk=m2d, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)
        ps2d = np.ma.masked_invalid(ps2d * 1e6)
        last_cf = ax.contourf(KX, KY, ps2d, levels=levels, cmap=cmap, norm=norm, extend="min")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$\ell_x$ [rad$^{-1}$]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
    axes[0].set_ylabel(r"$\ell_y$ [rad$^{-1}$]")
    fig.subplots_adjust(right=0.86, wspace=0.22)
    pos = axes[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    cb = fig.colorbar(last_cf, cax=cax, orientation="vertical", ticks=[float(vmin), float(vmax)])
    cb.ax.set_yticklabels([f"{float(vmin):.0e}", f"{float(vmax):.0e}"])
    cb.set_label(r"$C_\ell$ [$\mu K_{\rm CMB}^2$]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _exclude_ky_zero_row(KY: np.ndarray) -> np.ndarray:
    ky_abs = np.abs(np.asarray(KY, dtype=np.float64))
    ky_pos = ky_abs[ky_abs > 0.0]
    if ky_pos.size == 0:
        return np.ones_like(ky_abs, dtype=bool)
    return ky_abs >= float(np.min(ky_pos))


def _plot_nondegenerate(
    *,
    out_dir: pathlib.Path,
    naive: np.ndarray,
    rec: np.ndarray,
    hit_mask_2d: np.ndarray,
    pixel_res_rad: float,
    extent: list[float],
    w_mean: np.ndarray,
) -> None:
    w_norm = float(np.hypot(float(w_mean[0]), float(w_mean[1])))
    if not (np.isfinite(w_norm) and w_norm > 0):
        return
    KX, KY, ps_co = power.power2d_from_map(map_2d_mk=naive, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)
    _, _, ps_re = power.power2d_from_map(map_2d_mk=rec, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)
    ell = np.sqrt(KX * KX + KY * KY)
    denom = np.maximum(ell * w_norm, 1e-300)
    cosang = (KX * float(w_mean[0]) + KY * float(w_mean[1])) / denom
    keep = np.isfinite(cosang) & (np.abs(cosang) >= float(KDOTW_EXCLUDE_COS)) & _exclude_ky_zero_row(KY)
    ps_co_m = np.ma.masked_invalid(np.where(keep, ps_co * 1e6, np.nan))
    ps_re_m = np.ma.masked_invalid(np.where(keep, ps_re * 1e6, np.nan))
    disp = (np.abs(KX) <= 500.0) & (np.abs(KY) <= 500.0)
    vals = np.concatenate([ps_co_m[disp].compressed(), ps_re_m[disp].compressed()])
    if vals.size == 0:
        return
    vmax = float(np.max(vals))
    vmin = max(vmax * 1e-5, 1e-30)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 30)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True, dpi=150)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))
    cmap.set_under(cmap(0.0))
    cf0 = axes[0].contourf(KX, KY, ps_co_m, levels=levels, cmap=cmap, norm=norm, extend="min")
    axes[1].contourf(KX, KY, ps_re_m, levels=levels, cmap=cmap, norm=norm, extend="min")
    axes[0].set_title("Naive coadd (non-degenerate modes)", fontsize=10)
    axes[1].set_title("Recon coadd (non-degenerate modes)", fontsize=10)
    for ax in axes:
        ax.set_xlabel(r"$\ell_x$ [rad$^{-1}$]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
    axes[0].set_ylabel(r"$\ell_y$ [rad$^{-1}$]")
    fig.subplots_adjust(right=0.86, wspace=0.22)
    pos = axes[1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    cb = fig.colorbar(cf0, cax=cax, orientation="vertical", ticks=[vmin, vmax])
    cb.ax.set_yticklabels([f"{vmin:.0e}", f"{vmax:.0e}"])
    cb.set_label(r"$C_\ell$ [$\mu K_{\rm CMB}^2$]")
    fig.savefig(out_dir / "maps_coadd_vs_reconcoadd_power2d_nondegenerate_ml.png", bbox_inches="tight")
    plt.close(fig)
    ell_m, cl_co_m = power.radial_cl_1d_from_power2d(KX=KX, KY=KY, ps2d_mk2=ps_co, n_ell_bins=int(N_ELL_BINS), keep_mask_2d=keep)
    _, cl_re_m = power.radial_cl_1d_from_power2d(KX=KX, KY=KY, ps2d_mk2=ps_re, n_ell_bins=int(N_ELL_BINS), keep_mask_2d=keep)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ax.plot(ell_m, cl_co_m, color="k", lw=2.0, label="naive coadd (non-degenerate)")
    ax.plot(ell_m, cl_re_m, color="C3", lw=2.5, label="recon coadd (non-degenerate)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mK$^2$]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "cl_nondegenerate_ml.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from cad.parallel_solve import load_layout

    if not COMBINED_NPZ.exists():
        raise RuntimeError(f"Combined reconstruction not found: {COMBINED_NPZ}. Run run_synthesis.py first.")
    if not LAYOUT_NPZ.exists():
        raise RuntimeError(f"Layout not found: {LAYOUT_NPZ}. Run build_layout.py first.")

    layout = load_layout(LAYOUT_NPZ)
    bbox = map_util.BBox(ix0=layout.bbox_ix0, ix1=layout.bbox_ix0 + layout.nx - 1, iy0=layout.bbox_iy0, iy1=layout.bbox_iy0 + layout.ny - 1)
    pixel_size_deg = layout.pixel_size_deg
    pixel_res_rad = pixel_size_deg * np.pi / 180.0
    extent = _extent_deg(bbox=bbox, pixel_size_deg=pixel_size_deg)

    with np.load(COMBINED_NPZ, allow_pickle=False) as rc:
        rec_full = _img_from_vec(np.asarray(rc["c_hat_full_mk"], dtype=np.float64), nx=layout.nx, ny=layout.ny)

    s_naive = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.float64)
    c_naive = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.int64)
    for scan_path in layout.scan_paths:
        if not pathlib.Path(scan_path).exists():
            continue
        with np.load(scan_path, allow_pickle=False) as z:
            eff_tod_mk = np.asarray(z["eff_tod_mk"])
            pix_index = np.asarray(z["pix_index"], dtype=np.int64)
        ok = np.isfinite(eff_tod_mk)
        if not bool(np.any(ok)):
            continue
        ij = pix_index[ok]
        ixg = ij[:, 0] - int(bbox.ix0)
        iyg = ij[:, 1] - int(bbox.iy0)
        in_box = (ixg >= 0) & (ixg < int(bbox.nx)) & (iyg >= 0) & (iyg < int(bbox.ny))
        if not bool(np.any(in_box)):
            continue
        v = eff_tod_mk[ok].astype(np.float64, copy=False)[in_box]
        np.add.at(s_naive, (iyg[in_box], ixg[in_box]), v)
        np.add.at(c_naive, (iyg[in_box], ixg[in_box]), 1)

    naive = np.full((int(bbox.ny), int(bbox.nx)), np.nan, dtype=np.float32)
    hit_mask = c_naive > 0
    naive[hit_mask] = (s_naive[hit_mask] / c_naive[hit_mask]).astype(np.float32)
    rec = np.where(hit_mask, np.asarray(rec_full, dtype=np.float64), np.nan).astype(np.float32, copy=False)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    vmin, vmax = _robust_vmin_vmax(np.concatenate([naive.ravel(), rec.ravel()]))
    _plot_map_stack(
        out_path=PLOTS_DIR / "maps_coadd_vs_reconcoadd_ml.png",
        imgs=[naive, rec],
        titles=["Naive coadd (all scans) [mK]", "Recon combined (ML) [mK]"],
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    _plot_power2d_comparison(
        out_path=PLOTS_DIR / "maps_coadd_vs_reconcoadd_power2d_ml.png",
        maps_2d_mk=[("Naive coadd", naive), ("Recon combined (ML)", rec)],
        pixel_res_rad=pixel_res_rad,
        hit_mask_2d=hit_mask,
    )
    ell, cl_naive = power.radial_cl_1d_from_map(map_2d_mk=naive, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=int(N_ELL_BINS))
    _, cl_rec = power.radial_cl_1d_from_map(map_2d_mk=rec, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=int(N_ELL_BINS))
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
    fig.savefig(PLOTS_DIR / "cl_ml.png", bbox_inches="tight")
    plt.close(fig)

    wind_path = SCAN_NPZ_DIR / "scan_0000_ml.npz"
    if wind_path.exists():
        with np.load(wind_path, allow_pickle=False) as z:
            w = np.asarray(z["wind_deg_per_s"], dtype=np.float64).reshape(-1)
        if w.size >= 2 and np.all(np.isfinite(w[:2])):
            _plot_nondegenerate(
                out_dir=PLOTS_DIR,
                naive=naive,
                rec=rec,
                hit_mask_2d=hit_mask,
                pixel_res_rad=pixel_res_rad,
                extent=extent,
                w_mean=w[:2],
            )

    print(f"Wrote plots to {PLOTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
