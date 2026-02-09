#!/usr/bin/env python3
"""
Plot diagnostics for SPT binned-TOD reconstructions.

Writes figures to:
  cad/analysis/output/reconstruction_<pix>_<ml|map>/plots/

This script is loader-style (it reads the NPZs) and keeps the util modules
loader-free.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

THIS_DIR = pathlib.Path(__file__).resolve().parent

from cad import map as map_util
from cad import power


# -----------------------------------------------------------------------------
# Defaults (edit here)
# -----------------------------------------------------------------------------

CAD_DIR = THIS_DIR.parent
DATA_DIR = CAD_DIR / "data"
OUT_DIR = THIS_DIR / "output"

N_ELL_BINS = 64
NU_GHZ = 220.0
EPSILON_DEG = 44.75
KDOTW_EXCLUDE_COS = 0.5  # exclude modes with |cos(angle(k,w_mean))| < this (k ⟂ w)


def _exclude_ky_zero_row(KY: np.ndarray) -> np.ndarray:
    """
    Exclude the Fourier row closest to ky=0 (remove the ky=0 line).
    """
    ky_abs = np.abs(np.asarray(KY, dtype=np.float64))
    ky_pos = ky_abs[ky_abs > 0.0]
    if ky_pos.size == 0:
        return np.ones_like(ky_abs, dtype=bool)
    ky_min = float(np.min(ky_pos))
    return ky_abs >= ky_min


def _img_from_vec(vec: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    """
    vec uses pixel_index = iy + ix*ny. Return image (ny,nx) with iy as rows.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(int(nx), int(ny))
    return v.T  # (ny,nx)


def _extent_deg(*, bbox_ix0: int, bbox_iy0: int, nx: int, ny: int, pixel_size_deg: float) -> list[float]:
    x0 = float(bbox_ix0) * float(pixel_size_deg)
    x1 = float(bbox_ix0 + nx) * float(pixel_size_deg)
    y0 = float(bbox_iy0) * float(pixel_size_deg)
    y1 = float(bbox_iy0 + ny) * float(pixel_size_deg)
    return [x0, x1, y0, y1]


def _imshow(ax, img, *, extent, title: str, vmin=None, vmax=None, cmap="RdBu_r"):
    img = np.ma.masked_invalid(np.asarray(img))
    cm = plt.get_cmap(cmap).copy()
    # Render NaNs (unhit pixels) as solid white.
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


def _robust_vmin_vmax(x: np.ndarray, *, p_lo: float = 2.0, p_hi: float = 98.0, default=(-1.0, 1.0)) -> tuple[float, float]:
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(default[0]), float(default[1])
    lo, hi = np.percentile(v, [float(p_lo), float(p_hi)])
    return float(lo), float(hi)


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
    n = int(len(imgs))
    if n == 0:
        return
    if len(titles) != n:
        raise ValueError("titles must match imgs.")
    fig, axs = plt.subplots(n, 1, figsize=(9.0, 2.3 * n), dpi=150, sharex=True, sharey=True)
    if n == 1:
        axs = [axs]
    ims = []
    for i in range(n):
        im = _imshow(axs[i], imgs[i], extent=extent, title=titles[i], vmin=vmin, vmax=vmax)
        ims.append(im)
        if i < n - 1:
            axs[i].set_xlabel("")
    fig.subplots_adjust(right=0.86, hspace=0.28)
    cax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
    fig.colorbar(ims[-1], cax=cax, label=str(cbar_label))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _plot_power2d_comparison(
    *,
    out_path: pathlib.Path,
    maps_2d_mk: list[tuple[str, np.ndarray]],
    pixel_res_rad: float,
    hit_mask_2d: np.ndarray | None,
) -> None:
    # Compute 2D power maps (cropped to tight bbox of hit mask).
    KX0, KY0, _ = power.power2d_from_map(
        map_2d_mk=maps_2d_mk[0][1],
        pixel_res_rad=pixel_res_rad,
        hit_mask_2d=hit_mask_2d,
        crop_to_hit_mask=True,
    )
    ps_list = []
    for _lab, m in maps_2d_mk:
        KX, KY, ps = power.power2d_from_map(
            map_2d_mk=m,
            pixel_res_rad=pixel_res_rad,
            hit_mask_2d=hit_mask_2d,
            crop_to_hit_mask=True,
        )
        if KX.shape != KX0.shape or KY.shape != KY0.shape:
            raise RuntimeError("Internal error: power2d grids do not match.")
        ps_list.append(ps * 1e6)  # (mK)^2 -> (uK)^2 for display

    vmax = float(np.max([float(np.max(ps)) for ps in ps_list]))
    vmin = max(vmax * 1e-3, 1e-30)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 30)

    n_pan = int(len(maps_2d_mk))
    fig, axes = plt.subplots(1, n_pan, figsize=(4.8 * n_pan, 4.2), sharex=True, sharey=True, dpi=150)
    if n_pan == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.22, right=0.86)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under(cmap(0.0))

    last_cf = None
    for i, (label, _m) in enumerate(maps_2d_mk):
        ax = axes[i]
        cf = ax.contourf(KX0, KY0, ps_list[i], levels=levels, cmap=cmap, norm=norm, extend="min")
        last_cf = cf
        ax.set_title(str(label), fontsize=10)
        ax.set_xlabel(r"$\ell_x$ [rad$^{-1}$]")
        if i == 0:
            ax.set_ylabel(r"$\ell_y$ [rad$^{-1}$]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)

    if last_cf is not None:
        pan = axes[min(1, n_pan - 1)]
        pos = pan.get_position()
        cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
        cb = fig.colorbar(last_cf, cax=cax, orientation="vertical", ticks=[vmin, vmax])
        cb.ax.set_yticklabels([f"{vmin:.0e}", f"{vmax:.0e}"])
        cb.set_label(r"$C_\ell$ [$\mu K_{\rm CMB}^2$]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_nondegenerate_power2d_delta(
    *,
    out_path: pathlib.Path,
    pre_coadd_all: np.ndarray,
    c_comb: np.ndarray,
    pixel_res_rad: float,
    hit_mask_2d: np.ndarray,
    w_mean: np.ndarray,
) -> None:
    w_norm = float(np.hypot(float(w_mean[0]), float(w_mean[1])))
    if not (np.isfinite(w_norm) and w_norm > 0):
        return

    KX, KY, ps_co = power.power2d_from_map(map_2d_mk=pre_coadd_all, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)
    _, _, ps_re = power.power2d_from_map(map_2d_mk=c_comb, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)

    ell = np.sqrt(KX * KX + KY * KY)
    denom = np.maximum(ell * w_norm, 1e-300)
    cosang = (KX * float(w_mean[0]) + KY * float(w_mean[1])) / denom
    ky_keep = _exclude_ky_zero_row(KY)
    keep = np.isfinite(cosang) & (np.abs(cosang) >= float(KDOTW_EXCLUDE_COS)) & ky_keep

    delta = (ps_re - ps_co) * 1e6  # (mK)^2 -> (uK)^2
    delta = np.where(keep, delta, np.nan)

    disp = (np.abs(KX) <= 500.0) & (np.abs(KY) <= 500.0) & np.isfinite(delta)
    vals = np.abs(delta[disp])
    if vals.size == 0:
        return
    vmax = float(np.percentile(vals, 99))
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.2), dpi=150)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))
    im = ax.imshow(
        delta,
        origin="lower",
        extent=[float(np.min(KX)), float(np.max(KX)), float(np.min(KY)), float(np.max(KY))],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="equal",
    )
    ax.set_title("Non-degenerate Δ power (recon - coadd)", fontsize=10)
    ax.set_xlabel(r"$\ell_x$ [rad$^{-1}$]")
    ax.set_ylabel(r"$\ell_y$ [rad$^{-1}$]")
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    cb = fig.colorbar(im, ax=ax, orientation="vertical")
    cb.set_label(r"$\Delta C_\ell$ [$\mu K_{\rm CMB}^2$]")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_nondegenerate_fourier_and_realspace(
    *,
    out_dir: pathlib.Path,
    pre_coadd_all: np.ndarray,
    c_comb: np.ndarray,
    hit_mask_2d: np.ndarray,
    pixel_res_rad: float,
    extent: list[float],
    w_mean: np.ndarray,
) -> None:
    w_norm = float(np.hypot(float(w_mean[0]), float(w_mean[1])))
    if not (np.isfinite(w_norm) and w_norm > 0):
        return

    KX, KY, ps_co = power.power2d_from_map(map_2d_mk=pre_coadd_all, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)
    _, _, ps_re = power.power2d_from_map(map_2d_mk=c_comb, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)

    ell = np.sqrt(KX * KX + KY * KY)
    denom = np.maximum(ell * w_norm, 1e-300)
    cosang = (KX * float(w_mean[0]) + KY * float(w_mean[1])) / denom
    ky_keep = _exclude_ky_zero_row(KY)
    keep = np.isfinite(cosang) & (np.abs(cosang) >= float(KDOTW_EXCLUDE_COS)) & ky_keep

    ps_co_m = np.ma.masked_invalid(np.where(keep, ps_co * 1e6, np.nan))
    ps_re_m = np.ma.masked_invalid(np.where(keep, ps_re * 1e6, np.nan))

    disp = (np.abs(KX) <= 500.0) & (np.abs(KY) <= 500.0)
    vals = np.concatenate([ps_co_m[disp].compressed(), ps_re_m[disp].compressed()])
    vmax = float(np.max(vals)) if vals.size > 0 else 1.0
    vmin = max(vmax * 1e-5, 1e-30)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 30)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True, dpi=150)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))
    cmap.set_under(cmap(0.0))

    cf0 = axes[0].contourf(KX, KY, ps_co_m, levels=levels, cmap=cmap, norm=norm, extend="min")
    axes[0].set_title("Naive coadd (non-degenerate modes)", fontsize=10)
    axes[1].contourf(KX, KY, ps_re_m, levels=levels, cmap=cmap, norm=norm, extend="min")
    axes[1].set_title("Combined recon (non-degenerate modes)", fontsize=10)
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
    fig.savefig(out_dir / "maps_coadd_vs_combined_power2d_nondegenerate.png", bbox_inches="tight")
    plt.close(fig)

    ell_m, cl_co_m = power.radial_cl_1d_from_power2d(KX=KX, KY=KY, ps2d_mk2=ps_co, n_ell_bins=int(N_ELL_BINS), keep_mask_2d=keep)
    _, cl_re_m = power.radial_cl_1d_from_power2d(KX=KX, KY=KY, ps2d_mk2=ps_re, n_ell_bins=int(N_ELL_BINS), keep_mask_2d=keep)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ax.plot(ell_m, cl_co_m, color="k", lw=2.0, label="naive coadd (non-degenerate modes)")
    ax.plot(ell_m, cl_re_m, color="C3", lw=2.5, label="combined recon (non-degenerate modes)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mK$^2$]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "cl_nondegenerate.png", bbox_inches="tight")
    plt.close(fig)

    frac_excl = float(np.mean(~keep[np.isfinite(keep)]))
    print(
        f"[non-degenerate modes] w_mean=({float(w_mean[0]):.3f},{float(w_mean[1]):.3f}) deg/s  "
        f"cos_thr={float(KDOTW_EXCLUDE_COS):.2f}  frac_excluded={frac_excl:.3f}",
        flush=True,
    )

    def _band_ratio(ell1, a1, b1, e0, e1):
        m = (ell1 >= float(e0)) & (ell1 <= float(e1)) & np.isfinite(a1) & np.isfinite(b1) & (b1 > 0)
        if not bool(np.any(m)):
            return None
        r = a1[m] / b1[m]
        return float(np.median(r)), float(np.percentile(r, 16)), float(np.percentile(r, 84)), int(r.size)

    for (e0, e1) in [(50, 300), (300, 1000), (1000, 3000)]:
        rr = _band_ratio(ell_m, cl_re_m, cl_co_m, e0, e1)
        if rr is None:
            continue
        med, p16, p84, n = rr
        print(f"  [cl masked] recon/coadd ell∈[{e0},{e1}] median={med:.3f} p16={p16:.3f} p84={p84:.3f} n={n}", flush=True)

    # Real-space maps with excluded Fourier modes removed (set excluded modes -> 0).
    ny_full, nx_full = int(pre_coadd_all.shape[0]), int(pre_coadd_all.shape[1])
    ell_x_full = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx_full, d=float(pixel_res_rad)))
    ell_y_full = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny_full, d=float(pixel_res_rad)))
    KXf, KYf = np.meshgrid(ell_x_full, ell_y_full, indexing="xy")
    ellf = np.sqrt(KXf * KXf + KYf * KYf)
    denomf = np.maximum(ellf * w_norm, 1e-300)
    cosf = (KXf * float(w_mean[0]) + KYf * float(w_mean[1])) / denomf
    ky_keep_f = _exclude_ky_zero_row(KYf)
    keepf = np.isfinite(cosf) & (np.abs(cosf) >= float(KDOTW_EXCLUDE_COS)) & ky_keep_f

    def _realspace_filtered(m2d: np.ndarray) -> np.ndarray:
        img = np.asarray(m2d, dtype=np.float64)
        img = np.where(hit_mask_2d & np.isfinite(img), img, 0.0)
        F = np.fft.fftshift(np.fft.fft2(img))
        out = np.fft.ifft2(np.fft.ifftshift(F * keepf)).real
        return np.where(hit_mask_2d, out, np.nan).astype(np.float64, copy=False)

    pre_nd = _realspace_filtered(pre_coadd_all)
    rec_nd = _realspace_filtered(c_comb)
    vmin_nd, vmax_nd = _robust_vmin_vmax(np.concatenate([pre_nd.ravel(), rec_nd.ravel()]))

    _plot_map_stack(
        out_path=out_dir / "maps_coadd_vs_combined_nondegenerate.png",
        imgs=[pre_nd, rec_nd],
        titles=[
            "Naive coadd (non-degenerate modes) [mK]",
            "Combined reconstruction (non-degenerate modes) [mK]",
        ],
        extent=extent,
        vmin=vmin_nd,
        vmax=vmax_nd,
    )


def _grid_focal_plane_frame(
    *,
    eff_offsets_arcmin: np.ndarray,
    eff_counts: np.ndarray,
    eff_tod_mk_frame: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    box_arcmin: float,
) -> tuple[np.ndarray, list[float]]:
    """
    Grid a single frame of TOD onto the focal plane (bilinear deposition).
    Returns (img, extent).
    """
    nx = int(np.ceil((x_max - x_min) / box_arcmin))
    ny = int(np.ceil((y_max - y_min) / box_arcmin))

    x = np.asarray(eff_offsets_arcmin[:, 0], dtype=np.float64)
    y = np.asarray(eff_offsets_arcmin[:, 1], dtype=np.float64)
    xi = (x - float(x_min)) / float(box_arcmin) - 0.5
    yi = (y - float(y_min)) / float(box_arcmin) - 0.5
    xi = np.clip(xi, 0.0, float(nx - 1))
    yi = np.clip(yi, 0.0, float(ny - 1))

    ix0 = np.floor(xi).astype(np.int64)
    iy0 = np.floor(yi).astype(np.int64)
    ix1 = np.minimum(ix0 + 1, nx - 1)
    iy1 = np.minimum(iy0 + 1, ny - 1)
    fx = xi - ix0
    fy = yi - iy0

    eff_counts = np.asarray(eff_counts, dtype=np.float64)
    w00 = (1.0 - fx) * (1.0 - fy) * eff_counts
    w10 = fx * (1.0 - fy) * eff_counts
    w01 = (1.0 - fx) * fy * eff_counts
    w11 = fx * fy * eff_counts

    v = np.asarray(eff_tod_mk_frame, dtype=np.float64)
    m = np.isfinite(v)

    s = np.zeros((ny, nx), dtype=np.float64)
    w = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(s, (iy0[m], ix0[m]), v[m] * w00[m])
    np.add.at(w, (iy0[m], ix0[m]), w00[m])
    np.add.at(s, (iy0[m], ix1[m]), v[m] * w10[m])
    np.add.at(w, (iy0[m], ix1[m]), w10[m])
    np.add.at(s, (iy1[m], ix0[m]), v[m] * w01[m])
    np.add.at(w, (iy1[m], ix0[m]), w01[m])
    np.add.at(s, (iy1[m], ix1[m]), v[m] * w11[m])
    np.add.at(w, (iy1[m], ix1[m]), w11[m])

    img = np.full((ny, nx), np.nan, dtype=np.float32)
    hit = w > 0
    img[hit] = (s[hit] / w[hit]).astype(np.float32)
    extent_fp = [float(x_min), float(x_max), float(y_min), float(y_max)]
    return img, extent_fp


def _plot_focal_plane_bad_pixels(*, out_path: pathlib.Path, scans, recons) -> None:
    n_scans = int(len(scans))
    n_fp = min(4, n_scans)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    axs = axs.flatten()

    for i in range(n_fp):
        ax = axs[i]
        z = scans[i]
        r = recons[i]

        eff_offsets = np.asarray(z["eff_offsets_arcmin"])
        eff_tod0 = np.asarray(z["eff_tod_mk"])[0]
        eff_counts = np.asarray(z["eff_counts"])
        mask_good = np.asarray(r["wind_valid_mask"], dtype=bool)

        img, extent_fp = _grid_focal_plane_frame(
            eff_offsets_arcmin=eff_offsets,
            eff_counts=eff_counts,
            eff_tod_mk_frame=eff_tod0,
            x_min=float(z["focal_x_min_arcmin"]),
            x_max=float(z["focal_x_max_arcmin"]),
            y_min=float(z["focal_y_min_arcmin"]),
            y_max=float(z["focal_y_max_arcmin"]),
            box_arcmin=float(z["effective_box_arcmin"]),
        )
        vmin_fp, vmax_fp = _robust_vmin_vmax(img)

        ax.imshow(
            img,
            origin="lower",
            extent=extent_fp,
            cmap="coolwarm",
            interpolation="nearest",
            vmin=vmin_fp,
            vmax=vmax_fp,
        )
        bad_idx = np.where(~mask_good)[0]
        if bad_idx.size > 0:
            bx = eff_offsets[bad_idx, 0]
            by = eff_offsets[bad_idx, 1]
            ax.scatter(bx, by, marker="x", c="k", s=40, linewidths=1.0, label="Dropped" if i == 0 else None)

        ax.set_title(f"scan{i:03d} FP (t=0) dropped={bad_idx.size}")
        ax.set_aspect("equal")
        ax.set_xlabel("x [arcmin]")
        ax.set_ylabel("y [arcmin]")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class Config:
    outputs_dir: str = "outputs_30arcmin"
    recon_mode: str = "map"  # 'ml' or 'map'


def main(cfg: Config) -> None:
    outputs_dir_name = str(cfg.outputs_dir)
    recon_mode = str(cfg.recon_mode).lower()
    if recon_mode not in ("ml", "map"):
        raise ValueError("recon_mode must be 'ml' or 'map'.")

    outputs_dir = DATA_DIR / outputs_dir_name
    pix_tag = outputs_dir_name[len("outputs_") :] if outputs_dir_name.startswith("outputs_") else outputs_dir_name
    recon_dir = OUT_DIR / f"reconstruction_{pix_tag}_{recon_mode}"
    out_dir = recon_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove deprecated plot names so reruns don't leave stale files.
    old_snr = out_dir / "snr_ratio_power2d.png"
    if old_snr.exists():
        old_snr.unlink()

    scan_paths = sorted(outputs_dir.glob("extract_binned_tod_scan*.npz"))
    if len(scan_paths) == 0:
        raise RuntimeError(f"No scan NPZs found under {outputs_dir}.")
    recon_paths = [recon_dir / f"recon_scan{i:03d}.npz" for i in range(len(scan_paths))]
    comb_path = recon_dir / f"recon_combined_{recon_mode}.npz"

    scans = [np.load(p, allow_pickle=False) for p in scan_paths]
    recons = [np.load(p, allow_pickle=False) for p in recon_paths]
    comb = np.load(comb_path, allow_pickle=False)

    n_scans = int(len(scans))
    pixel_size_deg = float(scans[0]["pixel_size_deg"])
    bbox_ix0 = int(comb["bbox_ix0"])
    bbox_iy0 = int(comb["bbox_iy0"])
    nx = int(comb["nx"])
    ny = int(comb["ny"])
    extent = _extent_deg(bbox_ix0=bbox_ix0, bbox_iy0=bbox_iy0, nx=nx, ny=ny, pixel_size_deg=pixel_size_deg)
    pixel_res_rad = float(pixel_size_deg) * np.pi / 180.0

    bbox = map_util.BBox(ix0=bbox_ix0, ix1=bbox_ix0 + nx - 1, iy0=bbox_iy0, iy1=bbox_iy0 + ny - 1)

    pre_maps, pre_hits = [], []
    for z in scans:
        m, h = map_util.coadd_map(
            eff_tod_mk=np.asarray(z["eff_tod_mk"], dtype=np.float32),
            pix_index=np.asarray(z["pix_index"], dtype=np.int64),
            bbox=bbox,
        )
        pre_maps.append(m)
        pre_hits.append(h)

    pre_coadd_all, hit_all = map_util.coadd_map_global(
        scans_eff_tod_mk=[np.asarray(z["eff_tod_mk"], dtype=np.float32) for z in scans],
        scans_pix_index=[np.asarray(z["pix_index"], dtype=np.int64) for z in scans],
        bbox=bbox,
    )
    hit_mask_2d = hit_all > 0

    c_comb = _img_from_vec(comb["c_hat_full_mk"], nx=nx, ny=ny)
    c_scans = [_img_from_vec(r["c_hat_full_mk"], nx=nx, ny=ny) for r in recons]

    vmin_comb, vmax_comb = _robust_vmin_vmax(np.concatenate([pre_coadd_all.ravel(), c_comb.ravel()]))
    vmin_pre, vmax_pre = _robust_vmin_vmax(np.concatenate([m.ravel() for m in pre_maps]))
    vmin_rec, vmax_rec = _robust_vmin_vmax(np.concatenate([m.ravel() for m in c_scans]))

    # 1) theoretical Cl vs measured Cl (pre maps)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ell0 = None
    for i, m in enumerate(pre_maps):
        ell, cl = power.radial_cl_1d_from_map(map_2d_mk=m, pixel_res_rad=pixel_res_rad, hit_mask=(pre_hits[i] > 0), n_ell_bins=int(N_ELL_BINS))
        ell0 = ell if ell0 is None else ell0
        ax.plot(ell, cl, lw=1.0, alpha=0.6, label=f"scan{i:03d} (measured)")
    if ell0 is not None:
        cl_th = power.atmospheric_power_spectrum_TQU(ell0, "TT", nu_GHz=float(NU_GHZ), epsilon_deg=float(EPSILON_DEG))
        ax.plot(ell0, cl_th, color="k", lw=2.0, label="theory (TT, Kolmogorov)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mK$^2$]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "cl_atmosphere.png", bbox_inches="tight")
    plt.close(fig)

    # 2) maps: naive coadd vs combined recon
    _plot_map_stack(
        out_path=out_dir / "maps_coadd_vs_combined.png",
        imgs=[pre_coadd_all, c_comb],
        titles=["Naive coadd (all scans) [mK]", "Combined reconstruction [mK]"],
        extent=extent,
        vmin=vmin_comb,
        vmax=vmax_comb,
    )

    # 2b) 2D Fourier power comparison
    _plot_power2d_comparison(
        out_path=out_dir / "maps_coadd_vs_combined_power2d.png",
        maps_2d_mk=[("Naive coadd (all scans)", pre_coadd_all), ("Combined reconstruction", c_comb)],
        pixel_res_rad=pixel_res_rad,
        hit_mask_2d=hit_mask_2d,
    )

    # 2c/2d) non-degenerate Fourier modes: power2d + Cl + filtered realspace maps
    w_mean = None
    try:
        ww = np.asarray(comb["winds_deg_per_s"], dtype=np.float64)
        if ww.ndim == 2 and ww.shape[1] >= 2 and np.all(np.isfinite(ww[:, :2])):
            w_mean = np.mean(ww[:, :2], axis=0)
    except Exception:
        w_mean = None
    if w_mean is not None:
        _plot_nondegenerate_fourier_and_realspace(
            out_dir=out_dir,
            pre_coadd_all=pre_coadd_all,
            c_comb=c_comb,
            hit_mask_2d=hit_mask_2d,
            pixel_res_rad=pixel_res_rad,
            extent=extent,
            w_mean=w_mean,
        )
        if recon_mode == "ml":
            _plot_nondegenerate_power2d_delta(
                out_path=out_dir / "power2d_nondegenerate_delta.png",
                pre_coadd_all=pre_coadd_all,
                c_comb=c_comb,
                hit_mask_2d=hit_mask_2d,
                pixel_res_rad=pixel_res_rad,
                w_mean=w_mean,
            )

    # 3) per-scan naive coadds
    _plot_map_stack(
        out_path=out_dir / "maps_coadd_scans.png",
        imgs=pre_maps,
        titles=[f"scan{i:03d} naive coadd [mK]" for i in range(n_scans)],
        extent=extent,
        vmin=vmin_pre,
        vmax=vmax_pre,
    )

    # 4) per-scan reconstructed maps
    titles = []
    for i in range(n_scans):
        w = np.asarray(recons[i]["wind_deg_per_s"], dtype=np.float64).reshape(-1)
        sx = float(recons[i]["wind_sigma_x_deg_per_s"])
        sy = float(recons[i]["wind_sigma_y_deg_per_s"])
        if w.size >= 2 and np.all(np.isfinite(w[:2])):
            if np.isfinite(sx) and np.isfinite(sy):
                w_title = f"w=({w[0]:.2f},{w[1]:.2f}) ± ({sx:.2f},{sy:.2f}) deg/s"
            else:
                w_title = f"w=({w[0]:.2f},{w[1]:.2f}) deg/s"
        else:
            w_title = "w=(nan,nan) deg/s"
        titles.append(f"scan{i:03d} reconstructed CMB [mK]\n{w_title}")

    _plot_map_stack(
        out_path=out_dir / "maps_recon_scans.png",
        imgs=c_scans,
        titles=titles,
        extent=extent,
        vmin=vmin_rec,
        vmax=vmax_rec,
    )

    # 5) power spectrum summary
    ell, cl_naive = power.radial_cl_1d_from_map(map_2d_mk=pre_coadd_all, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask_2d, n_ell_bins=int(N_ELL_BINS))
    _, cl_comb = power.radial_cl_1d_from_map(map_2d_mk=c_comb, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask_2d, n_ell_bins=int(N_ELL_BINS))

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    for i in range(n_scans):
        _, cl_pre_i = power.radial_cl_1d_from_map(map_2d_mk=pre_maps[i], pixel_res_rad=pixel_res_rad, hit_mask=(pre_hits[i] > 0), n_ell_bins=int(N_ELL_BINS))
        ax.plot(ell, cl_pre_i, color="0.6", lw=1.0, alpha=0.35)
    for i in range(n_scans):
        _, cl_rec_i = power.radial_cl_1d_from_map(map_2d_mk=c_scans[i], pixel_res_rad=pixel_res_rad, hit_mask=hit_mask_2d, n_ell_bins=int(N_ELL_BINS))
        ax.plot(ell, cl_rec_i, color="C0", lw=1.0, alpha=0.35)

    ax.plot(ell, cl_naive, color="k", lw=2.0, label="naive coadd (all scans)")
    ax.plot(ell, cl_comb, color="C3", lw=2.5, label="combined reconstruction")
    ax.plot([], [], color="0.6", lw=2.0, label="coadd (per scan)")
    ax.plot([], [], color="C0", lw=2.0, label="recon (per scan)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mK$^2$]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "cl.png", bbox_inches="tight")
    plt.close(fig)

    # 6) focal plane snapshot with dropped detectors
    _plot_focal_plane_bad_pixels(out_path=out_dir / "focal_plane_bad_pixels.png", scans=scans, recons=recons)

    for z in scans + recons + [comb]:
        z.close()
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    for recon_mode in ("ml", "map"):
        main(Config(outputs_dir="outputs_10arcmin_combined", recon_mode=str(recon_mode)))

