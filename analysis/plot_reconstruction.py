#!/usr/bin/env python3
"""
Concise plotting for the current multi-field layout.

Assumptions (no backward compatibility):
  - input scans:
      cad/data/<dataset>/<obs_id>/binned_tod_*/<scan>.npz
  - per-scan recon outputs:
      cad/data/<dataset>_recon/<obs_id>/recon_scan###_<ml|map>.npz
  - combined recon outputs:
      cad/data/<dataset>_recon/recon_<ml|map>.npz

This script makes dataset-level plots by loading the precomputed combined
reconstruction (`recon_<ml|map>.npz`) and comparing it to naive coadd maps.

Outputs:
  cad/data/<dataset>_recon/plots/

Naming conventions:
  - every file ends with _ml or _map
  - per-obs scan stacks use <obs_id>_ as a prefix
"""

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from cad import map as map_util
from cad import power

# -----------------------------------------------------------------------------
# Edit here (single knob)
# -----------------------------------------------------------------------------

RECON_MODE = "ml"  # "ml" or "map"

# -----------------------------------------------------------------------------
# Internal defaults
# -----------------------------------------------------------------------------

N_ELL_BINS = 128
KDOTW_EXCLUDE_COS = 0.5  # keep modes with |cos(angle(k,w_mean))| >= this
PREFER_BINNED_SUBDIR = "binned_tod_10arcmin"
TOP_OBS_SCAN_STACKS = 5

THIS_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = THIS_DIR.parent
DATA_DIR = CAD_DIR / "data"


def _img_from_vec(vec: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    """
    vec uses pixel_index = iy + ix*ny. Return image (ny,nx) with iy as rows.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(int(nx), int(ny))
    return v.T  # (ny,nx)


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
    cm.set_bad(color=(1.0, 1.0, 1.0, 1.0))  # NaNs as white
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
    n = int(len(imgs))
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
    cb = fig.colorbar(ims[-1], cax=cax, orientation="vertical")
    cb.set_label(cbar_label)
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
    n_pan = int(len(maps_2d_mk))
    fig, axes = plt.subplots(1, n_pan, figsize=(4.8 * n_pan, 4.2), sharex=True, sharey=True, dpi=150)
    if n_pan == 1:
        axes = [axes]

    # Shared color range from displayed pixels.
    vmin, vmax = None, None
    for _name, m2d in maps_2d_mk:
        KX, KY, ps2d = power.power2d_from_map(map_2d_mk=m2d, pixel_res_rad=pixel_res_rad, hit_mask_2d=hit_mask_2d)
        ps2d = np.ma.masked_invalid(ps2d * 1e6)  # (mK)^2 -> (uK)^2
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
    ky_min = float(np.min(ky_pos))
    return ky_abs >= ky_min


def _plot_nondegenerate(
    *,
    out_dir: pathlib.Path,
    suffix: str,
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
    fig.savefig(out_dir / f"maps_coadd_vs_reconcoadd_power2d_nondegenerate{suffix}.png", bbox_inches="tight")
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
    fig.savefig(out_dir / f"cl_nondegenerate{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    # Real-space maps with excluded Fourier modes removed (set excluded modes -> 0).
    ny, nx = int(naive.shape[0]), int(naive.shape[1])
    ell_x = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=float(pixel_res_rad)))
    ell_y = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny, d=float(pixel_res_rad)))
    KXf, KYf = np.meshgrid(ell_x, ell_y, indexing="xy")
    ellf = np.sqrt(KXf * KXf + KYf * KYf)
    denomf = np.maximum(ellf * w_norm, 1e-300)
    cosf = (KXf * float(w_mean[0]) + KYf * float(w_mean[1])) / denomf
    keepf = np.isfinite(cosf) & (np.abs(cosf) >= float(KDOTW_EXCLUDE_COS)) & _exclude_ky_zero_row(KYf)

    def _realspace_filtered(m2d: np.ndarray) -> np.ndarray:
        img = np.asarray(m2d, dtype=np.float64)
        img = np.where(hit_mask_2d & np.isfinite(img), img, 0.0)
        F = np.fft.fftshift(np.fft.fft2(img))
        out = np.fft.ifft2(np.fft.ifftshift(F * keepf)).real
        return np.where(hit_mask_2d, out, np.nan).astype(np.float64, copy=False)

    naive_nd = _realspace_filtered(naive)
    rec_nd = _realspace_filtered(rec)
    vmin, vmax = _robust_vmin_vmax(np.concatenate([naive_nd.ravel(), rec_nd.ravel()]))
    _plot_map_stack(
        out_path=out_dir / f"maps_coadd_vs_reconcoadd_nondegenerate{suffix}.png",
        imgs=[naive_nd, rec_nd],
        titles=["Naive coadd (non-degenerate modes) [mK]", "Recon coadd (non-degenerate modes) [mK]"],
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )


def _discover_obs_entries(dataset_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    obs_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)], key=lambda p: p.name)
    if obs_dirs:
        return [(str(p.name), p) for p in obs_dirs]
    return [(str(dataset_dir.name), dataset_dir)]


def _discover_scan_paths(*, obs_dir: pathlib.Path, max_scans: int | None = None) -> list[pathlib.Path]:
    binned_dirs = sorted([p for p in obs_dir.iterdir() if p.is_dir() and p.name.startswith("binned_tod_")])
    if len(binned_dirs) == 0:
        return []
    chosen = None
    for p in binned_dirs:
        if p.name == str(PREFER_BINNED_SUBDIR):
            chosen = p
            break
    if chosen is None:
        chosen = binned_dirs[0]
    scan_paths = sorted([p for p in chosen.iterdir() if p.is_file() and p.suffix == ".npz" and not p.name.startswith(".")])
    if max_scans is not None and int(max_scans) > 0:
        scan_paths = scan_paths[: int(max_scans)]
    return scan_paths


def _suffix(mode: str) -> str:
    return f"_{str(mode).lower()}"


def _plot_dataset(
    *,
    dataset_dir: pathlib.Path,
    recon_root: pathlib.Path,
    out_dir: pathlib.Path,
    recon_mode: str,
    max_scans: int | None,
) -> None:
    recon_mode = str(recon_mode).lower()
    if recon_mode not in ("ml", "map"):
        raise ValueError("RECON_MODE must be 'ml' or 'map'.")

    combined_path = recon_root / f"recon_{recon_mode}.npz"
    if not combined_path.exists():
        raise RuntimeError(
            f"Combined reconstruction not found: {combined_path}. "
            "Run run_synthesis.py first."
        )

    with np.load(combined_path, allow_pickle=False) as rc:
        bbox_ix0 = int(rc["bbox_ix0"])
        bbox_iy0 = int(rc["bbox_iy0"])
        nx = int(rc["nx"])
        ny = int(rc["ny"])
        pixel_size_deg = float(rc["pixel_size_deg"])
        rec_full = _img_from_vec(np.asarray(rc["c_hat_full_mk"], dtype=np.float64), nx=nx, ny=ny)
        winds_arr = np.asarray(rc["winds_deg_per_s"], dtype=np.float64) if "winds_deg_per_s" in rc.files else np.empty((0, 2), dtype=np.float64)

    bbox = map_util.BBox(ix0=bbox_ix0, ix1=bbox_ix0 + nx - 1, iy0=bbox_iy0, iy1=bbox_iy0 + ny - 1)
    extent = _extent_deg(bbox=bbox, pixel_size_deg=pixel_size_deg)
    pixel_res_rad = float(pixel_size_deg) * np.pi / 180.0

    # Streaming naive coadd accumulation over all scan samples.
    s_naive = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.float64)
    c_naive = np.zeros((int(bbox.ny), int(bbox.nx)), dtype=np.int64)
    scan_paths = []
    for _obs_id, obs_dir in _discover_obs_entries(dataset_dir):
        scan_paths.extend(_discover_scan_paths(obs_dir=obs_dir, max_scans=max_scans))

    if len(scan_paths) == 0:
        raise RuntimeError(f"No scan NPZ files found under {dataset_dir}.")

    for scan_path in scan_paths:
        with np.load(scan_path, allow_pickle=False) as z:
            eff_tod_mk = np.asarray(z["eff_tod_mk"])
            pix_index = np.asarray(z["pix_index"], dtype=np.int64)
        ok = np.isfinite(eff_tod_mk)
        if not bool(np.any(ok)):
            continue
        ij = pix_index[ok]  # (n_hit, 2) global (ix,iy)
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

    # For direct map comparisons, mask combined map to data-supported naive hit mask.
    rec = np.where(hit_mask, np.asarray(rec_full, dtype=np.float64), np.nan).astype(np.float32, copy=False)

    suf = _suffix(recon_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    vmin, vmax = _robust_vmin_vmax(np.concatenate([naive.ravel(), rec.ravel()]))
    _plot_map_stack(
        out_path=out_dir / f"maps_coadd_vs_reconcoadd{suf}.png",
        imgs=[naive, rec],
        titles=["Naive coadd (all scans) [mK]", f"Recon combined (loaded recon_{recon_mode}.npz) [{recon_mode.upper()}] [mK]"],
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    _plot_power2d_comparison(
        out_path=out_dir / f"maps_coadd_vs_reconcoadd_power2d{suf}.png",
        maps_2d_mk=[("Naive coadd", naive), (f"Recon combined ({recon_mode})", rec)],
        pixel_res_rad=pixel_res_rad,
        hit_mask_2d=hit_mask,
    )

    ell, cl_naive = power.radial_cl_1d_from_map(map_2d_mk=naive, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=int(N_ELL_BINS))
    _, cl_rec = power.radial_cl_1d_from_map(map_2d_mk=rec, pixel_res_rad=pixel_res_rad, hit_mask=hit_mask, n_ell_bins=int(N_ELL_BINS))
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ax.plot(ell, cl_naive, color="k", lw=2.0, label="naive coadd")
    ax.plot(ell, cl_rec, color="C3", lw=2.5, label=f"recon combined ({recon_mode})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mK$^2$]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"cl{suf}.png", bbox_inches="tight")
    plt.close(fig)

    if winds_arr.ndim == 2 and winds_arr.shape[1] >= 2 and int(winds_arr.shape[0]) > 0:
        w_mean = np.mean(np.asarray(winds_arr[:, :2], dtype=np.float64), axis=0)
        _plot_nondegenerate(
            out_dir=out_dir,
            suffix=suf,
            naive=naive,
            rec=rec,
            hit_mask_2d=hit_mask,
            pixel_res_rad=pixel_res_rad,
            extent=extent,
            w_mean=w_mean,
        )


def _plot_obs_scan_stacks(*, obs_id: str, scan_paths: list[pathlib.Path], recon_dir: pathlib.Path, out_dir: pathlib.Path, recon_mode: str) -> None:
    recon_mode = str(recon_mode).lower()
    if recon_mode not in ("ml", "map"):
        raise ValueError("mode must be 'ml' or 'map'.")

    n_show = min(int(len(scan_paths)), int(TOP_OBS_SCAN_STACKS))
    scan_paths_show = list(scan_paths[:n_show])
    recon_paths = [recon_dir / f"recon_scan{i:03d}_{recon_mode}.npz" for i in range(n_show)]
    if any([not p.exists() for p in recon_paths]):
        return

    # Use bbox from the first recon scan.
    r0 = np.load(recon_paths[0], allow_pickle=False)
    bbox_ix0 = int(r0["bbox_ix0"])
    bbox_iy0 = int(r0["bbox_iy0"])
    nx = int(r0["nx"])
    ny = int(r0["ny"])
    pixel_size_deg = float(r0["pixel_size_deg"])
    r0.close()

    bbox = map_util.BBox(ix0=bbox_ix0, ix1=bbox_ix0 + nx - 1, iy0=bbox_iy0, iy1=bbox_iy0 + ny - 1)
    extent = _extent_deg(bbox=bbox, pixel_size_deg=pixel_size_deg)

    scans = [np.load(p, allow_pickle=False) for p in scan_paths_show]
    recons = [np.load(p, allow_pickle=False) for p in recon_paths]

    pre_maps = []
    for z in scans:
        m, _h = map_util.coadd_map(
            eff_tod_mk=np.asarray(z["eff_tod_mk"], dtype=np.float32),
            pix_index=np.asarray(z["pix_index"], dtype=np.int64),
            bbox=bbox,
        )
        pre_maps.append(m)

    c_scans = [_img_from_vec(r["c_hat_full_mk"], nx=nx, ny=ny) for r in recons]

    vmin_pre, vmax_pre = _robust_vmin_vmax(np.concatenate([m.ravel() for m in pre_maps]))
    vmin_rec, vmax_rec = _robust_vmin_vmax(np.concatenate([m.ravel() for m in c_scans]))
    suf = _suffix(recon_mode)

    _plot_map_stack(
        out_path=out_dir / f"{obs_id}_maps_coadd_scans{suf}.png",
        imgs=pre_maps,
        titles=[f"obs{obs_id} scan{i:03d} naive coadd [mK]" for i in range(len(pre_maps))],
        extent=extent,
        vmin=vmin_pre,
        vmax=vmax_pre,
    )

    titles = []
    for i, r in enumerate(recons):
        w = np.asarray(r["wind_deg_per_s"], dtype=np.float64).reshape(-1)
        sx = float(r["wind_sigma_x_deg_per_s"])
        sy = float(r["wind_sigma_y_deg_per_s"])
        if w.size >= 2 and np.all(np.isfinite(w[:2])):
            if np.isfinite(sx) and np.isfinite(sy):
                w_title = f"w=({w[0]:.2f},{w[1]:.2f}) ± ({sx:.2f},{sy:.2f}) deg/s"
            else:
                w_title = f"w=({w[0]:.2f},{w[1]:.2f}) deg/s"
        else:
            w_title = "w=(nan,nan) deg/s"
        titles.append(f"obs{obs_id} scan{i:03d} reconstructed CMB [mK]\n{w_title}")

    _plot_map_stack(
        out_path=out_dir / f"{obs_id}_maps_recon_scans{suf}.png",
        imgs=c_scans,
        titles=titles,
        extent=extent,
        vmin=vmin_rec,
        vmax=vmax_rec,
    )

    for z in scans + recons:
        z.close()


def main(*, dataset: str, max_scans: int | None = None) -> None:
    mode = str(RECON_MODE).lower()
    if mode not in ("ml", "map"):
        raise ValueError("RECON_MODE must be 'ml' or 'map'.")

    dataset_dir = DATA_DIR / str(dataset)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory does not exist: {dataset_dir}")
    recon_root = dataset_dir.parent / f"{dataset_dir.name}_recon"
    if not recon_root.exists():
        raise RuntimeError(f"Recon directory does not exist: {recon_root} (run run_reconstruction.py and run_synthesis.py first)")

    out_dir = recon_root / "plots"
    _plot_dataset(
        dataset_dir=dataset_dir,
        recon_root=recon_root,
        out_dir=out_dir,
        recon_mode=mode,
        max_scans=max_scans,
    )

    # Per-obs scan stacks (top-N by number of scan files).
    rows = []
    for obs_id, obs_dir in _discover_obs_entries(dataset_dir):
        scan_paths = _discover_scan_paths(obs_dir=obs_dir, max_scans=max_scans)
        if len(scan_paths) == 0:
            continue
        rows.append((int(len(scan_paths)), str(obs_id), scan_paths))
    rows.sort(key=lambda x: (-x[0], x[1]))
    for _n, obs_id, scan_paths in rows[: int(TOP_OBS_SCAN_STACKS)]:
        _plot_obs_scan_stacks(obs_id=obs_id, scan_paths=scan_paths, recon_dir=recon_root / obs_id, out_dir=out_dir, recon_mode=mode)

    print(f"Wrote plots to {recon_root}", flush=True)


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) >= 2 else "ra0hdec-59.75"
    max_scans = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    main(dataset=str(dataset), max_scans=max_scans)

