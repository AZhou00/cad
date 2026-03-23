#!/usr/bin/env python3
"""
Compare margined synthesis maps to Planck SMICA T in the same RA/Dec patch.

Requires a margined synthesis npz with uncertain_mode_vectors having at least max(K_MODES_REMOVED) columns.
NPZ schema: cad.parallel_solve.synthesize_scan (required keys validated via assert_synthesis_npz_keys).

CLI:
  python plot_comparison_planck.py <recon_combined_ml_margined.npz>

Writes under <synthesized>/plots_margined/planck_compare/:
  cl_vs_planck_ml.png, coherence_vs_planck_ml.png, maps_planck_vs_deproj_ml.png
Each includes naive coadd (dashed gray) for C_ell and coherence vs Planck. Map figure: Planck and synth rows share one color bar; naive coadd is last with its own scale.
"""

from __future__ import annotations

import pathlib
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

THIS_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = THIS_DIR.parent
DATA_DIR = CAD_DIR / "data"

# Processed Planck T + masks (see cad.planck.PlanckMapLoader).
PLANCK_DATA_DIR = pathlib.Path("/global/homes/j/junzhez/isotropy/data/cmb_full_sky")

K_MODES_REMOVED = (10, 50, 100, 200)
N_ELL_BINS = 48

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cad import map as map_util
from cad import power
from cad.parallel_solve.artifact_io import assert_synthesis_npz_keys
from cad.planck import PlanckMapLoader
from cad.plot_util import deproject_uncertain_modes, img_from_vec

import plot_reconstruction as pr


def _recon_deproj_2d(
    c_hat_obs: np.ndarray,
    good_mask: np.ndarray,
    obs_pix_global: np.ndarray,
    uncertain_vectors: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Deproject along columns of uncertain_vectors; return (ny, nx) in mK."""
    n_pix = int(nx * ny)
    rec_filt = deproject_uncertain_modes(c_hat_obs, good_mask, uncertain_vectors)
    rec_full = np.full(n_pix, np.nan, dtype=np.float64)
    rec_full[obs_pix_global] = rec_filt
    return img_from_vec(rec_full, nx=nx, ny=ny)


def _sample_healpix_to_patch(
    m_ring: np.ndarray,
    mask_ring: np.ndarray,
    *,
    bbox_ix0: int,
    bbox_iy0: int,
    nx: int,
    ny: int,
    pixel_size_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample HEALPix (Celestial RING, lon=RA, lat=Dec deg) onto the synthesis plate-carrée grid.

    Pixel centers match plot_reconstruction: RA = (ix0+ix+0.5)*pixel_size_deg,
    Dec = (iy0+iy+0.5)*pixel_size_deg (same axes as bbox / extent). This is not a gnomonic
    tangent-plane projection; it matches the flat-sky FFT geometry used in cad/power.py.
    RA is wrapped to [0, 360) deg for healpy; Dec is clipped to [-90, 90].

    Returns:
      t_mk: (ny, nx) temperature in mK (Planck SMICA is stored in microkelvin).
      ok: (ny, nx) bool, True where Planck common mask > 0.5 at interpolated point.
    """
    ps = float(pixel_size_deg)
    ra_c = (bbox_ix0 + np.arange(nx, dtype=np.float64) + 0.5) * ps
    dec_c = (bbox_iy0 + np.arange(ny, dtype=np.float64) + 0.5) * ps
    ra_g, dec_g = np.meshgrid(ra_c, dec_c, indexing="xy")  # (ny, nx)
    # healpy lonlat=True: first longitude (RA), second latitude (Dec), degrees
    ra_h = np.mod(ra_g, 360.0)
    dec_h = np.clip(dec_g, -90.0, 90.0)
    t_uK = hp.get_interp_val(m_ring, ra_h.ravel(), dec_h.ravel(), lonlat=True).reshape(ny, nx)
    w = hp.get_interp_val(mask_ring, ra_h.ravel(), dec_h.ravel(), lonlat=True).reshape(ny, nx)
    t_mk = np.asarray(t_uK, dtype=np.float64) * 1e-3
    ok = w > 0.5
    return t_mk, ok


def _radial_fft_coherence(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    pixel_res_rad: float,
    hit_mask: np.ndarray,
    n_ell_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per ell-bin magnitude coherence |sum F_a F_b*| / sqrt(sum|F_a|^2 sum|F_b|^2).

    map_*: (ny, nx) mK; hit_mask (ny, nx) bool. Returns ell_centers (n_bins,), rho (n_bins,).
    """
    a = np.where(hit_mask & np.isfinite(map_a), np.asarray(map_a, dtype=np.float64), 0.0)
    b = np.where(hit_mask & np.isfinite(map_b), np.asarray(map_b, dtype=np.float64), 0.0)
    ny, nx = a.shape
    dx = float(pixel_res_rad)
    ell_x = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))
    ell_y = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny, d=dx))
    kx, ky = np.meshgrid(ell_x, ell_y, indexing="xy")
    ell = np.sqrt(kx * kx + ky * ky)

    fa = np.fft.fftshift(np.fft.fft2(a))
    fb = np.fft.fftshift(np.fft.fft2(b))
    cross = fa * np.conj(fb)
    p1 = np.abs(fa) ** 2
    p2 = np.abs(fb) ** 2

    ell_m = ell.ravel()
    cross_m = cross.ravel()
    p1_m = p1.ravel()
    p2_m = p2.ravel()
    use = np.isfinite(ell_m) & (ell_m > 0)
    ell_max = float(np.max(ell_m[use])) if np.any(use) else 1.0
    edges = np.linspace(0.0, ell_max, int(n_ell_bins) + 1)
    ibin = np.digitize(ell_m, edges) - 1
    m = use & (ibin >= 0) & (ibin < int(n_ell_bins))

    num_c = np.zeros((int(n_ell_bins),), dtype=np.complex128)
    d1 = np.zeros((int(n_ell_bins),), dtype=np.float64)
    d2 = np.zeros((int(n_ell_bins),), dtype=np.float64)
    np.add.at(num_c, ibin[m], cross_m[m])
    np.add.at(d1, ibin[m], p1_m[m])
    np.add.at(d2, ibin[m], p2_m[m])

    rho = np.full((int(n_ell_bins),), np.nan, dtype=np.float64)
    good = (d1 > 0) & (d2 > 0)
    rho[good] = np.abs(num_c[good]) / np.sqrt(d1[good] * d2[good])
    ell_centers = 0.5 * (edges[:-1] + edges[1:])
    return ell_centers, rho


def main(synthesis_npz: pathlib.Path) -> None:
    npz = pathlib.Path(synthesis_npz).resolve()
    if "_margined_" not in npz.stem and not npz.stem.startswith("recon_combined_ml_margined"):
        raise ValueError("This script expects a margined synthesis npz (e.g. recon_combined_ml_margined.npz).")

    out_dir = npz.parent / "plots_margined" / "planck_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz, allow_pickle=True) as rc:
        assert_synthesis_npz_keys(rc)
        bbox_ix0 = int(rc["bbox_ix0"])
        bbox_iy0 = int(rc["bbox_iy0"])
        nx = int(rc["nx"])
        ny = int(rc["ny"])
        pixel_size_deg = float(rc["pixel_size_deg"])
        c_hat_obs = np.asarray(rc["c_hat_obs"], dtype=np.float64)
        obs_pix_global = np.asarray(rc["obs_pix_global"], dtype=np.int64)
        good_mask = np.asarray(rc["good_mask"], dtype=bool)
        uncertain_vectors = np.asarray(rc["uncertain_mode_vectors"], dtype=np.float64)
        scan_metadata = rc["scan_metadata"].tolist() if "scan_metadata" in rc else []

    k_max = uncertain_vectors.shape[1]
    k_use = [k for k in K_MODES_REMOVED if k <= k_max]
    if not k_use:
        raise ValueError(f"uncertain_mode_vectors has k={k_max}; need at least min({K_MODES_REMOVED}).")

    field_id = npz.parent.parent.name
    obs_ids = list(dict.fromkeys(m["observation_id"] for m in scan_metadata)) if scan_metadata else []
    tod_paths: list[pathlib.Path] = []
    for oid in obs_ids:
        tod_paths.extend(pr._binned_tod_paths(DATA_DIR / field_id / oid))

    bbox = map_util.BBox(ix0=bbox_ix0, ix1=bbox_ix0 + nx - 1, iy0=bbox_iy0, iy1=bbox_iy0 + ny - 1)
    extent = pr._extent_deg(bbox=bbox, pixel_size_deg=pixel_size_deg)
    pixel_res_rad = pixel_size_deg * np.pi / 180.0

    naive, hit_mask = pr._naive_coadd(tod_paths, bbox) if tod_paths else (
        np.full((ny, nx), np.nan, dtype=np.float32),
        np.zeros((ny, nx), dtype=bool),
    )
    if not np.any(hit_mask) and tod_paths:
        hit_mask = np.isfinite(naive)
    if not np.any(hit_mask):
        buf = np.zeros(int(nx * ny), dtype=np.float32)
        buf[obs_pix_global] = 1.0
        hit_mask = img_from_vec(buf, nx=nx, ny=ny) > 0.5

    pbar = tqdm(total=2 + len(k_use), desc="planck_compare", unit="step")
    pk = PlanckMapLoader(str(PLANCK_DATA_DIR)).load_smica_TQU1024(load_pol=False)
    t_planck_mk, planck_ok = _sample_healpix_to_patch(
        pk["T"],
        pk["T_mask_common"],
        bbox_ix0=bbox_ix0,
        bbox_iy0=bbox_iy0,
        nx=nx,
        ny=ny,
        pixel_size_deg=pixel_size_deg,
    )
    pbar.update(1)
    common = hit_mask & planck_ok

    naive_2d = np.asarray(naive, dtype=np.float64)
    naive_m = np.where(common, np.where(np.isfinite(naive_2d), naive_2d, np.nan), np.nan)

    ell_ref, cl_pl = power.radial_cl_1d_from_map(
        map_2d_mk=t_planck_mk,
        pixel_res_rad=pixel_res_rad,
        hit_mask=common,
        n_ell_bins=N_ELL_BINS,
    )
    _, cl_naive = power.radial_cl_1d_from_map(
        map_2d_mk=naive_m,
        pixel_res_rad=pixel_res_rad,
        hit_mask=common,
        n_ell_bins=N_ELL_BINS,
    )

    fig_cl, ax_cl = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ax_cl.plot(ell_ref, cl_pl, color="k", lw=2.0, label="Planck SMICA T")
    ax_cl.plot(ell_ref, cl_naive, color="0.45", lw=1.5, ls="--", label="Naive coadd")

    fig_rho, ax_rho = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ell_n, rho_pn = _radial_fft_coherence(
        t_planck_mk,
        naive_m,
        pixel_res_rad=pixel_res_rad,
        hit_mask=common,
        n_ell_bins=N_ELL_BINS,
    )
    ax_rho.plot(ell_n, rho_pn, color="0.45", lw=1.5, ls="--", label="Naive vs Planck")

    recons: dict[int, np.ndarray] = {}
    for k in k_use:
        vk = uncertain_vectors[:, :k]
        rec = _recon_deproj_2d(c_hat_obs, good_mask, obs_pix_global, vk, nx, ny)
        rec_m = np.where(common, np.where(np.isfinite(rec), rec, np.nan), np.nan).astype(np.float64)
        recons[k] = rec_m

        _, cl_r = power.radial_cl_1d_from_map(
            map_2d_mk=rec_m,
            pixel_res_rad=pixel_res_rad,
            hit_mask=common,
            n_ell_bins=N_ELL_BINS,
        )
        ax_cl.plot(ell_ref, cl_r, lw=1.2, label=f"synth, {k} modes removed")

        ell_c, rho = _radial_fft_coherence(
            t_planck_mk,
            rec_m,
            pixel_res_rad=pixel_res_rad,
            hit_mask=common,
            n_ell_bins=N_ELL_BINS,
        )
        ax_rho.plot(ell_c, rho, lw=1.2, label=f"{k} modes removed")
        pbar.update(1)

    ax_cl.set_xscale("log")
    ax_cl.set_yscale("log")
    ax_cl.set_xlabel(r"$\ell$")
    ax_cl.set_ylabel(r"$C_\ell$ [mK$^2$]")
    ax_cl.grid(True, which="both", alpha=0.2)
    ax_cl.legend(fontsize=7, loc="best")
    fig_cl.tight_layout()
    fig_cl.savefig(out_dir / "cl_vs_planck_ml.png", bbox_inches="tight")
    plt.close(fig_cl)

    ax_rho.set_xscale("log")
    ax_rho.set_xlabel(r"$\ell$")
    ax_rho.set_ylabel(r"FFT-bin coherence $|\sum F_P F_S^*| / \sqrt{\sum|F_P|^2\sum|F_S|^2}$")
    ax_rho.set_ylim(-0.05, 1.05)
    ax_rho.grid(True, which="both", alpha=0.2)
    ax_rho.legend(fontsize=7, loc="best")
    fig_rho.tight_layout()
    fig_rho.savefig(out_dir / "coherence_vs_planck_ml.png", bbox_inches="tight")
    plt.close(fig_rho)

    # Maps: Planck + synth rows share one symmetric color bar; naive coadd last with its own bar
    planck_m = np.where(common, t_planck_mk, np.nan)
    main_stacks = [planck_m] + [recons[k] for k in k_use]
    main_titles = ["Planck SMICA T [mK]"] + [f"Synth ML, {k} uncertain modes removed [mK]" for k in k_use]
    nrows = len(main_stacks) + 1
    fig_m, axs = plt.subplots(nrows, 1, figsize=(8.0, 2.4 * nrows), dpi=150, sharex=True, sharey=True)
    axs_list = [axs] if nrows == 1 else list(axs)
    main_axs = axs_list[:-1]
    naive_ax = axs_list[-1]

    flat_main = np.concatenate([s[np.isfinite(s)] for s in main_stacks])
    lo, hi = pr._robust_vmin_vmax(flat_main)
    vlim = float(max(abs(lo), abs(hi)))
    vmin, vmax = -vlim, vlim
    ims_main = []
    for ax, img, ti in zip(main_axs, main_stacks, main_titles):
        im = pr._imshow(ax, img, extent=extent, title=ti, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ims_main.append(im)

    flat_n = naive_m[np.isfinite(naive_m)]
    lon, hin = pr._robust_vmin_vmax(flat_n)
    vlim_n = float(max(abs(lon), abs(hin)))
    vmin_n, vmax_n = -vlim_n, vlim_n
    im_naive = pr._imshow(naive_ax, naive_m, extent=extent, title="Naive coadd [mK]", vmin=vmin_n, vmax=vmax_n, cmap="RdBu_r")

    fig_m.subplots_adjust(right=0.88, hspace=0.28)
    pr._add_shared_colorbar(fig_m, main_axs, ims_main[-1], label="mK")
    div = make_axes_locatable(naive_ax)
    cax_n = div.append_axes("right", size="2.8%", pad=0.12)
    fig_m.colorbar(im_naive, cax=cax_n).set_label("mK")
    fig_m.savefig(out_dir / "maps_planck_vs_deproj_ml.png", bbox_inches="tight")
    plt.close(fig_m)
    pbar.update(1)
    pbar.close()

    print(f"Wrote plots to {out_dir}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_comparison_planck.py <recon_combined_ml_margined.npz>", file=sys.stderr)
        sys.exit(1)
    main(pathlib.Path(sys.argv[1]))
