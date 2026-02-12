#!/usr/bin/env python3
"""
End-to-end real-data reconstruction using `cad` (open-boundary frozen-screen model).

Inputs:
  - A dataset directory under `cad/data/<dataset_dir>/`.

Supported layout (organized by observation id):
  `cad/data/<dataset_dir>/<obs_id>/binned_tod_*/<scan>.npz`

Outputs:
  - `cad/data/<dataset_dir>_recon/<field_id>/recon_combined_<ml|map>.npz`
  - `cad/data/<dataset_dir>_recon/<field_id>/recon_scanXXX_<ml|map>.npz`
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import numpy as np

import cad
from cad import map as map_util
from cad import power
from cad import util


BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent
DATA_DIR = CAD_DIR / "data"

#TODO: The only preprocessing improvement I’d consider is storing a dec_ref (e.g., median boresight Dec per scan) in the NPZ, so the prior’s cos_dec can be set from the actual scan geometry rather than inferred from the bbox center. This is optional and not required for correctness.

@dataclass(frozen=True)
class Config:
    dataset_dir: str | None = None
    estimator_mode: str = "ML"  # 'ML' or 'MAP'
    n_ell_bins: int = 128
    cl_floor_mk2: float = 1e-12
    min_hits_per_pix: int = 1
    cg_tol: float = 5e-4
    cg_maxiter: int = 1200
    noise_per_raw_detector_per_153hz_sample_mk: float = 10.0
    prefer_binned_subdir: str = "binned_tod_10arcmin"


def _discover_fields(dataset_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """
    Return [(field_id, field_input_dir)].

    Multi-field convention: subdirectories named with digits are treated as obs ids.
    Otherwise, treat the dataset directory itself as a single field.
    """
    subdirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    obs = sorted([p for p in subdirs if re.fullmatch(r"\d+", p.name)], key=lambda p: p.name)
    return [(p.name, p) for p in obs]


def _discover_scan_paths(*, field_dir: pathlib.Path, cfg: Config) -> list[pathlib.Path]:
    """
    Return sorted scan NPZ paths for a field directory.

    Supports:
      - binned_tod_*/ subdirectories under field_dir
    """
    binned_dirs = sorted([p for p in field_dir.iterdir() if p.is_dir() and p.name.startswith("binned_tod_")])
    if len(binned_dirs) == 0:
        return []

    prefer = str(cfg.prefer_binned_subdir)
    chosen = None
    for p in binned_dirs:
        if p.name == prefer:
            chosen = p
            break
    if chosen is None:
        chosen = binned_dirs[0]

    scan_paths = sorted([p for p in chosen.iterdir() if p.is_file() and p.suffix == ".npz" and not p.name.startswith(".")])
    return scan_paths


def _load_scan(npz_path: pathlib.Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        # Keep the original schema; we will mask detectors later but keep the wind mask
        # in the outputs for focal-plane diagnostics.
        return dict(
            eff_tod_mk=np.asarray(z["eff_tod_mk"], dtype=np.float32),  # (n_t,n_eff)
            pix_index=np.asarray(z["pix_index"], dtype=np.int64),  # (n_t,n_eff,2)
            t_s=np.asarray(z["t_bin_center_s"], dtype=np.float64),  # (n_t,)
            pixel_size_deg=float(z["pixel_size_deg"]),
            eff_pos_deg=np.asarray(z["eff_pos_deg"], dtype=np.float32),  # (n_t,n_eff,2)
            boresight_pos_deg=np.asarray(z["boresight_pos_deg"], dtype=np.float32),  # (n_t,2)
            eff_counts=np.asarray(z["eff_counts"], dtype=np.float64),  # (n_eff,)
            eff_offsets_arcmin=np.asarray(z["eff_offsets_arcmin"], dtype=np.float64),  # (n_eff,2)
            bin_sec=float(z["bin_sec"]),
            sample_rate_hz=float(z["sample_rate_hz"]),
            # focal plane metadata (for plotting)
            focal_x_min_arcmin=float(z["focal_x_min_arcmin"]),
            focal_x_max_arcmin=float(z["focal_x_max_arcmin"]),
            focal_y_min_arcmin=float(z["focal_y_min_arcmin"]),
            focal_y_max_arcmin=float(z["focal_y_max_arcmin"]),
            effective_box_arcmin=float(z["effective_box_arcmin"]),
        )


def _print_cl_comparison(
    *,
    label_a: str,
    cl_a: np.ndarray,
    label_b: str,
    cl_b: np.ndarray,
    ell: np.ndarray,
    ell_min: float,
    ell_max: float,
) -> None:
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    cl_a = np.asarray(cl_a, dtype=np.float64).reshape(-1)
    cl_b = np.asarray(cl_b, dtype=np.float64).reshape(-1)
    m = np.isfinite(ell) & np.isfinite(cl_a) & np.isfinite(cl_b) & (ell >= float(ell_min)) & (ell <= float(ell_max)) & (cl_b > 0)
    if not bool(np.any(m)):
        print(f"[cl] {label_a} vs {label_b}: no finite bins in [{ell_min},{ell_max}].", flush=True)
        return
    ratio = cl_a[m] / cl_b[m]
    print(
        f"[cl] {label_a}/{label_b} in ell∈[{ell_min:.0f},{ell_max:.0f}] : "
        f"median={float(np.median(ratio)):.3f}  "
        f"p16={float(np.percentile(ratio,16)):.3f}  p84={float(np.percentile(ratio,84)):.3f}  "
        f"n={int(np.sum(m))}",
        flush=True,
    )


def _run_field(
    *,
    field_id: str,
    scan_paths: list[pathlib.Path],
    out_dir: pathlib.Path,
    mode: str,
    cfg: Config,
) -> None:
    scans0 = [_load_scan(p) for p in scan_paths]
    pixel_size_deg0 = float(scans0[0]["pixel_size_deg"])
    for s in scans0[1:]:
        if abs(float(s["pixel_size_deg"]) - pixel_size_deg0) > 1e-12:
            raise ValueError(f"[{field_id}] pixel_size_deg mismatch across scans.")

    print(f"[field] {field_id}: n_scans={len(scan_paths)}  mode={mode}", flush=True)

    # --- wind estimation on unmasked scans ---
    winds = np.zeros((len(scans0), 2), dtype=np.float64)
    wind_diags: list[dict] = []
    masks_good: list[np.ndarray] = []
    for i, s in enumerate(scans0):
        w_x, w_y, diag, mask = cad.estimate_wind_deg_per_s(
            eff_tod_mk=s["eff_tod_mk"],
            eff_pos_deg=s["eff_pos_deg"],
            boresight_pos_deg=s["boresight_pos_deg"],
            t_s=s["t_s"],
            eff_counts=s["eff_counts"],
            physical_degree=False,
        )
        winds[i, :] = (float(w_x), float(w_y))
        wind_diags.append(diag)
        masks_good.append(np.asarray(mask, dtype=bool))
        print(
            f"[wind] scan{i:03d} w=({w_x:.4f},{w_y:.4f}) deg/s  "
            f"sigma=({diag['wind_sigma_x_deg_per_s']:.4f},{diag['wind_sigma_y_deg_per_s']:.4f})  "
            f"snr_mag={diag['wind_snr_mag']:.2f}  "
            f"rms={diag['resid_rms_mk_per_s']:.3f} mK/s  good={int(mask.sum())}/{int(mask.size)}",
            flush=True,
        )

    # --- bbox_obs and atmosphere C_ell estimate (use unmasked data on bbox_obs) ---
    boxes = []
    for s in scans0:
        valid = np.isfinite(s["eff_tod_mk"])
        boxes.append(map_util.scan_bbox_from_pix_index(pix_index=s["pix_index"], valid_mask=valid))
    bbox_obs = map_util.bbox_union(boxes)

    coadd_obs_mk, hit_obs_2d = map_util.coadd_map_global(
        scans_eff_tod_mk=[s["eff_tod_mk"] for s in scans0],
        scans_pix_index=[s["pix_index"] for s in scans0],
        bbox=bbox_obs,
    )
    hit_mask = hit_obs_2d > 0
    pixel_res_rad = float(pixel_size_deg0) * np.pi / 180.0
    coadd_obs_mk = np.asarray(coadd_obs_mk, dtype=np.float64)
    hm = hit_mask & np.isfinite(coadd_obs_mk)
    if bool(np.any(hm)):
        coadd_obs_mk = coadd_obs_mk.copy()
        coadd_obs_mk[hm] -= float(np.mean(coadd_obs_mk[hm]))

    ell_centers, cl_atm_bins_mk2 = power.radial_cl_1d_from_map(
        map_2d_mk=coadd_obs_mk,
        pixel_res_rad=pixel_res_rad,
        hit_mask=hit_mask,
        n_ell_bins=int(cfg.n_ell_bins),
    )
    cl_atm_bins_mk2 = np.asarray(cl_atm_bins_mk2, dtype=np.float64)
    cl_atm_bins_mk2 = np.where(np.isfinite(cl_atm_bins_mk2), cl_atm_bins_mk2, 0.0)
    cl_atm_bins_mk2 = np.maximum(cl_atm_bins_mk2, float(cfg.cl_floor_mk2))

    if mode == "MAP":
        cl_cmb_bins_mk2 = np.asarray(power.cmb_power_spectrum(ell_centers), dtype=np.float64)
    else:
        cl_cmb_bins_mk2 = np.full_like(ell_centers, float(cfg.cl_floor_mk2), dtype=np.float64)
    cl_cmb_bins_mk2 = np.maximum(cl_cmb_bins_mk2, float(cfg.cl_floor_mk2))

    # --- apply bad pixel masking (use mask_good per scan) ---
    scans = []
    for i, s in enumerate(scans0):
        msk = masks_good[i]
        if int(msk.size) != int(s["eff_tod_mk"].shape[1]):
            raise RuntimeError("Internal error: mask_good length mismatch.")
        if int(msk.size - msk.sum()) > 0:
            print(f"[mask] scan{i:03d}: dropped {int(msk.size-msk.sum())}/{int(msk.size)} detectors", flush=True)
        s2 = dict(s)
        s2["eff_tod_mk"] = np.asarray(s["eff_tod_mk"])[:, msk]
        s2["pix_index"] = np.asarray(s["pix_index"], dtype=np.int64)[:, msk, :]
        s2["eff_pos_deg"] = np.asarray(s["eff_pos_deg"])[:, msk, :]
        s2["eff_counts"] = np.asarray(s["eff_counts"], dtype=np.float64)[msk]
        s2["eff_offsets_arcmin"] = np.asarray(s["eff_offsets_arcmin"], dtype=np.float64)[msk]
        scans.append(s2)

    # --- noise model per scan (after masking) ---
    scans_noise = []
    for i, s in enumerate(scans):
        sig = util.noise_std_eff_mk_from_counts(
            eff_counts=np.asarray(s["eff_counts"], dtype=np.float64),
            bin_sec=float(s["bin_sec"]),
            sample_rate_hz=float(s["sample_rate_hz"]),
            noise_per_raw_detector_per_153hz_sample_mk=float(cfg.noise_per_raw_detector_per_153hz_sample_mk),
        )
        scans_noise.append(sig)
        print(
            f"[noise] scan{i:03d}: sigma_eff median={float(np.median(sig)):.3f} mK  min={float(np.min(sig)):.3f}  max={float(np.max(sig)):.3f}",
            flush=True,
        )

    # --- combined reconstruction ---
    combined = cad.synthesize_scans(
        scans_tod_mk=[np.asarray(s["eff_tod_mk"], dtype=np.float64) for s in scans],
        scans_pix_index=[np.asarray(s["pix_index"], dtype=np.int64) for s in scans],
        scans_t_s=[np.asarray(s["t_s"], dtype=np.float64) for s in scans],
        winds_deg_per_s=[(float(w[0]), float(w[1])) for w in winds],
        scans_noise_std_det_mk=[np.asarray(sig, dtype=np.float64) for sig in scans_noise],
        pixel_size_deg=float(pixel_size_deg0),
        cl_atm_bins_mk2=np.asarray(cl_atm_bins_mk2, dtype=np.float64),
        cl_cmb_bins_mk2=np.asarray(cl_cmb_bins_mk2, dtype=np.float64),
        estimator_mode=mode,
        cl_floor_mk2=float(cfg.cl_floor_mk2),
        min_hits_per_pix=int(cfg.min_hits_per_pix),
        cg_tol=float(cfg.cg_tol),
        cg_maxiter=int(cfg.cg_maxiter),
    )

    bbox_cmb = combined.bbox_cmb
    nx = int(bbox_cmb.nx)
    ny = int(bbox_cmb.ny)
    n_pix = int(nx * ny)
    obs_pix_global = np.asarray(combined.obs_pix_cmb, dtype=np.int64)

    out_comb = out_dir / f"recon_combined_{mode.lower()}.npz"
    out_payload = dict(
        estimator_mode=np.array(mode),
        pixel_size_deg=np.float64(pixel_size_deg0),
        bbox_ix0=np.int64(bbox_cmb.ix0),
        bbox_iy0=np.int64(bbox_cmb.iy0),
        nx=np.int64(nx),
        ny=np.int64(ny),
        winds_deg_per_s=np.asarray(winds, dtype=np.float64),
        obs_pix_global=np.asarray(obs_pix_global, dtype=np.int64),
        c_hat_full_mk=np.asarray(combined.c_hat_full_mk, dtype=np.float64),
        ell_centers=np.asarray(ell_centers, dtype=np.float64),
        cl_atm_bins_mk2=np.asarray(cl_atm_bins_mk2, dtype=np.float64),
        cl_cmb_bins_mk2=np.asarray(cl_cmb_bins_mk2, dtype=np.float64),
        boundary=np.array("open"),
    )
    np.savez_compressed(out_comb, **out_payload)
    print(f"[write] {out_comb}", flush=True)

    # --- per-scan reconstructions on the same grid (for diagnostics) ---
    global_to_obs = -np.ones((n_pix,), dtype=np.int64)
    global_to_obs[obs_pix_global] = np.arange(int(obs_pix_global.size), dtype=np.int64)

    # Calculate cos_dec from bbox_cmb center.
    iy_mid = (float(bbox_cmb.iy0) + float(bbox_cmb.iy1)) / 2.0
    dec_deg = iy_mid * float(pixel_size_deg0)
    cos_dec = float(np.cos(np.deg2rad(dec_deg)))

    prior_cmb = cad.FourierGaussianPrior(
        nx=int(bbox_cmb.nx),
        ny=int(bbox_cmb.ny),
        pixel_res_rad=pixel_res_rad,
        cl_bins_mk2=np.asarray(cl_cmb_bins_mk2, dtype=np.float64),
        cos_dec=cos_dec,
        cl_floor_mk2=float(cfg.cl_floor_mk2),
    )

    for i, s in enumerate(scans):
        valid_i = np.isfinite(s["eff_tod_mk"])
        bbox_obs_i = map_util.scan_bbox_from_pix_index(pix_index=np.asarray(s["pix_index"], dtype=np.int64), valid_mask=valid_i)
        bbox_atm_i = util.bbox_pad_for_open_boundary(
            bbox_obs=bbox_obs_i,
            scans_pix_index=[np.asarray(s["pix_index"], dtype=np.int64)],
            scans_tod_mk=[np.asarray(s["eff_tod_mk"], dtype=np.float64)],
            scans_t_s=[np.asarray(s["t_s"], dtype=np.float64)],
            winds_deg_per_s=[(float(winds[i, 0]), float(winds[i, 1]))],
            pixel_size_deg=float(pixel_size_deg0),
        )
        prior_atm_i = cad.FourierGaussianPrior(
            nx=int(bbox_atm_i.nx),
            ny=int(bbox_atm_i.ny),
            pixel_res_rad=pixel_res_rad,
            cl_bins_mk2=np.asarray(cl_atm_bins_mk2, dtype=np.float64),
            cos_dec=cos_dec,
            cl_floor_mk2=float(cfg.cl_floor_mk2),
        )

        sol = cad.reconstruct_scan.solve_single_scan(
            tod_mk=np.asarray(s["eff_tod_mk"], dtype=np.float64),
            pix_index=np.asarray(s["pix_index"], dtype=np.int64),
            t_s=np.asarray(s["t_s"], dtype=np.float64),
            pixel_size_deg=float(pixel_size_deg0),
            wind_deg_per_s=(float(winds[i, 0]), float(winds[i, 1])),
            noise_std_det_mk=np.asarray(scans_noise[i], dtype=np.float64),
            prior_atm=prior_atm_i,
            prior_cmb=prior_cmb,
            bbox_cmb=bbox_cmb,
            bbox_atm=bbox_atm_i,
            obs_pix_cmb=obs_pix_global,
            global_to_obs_cmb=global_to_obs,
            estimator_mode=mode,
            n_scans=int(len(scans)),
            cl_floor_mk2=float(cfg.cl_floor_mk2),
            cg_tol=float(cfg.cg_tol),
            cg_maxiter=max(200, int(cfg.cg_maxiter // 2)),
        )

        out_path = out_dir / f"recon_scan{i:03d}_{mode.lower()}.npz"
        diag = wind_diags[i]
        out_payload = dict(
            scan_index=np.int64(i),
            estimator_mode=np.array(mode),
            pixel_size_deg=np.float64(pixel_size_deg0),
            bbox_ix0=np.int64(bbox_cmb.ix0),
            bbox_iy0=np.int64(bbox_cmb.iy0),
            nx=np.int64(nx),
            ny=np.int64(ny),
            wind_deg_per_s=np.asarray(winds[i], dtype=np.float64),
            wind_valid_mask=np.asarray(masks_good[i], dtype=bool),
            wind_resid_rms_mk_per_s=np.float64(diag["resid_rms_mk_per_s"]),
            wind_sigma_x_deg_per_s=np.float64(diag["wind_sigma_x_deg_per_s"]),
            wind_sigma_y_deg_per_s=np.float64(diag["wind_sigma_y_deg_per_s"]),
            wind_snr_mag=np.float64(diag["wind_snr_mag"]),
            noise_per_raw_detector_per_153hz_sample_mk=np.float64(cfg.noise_per_raw_detector_per_153hz_sample_mk),
            obs_pix_global=np.asarray(obs_pix_global, dtype=np.int64),
            c_hat_full_mk=np.asarray(sol.c_hat_full_mk, dtype=np.float64),
            ell_centers=np.asarray(ell_centers, dtype=np.float64),
            cl_atm_bins_mk2=np.asarray(cl_atm_bins_mk2, dtype=np.float64),
            cl_cmb_bins_mk2=np.asarray(cl_cmb_bins_mk2, dtype=np.float64),
            boundary=np.array("open"),
        )
        np.savez_compressed(out_path, **out_payload)
        print(f"[write] {out_path}", flush=True)

    # --- numeric Cl comparison: coadd vs combined reconstruction (same bbox_cmb) ---
    bbox_cmb_map = map_util.BBox(ix0=bbox_cmb.ix0, ix1=bbox_cmb.ix1, iy0=bbox_cmb.iy0, iy1=bbox_cmb.iy1)
    coadd_cmb_mk_masked, hit_cmb_2d_masked = map_util.coadd_map_global(
        scans_eff_tod_mk=[s["eff_tod_mk"] for s in scans],
        scans_pix_index=[s["pix_index"] for s in scans],
        bbox=bbox_cmb_map,
    )
    hit_cmb_mask_masked = hit_cmb_2d_masked > 0
    ell, cl_coadd_masked = power.radial_cl_1d_from_map(
        map_2d_mk=np.asarray(coadd_cmb_mk_masked, dtype=np.float64),
        pixel_res_rad=pixel_res_rad,
        hit_mask=hit_cmb_mask_masked,
        n_ell_bins=int(cfg.n_ell_bins),
    )

    coadd_cmb_mk_unmasked, hit_cmb_2d_unmasked = map_util.coadd_map_global(
        scans_eff_tod_mk=[s["eff_tod_mk"] for s in scans0],
        scans_pix_index=[s["pix_index"] for s in scans0],
        bbox=bbox_cmb_map,
    )
    hit_cmb_mask_unmasked = hit_cmb_2d_unmasked > 0
    _, cl_coadd_unmasked = power.radial_cl_1d_from_map(
        map_2d_mk=np.asarray(coadd_cmb_mk_unmasked, dtype=np.float64),
        pixel_res_rad=pixel_res_rad,
        hit_mask=hit_cmb_mask_unmasked,
        n_ell_bins=int(cfg.n_ell_bins),
    )

    c2 = np.asarray(combined.c_hat_full_mk, dtype=np.float64).reshape(nx, ny).T
    _, cl_recon = power.radial_cl_1d_from_map(
        map_2d_mk=c2,
        pixel_res_rad=pixel_res_rad,
        hit_mask=hit_cmb_mask_masked,
        n_ell_bins=int(cfg.n_ell_bins),
    )
    print(
        f"[grid] bbox_cmb ix=[{bbox_cmb.ix0},{bbox_cmb.ix1}] iy=[{bbox_cmb.iy0},{bbox_cmb.iy1}] nx={nx} ny={ny} n_pix={n_pix}",
        flush=True,
    )
    _print_cl_comparison(label_a="coadd(masked)", cl_a=cl_coadd_masked, label_b="coadd(unmasked)", cl_b=cl_coadd_unmasked, ell=ell, ell_min=50, ell_max=3000)
    _print_cl_comparison(label_a="recon", cl_a=cl_recon, label_b="coadd(masked)", cl_b=cl_coadd_masked, ell=ell, ell_min=50, ell_max=300)
    _print_cl_comparison(label_a="recon", cl_a=cl_recon, label_b="coadd(masked)", cl_b=cl_coadd_masked, ell=ell, ell_min=300, ell_max=1000)
    _print_cl_comparison(label_a="recon", cl_a=cl_recon, label_b="coadd(masked)", cl_b=cl_coadd_masked, ell=ell, ell_min=1000, ell_max=3000)


def main(cfg: Config) -> None:
    mode = str(cfg.estimator_mode).upper()
    if mode not in ("ML", "MAP"):
        raise ValueError("estimator_mode must be 'ML' or 'MAP'.")

    dataset_dir = DATA_DIR / str(cfg.dataset_dir)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory does not exist: {dataset_dir}")

    out_root = dataset_dir.parent / f"{dataset_dir.name}_recon"
    out_root.mkdir(parents=True, exist_ok=True)

    fields = _discover_fields(dataset_dir)
    if len(fields) == 0:
        raise RuntimeError(
            f"No obs-id subdirectories found under {dataset_dir}. "
            "Expected layout: <dataset>/<obs_id>/binned_tod_*/<scan>.npz"
        )

    for field_id, field_in_dir in fields:
        scan_paths = _discover_scan_paths(field_dir=field_in_dir, cfg=cfg)
        if len(scan_paths) == 0:
            print(f"[skip] field {field_id}: no scan NPZs found under {field_in_dir}", flush=True)
            continue
        out_dir = out_root / str(field_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_field(field_id=str(field_id), scan_paths=scan_paths, out_dir=out_dir, mode=mode, cfg=cfg)


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) >= 2 else "ra0hdec-59.75"
    for estimator_mode in ("ML", "MAP"):
        main(Config(dataset_dir=str(dataset), estimator_mode=str(estimator_mode)))

