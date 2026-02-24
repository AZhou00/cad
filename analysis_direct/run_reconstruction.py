#!/usr/bin/env python3
"""
Per-scan real-data reconstruction using `cad` (open-boundary frozen-screen model).

Inputs:
  - A dataset directory under `cad/data/<dataset_dir>/`.

Supported layout (organized by observation id):
  `cad/data/<dataset_dir>/<obs_id>/binned_tod_*/<scan>.npz`

Outputs:
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

@dataclass(frozen=True)
class Config:
    dataset_dir: str | None = None
    estimator_mode: str = ""  # 'ML' or 'MAP'
    n_ell_bins: int = 128
    cl_floor_mk2: float = 1e-12
    min_hits_per_pix: int = 1
    cg_tol: float = 5e-4
    cg_maxiter: int = 1200
    noise_per_raw_detector_per_153hz_sample_mk: float = 10.0
    prefer_binned_subdir: str = "binned_tod_10arcmin"
    max_scans: int | None = None


def _discover_fields(dataset_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """
    Return [(field_id, field_input_dir)].

    Multi-field convention: subdirectories named with digits are treated as obs ids.
    Otherwise, treat the dataset directory itself as a single field.
    """
    subdirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    obs = sorted([p for p in subdirs if re.fullmatch(r"\d+", p.name)], key=lambda p: p.name)
    if obs:
        return [(p.name, p) for p in obs]
    return [(dataset_dir.name, dataset_dir)]


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
    if cfg.max_scans is not None and int(cfg.max_scans) > 0:
        scan_paths = scan_paths[: int(cfg.max_scans)]
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


def _prepare_group_inputs(
    *,
    label: str,
    scan_paths: list[pathlib.Path],
    mode: str,
    cfg: Config,
) -> dict:
    scans0 = [_load_scan(p) for p in scan_paths]
    pixel_size_deg0 = float(scans0[0]["pixel_size_deg"])
    for s in scans0[1:]:
        if abs(float(s["pixel_size_deg"]) - pixel_size_deg0) > 1e-12:
            raise ValueError(f"[{label}] pixel_size_deg mismatch across scans.")

    print(f"[group] {label}: n_scans={len(scan_paths)}  mode={mode}", flush=True)

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
            f"[wind] {label} scan{i:03d} w=({w_x:.4f},{w_y:.4f}) deg/s  "
            f"sigma=({diag['wind_sigma_x_deg_per_s']:.4f},{diag['wind_sigma_y_deg_per_s']:.4f})  "
            f"snr_mag={diag['wind_snr_mag']:.2f}  "
            f"rms={diag['resid_rms_mk_per_s']:.3f} mK/s  good={int(mask.sum())}/{int(mask.size)}",
            flush=True,
        )

    pixel_res_rad = float(pixel_size_deg0) * np.pi / 180.0

    # --- apply bad pixel masking (use mask_good per scan) ---
    scans = []
    for i, s in enumerate(scans0):
        msk = masks_good[i]
        if int(msk.size) != int(s["eff_tod_mk"].shape[1]):
            raise RuntimeError("Internal error: mask_good length mismatch.")
        if int(msk.size - msk.sum()) > 0:
            print(f"[mask] {label} scan{i:03d}: dropped {int(msk.size-msk.sum())}/{int(msk.size)} detectors", flush=True)
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
            f"[noise] {label} scan{i:03d}: sigma_eff median={float(np.median(sig)):.3f} mK  min={float(np.min(sig)):.3f}  max={float(np.max(sig)):.3f}",
            flush=True,
        )

    # --- per-scan atmosphere C_ell on the shared ell grid ---
    ell_centers: np.ndarray | None = None
    cl_atm_bins_scan_mk2: list[np.ndarray] = []
    for i, s in enumerate(scans):
        valid_i = np.isfinite(s["eff_tod_mk"])
        bbox_i = map_util.scan_bbox_from_pix_index(
            pix_index=np.asarray(s["pix_index"], dtype=np.int64),
            valid_mask=valid_i,
        )
        coadd_i_mk, hit_i_2d = map_util.coadd_map_global(
            scans_eff_tod_mk=[np.asarray(s["eff_tod_mk"], dtype=np.float64)],
            scans_pix_index=[np.asarray(s["pix_index"], dtype=np.int64)],
            bbox=bbox_i,
        )
        hit_i = hit_i_2d > 0
        coadd_i_mk = np.asarray(coadd_i_mk, dtype=np.float64)
        hm_i = hit_i & np.isfinite(coadd_i_mk)
        if bool(np.any(hm_i)):
            coadd_i_mk = coadd_i_mk.copy()
            coadd_i_mk[hm_i] -= float(np.mean(coadd_i_mk[hm_i]))

        ell_i, cl_i = power.radial_cl_1d_from_map(
            map_2d_mk=coadd_i_mk,
            pixel_res_rad=pixel_res_rad,
            hit_mask=hit_i,
            n_ell_bins=int(cfg.n_ell_bins),
        )
        ell_i = np.asarray(ell_i, dtype=np.float64).reshape(-1)
        cl_i = np.asarray(cl_i, dtype=np.float64)
        cl_i = np.where(np.isfinite(cl_i), cl_i, 0.0)
        cl_i = np.maximum(cl_i, float(cfg.cl_floor_mk2))
        if ell_centers is None:
            ell_centers = np.asarray(ell_i, dtype=np.float64)
            cl_atm_bins_scan_mk2.append(np.asarray(cl_i, dtype=np.float64))
        elif np.array_equal(np.asarray(ell_i, dtype=np.float64), np.asarray(ell_centers, dtype=np.float64)):
            cl_atm_bins_scan_mk2.append(np.asarray(cl_i, dtype=np.float64))
        else:
            cl_i_interp = np.interp(
                np.asarray(ell_centers, dtype=np.float64),
                np.asarray(ell_i, dtype=np.float64),
                cl_i,
                left=float(cl_i[0]),
                right=float(cl_i[-1]),
            )
            cl_atm_bins_scan_mk2.append(np.asarray(cl_i_interp, dtype=np.float64))

    if ell_centers is None:
        raise RuntimeError(f"[{label}] failed to estimate per-scan atmospheric C_ell.")
    ell_centers = np.asarray(ell_centers, dtype=np.float64)

    if mode == "MAP":
        cl_cmb_bins_mk2 = np.asarray(power.cmb_power_spectrum(ell_centers), dtype=np.float64)
    else:
        cl_cmb_bins_mk2 = np.full_like(ell_centers, float(cfg.cl_floor_mk2), dtype=np.float64)
    cl_cmb_bins_mk2 = np.maximum(cl_cmb_bins_mk2, float(cfg.cl_floor_mk2))

    return dict(
        scans=scans,
        scans_noise=scans_noise,
        winds=winds,
        wind_diags=wind_diags,
        masks_good=masks_good,
        pixel_size_deg=np.float64(pixel_size_deg0),
        pixel_res_rad=np.float64(pixel_res_rad),
        ell_centers=np.asarray(ell_centers, dtype=np.float64),
        cl_atm_bins_scan_mk2=[np.asarray(cl_i, dtype=np.float64) for cl_i in cl_atm_bins_scan_mk2],
        cl_cmb_bins_mk2=np.asarray(cl_cmb_bins_mk2, dtype=np.float64),
    )


def _run_field(
    *,
    field_id: str,
    scan_paths: list[pathlib.Path],
    out_dir: pathlib.Path,
    mode: str,
    cfg: Config,
) -> None:
    prep = _prepare_group_inputs(label=field_id, scan_paths=scan_paths, mode=mode, cfg=cfg)
    scans = prep["scans"]
    scans_noise = prep["scans_noise"]
    winds = prep["winds"]
    wind_diags = prep["wind_diags"]
    masks_good = prep["masks_good"]
    pixel_size_deg0 = float(prep["pixel_size_deg"])
    pixel_res_rad = float(prep["pixel_res_rad"])
    ell_centers = np.asarray(prep["ell_centers"], dtype=np.float64)
    cl_atm_bins_scan_mk2 = [np.asarray(cl_i, dtype=np.float64) for cl_i in prep["cl_atm_bins_scan_mk2"]]
    cl_cmb_bins_mk2 = np.asarray(prep["cl_cmb_bins_mk2"], dtype=np.float64)

    # --- shared CMB solve grid and observed-set basis for per-scan solves ---
    boxes_cmb = []
    for s in scans:
        valid = np.isfinite(s["eff_tod_mk"])
        boxes_cmb.append(map_util.scan_bbox_from_pix_index(pix_index=s["pix_index"], valid_mask=valid))
    bbox_cmb = map_util.bbox_union(boxes_cmb)
    nx = int(bbox_cmb.nx)
    ny = int(bbox_cmb.ny)
    n_pix = int(nx * ny)

    pointing_mats = []
    valid_masks = []
    for s in scans:
        pm, vm = util.pointing_from_pix_index(
            pix_index=np.asarray(s["pix_index"], dtype=np.int64),
            tod_mk=np.asarray(s["eff_tod_mk"], dtype=np.float64),
            bbox=bbox_cmb,
        )
        pointing_mats.append(pm)
        valid_masks.append(vm)

    if mode == "MAP":
        obs_pix_global = np.arange(n_pix, dtype=np.int64)
        global_to_obs = np.arange(n_pix, dtype=np.int64)
    else:
        obs_pix_global, global_to_obs = util.observed_pixel_index_set(
            pointing_matrices=pointing_mats,
            valid_masks=valid_masks,
            n_pix=n_pix,
            min_hits_per_pix=int(cfg.min_hits_per_pix),
        )
        if int(obs_pix_global.size) == 0:
            raise RuntimeError(
                f"[{field_id}] No observed pixels survive min_hits_per_pix={int(cfg.min_hits_per_pix)}. "
                "Lower min_hits_per_pix or check input scans."
            )

    print(
        f"[grid] field={field_id} bbox_cmb ix=[{bbox_cmb.ix0},{bbox_cmb.ix1}] iy=[{bbox_cmb.iy0},{bbox_cmb.iy1}] "
        f"nx={nx} ny={ny} n_pix={n_pix} n_obs={int(obs_pix_global.size)}",
        flush=True,
    )

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
            cl_bins_mk2=np.asarray(cl_atm_bins_scan_mk2[i], dtype=np.float64),
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
            source_scan_path=np.array(str(scan_paths[i])),
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
            sigma_eff_det_mk=np.asarray(scans_noise[i], dtype=np.float64),
            obs_pix_global=np.asarray(obs_pix_global, dtype=np.int64),
            c_hat_full_mk=np.asarray(sol.c_hat_full_mk, dtype=np.float64),
            ell_centers=np.asarray(ell_centers, dtype=np.float64),
            cl_atm_bins_mk2=np.asarray(cl_atm_bins_scan_mk2[i], dtype=np.float64),
            cl_cmb_bins_mk2=np.asarray(cl_cmb_bins_mk2, dtype=np.float64),
            boundary=np.array("open"),
        )
        np.savez_compressed(out_path, **out_payload)
        print(f"[write] {out_path}", flush=True)


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
    max_scans = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    for estimator_mode in ("ML", "MAP"):
        main(Config(dataset_dir=str(dataset), estimator_mode=str(estimator_mode), max_scans=max_scans))

