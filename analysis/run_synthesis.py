#!/usr/bin/env python3
"""
Run multi-scan synthesis (combined reconstruction) using `cad`.

Inputs:
  - A dataset directory under `cad/data/<dataset_dir>/`.

Supported layout (organized by observation id):
  `cad/data/<dataset_dir>/<obs_id>/binned_tod_*/<scan>.npz`

Outputs:
  - Per observation:
      `cad/data/<dataset_dir>_recon/<field_id>/recon_combined_<ml|map>.npz`
  - All observations combined:
      `cad/data/<dataset_dir>_recon/recon_<ml|map>.npz`
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
        )


def _prepare_synthesis_inputs(*, label: str, scan_paths: list[pathlib.Path], mode: str, cfg: Config) -> dict:
    scans0 = [_load_scan(p) for p in scan_paths]
    pixel_size_deg0 = float(scans0[0]["pixel_size_deg"])
    for s in scans0[1:]:
        if abs(float(s["pixel_size_deg"]) - pixel_size_deg0) > 1e-12:
            raise ValueError(f"[{label}] pixel_size_deg mismatch across scans.")

    print(f"[group] {label}: n_scans={len(scan_paths)}  mode={mode}", flush=True)

    # --- wind estimation on unmasked scans ---
    winds = np.zeros((len(scans0), 2), dtype=np.float64)
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
        masks_good.append(np.asarray(mask, dtype=bool))
        print(
            f"[wind] {label} scan{i:03d} w=({w_x:.4f},{w_y:.4f}) deg/s  "
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

    # --- apply bad pixel masking ---
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
            f"[noise] {label} scan{i:03d}: sigma_eff median={float(np.median(sig)):.3f} mK  "
            f"min={float(np.min(sig)):.3f}  max={float(np.max(sig)):.3f}",
            flush=True,
        )

    return dict(
        scans=scans,
        winds=winds,
        pixel_size_deg=float(pixel_size_deg0),
        ell_centers=np.asarray(ell_centers, dtype=np.float64),
        cl_atm_bins_mk2=np.asarray(cl_atm_bins_mk2, dtype=np.float64),
        cl_cmb_bins_mk2=np.asarray(cl_cmb_bins_mk2, dtype=np.float64),
        scans_noise=[np.asarray(sig, dtype=np.float64) for sig in scans_noise],
    )


def _run_synthesis_group(
    *,
    label: str,
    scan_paths: list[pathlib.Path],
    out_path: pathlib.Path,
    mode: str,
    cfg: Config,
) -> None:
    prep = _prepare_synthesis_inputs(label=label, scan_paths=scan_paths, mode=mode, cfg=cfg)

    combined = cad.synthesize_scans(
        scans_tod_mk=[np.asarray(s["eff_tod_mk"], dtype=np.float64) for s in prep["scans"]],
        scans_pix_index=[np.asarray(s["pix_index"], dtype=np.int64) for s in prep["scans"]],
        scans_t_s=[np.asarray(s["t_s"], dtype=np.float64) for s in prep["scans"]],
        winds_deg_per_s=[(float(w[0]), float(w[1])) for w in prep["winds"]],
        scans_noise_std_det_mk=[np.asarray(sig, dtype=np.float64) for sig in prep["scans_noise"]],
        pixel_size_deg=float(prep["pixel_size_deg"]),
        cl_atm_bins_mk2=np.asarray(prep["cl_atm_bins_mk2"], dtype=np.float64),
        cl_cmb_bins_mk2=np.asarray(prep["cl_cmb_bins_mk2"], dtype=np.float64),
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

    out_payload = dict(
        estimator_mode=np.array(mode),
        pixel_size_deg=np.float64(prep["pixel_size_deg"]),
        bbox_ix0=np.int64(bbox_cmb.ix0),
        bbox_iy0=np.int64(bbox_cmb.iy0),
        nx=np.int64(nx),
        ny=np.int64(ny),
        winds_deg_per_s=np.asarray(prep["winds"], dtype=np.float64),
        obs_pix_global=np.asarray(obs_pix_global, dtype=np.int64),
        c_hat_full_mk=np.asarray(combined.c_hat_full_mk, dtype=np.float64),
        ell_centers=np.asarray(prep["ell_centers"], dtype=np.float64),
        cl_atm_bins_mk2=np.asarray(prep["cl_atm_bins_mk2"], dtype=np.float64),
        cl_cmb_bins_mk2=np.asarray(prep["cl_cmb_bins_mk2"], dtype=np.float64),
        boundary=np.array("open"),
        n_scans=np.int64(len(scan_paths)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_payload)
    print(
        f"[write] {out_path}  [group={label} mode={mode} n_scans={len(scan_paths)} n_obs={int(obs_pix_global.size)} n_pix={n_pix}]",
        flush=True,
    )


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

    all_scan_paths: list[pathlib.Path] = []
    for field_id, field_in_dir in fields:
        scan_paths = _discover_scan_paths(field_dir=field_in_dir, cfg=cfg)
        if len(scan_paths) == 0:
            print(f"[skip] field {field_id}: no scan NPZs found under {field_in_dir}", flush=True)
            continue
        out_path = out_root / str(field_id) / f"recon_combined_{mode.lower()}.npz"
        _run_synthesis_group(label=str(field_id), scan_paths=scan_paths, out_path=out_path, mode=mode, cfg=cfg)
        all_scan_paths.extend(scan_paths)

    if len(all_scan_paths) == 0:
        raise RuntimeError(f"No scan NPZ files discovered under dataset: {dataset_dir}")

    # Synthesis over all scans from all observations in this dataset.
    out_all = out_root / f"recon_{mode.lower()}.npz"
    _run_synthesis_group(label="all_observations", scan_paths=all_scan_paths, out_path=out_all, mode=mode, cfg=cfg)


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) >= 2 else "ra0hdec-59.75"
    for estimator_mode in ("ML", "MAP"):
        main(Config(dataset_dir=str(dataset), estimator_mode=str(estimator_mode)))

