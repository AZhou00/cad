#!/usr/bin/env python3
"""
Run multi-scan synthesis (combined reconstruction) using `cad`.

Inputs:
  - A dataset directory under `cad/data/<dataset_dir>/`.
  - Per-scan reconstruction files from `run_reconstruction.py`:
      `cad/data/<dataset_dir>_recon/<obs_id>/recon_scanXXX_<ml|map>.npz`

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


BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent
DATA_DIR = CAD_DIR / "data"


@dataclass(frozen=True)
class Config:
    dataset_dir: str | None = None
    estimator_mode: str = ""  # "ML" or "MAP"
    synthesis_scope: str = "both"  # "both" or "all_only"
    cl_floor_mk2: float = 1e-12
    min_hits_per_pix: int = 1
    cg_tol: float = 5e-4
    cg_maxiter: int = 2000
    prefer_binned_subdir: str = "binned_tod_10arcmin"


def _discover_fields(dataset_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Return [(field_id, field_input_dir)]."""
    subdirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    obs = sorted([p for p in subdirs if re.fullmatch(r"\d+", p.name)], key=lambda p: p.name)
    if obs:
        return [(p.name, p) for p in obs]
    return [(dataset_dir.name, dataset_dir)]


def _discover_scan_paths(*, field_dir: pathlib.Path, cfg: Config) -> list[pathlib.Path]:
    """Return sorted scan NPZ paths for one field directory."""
    binned_dirs = sorted([p for p in field_dir.iterdir() if p.is_dir() and p.name.startswith("binned_tod_")])
    if not binned_dirs:
        return []
    chosen = next((p for p in binned_dirs if p.name == str(cfg.prefer_binned_subdir)), binned_dirs[0])
    return sorted([p for p in chosen.iterdir() if p.is_file() and p.suffix == ".npz" and not p.name.startswith(".")])


def _discover_recon_paths(*, recon_dir: pathlib.Path, n_scans: int, mode: str) -> list[pathlib.Path]:
    recon_mode = str(mode).lower()
    return [recon_dir / f"recon_scan{i:03d}_{recon_mode}.npz" for i in range(int(n_scans))]


def _load_scan(npz_path: pathlib.Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        return dict(
            eff_tod_mk=np.asarray(z["eff_tod_mk"], dtype=np.float32),  # (n_t,n_eff)
            pix_index=np.asarray(z["pix_index"], dtype=np.int64),  # (n_t,n_eff,2)
            t_s=np.asarray(z["t_bin_center_s"], dtype=np.float64),  # (n_t,)
            pixel_size_deg=float(z["pixel_size_deg"]),
            eff_counts=np.asarray(z["eff_counts"], dtype=np.float64),  # (n_eff,)
            bin_sec=float(z["bin_sec"]),
            sample_rate_hz=float(z["sample_rate_hz"]),
        )


def _load_recon_scan_meta(*, recon_path: pathlib.Path, mode: str, scan_path: pathlib.Path, label: str) -> dict:
    if not recon_path.exists():
        raise RuntimeError(f"[{label}] Missing reconstruction file: {recon_path}")

    with np.load(recon_path, allow_pickle=False) as r:
        mode_saved = str(np.asarray(r["estimator_mode"]).item()).upper()
        mode_req = str(mode).upper()
        if mode_saved != mode_req:
            raise RuntimeError(f"[{label}] Mode mismatch in {recon_path}: file has {mode_saved}, requested {mode_req}.")

        if "source_scan_path" not in r.files:
            raise RuntimeError(f"[{label}] Missing source_scan_path in {recon_path}. Re-run run_reconstruction.py.")
        source = str(np.asarray(r["source_scan_path"]).item())
        if source != str(scan_path):
            raise RuntimeError(
                f"[{label}] source_scan_path mismatch for {recon_path}. "
                f"metadata={source} discovered={scan_path}"
            )

        wind = np.asarray(r["wind_deg_per_s"], dtype=np.float64).reshape(-1)
        if wind.size < 2:
            raise RuntimeError(f"[{label}] Invalid wind_deg_per_s in {recon_path}.")

        if "wind_valid_mask" not in r.files:
            raise RuntimeError(f"[{label}] Missing wind_valid_mask in {recon_path}.")
        mask = np.asarray(r["wind_valid_mask"], dtype=bool).reshape(-1)

        if "sigma_eff_det_mk" not in r.files:
            raise RuntimeError(f"[{label}] Missing sigma_eff_det_mk in {recon_path}. Re-run run_reconstruction.py.")
        sigma = np.asarray(r["sigma_eff_det_mk"], dtype=np.float64).reshape(-1)

        if "ell_centers" not in r.files or "cl_atm_bins_mk2" not in r.files or "cl_cmb_bins_mk2" not in r.files:
            raise RuntimeError(
                f"[{label}] Missing C_ell metadata in {recon_path}. "
                "Re-run run_reconstruction.py to regenerate per-scan priors."
            )
        ell_centers = np.asarray(r["ell_centers"], dtype=np.float64)
        cl_atm = np.asarray(r["cl_atm_bins_mk2"], dtype=np.float64)
        cl_cmb = np.asarray(r["cl_cmb_bins_mk2"], dtype=np.float64)

    return dict(
        wind=np.asarray(wind[:2], dtype=np.float64),
        mask=mask,
        sigma=sigma,
        ell_centers=ell_centers,
        cl_atm_bins_mk2=cl_atm,
        cl_cmb_bins_mk2=cl_cmb,
    )


def _prepare_synthesis_inputs(
    *,
    label: str,
    scan_paths: list[pathlib.Path],
    recon_paths: list[pathlib.Path],
    mode: str,
    cfg: Config,
) -> dict:
    if len(scan_paths) != len(recon_paths):
        raise RuntimeError(f"[{label}] scan_paths and recon_paths length mismatch.")
    if len(scan_paths) == 0:
        raise RuntimeError(f"[{label}] No scans to synthesize.")

    scans0 = [_load_scan(p) for p in scan_paths]
    pixel_size_deg0 = float(scans0[0]["pixel_size_deg"])
    for s in scans0[1:]:
        if abs(float(s["pixel_size_deg"]) - pixel_size_deg0) > 1e-12:
            raise RuntimeError(f"[{label}] pixel_size_deg mismatch across scans.")

    scans = []
    winds = np.zeros((len(scan_paths), 2), dtype=np.float64)
    scans_noise = []
    ell_ref = None
    cl_atm_scans_ref: list[np.ndarray] = []
    cl_cmb_ref_list: list[np.ndarray] = []

    for i, (scan_path, recon_path, s0) in enumerate(zip(scan_paths, recon_paths, scans0, strict=True)):
        meta = _load_recon_scan_meta(recon_path=recon_path, mode=mode, scan_path=scan_path, label=label)
        mask = np.asarray(meta["mask"], dtype=bool)
        if mask.shape != (int(s0["eff_tod_mk"].shape[1]),):
            raise RuntimeError(
                f"[{label}] Detector mask length mismatch for scan{i:03d}: "
                f"mask={mask.shape} n_det={int(s0['eff_tod_mk'].shape[1])}"
            )
        sigma = np.asarray(meta["sigma"], dtype=np.float64).reshape(-1)
        if sigma.shape != (int(mask.sum()),):
            raise RuntimeError(
                f"[{label}] sigma_eff_det_mk length mismatch for scan{i:03d}: "
                f"sigma={sigma.shape} n_good={int(mask.sum())}"
            )
        if not bool(np.all(np.isfinite(sigma))) or not bool(np.all(sigma > 0)):
            raise RuntimeError(f"[{label}] invalid sigma_eff_det_mk values for scan{i:03d}.")

        s = dict(s0)
        s["eff_tod_mk"] = np.asarray(s0["eff_tod_mk"])[:, mask]
        s["pix_index"] = np.asarray(s0["pix_index"], dtype=np.int64)[:, mask, :]
        s["eff_counts"] = np.asarray(s0["eff_counts"], dtype=np.float64)[mask]
        scans.append(s)

        winds[i, :] = np.asarray(meta["wind"], dtype=np.float64)
        scans_noise.append(sigma)

        ell_i = np.asarray(meta["ell_centers"], dtype=np.float64).reshape(-1)
        cl_atm_i = np.asarray(meta["cl_atm_bins_mk2"], dtype=np.float64).reshape(-1)
        cl_cmb_i = np.asarray(meta["cl_cmb_bins_mk2"], dtype=np.float64).reshape(-1)

        cl_atm_i = np.where(np.isfinite(cl_atm_i), cl_atm_i, 0.0)
        cl_atm_i = np.maximum(cl_atm_i, float(cfg.cl_floor_mk2))
        cl_cmb_i = np.where(np.isfinite(cl_cmb_i), cl_cmb_i, 0.0)
        cl_cmb_i = np.maximum(cl_cmb_i, float(cfg.cl_floor_mk2))

        if ell_ref is None:
            ell_ref = np.asarray(ell_i, dtype=np.float64)
            cl_atm_scans_ref.append(np.asarray(cl_atm_i, dtype=np.float64))
            cl_cmb_ref_list.append(np.asarray(cl_cmb_i, dtype=np.float64))
        else:
            if int(ell_i.size) <= 0 or int(ell_ref.size) <= 0:
                raise RuntimeError(f"[{label}] invalid ell arrays while preparing synthesis.")
            if np.array_equal(np.asarray(ell_i, dtype=np.float64), np.asarray(ell_ref, dtype=np.float64)):
                cl_atm_scans_ref.append(np.asarray(cl_atm_i, dtype=np.float64))
                cl_cmb_ref_list.append(np.asarray(cl_cmb_i, dtype=np.float64))
            else:
                cl_atm_scans_ref.append(
                    np.interp(np.asarray(ell_ref, dtype=np.float64), np.asarray(ell_i, dtype=np.float64), np.asarray(cl_atm_i, dtype=np.float64), left=float(cl_atm_i[0]), right=float(cl_atm_i[-1]))
                )
                cl_cmb_ref_list.append(
                    np.interp(np.asarray(ell_ref, dtype=np.float64), np.asarray(ell_i, dtype=np.float64), np.asarray(cl_cmb_i, dtype=np.float64), left=float(cl_cmb_i[0]), right=float(cl_cmb_i[-1]))
                )

        print(
            f"[meta] {label} scan{i:03d}: w=({winds[i,0]:.4f},{winds[i,1]:.4f}) "
            f"good={int(mask.sum())}/{int(mask.size)} sigma_med={float(np.median(sigma)):.3f} mK",
            flush=True,
        )

    cl_cmb_ref = np.median(np.stack(cl_cmb_ref_list, axis=0), axis=0)
    cl_atm_summary = np.median(np.stack(cl_atm_scans_ref, axis=0), axis=0)

    return dict(
        scans=scans,
        winds=winds,
        pixel_size_deg=float(pixel_size_deg0),
        ell_centers=np.asarray(ell_ref, dtype=np.float64),
        cl_atm_bins_mk2_scans=[np.asarray(cl, dtype=np.float64) for cl in cl_atm_scans_ref],
        cl_atm_bins_mk2_summary=np.asarray(cl_atm_summary, dtype=np.float64),
        cl_cmb_bins_mk2=np.asarray(cl_cmb_ref, dtype=np.float64),
        scans_noise=[np.asarray(sig, dtype=np.float64) for sig in scans_noise],
    )


def _run_synthesis_group(*, label: str, prep: dict, out_path: pathlib.Path, mode: str, cfg: Config) -> None:
    combined = cad.synthesize_scans(
        scans_tod_mk=[np.asarray(s["eff_tod_mk"], dtype=np.float64) for s in prep["scans"]],
        scans_pix_index=[np.asarray(s["pix_index"], dtype=np.int64) for s in prep["scans"]],
        scans_t_s=[np.asarray(s["t_s"], dtype=np.float64) for s in prep["scans"]],
        winds_deg_per_s=[(float(w[0]), float(w[1])) for w in prep["winds"]],
        scans_noise_std_det_mk=[np.asarray(sig, dtype=np.float64) for sig in prep["scans_noise"]],
        pixel_size_deg=float(prep["pixel_size_deg"]),
        cl_atm_bins_mk2=[np.asarray(cl, dtype=np.float64) for cl in prep["cl_atm_bins_mk2_scans"]],
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
        cl_atm_bins_mk2=np.asarray(prep["cl_atm_bins_mk2_summary"], dtype=np.float64),
        cl_atm_bins_scan_mk2=np.asarray(prep["cl_atm_bins_mk2_scans"], dtype=np.float64),
        cl_cmb_bins_mk2=np.asarray(prep["cl_cmb_bins_mk2"], dtype=np.float64),
        boundary=np.array("open"),
        n_scans=np.int64(len(prep["scans"])),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_payload)
    print(
        f"[write] {out_path}  [group={label} mode={mode} n_scans={len(prep['scans'])} n_obs={int(obs_pix_global.size)} n_pix={n_pix}]",
        flush=True,
    )


def _merge_preps_for_all_observations(*, preps: list[dict], mode: str) -> dict:
    if not preps:
        raise RuntimeError("No prepared groups to merge.")

    pixel_size_deg = float(preps[0]["pixel_size_deg"])
    for p in preps[1:]:
        if abs(float(p["pixel_size_deg"]) - pixel_size_deg) > 1e-12:
            raise RuntimeError("pixel_size_deg mismatch across observation groups.")

    ell_ref = np.asarray(preps[0]["ell_centers"], dtype=np.float64).reshape(-1)
    if ell_ref.size == 0:
        raise RuntimeError("Empty ell_centers in prepared group.")

    cl_atm_scans_all: list[np.ndarray] = []
    cl_cmb_all = []
    scans = []
    scans_noise = []
    winds = []
    for p in preps:
        ell_i = np.asarray(p["ell_centers"], dtype=np.float64).reshape(-1)
        cl_atm_scans_i = [np.asarray(cl, dtype=np.float64).reshape(-1) for cl in p["cl_atm_bins_mk2_scans"]]
        cl_cmb_i = np.asarray(p["cl_cmb_bins_mk2"], dtype=np.float64).reshape(-1)
        if ell_i.size != ell_ref.size:
            raise RuntimeError("n_ell_bins mismatch across observation groups.")

        if np.array_equal(ell_i, ell_ref):
            for cl_scan in cl_atm_scans_i:
                cl_atm_scans_all.append(np.asarray(cl_scan, dtype=np.float64))
            cl_cmb_interp = cl_cmb_i
        else:
            for cl_scan in cl_atm_scans_i:
                cl_atm_scans_all.append(
                    np.interp(ell_ref, ell_i, cl_scan, left=float(cl_scan[0]), right=float(cl_scan[-1]))
                )
            cl_cmb_interp = np.interp(ell_ref, ell_i, cl_cmb_i, left=cl_cmb_i[0], right=cl_cmb_i[-1])

        cl_cmb_all.append(cl_cmb_interp)
        scans.extend(p["scans"])
        scans_noise.extend(p["scans_noise"])
        winds.extend([np.asarray(w, dtype=np.float64) for w in np.asarray(p["winds"], dtype=np.float64)])

    cl_atm_agg = np.median(np.stack(cl_atm_scans_all, axis=0), axis=0)
    cl_cmb_agg = np.median(np.stack(cl_cmb_all, axis=0), axis=0)
    if str(mode).upper() == "ML":
        cl_cmb_agg = np.maximum(cl_cmb_agg, 0.0)

    return dict(
        scans=scans,
        winds=np.asarray(winds, dtype=np.float64),
        pixel_size_deg=float(pixel_size_deg),
        ell_centers=ell_ref,
        cl_atm_bins_mk2_scans=[np.asarray(cl, dtype=np.float64) for cl in cl_atm_scans_all],
        cl_atm_bins_mk2_summary=np.asarray(cl_atm_agg, dtype=np.float64),
        cl_cmb_bins_mk2=np.asarray(cl_cmb_agg, dtype=np.float64),
        scans_noise=[np.asarray(sig, dtype=np.float64) for sig in scans_noise],
    )


def main(cfg: Config) -> None:
    mode = str(cfg.estimator_mode).upper()
    if mode not in ("ML", "MAP"):
        raise ValueError("estimator_mode must be 'ML' or 'MAP'.")
    scope = str(cfg.synthesis_scope).lower()
    if scope not in ("both", "all_only"):
        raise ValueError("synthesis_scope must be 'both' or 'all_only'.")

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

    per_field_preps: list[dict] = []
    for field_id, field_in_dir in fields:
        scan_paths = _discover_scan_paths(field_dir=field_in_dir, cfg=cfg)
        if len(scan_paths) == 0:
            print(f"[skip] field {field_id}: no scan NPZs found under {field_in_dir}", flush=True)
            continue

        recon_dir = out_root / str(field_id)
        recon_paths = _discover_recon_paths(recon_dir=recon_dir, n_scans=len(scan_paths), mode=mode)
        prep = _prepare_synthesis_inputs(label=str(field_id), scan_paths=scan_paths, recon_paths=recon_paths, mode=mode, cfg=cfg)
        if scope == "both":
            out_path = out_root / str(field_id) / f"recon_combined_{mode.lower()}.npz"
            _run_synthesis_group(label=str(field_id), prep=prep, out_path=out_path, mode=mode, cfg=cfg)
        per_field_preps.append(prep)

    if len(per_field_preps) == 0:
        raise RuntimeError(f"No scan NPZ files discovered under dataset: {dataset_dir}")

    prep_all = _merge_preps_for_all_observations(preps=per_field_preps, mode=mode)
    out_all = out_root / f"recon_{mode.lower()}.npz"
    _run_synthesis_group(label="all_observations", prep=prep_all, out_path=out_all, mode=mode, cfg=cfg)


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) >= 2 else "ra0hdec-59.75"
    scope = sys.argv[2] if len(sys.argv) >= 3 else "both"
    for estimator_mode in ("ML", "MAP"):
        main(Config(dataset_dir=str(dataset), estimator_mode=str(estimator_mode), synthesis_scope=str(scope)))
