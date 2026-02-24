"""
Direct solve: multi-field synthesis (run_synthesis, run_all_reconstruction). Max 8 scans per group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cad import dataset_io

from .synthesize_scan import synthesize_scans
from .workflow_single import (
    MAX_SCANS_DIRECT,
    DirectConfig,
    discover_recon_paths,
    load_recon_scan_meta,
)


def prepare_synthesis_inputs(
    *,
    label: str,
    scan_paths: list[Path],
    recon_paths: list[Path],
    mode: str,
    cfg: DirectConfig,
) -> dict:
    """Raises if len(scan_paths) > 8."""
    if len(scan_paths) > MAX_SCANS_DIRECT:
        raise RuntimeError(
            f"Direct solve allows at most {MAX_SCANS_DIRECT} scans per group; got {len(scan_paths)}."
        )
    if len(scan_paths) != len(recon_paths):
        raise RuntimeError(f"[{label}] scan_paths and recon_paths length mismatch.")
    if len(scan_paths) == 0:
        raise RuntimeError(f"[{label}] No scans to synthesize.")

    scans0 = [dataset_io.load_scan(p) for p in scan_paths]
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
        meta = load_recon_scan_meta(recon_path=recon_path, mode=mode, scan_path=scan_path, label=label)
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
                    np.interp(
                        np.asarray(ell_ref, dtype=np.float64),
                        np.asarray(ell_i, dtype=np.float64),
                        np.asarray(cl_atm_i, dtype=np.float64),
                        left=float(cl_atm_i[0]),
                        right=float(cl_atm_i[-1]),
                    )
                )
                cl_cmb_ref_list.append(
                    np.interp(
                        np.asarray(ell_ref, dtype=np.float64),
                        np.asarray(ell_i, dtype=np.float64),
                        np.asarray(cl_cmb_i, dtype=np.float64),
                        left=float(cl_cmb_i[0]),
                        right=float(cl_cmb_i[-1]),
                    )
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


def merge_preps_for_all_observations(*, preps: list[dict], mode: str) -> dict:
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


def run_synthesis_group(
    *,
    label: str,
    prep: dict,
    out_path: Path,
    mode: str,
    cfg: DirectConfig,
) -> None:
    """Raises if len(prep['scans']) > 8."""
    if len(prep["scans"]) > MAX_SCANS_DIRECT:
        raise RuntimeError(
            f"Direct solve allows at most {MAX_SCANS_DIRECT} scans per group; got {len(prep['scans'])}."
        )
    combined = synthesize_scans(
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
