"""
Parallel solve: exact global synthesis from per-scan cov_inv and Pt_Ninv_d.

Sum [Cov(hat c_s)]^{-1} and P^T tilde N^{-1} d at global observed indices; solve for hat c.
We must form cov_inv_tot to save it; the solve (cov_inv_tot @ c_hat = RHS) is O(n_obs^3).
Solve runs on GPU via JAX (fail if solve errors). CG is an option for the solve
(matrix-free CG would avoid storing cov_inv_tot during solve but we still form it once to save).
Most efficient with dense precision required: form once, solve on GPU (Cholesky/solve).
The dense precision matrix cov_inv_tot (n_obs x n_obs) is always saved in the output NPZ.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .layout import GlobalLayout, load_layout
from .reconstruct_scan import load_scan_artifact


def _solve_synthesis_gpu(cov_inv_good: np.ndarray, Pt_Ninv_d_good: np.ndarray) -> np.ndarray:
    """Solve cov_inv_good @ x = Pt_Ninv_d_good on GPU with JAX. Returns c_hat_good (n_good,)."""
    import jax.numpy as jnp
    A = jnp.asarray(cov_inv_good)
    b = jnp.asarray(Pt_Ninv_d_good)
    x = jnp.linalg.solve(A, b)
    return np.asarray(x, dtype=np.float64)


def run_synthesis(
    layout: GlobalLayout,
    scan_npz_dir: Path,
    out_path: Path,
    timings: dict | None = None,
) -> None:
    """
    Exact marginalized synthesis: accumulate cov_inv_tot and Pt_Ninv_d_tot from per-scan npzs,
    solve cov_inv_tot @ c_hat_obs = Pt_Ninv_d_tot, embed to full CMB grid.
    Always saves the dense precision matrix cov_inv_tot (n_obs x n_obs). If timings, keys: load_s, accumulate_s, solve_s.
    """
    n_obs = layout.n_obs
    global_to_obs = layout.global_to_obs
    cov_inv_tot = np.zeros((n_obs, n_obs), dtype=np.float64)
    Pt_Ninv_d_tot = np.zeros((n_obs,), dtype=np.float64)

    existing = [i for i in range(layout.n_scans) if (scan_npz_dir / f"scan_{i:04d}_ml.npz").exists()]
    if not existing:
        raise FileNotFoundError(f"No scan_*_ml.npz found in {scan_npz_dir}")

    t0 = time.perf_counter()
    scan_metadata: list[dict] = []
    for scan_index in tqdm(existing, desc="load+accumulate", leave=True):
        npz_path = scan_npz_dir / f"scan_{scan_index:04d}_ml.npz"
        art = load_scan_artifact(npz_path)
        obs_pix_global_scan = art["obs_pix_global_scan"]
        cov_inv_s = art["cov_inv"]
        Pt_Ninv_d_s = art["Pt_Ninv_d"]
        obs_idx = np.asarray(global_to_obs[obs_pix_global_scan], dtype=np.int64)
        valid = obs_idx >= 0
        obs_idx_valid = obs_idx[valid]
        if obs_idx_valid.size > 0:
            cov_inv_s_valid = cov_inv_s[np.ix_(valid, valid)]
            Pt_Ninv_d_s_valid = Pt_Ninv_d_s[valid]
            cov_inv_tot[np.ix_(obs_idx_valid, obs_idx_valid)] += cov_inv_s_valid
            Pt_Ninv_d_tot[obs_idx_valid] += Pt_Ninv_d_s_valid
        scan_metadata.append({
            "scan_index": scan_index,
            "wind_deg_per_s": np.asarray(art["wind_deg_per_s"], dtype=np.float64).copy(),
            "wind_sigma_x_deg_per_s": art["wind_sigma_x_deg_per_s"],
            "wind_sigma_y_deg_per_s": art["wind_sigma_y_deg_per_s"],
            "ell_atm": art["ell_atm"].copy(),
            "cl_atm_mk2": art["cl_atm_mk2"].copy(),
        })
    if timings is not None:
        timings["load_s"] = 0.0
        timings["accumulate_s"] = time.perf_counter() - t0

    zero_precision_mask = np.all(cov_inv_tot == 0.0, axis=1)
    good = ~zero_precision_mask
    c_hat_obs = np.zeros((n_obs,), dtype=np.float64)
    t0 = time.perf_counter()
    if np.any(good):
        cov_inv_good = cov_inv_tot[np.ix_(good, good)]
        Pt_Ninv_d_good = Pt_Ninv_d_tot[good]
        c_hat_good = _solve_synthesis_gpu(cov_inv_good, Pt_Ninv_d_good)
        c_hat_good = np.asarray(c_hat_good, dtype=np.float64)
        c_hat_obs[good] = c_hat_good
        c_hat_obs -= float(np.mean(c_hat_obs[good]))
    if timings is not None:
        timings["solve_s"] = time.perf_counter() - t0

    precision_diag_total = np.diag(cov_inv_tot).copy()
    var_diag_total = np.full((n_obs,), np.inf, dtype=np.float64)
    var_diag_total[good] = np.where(
        precision_diag_total[good] > 0.0,
        1.0 / precision_diag_total[good],
        np.inf,
    )

    n_pix = layout.n_pix
    c_hat_full_mk = np.zeros((n_pix,), dtype=np.float64)
    c_hat_full_mk[layout.obs_pix_global] = c_hat_obs  # (n_pix_cmb,)

    n_good = int(np.sum(good))
    if n_good > 0:
        evals, evecs = np.linalg.eigh(cov_inv_tot[np.ix_(good, good)])
        k = min(10, max(1, int(0.1 * n_good)))
        uncertain_vectors = np.asarray(evecs[:, :k], dtype=np.float64)
        uncertain_variances = np.asarray(1.0 / evals[:k], dtype=np.float64)
    else:
        uncertain_vectors = np.empty((n_obs, 0), dtype=np.float64)
        uncertain_variances = np.empty((0,), dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _out = dict(
        estimator_mode=np.array("ML", dtype=object),
        bbox_ix0=np.int64(layout.bbox_ix0),
        bbox_iy0=np.int64(layout.bbox_iy0),
        nx=np.int64(layout.nx),
        ny=np.int64(layout.ny),
        pixel_size_deg=np.float64(layout.pixel_size_deg),
        obs_pix_global=layout.obs_pix_global,
        c_hat_full_mk=c_hat_full_mk,
        c_hat_obs=c_hat_obs,
        precision_diag_total=precision_diag_total,
        var_diag_total=var_diag_total,
        zero_precision_mask=zero_precision_mask,
        n_scans=np.int64(layout.n_scans),
        n_scans_used=np.int64(len(existing)),
        cov_inv_tot=cov_inv_tot,
        good_mask=good,
        uncertain_mode_vectors=uncertain_vectors,
        uncertain_mode_variances=uncertain_variances,
        scan_metadata=np.array(scan_metadata, dtype=object),
    )
    np.savez_compressed(out_path, **_out)
    print(f"[write] {out_path} n_obs={n_obs} n_scans_used={len(existing)}/{layout.n_scans}", flush=True)


def run_synthesis_multi_obs(
    out_base: Path,
    field_id: str,
    observation_ids: list[str],
    out_subdir: str = "synthesized",
    timings: dict | None = None,
) -> tuple[GlobalLayout, Path, Path]:
    """
    Load all scan npzs from OUT_BASE/field_id/obs_id/scans/ for each obs_id, build combined layout
    (union of observed pixels), accumulate cov_inv and Pt_Ninv_d, solve, write to
    OUT_BASE/field_id/<out_subdir>/recon_combined_ml.npz and winds_list.npz.
    Returns (combined_layout, out_npz_path, winds_npz_path).
    """
    if not observation_ids:
        raise ValueError("observation_ids must be non-empty")
    layouts = []
    artifact_paths: list[tuple[str, int, Path]] = []
    for obs_id in observation_ids:
        layout_path = out_base / field_id / obs_id / "layout.npz"
        scan_dir = out_base / field_id / obs_id / "scans"
        if not layout_path.exists():
            raise FileNotFoundError(f"Layout not found: {layout_path}")
        lay = load_layout(layout_path)
        layouts.append((obs_id, lay))
        for scan_index in range(lay.n_scans):
            p = scan_dir / f"scan_{scan_index:04d}_ml.npz"
            if p.exists():
                artifact_paths.append((obs_id, scan_index, p))

    if not artifact_paths:
        raise FileNotFoundError(f"No scan_*_ml.npz found under {out_base / field_id} for {observation_ids}")

    first = layouts[0][1]
    obs_pix_union = sorted(set().union(*(set(lay.obs_pix_global.tolist()) for _, lay in layouts)))
    n_obs = len(obs_pix_union)
    n_pix = first.n_pix
    global_to_obs = np.full(n_pix, -1, dtype=np.int64)
    for i, p in enumerate(obs_pix_union):
        global_to_obs[p] = i
    combined_layout = GlobalLayout(
        bbox_ix0=first.bbox_ix0,
        bbox_iy0=first.bbox_iy0,
        nx=first.nx,
        ny=first.ny,
        obs_pix_global=np.array(obs_pix_union, dtype=np.int64),
        global_to_obs=global_to_obs,
        scan_paths=(),
        pixel_size_deg=first.pixel_size_deg,
        field_id=field_id,
    )

    cov_inv_tot = np.zeros((n_obs, n_obs), dtype=np.float64)
    Pt_Ninv_d_tot = np.zeros((n_obs,), dtype=np.float64)
    winds_list: list[tuple[str, int, float, float, float, float]] = []
    scan_metadata: list[dict] = []

    t0 = time.perf_counter()
    for obs_id, scan_index, npz_path in tqdm(artifact_paths, desc="load scans", leave=True):
        art = load_scan_artifact(npz_path)
        obs_pix_global_scan = art["obs_pix_global_scan"]
        cov_inv_s = art["cov_inv"]
        Pt_Ninv_d_s = art["Pt_Ninv_d"]
        obs_idx = np.asarray(combined_layout.global_to_obs[obs_pix_global_scan], dtype=np.int64)
        valid = obs_idx >= 0
        obs_idx_valid = obs_idx[valid]
        if obs_idx_valid.size > 0:
            cov_inv_s_valid = cov_inv_s[np.ix_(valid, valid)]
            Pt_Ninv_d_s_valid = Pt_Ninv_d_s[valid]
            cov_inv_tot[np.ix_(obs_idx_valid, obs_idx_valid)] += cov_inv_s_valid
            Pt_Ninv_d_tot[obs_idx_valid] += Pt_Ninv_d_s_valid
        w = art["wind_deg_per_s"]
        sx = float(art["wind_sigma_x_deg_per_s"])
        sy = float(art["wind_sigma_y_deg_per_s"])
        winds_list.append((obs_id, scan_index, float(w[0]), float(w[1]), sx, sy))
        scan_metadata.append({
            "observation_id": obs_id,
            "scan_index": scan_index,
            "wind_deg_per_s": np.asarray(art["wind_deg_per_s"], dtype=np.float64).copy(),
            "wind_sigma_x_deg_per_s": sx,
            "wind_sigma_y_deg_per_s": sy,
            "ell_atm": art["ell_atm"].copy(),
            "cl_atm_mk2": art["cl_atm_mk2"].copy(),
        })
    if timings is not None:
        timings["load_s"] = time.perf_counter() - t0

    zero_precision_mask = np.all(cov_inv_tot == 0.0, axis=1)
    good = ~zero_precision_mask
    c_hat_obs = np.zeros((n_obs,), dtype=np.float64)
    if np.any(good):
        cov_inv_good = cov_inv_tot[np.ix_(good, good)]
        Pt_Ninv_d_good = Pt_Ninv_d_tot[good]
        c_hat_good = _solve_synthesis_gpu(cov_inv_good, Pt_Ninv_d_good)
        c_hat_obs[good] = np.asarray(c_hat_good, dtype=np.float64)
        c_hat_obs -= float(np.mean(c_hat_obs[good]))

    precision_diag_total = np.diag(cov_inv_tot).copy()
    var_diag_total = np.full((n_obs,), np.inf, dtype=np.float64)
    var_diag_total[good] = np.where(
        precision_diag_total[good] > 0.0,
        1.0 / precision_diag_total[good],
        np.inf,
    )
    c_hat_full_mk = np.zeros((n_pix,), dtype=np.float64)
    c_hat_full_mk[combined_layout.obs_pix_global] = c_hat_obs

    n_good = int(np.sum(good))
    if n_good > 0:
        evals, evecs = np.linalg.eigh(cov_inv_tot[np.ix_(good, good)])
        k = min(10, max(1, int(0.1 * n_good)))
        uncertain_vectors = np.asarray(evecs[:, :k], dtype=np.float64)
        uncertain_variances = np.asarray(1.0 / evals[:k], dtype=np.float64)
    else:
        uncertain_vectors = np.empty((n_obs, 0), dtype=np.float64)
        uncertain_variances = np.empty((0,), dtype=np.float64)

    out_dir = out_base / field_id / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "recon_combined_ml.npz"
    _out = dict(
        estimator_mode=np.array("ML", dtype=object),
        bbox_ix0=np.int64(combined_layout.bbox_ix0),
        bbox_iy0=np.int64(combined_layout.bbox_iy0),
        nx=np.int64(combined_layout.nx),
        ny=np.int64(combined_layout.ny),
        pixel_size_deg=np.float64(combined_layout.pixel_size_deg),
        obs_pix_global=combined_layout.obs_pix_global,
        c_hat_full_mk=c_hat_full_mk,
        c_hat_obs=c_hat_obs,
        precision_diag_total=precision_diag_total,
        var_diag_total=var_diag_total,
        zero_precision_mask=zero_precision_mask,
        n_scans=np.int64(len(artifact_paths)),
        n_scans_used=np.int64(len(artifact_paths)),
        cov_inv_tot=cov_inv_tot,
        good_mask=good,
        uncertain_mode_vectors=uncertain_vectors,
        uncertain_mode_variances=uncertain_variances,
        scan_metadata=np.array(scan_metadata, dtype=object),
    )
    np.savez_compressed(out_npz, **_out)

    obs_ids_arr = np.array([t[0] for t in winds_list], dtype=object)
    scan_indices = np.array([t[1] for t in winds_list], dtype=np.int64)
    wind_deg_per_s = np.array([[t[2], t[3]] for t in winds_list], dtype=np.float64)
    wind_sigma = np.array([[t[4], t[5]] for t in winds_list], dtype=np.float64)
    winds_npz = out_dir / "winds_list.npz"
    np.savez_compressed(
        winds_npz,
        observation_id=obs_ids_arr,
        scan_index=scan_indices,
        wind_deg_per_s=wind_deg_per_s,
        wind_sigma_deg_per_s=wind_sigma,
    )
    print(f"[write] {out_npz} n_obs={n_obs} n_scans={len(artifact_paths)}", flush=True)
    print(f"[write] {winds_npz}", flush=True)
    return combined_layout, out_npz, winds_npz
