"""
Parallel solve: exact global synthesis from per-scan cov_inv and Pt_Ninv_d.

Joint synthesis (theory): cov_inv_tot = sum_s [Cov(hat c_s)]^{-1}, Pt_Ninv_d_tot = sum_s P_s' tilde N_s^{-1} d_s.
Solve (cov_inv_tot @ c_hat = Pt_Ninv_d_tot) for ML map on observed pixels; gauge by mean subtraction.
Unconstrained modes: eigenvectors of the precision [Cov(hat c)]^{-1} with smallest eigenvalues
(largest posterior variance). Estimated via Lanczos on cov_inv_tot; stored as uncertain_mode_vectors (n_good, k)
and uncertain_mode_variances = 1/lambda (covariance eigenvalues). The dense cov_inv_tot is always saved.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .layout import GlobalLayout, load_layout
from .reconstruct_scan import load_scan_artifact

jax.config.update("jax_enable_x64", True)

N_UNCERTAIN_MODES = 100
LANCZOS_OVERSAMPLE = 64
LANCZOS_MAXITER = 256
LANCZOS_SEED = 0


def _solve_synthesis(cov_inv_good: np.ndarray, Pt_Ninv_d_good: np.ndarray) -> np.ndarray:
    """Single synthesis solve path: JAX dense solve."""
    A = jnp.asarray(cov_inv_good)
    b = jnp.asarray(Pt_Ninv_d_good)
    x = jnp.linalg.solve(A, b)
    return np.asarray(x, dtype=np.float64)


def _lanczos_smallest_modes(
    cov_inv_good: np.ndarray,
    *,
    n_modes: int = N_UNCERTAIN_MODES,
    oversample: int = LANCZOS_OVERSAMPLE,
    maxiter: int = LANCZOS_MAXITER,
    seed: int = LANCZOS_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate smallest eigenpairs of SPD precision matrix via Lanczos + Ritz projection.
    These are the pixel-space unconstrained directions (largest posterior variance).

    Returns:
      uncertain_variances: (k,) approximate covariance eigenvalues = 1 / lambda_min
      uncertain_vectors: (n_good, k) eigenvectors on the good-pixel subspace
    """
    n = int(cov_inv_good.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    k = min(int(n_modes), n)
    if k <= 0:
        return np.empty((0,), dtype=np.float64), np.empty((n, 0), dtype=np.float64)

    m = min(n, max(k + int(oversample), 2 * k))
    m = min(m, int(maxiter))
    m = max(m, k)

    A_j = jnp.asarray(cov_inv_good)
    Q = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros((m,), dtype=np.float64)
    beta = np.zeros((max(0, m - 1),), dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    q = rng.standard_normal(n).astype(np.float64)
    q /= np.linalg.norm(q) + 1e-30
    q_prev = np.zeros((n,), dtype=np.float64)
    b_prev = 0.0
    m_eff = m

    for j in tqdm(range(m), desc="lanczos-uncertain-modes", leave=True):
        Q[:, j] = q
        z = np.array(A_j @ jnp.asarray(q), dtype=np.float64, copy=True)
        if j > 0:
            z -= b_prev * q_prev
        a_j = float(np.dot(q, z))
        alpha[j] = a_j
        z -= a_j * q

        # Full re-orthogonalization for numerical stability.
        if j > 0:
            proj = Q[:, :j].T @ z
            z -= Q[:, :j] @ proj
            proj2 = Q[:, :j].T @ z
            z -= Q[:, :j] @ proj2

        if j == m - 1:
            break
        b_j = float(np.linalg.norm(z))
        beta[j] = b_j
        if b_j < 1e-14:
            m_eff = j + 1
            break
        q_prev = q
        q = z / b_j
        b_prev = b_j

    alpha = alpha[:m_eff]
    beta = beta[: max(0, m_eff - 1)]
    T = np.diag(alpha)
    if beta.size > 0:
        T += np.diag(beta, k=1) + np.diag(beta, k=-1)

    evals_t, evecs_t = np.linalg.eigh(T)
    idx = np.argsort(evals_t)[:k]
    V_ritz = Q[:, :m_eff] @ evecs_t[:, idx]

    # Orthonormalize and refine Rayleigh quotients.
    V_ritz, _ = np.linalg.qr(V_ritz, mode="reduced")
    V_ritz = V_ritz[:, :k]
    AV = np.array(A_j @ jnp.asarray(V_ritz), dtype=np.float64, copy=False)
    evals = np.sum(V_ritz * AV, axis=0)
    order = np.argsort(evals)
    evals = evals[order]
    V_ritz = V_ritz[:, order]

    evals_safe = np.maximum(evals, 1e-18)
    uncertain_variances = 1.0 / evals_safe
    return uncertain_variances.astype(np.float64), V_ritz.astype(np.float64)


def _scan_meta_entry(
    observation_id: str,
    scan_index: int,
    art: dict,
) -> dict:
    """One scan_metadata dict; same keys for single- and multi-obs."""
    return {
        "observation_id": observation_id,
        "scan_index": scan_index,
        "wind_deg_per_s": np.asarray(art["wind_deg_per_s"], dtype=np.float64).copy(),
        "wind_sigma_x_deg_per_s": art["wind_sigma_x_deg_per_s"],
        "wind_sigma_y_deg_per_s": art["wind_sigma_y_deg_per_s"],
        "ell_atm": art["ell_atm"].copy(),
        "cl_atm_mk2": art["cl_atm_mk2"].copy(),
    }


def run_synthesis(
    layout: GlobalLayout,
    scan_npz_dir: Path,
    out_path: Path,
    observation_id: str = "",
    timings: dict | None = None,
) -> None:
    """
    Exact marginalized synthesis: accumulate cov_inv_tot and Pt_Ninv_d_tot from per-scan npzs,
    solve cov_inv_tot @ c_hat_obs = Pt_Ninv_d_tot, embed to full CMB grid.
    Writes one npz with same structure as run_synthesis_multi_obs. observation_id labels the scans (e.g. single-obs id).
    """
    n_obs = layout.n_obs
    global_to_obs = layout.global_to_obs
    cov_inv_tot = np.zeros((n_obs, n_obs), dtype=np.float64)
    Pt_Ninv_d_tot = np.zeros((n_obs,), dtype=np.float64)

    existing = [i for i in range(layout.n_scans) if (scan_npz_dir / f"scan_{i:04d}_ml.npz").exists()]
    if not existing:
        raise FileNotFoundError(f"No scan_*_ml.npz found in {scan_npz_dir}")

    t_load = 0.0
    t_accum = 0.0
    scan_metadata: list[dict] = []
    for scan_index in tqdm(existing, desc="load+accumulate", leave=True):
        npz_path = scan_npz_dir / f"scan_{scan_index:04d}_ml.npz"
        t0 = time.perf_counter()
        art = load_scan_artifact(npz_path)
        t_load += time.perf_counter() - t0

        t0 = time.perf_counter()
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
        scan_metadata.append(_scan_meta_entry(observation_id, scan_index, art))
        t_accum += time.perf_counter() - t0
    if timings is not None:
        timings["load_s"] = t_load
        timings["accumulate_s"] = t_accum

    precision_diag_total = np.diag(cov_inv_tot).copy()
    good = precision_diag_total > 0.0
    zero_precision_mask = ~good
    c_hat_obs = np.zeros((n_obs,), dtype=np.float64)

    print(f"[solve] n_obs={n_obs} n_good={int(np.sum(good))}", flush=True)
    t0 = time.perf_counter()
    if np.any(good):
        cov_inv_good = cov_inv_tot[np.ix_(good, good)]
        Pt_Ninv_d_good = Pt_Ninv_d_tot[good]
        c_hat_good = _solve_synthesis(cov_inv_good, Pt_Ninv_d_good)
        c_hat_good = np.asarray(c_hat_good, dtype=np.float64)
        c_hat_obs[good] = c_hat_good
        c_hat_obs -= float(np.mean(c_hat_obs[good]))
    if timings is not None:
        timings["solve_s"] = time.perf_counter() - t0

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
    t0 = time.perf_counter()
    if n_good > 0:
        uncertain_variances, uncertain_vectors = _lanczos_smallest_modes(
            cov_inv_tot[np.ix_(good, good)],
            n_modes=N_UNCERTAIN_MODES,
            oversample=LANCZOS_OVERSAMPLE,
            maxiter=LANCZOS_MAXITER,
            seed=LANCZOS_SEED,
        )
        k = uncertain_vectors.shape[1]
        full_uncertain = np.zeros((n_obs, k), dtype=np.float64)
        full_uncertain[good] = uncertain_vectors
    else:
        full_uncertain = np.empty((n_obs, 0), dtype=np.float64)
        uncertain_variances = np.empty((0,), dtype=np.float64)
    if timings is not None:
        timings["uncertain_modes_s"] = time.perf_counter() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
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
    if timings is not None:
        timings["write_s"] = time.perf_counter() - t0
    print(f"[write] {out_path} n_obs={n_obs} n_scans_used={len(existing)}/{layout.n_scans}", flush=True)


def run_synthesis_multi_obs(
    out_base: Path,
    field_id: str,
    observation_ids: list[str],
    out_subdir: str = "synthesized",
    timings: dict | None = None,
) -> tuple[GlobalLayout, Path]:
    """
    Load all scan npzs from OUT_BASE/field_id/obs_id/scans/ for each obs_id, build combined layout,
    accumulate cov_inv and Pt_Ninv_d, solve, write single OUT_BASE/field_id/<out_subdir>/recon_combined_ml.npz.
    Output npz has same structure as run_synthesis (scan_metadata includes observation_id per scan).
    Returns (combined_layout, out_npz_path).
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
    scan_metadata: list[dict] = []

    t_load = 0.0
    t_accum = 0.0
    for obs_id, scan_index, npz_path in tqdm(artifact_paths, desc="load+accumulate", leave=True):
        t0 = time.perf_counter()
        art = load_scan_artifact(npz_path)
        t_load += time.perf_counter() - t0

        t0 = time.perf_counter()
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
        scan_metadata.append(_scan_meta_entry(obs_id, scan_index, art))
        t_accum += time.perf_counter() - t0
    if timings is not None:
        timings["load_s"] = t_load
        timings["accumulate_s"] = t_accum

    precision_diag_total = np.diag(cov_inv_tot).copy()
    good = precision_diag_total > 0.0
    zero_precision_mask = ~good
    c_hat_obs = np.zeros((n_obs,), dtype=np.float64)
    print(f"[solve] n_obs={n_obs} n_good={int(np.sum(good))}", flush=True)
    t0 = time.perf_counter()
    if np.any(good):
        cov_inv_good = cov_inv_tot[np.ix_(good, good)]
        Pt_Ninv_d_good = Pt_Ninv_d_tot[good]
        c_hat_good = _solve_synthesis(cov_inv_good, Pt_Ninv_d_good)
        c_hat_obs[good] = np.asarray(c_hat_good, dtype=np.float64)
        c_hat_obs -= float(np.mean(c_hat_obs[good]))
    if timings is not None:
        timings["solve_s"] = time.perf_counter() - t0

    var_diag_total = np.full((n_obs,), np.inf, dtype=np.float64)
    var_diag_total[good] = np.where(
        precision_diag_total[good] > 0.0,
        1.0 / precision_diag_total[good],
        np.inf,
    )
    c_hat_full_mk = np.zeros((n_pix,), dtype=np.float64)
    c_hat_full_mk[combined_layout.obs_pix_global] = c_hat_obs

    n_good = int(np.sum(good))
    t0 = time.perf_counter()
    if n_good > 0:
        uncertain_variances, uncertain_vectors = _lanczos_smallest_modes(
            cov_inv_tot[np.ix_(good, good)],
            n_modes=N_UNCERTAIN_MODES,
            oversample=LANCZOS_OVERSAMPLE,
            maxiter=LANCZOS_MAXITER,
            seed=LANCZOS_SEED,
        )
    else:
        uncertain_vectors = np.empty((0, 0), dtype=np.float64)
        uncertain_variances = np.empty((0,), dtype=np.float64)
    if timings is not None:
        timings["uncertain_modes_s"] = time.perf_counter() - t0

    out_dir = out_base / field_id / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "recon_combined_ml.npz"
    t0 = time.perf_counter()
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
    if timings is not None:
        timings["write_s"] = time.perf_counter() - t0
    print(f"[write] {out_npz} n_obs={n_obs} n_scans={len(artifact_paths)}", flush=True)
    return combined_layout, out_npz
