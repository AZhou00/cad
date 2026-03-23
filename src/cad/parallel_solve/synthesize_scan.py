"""
Parallel solve: exact global synthesis from per-scan cov_inv and Pt_Ninv_d.

Joint synthesis (theory): cov_inv_tot = sum_s [Cov(hat c_s)]^{-1}, Pt_Ninv_d_tot = sum_s P_s' tilde N_s^{-1} d_s.
Solve (cov_inv_tot @ c_hat = Pt_Ninv_d_tot) for ML map on observed pixels; gauge by mean subtraction.
Unconstrained modes: eigenvectors of the precision [Cov(hat c)]^{-1} with smallest eigenvalues
(largest posterior variance). Lanczos estimates the first n_uncertain_modes directions on the good-pixel
subspace; plotting may slice fewer columns (deprojection) without re-running synthesis.

Output NPZ (single file per run, path = out_path argument)
---------------------------------------------------------
Scalars / small: estimator_mode (str), bbox_ix0, bbox_iy0, nx, ny, pixel_size_deg,
  n_scans, n_scans_used, lanczos_n_modes, n_uncertain_modes_stored.
Maps / vectors on observed index ordering (length n_obs = |obs_pix_global|):
  c_hat_obs (n_obs,), obs_pix_global (n_obs,) global flat indices pix = iy + ix*ny,
  precision_diag_total (n_obs,), var_diag_total (n_obs,), zero_precision_mask (n_obs,) bool,
  good_mask (n_obs,) bool (True where diag precision > 0).
Full plate grid (length n_pix = nx * ny): c_hat_full_mk (n_pix,) mK, NaN/unobserved as 0 off footprint.
Dense: cov_inv_tot (n_obs, n_obs) global ML precision sum.
Uncertain subspace on good pixels only: uncertain_mode_vectors (n_good, k), uncertain_mode_variances (k,)
  with n_good = sum(good_mask), k <= n_uncertain_modes requested (capped by n_good).
Provenance: scan_metadata object array of dicts (observation_id, scan_index, wind_*, ell_atm, cl_atm_mk2).
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .artifact_io import load_scan_artifact
from .layout import GlobalLayout, load_layout

jax.config.update("jax_enable_x64", True)

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
    n_modes: int,
    oversample: int,
    maxiter: int,
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


def _save_combined_npz(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


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


def _trim_sides(n: int, margin_frac: float) -> int:
    return int(np.floor(float(n) * float(margin_frac)))


def _margined_bbox(
    *,
    bbox_ix0: int,
    bbox_iy0: int,
    nx: int,
    ny: int,
    margin_frac: float,
) -> tuple[int, int, int, int, int, int]:
    mx = _trim_sides(nx, margin_frac)
    my = _trim_sides(ny, margin_frac)
    nx_inner = int(nx - 2 * mx)
    ny_inner = int(ny - 2 * my)
    if nx_inner <= 0 or ny_inner <= 0:
        raise ValueError(f"margin_frac={margin_frac} too large for nx={nx}, ny={ny}")
    return int(bbox_ix0 + mx), int(bbox_iy0 + my), nx_inner, ny_inner, mx, my


def _remap_pixels_to_inner_bbox(
    pix_old: np.ndarray,
    *,
    nx_old: int,
    ny_old: int,
    mx: int,
    my: int,
    ny_inner: int,
) -> tuple[np.ndarray, np.ndarray]:
    pix = np.asarray(pix_old, dtype=np.int64)
    ix_old = pix // int(ny_old)
    iy_old = pix - ix_old * int(ny_old)
    keep = (
        (ix_old >= int(mx))
        & (ix_old < int(nx_old - mx))
        & (iy_old >= int(my))
        & (iy_old < int(ny_old - my))
    )
    if not np.any(keep):
        return np.empty((0,), dtype=np.int64), keep
    ix_new = ix_old[keep] - int(mx)
    iy_new = iy_old[keep] - int(my)
    pix_new = iy_new + ix_new * int(ny_inner)
    return np.asarray(pix_new, dtype=np.int64), keep


def _margined_obs_index(
    layout: GlobalLayout,
    *,
    margin_frac: float,
) -> tuple[int, int, int, int, int, int, np.ndarray, np.ndarray]:
    bbox_ix0, bbox_iy0, nx, ny, mx, my = _margined_bbox(
        bbox_ix0=layout.bbox_ix0,
        bbox_iy0=layout.bbox_iy0,
        nx=layout.nx,
        ny=layout.ny,
        margin_frac=margin_frac,
    )
    obs_pix_global, _ = _remap_pixels_to_inner_bbox(
        layout.obs_pix_global,
        nx_old=layout.nx,
        ny_old=layout.ny,
        mx=mx,
        my=my,
        ny_inner=ny,
    )
    n_pix = int(nx * ny)
    global_to_obs = np.full((n_pix,), -1, dtype=np.int64)
    if obs_pix_global.size > 0:
        global_to_obs[obs_pix_global] = np.arange(obs_pix_global.size, dtype=np.int64)
    return bbox_ix0, bbox_iy0, nx, ny, mx, my, obs_pix_global, global_to_obs


def run_synthesis(
    layout: GlobalLayout,
    scan_npz_dir: Path,
    out_path: Path,
    n_uncertain_modes: int,
    lanczos_oversample: int,
    lanczos_maxiter: int,
    observation_id: str = "",
    timings: dict | None = None,
    margin_frac: float = 0.0,
) -> None:
    """
    Exact marginalized synthesis: accumulate cov_inv_tot and Pt_Ninv_d_tot from per-scan npzs,
    solve cov_inv_tot @ c_hat_obs = Pt_Ninv_d_tot, embed to full CMB grid.

    Writes exactly one NPZ at out_path. Lanczos requests n_uncertain_modes smallest-eigenvalue
    directions of precision on good pixels; store full columns in uncertain_mode_vectors (see module docstring).
    """
    k_lanczos = int(n_uncertain_modes)
    if k_lanczos <= 0:
        raise ValueError("n_uncertain_modes must be a positive integer.")
    bbox_ix0, bbox_iy0, nx, ny, mx, my, obs_pix_global, global_to_obs = _margined_obs_index(
        layout,
        margin_frac=margin_frac,
    )
    n_obs = int(obs_pix_global.size)
    n_pix = int(nx * ny)
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
        obs_pix_global_scan_inner, keep = _remap_pixels_to_inner_bbox(
            art["obs_pix_global_scan"],
            nx_old=layout.nx,
            ny_old=layout.ny,
            mx=mx,
            my=my,
            ny_inner=ny,
        )
        cov_inv_s = art["cov_inv"]
        Pt_Ninv_d_s = art["Pt_Ninv_d"]
        if not np.any(keep):
            scan_metadata.append(_scan_meta_entry(observation_id, scan_index, art))
            t_accum += time.perf_counter() - t0
            continue
        cov_inv_s = cov_inv_s[np.ix_(keep, keep)]
        Pt_Ninv_d_s = Pt_Ninv_d_s[keep]
        obs_idx = np.asarray(global_to_obs[obs_pix_global_scan_inner], dtype=np.int64)
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

    c_hat_full_mk = np.zeros((n_pix,), dtype=np.float64)
    c_hat_full_mk[obs_pix_global] = c_hat_obs

    n_good = int(np.sum(good))
    t0 = time.perf_counter()
    if n_good > 0:
        uncertain_variances, uncertain_vectors = _lanczos_smallest_modes(
            cov_inv_tot[np.ix_(good, good)],
            n_modes=k_lanczos,
            oversample=lanczos_oversample,
            maxiter=lanczos_maxiter,
            seed=LANCZOS_SEED,
        )
    else:
        uncertain_vectors = np.empty((0, 0), dtype=np.float64)
        uncertain_variances = np.empty((0,), dtype=np.float64)
    if timings is not None:
        timings["uncertain_modes_s"] = time.perf_counter() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_stored = int(uncertain_vectors.shape[1]) if uncertain_vectors.size else 0
    _save_combined_npz(
        out_path,
        dict(
            estimator_mode=np.array("ML", dtype=object),
            bbox_ix0=np.int64(bbox_ix0),
            bbox_iy0=np.int64(bbox_iy0),
            nx=np.int64(nx),
            ny=np.int64(ny),
            pixel_size_deg=np.float64(layout.pixel_size_deg),
            obs_pix_global=obs_pix_global,
            c_hat_full_mk=c_hat_full_mk,
            c_hat_obs=c_hat_obs,
            precision_diag_total=precision_diag_total,
            var_diag_total=var_diag_total,
            zero_precision_mask=zero_precision_mask,
            n_scans=np.int64(layout.n_scans),
            n_scans_used=np.int64(len(existing)),
            cov_inv_tot=cov_inv_tot,
            good_mask=good,
            scan_metadata=np.array(scan_metadata, dtype=object),
            uncertain_mode_vectors=uncertain_vectors,
            uncertain_mode_variances=uncertain_variances,
            n_uncertain_modes_stored=np.int64(n_stored),
            lanczos_n_modes=np.int64(k_lanczos),
        ),
    )
    print(f"[write] {out_path} n_obs={n_obs} n_scans_used={len(existing)}/{layout.n_scans}", flush=True)
    if timings is not None:
        timings["write_s"] = time.perf_counter() - t0


def run_synthesis_multi_obs(
    out_base: Path,
    field_id: str,
    observation_ids: list[str],
    n_uncertain_modes: int,
    lanczos_oversample: int,
    lanczos_maxiter: int,
    out_subdir: str = "synthesized",
    out_filename: str = "recon_combined_ml.npz",
    timings: dict | None = None,
    margin_frac: float = 0.0,
) -> tuple[GlobalLayout, Path]:
    """
    Load all scan npzs from OUT_BASE/field_id/obs_id/scans/ for each obs_id, build combined layout,
    accumulate cov_inv and Pt_Ninv_d, solve, write one NPZ at out_dir / out_filename (see run_synthesis).
    Returns (combined_layout, written_npz_path).
    """
    if not observation_ids:
        raise ValueError("observation_ids must be non-empty")
    k_lanczos = int(n_uncertain_modes)
    if k_lanczos <= 0:
        raise ValueError("n_uncertain_modes must be a positive integer.")
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

    # Preserve original trajectory: build one combined/global layout first.
    first = layouts[0][1]
    obs_pix_union = sorted(set().union(*(set(lay.obs_pix_global.tolist()) for _, lay in layouts)))
    n_pix_full = first.n_pix
    global_to_obs_full = np.full(n_pix_full, -1, dtype=np.int64)
    for i, p in enumerate(obs_pix_union):
        global_to_obs_full[p] = i
    combined_layout_full = GlobalLayout(
        bbox_ix0=first.bbox_ix0,
        bbox_iy0=first.bbox_iy0,
        nx=first.nx,
        ny=first.ny,
        obs_pix_global=np.array(obs_pix_union, dtype=np.int64),
        global_to_obs=global_to_obs_full,
        scan_paths=(),
        pixel_size_deg=first.pixel_size_deg,
        field_id=field_id,
    )

    # Then apply one global margin in that combined coordinate system.
    bbox_ix0, bbox_iy0, nx, ny, mx, my, obs_pix_global, global_to_obs = _margined_obs_index(
        combined_layout_full,
        margin_frac=margin_frac,
    )
    n_obs = int(obs_pix_global.size)
    n_pix = int(nx * ny)
    combined_layout = GlobalLayout(
        bbox_ix0=bbox_ix0,
        bbox_iy0=bbox_iy0,
        nx=nx,
        ny=ny,
        obs_pix_global=obs_pix_global,
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
        obs_pix_global_scan_inner, keep = _remap_pixels_to_inner_bbox(
            art["obs_pix_global_scan"],
            nx_old=combined_layout_full.nx,
            ny_old=combined_layout_full.ny,
            mx=mx,
            my=my,
            ny_inner=combined_layout.ny,
        )
        cov_inv_s = art["cov_inv"]
        Pt_Ninv_d_s = art["Pt_Ninv_d"]
        if not np.any(keep):
            scan_metadata.append(_scan_meta_entry(obs_id, scan_index, art))
            t_accum += time.perf_counter() - t0
            continue
        cov_inv_s = cov_inv_s[np.ix_(keep, keep)]
        Pt_Ninv_d_s = Pt_Ninv_d_s[keep]
        obs_idx = np.asarray(combined_layout.global_to_obs[obs_pix_global_scan_inner], dtype=np.int64)
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
            n_modes=k_lanczos,
            oversample=lanczos_oversample,
            maxiter=lanczos_maxiter,
            seed=LANCZOS_SEED,
        )
    else:
        uncertain_vectors = np.empty((0, 0), dtype=np.float64)
        uncertain_variances = np.empty((0,), dtype=np.float64)
    if timings is not None:
        timings["uncertain_modes_s"] = time.perf_counter() - t0

    out_dir = out_base / field_id / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / out_filename
    t0 = time.perf_counter()
    n_stored = int(uncertain_vectors.shape[1]) if uncertain_vectors.size else 0
    _save_combined_npz(
        out_npz,
        dict(
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
            scan_metadata=np.array(scan_metadata, dtype=object),
            uncertain_mode_vectors=uncertain_vectors,
            uncertain_mode_variances=uncertain_variances,
            n_uncertain_modes_stored=np.int64(n_stored),
            lanczos_n_modes=np.int64(k_lanczos),
        ),
    )
    print(f"[write] {out_npz} n_obs={n_obs} n_scans={len(artifact_paths)}", flush=True)
    if timings is not None:
        timings["write_s"] = time.perf_counter() - t0
    return combined_layout, out_npz
