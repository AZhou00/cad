"""
Build per-scan information form F_s = P^T Ntilde^{-1} P and b_s = P^T Ntilde^{-1} d.

Woodbury: M_s = C_a^{-1} + W^T N^{-1} W; Ntilde_s^{-1} = N^{-1} - N^{-1} W M_s^{-1} W^T N^{-1}.
Uses matrix-free M_s (CG) when n_pix_atm is large; builds F_s column-by-column.
Optional/advanced path; default parallel pipeline uses diagonal covariance.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from cad.prior import FourierGaussianPrior


def _w_apply(idx4: np.ndarray, w4: np.ndarray, a: np.ndarray) -> np.ndarray:
    """(n_valid,) = W a."""
    return np.sum(w4 * a[idx4], axis=1)


def _wt_apply(idx4: np.ndarray, w4: np.ndarray, x: np.ndarray, n_pix_atm: int) -> np.ndarray:
    """(n_pix_atm,) = W^T x."""
    out = np.zeros((n_pix_atm,), dtype=np.float64)
    np.add.at(out, idx4.reshape(-1), (w4 * x[:, None]).reshape(-1))
    return out


def build_fisher_and_rhs(
    *,
    d: np.ndarray,
    inv_var: np.ndarray,
    pix_obs_local: np.ndarray,
    idx4: np.ndarray,
    w4: np.ndarray,
    prior_atm: FourierGaussianPrior,
    n_obs_scan: int,
    cg_tol: float = 1e-4,
    cg_maxiter: int = 20000,
    max_exact_obs_columns: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute F_s (n_obs_scan, n_obs_scan), b_s (n_obs_scan), c_hat_s (n_obs_scan).
    """
    n_valid = int(d.size)
    n_pix_atm = int(prior_atm.nx * prior_atm.ny)
    if n_obs_scan > int(max_exact_obs_columns):
        raise RuntimeError(
            "Exact dense F_s construction is too expensive for this scan: "
            f"n_obs_scan={n_obs_scan} exceeds max_exact_obs_columns={max_exact_obs_columns}. "
            "Use a smaller observed set (e.g. higher min_hits_per_pix) or switch to an operator/factor storage path."
        )

    d = np.asarray(d, dtype=np.float64).reshape(-1)
    inv_var = np.asarray(inv_var, dtype=np.float64).reshape(-1)
    pix_obs_local = np.asarray(pix_obs_local, dtype=np.int64).reshape(-1)

    reg_eps = 1e-10 * (float(np.mean(inv_var)) * 4.0 + 1e-12)

    def M_matvec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        return (
            prior_atm.apply_Cinv(v)
            + _wt_apply(idx4, w4, inv_var * _w_apply(idx4, w4, v), n_pix_atm)
            + reg_eps * v
        )

    M_op = spla.LinearOperator(
        (n_pix_atm, n_pix_atm),
        matvec=M_matvec,
        dtype=np.float64,
    )

    diag_WtNW = np.bincount(
        idx4.reshape(-1),
        weights=(w4 * w4 * inv_var[:, None]).reshape(-1),
        minlength=n_pix_atm,
    ).astype(np.float64)
    e0 = np.zeros((n_pix_atm,), dtype=np.float64)
    e0[0] = 1.0
    diag_Ca_inv_0 = float(prior_atm.apply_Cinv(e0)[0])
    diag_M = np.maximum(diag_WtNW + diag_Ca_inv_0 + reg_eps, 1e-14)
    M_pre = spla.LinearOperator(
        (n_pix_atm, n_pix_atm),
        matvec=lambda x: np.asarray(x, dtype=np.float64).reshape(-1) / diag_M,
        dtype=np.float64,
    )

    rhs_u = _wt_apply(idx4, w4, inv_var * d, n_pix_atm)
    u, info = spla.cg(M_op, rhs_u, M=M_pre, atol=0.0, rtol=float(cg_tol), maxiter=int(cg_maxiter))
    if info != 0:
        raise RuntimeError(f"M_s solve did not converge (info={info})")
    u = np.asarray(u, dtype=np.float64).reshape(-1)

    y = inv_var * d - inv_var * _w_apply(idx4, w4, u)
    y = np.asarray(y, dtype=np.float64)

    b_s = np.bincount(pix_obs_local, weights=y, minlength=n_obs_scan).astype(np.float64)
    F_s = np.zeros((n_obs_scan, n_obs_scan), dtype=np.float64)
    for j in range(n_obs_scan):
        z = np.zeros((n_valid,), dtype=np.float64)
        z[pix_obs_local == j] = 1.0
        rhs_u_j = _wt_apply(idx4, w4, inv_var * z, n_pix_atm)
        u_j, info_j = spla.cg(M_op, rhs_u_j, M=M_pre, atol=0.0, rtol=float(cg_tol), maxiter=int(cg_maxiter))
        if info_j != 0:
            raise RuntimeError(f"M_s solve for column {j} did not converge (info={info_j})")
        u_j = np.asarray(u_j, dtype=np.float64).reshape(-1)
        w_j = inv_var * z - inv_var * _w_apply(idx4, w4, u_j)
        F_s[:, j] = np.bincount(pix_obs_local, weights=w_j, minlength=n_obs_scan).astype(np.float64)
    F_s = 0.5 * (F_s + F_s.T)

    try:
        c_hat_s = la.solve(F_s, b_s, assume_a="sym", check_finite=False)
    except la.LinAlgError as e:
        raise RuntimeError(
            "Per-scan information solve failed (singular/ill-conditioned F_s). "
            "Check scan coverage or gauge handling."
        ) from e
    c_hat_s = np.asarray(c_hat_s, dtype=np.float64)
    c_hat_s -= float(np.mean(c_hat_s))

    return F_s, b_s, c_hat_s
