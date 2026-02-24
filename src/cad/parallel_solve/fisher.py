"""
Build per-scan [Cov(hat c_s)]^{-1} and P^T tilde N^{-1} d; point estimate from the normal equation.

M_s = C_a^{-1} + W^T N^{-1} W; tilde N^{-1} applied via Woodbury.
Column j of cov_inv from one M_s solve with RHS W^T N^{-1} z_j (z_j unit on pixel j).
Pt_Ninv_d = P^T y with y = N^{-1} d - N^{-1} W u, u = M_s^{-1}(W^T N^{-1} d).
Requires JAX with GPU (set CUDA_VISIBLE_DEVICES per process).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

import jax
import jax.numpy as jnp
from jax import jit

jax.config.update("jax_enable_x64", True)

# Fixed batch size for cov_inv column builds to avoid recompilation
COV_INV_BATCH_SIZE = 64
# Fixed CG iterations (enough for convergence in benchmark)
CG_NITER = 328


def _apply_cinv(
    x_pix: jnp.ndarray,
    nx: int,
    ny: int,
    cl_per_mode: jnp.ndarray,
    dxdy: float,
) -> jnp.ndarray:
    """Apply C_a^{-1}. x_pix (n_pix_atm,), out (n_pix_atm,)."""
    x2 = jnp.reshape(x_pix, (nx, ny))
    X = jnp.fft.rfft2(x2)
    X = jnp.fft.fftshift(X, axes=0)
    X = X * (dxdy / cl_per_mode)
    X = jnp.fft.ifftshift(X, axes=0)
    out = jnp.fft.irfft2(X, s=(nx, ny))
    return jnp.real(jnp.reshape(out, (-1,)))


def _w_apply(
    idx4: jnp.ndarray,
    w4: jnp.ndarray,
    a: jnp.ndarray,
) -> jnp.ndarray:
    """(n_valid,) = W a. idx4 (n_valid,4), w4 (n_valid,4), a (n_pix_atm,)."""
    return jnp.sum(w4 * a[idx4], axis=1)


def _wt_apply(
    idx4: jnp.ndarray,
    w4: jnp.ndarray,
    x: jnp.ndarray,
    n_pix_atm: int,
) -> jnp.ndarray:
    """(n_pix_atm,) = W^T x. Scatter-add."""
    flat_idx = jnp.reshape(idx4, (-1,))
    flat_val = jnp.reshape(w4 * jnp.expand_dims(x, axis=1), (-1,))
    return jnp.zeros((n_pix_atm,), dtype=x.dtype).at[flat_idx].add(flat_val)


def _cg_batched(
    M_matvec_batched: callable,
    rhs_batch: jnp.ndarray,
    diag_precond: jnp.ndarray,
    niter: int = CG_NITER,
) -> jnp.ndarray:
    """Fixed-iteration batched CG. rhs_batch (K, n_pix_atm), out (K, n_pix_atm)."""
    x = jnp.zeros_like(rhs_batch)
    r = rhs_batch
    z = rhs_batch / diag_precond
    p = z
    rho = jnp.sum(r * z, axis=1)

    def body(carry, _):
        x, r, z, p, rho = carry
        Ap = M_matvec_batched(p)
        alpha = rho / (jnp.sum(p * Ap, axis=1) + 1e-30)
        x = x + alpha[:, None] * p
        r = r - alpha[:, None] * Ap
        z = r / diag_precond
        rho_new = jnp.sum(r * z, axis=1)
        beta = rho_new / (rho + 1e-30)
        p = z + beta[:, None] * p
        return (x, r, z, p, rho_new), None

    (x, _, _, _, _), _ = jax.lax.scan(body, (x, r, z, p, rho), None, length=niter)
    return x


def _cg_single(
    M_matvec: callable,
    rhs: jnp.ndarray,
    diag_precond: jnp.ndarray,
    niter: int = CG_NITER,
) -> jnp.ndarray:
    """Fixed-iteration single-RHS CG. rhs (n_pix_atm,), out (n_pix_atm,)."""
    x = jnp.zeros_like(rhs)
    r = rhs
    z = rhs / diag_precond
    p = z
    rho = jnp.dot(r, z)

    def body(carry, _):
        x, r, z, p, rho = carry
        Ap = M_matvec(p)
        alpha = rho / (jnp.dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = r / diag_precond
        rho_new = jnp.dot(r, z)
        beta = rho_new / (rho + 1e-30)
        p = z + beta * p
        return (x, r, z, p, rho_new), None

    (x, _, _, _, _), _ = jax.lax.scan(body, (x, r, z, p, rho), None, length=niter)
    return x


def build_scan_information(
    d: np.ndarray,
    inv_var: np.ndarray,
    pix_obs_local: np.ndarray,
    idx4: np.ndarray,
    w4: np.ndarray,
    nx: int,
    ny: int,
    cl_per_mode: np.ndarray,
    dxdy: float,
    diag_M: np.ndarray,
    n_obs_scan: int,
    n_pix_atm: int,
    reg_eps: float,
    batch_size: int = COV_INV_BATCH_SIZE,
    cg_niter: int = CG_NITER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cov_inv_s (n_obs_scan, n_obs_scan), Pt_Ninv_d_s (n_obs_scan), c_hat_scan_obs (n_obs_scan).

    cov_inv_s = [Cov(hat c_s)]^{-1} = P^T tilde N^{-1} P; Pt_Ninv_d_s = P^T tilde N^{-1} d.
    Inputs: d (n_valid,) valid TOD; inv_var (n_valid,); pix_obs_local (n_valid,) in [0, n_obs_scan);
    idx4 (n_valid, 4), w4 (n_valid, 4); nx, ny, cl_per_mode, dxdy for C_a^{-1}; diag_M (n_pix_atm,) preconditioner.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    inv_var = np.asarray(inv_var, dtype=np.float64).reshape(-1)
    pix_obs_local = np.asarray(pix_obs_local, dtype=np.int64).reshape(-1)
    n_valid = int(d.size)

    idx4_j = jnp.asarray(idx4)
    w4_j = jnp.asarray(w4)
    inv_var_j = jnp.asarray(inv_var)
    pix_obs_local_j = jnp.asarray(pix_obs_local)
    cl_per_mode_j = jnp.asarray(cl_per_mode)
    diag_M_j = jnp.asarray(diag_M)

    def M_matvec(v: jnp.ndarray) -> jnp.ndarray:
        wv = _w_apply(idx4_j, w4_j, v)
        wt_term = _wt_apply(idx4_j, w4_j, inv_var_j * wv, n_pix_atm)
        cinv_term = _apply_cinv(v, nx, ny, cl_per_mode_j, dxdy)
        return cinv_term + wt_term + reg_eps * v

    M_matvec_batched = jax.vmap(M_matvec, in_axes=0, out_axes=0)

    # Pt_Ninv_d_s: one M_s solve for u, then y = inv_var*d - inv_var*W*u, Pt_Ninv_d_s = P^T y
    Wt_Ninv_d = _wt_apply(idx4_j, w4_j, inv_var_j * jnp.asarray(d), n_pix_atm)
    run_single = jit(lambda rhs: _cg_single(M_matvec, rhs, diag_M_j, niter=cg_niter))
    u_jax = run_single(Wt_Ninv_d)
    wu_jax = _w_apply(idx4_j, w4_j, u_jax)
    y = np.asarray(inv_var_j * jnp.asarray(d, dtype=jnp.float64) - inv_var_j * wu_jax, dtype=np.float64)
    Pt_Ninv_d_s = np.bincount(pix_obs_local, weights=y, minlength=n_obs_scan).astype(np.float64)

    # cov_inv_s: column j from M_s solve with RHS W^T (inv_var * z_j), then w_j = inv_var*z_j - inv_var*W*u_j, cov_inv_s[:,j] = P^T w_j
    obs_uniq = np.unique(pix_obs_local)
    n_obs = int(obs_uniq.size)
    cov_inv_s = np.zeros((n_obs_scan, n_obs_scan), dtype=np.float64)

    @jit
    def build_chunk_jitted(z_batch: jnp.ndarray) -> jnp.ndarray:
        """z_batch (K, n_valid). Returns cov_inv_cols (K, n_obs_scan)."""
        rhs_batch = jax.vmap(
            lambda z: _wt_apply(idx4_j, w4_j, inv_var_j * z, n_pix_atm),
            in_axes=0,
            out_axes=0,
        )(z_batch)
        u_batch = _cg_batched(M_matvec_batched, rhs_batch, diag_M_j, niter=cg_niter)
        w_batch = inv_var_j * z_batch - inv_var_j * jax.vmap(
            lambda a: _w_apply(idx4_j, w4_j, a)
        )(u_batch)
        cov_inv_cols = jax.vmap(
            lambda w: jnp.bincount(pix_obs_local_j, weights=w, minlength=n_obs_scan),
            in_axes=0,
            out_axes=0,
        )(w_batch)
        return cov_inv_cols

    for j_start in range(0, n_obs, batch_size):
        j_end = min(j_start + batch_size, n_obs)
        j_list = obs_uniq[j_start:j_end]
        n_chunk = j_list.size
        z_batch = np.zeros((batch_size, n_valid), dtype=np.float64)
        z_batch[:n_chunk] = (pix_obs_local[None, :] == j_list[:, None]).astype(np.float64)
        z_batch_j = jnp.asarray(z_batch)
        cov_inv_cols = np.asarray(build_chunk_jitted(z_batch_j))
        for k in range(n_chunk):
            cov_inv_s[:, j_list[k]] = cov_inv_cols[k, :]
    cov_inv_s = 0.5 * (cov_inv_s + cov_inv_s.T)

    try:
        c_hat_scan_obs = la.solve(cov_inv_s, Pt_Ninv_d_s, assume_a="sym", check_finite=False)
    except la.LinAlgError as err:
        raise RuntimeError(
            "Per-scan normal-equation solve failed (singular/ill-conditioned cov_inv). "
            "Check scan coverage or gauge handling."
        ) from err
    c_hat_scan_obs = np.asarray(c_hat_scan_obs, dtype=np.float64)
    c_hat_scan_obs -= float(np.mean(c_hat_scan_obs))

    return cov_inv_s, Pt_Ninv_d_s, c_hat_scan_obs
