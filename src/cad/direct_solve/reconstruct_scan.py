"""
Single-scan CMB reconstruction via a joint linear solve.

We solve the normal equations for the CMB map c and a per-scan atmosphere
reference screen a0 under the frozen-screen model:

  d = P c + W a0 + n,    n ~ N(0, N)

with diagonal N (white, per-detector) and stationary FFT-diagonal priors on a0
(and optionally on c for MAP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse.linalg as spla

from cad.map import BBox
from cad.prior import FourierGaussianPrior
from cad.util import frozen_screen_bilinear_weights, pointing_from_pix_index


@dataclass
class ScanSolve:
    """
    Single-scan reconstruction output + operator state for multi-scan synthesis.

    Public fields:
      - c_hat_full_mk: (n_pix_cmb,) CMB map on the full CMB bbox grid (mK)
      - bbox_cmb: CMB bbox on global pixel grid
      - bbox_atm: atmosphere bbox on global pixel grid (typically padded)
      - obs_pix_cmb: (n_obs,) indices into the CMB full grid for the shared observed-pixel set
      - estimator_mode: 'ML' or 'MAP'
      - n_scans: total scans intended for synthesis (controls MAP prior scaling)

    Internal fields (used by `cad.synthesize_scan`):
      - pix_obs_local: (n_valid,) obs-pixel indices for each valid TOD sample
      - idx4, w4: (n_valid,4) atmosphere bilinear corners/weights into a0.ravel()
      - inv_var: (n_valid,) per-sample inverse variances
      - tod_valid_mk: (n_valid,) valid TOD vector d in same order as pix_obs_local
      - rhs_c: (n_obs,) P^T N^{-1} d on obs basis
      - rhs_a: (n_pix_atm,) W^T N^{-1} d on atmosphere grid
      - prior_atm, prior_cmb: FFT priors (stationary, periodic on their domains)
    """

    c_hat_full_mk: np.ndarray
    bbox_cmb: BBox
    bbox_atm: BBox
    obs_pix_cmb: np.ndarray
    estimator_mode: str
    n_scans: int

    # internal
    pix_obs_local: np.ndarray
    idx4: np.ndarray
    w4: np.ndarray
    inv_var: np.ndarray
    tod_valid_mk: np.ndarray
    rhs_c: np.ndarray
    rhs_a: np.ndarray
    prior_atm: FourierGaussianPrior
    prior_cmb: FourierGaussianPrior


def solve_single_scan(
    *,
    tod_mk: np.ndarray,
    pix_index: np.ndarray,
    t_s: np.ndarray,
    pixel_size_deg: float,
    wind_deg_per_s: tuple[float, float],
    noise_std_det_mk: np.ndarray,
    prior_atm: FourierGaussianPrior,
    prior_cmb: FourierGaussianPrior,
    bbox_cmb: BBox,
    bbox_atm: BBox,
    obs_pix_cmb: np.ndarray,
    global_to_obs_cmb: np.ndarray,
    estimator_mode: Literal["ML", "MAP"] = "ML",
    n_scans: int = 1,
    cl_floor_mk2: float = 1e-12,
    cg_tol: float = 1e-3,
    cg_maxiter: int = 400,
) -> ScanSolve:
    """
    Matrix-free single-scan solve for (c, a0).

    Model (one scan):

    - **Data**: d ∈ R^(n_valid) are the finite TOD samples (mK).
    - **Pointing**: P selects one CMB pixel per sample from the CMB bbox grid.
    - **Atmosphere**: W bilinear-samples the reference screen a0 on `bbox_atm`.

    Frozen-screen sampling:

      x_src(t) = i_x - (w_x / pixel_size_deg) * (t - t0)
      y_src(t) = i_y - (w_y / pixel_size_deg) * (t - t0)

    with t0 = t_s[0] and pixel_size_deg converting deg/s to pixels/s.

    Priors:

    - a0 ~ N(0, C_a) always.
    - MAP only: c ~ N(0, C_c) applied as (1/n_scans) * C_c^{-1} per scan.

    Joint normal equations (MAP adds the Cc⁻¹ term; ML omits it):

      [ Pᵀ N⁻¹ P + I_MAP*(1/n_scans) Cc⁻¹    Pᵀ N⁻¹ W ] [ c  ] = [ Pᵀ N⁻¹ d ]
      [ Wᵀ N⁻¹ P                            Wᵀ N⁻¹ W + Ca⁻¹ ] [ a0 ]   [ Wᵀ N⁻¹ d ]

    Args:
      tod_mk: (n_t, n_det) TOD in mK.
      pix_index: (n_t, n_det, 2) global (ix,iy) indices corresponding to tod samples.
      t_s: (n_t,) time stamps in seconds.
      pixel_size_deg: scalar pixel size in degrees.
      wind_deg_per_s: (w_x, w_y) deg/s in the RA/Dec-degree basis.
      noise_std_det_mk: (n_det,) per-detector noise std in mK (diagonal N).
      prior_atm: C_a on the atmosphere grid `bbox_atm`.
      prior_cmb: C_c on the CMB grid `bbox_cmb` (MAP only).
      bbox_cmb: CMB bbox (global pixel indices).
      bbox_atm: atmosphere bbox (global pixel indices; must contain all back-advected corners).
      obs_pix_cmb: (n_obs,) indices into full CMB grid used as the shared observed basis.
      global_to_obs_cmb: (n_pix_cmb,) mapping full->obs, -1 for unobserved.
      estimator_mode: 'ML' or 'MAP'.
      n_scans: total scans intended for synthesis; MAP uses (1/n_scans) C_c^{-1} per scan.
      cl_floor_mk2: small positive floor used only for numerical stabilization of diagonals.
      cg_tol, cg_maxiter: CG controls for the joint solve.

    Returns:
      ScanSolve.
    """
    mode = str(estimator_mode).upper()
    if mode not in ("ML", "MAP"):
        raise ValueError("estimator_mode must be 'ML' or 'MAP'.")
    if int(n_scans) <= 0:
        raise ValueError("n_scans must be positive.")

    tod_mk = np.asarray(tod_mk)
    pix_index = np.asarray(pix_index, dtype=np.int64)
    t_s = np.asarray(t_s, dtype=np.float64)
    if pix_index.shape[:2] != tod_mk.shape or pix_index.shape[-1] != 2:
        raise ValueError("pix_index must have shape (n_t,n_det,2) matching tod_mk.")
    if t_s.shape != (tod_mk.shape[0],):
        raise ValueError("t_s must have shape (n_t,).")

    # Validate prior/grid sizes vs bboxes.
    n_pix_cmb = int(bbox_cmb.nx * bbox_cmb.ny)
    if int(prior_cmb.nx * prior_cmb.ny) != n_pix_cmb:
        raise ValueError("prior_cmb grid size must match bbox_cmb.")
    n_pix_atm = int(bbox_atm.nx * bbox_atm.ny)
    if int(prior_atm.nx * prior_atm.ny) != n_pix_atm:
        raise ValueError("prior_atm grid size must match bbox_atm.")

    # Build pointing and valid mask on the CMB grid.
    pm, vm = pointing_from_pix_index(pix_index=pix_index, tod_mk=tod_mk, bbox=bbox_cmb)

    # Restrict to the shared observed-pixel set: drop samples whose pix is not in obs set.
    valid_pix = pm[vm].astype(np.int64, copy=False)  # (n_valid_all,)
    pix_obs_local_all = np.asarray(global_to_obs_cmb, dtype=np.int64)[valid_pix]
    keep = pix_obs_local_all >= 0
    if not bool(np.all(keep)):
        vm2 = vm.copy()
        vm2[vm] = keep
        vm = vm2
        valid_pix = pm[vm].astype(np.int64, copy=False)
        pix_obs_local_all = np.asarray(global_to_obs_cmb, dtype=np.int64)[valid_pix]
    pix_obs_local = pix_obs_local_all.astype(np.int64, copy=False)  # (n_valid,)

    n_obs = int(np.asarray(obs_pix_cmb, dtype=np.int64).size)
    hits = np.bincount(pix_obs_local, minlength=n_obs).astype(np.float64, copy=False)

    # Per-sample inverse variances (diagonal N).
    n_det = int(tod_mk.shape[1])
    eff_sigma = np.asarray(noise_std_det_mk, dtype=np.float64).reshape(-1)
    if eff_sigma.shape != (n_det,):
        raise ValueError("noise_std_det_mk must have shape (n_det,).")
    if not bool(np.all(np.isfinite(eff_sigma))) or not bool(np.all(eff_sigma > 0)):
        raise ValueError("noise_std_det_mk must be finite and > 0.")

    _, det_idx = np.where(vm)
    det_idx = det_idx.astype(np.int64, copy=False)
    sigma_samp = eff_sigma[det_idx]
    inv_var = 1.0 / (sigma_samp * sigma_samp)  # (n_valid,)

    # Dense TOD vector of valid samples.
    d = np.asarray(tod_mk, dtype=np.float64)[vm].reshape(-1)  # (n_valid,)

    # Atmosphere advection operator W on bbox_atm (open boundary).
    idx4, w4 = frozen_screen_bilinear_weights(
        pointing_matrix=pm,
        valid_mask=vm,
        bbox_cmb=bbox_cmb,
        bbox_atm=bbox_atm,
        wind_deg_per_s=(float(wind_deg_per_s[0]), float(wind_deg_per_s[1])),
        t_s=t_s,
        pixel_size_deg=float(pixel_size_deg),
        strict=True,
    )
    n_valid = int(idx4.shape[0])
    if n_valid != int(np.sum(vm)):
        raise RuntimeError("Internal error: idx4 length must equal number of valid samples.")

    def W_apply(a0: np.ndarray) -> np.ndarray:
        a = np.asarray(a0, dtype=np.float64).reshape(-1)
        if a.size != n_pix_atm:
            raise ValueError("W_apply got wrong shape for a0.")
        return np.sum(w4 * a[idx4], axis=1)

    def WT_apply(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        out = np.zeros((n_pix_atm,), dtype=np.float64)
        np.add.at(out, idx4.reshape(-1), (w4 * xx[:, None]).reshape(-1))
        return out

    # Choose CMB unknown parameterization.
    if mode == "MAP":
        if n_obs != n_pix_cmb:
            raise ValueError("MAP requires full-grid solve on bbox_cmb (n_obs == n_pix_cmb).")
        idx_c = np.arange(n_obs, dtype=np.int64)
        obs_to_act = np.arange(n_obs, dtype=np.int64)
        act_idx = pix_obs_local  # (n_valid,) values in [0,n_pix_cmb)
        n_c = int(n_obs)
    else:
        idx_c = np.nonzero(hits > 0)[0].astype(np.int64)
        n_c = int(idx_c.size)
        obs_to_act = -np.ones((n_obs,), dtype=np.int64)
        obs_to_act[idx_c] = np.arange(n_c, dtype=np.int64)
        act_idx = obs_to_act[pix_obs_local]
        if not bool(np.all(act_idx >= 0)):
            raise RuntimeError("Internal error: act_idx must be non-negative for all valid samples.")

    ninv_d = inv_var * d  # (n_valid,)
    rhs_c_act = np.bincount(act_idx, weights=ninv_d, minlength=n_c).astype(np.float64, copy=False)  # (n_c,)
    rhs_a = WT_apply(ninv_d)  # (n_pix_atm,)

    # Store RHS on the full observed basis for synthesis.
    rhs_c_full = np.bincount(pix_obs_local, weights=ninv_d, minlength=n_obs).astype(np.float64, copy=False)  # (n_obs,)

    # Preconditioner diagonals.
    diag_c = np.bincount(act_idx, weights=inv_var, minlength=n_c).astype(np.float64, copy=False)
    diag_c = np.maximum(diag_c, float(cl_floor_mk2))

    diag_WtNW = np.bincount(
        idx4.reshape(-1),
        weights=(w4 * w4 * inv_var[:, None]).reshape(-1),
        minlength=n_pix_atm,
    ).astype(np.float64, copy=False)

    e0_atm = np.zeros((n_pix_atm,), dtype=np.float64)
    e0_atm[0] = 1.0
    diag_Ca_inv = float(prior_atm.apply_Cinv(e0_atm)[0])
    if not np.isfinite(diag_Ca_inv) or diag_Ca_inv <= 0:
        raise ValueError("Non-positive diagonal estimate for C_a^{-1}; check cl_floor_mk2 and cl bins.")
    diag_a = np.maximum(diag_WtNW + diag_Ca_inv, float(cl_floor_mk2))

    if mode == "MAP":
        e0_cmb = np.zeros((n_pix_cmb,), dtype=np.float64)
        e0_cmb[0] = 1.0
        diag_Cc_inv = float(prior_cmb.apply_Cinv(e0_cmb)[0])
        if np.isfinite(diag_Cc_inv) and diag_Cc_inv > 0:
            diag_c = diag_c + (1.0 / float(n_scans)) * diag_Cc_inv

    # Joint operator A * [c_act; a0_atm].
    def A_matvec(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        c_act = xx[:n_c]
        a0 = xx[n_c:]
        if a0.size != n_pix_atm:
            raise ValueError("A_matvec got wrong shape.")

        Pc = c_act[act_idx]  # (n_valid,) = (P c) on valid TOD samples
        Wa = W_apply(a0)  # (n_valid,) = (W a0) on valid TOD samples
        u = inv_var * (Pc + Wa)  # (n_valid,) = N^{-1}(P c + W a0) with diagonal N

        out_c = np.bincount(act_idx, weights=u, minlength=n_c).astype(np.float64, copy=False)  # (n_c,) = P^T u
        if mode == "MAP":
            out_c = out_c + (1.0 / float(n_scans)) * prior_cmb.apply_Cinv(c_act)  # (n_c,) += (1/S) Cc^{-1} c
        out_a = WT_apply(u) + prior_atm.apply_Cinv(a0)  # (n_pix_atm,) = W^T u + Ca^{-1} a0
        return np.concatenate([out_c, out_a], axis=0)  # (n_c + n_pix_atm,)

    # Matrix-free operator A: R^(n_c+n_pix_atm) -> R^(n_c+n_pix_atm)
    A_op = spla.LinearOperator((n_c + n_pix_atm, n_c + n_pix_atm), matvec=A_matvec, dtype=np.float64)

    def Pinv_matvec(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)  # (n_c + n_pix_atm,)
        xc = xx[:n_c] / diag_c  # (n_c,) elementwise divide by diag(A_cc) approximation
        xa = xx[n_c:] / diag_a  # (n_pix_atm,) elementwise divide by diag(A_aa) approximation
        return np.concatenate([xc, xa], axis=0)  # (n_c + n_pix_atm,)

    # Diagonal preconditioner M ≈ A^{-1} applied as x -> [x_c/diag_c; x_a/diag_a].
    P_pre = spla.LinearOperator(A_op.shape, matvec=Pinv_matvec, dtype=np.float64)

    rhs = np.concatenate([rhs_c_act, rhs_a], axis=0)  # (n_c + n_pix_atm,) = [P^T N^{-1} d; W^T N^{-1} d]
    sol, info = spla.cg(A_op, rhs, M=P_pre, atol=0.0, rtol=float(cg_tol), maxiter=int(cg_maxiter))
    if info != 0:
        raise RuntimeError(f"Joint CG did not converge (info={info}). Increase cg_maxiter or adjust preconditioning.")
    sol = np.asarray(sol, dtype=np.float64).reshape(-1)

    c_act = sol[:n_c].copy()
    c_act -= float(np.mean(c_act))  # monopole: remove mean on the solved subspace

    c_obs = np.zeros((n_obs,), dtype=np.float64)
    c_obs[idx_c] = c_act
    c_hat_full = np.zeros((n_pix_cmb,), dtype=np.float64)
    c_hat_full[np.asarray(obs_pix_cmb, dtype=np.int64)] = c_obs

    return ScanSolve(
        c_hat_full_mk=c_hat_full,
        bbox_cmb=bbox_cmb,
        bbox_atm=bbox_atm,
        obs_pix_cmb=np.asarray(obs_pix_cmb, dtype=np.int64),
        estimator_mode=mode,
        n_scans=int(n_scans),
        pix_obs_local=pix_obs_local,
        idx4=idx4,
        w4=w4,
        inv_var=inv_var,
        tod_valid_mk=d,
        rhs_c=rhs_c_full,
        rhs_a=rhs_a,
        prior_atm=prior_atm,
        prior_cmb=prior_cmb,
    )
