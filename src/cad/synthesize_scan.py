"""
Multi-scan CMB reconstruction via a joint linear solve.

We solve for a shared CMB map c and per-scan atmosphere reference screens {a0_s}:

  d_s = P_s c + W_s a0_s + n_s,    n_s ~ N(0, N_s)

with Gaussian stationary priors on each a0_s (and optionally on c for MAP).

This is equivalent to the theory's marginal solution for c (Woodbury / Schur
complement), but avoids nested solves by working on the augmented system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse.linalg as spla

from .map import BBox, bbox_union, scan_bbox_from_pix_index
from .prior import FourierGaussianPrior
from .util import (
    bbox_pad_for_open_boundary,
    frozen_screen_bilinear_weights,
    observed_pixel_index_set,
    pointing_from_pix_index,
)


@dataclass(frozen=True)
class MultiScanSolve:
    """
    Output of the multi-scan solve.

    Fields:
      c_hat_full_mk: (n_pix_cmb,) CMB map on the full CMB bbox grid (mK).
      bbox_cmb: bbox of the CMB grid (global indices).
      bbox_atm: bbox of the atmosphere grid (global indices; padded).
      obs_pix_cmb: (n_obs,) indices into the CMB full grid used as the solve basis.
      estimator_mode: 'ML' or 'MAP'.
      n_scans: number of scans.
    """

    c_hat_full_mk: np.ndarray
    bbox_cmb: BBox
    bbox_atm: BBox
    obs_pix_cmb: np.ndarray
    estimator_mode: str
    n_scans: int

    
def synthesize_scans(
    *,
    scans_tod_mk: list[np.ndarray],
    scans_pix_index: list[np.ndarray],
    scans_t_s: list[np.ndarray],
    winds_deg_per_s: list[tuple[float, float]],
    scans_noise_std_det_mk: list[np.ndarray],
    pixel_size_deg: float,
    cl_atm_bins_mk2: np.ndarray,
    cl_cmb_bins_mk2: np.ndarray | None = None,
    estimator_mode: Literal["ML", "MAP"] = "ML",
    cl_floor_mk2: float = 1e-12,
    min_hits_per_pix: int = 1,
    cg_tol: float = 1e-3,
    cg_maxiter: int = 800,
) -> MultiScanSolve:
    """
    Solve the joint multi-scan problem for a shared CMB map.

    Inputs are deliberately simple: provide per-scan TOD + per-sample pixel indices.

    Unknowns:

      x = [ c ; a0_0 ; a0_1 ; ... ; a0_{S-1} ]

    Normal equations (block form):

      A_cc = Σ_s P_sᵀ N_s⁻¹ P_s   + I_MAP * Cc⁻¹
      A_cas =      P_sᵀ N_s⁻¹ W_s
      A_asc =      W_sᵀ N_s⁻¹ P_s
      A_asas =     W_sᵀ N_s⁻¹ W_s + Ca⁻¹

      b_c  = Σ_s P_sᵀ N_s⁻¹ d_s
      b_as =      W_sᵀ N_s⁻¹ d_s

    where:
      - P_s selects one CMB pixel per valid sample on `bbox_cmb`
      - W_s is the frozen-screen bilinear operator on `bbox_atm`, using
        x_src(t) = i_x - (w_x / pixel_size_deg) * (t - t0) and
        y_src(t) = i_y - (w_y / pixel_size_deg) * (t - t0) with t0 = t_s[0]
      - Ca and Cc are stationary FFT-diagonal priors on their respective grids.

    Args:
      scans_tod_mk: list of (n_t, n_det) TOD arrays in mK.
      scans_pix_index: list of (n_t, n_det, 2) int64 global (ix,iy) indices.
      scans_t_s: list of (n_t,) time stamps in seconds.
      winds_deg_per_s: list of (w_x,w_y) deg/s per scan in the RA/Dec-degree basis.
      scans_noise_std_det_mk: list of (n_det,) per-detector noise std in mK.
      pixel_size_deg: scalar pixel size in degrees (shared across scans).
      cl_atm_bins_mk2: (n_ell_bins,) atmospheric C_ell bins in mK^2.
      cl_cmb_bins_mk2: (n_ell_bins,) CMB C_ell bins in mK^2 (required for MAP; optional for ML).
      estimator_mode: 'ML' or 'MAP'.
      cl_floor_mk2: floor for numerical stability / positivity.
      min_hits_per_pix: observed-pixel threshold on the CMB grid (ML only).
      cg_tol, cg_maxiter: CG controls for the joint solve.

    Returns:
      MultiScanSolve with c_hat_full_mk on the CMB bbox grid.
    """
    mode = str(estimator_mode).upper()
    if mode not in ("ML", "MAP"):
        raise ValueError("estimator_mode must be 'ML' or 'MAP'.")

    if not scans_tod_mk:
        raise ValueError("Must provide at least one scan.")
    n_scans = int(len(scans_tod_mk))
    if not (len(scans_pix_index) == n_scans == len(scans_t_s) == len(winds_deg_per_s) == len(scans_noise_std_det_mk)):
        raise ValueError("All per-scan input lists must have the same length.")

    # --- CMB bbox on the shared global pixel grid ---
    boxes: list[BBox] = []
    for tod_mk, pix_index in zip(scans_tod_mk, scans_pix_index, strict=True):
        tod_mk = np.asarray(tod_mk)
        pix_index = np.asarray(pix_index, dtype=np.int64)
        valid = np.isfinite(tod_mk)
        boxes.append(scan_bbox_from_pix_index(pix_index=pix_index, valid_mask=valid))
    bbox_cmb = bbox_union(boxes)

    # --- atmosphere bbox padding for open-boundary advection ---
    bbox_atm = bbox_pad_for_open_boundary(
        bbox_obs=bbox_cmb,
        scans_pix_index=scans_pix_index,
        scans_tod_mk=scans_tod_mk,
        scans_t_s=scans_t_s,
        winds_deg_per_s=winds_deg_per_s,
        pixel_size_deg=float(pixel_size_deg),
    )

    nx_c, ny_c = int(bbox_cmb.nx), int(bbox_cmb.ny)
    nx_a, ny_a = int(bbox_atm.nx), int(bbox_atm.ny)
    n_pix_cmb = int(nx_c * ny_c)
    n_pix_atm = int(nx_a * ny_a)
    pixel_res_rad = float(pixel_size_deg) * np.pi / 180.0

    # Calculate cos_dec from bbox_cmb center
    iy_mid = (float(bbox_cmb.iy0) + float(bbox_cmb.iy1)) / 2.0
    dec_deg = iy_mid * float(pixel_size_deg)
    cos_dec = float(np.cos(np.deg2rad(dec_deg)))
    if cos_dec <= 0.1:
         cos_dec = 0.1

    cl_atm_bins_mk2 = np.asarray(cl_atm_bins_mk2, dtype=np.float64).reshape(-1)
    if int(cl_atm_bins_mk2.size) <= 0:
        raise ValueError("cl_atm_bins_mk2 must be non-empty.")
    n_ell_bins = int(cl_atm_bins_mk2.size)
    prior_atm = FourierGaussianPrior(
        nx=nx_a,
        ny=ny_a,
        pixel_res_rad=pixel_res_rad,
        cl_bins_mk2=cl_atm_bins_mk2,
        cos_dec=cos_dec,
        cl_floor_mk2=float(cl_floor_mk2),
    )

    if mode == "MAP":
        if cl_cmb_bins_mk2 is None:
            raise ValueError("MAP requires cl_cmb_bins_mk2.")
        cl_cmb_bins_mk2 = np.asarray(cl_cmb_bins_mk2, dtype=np.float64).reshape(-1)
        if cl_cmb_bins_mk2.size != int(n_ell_bins):
            raise ValueError("cl_cmb_bins_mk2 must have the same length as cl_atm_bins_mk2.")
    else:
        # ML does not apply the CMB prior in the objective, but we still build a placeholder prior for symmetry.
        if cl_cmb_bins_mk2 is None:
            cl_cmb_bins_mk2 = np.full((int(n_ell_bins),), float(cl_floor_mk2), dtype=np.float64)
        else:
            cl_cmb_bins_mk2 = np.asarray(cl_cmb_bins_mk2, dtype=np.float64).reshape(-1)
            if cl_cmb_bins_mk2.size != int(n_ell_bins):
                raise ValueError("cl_cmb_bins_mk2 must have the same length as cl_atm_bins_mk2.")

    prior_cmb = FourierGaussianPrior(
        nx=nx_c,
        ny=ny_c,
        pixel_res_rad=pixel_res_rad,
        cl_bins_mk2=np.asarray(cl_cmb_bins_mk2, dtype=np.float64),
        cos_dec=cos_dec,
        cl_floor_mk2=float(cl_floor_mk2),
    )

    # --- pointing matrices per scan on the CMB bbox ---
    pointing_mats: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    for tod_mk, pix_index in zip(scans_tod_mk, scans_pix_index, strict=True):
        pm, vm = pointing_from_pix_index(pix_index=pix_index, tod_mk=tod_mk, bbox=bbox_cmb)
        pointing_mats.append(pm)
        valid_masks.append(vm)

    # --- observed pixel index set (union across scans) ---
    obs_pix_hit, global_to_obs_hit = observed_pixel_index_set(
        pointing_matrices=pointing_mats,
        valid_masks=valid_masks,
        n_pix=n_pix_cmb,
        min_hits_per_pix=int(min_hits_per_pix),
    )
    if mode == "MAP":
        obs_pix_cmb = np.arange(n_pix_cmb, dtype=np.int64)
        global_to_obs_cmb = np.arange(n_pix_cmb, dtype=np.int64)
    else:
        obs_pix_cmb, global_to_obs_cmb = obs_pix_hit, global_to_obs_hit
    n_obs = int(obs_pix_cmb.size)

    # --- per-scan operator state ---
    pix_obs_locals: list[np.ndarray] = []
    idx4_list: list[np.ndarray] = []
    w4_list: list[np.ndarray] = []
    inv_var_list: list[np.ndarray] = []
    rhs_c_list: list[np.ndarray] = []
    rhs_a_list: list[np.ndarray] = []

    for tod_mk, pix_index, t_s, wind, noise_std_det, pm0, vm0 in zip(
        scans_tod_mk,
        scans_pix_index,
        scans_t_s,
        winds_deg_per_s,
        scans_noise_std_det_mk,
        pointing_mats,
        valid_masks,
        strict=True,
    ):
        pm = np.asarray(pm0, dtype=np.int64)
        vm = np.asarray(vm0, dtype=bool)

        # Drop samples outside the shared obs set.
        valid_pix = pm[vm].astype(np.int64, copy=False)
        pix_obs_local_all = np.asarray(global_to_obs_cmb, dtype=np.int64)[valid_pix]
        keep = pix_obs_local_all >= 0
        if not bool(np.all(keep)):
            vm2 = vm.copy()
            vm2[vm] = keep
            vm = vm2
            valid_pix = pm[vm].astype(np.int64, copy=False)
            pix_obs_local_all = np.asarray(global_to_obs_cmb, dtype=np.int64)[valid_pix]
        pix_obs_local = pix_obs_local_all.astype(np.int64, copy=False)  # (n_valid,)

        # Sample weights.
        tod_mk = np.asarray(tod_mk)
        noise_std_det = np.asarray(noise_std_det, dtype=np.float64).reshape(-1)
        if noise_std_det.shape != (tod_mk.shape[1],):
            raise ValueError("Each scans_noise_std_det_mk entry must have shape (n_det,).")
        if not bool(np.all(np.isfinite(noise_std_det))) or not bool(np.all(noise_std_det > 0)):
            raise ValueError("noise stds must be finite and > 0.")

        _, det_idx = np.where(vm)
        det_idx = det_idx.astype(np.int64, copy=False)
        sigma_samp = noise_std_det[det_idx]
        inv_var = 1.0 / (sigma_samp * sigma_samp)  # (n_valid,)

        d = np.asarray(tod_mk, dtype=np.float64)[vm].reshape(-1)  # (n_valid,)
        ninv_d = inv_var * d

        # Atmosphere advection indices on bbox_atm.
        t_s = np.asarray(t_s, dtype=np.float64)
        idx4, w4 = frozen_screen_bilinear_weights(
            pointing_matrix=pm,
            valid_mask=vm,
            bbox_cmb=bbox_cmb,
            bbox_atm=bbox_atm,
            wind_deg_per_s=(float(wind[0]), float(wind[1])),
            t_s=t_s,
            pixel_size_deg=float(pixel_size_deg),
            strict=True,
        )

        # RHS blocks.
        rhs_c = np.bincount(pix_obs_local, weights=ninv_d, minlength=n_obs).astype(np.float64, copy=False)  # (n_obs,)
        rhs_a = np.zeros((n_pix_atm,), dtype=np.float64)
        np.add.at(rhs_a, idx4.reshape(-1), (w4 * ninv_d[:, None]).reshape(-1))

        pix_obs_locals.append(pix_obs_local)
        idx4_list.append(idx4)
        w4_list.append(w4)
        inv_var_list.append(inv_var)
        rhs_c_list.append(rhs_c)
        rhs_a_list.append(rhs_a)

    # Combined RHS.
    rhs_c = np.sum(rhs_c_list, axis=0)
    rhs_a = np.concatenate(rhs_a_list, axis=0)  # (n_scans*n_pix_atm,)
    rhs = np.concatenate([rhs_c, rhs_a], axis=0)

    # Preconditioner diagonals.
    diag_c = np.zeros((n_obs,), dtype=np.float64)
    diag_a_blocks: list[np.ndarray] = []

    e0_atm = np.zeros((n_pix_atm,), dtype=np.float64)
    e0_atm[0] = 1.0
    diag_Ca_inv = float(prior_atm.apply_Cinv(e0_atm)[0])
    if not np.isfinite(diag_Ca_inv) or diag_Ca_inv <= 0:
        raise ValueError("Non-positive diagonal estimate for C_a^{-1}.")

    diag_Cc_inv = 0.0
    if mode == "MAP":
        e0_cmb = np.zeros((n_pix_cmb,), dtype=np.float64)
        e0_cmb[0] = 1.0
        diag_Cc_inv = float(prior_cmb.apply_Cinv(e0_cmb)[0])

    for pix_obs_local, idx4, w4, inv_var in zip(pix_obs_locals, idx4_list, w4_list, inv_var_list, strict=True):
        diag_c += np.bincount(pix_obs_local, weights=inv_var, minlength=n_obs).astype(np.float64, copy=False)
        diag_WtNW = np.bincount(
            idx4.reshape(-1),
            weights=(w4 * w4 * inv_var[:, None]).reshape(-1),
            minlength=n_pix_atm,
        ).astype(np.float64, copy=False)
        diag_a_blocks.append(np.maximum(diag_WtNW + diag_Ca_inv, float(cl_floor_mk2)))

    if mode == "MAP":
        if n_obs != n_pix_cmb:
            raise ValueError("MAP combined solve requires full-grid CMB solve (n_obs == n_pix_cmb).")
        if np.isfinite(diag_Cc_inv) and diag_Cc_inv > 0:
            diag_c = diag_c + diag_Cc_inv

    diag_c = np.maximum(diag_c, float(cl_floor_mk2))
    diag_a = np.concatenate(diag_a_blocks, axis=0)

    # Combined operator A * [c; a0_1; ...; a0_S].
    def A_matvec(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        c = xx[:n_obs]
        a_all = xx[n_obs:]
        if a_all.size != n_scans * n_pix_atm:
            raise ValueError("A_matvec got wrong shape.")
        a_blocks = a_all.reshape(n_scans, n_pix_atm)

        out_c = np.zeros((n_obs,), dtype=np.float64)
        out_a_blocks: list[np.ndarray] = []

        for si in range(n_scans):
            pix_obs_local = pix_obs_locals[si]
            idx4 = idx4_list[si]
            w4 = w4_list[si]
            inv_var = inv_var_list[si]
            a0 = a_blocks[si]

            Pc = c[pix_obs_local]
            Wa = np.sum(w4 * a0[idx4], axis=1)
            u = inv_var * (Pc + Wa)

            out_c += np.bincount(pix_obs_local, weights=u, minlength=n_obs).astype(np.float64, copy=False)

            out_a = np.zeros((n_pix_atm,), dtype=np.float64)
            np.add.at(out_a, idx4.reshape(-1), (w4 * u[:, None]).reshape(-1))
            out_a = out_a + prior_atm.apply_Cinv(a0)
            out_a_blocks.append(out_a)

        if mode == "MAP":
            out_c = out_c + prior_cmb.apply_Cinv(c)

        return np.concatenate([out_c, np.concatenate(out_a_blocks, axis=0)], axis=0)

    A_op = spla.LinearOperator((n_obs + n_scans * n_pix_atm, n_obs + n_scans * n_pix_atm), matvec=A_matvec, dtype=np.float64)

    def Pinv_matvec(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        xc = xx[:n_obs] / diag_c
        xa = xx[n_obs:] / diag_a
        return np.concatenate([xc, xa], axis=0)

    P_pre = spla.LinearOperator(A_op.shape, matvec=Pinv_matvec, dtype=np.float64)

    sol, info = spla.cg(A_op, rhs, M=P_pre, atol=0.0, rtol=float(cg_tol), maxiter=int(cg_maxiter))
    if info != 0:
        raise RuntimeError(f"Combined joint CG did not converge (info={info}).")
    sol = np.asarray(sol, dtype=np.float64).reshape(-1)

    c = sol[:n_obs].copy()
    c -= float(np.mean(c))  # gauge: remove monopole

    c_hat_full = np.zeros((n_pix_cmb,), dtype=np.float64)
    c_hat_full[np.asarray(obs_pix_cmb, dtype=np.int64)] = c

    return MultiScanSolve(
        c_hat_full_mk=c_hat_full,
        bbox_cmb=bbox_cmb,
        bbox_atm=bbox_atm,
        obs_pix_cmb=np.asarray(obs_pix_cmb, dtype=np.int64),
        estimator_mode=mode,
        n_scans=int(n_scans),
    )

