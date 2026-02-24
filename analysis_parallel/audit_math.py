"""
Equation-to-code mapping and correctness-sensitive points for the parallel ML pipeline.

Theory reference: latex_cmb_atmosphere/main.tex.
Code references: cad/src/cad/reconstruct_scan.py, cad/analysis/run_reconstruction.py,
  notebooks/notes_realistic_wind_examples/util_inference.py (information-form style).

--- EQUATIONS (ML, per scan s) ---

Forward model:
  d_s = P_s c + W_s a_s^0 + n_s,   n_s ~ N(0, N_s),  N_s = diag(sigma_i^2).

Atmosphere prior:
  a_s^0 ~ N(0, C_a).

Marginalized TOD covariance:
  Ntilde_s = N_s + W_s C_a W_s^T.

Woodbury:
  M_s = C_a^{-1} + W_s^T N_s^{-1} W_s.
  Ntilde_s^{-1} = N_s^{-1} - N_s^{-1} W_s M_s^{-1} W_s^T N_s^{-1}.

Information form (observed-pixel restriction):
  F_s = P_s^T Ntilde_s^{-1} P_s.   (Fisher / precision on observed pixels)
  b_s = P_s^T Ntilde_s^{-1} d_s.

Per-scan ML solve:
  F_s c_s = b_s  =>  c_hat_s = F_s^{-1} b_s  (with gauge: mean-zero on obs pixels).

Global synthesis:
  F_tot = sum_s S_s^T F_s S_s,
  b_tot = sum_s S_s^T b_s.
  F_tot c_hat = b_tot.

S_s maps scan-local observed pixel indices to global observed indices. Pixels not
observed in scan s contribute zero to F_s and b_s from that scan.

--- CODE MAPPING ---

cad/reconstruct_scan.solve_single_scan:
  - Implements joint (c, a0) normal equations; matrix-free A_matvec = [P^T N^{-1} P c + ..., W^T N^{-1} P c + W^T N^{-1} W a0 + C_a^{-1} a0].
  - rhs_c_full = P^T N^{-1} d (not P^T Ntilde^{-1} d); rhs_a = W^T N^{-1} d.
  - So current cad path does NOT form F_s or b_s; it solves the joint system by CG.

util_inference.solve_cmb_single_scan:
  - Builds M = C_a^{-1} + W^T N^{-1} W (dense), chol_M = cholesky(M).
  - u = M^{-1} (W^T N^{-1} d); y = N^{-1} d - N^{-1} W u = Ntilde^{-1} d.
  - F = P^T Ntilde^{-1} P = inv_noise2*diag(hits) - inv_noise2*(PW @ U) with U = M^{-1}(W^T P).
  - b = P^T y.
  - c_obs = solve(F, b); gauge: c_obs -= c_obs.mean().

util_inference.synthesize_scans:
  - F = sum_s F_s, b = sum_s b_s; c_obs = solve(F, b); c_obs -= c_obs.mean().

--- CORRECTNESS-SENSITIVE POINTS ---

1. Noise: N_s is diagonal; per-sample inv_var = 1/sigma_i^2. cad uses per-detector
   noise (noise_std_det_mk); each TOD sample gets sigma from its detector index.

2. P and W: P is (n_valid, n_obs) one-hot: row i has 1 at column pix_obs_local[i].
   W is (n_valid, n_pix_atm) with 4 nonzeros per row from bilinear weights (idx4, w4).
   util.frozen_screen_bilinear_weights and util.pointing_from_pix_index must use
   the same global bbox and t0 = t_s[0] convention.

3. Global observed index set: For synthesis, all scans must use the same strict
   global pixel ordering. obs_pix_global_scan for each scan are indices into the
   same global CMB grid (bbox_cmb union). Scatter S_s^T: scan's obs index j
   corresponds to global obs index global_index(obs_pix_global_scan[j]).

4. C_a^{-1}: FFT-diagonal prior; apply_Cinv applies in Fourier space. For dense M_s
   we need columns of C_a^{-1} or matrix-free M_s matvec (C_a^{-1} v + W^T N^{-1} W v).

5. Gauge: ML has a null direction (monopole). Both util_inference and cad remove
   the mean of c_obs after solve. Synthesis must apply the same gauge once.

6. Symmetrization: F_s can be numerically asymmetric; use 0.5*(F_s + F_s.T) before
   save and before global solve.
"""
