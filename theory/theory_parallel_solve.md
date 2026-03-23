# Parallel ML Solve: Theory-to-Code Mapping

This note documents the current parallel pipeline implemented in:

- `cad/analysis_parallel/run_reconstruction.py`
- `cad/analysis_parallel/run_synthesis_full.py`
- `cad/analysis_parallel/run_synthesis_margined.py`
- `cad/src/cad/parallel_solve/reconstruct_scan.py`
- `cad/src/cad/parallel_solve/fisher.py`
- `cad/src/cad/parallel_solve/synthesize_scan.py`

Reference exact joint (augmented) solve remains available in:

- `cad/src/cad/direct_solve/reconstruct_scan.py` (single scan)
- `cad/src/cad/direct_solve/synthesize_scan.py` (all scans)

This document covers ML only (`estimator_mode="ML"`), consistent with the current parallel path.

---

## 1) Symbols, index sets, and array conventions

### 1.1 Indices and dimensions

- $s \in \{1,\dots,S\}$: scan index
- $i \in \{1,\dots,n_{\mathrm{valid},s}\}$: valid TOD sample index in scan $s$
- $p \in \{1,\dots,n_{\mathrm{pix,cmb}}\}$: CMB pixel index on the CMB bbox grid
- $q \in \{1,\dots,n_{\mathrm{pix,atm},s}\}$: atmosphere pixel index on scan-$s$ atmosphere grid

### 1.2 Core variables

For each scan $s$:

- $d_s \in \mathbb{R}^{n_{\mathrm{valid},s}}$: valid TOD
- $c \in \mathbb{R}^{n_{\mathrm{pix,cmb}}}$: static sky map
- $a_s^0 \in \mathbb{R}^{n_{\mathrm{pix,atm},s}}$: atmosphere screen at reference time
- $P_s$: CMB pointing operator (CMB pixels $\to$ valid TOD)
- $W_s$: frozen-flow advection/interpolation operator (atmosphere pixels $\to$ valid TOD)
- $N_s = \mathrm{diag}(\sigma_i^2)$: diagonal TOD noise covariance
- $C_a$: atmosphere prior covariance

Derived:

- $\tilde N_s = N_s + W_s C_a W_s^\top$
- $\mathrm{Cov}(\hat c_s)^{-1} = P_s^\top \tilde N_s^{-1} P_s$ (per-scan precision on observed CMB pixels)
- $P_s^\top \tilde N_s^{-1} d_s$ (per-scan information vector; stored as `Pt_Ninv_d`)

### 1.3 Flattening convention used in code

For a bbox with shape $(n_x, n_y)$ and local integer coordinates $(i_x, i_y)$:

$$
\mathrm{pix} = i_y + i_x n_y.
$$

This convention is used consistently in:

- layout and pointing (`obs_pix_global`, `global_to_obs`)
- synthesis inputs/outputs
- plotting reshapes (`img_from_vec`, `naive_2d.T.ravel()`)

### 1.4 Main stored arrays

- `obs_pix_global`: $(n_{\mathrm{obs}},)$ observed CMB pixel indices in global flat indexing
- `global_to_obs`: $(n_{\mathrm{pix,cmb}},)$ map from global pixel to observed index or `-1`
- `c_hat_scan_obs`: $(n_{\mathrm{obs},s},)$ per-scan point estimate
- `cov_inv`: $(n_{\mathrm{obs},s}, n_{\mathrm{obs},s}) = \mathrm{Cov}(\hat c_s)^{-1}$
- `Pt_Ninv_d`: $(n_{\mathrm{obs},s},) = P_s^\top \tilde N_s^{-1} d_s$
- `c_hat_obs`: $(n_{\mathrm{obs}},)$ synthesized ML map on observed set
- `cov_inv_tot`: $(n_{\mathrm{obs}}, n_{\mathrm{obs}})$ global precision (sum of remapped per-scan precisions)

---

## 2) Per-scan model and ML equations

For each scan $s$:

$$
d_s = P_s c + W_s a_s^0 + n_s,\qquad n_s \sim \mathcal N(0, N_s),
$$
$$
a_s^0 \sim \mathcal N(0, C_a).
$$

Define the per-scan objective (negative log posterior up to constants):

$$
\Phi_s(c,a_s^0)
=\frac12(d_s-P_sc-W_sa_s^0)^\top N_s^{-1}(d_s-P_sc-W_sa_s^0)
+\frac12 (a_s^0)^\top C_a^{-1} a_s^0.
$$

Stationarity gives the augmented block normal equation:

$$
\begin{bmatrix}
P_s^\top N_s^{-1}P_s & P_s^\top N_s^{-1}W_s \\
W_s^\top N_s^{-1}P_s & W_s^\top N_s^{-1}W_s + C_a^{-1}
\end{bmatrix}
\begin{bmatrix}
c \\ a_s^0
\end{bmatrix}
=
\begin{bmatrix}
P_s^\top N_s^{-1}d_s \\
W_s^\top N_s^{-1}d_s
\end{bmatrix}.
$$

Eliminating $a_s^0$ (Schur complement) yields:

$$
\mathrm{Cov}(\hat c_s)^{-1} c
= P_s^\top \tilde N_s^{-1} d_s,\qquad
\mathrm{Cov}(\hat c_s)^{-1}
= P_s^\top \tilde N_s^{-1} P_s,
$$
$$
\tilde N_s = N_s + W_s C_a W_s^\top.
$$

Hence the per-scan ML estimator is:

$$
\hat c_s
= \left(P_s^\top \tilde N_s^{-1} P_s\right)^{-1}
  P_s^\top \tilde N_s^{-1} d_s,
$$
with covariance
$$
\mathrm{Cov}(\hat c_s)
= \left(P_s^\top \tilde N_s^{-1} P_s\right)^{-1}.
$$

So `cov_inv` in scan artifacts is exactly $\mathrm{Cov}(\hat c_s)^{-1}$, and
`Pt_Ninv_d` is exactly $P_s^\top \tilde N_s^{-1} d_s$.

---

## 3) How `fisher.py` builds per-scan information exactly

The implementation uses Woodbury with
$$
M_s = C_a^{-1} + W_s^\top N_s^{-1}W_s,\qquad
\tilde N_s^{-1}
=N_s^{-1}-N_s^{-1}W_s M_s^{-1} W_s^\top N_s^{-1}.
$$

For any vector $u$ on valid TOD samples:

1. Solve $M_s x = W_s^\top N_s^{-1}u$
2. Return $\tilde N_s^{-1}u = N_s^{-1}u - N_s^{-1}W_s x$

This gives exact actions of $\tilde N_s^{-1}$ without explicitly forming $\tilde N_s$.

To build `Pt_Ninv_d`:

$$
\texttt{Pt\_Ninv\_d} = P_s^\top \tilde N_s^{-1} d_s
$$
via one $M_s$ solve.

To build `cov_inv`:

$$
\texttt{cov\_inv}[:,j] = P_s^\top \tilde N_s^{-1}(P_s e_j),
$$
for each observed pixel basis vector $e_j$; in code this is batched across columns.

Then solve
$$
\texttt{cov\_inv}\,\hat c_s = \texttt{Pt\_Ninv\_d}
$$
for `c_hat_scan_obs` and remove mean (gauge).

Outputs per scan (`scan_XXXX_ml.npz`):

- `obs_pix_global_scan`, `c_hat_scan_obs`
- `cov_inv` ($\mathrm{Cov}(\hat c_s)^{-1}$), `Pt_Ninv_d` ($P_s^\top \tilde N_s^{-1} d_s$)
- scan metadata (`wind_*`, `ell_atm`, `cl_atm_mk2`, etc.)

---

## 4) Global synthesis from per-scan information

### 4.1 Exact global ML equation

Define global observed index space with size $n_{\mathrm{obs}}$.
Each scan contribution is remapped from scan-local observed indices to this global observed index set (the same remap implemented by `global_to_obs` in code).
Then:

$$
\mathrm{Cov}(\hat c)^{-1}
= \sum_{s=1}^S R_s^\top \mathrm{Cov}(\hat c_s)^{-1} R_s,\qquad
\texttt{Pt\_Ninv\_d\_tot}
= \sum_{s=1}^S R_s^\top \left(P_s^\top \tilde N_s^{-1} d_s\right).
$$

Global ML map on observed pixels:

$$
\mathrm{Cov}(\hat c)^{-1}\hat c = \texttt{Pt\_Ninv\_d\_tot}.
$$

This is exactly the marginalized all-scan ML equation:

$$
\left(\sum_s P_s^\top \tilde N_s^{-1} P_s\right)c
=\sum_s P_s^\top \tilde N_s^{-1} d_s,
$$
expressed in the global observed-index basis.

### 4.2 Good-pixel restriction and gauge

Code uses:

- `precision_diag_total = diag(cov_inv_tot)`
- `good_mask = precision_diag_total > 0`

Solve on good subspace $G$:

$$
\left[\mathrm{Cov}(\hat c)^{-1}\right]_{GG}\hat c_G
= \left[\texttt{Pt\_Ninv\_d\_tot}\right]_G,\qquad
\hat c_{\bar G}=0.
$$

Then remove monopole on solved subspace:

$$
\hat c_G \leftarrow \hat c_G - \frac{1}{|G|}\sum_{p\in G}\hat c_p.
$$

Finally embed to full bbox vector:

$$
\hat c_{\mathrm{full}}[\mathrm{obs\_pix\_global}] = \hat c,\qquad
\hat c_{\mathrm{full}}[\text{unobserved}] = 0.
$$

### 4.3 Margined synthesis (`run_synthesis_margined.py`)

Given `margin_frac = f`, define:

$$
m_x = \lfloor f n_x \rfloor,\quad
m_y = \lfloor f n_y \rfloor,\quad
n_x' = n_x - 2m_x,\quad
n_y' = n_y - 2m_y.
$$

For old flat index $p$:

$$
i_x = \left\lfloor \frac{p}{n_y}\right\rfloor,\qquad
i_y = p - i_x n_y.
$$

Keep pixels with
$$
m_x \le i_x < n_x-m_x,\qquad m_y \le i_y < n_y-m_y,
$$
and remap to inner index
$$
p' = (i_y-m_y) + (i_x-m_x)n_y'.
$$

All accumulation, solve, diagnostics, and uncertain-mode estimation run on this inner footprint.

---

## 5) Uncertain eigenmodes in synthesis

Let
$$
\texttt{cov\_inv\_good}
= \left[\mathrm{Cov}(\hat c)^{-1}\right]_{GG}
$$
be the global precision on good pixels ($n_g = |G|$).
Uncertain directions are eigenvectors with smallest eigenvalues:

$$
\texttt{cov\_inv\_good}\, v_i = \lambda_i v_i,\qquad
0 < \lambda_1 \le \lambda_2 \le \cdots,
$$
$$
\mathrm{Var}(\text{mode }i) \approx \lambda_i^{-1}.
$$

So smallest $\lambda_i$ correspond to largest posterior variance.

### 5.1 Lanczos-Ritz algorithm used in code

Inputs:

- target modes: $k = \min(n_{\text{uncertain\_modes}}, n_g)$
- Krylov dimension:
  $$
  m = \min\!\left(n_g,\ \max(k+\text{oversample},2k),\ \text{maxiter}\right),\quad m\ge k
  $$

Procedure:

1. Start normalized random $q_1$.
2. Lanczos recurrence with full re-orthogonalization to build:
   - $Q_m = [q_1,\dots,q_m] \in \mathbb R^{n_g\times m}$
   - tridiagonal $T_m$ from $\alpha_j,\beta_j$
3. Solve small eigenproblem:
   $$
   T_m y_i = \theta_i y_i
   $$
4. Ritz vectors:
   $$
   \tilde v_i = Q_m y_i
   $$
   keep the $k$ smallest $\theta_i$
5. QR orthonormalize $\tilde V$, then refine with Rayleigh quotients:
   $$
   \lambda_i = \tilde v_i^\top \texttt{cov\_inv\_good}\,\tilde v_i
   $$
6. Sort ascending $\lambda_i$, store:
   $$
   \text{uncertain\_mode\_variances}_i = 1/\max(\lambda_i, 10^{-18}),
   $$
   and corresponding vectors in `uncertain_mode_vectors`.

Stored shapes:

- `uncertain_mode_vectors`: $(n_g, k_{\mathrm{stored}})$
- `uncertain_mode_variances`: $(k_{\mathrm{stored}},)$
- `lanczos_n_modes`: requested target
- `n_uncertain_modes_stored`: actual stored mode count

These are used by plotting scripts for projection/deprojection without re-running synthesis.

---

## 6) Combined NPZ outputs from synthesis

Single output file per synthesis run (`recon_combined_ml_full.npz` or `recon_combined_ml_margined.npz`):

- Geometry: `bbox_ix0`, `bbox_iy0`, `nx`, `ny`, `pixel_size_deg`
- Indexing: `obs_pix_global`
- Map estimates: `c_hat_obs`, `c_hat_full_mk`
- Precision diagnostics: `cov_inv_tot`, `precision_diag_total`, `var_diag_total`, `zero_precision_mask`, `good_mask`
- Provenance: `scan_metadata`, `n_scans`, `n_scans_used`, `estimator_mode`
- Uncertain modes: `uncertain_mode_vectors`, `uncertain_mode_variances`, `lanczos_n_modes`, `n_uncertain_modes_stored`

`var_diag_total` is the reciprocal of the precision diagonal where positive; it is a diagonal diagnostic, not the diagonal of $\mathrm{Cov}(\hat c)$ computed by full matrix inversion.

---

## 7) End-to-end execution order in practice

1. Build/load layout (`layout.npz`) for each observation.
2. Run per-scan reconstruction (`run_reconstruction.py`):
   - solve scan model
   - build exact `cov_inv` and `Pt_Ninv_d`
   - write `scan_XXXX_ml.npz`
3. Run synthesis:
   - `run_synthesis_full.py`: full footprint (`margin_frac=0`)
   - `run_synthesis_margined.py`: inner footprint (`margin_frac=0.10`)
4. Accumulate global `cov_inv_tot` and `Pt_Ninv_d_tot`, solve for $\hat c$, compute uncertain modes, write one combined NPZ.

This decomposition is mathematically equivalent to solving the all-scan marginalized ML system directly, while enabling scan-level parallelism and artifact reuse.