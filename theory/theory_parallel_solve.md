# Parallel ML Solve: Theory-to-Code Mapping

This note documents the **current implementation only** in:

- `cad/analysis_parallel/` (CLI entry points)
- `cad/src/cad/direct_solve/reconstruct_scan.py` (single-scan augmented solve: `solve_single_scan`)
- `cad/src/cad/parallel_solve/reconstruct_scan.py` (per-scan: CPU solve then build of cov_inv, Pt_Ninv_d: `run_one_scan`)
- `cad/src/cad/parallel_solve/fisher.py` (build of [Cov(\hat c_s)]^{-1}, P^T \tilde N^{-1} d, c_hat_s)
- `cad/src/cad/parallel_solve/synthesize_scan.py` (exact global synthesis: `run_synthesis`)

The exact joint path (all-scan augmented solve) is in `cad/src/cad/direct_solve/synthesize_scan.py` (`synthesize_scans`). Notation is consistent with `latex_cmb_atmosphere/main.tex`.

We consider **ML only** throughout this document. MAP is intentionally not discussed.

---

## 1) Variables and shapes

### 1.1 Indices

- $s \in \{1,\dots,S\}$: scan index
- $i$: valid TOD sample index within one scan
- $p$: CMB pixel index on the CMB bbox grid
- $q$: atmosphere pixel index on the padded atmosphere bbox grid

### 1.2 Fields and operators

For scan $s$:

- $d_s \in \mathbb{R}^{n_{\mathrm{valid},s}}$: valid TOD vector
- $c \in \mathbb{R}^{n_{\mathrm{pix,cmb}}}$: static CMB map
- $a_s^0 \in \mathbb{R}^{n_{\mathrm{pix,atm},s}}$: atmosphere screen at reference time $t_0$
- $P_s$: pointing operator from CMB pixels to valid TOD samples
- $W_s$: frozen-screen bilinear advection operator from atmosphere pixels to valid TOD samples
- $N_s=\mathrm{diag}(\sigma_i^2)$: diagonal noise covariance for valid samples
- $C_a$: atmospheric prior covariance

Flattening convention used in code:

- `pix = iy + ix * ny`

### 1.3 Implementation arrays (core)

- `pix_index`: `(n_t, n_det, 2)` integer global `(ix, iy)`
- `obs_pix_global`: `(n_obs_global,)` global observed CMB pixel indices
- `global_to_obs`: `(n_pix_cmb,)`, value in `[0, n_obs_global)` or `-1`
- `c_hat_full_mk`: `(n_pix_cmb,)`
- `c_hat_scan_obs`: `(n_obs_scan,)` (point estimate on scan-observed pixels)
- `cov_inv`: `(n_obs_scan, n_obs_scan)` per-scan inverse covariance [Cov(\hat c_s)]^{-1} = P_s^\top \tilde N_s^{-1} P_s
- `Pt_Ninv_d`: `(n_obs_scan,)` P_s^\top \tilde N_s^{-1} d_s

---

## 2) Problem and assumptions

### 2.1 Forward model (per scan)

$$
d_s = P_s c + W_s a_s^0 + n_s,\qquad n_s\sim\mathcal{N}(0,N_s),\quad N_s=\mathrm{diag}(\sigma_i^2).
$$

Atmosphere prior:

$$
a_s^0 \sim \mathcal{N}(0,C_a).
$$

### 2.2 Assumptions used by implementation

1. **ML only** (`estimator_mode="ML"`).
2. **Frozen-flow atmosphere** within each scan; advection referenced to `t_s[0]`.
3. **Diagonal noise** with per-sample variance from detector-dependent noise.
4. **Stationary atmospheric prior** implemented by FFT-diagonal `apply_Cinv`.
5. **Observed-pixel restriction** via `obs_pix_global` and `global_to_obs`.
6. **Gauge fixing** by removing mean on solved CMB subspace.
7. In `analysis_parallel`, per-scan **exact** [Cov(\hat c_s)]^{-1} and P^\top \tilde N^{-1} d are built; point estimate from the normal equation; global synthesis uses the summed information and RHS.

---

## 3) Single-scan ML solution

### 3.1 Derivation of augmented linear system

For one scan (drop subscript $s$), define the negative log-posterior (up to constants):

$$
\Phi(c,a^0)=\frac{1}{2}(d-Pc-Wa^0)^\top N^{-1}(d-Pc-Wa^0)+\frac{1}{2}(a^0)^\top C_a^{-1}a^0.
$$

Set gradients to zero:

$$
\frac{\partial\Phi}{\partial c}=-P^\top N^{-1}(d-Pc-Wa^0)=0,
$$
$$
\frac{\partial\Phi}{\partial a^0}=-W^\top N^{-1}(d-Pc-Wa^0)+C_a^{-1}a^0=0.
$$

Rearrange:

$$
\begin{bmatrix}
P^\top N^{-1}P & P^\top N^{-1}W \\
W^\top N^{-1}P & W^\top N^{-1}W + C_a^{-1}
\end{bmatrix}
\begin{bmatrix}
c\\
a^0
\end{bmatrix}
=
\begin{bmatrix}
P^\top N^{-1}d\\
W^\top N^{-1}d
\end{bmatrix}.
$$

This is exactly what `cad.reconstruct_scan.solve_single_scan(...)` (in `cad/src/cad/direct_solve/reconstruct_scan.py`) solves (matrix-free CG).

### 3.2 Marginalized form and equivalence to LaTeX

Define:

$$
\tilde N = N + WC_aW^\top,\qquad M=C_a^{-1}+W^\top N^{-1}W.
$$

Woodbury:

$$
\tilde N^{-1}=N^{-1}-N^{-1}WM^{-1}W^\top N^{-1}.
$$

Eliminating $a^0$ from the augmented system (Schur complement) gives:

$$
\left(P^\top\tilde N^{-1}P\right)c = P^\top\tilde N^{-1}d.
$$

So the augmented solve and marginalized single-scan ML solve are mathematically equivalent for the point estimate $c$.

The **covariance** of $\hat c$ is also the same in both formulations. In the augmented system, the block for $c$ has precision $P^\top N^{-1} P - P^\top N^{-1} W (W^\top N^{-1} W + C_a^{-1})^{-1} W^\top N^{-1} P$; by Woodbury this equals $P^\top \tilde N^{-1} P$. So $\mathrm{Cov}(\hat c_s) = [P_s^\top \tilde N_s^{-1} P_s]^{-1}$ in both the augmented and the marginalized formulation.
This is the same structure as the marginalized derivation in `latex_cmb_atmosphere/main.tex`.

### 3.3 Covariance of the single-scan estimator

The single-scan ML estimator on the observed CMB subspace is

$$
\tilde N_s \equiv N_s + W_s C_a W_s^\top,
\qquad
\hat c_s = \left(P_s^\top \tilde N_s^{-1} P_s\right)^{-1} P_s^\top \tilde N_s^{-1} d_s.
$$

The covariance of $\hat c_s$ is

$$
\mathrm{Cov}(\hat c_s) = \left(P_s^\top \tilde N_s^{-1} P_s\right)^{-1}.
$$

We denote the inverse covariance (precision) in code as `cov_inv`; then $\mathrm{Cov}(\hat c_s)^{-1} = P_s^\top \tilde N_s^{-1} P_s$. The normal equation is $[\mathrm{Cov}(\hat c_s)]^{-1} \, \hat c_s = P_s^\top \tilde N_s^{-1} d_s$, so the RHS is $P_s^\top \tilde N_s^{-1} d_s$ (in code `Pt_Ninv_d`).

For any observed CMB pixel $p$, if $I_{s,p}$ is the set of valid TOD samples in scan $s$ that hit pixel $p$, then the $(p,p)$ entry of $[\mathrm{Cov}(\hat c_s)]^{-1}$ is

$$
\left(P_s^\top \tilde N_s^{-1} P_s\right)_{pp} = \sum_{i\in I_{s,p}} \sum_{j\in I_{s,p}} (\tilde N_s^{-1})_{ij}.
$$

This is exact and includes atmospheric correlations through $\tilde N_s$.

### 3.4 Per-scan build of cov_inv and Pt_Ninv_d

$[\mathrm{Cov}(\hat c_s)]^{-1} = P_s^\top \tilde N_s^{-1} P_s$ and $P_s^\top \tilde N_s^{-1} d_s$ are built in `cad/src/cad/parallel_solve/fisher.py`. Woodbury gives $\tilde N_s^{-1}$ applied via $M_s = C_a^{-1} + W^\top N^{-1} W$: column $j$ of the inverse covariance requires one $M_s$ solve with RHS $W^\top N^{-1} z_j$ ($z_j$ unit on samples hitting pixel $j$); $P_s^\top \tilde N_s^{-1} d_s$ requires one $M_s$ solve with RHS $W^\top N^{-1} d$, then $y = N^{-1}d - N^{-1} W u$ and $P^\top y$. Batched CG over columns is used. Point estimate: solve $[\mathrm{Cov}(\hat c_s)]^{-1} x = P_s^\top \tilde N_s^{-1} d_s$ for $x = \hat c_s$, then apply mean subtraction for gauge. Requires JAX/GPU (set `CUDA_VISIBLE_DEVICES` per process).

### 3.5 Single-scan implementation mapping

In `cad/analysis_parallel/run_reconstruction.py` (multi-GPU) and `cad/src/cad/parallel_solve/reconstruct_scan.py`:

1. Load scan, estimate wind, mask bad detectors, estimate atmospheric spectrum, build bboxes and priors (same as before).
2. Call `solve_single_scan(..., estimator_mode="ML")` on CPU (from `cad/src/cad/direct_solve/reconstruct_scan.py`) to obtain `sol` (idx4, w4, inv_var, pix_obs_local, tod_valid_mk, etc.).
3. Call `fisher.build_scan_information(...)` to build cov_inv, Pt_Ninv_d, and c_hat_scan_obs from the normal equation.
4. Write a single NPZ per scan.

### 3.6 Single-scan stored outputs

Saved in `scan_XXXX_ml.npz`:

- `scan_index`, `source_scan_path`
- `bbox_ix0`, `bbox_iy0`, `nx`, `ny`
- `obs_pix_global_scan` `(n_obs_scan,)`
- `c_hat_scan_obs` `(n_obs_scan,)` (point estimate from the normal equation)
- `cov_inv` `(n_obs_scan, n_obs_scan)`, `Pt_Ninv_d` `(n_obs_scan,)`
- `pixel_size_deg`, `wind_deg_per_s`, `n_obs_scan`, `estimator_mode`

---

## 4) Joint (all-scan) ML solution

### 4.1 Exact joint augmented system (reference)

Unknown:

$$
x=[c;\,a_1^0;\dots;a_S^0].
$$

Negative log-posterior:

$$
\Phi(c,\{a_s^0\})
=\frac{1}{2}\sum_{s=1}^S(d_s-P_sc-W_sa_s^0)^\top N_s^{-1}(d_s-P_sc-W_sa_s^0)
+\frac{1}{2}\sum_{s=1}^S(a_s^0)^\top C_a^{-1}a_s^0.
$$

Stationarity gives the block system solved by
`cad.synthesize_scan.synthesize_scans(...)` (legacy exact joint path in `cad/src/cad/direct_solve/synthesize_scan.py`).

### 4.2 Exact marginalized global system and equivalence to LaTeX

Eliminate each $a_s^0$ scan-by-scan:

$$
\left(\sum_{s=1}^S P_s^\top\tilde N_s^{-1}P_s\right)c
=\sum_{s=1}^S P_s^\top\tilde N_s^{-1}d_s,
\qquad
\tilde N_s=N_s+W_sC_aW_s^\top.
$$

This is exactly the global marginalized ML equation in the LaTeX derivation.

So exact joint augmented and exact global marginalized systems are equivalent for the CMB point estimate.

### 4.3 Exact parallel joint implementation (ML only)

No $C_c^{-1}$ term is used anywhere in the parallel path; the global solution is pure ML.

In `cad/analysis_parallel/run_synthesis.py` and `cad/src/cad/parallel_solve/synthesize_scan.py`:

$$
[\mathrm{Cov}(\hat c)]^{-1}_{\mathrm{tot}} = \sum_s [\mathrm{Cov}(\hat c_s)]^{-1} \quad\text{(at global observed indices)},\qquad
(P^\top \tilde N^{-1} d)_{\mathrm{tot}} = \sum_s P_s^\top \tilde N_s^{-1} d_s,\qquad
[\mathrm{Cov}(\hat c)]^{-1}_{\mathrm{tot}} \, \hat c = (P^\top \tilde N^{-1} d)_{\mathrm{tot}}.
$$

Each scan NPZ contributes cov_inv and Pt_Ninv_d; indices are mapped to the global observed set via `global_to_obs`. The linear solve (cov_inv_tot @ c_hat = Pt_Ninv_d_tot) is performed with the same math either on GPU (JAX `linalg.solve`, when available) or on CPU (scipy). The dense precision matrix must be formed to save it; CG could be used for the solve instead of a direct Cholesky/solve, but with the matrix already formed the direct solve is simple and on GPU is efficient. Gauge: subtract mean of $\hat c$ on observed pixels. Then:

- `c_hat_obs`: `(n_obs_global,)`
- embed to full `c_hat_full_mk`: `(n_pix_cmb,)`.

`precision_diag_total` is the diagonal of the summed inverse covariance; `var_diag_total` is set to the reciprocal where positive (approximation to diagonal of the global covariance). Pixels with no contribution are `zero_precision_mask` with `var_diag_total = \infty`. The dense precision matrix `cov_inv_tot` is always saved.

### 4.4 Joint stored outputs

Saved in combined NPZ (`recon_combined_ml.npz`):

- `bbox_ix0`, `bbox_iy0`, `nx`, `ny`, `pixel_size_deg`, `obs_pix_global`
- `c_hat_obs`, `c_hat_full_mk`
- `precision_diag_total`, `var_diag_total`, `zero_precision_mask`
- `cov_inv_tot` (dense precision matrix, n_obs x n_obs)
- `n_scans`, `n_scans_used`, `estimator_mode`

---

## 5) End-to-end computation order

1. Build global layout: discover scans, global CMB bbox, global observed pixel set.
2. Run reconstruction (`run_reconstruction.py`): one scan per process across 4 GPUs, skip completed:
   - CPU: `solve_single_scan` to get `sol`
   - Build cov_inv, Pt_Ninv_d, c_hat_scan_obs
   - save one NPZ per scan (c_hat_scan_obs, cov_inv, Pt_Ninv_d, metadata).
3. Synthesize all scans (`run_synthesis.py`): load per-scan NPZs, accumulate cov_inv_tot and Pt_Ninv_d_tot, solve for $\hat c$, write combined map.
