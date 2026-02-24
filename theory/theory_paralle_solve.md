# Parallel ML Solve: Theory-to-Code Mapping

This note documents the **current implementation only** in:

- `cad/analysis_parallel/`
- `cad/src/cad/reconstruct_scan.py`
- `cad/src/cad/synthesize_scan.py` (for exact joint reference)

and uses notation consistent with `latex_cmb_atmosphere/main.tex`.

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
- `c_hat_scan_obs`: `(n_obs_scan,)`
- `precision_diag_scan_obs`: `(n_obs_scan,)`
- `var_diag_scan_obs`: `(n_obs_scan,)`

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
7. In `analysis_parallel`, covariance is approximated as **diagonal**; point estimate remains exact from augmented single-scan solve.

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

This is exactly what `cad.reconstruct_scan.solve_single_scan(...)` solves (matrix-free CG).

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

So the augmented solve and marginalized single-scan ML solve are mathematically equivalent for $c$.
This is the same structure as the marginalized derivation in `latex_cmb_atmosphere/main.tex`.

### 3.3 Covariance of the single-scan estimator

Define the single-scan ML estimator on the observed CMB subspace:

$$
\tilde N_s \equiv N_s + W_s C_a W_s^\top,
\qquad
\hat c_s = \left(P_s^\top \tilde N_s^{-1} P_s\right)^{-1} P_s^\top \tilde N_s^{-1} d_s.
$$

Define:

$$
F_s \equiv P_s^\top \tilde N_s^{-1} P_s,
\qquad
\Sigma_s \equiv \mathrm{Cov}(\hat c_s) = F_s^{-1}.
$$

So the exact covariance is $\Sigma_s$, and the exact diagonal variance is
$\mathrm{diag}(\Sigma_s)$.

For any observed CMB pixel $p$, if $I_{s,p}$ is the set of valid TOD samples in
scan $s$ that hit pixel $p$, then:

$$
(F_s)_{pp} = \sum_{i\in I_{s,p}} \sum_{j\in I_{s,p}} (\tilde N_s^{-1})_{ij}.
$$

This is exact and includes atmospheric correlations through $\tilde N_s$.

### 3.4 Diagonal covariance approximation used in implementation

Current implementation keeps the point estimate exact (from the augmented solve),
and approximates only the covariance diagonal by approximating $\tilde N_s$ as
sample-diagonal:

$$
\tilde N_s \approx \mathrm{diag}\!\left(\sigma_i^2 + (W_s C_a W_s^\top)_{ii}\right).
$$

Then:

$$
(F_s)_{pp} \approx \sum_{i\in I_{s,p}} \frac{1}{\sigma_i^2 + (W_s C_a W_s^\top)_{ii}},
\qquad
\mathrm{Var}_{s,p} \approx \frac{1}{(F_s)_{pp}}.
$$

`run_reconstruction_single.py` computes this by:

1. exact `c_hat_full_mk` from `reconstruct_scan.solve_single_scan(..., estimator_mode="ML")`,
2. sample noise variances from `sol.inv_var`,
3. atmosphere diagonal term $(W_s C_a W_s^\top)_{ii}$ from bilinear weights `sol.w4` and the stationary prior `prior_atm.apply_C`,
4. per-pixel accumulation of $1/(\sigma_i^2 + (W_s C_a W_s^\top)_{ii})$ over samples hitting each observed pixel.

### 3.5 Single-scan implementation mapping

In `cad/analysis_parallel/run_reconstruction_single.py`:

1. Load scan NPZ (`_load_scan`).
2. Estimate wind: `cad.estimate_wind_deg_per_s`.
3. Mask bad detectors (wind-based mask).
4. Estimate per-scan atmospheric spectrum from scan coadd:
   - `map_util.coadd_map_global`
   - `power.radial_cl_1d_from_map`
5. Build CMB and atmosphere boxes:
   - CMB box from global layout (`bbox_ix0`, `bbox_iy0`, `nx`, `ny`)
   - atmosphere box from `util.bbox_pad_for_open_boundary`
6. Build priors:
   - atmosphere prior from estimated `cl_i`
   - placeholder CMB prior (floor-only) for ML interface compatibility
7. Exact single-scan ML point estimate:
   - call `reconstruct_scan.solve_single_scan(..., estimator_mode="ML")`
   - read `sol.c_hat_full_mk`
   - restrict to scan-observed pixels: `c_hat_scan_obs`

### 3.6 Single-scan stored outputs

Saved in `scan_XXXX_ml.npz`:

- `scan_index`
- `source_scan_path`
- `bbox_ix0`, `bbox_iy0`, `nx`, `ny`
- `obs_pix_global_scan`
- `c_hat_scan_obs` (exact point estimate on scan-observed pixels)
- `precision_diag_scan_obs` (diagonal precision approximation)
- `var_diag_scan_obs` (diagonal variance approximation)
- `pixel_size_deg`, `wind_deg_per_s`
- `n_obs_scan`, `estimator_mode`

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
`cad.synthesize_scan.synthesize_scans(...)` (legacy exact joint path).

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

### 4.3 Current parallel joint implementation

In `cad/analysis_parallel/run_synthesis.py`, we combine per-scan point estimates
$\hat c_s[p]$ using the diagonal variances $\mathrm{Var}_{s,p}$ defined above:

$$
\hat c[p] =
\frac{\sum_s \hat c_s[p] / \mathrm{Var}_{s,p}}
{\sum_s 1 / \mathrm{Var}_{s,p}},
\qquad
\mathrm{Var}_{\mathrm{total},p} =
\left(\sum_s \frac{1}{\mathrm{Var}_{s,p}}\right)^{-1}.
$$

If $\sum_s 1/\mathrm{Var}_{s,p}=0$, implementation stores
$\mathrm{Var}_{\mathrm{total},p}=\infty$ and sets $\hat c[p]=0$.

Then:

- `c_hat_obs`: `(n_obs_global,)`
- embed to full `c_hat_full_mk`: `(n_pix_cmb,)`.

Pixels with $\sum_s 1/\mathrm{Var}_{s,p}=0$ are tracked by
`zero_precision_mask` and reported with `var_diag_total = inf`.

### 4.4 Joint stored outputs

Saved in combined NPZ:

- `bbox_ix0`, `bbox_iy0`, `nx`, `ny`
- `obs_pix_global`
- `c_hat_obs`, `c_hat_full_mk`
- `precision_diag_total`, `var_diag_total`
- `zero_precision_mask`
- `n_scans`, `estimator_mode`

---

## 5) End-to-end computation order in current implementation

1. Build global layout (`build_layout.py`):
   - discover scans, global CMB bbox, global observed pixel index set.
2. Run one scan per process (`run_reconstruction_single.py`):
   - exact single-scan ML point estimate
   - diagonal covariance approximation
   - save one NPZ per scan.
3. Synthesize all scans (`run_synthesis.py`):
   - diagonal inverse-variance weighting on global observed pixels
   - save combined map and diagonal uncertainty summaries.

This is the full ML implementation path currently used in `analysis_parallel`.
