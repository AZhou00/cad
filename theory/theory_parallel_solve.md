# Parallel ML Solve: Theory-to-Code Mapping

This note documents the ML pipeline implemented in:

- `cad/analysis_parallel/run_reconstruction.py`
- `cad/analysis_parallel/run_synthesis_full.py`
- `cad/analysis_parallel/run_synthesis_margined.py`
- `cad/src/cad/parallel_solve/reconstruct_scan.py`
- `cad/src/cad/parallel_solve/fisher.py`
- `cad/src/cad/parallel_solve/synthesize_scan.py`

Goal: keep notation light, define symbols before use, and match code names directly.

---

## 1) Minimal symbols and code naming

For each scan `s`:

- $d_s$: valid TOD vector (`sol.tod_valid_mk`)
- $c$: static CMB map coefficients on the chosen pixel basis
- $a_s^0$: atmosphere screen at scan reference time
- $P_s$: CMB pointing operator (implemented by `pix_obs_local` binning/scatter)
- $W_s$: atmosphere advection+interpolation operator (implemented by `idx4`, `w4`)
- $N_s$: diagonal TOD noise covariance (`inv_var = diag(N_s^{-1})`)
- $C_a$: atmosphere prior covariance

Derived effective covariance:

$$
\tilde N_s = N_s + W_s C_a W_s^\top.
$$

Main stored arrays:

- per scan: `cov_inv = P_s^\top \tilde N_s^{-1} P_s`
- per scan: `Pt_Ninv_d = P_s^\top \tilde N_s^{-1} d_s`
- synthesis: `cov_inv_tot = \sum_s` remapped per-scan `cov_inv`
- synthesis: `Pt_Ninv_d_tot = \sum_s` remapped per-scan `Pt_Ninv_d`
- solution: `c_hat_obs` from `cov_inv_tot @ c_hat_obs = Pt_Ninv_d_tot`

Flattening convention used everywhere:

$$
\texttt{pix} = i_y + i_x n_y.
$$

This is the convention behind `obs_pix_global`, `global_to_obs`, and all map reshapes.

---

## 2) Per-scan ML model and derivation

### 2.1 Forward model and objective

For one scan:

$$
d_s = P_s c + W_s a_s^0 + n_s, \qquad n_s \sim \mathcal N(0, N_s),
$$
$$
a_s^0 \sim \mathcal N(0, C_a).
$$

The per-scan ML objective (negative log posterior up to constants) is:

$$
\Phi_s(c,a_s^0)
= \frac12(d_s-P_sc-W_sa_s^0)^\top N_s^{-1}(d_s-P_sc-W_sa_s^0)
+ \frac12(a_s^0)^\top C_a^{-1}a_s^0.
$$

### 2.2 Block normal equation

Taking derivatives w.r.t. $c$ and $a_s^0$ gives:

$$
\begin{bmatrix}
P_s^\top N_s^{-1}P_s & P_s^\top N_s^{-1}W_s \\
W_s^\top N_s^{-1}P_s & W_s^\top N_s^{-1}W_s + C_a^{-1}
\end{bmatrix}
\begin{bmatrix}
c\\a_s^0
\end{bmatrix}
=
\begin{bmatrix}
P_s^\top N_s^{-1}d_s\\
W_s^\top N_s^{-1}d_s
\end{bmatrix}.
$$

To keep notation short, define:

$$
A=P_s^\top N_s^{-1}P_s,\quad
B=P_s^\top N_s^{-1}W_s,\quad
D=W_s^\top N_s^{-1}W_s + C_a^{-1},
$$
$$
b=P_s^\top N_s^{-1}d_s,\quad
e=W_s^\top N_s^{-1}d_s.
$$

Then the block system is:

$$
Ac+Ba=b,\qquad B^\top c+Da=e.
$$

Eliminate $a$ via $a=D^{-1}(e-B^\top c)$:

$$
(A-BD^{-1}B^\top)c=b-BD^{-1}e.
$$

Using Woodbury identities, this is exactly:

$$
P_s^\top \tilde N_s^{-1}P_s\,c = P_s^\top \tilde N_s^{-1}d_s.
$$

Therefore:

$$
\texttt{cov\_inv} = P_s^\top \tilde N_s^{-1}P_s,\qquad
\texttt{Pt\_Ninv\_d} = P_s^\top \tilde N_s^{-1}d_s.
$$

And per-scan estimate:

$$
\hat c_s = \texttt{cov\_inv}^{-1}\texttt{Pt\_Ninv\_d},
$$

with monopole gauge fixing in code (subtract mean on solved subspace).

---

## 3) How `fisher.py` computes these objects

`cad/src/cad/parallel_solve/fisher.py` never forms $\tilde N_s$ explicitly.
It uses:

$$
M_s = C_a^{-1} + W_s^\top N_s^{-1}W_s,
$$
$$
\tilde N_s^{-1}
=N_s^{-1}-N_s^{-1}W_s M_s^{-1}W_s^\top N_s^{-1}.
$$

For any TOD vector $u$:

1. compute $r = W_s^\top N_s^{-1}u$
2. solve $M_s x = r$
3. return $\tilde N_s^{-1}u = N_s^{-1}u - N_s^{-1}W_sx$

Then:

- `Pt_Ninv_d`: apply the above with $u=d_s$, then apply $P_s^\top$
- `cov_inv[:, j]`: apply the above with $u=P_se_j$, then apply $P_s^\top$

where $e_j$ is scan-local observed-pixel basis vector.

### 3.1 Exact formula vs numerical implementation

The algebra above is exact.  
Numerically, the $M_s$ solves are iterative (PCG), so the realized action is accurate up to iteration budget/tolerance.

---

## 4) What “CG solve of \(M_s\)” means in this code

### 4.1 Which CG routines are used

In `cad/src/cad/parallel_solve/fisher.py`:

- `_cg_single(...)`: fixed-iteration preconditioned CG for one RHS
- `_cg_batched(...)`: fixed-iteration preconditioned CG for many RHS in parallel
- both are JAX implementations (`jax.lax.scan`) used in production path
- `run_one_M_s_solve_converged(...)` uses `_cg_single_converged(...)` (residual-stop variant) for benchmarking, not the main production path

Production call path:

- `build_scan_information(..., cg_niter=...)` sets iteration count
- `run_reconstruction.py` sets `CG_MAXITER` and passes it through as `cg_niter`

### 4.2 Brief CG sketch (self-contained)

For SPD system $Ax=b$, CG builds iterates in Krylov spaces:

$$
\mathcal K_k(A,r_0)=\text{span}\{r_0,Ar_0,\dots,A^{k-1}r_0\},
$$

and at step $k$ picks $x_k$ that minimizes quadratic energy
$\frac12 x^\top A x - b^\top x$ over that space.

Standard recurrence (preconditioned form, $M\approx A$):

1. $r_0=b-Ax_0$
2. $z_0=M^{-1}r_0$, $p_0=z_0$
3. iterate
   - $\alpha_k=(r_k^\top z_k)/(p_k^\top A p_k)$
   - $x_{k+1}=x_k+\alpha_k p_k$
   - $r_{k+1}=r_k-\alpha_k A p_k$
   - $z_{k+1}=M^{-1}r_{k+1}$
   - $\beta_k=(r_{k+1}^\top z_{k+1})/(r_k^\top z_k)$
   - $p_{k+1}=z_{k+1}+\beta_k p_k$

In this code:

- $A$ is $M_s$
- preconditioner is diagonal (`diag_M`) built from approximate diagonal of $M_s$
- stopping rule in production is **fixed iteration count**, not residual threshold

### 4.3 Why this is used

- avoids dense factorization of $M_s$ for each scan
- easy to batch for many RHS when building `cov_inv` columns
- JAX-friendly and GPU-friendly implementation

---

## 5) Global synthesis derivation

Each scan has local observed indexing. Let $R_s$ be the remap from global observed basis to scan-local observed basis (implemented by `obs_pix_global_scan` + `global_to_obs`).

Per scan:

$$
\texttt{cov\_inv}_s = P_s^\top \tilde N_s^{-1}P_s,\qquad
\texttt{Pt\_Ninv\_d}_s = P_s^\top \tilde N_s^{-1}d_s.
$$

Global accumulation:

$$
\texttt{cov\_inv\_tot}
= \sum_s R_s^\top \texttt{cov\_inv}_s R_s,
$$
$$
\texttt{Pt\_Ninv\_d\_tot}
= \sum_s R_s^\top \texttt{Pt\_Ninv\_d}_s.
$$

This is the same as:

$$
\left(\sum_s P_s^\top \tilde N_s^{-1}P_s\right)c
=\sum_s P_s^\top \tilde N_s^{-1}d_s,
$$

written in one shared observed-index basis.

Code then solves on `good_mask` where `diag(cov_inv_tot) > 0`, sets other pixels to zero, and subtracts mean on solved subspace.

---

## 6) Margined synthesis

`run_synthesis_margined.py` uses `margin_frac=f` to trim each side:

$$
m_x=\lfloor fn_x\rfloor,\qquad m_y=\lfloor fn_y\rfloor.
$$

Inner grid:

$$
n_x'=n_x-2m_x,\qquad n_y'=n_y-2m_y.
$$

Old flat index decode:

$$
i_x=\left\lfloor p/n_y\right\rfloor,\qquad i_y=p-i_xn_y.
$$

Inner remap:

$$
p'=(i_y-m_y)+(i_x-m_x)n_y'.
$$

All sums, solve, diagnostics, and uncertain modes are done on the inner footprint.

---

## 7) Uncertain modes: derivation + algorithm details

Define:

$$
A_g = \texttt{cov\_inv\_good}
=\left[\texttt{cov\_inv\_tot}\right]_{\texttt{good\_mask},\texttt{good\_mask}}.
$$

Eigenproblem:

$$
A_g v_i = \lambda_i v_i,\qquad
0<\lambda_1\le\lambda_2\le\cdots.
$$

Since covariance is approximately $A_g^{-1}$, mode variance scales as:

$$
\mathrm{Var}(v_i)\approx 1/\lambda_i.
$$

So **smallest** $\lambda_i$ are the most weakly constrained (most uncertain).

### 7.1 Lanczos-Ritz in `synthesize_scan.py`

Implemented in `_lanczos_smallest_modes(...)`:

1. pick random normalized start vector (`seed = LANCZOS_SEED`)
2. run Lanczos recurrence with full re-orthogonalization
3. build tridiagonal $T_m$ from $\alpha_j,\beta_j$
4. solve small eigensystem of $T_m$
5. map Ritz vectors back to pixel space
6. QR orthonormalize, compute Rayleigh quotients, sort ascending
7. store:
   - `uncertain_mode_vectors`
   - `uncertain_mode_variances = 1/max(lambda, 1e-18)`

### 7.2 What each hyperparameter means

Used in `run_synthesis_full.py` and `run_synthesis_margined.py`:

- `N_UNCERTAIN_MODES`:
  requested number of smallest-eigenvalue modes to return (`k`)
- `LANCZOS_OVERSAMPLE`:
  extra Krylov dimension beyond `k` to improve Ritz accuracy
- `LANCZOS_MAXITER`:
  hard cap on Krylov dimension / iterations

Actual Lanczos dimension in code:

$$
m = \min\!\Big(n_g,\ \max(k+\text{oversample},2k),\ \text{maxiter}\Big),\quad m\ge k.
$$

Where $n_g = \sum \texttt{good\_mask}$.

Interpretation:

- larger `N_UNCERTAIN_MODES`: returns more uncertain directions but costs more memory/time
- larger `LANCZOS_OVERSAMPLE`: better eigenpair quality near the cutoff, modest extra cost
- larger `LANCZOS_MAXITER`: allows deeper Krylov basis; useful when spectrum is clustered

---

## 8) Key reconstruction/synthesis hyperparameters in code

### 8.1 Per-scan reconstruction (`run_reconstruction.py`)

- `N_ELL_BINS`: bins for atmosphere power estimate used in atmosphere prior
- `CL_FLOOR_MK2`: floor on $C_\ell$ to keep prior operators well-conditioned
- `NOISE_MK`: raw-detector white-noise scale used to build `inv_var`
- `CG_TOL`, `CG_MAXITER`: passed to `solve_single_scan` joint CG (`solver.py`)
- `CG_MAXITER` also passed as `cg_niter` for fixed-iteration PCG in `fisher.py`

### 8.2 Synthesis

- `MARGIN_FRAC`: 0 for full solve, >0 for inner-footprint solve
- `N_UNCERTAIN_MODES`, `LANCZOS_OVERSAMPLE`, `LANCZOS_MAXITER`: uncertain-mode settings above

---

## 9) Output interface

Per-scan artifact (`scan_XXXX_ml.npz`):

- `obs_pix_global_scan`
- `c_hat_scan_obs`
- `cov_inv`
- `Pt_Ninv_d`
- wind + atmosphere metadata (`wind_*`, `ell_atm`, `cl_atm_mk2`)

Combined synthesis artifact (`recon_combined_ml_full.npz` or `recon_combined_ml_margined.npz`):

- geometry: `bbox_ix0`, `bbox_iy0`, `nx`, `ny`, `pixel_size_deg`
- indexing: `obs_pix_global`
- map solution: `c_hat_obs`, `c_hat_full_mk`
- precision diagnostics: `cov_inv_tot`, `precision_diag_total`, `good_mask`, `zero_precision_mask`, `var_diag_total`
- uncertain modes: `uncertain_mode_vectors`, `uncertain_mode_variances`, `lanczos_n_modes`, `n_uncertain_modes_stored`
- provenance: `scan_metadata`, `n_scans`, `n_scans_used`, `estimator_mode`

`var_diag_total` is a diagonal proxy $1/\mathrm{diag}(\texttt{cov\_inv\_tot})$ where positive, not the diagonal of the fully inverted covariance.

---

## 10) Execution order

1. Build layout (`build_layout.py`).
2. Run per-scan reconstruction (`run_reconstruction.py`) to write `scan_XXXX_ml.npz`.
3. Run synthesis:
   - `run_synthesis_full.py`
   - `run_synthesis_margined.py`
4. Optional plotting: `plot_reconstruction.py`.

This parallel decomposition is mathematically the same marginalized ML system as direct all-scan solve, while enabling scan-level parallelism and artifact reuse.
