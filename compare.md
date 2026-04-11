# Detailed math comparison: `paper_draft` vs local theory + implementation

Date: 2026-04-06  
Paper source reviewed:
- `cad/paper_draft/main.tex` (active includes)
- `cad/paper_draft/3inference_analytic.tex`
- `cad/paper_draft/appendix_wind.tex`

Local references reviewed:
- Theory note: `cad/theory/theory_parallel_solve.md`
- Core implementation:
  - `cad/src/cad/parallel_solve/solver.py`
  - `cad/src/cad/parallel_solve/fisher.py`
  - `cad/src/cad/parallel_solve/reconstruct_scan.py`
  - `cad/src/cad/parallel_solve/synthesize_scan.py`
  - `cad/src/cad/util.py`
  - `cad/src/cad/wind.py`

---

## 1) Active paper math scope

From `paper_draft/main.tex`, the active content is:
- `\input{3inference_analytic}`
- `\input{appendix_wind}`

So this draft is currently:
- single-scan formalism and toy 1D/2D demonstrations,
- conceptual multi-scan claim,
- draft appendix for initial wind estimate.

It does **not** include the more complete analytic appendices (`appendix_numerics`, `app_per_scan`, `app_posterior_marginal`) that exist in other branches of your theory work.

---

## 2) Equation-level parity map (paper -> local theory -> code)

## A. Forward model and frozen-flow advection
- **Paper** (`3inference_analytic.tex`):  
  $d_t^\alpha = P_{t,i}^\alpha T_i + Q_{t,i}^\alpha a_i^0 + \epsilon_t^\alpha$,  
  $a(\theta,t)=a^0(\theta-\mathbf{w}t)$.
- **Local theory** (`theory_parallel_solve.md`): same model with `W_s` and scan index.
- **Code parity**:
  - `solver.py` uses `d = P c + W a0 + n` in joint normal equations.
  - `util.frozen_screen_bilinear_weights()` implements source coordinates
    `x_src = i_x - (w_x/pixel_size_deg)*(t-t0)` and similarly for `y`.
- **Verdict**: mathematically aligned.

## B. Gaussian posterior and atmosphere marginalization
- **Paper**: posterior quadratic in `(T, a0)`, then marginalization gives
  $\tilde N_d = N_d + Q C_a Q^T$ and
  $\hat T = C_{\hat T} P^T \tilde N_d^{-1} d$.
- **Local theory**: same derivation via block elimination/Woodbury.
- **Code parity**:
  - `fisher.py` applies $\tilde N^{-1}$ through
    $N^{-1} - N^{-1}W(C_a^{-1}+W^T N^{-1}W)^{-1}W^T N^{-1}$.
  - builds exactly `cov_inv = P^T \tilde N^{-1} P`, `Pt_Ninv_d = P^T \tilde N^{-1} d`.
- **Verdict**: mathematically aligned.

## C. ML vs MAP handling
- **Paper**: derives MAP covariance with `+ C_c^{-1}` and says focus on ML later.
- **Local theory**: documents both MAP and ML limits.
- **Code behavior**:
  - `solver.py` supports both `ML` and `MAP` (`(1/n_scans) C_c^{-1}` term in MAP path).
  - pipeline drivers currently run `estimator_mode="ML"` (production path).
- **Verdict**: theory aligns; implementation is currently ML-focused by configuration.

## D. Woodbury white-noise expression
- **Paper** Eq-like block:
  $\tilde N_d^{-1}=\sigma_d^{-2}[I-Q(\sigma_d^2 C_a^{-1}+Q^TQ)^{-1}Q^T]$.
- **Code parity**:
  - assumes diagonal/white sample noise via `inv_var`.
  - evaluates the same operator action implicitly with iterative solves.
- **Verdict**: algebra aligns with implementation strategy.

## E. Large-atmosphere geometric projection intuition
- **Paper**: uses projector interpretation
  $I-Q(Q^TQ)^{-1}Q^T$ and mode-filter discussion.
- **Local**:
  - does not rely on this limit analytically; it solves finite-`C_a` systems.
  - still computes uncertain modes from precision eigenspectrum (`synthesize_scan.py`), consistent with geometric interpretation.
- **Verdict**: conceptual alignment; local is more general numerically.

## F. Mode removal / weakly constrained subspace
- **Paper**: removes manually selected contaminated modes in toy examples.
- **Local**:
  - computes smallest-eigenvalue precision modes with Lanczos (`uncertain_mode_vectors`),
  - deprojects these modes in plotting (`plot_util.deproject_uncertain_modes`).
- **Verdict**: local method is a principled extension of the paper idea.

## G. Wind estimation
- **Paper Appendix A**: FFT cross-correlation shift estimator over map pairs.
- **Local**:
  - uses different estimator (`wind.py`): per-time plane-fit gradients + time-regression.
- **Verdict**: algorithm differs; local estimator is not the same as Appendix A.

---

## 3) Confirmed mathematical/technical issues in `paper_draft`

## High-priority issues
- **(H1) Subspace terminology is incorrect** (`3inference_analytic.tex`):  
  text says orthogonal complement of `col(Q)` is "also called the null space of `Q`".  
  Correctly, it is the orthogonal complement / left-null space (`null(Q^T)`), not `null(Q)`.

- **(H2) Large-atmosphere inverse should be pseudoinverse-safe**:  
  expression uses `(Q^TQ)^{-1}` directly in atmosphere-dominant limit.  
  Because `Q` is generally rank-deficient, this should be `(Q^TQ)^+` (or state full-rank assumption explicitly).

- **(H3) Appendix wind model sign conflicts with main frozen-flow equation**:
  - Main text: $a(\theta,t)=a^0(\theta-\mathbf{w}t)$.
  - Appendix A currently writes `a(x + v_x t, y + v_y t)`.
  This is either a sign error or an unstated convention change; currently inconsistent.

## Medium-priority issues
- **(M1) Dimension statement for `Q` is inconsistent with later discussion**:  
  paper states `P` and `Q` are both `N_TOD x N_pix`, then later says atmosphere parameter count is typically larger than CMB.  
  Better to define separate `N_atm` and write `Q in R^{N_TOD x N_atm}`.

- **(M2) Noise symbol inconsistency**:  
  same derivation block uses both `sigma_d` and `sigma_n` for detector noise scale.

- **(M3) ML estimator inverse in degenerate case**:  
  formula uses ordinary inverse for `(P^T Π_perp P)^{-1}` although text later acknowledges rank deficiency (mean mode).  
  Should indicate pseudoinverse or explicitly mention restricted subspace.

## Low-priority / polish issues
- **(L1) Appendix A is mathematically under-specified and typographically broken**:
  malformed products (`;`), malformed FFT expression, malformed interpolation denominator, set notation without `\{ \}`.
  This appendix is not publication-ready in current form.

- **(L2) `\vec l`-mode information expression** (`fishexp`) is presented without normalization and with cosine-only basis choice; valid as diagnostic but should be labeled as such to avoid confusion with orthonormal Fourier power.

---

## 4) Where local implementation goes beyond paper math

- Real scan geometry, missing-sample masks, and per-detector noise weighting.
- Open-boundary atmosphere grid construction (`bbox_pad_for_open_boundary`) and bilinear advection.
- Two-stage compute (joint scan solve + Fisher operator build) with GPU-friendly iterative solvers.
- Multi-observation synthesis and optional inner-footprint margin solve.
- Quantitative uncertain-mode extraction via Lanczos and downstream deprojection.

These are not paper errors; they are implementation-level extensions.

---

## 5) Direct answer: is paper math implemented faithfully?

**Core answer:** yes, the principal linear-Gaussian marginalized inference math in the paper is implemented faithfully in local code (forward model, Woodbury-based effective inverse action, and synthesis-by-precision accumulation).  

**Caveat:** the active `paper_draft` is currently a simplified theory narrative plus toy demonstrations, while local code implements a more complete and practical real-data pipeline with different wind-estimation machinery and additional numerical layers.

---

## 6) Detailed answers to current questions

## Q1. How do the two wind algorithms differ? Which is more robust? What assumptions differ?

### Paper draft wind algorithm (`appendix_wind.tex`)
- Uses FFT cross-correlation between map snapshots separated by `Delta t`.
- Finds displacement peak `(Delta x, Delta y)` and divides by `Delta t` to infer velocity.
- Uses windowing and zero-padding to reduce edge/circular artifacts.
- Aggregates many pairwise estimates via median.

### Local wind algorithm (`cad/src/cad/wind.py`)
- Fits, at each time bin, a focal-plane linear model
  `TOD ~ T0 + dT/dx * dx + dT/dy * dy` across detectors.
- Regresses temporal equation
  `dT/dt - v_scan·grad(T) = - w·grad(T)` to solve `(w_x, w_y)`.
- Uses detector weighting (`eff_counts`), bad-detector masking, and uncertainty diagnostics (`wind_sigma_*`, condition number).

### Fundamental assumption differences
- **Paper appendix method** assumes:
  - atmosphere dominates map morphology strongly enough for correlation peaks,
  - shift estimation from map-to-map translation is reliable,
  - detector geometry can be treated as rasterized map snapshots.
- **Local method** assumes:
  - atmosphere is locally smooth enough over focal plane to support per-time planar approximation,
  - boresight velocity and detector offsets are known and usable in gradient-based regression,
  - enough good detectors/time bins for stable least-squares.

### Robustness comparison (practical)
- For real TOD with detector masking/nonuniformity, the **local gradient-regression method is currently more operationally robust** in this repo because:
  - it directly uses binned detector geometry and scan kinematics,
  - has bad-detector rejection and uncertainty outputs,
  - is integrated in reconstruction path and already exercised on your scan set.
- The paper appendix cross-correlation method can still be useful as:
  - initialization,
  - cross-check in high-atmosphere-SNR periods,
  - fallback when plane-fit assumptions fail.

## Q2. How is rank deficiency handled locally? Why no runtime errors so far?

### Where rank deficiency can appear
- Theoretical weak/null directions (e.g. monopole-like modes, single-scan wind-degenerate subspace) can make precision matrices singular or ill-conditioned.

### Current handling in code
- **Per-scan joint solve (`solver.py`)**
  - ML solve is on active observed subspace (`hits > 0`).
  - Atmosphere block includes `C_a^{-1}` and diagonal preconditioning; solve done by CG.
  - Gauge fixed by subtracting mean from solved `c_act`.
- **Per-scan Fisher solve (`fisher.py`)**
  - Builds `cov_inv_s = P^T Ntilde^{-1} P`, symmetrizes it.
  - Solves `cov_inv_s c_hat = Pt_Ninv_d` with dense linear solve.
  - If singular/ill-conditioned, it raises `RuntimeError` (explicit guard).
- **Global synthesis (`synthesize_scan.py`)**
  - Accumulates `cov_inv_tot`, then solves only on `good_mask = diag(cov_inv_tot) > 0`.
  - Sets zero-precision pixels aside (`zero_precision_mask`).
  - Gauge fixed by subtracting mean on solved subspace.
  - Uncertain directions are explicitly characterized by Lanczos smallest eigenmodes.

### Why it has not crashed in your current runs
- Current produced artifacts are numerically invertible in solved blocks:
  - margined synthesis: all solved pixels had positive precision diagonal (`n_good = n_obs`, no zero-precision pixels),
  - full synthesis: only a tiny fraction dropped by `good_mask`,
  - uncertain directions remain finite and are represented in `uncertain_mode_variances`.
- So rank deficiency is managed by subspace restriction + gauge fixing + mode diagnostics, and your current data realization has not pushed the solve into singular failure.

## Q3. How are boundary conditions handled differently in paper vs local?

### Paper draft
- Toy derivations mostly imply regular-grid operators and Fourier interpretation.
- Boundary treatment is not fully formalized; examples effectively rely on simplified discretizations.

### Local implementation
- **Advection/sampling boundary for atmosphere**: handled as **open boundary**:
  - atmosphere grid is padded (`bbox_pad_for_open_boundary`) so back-advected bilinear source corners stay in-domain,
  - strict bounds checks in `frozen_screen_bilinear_weights`.
- **Prior operator boundary**:
  - `FourierGaussianPrior` applies `C^{-1}` via FFT on chosen finite grid (periodic operator on that grid).
- **Synthesis robustness near edges**:
  - optional `margin_frac` (`run_synthesis_margined.py`) trims outer footprint before global accumulation/solve to reduce edge-driven weak modes/artifacts.

### Net effect
- Relative to paper toy treatment, local code uses a hybrid practical boundary strategy:
  - open-boundary geometry for advection operator,
  - FFT-diagonal prior on finite patch,
  - optional inner-footprint trimming for stable synthesis.
