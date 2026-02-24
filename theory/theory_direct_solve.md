## 1. Model (per scan)

For scan $s \in \{1,\dots,S\}$ we model the binned TOD as

$$
d_s = P_s\,c + W_s\,a_s^0 + n_s
$$

- $d_s$: $(n_t,n_{\rm det})$ TOD in mK (after binning). In the solver we flatten to a vector of the finite samples, $d_s \in \mathbb{R}^{N_{{\rm TOD},s}}$.
- $c$: static sky map on the **CMB bbox** (a global RA/DEC-degree pixel grid), $c \in \mathbb{R}^{N_{\rm pix}}$.
- $a_s^0$: per-scan frozen atmosphere reference screen on the **atmosphere bbox** (a padded global pixel grid), $a_s^0 \in \mathbb{R}^{N_{\rm atm}}$.
- $n_s \sim \mathcal{N}(0, N_s)$: white instrumental noise with diagonal $N_s$ in the TOD basis (per-sample variances inferred from per-detector noise).
- $P_s$: pointing operator that selects one CMB pixel per TOD sample (nearest-pixel), built from `pix_index`.
- $W_s$: frozen-screen advection + sampling operator that bilinear-interpolates $a_s^0$ at back-advected coordinates (open boundary on the atmosphere bbox).

### Grids and unknown bases (implementation)
- **CMB bbox**: for multi-scan inference we use the tight union bbox over observed pixels across scans (no padding).
- **Atmosphere bbox**: padded bbox large enough to keep open-boundary bilinear advection in-bounds (see `cad.util.bbox_pad_for_open_boundary`).
- **Observed pixel basis**: for ML we solve only for CMB pixels with hits above `min_hits_per_pix`; for MAP we solve the full CMB bbox grid.
- **Monopole gauge**: there is a near-degeneracy between the mean of $c$ and the means of $a_s^0$; we fix this gauge by subtracting the mean of the reconstructed $c$ map after solving.

## 2. Frozen-screen advection ($W_s$)

Let the wind for scan $s$ be $w_s=(w_x,w_y)$ in deg/s (it's estimation is
discussed below) in the same RA/DEC-degree basis as `pix_index`.

For each valid TOD sample at time $t$ with **global** hit pixel $(i_x,i_y)$, the atmosphere reference screen is sampled at

$$
x_{\rm src} = i_x - \frac{w_x}{\Delta_{\rm pix}}(t-t_{\rm ref}),\qquad
y_{\rm src} = i_y - \frac{w_y}{\Delta_{\rm pix}}(t-t_{\rm ref}),
$$

where $\Delta_{\rm pix}=$ `pixel_size_deg` and $t_{\rm ref} = t_s[0]$ in the implementation. We evaluate $a_s^0(x_{\rm src},y_{\rm src})$ by bilinear interpolation on the atmosphere grid (4 corners and weights). Samples whose bilinear corners fall outside the atmosphere bbox are not used; in code this is enforced as a strict check (so the atmosphere bbox must be padded sufficiently).

## Priors

Atmosphere (each scan independently):

$$
a_s^0 \sim \mathcal{N}(0, C_a)
$$

CMB:

$$
c \sim \mathcal{N}(0, C_c) \quad \text{(MAP only)}
$$

Both $C_a$ and $C_c$ are implemented as stationary FFT-diagonal operators on their respective finite bboxes (periodic boundary on that bbox). Units are mK$^2$.

### Anisotropic RA/Dec metric in the prior
The pixel grid is uniform in RA/DEC degrees, but physical length in RA is smaller by $\cos({\rm DEC})$.
We therefore define Fourier radius using

$$
\ell = \sqrt{\left(\frac{k_x}{\cos({\rm DEC}_{\rm ref})}\right)^2 + k_y^2},
$$

so that isotropic spectra on the sky remain isotropic in physical space. The FFT prior normalization uses
$dx\,dy = \Delta_{\rm pix}^2 \cos({\rm DEC}_{\rm ref})$, where $\Delta_{\rm pix}$ is in radians. Here ${\rm DEC}_{\rm ref}$
is taken as a constant representative DEC for the field (implementation: derived from the field's bbox center).
The pointing grid itself remains unchanged; only the prior's spectral weighting and normalization are anisotropic.

## 3. Estimators

- **ML**: maximize likelihood w.r.t. $(c,\{a_s^0\})$ (equivalently a flat prior on $c$; $C_c^{-1}=0$).
- **MAP**: maximize posterior w.r.t. $(c,\{a_s^0\})$ using both priors.

In the combined multi-scan solve, $c$ is a single shared unknown so the MAP system includes $C_c^{-1}$ once. In the per-scan diagnostic solve (solving each scan separately on the same CMB grid), we use $(1/S)C_c^{-1}$ so that summing per-scan contributions corresponds to a single $C_c^{-1}$ term.

## 4. Solver (joint conjugate gradient)

### Objective
Let $N_s$ be diagonal in the TOD basis. For ML, we minimize

$$
\frac{1}{2}\sum_{s=1}^S (d_s - P_s c - W_s a_s^0)^T N_s^{-1} (d_s - P_s c - W_s a_s^0)
\;+\;
\frac{1}{2}\sum_{s=1}^S (a_s^0)^T C_a^{-1} a_s^0.
$$

For MAP, we add $\frac{1}{2}c^T C_c^{-1} c$.

### Unknowns and normal equations
We solve the (symmetric positive semidefinite; typically positive definite) normal equations for the stacked unknown vector

$$
x = \begin{bmatrix} c \\ a_1^0 \\ \vdots \\ a_S^0 \end{bmatrix}
$$

The solution $x^\star$ contains the reconstructed CMB map $c^\star$ (primary
output) and, optionally, per-scan atmosphere screens (nuisance). The linear
system is

$$
A x = b,
$$

where $A$ and $b$ have the block structure:

$$
A =
\begin{bmatrix}
A_{cc} & A_{c a_1} & \cdots & A_{c a_S} \\
A_{a_1 c} & A_{a_1 a_1} & & 0 \\
\vdots & & \ddots & \\
A_{a_S c} & 0 & & A_{a_S a_S}
\end{bmatrix},
\qquad
b =
\begin{bmatrix}
b_c \\ b_{a_1} \\ \vdots \\ b_{a_S}
\end{bmatrix}.
$$

The off-diagonal blocks between different scans are zero because each $a_s^0$ only appears in scan $s$'s likelihood and prior.

$$
\begin{aligned}
A_{cc} &= \sum_{s=1}^S P_s^T N_s^{-1} P_s \\
A_{c a_s} &= P_s^T N_s^{-1} W_s \\
A_{a_s c} &= W_s^T N_s^{-1} P_s \\
A_{a_s a_s} &= W_s^T N_s^{-1} W_s + C_a^{-1} \qquad (s=1,\dots,S)
\end{aligned}
$$

In MAP, the CMB prior adds once to the shared CMB block:

$$
A_{cc} \leftarrow A_{cc} + C_c^{-1}.
$$

The right-hand side vector $b$ is

$$
b_c = \sum_s P_s^T N_s^{-1} d_s,\qquad
b_{a_s} = W_s^T N_s^{-1} d_s.
$$

In the implementation we solve for the full stacked vector
$$
x=\begin{bmatrix} c \\ a_1^0 \\ \vdots \\ a_S^0 \end{bmatrix},
$$
i.e. the atmosphere screens are part of the numerical solution. The public reconstruction outputs currently keep only $c^\star$ and discard the solved $a_s^{0\star}$ values.

Concretely, the atmosphere solution lives in the tail of the conjugate gradient solution vector:
- single scan: `cad.reconstruct_scan.solve_single_scan` solves for the stacked unknown $\begin{bmatrix}c_{\rm act} \\ a_0\end{bmatrix}$ (the `a0` block is `sol[n_c:]`) but only returns `c_hat_full_mk`.
- multi scan: `cad.synthesize_scan.synthesize_scans` solves for the stacked unknown $\begin{bmatrix}c \\ a_1^0 \\ \vdots \\ a_S^0\end{bmatrix}$ (the atmosphere blocks are `sol[n_obs:]`) but only returns `c_hat_full_mk`.

### Physical meaning of the terms
- $P_s^T N_s^{-1} P_s$ (in $A_{cc}$): maps TOD weights back to CMB pixels. With diagonal $N_s$, this is a (weighted) hit-count operator: it upweights pixels that are observed often with low noise.
- $P_s^T N_s^{-1} W_s$ (in $A_{c a_s}$): couples sky pixels to atmosphere pixels through the same TOD samples. This term is what allows the solver to assign parts of the TOD to $c$ vs $a_s^0$ based on how $W_s$ shifts the atmosphere in time.
- $W_s^T N_s^{-1} W_s$ (in $A_{a_s a_s}$): maps TOD weights back to atmosphere pixels through back-advection and bilinear interpolation; it encodes which parts of the atmosphere screen are constrained by the scan geometry and sampling.
- $C_a^{-1}$: regularizes atmosphere modes disfavored by the stationary spectrum (FFT prior on the atmosphere bbox).
- $C_c^{-1}$ (MAP only): Wiener regularization of CMB modes that are weakly constrained by the data.

### Numerical solver (conjugate gradient + diagonal preconditioner)
We never form $A$ explicitly. We apply $A$ as a matrix-free operator using the sparse structure of $P_s$ (nearest-pixel) and $W_s$ (4-point bilinear per sample).

We solve with conjugate gradient using a block-diagonal diagonal preconditioner. For each scan:

- **CMB block**: $\mathrm{diag}(P_s^T N_s^{-1} P_s)$ (hit-count / noise) plus, for MAP, a diagonal
  approximation to $C_c^{-1}$ using `FourierGaussianPrior.apply_Cinv(e0)[0]`.
- **Atmosphere block**: $\mathrm{diag}(W_s^T N_s^{-1} W_s)$ plus a diagonal approximation to $C_a^{-1}$
  using `FourierGaussianPrior.apply_Cinv(e0)[0]`.

This block-diagonal preconditioner is used in both the single-scan and multi-scan conjugate gradient solves.

## 5. Wind estimation (constant wind per scan)

We estimate a constant wind vector $w_s=(w_x,w_y)$ for each scan using only the binned time-ordered data and pointing. We work in a local Cartesian coordinate system:

- $x$: right ascension in degrees (stored on a continuous branch so it can be averaged).
- $y$: declination in degrees.

Let $(x_b(t_i), y_b(t_i))$ be the boresight position at time bin $t_i$, and let $(x_k(t_i), y_k(t_i))$ be the effective detector $k$ position at the same time bin. Define detector offsets relative to the boresight:

$$
\Delta x_{ik} = x_k(t_i) - x_b(t_i),\qquad
\Delta y_{ik} = y_k(t_i) - y_b(t_i).
$$

Let $d_{ik}$ be the binned time-ordered temperature for detector $k$ at time bin $t_i$ (mK).

### Step 1: per-time fit across the focal plane
At each time bin $t_i$, approximate the field across the focal plane by a first-order expansion:

$$
d_{ik} \approx T_i + G_{x,i}\,\Delta x_{ik} + G_{y,i}\,\Delta y_{ik},
$$

where $T_i$ is an offset and $(G_{x,i},G_{y,i})$ are local spatial gradients (mK/deg). We estimate $(T_i,G_{x,i},G_{y,i})$ by weighted least squares over detectors $k$, using weights proportional to the number of raw detectors in each effective detector bin.

### Step 2: linear regression for a constant wind
Under the frozen-screen model, a drifting field $a(x,y,t)=a_0(x-w_x t, y-w_y t)$ evaluated along the boresight trajectory satisfies

$$
\frac{d}{dt}a(x_b(t),y_b(t),t)
=
\left(\frac{dx_b}{dt}-w_x\right)\partial_x a
\;+\;
\left(\frac{dy_b}{dt}-w_y\right)\partial_y a.
$$

Replacing $(a,\partial_x a,\partial_y a)$ by the fitted quantities $(T,G_x,G_y)$ gives

$$
Y_i \equiv \frac{dT_i}{dt} - v_{x,i}\,G_{x,i} - v_{y,i}\,G_{y,i}
= -w_x\,G_{x,i} - w_y\,G_{y,i},
$$

where $v_{x,i}=dx_b/dt$ and $v_{y,i}=dy_b/dt$ are boresight velocities (deg/s). We compute time derivatives using finite differences on the binned arrays and fit $(w_x,w_y)$ by least squares over time bins $i$.

### Detector consistency filtering (implementation detail)
To reduce the impact of bad detectors, the implementation performs a first-pass
affine fit, then regresses each detector time stream against the first-pass
model prediction and drops detectors with inconsistent slope (and optionally low
correlation). The affine fit is then repeated using only the retained detectors
before estimating $(w_x,w_y)$.

## 6. Code–math mapping

- **Pointing operator $P_s$ and valid mask**: `cad.util.pointing_from_pix_index`
- **Atmosphere operator $W_s$ (indices + bilinear weights)**: `cad.util.frozen_screen_bilinear_weights`
- **Atmosphere bbox padding for open boundary**: `cad.util.bbox_pad_for_open_boundary`
- **Stationary FFT priors $C_a, C_c$ and application of $C^{-1}$**: `cad.prior.FourierGaussianPrior.apply_Cinv`
- **Single-scan joint conjugate gradient solve**: `cad.reconstruct_scan.solve_single_scan`
- **Multi-scan joint conjugate gradient solve**: `cad.synthesize_scan.synthesize_scans`
- **Wind estimate (from binned TOD + pointing)**: `cad.wind.estimate_wind_deg_per_s`

## 7. Single-scan conjugate gradient: matrix-free operator and preconditioner

In `cad.reconstruct_scan.solve_single_scan`, we solve a linear system $A x = b$ with conjugate gradient without explicitly forming $A$ (see `cad/src/cad/reconstruct_scan.py`). The unknown is stacked as

$$
x = \begin{bmatrix} c \\ a_0 \end{bmatrix},
$$

where $c \in \mathbb{R}^{n_c}$ is the CMB unknown vector on the chosen CMB basis (ML: hit-pixels; MAP: full CMB bbox) and $a_0 \in \mathbb{R}^{n_{\rm atm}}$ is the per-scan atmosphere reference screen on the atmosphere bbox.

The code defines a function `A_matvec(x)` that returns $A x$ by applying the block normal-equation operator:

- Form time-ordered data contributions on the valid samples:
  - $P c \in \mathbb{R}^{n_{\rm valid}}$ via indexing (`Pc = c_act[act_idx]`).
  - $W a_0 \in \mathbb{R}^{n_{\rm valid}}$ via bilinear sampling (`Wa = W_apply(a0)`).
  - $u \equiv N^{-1}(P c + W a_0) \in \mathbb{R}^{n_{\rm valid}}$ via elementwise multiplication by the diagonal inverse noise (`u = inv_var * (Pc + Wa)`).

- Apply the adjoints to build the block outputs:
  - CMB block: $P^T u \in \mathbb{R}^{n_c}$ via `np.bincount` (`out_c = np.bincount(...)`), plus MAP prior $(1/S)C_c^{-1}c$ when enabled (`out_c = out_c + (1/S)*prior_cmb.apply_Cinv(c_act)`).
  - Atmosphere block: $W^T u + C_a^{-1} a_0 \in \mathbb{R}^{n_{\rm atm}}$ (`out_a = WT_apply(u) + prior_atm.apply_Cinv(a0)`).

This implements the single-scan block operator

$$
A
\begin{bmatrix} c \\ a_0 \end{bmatrix}
=
\begin{bmatrix}
P^T N^{-1}(P c + W a_0) + \mathbf{1}_{\rm MAP}\,\frac{1}{S}C_c^{-1}c \\
W^T N^{-1}(P c + W a_0) + C_a^{-1}a_0
\end{bmatrix},
$$

with all operators applied in a matrix-free way using indexing and 4-point bilinear weights.

In code, we pass this matrix-free matvec to a `LinearOperator` object `A_op` with shape $(n_c+n_{\rm atm},\,n_c+n_{\rm atm})$ (constructed as `A_op = spla.LinearOperator(..., matvec=A_matvec, ...)`), so conjugate gradient can repeatedly apply $x \mapsto A x$ without forming $A$ explicitly.

We also define a diagonal preconditioner $M \approx A^{-1}$ by constructing a `LinearOperator` that applies elementwise division by diagonal approximations (implemented by `Pinv_matvec` and `P_pre = spla.LinearOperator(..., matvec=Pinv_matvec, ...)`). Applied to a stacked vector $x=\begin{bmatrix}x_c\\x_a\end{bmatrix}$:

$$
M x \equiv
\begin{bmatrix}
x_c / \mathrm{diag}_c \\
x_a / \mathrm{diag}_a
\end{bmatrix},
$$

where $\mathrm{diag}_c$ is a diagonal approximation to $\mathrm{diag}(A_{cc})$ (hit-count / noise, plus a MAP prior diagonal when enabled) and $\mathrm{diag}_a$ is a diagonal approximation to $\mathrm{diag}(A_{a_0 a_0}) \simeq \mathrm{diag}(W^T N^{-1} W + C_a^{-1})$.

Finally, we form the right-hand side vector

$$
b=
\begin{bmatrix}
P^T N^{-1} d \\
W^T N^{-1} d
\end{bmatrix},
$$

The implementation stacks this as `rhs = np.concatenate([rhs_c_act, rhs_a])` and calls `spla.cg(A_op, rhs, M=P_pre, ...)` to obtain the stacked solution $x^\star=\begin{bmatrix}c^\star\\a_0^\star\end{bmatrix}$.

