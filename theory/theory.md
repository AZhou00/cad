## Model (per scan)

For scan $s \in \{1,\dots,S\}$ we model the binned TOD as

$$
d_s = P_s\,c + W_s\,a_s^0 + n_s
$$

- $d_s$: $(n_t,n_{\rm det})$ TOD in mK (after binning).
- $c$: CMB pixel map on the **CMB bbox** (global grid, RA/Dec degrees).
- $a_s^0$: atmosphere screen at reference time $t_0$ on the **atmosphere bbox** (a padded global grid).
- $n_s \sim \mathcal{N}(0, N_s)$ with diagonal $N_s$ (white, per-detector).
- $P_s$: pointing operator (nearest-pixel) from `pix_index` (global $(i_x,i_y)$ per sample).
- $W_s$: frozen-screen advection operator (bilinear interpolation; open boundary).

This matches the theory notes around Eq. (105) in `latex_cmb_atmosphere/main.tex`.

## Frozen-screen advection ($W_s$)

Let the wind for scan $s$ be $w_s=(w_x,w_y)$ in deg/s in the same RA/Dec-degree basis as `pix_index`.

For each valid TOD sample at time $t$ with **global** hit pixel $(i_x,i_y)$, the atmosphere reference screen is sampled at

$$
x_{\rm src} = i_x - \frac{w_x}{\Delta_{\rm pix}}(t-t_0),\qquad
y_{\rm src} = i_y - \frac{w_y}{\Delta_{\rm pix}}(t-t_0),
$$

where $\Delta_{\rm pix}=$ `pixel_size_deg`. We evaluate $a_s^0(x_{\rm src},y_{\rm src})$ by bilinear interpolation on the atmosphere grid (4 corners and weights). Samples whose bilinear corners fall outside the atmosphere bbox are **not allowed** (strict open boundary).

### Choice of $t_0$

In the infinite-plane model with a stationary prior for $a_s^0$, changing $t_0$ is only a re-parameterization of the nuisance field and does not change the marginal posterior for $c$.

In the implementation we must still choose a reference time on a finite padded atmosphere bbox. We use the simple convention:

$$
t_0 = t_s[0].
$$

This is a coordinate choice for the definition of $a_s^0$; it affects how the padded atmosphere bbox is positioned on the global pixel grid, but does not introduce additional information.

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
The pixel grid is uniform in RA/Dec degrees, but physical length in RA is smaller by $\cos\delta$.
We therefore define Fourier radius using

$$
\ell = \sqrt{\left(\frac{k_x}{\cos\delta}\right)^2 + k_y^2},
$$

so that isotropic spectra on the sky remain isotropic in physical space. The FFT prior
normalization uses $dx\,dy = (\Delta_{\rm pix}^2)\cos\delta$, where $\Delta_{\rm pix}$ is in radians.
The pointing grid itself remains unchanged; only the prior's spectral weighting is anisotropic.

## Estimators

- **ML**: maximize likelihood w.r.t. $(c,\{a_s^0\})$ (equivalently a flat prior on $c$; $C_c^{-1}=0$).
- **MAP**: maximize posterior w.r.t. $(c,\{a_s^0\})$ using both priors.

In the combined multi-scan solve, $c$ is a single shared unknown so the MAP system includes $C_c^{-1}$ once. In the per-scan diagnostic solve (solving each scan separately on the same CMB grid), we use $(1/S)C_c^{-1}$ so that summing diagnostics across scans corresponds to a single $C_c^{-1}$ term.

## Linear system solved (joint CG)

We solve the augmented normal equations for the unknown vector

$$
x = \begin{bmatrix} c \\ a_1^0 \\ \vdots \\ a_S^0 \end{bmatrix}
$$

without explicitly forming the marginal covariance $\tilde N_s = N_s + W_s C_a W_s^T$ (cf. Eq. (322) in the theory).

For each scan $s$, the blocks are

$$
\begin{aligned}
A_{cc} &+= P_s^T N_s^{-1} P_s + \mathbf{1}_{\rm MAP} \, C_c^{-1} \\\\
A_{c a_s} &= P_s^T N_s^{-1} W_s \\\\
A_{a_s c} &= W_s^T N_s^{-1} P_s \\\\
A_{a_s a_s} &= W_s^T N_s^{-1} W_s + C_a^{-1}
\end{aligned}
$$

and the right-hand side is

$$
b_c = \sum_s P_s^T N_s^{-1} d_s,\qquad
b_{a_s} = W_s^T N_s^{-1} d_s.
$$

The multi-scan solve uses Conjugate Gradient with a diagonal preconditioner. For each scan:

- **CMB block**: $\mathrm{diag}(P_s^T N_s^{-1} P_s)$ (hit-count / noise) plus, for MAP, a diagonal
  approximation to $C_c^{-1}$ using `FourierGaussianPrior.apply_Cinv(e0)[0]`.
- **Atmosphere block**: $\mathrm{diag}(W_s^T N_s^{-1} W_s)$ plus a diagonal approximation to $C_a^{-1}$
  using `FourierGaussianPrior.apply_Cinv(e0)[0]`.

This block-diagonal preconditioner is used in both the single-scan and multi-scan CG solves.

## Grids, masks, and monopole

- **CMB bbox**: tight union bbox over observed pixels across scans (no padding).
- **Atmosphere bbox**: padded bbox large enough to keep open-boundary bilinear advection in-bounds. The combined solve uses a shared padded bbox; per-scan diagnostic solves may use per-scan padded bboxes.
- **Observed pixel basis**: for ML we solve for pixels hit at least `min_hits_per_pix` times; for MAP we solve the full bbox grid.
- **Monopole**: the solution has a near-degeneracy between the mean of $c$ and the mean of $a_s^0$; we fix a subtract the mean of the reconstructed $c$ map after solving.

