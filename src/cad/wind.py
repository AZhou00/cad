"""
Estimate a constant wind angular-velocity vector from one scan's binned arrays.

This is an NPZ-only implementation (no SPT3G dependency).

Coordinate basis
----------------
We work in a local Cartesian x/y plane. By default:
  - x = RA   [deg]  (on the continuous branch used by the extractor, typically [180,540))
  - y = Dec  [deg]
If `physical_degree=True`, we instead use:
  - x = RA*cos(Dec_ref)  [deg_physical]
  - y = Dec              [deg_physical]
with Dec_ref = median boresight Dec over the scan.

Math summary (single scan)
--------------------------
We approximate the atmosphere as a frozen 2D screen advected at a constant wind
(w_x,w_y) during a scan. Let the drifting atmosphere be:
  a(x,y,t) = a_0(x - w_x t, y - w_y t).
Along the boresight trajectory (x_bore(t), y_bore(t)), the time derivative obeys:
  d/dt a(x_bore(t),t) = (v_x - w_x) * ∂_x a + (v_y - w_y) * ∂_y a,
where v_x = dx_bore/dt and v_y = dy_bore/dt are the scan velocities in the same
x/y basis.

Algorithm
---------
Eq. (1) Plane fit (per time bin i):
  eff_tod_mk[i,d] ≈ T[i] + dT_dx[i] * Δx[i,d] + dT_dy[i] * Δy[i,d],
where Δx,Δy are detector sky offsets relative to boresight at time i.
We solve by weighted least squares with weights proportional to `eff_counts[d]`.

Eq. (2) Wind regression (over time bins):
Define:
  Y(t) = dT/dt - v_x(t) * dT_dx(t) - v_y(t) * dT_dy(t) = - w_x * dT_dx(t) - w_y * dT_dy(t).
We fit Y ≈ -[dT_dx, dT_dy]·w by least squares and return (w_x,w_y).

Units
-----
- T in mK, gradients in mK/deg, velocities in deg/s, so advection terms are mK/s.
"""

from __future__ import annotations

import numpy as np


def estimate_wind_deg_per_s(
    *,
    eff_tod_mk: np.ndarray,
    eff_pos_deg: np.ndarray,
    boresight_pos_deg: np.ndarray,
    t_s: np.ndarray,
    eff_counts: np.ndarray,
    physical_degree: bool = False,
    bad_pixel_min_slope: float = 0.7,
    bad_pixel_min_corr: float | None = None,
    bad_pixel_min_samples: int = 10,
) -> tuple[float, float, dict, np.ndarray]:
    """
    Estimate the constant wind components (w_x, w_y) in deg/s.

    Args:
      eff_tod_mk: (n_t, n_eff) in mK.
      eff_pos_deg: (n_t, n_eff, 2) with columns (RA, Dec) in deg.
      boresight_pos_deg: (n_t, 2) with columns (RA, Dec) in deg.
      t_s: (n_t,) time in seconds.
      eff_counts: (n_eff,) weights, proportional to inverse variance.
      physical_degree:
        - False: x=RA (deg), y=Dec (deg) so wind is in deg_RA/s and deg_Dec/s.
        - True: x=RA*cos(Dec_ref), y=Dec so wind is in physical small-angle degrees/s.
      bad_pixel_min_slope: mark a detector as good only if its regression slope vs the
        first-pass plane prediction exceeds this threshold.
      bad_pixel_min_corr: optional minimum Pearson correlation coefficient vs the
        first-pass plane prediction. If None, no correlation cut is applied.
      bad_pixel_min_samples: require at least this many finite time bins to evaluate a
        detector.

    Returns:
      w_x, w_y: floats in deg/s (basis depends on physical_degree).
      diagnostics: dict with fit metadata (RMS residual, Dec_ref, etc.).
      mask_good: (n_eff,) boolean array, True for good detectors, False for dropped detectors.
    """
    eff_tod_mk = np.asarray(eff_tod_mk, dtype=np.float64)
    eff_pos_deg = np.asarray(eff_pos_deg, dtype=np.float64)
    bore_pos_deg = np.asarray(boresight_pos_deg, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    eff_counts = np.asarray(eff_counts, dtype=np.float64)

    if eff_pos_deg.shape[:2] != eff_tod_mk.shape or eff_pos_deg.shape[-1] != 2:
        raise ValueError("eff_pos_deg must have shape (n_t, n_eff, 2) matching eff_tod_mk.")
    if bore_pos_deg.shape != (eff_tod_mk.shape[0], 2):
        raise ValueError("boresight_pos_deg must have shape (n_t, 2).")
    if t_s.shape != (eff_tod_mk.shape[0],):
        raise ValueError("t_s must have shape (n_t,).")
    if eff_counts.shape != (eff_tod_mk.shape[1],):
        raise ValueError("eff_counts must have shape (n_eff,).")

    n_t, n_d = eff_tod_mk.shape
    if int(n_t) < 3:
        raise ValueError("Need at least 3 time bins to estimate wind.")

    # Reference Dec for optional RA->x scaling.
    dec_ref_deg = float(np.median(bore_pos_deg[:, 1]))
    cos_dec_ref = float(np.cos(np.deg2rad(dec_ref_deg)))
    x_scale = cos_dec_ref if bool(physical_degree) else 1.0

    # Detector offsets relative to boresight (degrees).
    # shapes: (n_t, n_d)
    dRA_deg = (eff_pos_deg[:, :, 0] - bore_pos_deg[:, None, 0]) * x_scale
    dDEC_deg = eff_pos_deg[:, :, 1] - bore_pos_deg[:, None, 1]

    # Eq. (1): per-time plane fit for θ(t) = [T(t), dT_dx(t), dT_dy(t)] via WLS.
    # Shapes:
    #   T, dT_dx, dT_dy: (n_t,)
    #   w_d, sw: (n_d,)
    w_d = eff_counts  # (n_d,) weights ~ inverse variance
    if not bool(np.all(np.isfinite(w_d))):
        raise ValueError("eff_counts must be finite.")
    if not bool(np.all(w_d >= 0)):
        raise ValueError("eff_counts must be non-negative.")
    sw = np.sqrt(w_d)  # (n_d,)

    T = np.full((n_t,), np.nan, dtype=np.float64)
    dT_dx = np.full((n_t,), np.nan, dtype=np.float64)
    dT_dy = np.full((n_t,), np.nan, dtype=np.float64)
    ones = np.ones((n_d,), dtype=np.float64)

    # First pass: fit planes using all detectors and store plane predictions.
    # model_pred: (n_t, n_d)
    model_pred = np.full((n_t, n_d), np.nan, dtype=np.float64)

    for ti in range(n_t):
        Td = eff_tod_mk[ti]  # (n_d,)
        m = np.isfinite(Td)
        if m.sum() < 3:
            continue
        P = np.stack([ones, dRA_deg[ti], dDEC_deg[ti]], axis=1)  # (n_d, 3)
        Pw = P[m] * sw[m, None]  # (n_m, 3)
        yw = Td[m] * sw[m]  # (n_m,)
        th, _, _, _ = np.linalg.lstsq(Pw, yw, rcond=None)
        model_pred[ti] = P @ th

    # Bad-pixel detection: regress each detector timestream against the first-pass plane prediction.
    # This mainly catches sign flips and severe gain issues.
    alpha_k = np.full((n_d,), np.nan, dtype=np.float64)
    corr_k = np.full((n_d,), np.nan, dtype=np.float64)

    for k in range(n_d):
        y = eff_tod_mk[:, k]  # (n_t,)
        x = model_pred[:, k]  # (n_t,)
        m = np.isfinite(y) & np.isfinite(x)
        if int(np.sum(m)) < int(bad_pixel_min_samples):
            continue
        xm = x[m]
        ym = y[m]

        A = np.vstack([xm, np.ones(int(xm.size))]).T  # (n_samp, 2)
        th, _, _, _ = np.linalg.lstsq(A, ym, rcond=None)
        alpha_k[k] = float(th[0])

        # Pearson correlation (diagnostic / optional cut).
        sx = float(np.std(xm))
        sy = float(np.std(ym))
        if sx > 0 and sy > 0:
            corr_k[k] = float(np.corrcoef(xm, ym)[0, 1])

    mask_good = np.isfinite(alpha_k) & (alpha_k > float(bad_pixel_min_slope))
    if bad_pixel_min_corr is not None:
        mask_good = mask_good & np.isfinite(corr_k) & (corr_k > float(bad_pixel_min_corr))

    # Second pass: re-fit planes using only good detectors.
    for ti in range(n_t):
        Td = eff_tod_mk[ti]
        m = np.isfinite(Td) & mask_good  # (n_d,)
        if m.sum() < 3:
            continue
        P = np.stack([ones, dRA_deg[ti], dDEC_deg[ti]], axis=1)
        Pw = P[m] * sw[m, None]
        yw = Td[m] * sw[m]
        th, _, _, _ = np.linalg.lstsq(Pw, yw, rcond=None)
        T[ti] = float(th[0])
        dT_dx[ti] = float(th[1])
        dT_dy[ti] = float(th[2])

    # Eq (2): time derivatives and scan velocity.
    dT_dt = np.gradient(T, t_s)
    v_x = np.gradient(bore_pos_deg[:, 0] * x_scale, t_s)  # d(RA)/dt (scaled if enabled)
    v_y = np.gradient(bore_pos_deg[:, 1], t_s)  # d(Dec)/dt

    # Eq (3): linear regression for constant wind in the same x/y basis:
    #   Y(t) = w_x*(∂xT)(t) + w_y*(∂yT)(t)
    Y = dT_dt - v_x * dT_dx - v_y * dT_dy  # (n_t,)
    M = np.stack([dT_dx, dT_dy], axis=1)  # (n_t, 2)
    mfit = np.isfinite(Y) & np.isfinite(M).all(axis=1)
    if int(np.sum(mfit)) < 3:
        raise ValueError("Not enough finite time bins to fit wind.")

    w_hat, _, _, _ = np.linalg.lstsq(M[mfit], Y[mfit], rcond=None)  # (2,)

    w_x, w_y = -float(w_hat[0]), -float(w_hat[1])

    Yhat = M[mfit] @ w_hat
    resid = Y[mfit] - Yhat
    resid_rms = float(np.sqrt(np.mean(resid**2)))

    # Wind-fit uncertainty (simple homoscedastic OLS estimate).
    # Cov(w_hat) ≈ σ² (MᵀM)⁻¹ with σ² = RSS/(n-2). This is a diagnostic only.
    w_cov = None
    w_sigma_x = np.nan
    w_sigma_y = np.nan
    w_snr_mag = np.nan
    cond_MtM = np.nan
    try:
        Mf = M[mfit]  # (n_fit,2)
        MtM = Mf.T @ Mf  # (2,2)
        cond_MtM = float(np.linalg.cond(MtM))
        n_fit = int(Mf.shape[0])
        if n_fit > 2:
            rss = float(np.sum(resid**2))
            sigma2 = rss / float(n_fit - 2)
            w_cov = (sigma2 * np.linalg.inv(MtM)).astype(np.float64)
            w_sigma = np.sqrt(np.diag(w_cov))
            w_sigma_x = float(w_sigma[0])
            w_sigma_y = float(w_sigma[1])
            w_mag = float(np.hypot(w_x, w_y))
            denom = float(np.hypot(w_sigma_x, w_sigma_y))
            w_snr_mag = (w_mag / denom) if denom > 0 else np.nan
    except Exception:
        # Keep uncertainty diagnostics as NaN if numerical issues occur.
        pass

    diagnostics = dict(
        mode=("physical(E/N)" if physical_degree else "RA/Dec"),
        bad_pixel_min_slope=float(bad_pixel_min_slope),
        bad_pixel_min_corr=(None if bad_pixel_min_corr is None else float(bad_pixel_min_corr)),
        bad_pixel_min_samples=int(bad_pixel_min_samples),
        n_t=int(n_t),
        n_eff=int(n_d),
        n_good_pixels=int(mask_good.sum()),
        fit_samples=int(np.sum(mfit)),
        dec_ref_deg=float(dec_ref_deg),
        cos_dec_ref=float(cos_dec_ref),
        resid_rms_mk_per_s=float(resid_rms),
        wind_sigma_x_deg_per_s=float(w_sigma_x),
        wind_sigma_y_deg_per_s=float(w_sigma_y),
        wind_snr_mag=float(w_snr_mag),
        wind_cond_MtM=float(cond_MtM),
        w_mag_deg_per_s=float(np.hypot(w_x, w_y)),
    )
    return w_x, w_y, diagnostics, mask_good

