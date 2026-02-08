"""
Core geometry/operator utilities (no I/O).

Conventions (matches the SPT binned NPZ schema):
  - Global pixel indices are integer (ix, iy) on an RA/Dec-degree grid.
  - Within a bbox with sizes (nx, ny), we flatten pixels as:
      pix = iy + ix * ny
    where ix is axis-0 and iy is axis-1 of the reshaped (nx, ny) array.
"""

from __future__ import annotations

import numpy as np

from .map import BBox


def pointing_from_pix_index(*, pix_index: np.ndarray, tod_mk: np.ndarray, bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a local pointing matrix (flattened pixel indices) and a valid mask.

    Args:
      pix_index: (n_t, n_det, 2) int64 global (ix,iy).
      tod_mk: (n_t, n_det) float, mK.
      bbox: bbox defining the local grid.

    Returns:
      pointing_matrix: (n_t, n_det) int64, -1 invalid, else pix = iy + ix*ny in [0, nx*ny).
      valid_mask: (n_t, n_det) bool, finite TOD and inside bbox.
    """
    pix_index = np.asarray(pix_index, dtype=np.int64)
    tod_mk = np.asarray(tod_mk)
    if pix_index.shape[:2] != tod_mk.shape or pix_index.shape[-1] != 2:
        raise ValueError("pix_index must have shape (n_t,n_det,2) matching tod_mk.")

    ix = pix_index[..., 0] - int(bbox.ix0)
    iy = pix_index[..., 1] - int(bbox.iy0)
    inside = (ix >= 0) & (ix < int(bbox.nx)) & (iy >= 0) & (iy < int(bbox.ny))
    finite = np.isfinite(tod_mk)
    valid = inside & finite

    pm = np.full(tod_mk.shape, -1, dtype=np.int64)
    ny = int(bbox.ny)
    pm[valid] = (iy[valid] + ix[valid] * ny).astype(np.int64, copy=False)
    return pm, valid


def observed_pixel_index_set(
    *,
    pointing_matrices: list[np.ndarray],
    valid_masks: list[np.ndarray],
    n_pix: int,
    min_hits_per_pix: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Observed pixel set (union across scans) above a hit-count threshold.

    Returns:
      obs_pix_global: (n_obs,) int64, indices in [0,n_pix)
      global_to_obs: (n_pix,) int64, -1 for unobserved, else 0..n_obs-1
    """
    if len(pointing_matrices) != len(valid_masks):
        raise ValueError("pointing_matrices and valid_masks must have same length.")
    if not pointing_matrices:
        raise ValueError("Must provide at least one scan.")

    hit_count = np.zeros((int(n_pix),), dtype=np.int64)
    for pm, vm in zip(pointing_matrices, valid_masks, strict=True):
        pm = np.asarray(pm, dtype=np.int64)
        vm = np.asarray(vm, dtype=bool)
        pix = pm[vm]
        if pix.size == 0:
            continue
        hit_count += np.bincount(pix, minlength=int(n_pix)).astype(np.int64, copy=False)

    hit = hit_count >= int(min_hits_per_pix)
    obs_pix_global = np.nonzero(hit)[0].astype(np.int64)
    global_to_obs = -np.ones((int(n_pix),), dtype=np.int64)
    global_to_obs[obs_pix_global] = np.arange(obs_pix_global.size, dtype=np.int64)
    return obs_pix_global, global_to_obs

    
def noise_std_eff_mk_from_counts(
    *,
    eff_counts: np.ndarray,
    bin_sec: float,
    sample_rate_hz: float,
    noise_per_raw_detector_per_153hz_sample_mk: float = 10.0,
) -> np.ndarray:
    """
    White-noise model for binned effective-detector TOD.

    Assumptions / equations:

    - Per raw detector, per raw sample: white noise with std σ_raw [mK].
    - Let n_samp = round(bin_sec * sample_rate_hz) be the number of raw samples
      averaged into one time bin.
    - Bin-averaged noise for one raw detector:
        σ_raw_bin = σ_raw / sqrt(n_samp)
    - Effective detector d averages eff_counts[d] raw detectors:
        σ_eff[d] = σ_raw_bin / sqrt(eff_counts[d])

    Args:
      eff_counts: (n_eff,) number of raw detectors per effective detector.
      bin_sec: bin duration in seconds.
      sample_rate_hz: raw sampling rate (≈152.6 Hz for SPT3G).
      noise_per_raw_detector_per_153hz_sample_mk: scalar, mK per raw-detector per raw sample.

    Returns:
      noise_std_eff_mk: (n_eff,) mK, per-effective-detector bin noise std.
    """
    eff_counts = np.asarray(eff_counts, dtype=np.float64).reshape(-1)
    if not bool(np.all(np.isfinite(eff_counts))) or not bool(np.all(eff_counts > 0)):
        raise ValueError("eff_counts must be finite and > 0.")
    n_samp_per_bin = int(round(float(bin_sec) * float(sample_rate_hz)))
    if n_samp_per_bin <= 0:
        raise ValueError("Invalid binning: bin_sec*sample_rate_hz must be positive.")
    sigma_raw_bin_mk = float(noise_per_raw_detector_per_153hz_sample_mk) / np.sqrt(float(n_samp_per_bin))
    return (sigma_raw_bin_mk / np.sqrt(eff_counts)).astype(np.float64, copy=False)


def bbox_pad_for_open_boundary(
    *,
    bbox_obs: BBox,
    scans_pix_index: list[np.ndarray],
    scans_tod_mk: list[np.ndarray],
    scans_t_s: list[np.ndarray],
    winds_deg_per_s: list[tuple[float, float]],
    pixel_size_deg: float,
) -> BBox:
    """
    Pad `bbox_obs` so open-boundary bilinear advection stays in-bounds.

    This uses the *global* pixel indices and back-advection source corners with
    the reference time at scan start (t0 = t_s[0]):

      x_src(t) = i_x(t) - (w_x / pixel_size_deg) * (t - t0)
      y_src(t) = i_y(t) - (w_y / pixel_size_deg) * (t - t0)

    where:
      - t0 = t_s[0]
      - (i_x, i_y) are global integer pixel indices for each sample
      - pixel_size_deg converts deg/s to pixels/s.

    Returns:
      bbox_pad: global bbox that contains all bilinear corners (floor/ceil) needed by W.
    """
    if len(scans_pix_index) != len(scans_tod_mk) or len(scans_pix_index) != len(scans_t_s) or len(scans_pix_index) != len(winds_deg_per_s):
        raise ValueError("All scan lists must have the same length.")
    if not scans_pix_index:
        raise ValueError("Must provide at least one scan.")

    ix0, ix1, iy0, iy1 = int(bbox_obs.ix0), int(bbox_obs.ix1), int(bbox_obs.iy0), int(bbox_obs.iy1)
    pix_deg = float(pixel_size_deg)

    for pix_index, tod_mk, t_s, wind in zip(scans_pix_index, scans_tod_mk, scans_t_s, winds_deg_per_s, strict=True):
        pix_index = np.asarray(pix_index, dtype=np.int64)
        tod_mk = np.asarray(tod_mk)
        t_s = np.asarray(t_s, dtype=np.float64)
        if pix_index.shape[:2] != tod_mk.shape or pix_index.shape[-1] != 2:
            raise ValueError("Shape mismatch: pix_index must match tod_mk and have last dim 2.")
        if t_s.shape != (tod_mk.shape[0],):
            raise ValueError("t_s must have shape (n_t,).")

        ok = np.isfinite(tod_mk)
        if not bool(np.any(ok)):
            continue
        ij = pix_index[ok]  # (n_valid,2)
        ix = ij[:, 0].astype(np.float64, copy=False)
        iy = ij[:, 1].astype(np.float64, copy=False)

        t_idx, _ = np.where(ok)
        t0 = float(t_s[0])
        dt = t_s[t_idx].astype(np.float64, copy=False) - t0

        w_x = float(wind[0])
        w_y = float(wind[1])
        dx_pix = w_x * dt / pix_deg
        dy_pix = w_y * dt / pix_deg
        x_src = ix - dx_pix
        y_src = iy - dy_pix

        x0 = np.floor(x_src).astype(np.int64)
        y0 = np.floor(y_src).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1

        ix0 = min(ix0, int(np.min(x0)))
        ix1 = max(ix1, int(np.max(x1)))
        iy0 = min(iy0, int(np.min(y0)))
        iy1 = max(iy1, int(np.max(y1)))

    return BBox(ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)


def frozen_screen_bilinear_weights(
    *,
    pointing_matrix: np.ndarray,
    valid_mask: np.ndarray,
    bbox_cmb: BBox,
    bbox_atm: BBox,
    wind_deg_per_s: tuple[float, float],
    t_s: np.ndarray,
    pixel_size_deg: float,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinear frozen-screen weights for the atmosphere operator W (open boundary).

    We model the per-scan atmosphere as a frozen screen a0 at reference time
    t0 = t_s[0] advected by a constant wind w = (w_x, w_y) (deg/s in the same
    RA/Dec-degree basis as the pixel grid).

    For each valid sample at time t with *global* hit pixel (i_x, i_y), we sample
    the reference-time screen at:

      x_src(t) = i_x - (w_x / pixel_size_deg) * (t - t0)
      y_src(t) = i_y - (w_y / pixel_size_deg) * (t - t0)

    where t0 = t_s[0].

    We bilinear-interpolate a0(x_src, y_src) on the atmosphere grid,
    returning per-sample corner indices and weights so that:

      (W a0)[i] = sum_{j=0..3} w4[i,j] * a0[idx4[i,j]]

    Args:
      pointing_matrix: (n_t,n_det) int64, pixel_index = iy + ix*ny_cmb, -1 invalid.
      valid_mask: (n_t,n_det) bool.
      bbox_cmb: defines the local CMB grid (for decoding global ix/iy).
      bbox_atm: defines the local atmosphere grid (for building idx4).
      wind_deg_per_s: (w_x, w_y) in deg/s, in the RA/Dec-degree basis.
      t_s: (n_t,) seconds.
      pixel_size_deg: scalar pixel size in degrees.
      strict: if True, raise when advection goes out of bounds.

    Returns:
      idx4: (n_valid, 4) int64 corner indices into a0.ravel() (pix = iy + ix*ny_atm)
      w4: (n_valid, 4) float64 bilinear weights
    """
    pm = np.asarray(pointing_matrix, dtype=np.int64)
    vm = np.asarray(valid_mask, dtype=bool)
    if pm.shape != vm.shape:
        raise ValueError("pointing_matrix and valid_mask must have the same shape.")

    t_idx, det_idx = np.where(vm)
    t_idx = t_idx.astype(np.int64, copy=False)
    det_idx = det_idx.astype(np.int64, copy=False)
    pix = pm[vm].astype(np.int64, copy=False)  # (n_valid,)
    if pix.size == 0:
        raise ValueError("No valid samples for advection.")

    ny_cmb = int(bbox_cmb.ny)
    iy_cmb = pix % ny_cmb
    ix_cmb = pix // ny_cmb

    # Convert CMB local pixel hits to *global* pixel indices.
    ix_g = ix_cmb.astype(np.float64) + float(bbox_cmb.ix0)
    iy_g = iy_cmb.astype(np.float64) + float(bbox_cmb.iy0)

    t_full = np.asarray(t_s, dtype=np.float64)
    t = t_full[t_idx]  # (n_valid,)
    t0 = float(t_full[0])
    dt = t - t0

    w_x, w_y = float(wind_deg_per_s[0]), float(wind_deg_per_s[1])
    dx_pix = w_x * dt / float(pixel_size_deg)
    dy_pix = w_y * dt / float(pixel_size_deg)

    x_src_g = ix_g - dx_pix
    y_src_g = iy_g - dy_pix

    # Convert to atmosphere-local coordinates.
    x_src = x_src_g - float(bbox_atm.ix0)
    y_src = y_src_g - float(bbox_atm.iy0)

    x0 = np.floor(x_src).astype(np.int64)
    y0 = np.floor(y_src).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    nx_atm, ny_atm = int(bbox_atm.nx), int(bbox_atm.ny)
    inside = (x0 >= 0) & (x1 < nx_atm) & (y0 >= 0) & (y1 < ny_atm)
    if strict and not bool(np.all(inside)):
        bad = np.argwhere(~inside)
        i0 = int(bad[0][0])
        raise ValueError(
            "Open-boundary bilinear advection goes out of bounds.\n"
            f"First bad valid sample idx={i0}: t_idx={int(t_idx[i0])}, det_idx={int(det_idx[i0])}, "
            f"x_src_local={x_src[i0]:.3f}, y_src_local={y_src[i0]:.3f}, "
            f"x0={int(x0[i0])}, x1={int(x1[i0])}, y0={int(y0[i0])}, y1={int(y1[i0])}, "
            f"nx_atm={nx_atm}, ny_atm={ny_atm}"
        )

    wx1 = x_src - x0.astype(np.float64)
    wy1 = y_src - y0.astype(np.float64)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    idx4 = np.stack(
        [
            y0 + x0 * ny_atm,
            y0 + x1 * ny_atm,
            y1 + x0 * ny_atm,
            y1 + x1 * ny_atm,
        ],
        axis=1,
    ).astype(np.int64, copy=False)
    w4 = np.stack([wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1], axis=1).astype(np.float64, copy=False)
    return idx4, w4

