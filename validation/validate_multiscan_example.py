#!/usr/bin/env python3
"""
Toy multi-scan validation for `cad.synthesize_scans`.

Checks:
  - ML solve across scans with different winds improves recovery of c.
"""

from __future__ import annotations

import numpy as np

import cad


def main() -> None:
    rng = np.random.default_rng(1)

    pixel_size_deg = 1.0
    n_ell_bins = 32

    # Two scans, same pointing geometry, different winds.
    winds = [(0.25, -0.05), (-0.10, 0.20)]
    n_scans = len(winds)

    n_t = 40
    n_det = 6
    t_s = np.linspace(0.0, 20.0, n_t).astype(np.float64)

    # Scan geometry: boresight sweeps in +ix; detectors offset in iy.
    ix_base = np.linspace(1, 6, n_t).round().astype(np.int64)
    iy_base = np.full((n_t,), 3, dtype=np.int64)
    det_dy = np.arange(n_det, dtype=np.int64) - (n_det // 2)

    pix_index0 = np.zeros((n_t, n_det, 2), dtype=np.int64)
    pix_index0[..., 0] = ix_base[:, None]
    pix_index0[..., 1] = iy_base[:, None] + det_dy[None, :]

    # BBox from hits (matches cad's behavior).
    valid_all = np.ones((n_t, n_det), dtype=bool)
    bbox_cmb = cad.map.scan_bbox_from_pix_index(pix_index=pix_index0, valid_mask=valid_all)

    scans_pix_index = [pix_index0.copy() for _ in range(n_scans)]
    scans_t_s = [t_s.copy() for _ in range(n_scans)]

    # Truth CMB on the bbox grid implied by pointing.
    n_pix_cmb = int(bbox_cmb.nx * bbox_cmb.ny)
    c_true = rng.normal(size=(n_pix_cmb,)).astype(np.float64)

    # Synthetic TOD: d = P c + W a + n, using cad's operators.
    sigma_det = 0.1 * np.ones((n_det,), dtype=np.float64)
    cl_atm = np.full((n_ell_bins,), 10.0, dtype=np.float64)
    cl_cmb = np.full((n_ell_bins,), 1.0, dtype=np.float64)

    # Atmosphere bbox padded for open-boundary advection.
    bbox_atm = cad.util.bbox_pad_for_open_boundary(
        bbox_obs=bbox_cmb,
        scans_pix_index=scans_pix_index,
        scans_tod_mk=[np.zeros((n_t, n_det), dtype=np.float64) for _ in range(n_scans)],
        scans_t_s=scans_t_s,
        winds_deg_per_s=winds,
        pixel_size_deg=float(pixel_size_deg),
    )
    nx_a, ny_a = int(bbox_atm.nx), int(bbox_atm.ny)
    n_pix_atm = int(nx_a * ny_a)

    scans_tod = []
    for si in range(n_scans):
        a_true = rng.normal(size=(n_pix_atm,)).astype(np.float64)
        # Build P/W operators for this scan.
        pm, vm = cad.util.pointing_from_pix_index(pix_index=scans_pix_index[si], tod_mk=np.zeros((n_t, n_det)), bbox=bbox_cmb)
        pix = pm[vm].astype(np.int64)
        Pc = c_true[pix]
        idx4, w4 = cad.util.frozen_screen_bilinear_weights(
            pointing_matrix=pm,
            valid_mask=vm,
            bbox_cmb=bbox_cmb,
            bbox_atm=bbox_atm,
            wind_deg_per_s=winds[si],
            t_s=t_s,
            pixel_size_deg=float(pixel_size_deg),
            strict=True,
        )
        Wa = np.sum(w4 * a_true[idx4], axis=1)
        d_valid = Pc + Wa

        tod = np.full((n_t, n_det), np.nan, dtype=np.float64)
        tod[vm] = d_valid + rng.normal(scale=sigma_det[np.where(vm)[1]])
        scans_tod.append(tod)

    # ML solve across scans and compare on hit pixels (mean removed).
    sol = cad.synthesize_scans(
        scans_tod_mk=scans_tod,
        scans_pix_index=scans_pix_index,
        scans_t_s=scans_t_s,
        winds_deg_per_s=winds,
        scans_noise_std_det_mk=[sigma_det for _ in range(n_scans)],
        pixel_size_deg=pixel_size_deg,
        cl_atm_bins_mk2=cl_atm,
        cl_cmb_bins_mk2=cl_cmb,
        estimator_mode="ML",
        cg_tol=1e-6,
        cg_maxiter=800,
    )

    c_hat = np.asarray(sol.c_hat_full_mk, dtype=np.float64)
    # Compare on hit pixels (gauge matched).
    pm0, vm0 = cad.util.pointing_from_pix_index(pix_index=pix_index0, tod_mk=np.zeros((n_t, n_det)), bbox=bbox_cmb)
    hits = np.bincount(pm0[vm0], minlength=n_pix_cmb)
    m = hits > 0
    ct = c_true[m] - float(np.mean(c_true[m]))
    ch = c_hat[m] - float(np.mean(c_hat[m]))
    corr = float(np.corrcoef(ct, ch)[0, 1])
    rel = float(np.linalg.norm(ch - ct) / np.linalg.norm(ct))
    print(f"[multi] hit_pix={int(np.sum(m))}/{n_pix_cmb} corr={corr:.3f} rel_err={rel:.3e}")


if __name__ == "__main__":
    main()

