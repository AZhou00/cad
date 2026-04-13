"""
Tests for the parallel solve path: layout, scan artifacts, synthesis.

Run on compute node nid001097: ssh nid001097, then
  module load conda; module load gpu/1.0; conda activate jax;
  cd <repo_root>; python -m pytest cad/test/test_parallel.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cad.parallel_solve.layout import GlobalLayout, load_layout
from cad.parallel_solve.artifact_io import load_scan_artifact
from cad.parallel_solve.synthesize_scan import (
    run_synthesis,
    _local_flat_pixels_to_super_flat,
    _union_super_plate_from_layouts,
)


# --- Layout ---

def test_load_layout_shape_and_props():
    """load_layout returns GlobalLayout with correct n_scans, n_obs, n_pix."""
    layout_npz = Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data/ra0hdec-59.75/101706388/layout.npz")
    if not layout_npz.exists():
        pytest.skip("layout.npz not found")
    layout = load_layout(layout_npz)
    assert layout.n_scans >= 1
    assert layout.n_obs == layout.obs_pix_global.size
    assert layout.n_pix == layout.nx * layout.ny
    assert layout.global_to_obs.size == layout.n_pix
    assert layout.obs_pix_global.dtype == np.int64
    assert layout.global_to_obs.dtype == np.int64


# --- Scan artifact load ---

def test_load_scan_artifact_keys_and_shapes():
    """load_scan_artifact returns cov_inv, Pt_Ninv_d with correct shapes."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = Path(f.name)
    try:
        n_obs_scan = 4
        n_ell = 5
        np.savez_compressed(
            path,
            obs_pix_global_scan=np.arange(n_obs_scan, dtype=np.int64),
            c_hat_scan_obs=np.zeros((n_obs_scan,), dtype=np.float64),
            cov_inv=np.eye(n_obs_scan, dtype=np.float64),
            Pt_Ninv_d=np.zeros((n_obs_scan,), dtype=np.float64),
            wind_deg_per_s=np.array([0.1, -0.05], dtype=np.float64),
            wind_sigma_x_deg_per_s=np.float64(0.01),
            wind_sigma_y_deg_per_s=np.float64(0.01),
            ell_atm=np.linspace(10.0, 500.0, n_ell, dtype=np.float64),
            cl_atm_mk2=np.ones((n_ell,), dtype=np.float64) * 1e-6,
        )
        art = load_scan_artifact(path)
        assert art["obs_pix_global_scan"].shape == (n_obs_scan,)
        assert art["c_hat_scan_obs"].shape == (n_obs_scan,)
        assert art["cov_inv"].shape == (n_obs_scan, n_obs_scan)
        assert art["Pt_Ninv_d"].shape == (n_obs_scan,)
        assert art["wind_deg_per_s"].shape == (2,)
    finally:
        path.unlink(missing_ok=True)


# --- Synthesis ---

def test_synthesis_two_scans_minimal():
    """run_synthesis with two minimal scan npzs yields finite c_hat_obs."""
    n_obs = 5
    n_pix = 10
    obs_pix_global = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    global_to_obs = np.full(n_pix, -1, dtype=np.int64)
    global_to_obs[obs_pix_global] = np.arange(n_obs, dtype=np.int64)
    layout = GlobalLayout(
        bbox_ix0=0,
        bbox_iy0=0,
        nx=2,
        ny=5,
        obs_pix_global=obs_pix_global,
        global_to_obs=global_to_obs,
        scan_paths=(Path("/dummy/scan0"), Path("/dummy/scan1")),
        pixel_size_deg=0.25,
        field_id="test",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_dir = Path(tmpdir)
        out_path = Path(tmpdir) / "combined.npz"
        for scan_index in range(2):
            obs_scan = np.array([0, 1, 2], dtype=np.int64)
            n_s = obs_scan.size
            cov_inv_s = np.eye(n_s, dtype=np.float64) * (1.0 + scan_index)
            Pt_Ninv_d_s = np.ones((n_s,), dtype=np.float64) * (0.5 + scan_index)
            n_ell = 5
            np.savez_compressed(
                scan_dir / f"scan_{scan_index:04d}_ml.npz",
                scan_index=np.int64(scan_index),
                obs_pix_global_scan=obs_pix_global[obs_scan],
                c_hat_scan_obs=np.zeros((n_s,), dtype=np.float64),
                cov_inv=cov_inv_s,
                Pt_Ninv_d=Pt_Ninv_d_s,
                wind_deg_per_s=np.array([0.0, 0.0], dtype=np.float64),
                wind_sigma_x_deg_per_s=np.float64(0.0),
                wind_sigma_y_deg_per_s=np.float64(0.0),
                ell_atm=np.linspace(10.0, 500.0, n_ell, dtype=np.float64),
                cl_atm_mk2=np.ones((n_ell,), dtype=np.float64) * 1e-6,
            )
        run_synthesis(
            layout,
            scan_dir,
            out_path,
            n_uncertain_modes=8,
            lanczos_oversample=4,
            lanczos_maxiter=16,
        )
        with np.load(out_path, allow_pickle=True) as z:
            c_hat_obs = np.asarray(z["c_hat_obs"], dtype=np.float64)
            c_hat_full_mk = np.asarray(z["c_hat_full_mk"], dtype=np.float64)
        assert c_hat_obs.shape == (n_obs,)
        assert np.all(np.isfinite(c_hat_obs))
        assert c_hat_full_mk.shape == (n_pix,)
        assert np.all(np.isfinite(c_hat_full_mk[layout.obs_pix_global]))


def test_union_super_plate_identity_when_all_layouts_share_grid():
    """Same bbox and ny => super plate equals each layout; flat remap is identity."""
    n_pix = 12
    obs = np.array([0, 5, 11], dtype=np.int64)
    g2o = np.full(n_pix, -1, dtype=np.int64)
    g2o[obs] = np.arange(3, dtype=np.int64)
    lay = GlobalLayout(
        bbox_ix0=10,
        bbox_iy0=-3,
        nx=4,
        ny=3,
        obs_pix_global=obs,
        global_to_obs=g2o,
        scan_paths=(Path("/dummy/s"),),
        pixel_size_deg=0.16666666666666666,
        field_id="t",
    )
    layouts = [("a", lay), ("b", lay)]
    ix0_s, iy0_s, nx_s, ny_s, ps = _union_super_plate_from_layouts(layouts)
    assert (ix0_s, iy0_s, nx_s, ny_s) == (10, -3, 4, 3)
    assert ps == lay.pixel_size_deg
    p_all = np.arange(n_pix, dtype=np.int64)
    out = _local_flat_pixels_to_super_flat(
        p_all,
        bbox_ix0=lay.bbox_ix0,
        bbox_iy0=lay.bbox_iy0,
        ny_local=lay.ny,
        ix0_s=ix0_s,
        iy0_s=iy0_s,
        ny_super=ny_s,
    )
    assert np.array_equal(out, p_all)


def test_union_super_plate_remap_offset_bbox():
    """Two non-overlapping same-size patches: super bbox is the union; locals map to distinct super flats."""
    ny = 3

    def _mini(bix: int, biy: int) -> GlobalLayout:
        n_pix = 6
        obs = np.arange(n_pix, dtype=np.int64)
        g2o = np.arange(n_pix, dtype=np.int64)
        return GlobalLayout(
            bbox_ix0=bix,
            bbox_iy0=biy,
            nx=2,
            ny=ny,
            obs_pix_global=obs,
            global_to_obs=g2o,
            scan_paths=(Path("/x"),),
            pixel_size_deg=0.1,
            field_id="t",
        )

    lay0 = _mini(0, 0)
    lay1 = _mini(2, 0)
    ix0_s, iy0_s, nx_s, ny_s, _ps = _union_super_plate_from_layouts([("0", lay0), ("1", lay1)])
    assert (ix0_s, iy0_s, nx_s, ny_s) == (0, 0, 4, 3)
    s0 = _local_flat_pixels_to_super_flat(
        np.array([0], dtype=np.int64),
        bbox_ix0=0,
        bbox_iy0=0,
        ny_local=ny,
        ix0_s=ix0_s,
        iy0_s=iy0_s,
        ny_super=ny_s,
    )
    s1 = _local_flat_pixels_to_super_flat(
        np.array([0], dtype=np.int64),
        bbox_ix0=2,
        bbox_iy0=0,
        ny_local=ny,
        ix0_s=ix0_s,
        iy0_s=iy0_s,
        ny_super=ny_s,
    )
    assert int(s0[0]) == 0
    assert int(s1[0]) == 6
