import numpy as np
from scipy.constants import k as k_B, h


def cmb_power_spectrum(
    ell,
):
    """
    Calculate CMB temperature power spectrum at the fiducial cosmology.

    Args:
        ell (array-like): Multipole moments to evaluate

    Returns:
        ndarray: CMB temperature power spectrum [mK^2] in CMB thermodynamic units.
    """
    try:
        import camb  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "camb is required for cmb_power_spectrum(). "
            "Either install camb in your environment, or run ML (no CMB prior) and skip this call."
        ) from e

    cosmology = camb.CAMBparams()
    cosmology.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.054)
    cosmology.InitPower.set_params(As=2.1e-9, ns=0.965, r=0)
    cosmology.set_for_lmax(5000)

    results = camb.get_results(cosmology)
    powers = results.get_cmb_power_spectra(cosmology, CMB_unit="muK")
    ell_camb = np.arange(powers["total"].shape[0])  # l starts from 0
    D_ell = powers["total"][:, 0]  # TT power spectrum in D_l = l(l+1)C_l/(2π)

    factor = ell_camb * (ell_camb + 1) / (2 * np.pi)
    # Avoid divide-by-zero at ell=0 (and ell=1 where factor is finite but small).
    factor = np.where(factor > 0, factor, np.inf)
    C_ell = D_ell / factor  # [μK^2]
    C_ell *= 1e-12  # Convert to [K^2]
    C_ell *= 1e6  # Convert to [mK^2]

    # interpolate C_ell to grid_s.l
    C_ell = np.interp(ell, ell_camb, C_ell)  # mK_CMB^2

    return C_ell


def atmospheric_power_spectrum_TQU(
    ell: np.ndarray,
    component: str,
    *,
    nu_GHz: float = 220.0,
    epsilon_deg: float = 44.75,
) -> np.ndarray:
    """
    Atmospheric C_ell for TT, QQ, or UU following Coerver et al. Eq. 41.

    Returns:
      C_ell in mK^2 (CMB thermodynamic temperature units).
    """
    def _dT_CMB_dT_RJ(nu_GHz: float) -> float:
        """
        Conversion factor dT_CMB / dT_RJ at frequency nu (GHz).
        """
        nu = float(nu_GHz) * 1e9  # Hz
        T_cmb = 2.725  # K
        x = h * nu / (k_B * T_cmb)
        ex = np.exp(x)
        return float((ex - 1) ** 2 / (x**2 * ex))

    ell = np.asarray(ell, dtype=np.float64)
    epsilon = np.deg2rad(float(epsilon_deg))
    sin_e = np.sin(epsilon)
    cos_e = np.cos(epsilon)
    beta = 11.0 / 3.0

    comp = str(component).upper()
    if comp == "TT":
        B_amp = 43.0  # median at 220GHz, Table 1 (mK^2 in the paper convention)
        f_epsilon = sin_e ** (1.0 - beta)
    elif comp == "QQ":
        B_amp = 0.064  # median at 220GHz, Table 2
        f_epsilon = cos_e**4 * sin_e ** (-8.0 / 3.0)
        f_epsilon /= (3.0 + sin_e**2) ** 2
    elif comp == "UU":
        return np.zeros_like(ell)
    else:
        raise ValueError(f"Unknown component: {component}")

    conv = _dT_CMB_dT_RJ(float(nu_GHz)) ** 2
    powerlaw = (ell / (2.0 * np.pi)) ** (-beta)
    return float(B_amp) * float(conv) * float(f_epsilon) * powerlaw


def power2d_from_map(
    *,
    map_2d_mk: np.ndarray,
    pixel_res_rad: float,
    hit_mask_2d: np.ndarray | None = None,
    crop_to_hit_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a flat-sky 2D power spectrum map via FFT2.

    Definition:
      ps2d(ell_x, ell_y) = |FFT2(map)|^2 * (dx*dy) / (Nx*Ny), with dx=dy=pixel_res_rad.

    Shapes / units:
      - map_2d_mk: (ny,nx) in mK (or any linear temperature unit; ps2d is that unit^2).
      - returns KX, KY, ps2d with shape (ny_eff,nx_eff)
      - KX, KY are in ell units (rad^-1), ps2d is in mK^2 (or input_unit^2).

    Masking:
      - if hit_mask_2d is provided: unhit pixels are set to 0 prior to FFT.
      - if crop_to_hit_mask is True: crop to the tight bbox of hit_mask_2d before FFT.
    """
    img0 = np.asarray(map_2d_mk, dtype=np.float64)
    if img0.ndim != 2:
        raise ValueError("map_2d_mk must be 2D (ny,nx).")

    crop = None
    mask = None
    if hit_mask_2d is not None:
        mask = np.asarray(hit_mask_2d, dtype=bool)
        if mask.shape != img0.shape:
            raise ValueError("hit_mask_2d must match map_2d_mk shape.")
        if bool(crop_to_hit_mask):
            ys, xs = np.nonzero(mask)
            if ys.size > 0:
                iy0, iy1 = int(ys.min()), int(ys.max())
                ix0, ix1 = int(xs.min()), int(xs.max())
                crop = (slice(iy0, iy1 + 1), slice(ix0, ix1 + 1))

    img = img0[crop] if crop is not None else img0
    mask_eff = (mask[crop] if (mask is not None and crop is not None) else mask)

    if mask_eff is not None:
        img = np.where(mask_eff & np.isfinite(img), img, 0.0)
    else:
        img = np.where(np.isfinite(img), img, 0.0)

    ny_eff, nx_eff = int(img.shape[0]), int(img.shape[1])
    ell_x = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx_eff, d=float(pixel_res_rad)))
    ell_y = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny_eff, d=float(pixel_res_rad)))
    KX, KY = np.meshgrid(ell_x, ell_y, indexing="xy")  # (ny_eff,nx_eff)

    F = np.fft.fftshift(np.fft.fft2(img))
    ps2d = (np.abs(F) ** 2) * (float(pixel_res_rad) ** 2) / (float(nx_eff) * float(ny_eff))
    return KX, KY, ps2d


def radial_cl_1d_from_power2d(
    *,
    KX: np.ndarray,
    KY: np.ndarray,
    ps2d_mk2: np.ndarray,
    n_ell_bins: int = 64,
    keep_mask_2d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Radially bin a 2D power spectrum map into a 1D pseudo-C_ell(ell).

    Shapes:
      - KX, KY, ps2d_mk2: (ny,nx)
      - keep_mask_2d (optional): (ny,nx) boolean; False pixels are excluded.
    """
    KX = np.asarray(KX, dtype=np.float64)
    KY = np.asarray(KY, dtype=np.float64)
    ps2d = np.asarray(ps2d_mk2, dtype=np.float64)
    if KX.shape != KY.shape or KX.shape != ps2d.shape:
        raise ValueError("KX, KY, ps2d_mk2 must have the same shape.")

    ell = np.sqrt(KX * KX + KY * KY).reshape(-1)
    ps = ps2d.reshape(-1)
    ok = np.isfinite(ell) & np.isfinite(ps)
    if keep_mask_2d is not None:
        km = np.asarray(keep_mask_2d, dtype=bool)
        if km.shape != ps2d.shape:
            raise ValueError("keep_mask_2d must match ps2d_mk2 shape.")
        ok = ok & km.reshape(-1)

    ell_max = float(np.max(ell[np.isfinite(ell)]))
    edges = np.linspace(0.0, ell_max, int(n_ell_bins) + 1)
    idx = np.digitize(ell, edges) - 1

    cl = np.zeros((int(n_ell_bins),), dtype=np.float64)
    counts = np.zeros((int(n_ell_bins),), dtype=np.float64)
    m = ok & (idx >= 0) & (idx < int(n_ell_bins))
    np.add.at(cl, idx[m], ps[m])
    np.add.at(counts, idx[m], 1.0)
    cl = np.where(counts > 0, cl / counts, np.nan)
    ell_centers = 0.5 * (edges[:-1] + edges[1:])
    return ell_centers, cl



def radial_cl_1d_from_map(
    *,
    map_2d_mk: np.ndarray,
    pixel_res_rad: float,
    hit_mask: np.ndarray | None = None,
    n_ell_bins: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Radially binned flat-sky pseudo-C_ell from a 2D map.

    Args:
      map_2d_mk: (ny, nx) map in mK.
      pixel_res_rad: pixel size in radians.
      hit_mask: optional (ny, nx) bool mask; if provided, masked pixels are set to 0.
      n_ell_bins: number of radial bins.

    Returns:
      ell_centers: (n_ell_bins,) float64
      cl_mk2: (n_ell_bins,) float64
    """
    KX, KY, ps2d = power2d_from_map(
        map_2d_mk=np.asarray(map_2d_mk),
        pixel_res_rad=float(pixel_res_rad),
        hit_mask_2d=hit_mask,
        crop_to_hit_mask=False,
    )
    return radial_cl_1d_from_power2d(
        KX=KX,
        KY=KY,
        ps2d_mk2=ps2d,
        n_ell_bins=int(n_ell_bins),
    )