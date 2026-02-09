
import numpy as np

from cad.prior import FourierGaussianPrior


def main() -> None:
    """
    Check anisotropic prior geometry.

    The prior bins ell with:
      ell = sqrt((kx / cos_dec)^2 + ky^2).

    A delta-function covariance (C e0) should be stretched by ~1/cos_dec in x.
    """
    nx, ny = 256, 256
    pixel_res_rad = 1.0 * np.pi / 180.0
    cos_dec = 0.5

    # Narrow spectrum in ell to get a wide real-space correlation.
    n_ell_bins = 100
    bin_idx = np.arange(n_ell_bins)
    sigma_bin = 4.0
    cl_bins = np.exp(-0.5 * (bin_idx / sigma_bin) ** 2)

    prior = FourierGaussianPrior(
        nx=nx,
        ny=ny,
        pixel_res_rad=pixel_res_rad,
        cl_bins_mk2=cl_bins,
        cos_dec=cos_dec,
        cl_floor_mk2=1e-20,
    )

    # Delta vector -> covariance column (real-space correlation).
    delta = np.zeros(nx * ny, dtype=np.float64)
    mid_idx = (nx // 2) * ny + (ny // 2)
    delta[mid_idx] = 1.0
    corr_map = prior.apply_C(delta).reshape(nx, ny)

    # Central cuts in x and y.
    cut_x = corr_map[:, ny // 2]
    cut_y = corr_map[nx // 2, :]

    def _fwhm(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr / np.max(arr)
        above = np.where(arr > 0.5)[0]
        if above.size < 1:
            return 0.0
        return float(above[-1] - above[0])

    fwhm_x = _fwhm(cut_x)
    fwhm_y = _fwhm(cut_y)
    if fwhm_y <= 0:
        raise RuntimeError("FWHM check failed (y-direction).")

    ratio = fwhm_x / fwhm_y
    expected = 1.0 / cos_dec
    print(f"[anisotropy] fwhm_x={fwhm_x:.2f} fwhm_y={fwhm_y:.2f} ratio={ratio:.2f} expected={expected:.2f}")

    if abs(ratio - expected) > 0.2:
        raise RuntimeError("Anisotropy check failed.")


if __name__ == "__main__":
    main()
