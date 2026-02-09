
import numpy as np
import matplotlib.pyplot as plt
from cad.prior import FourierGaussianPrior

def test_anisotropy():
    nx, ny = 256, 256
    pixel_res_rad = 1.0 * np.pi / 180.0  # 1 degree pixels (dummy)
    cos_dec = 0.5  # extreme case, 60 deg latitude
    
    # White noise power spectrum to check pure geometry? 
    # No, we need spatial correlations to see the stretch.
    # Use a Gaussian power spectrum for easy correlation length measurement.
    # C_ell = exp(-ell^2 / (2 sigma_ell^2))
    # Correlation function should be Gaussian width sigma_r ~ 1/sigma_ell.
    
    # We want wide correlation in real space => Narrow in k-space.
    # We define cl_bins directly in terms of bin index to ensure we are narrow relative to Nyquist.
    n_ell_bins = 100
    bin_idx = np.arange(n_ell_bins)
    # Width of 4 bins out of 100.
    sigma_bin = 4.0
    cl_bins = np.exp(-0.5 * (bin_idx / sigma_bin)**2)
    
    prior = FourierGaussianPrior(
        nx=nx, 
        ny=ny, 
        pixel_res_rad=pixel_res_rad, 
        cl_bins_mk2=cl_bins, 
        cos_dec=cos_dec,
        cl_floor_mk2=1e-20
    )
    
    # Generate a realization? Or just look at the Covariance of a central pixel?
    # apply_C to a delta function gives the correlation function relative to that pixel.
    
    delta = np.zeros(nx*ny)
    mid_idx = (nx//2)*ny + (ny//2)
    delta[mid_idx] = 1.0
    
    corr_map = prior.apply_C(delta).reshape(nx, ny)
    
    # Measure width in x and y
    # x is axis 0, y is axis 1
    
    center_x, center_y = nx//2, ny//2
    
    # Cut along x (iy = center_y)
    cut_x = corr_map[:, center_y]
    # Cut along y (ix = center_x)
    cut_y = corr_map[center_x, :]
    
    # Fit Gaussians or measure FWHM
    # Simple FWHM estimator via interpolation
    def get_fwhm(arr):
        # Normalize
        arr = arr / np.max(arr)
        # Find indices where it crosses 0.5
        above = np.where(arr > 0.5)[0]
        if len(above) < 1: return 0.0
        return above[-1] - above[0]

    fwhm_x = get_fwhm(cut_x)
    fwhm_y = get_fwhm(cut_y)
    
    print(f"FWHM X (grid pixels): {fwhm_x}")
    print(f"FWHM Y (grid pixels): {fwhm_y}")
    if fwhm_y > 0:
        ratio = fwhm_x / fwhm_y
        print(f"Ratio X/Y: {ratio:.2f}")
        print(f"Expected Ratio (1/cos_dec): {1.0/cos_dec:.2f}")
        
        if abs(ratio - (1.0/cos_dec)) < 0.2:
             print("Anisotropy verified.")
        else:
             print("Anisotropy CHECK FAILED.")
    else:
        print("Could not measure FWHM.")

if __name__ == "__main__":
    test_anisotropy()
