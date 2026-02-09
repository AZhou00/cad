
import numpy as np
from cad.prior import FourierGaussianPrior

def test_prior_normalization():
    nx, ny = 100, 100
    pixel_res_rad = 1.0 * np.pi / 180.0  # 1 degree pixels
    
    # Case 1: White noise prior (Cl = constant)
    # If Cl is constant A (units mK^2 sr), then C_pix = A / (dx dy).
    # Variance per pixel should be A / (dx dy).
    
    A = 100.0 # mK^2 sr
    cl_bins = np.full(100, A)
    
    prior = FourierGaussianPrior(nx=nx, ny=ny, pixel_res_rad=pixel_res_rad, cl_bins_mk2=cl_bins)
    
    # Check C applied to delta function
    x = np.zeros(nx*ny)
    x[0] = 1.0
    Cx = prior.apply_C(x)
    
    # The value at x[0] should be the variance?
    # C is the covariance matrix. C_ij = Cov(x_i, x_j).
    # If x is delta at 0, Cx is the 0-th column of C.
    # (Cx)[0] is C_00, the variance.
    
    dx = pixel_res_rad
    dy = pixel_res_rad
    expected_var = A / (dx * dy)
    
    print(f"Expected variance: {expected_var:.4e}")
    print(f"Computed variance: {Cx[0]:.4e}")
    
    ratio = Cx[0] / expected_var
    print(f"Ratio: {ratio:.4f}")
    
    if abs(ratio - 1.0) < 1e-3:
        print("Normalization verified for White Noise.")
    else:
        print("Normalization MISMATCH for White Noise.")

    # Case 2: Invertibility
    x_rand = np.random.randn(nx*ny)
    y = prior.apply_C(x_rand)
    x_rec = prior.apply_Cinv(y)
    
    err = np.max(np.abs(x_rec - x_rand))
    print(f"Inversion error: {err:.4e}")
    if err < 1e-10:
        print("Invertibility verified.")
    else:
        print("Invertibility FAILED.")

if __name__ == "__main__":
    test_prior_normalization()
