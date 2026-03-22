"""
cad: CMB atmosphere decontamination utilities.

Public exports at package root:
  - `FourierGaussianPrior`
  - `estimate_wind_deg_per_s`

Parallel reconstruction/synthesis entry points live under `cad.parallel_solve`.
"""

from .prior import FourierGaussianPrior
from .wind import estimate_wind_deg_per_s

__all__ = [
    "FourierGaussianPrior",
    "estimate_wind_deg_per_s",
]

