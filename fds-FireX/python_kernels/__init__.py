"""
FDS-FireX GPU Kernels Package
Triton-based GPU acceleration for Fire Dynamics Simulator
"""

from .radiation import (
    radiation_kernel,
    compute_radiation_source,
    check_gpu_available
)
from .turb import (
    test_filter_kernel,
    compute_test_filter
)
from .velo import (
    viscosity_kernel,
    strain_rate_kernel,
    compute_viscosity,
    compute_strain_rate
)
from .divg import (
    divergence_kernel,
    compute_divergence
)

__version__ = "1.0.0"
__all__ = [
    "radiation_kernel", "compute_radiation_source",
    "test_filter_kernel", "compute_test_filter",
    "viscosity_kernel", "strain_rate_kernel",
    "compute_viscosity", "compute_strain_rate",
    "divergence_kernel", "compute_divergence",
    "check_gpu_available"
]
