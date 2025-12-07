"""
FDS-FireX Triton GPU Kernels

This package contains GPU-accelerated kernels for fire simulation
using OpenAI Triton for the FDS-FireX project.

Modules:
    radiation: Radiative heat transfer kernels
    transport: Advection and diffusion kernels (planned)
    chemistry: Chemical reaction kernels (planned)
"""

__version__ = "0.1.0"
__author__ = "FDS-FireX Team"

from . import radiation

__all__ = ['radiation']
