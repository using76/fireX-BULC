"""
Memory management utilities for Fortran-Python interoperability.

This module provides utilities for converting between Fortran and Python
array layouts and managing GPU memory.

Key considerations:
- Fortran uses column-major (I,J,K) ordering
- Python/C use row-major ordering
- GPU memory transfers should be minimized (Zero-Copy when possible)
"""

import torch
import numpy as np
import ctypes
from typing import Tuple, Optional, Union


class GPUMemoryManager:
    """
    Manages GPU memory allocation and transfers for Triton kernels.

    Supports pinned (page-locked) memory for optimal CPU-GPU transfer
    and tracks allocations for cleanup.
    """

    def __init__(self, device: str = 'cuda:0'):
        """
        Initialize the memory manager.

        Args:
            device: CUDA device to use (default 'cuda:0')
        """
        self.device = torch.device(device)
        self.allocations = {}
        self._allocation_id = 0

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        name: Optional[str] = None,
        pinned: bool = False,
    ) -> torch.Tensor:
        """
        Allocate a tensor on GPU.

        Args:
            shape: Tensor shape
            dtype: Data type (default float32)
            name: Optional name for tracking
            pinned: If True, allocate pinned CPU memory (for transfers)

        Returns:
            Allocated tensor
        """
        if pinned:
            tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            tensor = torch.zeros(shape, dtype=dtype, device=self.device)

        if name is None:
            name = f"tensor_{self._allocation_id}"
            self._allocation_id += 1

        self.allocations[name] = tensor
        return tensor

    def free(self, name: str) -> None:
        """
        Free a named allocation.

        Args:
            name: Name of allocation to free
        """
        if name in self.allocations:
            del self.allocations[name]

    def free_all(self) -> None:
        """Free all tracked allocations."""
        self.allocations.clear()
        torch.cuda.empty_cache()

    def get_info(self) -> dict:
        """
        Get GPU memory information.

        Returns:
            Dictionary with memory stats
        """
        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(self.device),
            "total_memory": torch.cuda.get_device_properties(self.device).total_memory,
            "allocated": torch.cuda.memory_allocated(self.device),
            "cached": torch.cuda.memory_reserved(self.device),
            "n_allocations": len(self.allocations),
        }


class FortranArrayAdapter:
    """
    Adapter for converting between Fortran and Python array layouts.

    Handles the column-major to row-major conversion and back.
    """

    @staticmethod
    def from_pointer(
        ptr: int,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        to_gpu: bool = True,
    ) -> torch.Tensor:
        """
        Create a PyTorch tensor from a Fortran array pointer.

        Args:
            ptr: Memory address (as integer)
            shape: Shape in Fortran order (I, J, K)
            dtype: NumPy dtype of the source array
            to_gpu: If True, move to GPU

        Returns:
            PyTorch tensor in row-major order
        """
        if ptr == 0:
            return torch.zeros(shape[::-1], dtype=torch.float32,
                             device='cuda' if to_gpu else 'cpu')

        # Cast pointer to ctypes
        ctypes_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        n_elements = int(np.prod(shape))

        # Create numpy array view (no copy)
        np_arr = np.ctypeslib.as_array(ctypes_ptr, shape=(n_elements,))
        np_arr = np_arr.reshape(shape, order='F')  # Fortran column-major

        # Convert to PyTorch
        tensor = torch.from_numpy(np_arr.copy())

        # Transpose to row-major order
        if len(shape) == 3:
            tensor = tensor.permute(2, 1, 0).contiguous()
        elif len(shape) == 2:
            tensor = tensor.T.contiguous()

        if to_gpu:
            tensor = tensor.cuda()

        return tensor

    @staticmethod
    def to_pointer(
        tensor: torch.Tensor,
        ptr: int,
        shape: Tuple[int, ...],
    ) -> None:
        """
        Write a PyTorch tensor back to a Fortran array pointer.

        Args:
            tensor: Source tensor
            ptr: Target memory address
            shape: Shape in Fortran order (I, J, K)
        """
        if ptr == 0:
            return

        # Transpose back to Fortran order
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 1, 0).contiguous()
        elif tensor.dim() == 2:
            tensor = tensor.T.contiguous()

        # Convert to numpy
        np_arr = tensor.cpu().numpy()
        np_arr = np.asfortranarray(np_arr)

        # Write to pointer
        ctypes_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        n_elements = int(np.prod(shape))
        target = np.ctypeslib.as_array(ctypes_ptr, shape=(n_elements,))
        target[:] = np_arr.ravel(order='F')

    @staticmethod
    def fortran_to_python_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Convert Fortran shape (I, J, K) to Python shape (K, J, I).
        """
        return shape[::-1]

    @staticmethod
    def python_to_fortran_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Convert Python shape (K, J, I) to Fortran shape (I, J, K).
        """
        return shape[::-1]


def create_contiguous_view(
    tensor: torch.Tensor,
    order: str = 'C'
) -> torch.Tensor:
    """
    Create a contiguous view of a tensor.

    Args:
        tensor: Input tensor
        order: Memory order ('C' for row-major, 'F' for column-major)

    Returns:
        Contiguous tensor
    """
    if order == 'F':
        # For Fortran order, we need to permute first
        dims = list(range(tensor.dim()))[::-1]
        return tensor.permute(dims).contiguous().permute(dims)
    return tensor.contiguous()


def estimate_memory_usage(
    grid_shape: Tuple[int, int, int],
    n_angles: int,
    n_bands: int = 1,
    dtype_size: int = 4,  # float32 = 4 bytes
) -> dict:
    """
    Estimate GPU memory usage for radiation computation.

    Args:
        grid_shape: (IBAR, JBAR, KBAR)
        n_angles: Number of radiation angles (NRA)
        n_bands: Number of spectral bands (NSB)
        dtype_size: Bytes per element

    Returns:
        Dictionary with memory estimates in bytes
    """
    n_cells = int(np.prod(grid_shape))

    # Core arrays (3D)
    arrays_3d = [
        "TMP", "KAPPA_GAS", "KFST4_GAS", "KFST4_PART",
        "IL", "EXTCOE", "SCAEFF", "UIIOLD", "QR", "SOURCE"
    ]
    memory_3d = len(arrays_3d) * n_cells * dtype_size

    # 4D arrays
    memory_4d = n_cells * n_bands * dtype_size  # UIID

    # 1D arrays
    arrays_1d = ["DX", "DY", "DZ", "DLX", "DLY", "DLZ", "RSA"]
    memory_1d = (grid_shape[0] + grid_shape[1] + grid_shape[2] +
                 4 * n_angles) * dtype_size

    total = memory_3d + memory_4d + memory_1d

    return {
        "grid_shape": grid_shape,
        "n_cells": n_cells,
        "n_angles": n_angles,
        "memory_3d_arrays": memory_3d,
        "memory_4d_arrays": memory_4d,
        "memory_1d_arrays": memory_1d,
        "total_bytes": total,
        "total_mb": total / (1024 * 1024),
        "total_gb": total / (1024 * 1024 * 1024),
    }
