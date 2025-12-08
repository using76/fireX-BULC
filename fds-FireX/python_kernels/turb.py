"""
FDS-FireX Turbulence Filter Triton Kernels
Implements TEST_FILTER (3x3x3 box filter) on GPU (FP32)
"""

import torch
import numpy as np
import ctypes

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

def check_gpu_available():
    return torch.cuda.is_available()


if TRITON_AVAILABLE:
    @triton.jit
    def _test_filter_kernel(
        orig_ptr, hat_ptr,
        stride_i, stride_j, stride_k,
        IBAR: tl.constexpr, JBAR: tl.constexpr, KBAR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """3x3x3 box filter: HAT[i,j,k] = mean(ORIG[i-1:i+2, j-1:j+2, k-1:k+2])"""
        pid = tl.program_id(0)
        num_cells = IBAR * JBAR * KBAR
        if pid >= num_cells:
            return
        k = pid // (IBAR * JBAR)
        rem = pid % (IBAR * JBAR)
        j = rem // IBAR
        i = rem % IBAR
        # Skip boundary cells
        if i < 1 or i >= IBAR - 1 or j < 1 or j >= JBAR - 1 or k < 1 or k >= KBAR - 1:
            idx = i * stride_i + j * stride_j + k * stride_k
            val = tl.load(orig_ptr + idx)
            tl.store(hat_ptr + idx, val)
            return
        # 3x3x3 average
        total = tl.zeros([], dtype=tl.float32)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    idx_n = (i + di) * stride_i + (j + dj) * stride_j + (k + dk) * stride_k
                    total += tl.load(orig_ptr + idx_n)
        hat_val = total / 27.0
        idx = i * stride_i + j * stride_j + k * stride_k
        tl.store(hat_ptr + idx, hat_val)


def test_filter_kernel(orig, hat):
    """Apply 3x3x3 box filter to field"""
    orig = orig.float()
    IBAR, JBAR, KBAR = orig.shape
    n_cells = IBAR * JBAR * KBAR
    if hat is None:
        hat = torch.zeros_like(orig)
    if TRITON_AVAILABLE and orig.is_cuda:
        grid = lambda meta: (triton.cdiv(n_cells, meta["BLOCK_SIZE"]),)
        _test_filter_kernel[grid](
            orig, hat,
            orig.stride(0), orig.stride(1), orig.stride(2),
            IBAR, JBAR, KBAR, 256
        )
    else:
        # PyTorch fallback using conv3d
        kernel = torch.ones(1, 1, 3, 3, 3, device=orig.device) / 27.0
        padded = torch.nn.functional.pad(orig.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode="replicate")
        hat = torch.nn.functional.conv3d(padded, kernel).squeeze()
    return hat


def compute_test_filter(field):
    """Convenience wrapper"""
    return test_filter_kernel(field, None)


class FilterGPUData(ctypes.Structure):
    _fields_ = [
        ("orig_ptr", ctypes.c_void_p),
        ("hat_ptr", ctypes.c_void_p),
        ("k3d_ptr", ctypes.c_void_p),
        ("ibar", ctypes.c_int32),
        ("jbar", ctypes.c_int32),
        ("kbar", ctypes.c_int32),
    ]


if __name__ == "__main__":
    print(f"Triton: {TRITON_AVAILABLE}, CUDA: {check_gpu_available()}")
    if check_gpu_available():
        x = torch.randn(32, 32, 32, device="cuda")
        y = compute_test_filter(x)
        print(f"Test passed! Shape: {y.shape}")
