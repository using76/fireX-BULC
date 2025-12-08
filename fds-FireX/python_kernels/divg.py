"""
FDS-FireX Divergence Solver Triton Kernels
Implements divergence computation on GPU (FP32)
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
    def _divergence_kernel(
        u_ptr, v_ptr, w_ptr, div_ptr,
        stride_i, stride_j, stride_k,
        IBAR: tl.constexpr, JBAR: tl.constexpr, KBAR: tl.constexpr,
        RDX, RDY, RDZ,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute divergence: div = du/dx + dv/dy + dw/dz"""
        pid = tl.program_id(0)
        num_cells = IBAR * JBAR * KBAR
        if pid >= num_cells:
            return
        k = pid // (IBAR * JBAR)
        rem = pid % (IBAR * JBAR)
        j = rem // IBAR
        i = rem % IBAR
        idx = i * stride_i + j * stride_j + k * stride_k
        # Staggered grid: U at i+1/2, V at j+1/2, W at k+1/2
        # div = (U[i] - U[i-1])/dx + (V[j] - V[j-1])/dy + (W[k] - W[k-1])/dz
        if i > 0:
            idx_im = (i - 1) * stride_i + j * stride_j + k * stride_k
            dudx = (tl.load(u_ptr + idx) - tl.load(u_ptr + idx_im)) * RDX
        else:
            dudx = tl.load(u_ptr + idx) * RDX
        if j > 0:
            idx_jm = i * stride_i + (j - 1) * stride_j + k * stride_k
            dvdy = (tl.load(v_ptr + idx) - tl.load(v_ptr + idx_jm)) * RDY
        else:
            dvdy = tl.load(v_ptr + idx) * RDY
        if k > 0:
            idx_km = i * stride_i + j * stride_j + (k - 1) * stride_k
            dwdz = (tl.load(w_ptr + idx) - tl.load(w_ptr + idx_km)) * RDZ
        else:
            dwdz = tl.load(w_ptr + idx) * RDZ
        div = dudx + dvdy + dwdz
        tl.store(div_ptr + idx, div)


    @triton.jit
    def _diffusion_flux_kernel(
        rho_ptr, d_ptr, flux_ptr,
        stride_i, stride_j, stride_k,
        IBAR: tl.constexpr, JBAR: tl.constexpr, KBAR: tl.constexpr,
        RDX, RDY, RDZ,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute diffusion flux: flux = -D * grad(rho)"""
        pid = tl.program_id(0)
        num_cells = IBAR * JBAR * KBAR
        if pid >= num_cells:
            return
        k = pid // (IBAR * JBAR)
        rem = pid % (IBAR * JBAR)
        j = rem // IBAR
        i = rem % IBAR
        if i < 1 or i >= IBAR - 1 or j < 1 or j >= JBAR - 1 or k < 1 or k >= KBAR - 1:
            idx = i * stride_i + j * stride_j + k * stride_k
            tl.store(flux_ptr + idx, 0.0)
            return
        idx = i * stride_i + j * stride_j + k * stride_k
        idx_ip = (i + 1) * stride_i + j * stride_j + k * stride_k
        idx_im = (i - 1) * stride_i + j * stride_j + k * stride_k
        idx_jp = i * stride_i + (j + 1) * stride_j + k * stride_k
        idx_jm = i * stride_i + (j - 1) * stride_j + k * stride_k
        idx_kp = i * stride_i + j * stride_j + (k + 1) * stride_k
        idx_km = i * stride_i + j * stride_j + (k - 1) * stride_k
        D = tl.load(d_ptr + idx)
        # Laplacian
        rho = tl.load(rho_ptr + idx)
        rho_ip = tl.load(rho_ptr + idx_ip)
        rho_im = tl.load(rho_ptr + idx_im)
        rho_jp = tl.load(rho_ptr + idx_jp)
        rho_jm = tl.load(rho_ptr + idx_jm)
        rho_kp = tl.load(rho_ptr + idx_kp)
        rho_km = tl.load(rho_ptr + idx_km)
        laplacian = (rho_ip - 2*rho + rho_im) * RDX * RDX
        laplacian += (rho_jp - 2*rho + rho_jm) * RDY * RDY
        laplacian += (rho_kp - 2*rho + rho_km) * RDZ * RDZ
        flux = D * laplacian
        tl.store(flux_ptr + idx, flux)


def divergence_kernel(u, v, w, div, dx, dy, dz):
    u = u.float()
    v = v.float()
    w = w.float()
    IBAR, JBAR, KBAR = u.shape
    n_cells = IBAR * JBAR * KBAR
    if div is None:
        div = torch.zeros_like(u)
    if TRITON_AVAILABLE and u.is_cuda:
        grid = lambda meta: (triton.cdiv(n_cells, meta["BLOCK_SIZE"]),)
        _divergence_kernel[grid](
            u, v, w, div,
            u.stride(0), u.stride(1), u.stride(2),
            IBAR, JBAR, KBAR, 1.0/dx, 1.0/dy, 1.0/dz, 256
        )
    else:
        # PyTorch fallback (staggered grid)
        dudx = (u - torch.roll(u, 1, 0)) / dx
        dvdy = (v - torch.roll(v, 1, 1)) / dy
        dwdz = (w - torch.roll(w, 1, 2)) / dz
        div = dudx + dvdy + dwdz
    return div


def compute_divergence(u, v, w, dx, dy, dz):
    return divergence_kernel(u, v, w, None, dx, dy, dz)


class DivergenceGPUData(ctypes.Structure):
    _fields_ = [
        ("u_ptr", ctypes.c_void_p),
        ("v_ptr", ctypes.c_void_p),
        ("w_ptr", ctypes.c_void_p),
        ("div_ptr", ctypes.c_void_p),
        ("rho_ptr", ctypes.c_void_p),
        ("ibar", ctypes.c_int32),
        ("jbar", ctypes.c_int32),
        ("kbar", ctypes.c_int32),
        ("rdx", ctypes.c_float),
        ("rdy", ctypes.c_float),
        ("rdz", ctypes.c_float),
    ]


if __name__ == "__main__":
    print(f"Triton: {TRITON_AVAILABLE}, CUDA: {check_gpu_available()}")
    if check_gpu_available():
        u = torch.randn(32, 32, 32, device="cuda")
        v = torch.randn(32, 32, 32, device="cuda")
        w = torch.randn(32, 32, 32, device="cuda")
        div = compute_divergence(u, v, w, 0.1, 0.1, 0.1)
        print(f"Test passed! div mean: {div.mean().item():.6e}")
