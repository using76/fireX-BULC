"""
FDS-FireX Velocity Solver Triton Kernels
Implements viscosity and strain rate computation on GPU (FP32)
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
    def _viscosity_kernel(
        tmp_ptr, zz_ptr, mu_ptr,
        stride_i, stride_j, stride_k,
        IBAR: tl.constexpr, JBAR: tl.constexpr, KBAR: tl.constexpr,
        MU_0: tl.constexpr,  # Reference viscosity
        T_0: tl.constexpr,   # Reference temperature
        BLOCK_SIZE: tl.constexpr,
    ):
        """Sutherland viscosity: mu = mu_0 * (T/T_0)^1.5 * (T_0 + S) / (T + S)"""
        pid = tl.program_id(0)
        num_cells = IBAR * JBAR * KBAR
        if pid >= num_cells:
            return
        k = pid // (IBAR * JBAR)
        rem = pid % (IBAR * JBAR)
        j = rem // IBAR
        i = rem % IBAR
        idx = i * stride_i + j * stride_j + k * stride_k
        T = tl.load(tmp_ptr + idx)
        S = 110.4  # Sutherland constant for air
        T_ratio = T / T_0
        mu = MU_0 * T_ratio * tl.sqrt(T_ratio) * (T_0 + S) / (T + S)
        tl.store(mu_ptr + idx, mu)


    @triton.jit
    def _strain_rate_kernel(
        u_ptr, v_ptr, w_ptr, strain_ptr,
        stride_i, stride_j, stride_k,
        IBAR: tl.constexpr, JBAR: tl.constexpr, KBAR: tl.constexpr,
        RDX, RDY, RDZ,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute strain rate magnitude: S = sqrt(2 * Sij * Sij)"""
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
            tl.store(strain_ptr + idx, 0.0)
            return
        idx = i * stride_i + j * stride_j + k * stride_k
        idx_ip = (i + 1) * stride_i + j * stride_j + k * stride_k
        idx_im = (i - 1) * stride_i + j * stride_j + k * stride_k
        idx_jp = i * stride_i + (j + 1) * stride_j + k * stride_k
        idx_jm = i * stride_i + (j - 1) * stride_j + k * stride_k
        idx_kp = i * stride_i + j * stride_j + (k + 1) * stride_k
        idx_km = i * stride_i + j * stride_j + (k - 1) * stride_k
        # Velocity gradients (central difference)
        dudx = (tl.load(u_ptr + idx_ip) - tl.load(u_ptr + idx_im)) * 0.5 * RDX
        dvdy = (tl.load(v_ptr + idx_jp) - tl.load(v_ptr + idx_jm)) * 0.5 * RDY
        dwdz = (tl.load(w_ptr + idx_kp) - tl.load(w_ptr + idx_km)) * 0.5 * RDZ
        dudy = (tl.load(u_ptr + idx_jp) - tl.load(u_ptr + idx_jm)) * 0.5 * RDY
        dvdx = (tl.load(v_ptr + idx_ip) - tl.load(v_ptr + idx_im)) * 0.5 * RDX
        dudz = (tl.load(u_ptr + idx_kp) - tl.load(u_ptr + idx_km)) * 0.5 * RDZ
        dwdx = (tl.load(w_ptr + idx_ip) - tl.load(w_ptr + idx_im)) * 0.5 * RDX
        dvdz = (tl.load(v_ptr + idx_kp) - tl.load(v_ptr + idx_km)) * 0.5 * RDZ
        dwdy = (tl.load(w_ptr + idx_jp) - tl.load(w_ptr + idx_jm)) * 0.5 * RDY
        # Strain rate tensor components
        S11 = dudx
        S22 = dvdy
        S33 = dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)
        # Strain rate magnitude
        S2 = 2.0 * (S11*S11 + S22*S22 + S33*S33 + 2.0*(S12*S12 + S13*S13 + S23*S23))
        S_mag = tl.sqrt(tl.maximum(S2, 0.0))
        tl.store(strain_ptr + idx, S_mag)


def viscosity_kernel(tmp, zz, mu, mu_0=1.8e-5, T_0=293.0):
    tmp = tmp.float()
    IBAR, JBAR, KBAR = tmp.shape
    n_cells = IBAR * JBAR * KBAR
    if mu is None:
        mu = torch.zeros_like(tmp)
    if TRITON_AVAILABLE and tmp.is_cuda:
        grid = lambda meta: (triton.cdiv(n_cells, meta["BLOCK_SIZE"]),)
        _viscosity_kernel[grid](
            tmp, zz, mu,
            tmp.stride(0), tmp.stride(1), tmp.stride(2),
            IBAR, JBAR, KBAR, mu_0, T_0, 256
        )
    else:
        S = 110.4
        T_ratio = tmp / T_0
        mu = mu_0 * T_ratio * torch.sqrt(T_ratio) * (T_0 + S) / (tmp + S)
    return mu


def strain_rate_kernel(u, v, w, strain, dx, dy, dz):
    u = u.float()
    v = v.float()
    w = w.float()
    IBAR, JBAR, KBAR = u.shape
    n_cells = IBAR * JBAR * KBAR
    if strain is None:
        strain = torch.zeros_like(u)
    if TRITON_AVAILABLE and u.is_cuda:
        grid = lambda meta: (triton.cdiv(n_cells, meta["BLOCK_SIZE"]),)
        _strain_rate_kernel[grid](
            u, v, w, strain,
            u.stride(0), u.stride(1), u.stride(2),
            IBAR, JBAR, KBAR, 1.0/dx, 1.0/dy, 1.0/dz, 256
        )
    else:
        # PyTorch fallback
        dudx = (torch.roll(u, -1, 0) - torch.roll(u, 1, 0)) / (2 * dx)
        dvdy = (torch.roll(v, -1, 1) - torch.roll(v, 1, 1)) / (2 * dy)
        dwdz = (torch.roll(w, -1, 2) - torch.roll(w, 1, 2)) / (2 * dz)
        dudy = (torch.roll(u, -1, 1) - torch.roll(u, 1, 1)) / (2 * dy)
        dvdx = (torch.roll(v, -1, 0) - torch.roll(v, 1, 0)) / (2 * dx)
        dudz = (torch.roll(u, -1, 2) - torch.roll(u, 1, 2)) / (2 * dz)
        dwdx = (torch.roll(w, -1, 0) - torch.roll(w, 1, 0)) / (2 * dx)
        dvdz = (torch.roll(v, -1, 2) - torch.roll(v, 1, 2)) / (2 * dz)
        dwdy = (torch.roll(w, -1, 1) - torch.roll(w, 1, 1)) / (2 * dy)
        S11, S22, S33 = dudx, dvdy, dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)
        S2 = 2 * (S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2))
        strain = torch.sqrt(torch.clamp(S2, min=0))
    return strain


def compute_viscosity(tmp, zz=None):
    return viscosity_kernel(tmp, zz, None)


def compute_strain_rate(u, v, w, dx, dy, dz):
    return strain_rate_kernel(u, v, w, None, dx, dy, dz)


class VelocityGPUData(ctypes.Structure):
    _fields_ = [
        ("u_ptr", ctypes.c_void_p),
        ("v_ptr", ctypes.c_void_p),
        ("w_ptr", ctypes.c_void_p),
        ("tmp_ptr", ctypes.c_void_p),
        ("mu_ptr", ctypes.c_void_p),
        ("strain_ptr", ctypes.c_void_p),
        ("ibar", ctypes.c_int32),
        ("jbar", ctypes.c_int32),
        ("kbar", ctypes.c_int32),
        ("rdx", ctypes.c_float),
        ("rdy", ctypes.c_float),
        ("rdz", ctypes.c_float),
    ]


if __name__ == "__main__":
    print(f"Triton: {TRITON_AVAILABLE}, CUDA: {check_gpu_available()}")
