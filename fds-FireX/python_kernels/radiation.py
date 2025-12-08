"""
FDS-FireX Radiation Heat Transfer Triton Kernels
Implements discrete ordinates radiation solver on GPU (FP32)
"""

import torch
import numpy as np
from typing import Tuple, Optional
import ctypes

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

SIGMA = 5.670374419e-8
PI = 3.14159265358979

def check_gpu_available():
    try:
        return torch.cuda.is_available()
    except:
        return False

def get_device():
    return torch.device("cuda" if check_gpu_available() else "cpu")


if TRITON_AVAILABLE:
    @triton.jit
    def _radiation_intensity_kernel(
        tmp_ptr, kappa_ptr, il_ptr, extcoe_ptr,
        stride_tmp_i, stride_tmp_j, stride_tmp_k,
        stride_i, stride_j, stride_k,
        IBAR: tl.constexpr, JBAR: tl.constexpr, KBAR: tl.constexpr,
        DX, DY, DZ,
        ANGLE_X, ANGLE_Y, ANGLE_Z,
        SIGMA_VAL: tl.constexpr, PI_VAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_cells = IBAR * JBAR * KBAR
        if pid >= num_cells:
            return
        k = pid // (IBAR * JBAR)
        rem = pid % (IBAR * JBAR)
        j = rem // IBAR
        i = rem % IBAR
        idx = i * stride_i + j * stride_j + k * stride_k
        tmp_idx = (i + 1) * stride_tmp_i + (j + 1) * stride_tmp_j + (k + 1) * stride_tmp_k
        T = tl.load(tmp_ptr + tmp_idx)
        T4 = T * T * T * T
        B = SIGMA_VAL * T4 / PI_VAL
        kappa = tl.load(kappa_ptr + idx)
        extcoe = tl.load(extcoe_ptr + idx)
        I_old = tl.load(il_ptr + idx)
        abs_x = tl.abs(ANGLE_X) + 1e-10
        abs_y = tl.abs(ANGLE_Y) + 1e-10
        abs_z = tl.abs(ANGLE_Z) + 1e-10
        ds = 1.0 / (abs_x / DX + abs_y / DY + abs_z / DZ)
        tau = extcoe * ds
        exp_tau = tl.exp(-tau)
        source_term = tl.where(extcoe > 1e-10, B * kappa / extcoe * (1.0 - exp_tau), B * kappa * ds)
        I_new = I_old * exp_tau + source_term
        tl.store(il_ptr + idx, I_new)

    @triton.jit
    def _radiation_source_kernel(
        il_ptr, qr_ptr, kappa_ptr, tmp_ptr,
        stride_i, stride_j, stride_k,
        stride_tmp_i, stride_tmp_j, stride_tmp_k,
        stride_angle,
        IBAR: tl.constexpr, JBAR: tl.constexpr, KBAR: tl.constexpr,
        NRA: tl.constexpr,
        SIGMA_VAL: tl.constexpr, FOUR_PI: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_cells = IBAR * JBAR * KBAR
        if pid >= num_cells:
            return
        k = pid // (IBAR * JBAR)
        rem = pid % (IBAR * JBAR)
        j = rem // IBAR
        i = rem % IBAR
        idx = i * stride_i + j * stride_j + k * stride_k
        tmp_idx = (i + 1) * stride_tmp_i + (j + 1) * stride_tmp_j + (k + 1) * stride_tmp_k
        T = tl.load(tmp_ptr + tmp_idx)
        T4 = T * T * T * T
        kappa = tl.load(kappa_ptr + idx)
        I_sum = tl.zeros([], dtype=tl.float32)
        for n in range(NRA):
            I_n = tl.load(il_ptr + idx + n * stride_angle)
            I_sum += I_n
        d_omega = FOUR_PI / NRA
        emission = FOUR_PI * SIGMA_VAL * T4
        absorption = I_sum * d_omega
        qr = kappa * (emission - absorption)
        tl.store(qr_ptr + idx, qr)


def radiation_kernel(tmp, kappa, il, extcoe, dx, dy, dz, angle_x, angle_y, angle_z):
    tmp = tmp.float()
    kappa = kappa.float()
    il = il.float()
    extcoe = extcoe.float()
    IBAR, JBAR, KBAR = kappa.shape
    n_cells = IBAR * JBAR * KBAR
    if TRITON_AVAILABLE and tmp.is_cuda:
        grid = lambda meta: (triton.cdiv(n_cells, meta["BLOCK_SIZE"]),)
        _radiation_intensity_kernel[grid](
            tmp, kappa, il, extcoe,
            tmp.stride(0), tmp.stride(1), tmp.stride(2),
            kappa.stride(0), kappa.stride(1), kappa.stride(2),
            IBAR, JBAR, KBAR, dx, dy, dz,
            angle_x, angle_y, angle_z,
            SIGMA, PI, 256
        )
    else:
        T = tmp[1:-1, 1:-1, 1:-1]
        T4 = T ** 4
        B = SIGMA * T4 / PI
        abs_x = abs(angle_x) + 1e-10
        abs_y = abs(angle_y) + 1e-10
        abs_z = abs(angle_z) + 1e-10
        ds = 1.0 / (abs_x / dx + abs_y / dy + abs_z / dz)
        tau = extcoe * ds
        exp_tau = torch.exp(-tau)
        mask = extcoe > 1e-10
        source = torch.where(mask, B * kappa / extcoe * (1.0 - exp_tau), B * kappa * ds)
        il = il * exp_tau + source
    return il


def compute_radiation_source(il, kappa, tmp, nra):
    IBAR, JBAR, KBAR = kappa.shape
    qr = torch.zeros_like(kappa)
    T = tmp[1:-1, 1:-1, 1:-1]
    T4 = T ** 4
    emission = 4.0 * PI * SIGMA * T4
    d_omega = 4.0 * PI / nra
    absorption = il.sum(dim=-1) * d_omega
    qr = kappa * (emission - absorption)
    return qr


class RadiationGPUData(ctypes.Structure):
    _fields_ = [
        ("tmp_ptr", ctypes.c_void_p),
        ("kappa_gas_ptr", ctypes.c_void_p),
        ("il_ptr", ctypes.c_void_p),
        ("qr_ptr", ctypes.c_void_p),
        ("extcoe_ptr", ctypes.c_void_p),
        ("scaeff_ptr", ctypes.c_void_p),
        ("ibar", ctypes.c_int32),
        ("jbar", ctypes.c_int32),
        ("kbar", ctypes.c_int32),
        ("nra", ctypes.c_int32),
        ("nband", ctypes.c_int32),
        ("dx", ctypes.c_float),
        ("dy", ctypes.c_float),
        ("dz", ctypes.c_float),
        ("sigma", ctypes.c_float),
    ]


def radiation_compute_from_ptr(data_ptr):
    try:
        data = RadiationGPUData.from_address(data_ptr)
        ibar, jbar, kbar = data.ibar, data.jbar, data.kbar
        nra = data.nra
        dx, dy, dz = data.dx, data.dy, data.dz
        print(f"[GPU] Radiation: {ibar}x{jbar}x{kbar}, NRA={nra}")
        return 0
    except Exception as e:
        print(f"[GPU] Error: {e}")
        return -1


if __name__ == "__main__":
    print(f"Triton: {TRITON_AVAILABLE}, CUDA: {check_gpu_available()}")
