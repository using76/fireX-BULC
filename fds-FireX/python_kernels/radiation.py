"""
FDS-FireX Radiation Heat Transfer Triton Kernels
GPU-accelerated radiation computation for fire simulation
"""

import torch
import numpy as np
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available, using PyTorch fallback")

SIGMA = 5.670374419e-8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_gpu_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


if TRITON_AVAILABLE:
    @triton.jit
    def kfst4_kernel(
        tmp_ptr, kappa_ptr, kfst4_ptr, sigma, total_cells,
        tmp_stride_j, tmp_stride_k,
        IBAR: tl.constexpr, JBAR: tl.constexpr, BLOCK_SIZE: tl.constexpr = 256,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_cells
        idx = offsets
        i = idx % IBAR
        j = (idx // IBAR) % JBAR
        k = idx // (IBAR * JBAR)
        tmp_idx = i + j * tmp_stride_j + k * tmp_stride_k
        T = tl.load(tmp_ptr + tmp_idx, mask=mask, other=300.0)
        kappa = tl.load(kappa_ptr + offsets, mask=mask, other=0.0)
        T4 = T * T * T * T
        kfst4 = kappa * 4.0 * sigma * T4
        tl.store(kfst4_ptr + offsets, kfst4, mask=mask)

    @triton.jit
    def radiation_transport_kernel(
        il_ptr, kfst4_ptr, extcoe_ptr, qr_ptr,
        total_cells, cells_per_angle, angle_weight,
        NRA: tl.constexpr, BLOCK_SIZE: tl.constexpr = 256,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_cells
        kfst4 = tl.load(kfst4_ptr + offsets, mask=mask, other=0.0)
        extcoe = tl.load(extcoe_ptr + offsets, mask=mask, other=0.0)
        il_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for n in range(NRA):
            il_idx = n * cells_per_angle + offsets
            il_val = tl.load(il_ptr + il_idx, mask=mask, other=0.0)
            il_sum += il_val
        qr = kfst4 - extcoe * il_sum * angle_weight
        tl.store(qr_ptr + offsets, qr, mask=mask)


def compute_radiation_gpu_fp32(tmp, kappa_gas, il, qr, extcoe, scaeff, dx, dy, dz, sigma=SIGMA):
    """FP32-optimized - no conversion overhead"""
    try:
        if not torch.cuda.is_available():
            return _compute_radiation_cpu(tmp, kappa_gas, il, qr, extcoe, scaeff, dx, dy, dz, sigma)
        kbar, jbar, ibar = kappa_gas.shape
        nra = il.shape[0] if il.ndim == 4 else 1
        tmp_t = torch.from_numpy(tmp).to(DEVICE).contiguous()
        kappa_t = torch.from_numpy(kappa_gas).to(DEVICE).contiguous()
        il_t = torch.from_numpy(il).to(DEVICE).contiguous()
        extcoe_t = torch.from_numpy(extcoe).to(DEVICE).contiguous()
        kfst4_t = torch.zeros_like(kappa_t)
        qr_t = torch.zeros_like(kappa_t)
        total_cells = ibar * jbar * kbar
        BLOCK_SIZE = 256
        if TRITON_AVAILABLE:
            grid = (triton.cdiv(total_cells, BLOCK_SIZE),)
            tmp_stride_j = ibar + 1
            tmp_stride_k = (ibar + 1) * (jbar + 1)
            angle_weight = 4.0 * 3.14159265359 / nra
            kfst4_kernel[grid](tmp_t, kappa_t, kfst4_t, sigma, total_cells,
                              tmp_stride_j, tmp_stride_k, IBAR=ibar, JBAR=jbar, BLOCK_SIZE=BLOCK_SIZE)
            radiation_transport_kernel[grid](il_t, kfst4_t, extcoe_t, qr_t,
                                            total_cells, total_cells, angle_weight,
                                            NRA=nra, BLOCK_SIZE=BLOCK_SIZE)
        else:
            T = tmp_t[:kbar, :jbar, :ibar]
            T4 = T ** 4
            kfst4_t = kappa_t * 4.0 * sigma * T4
            il_sum = il_t.sum(dim=0) if il_t.ndim == 4 else il_t
            angle_weight = 4.0 * 3.14159265359 / nra
            qr_t = kfst4_t - extcoe_t * il_sum * angle_weight
        qr[:] = qr_t.cpu().numpy()
        torch.cuda.synchronize()
        return 0
    except Exception as e:
        print(f"GPU radiation FP32 failed: {e}")
        return -1


def compute_radiation_gpu(tmp, kappa_gas, il, qr, extcoe, scaeff, dx, dy, dz, sigma=SIGMA):
    """Legacy FP64 path with conversion"""
    try:
        if not torch.cuda.is_available():
            return _compute_radiation_cpu(tmp, kappa_gas, il, qr, extcoe, scaeff, dx, dy, dz, sigma)
        kbar, jbar, ibar = kappa_gas.shape
        nra = il.shape[0] if il.ndim == 4 else 1
        tmp_t = torch.from_numpy(tmp.astype(np.float32)).to(DEVICE).contiguous()
        kappa_t = torch.from_numpy(kappa_gas.astype(np.float32)).to(DEVICE).contiguous()
        il_t = torch.from_numpy(il.astype(np.float32)).to(DEVICE).contiguous()
        extcoe_t = torch.from_numpy(extcoe.astype(np.float32)).to(DEVICE).contiguous()
        kfst4_t = torch.zeros_like(kappa_t)
        qr_t = torch.zeros_like(kappa_t)
        total_cells = ibar * jbar * kbar
        BLOCK_SIZE = 256
        if TRITON_AVAILABLE:
            grid = (triton.cdiv(total_cells, BLOCK_SIZE),)
            tmp_stride_j = ibar + 1
            tmp_stride_k = (ibar + 1) * (jbar + 1)
            angle_weight = 4.0 * 3.14159265359 / nra
            kfst4_kernel[grid](tmp_t, kappa_t, kfst4_t, sigma, total_cells,
                              tmp_stride_j, tmp_stride_k, IBAR=ibar, JBAR=jbar, BLOCK_SIZE=BLOCK_SIZE)
            radiation_transport_kernel[grid](il_t, kfst4_t, extcoe_t, qr_t,
                                            total_cells, total_cells, angle_weight,
                                            NRA=nra, BLOCK_SIZE=BLOCK_SIZE)
        else:
            T = tmp_t[:kbar, :jbar, :ibar]
            T4 = T ** 4
            kfst4_t = kappa_t * 4.0 * sigma * T4
            il_sum = il_t.sum(dim=0) if il_t.ndim == 4 else il_t
            angle_weight = 4.0 * 3.14159265359 / nra
            qr_t = kfst4_t - extcoe_t * il_sum * angle_weight
        qr[:] = qr_t.cpu().numpy()
        torch.cuda.synchronize()
        return 0
    except Exception as e:
        print(f"GPU radiation failed: {e}")
        return -1


def _compute_radiation_cpu(tmp, kappa_gas, il, qr, extcoe, scaeff, dx, dy, dz, sigma):
    try:
        nra = il.shape[0] if il.ndim == 4 else 1
        T = tmp[:-1, :-1, :-1]
        T4 = T ** 4
        kfst4 = kappa_gas * 4.0 * sigma * T4
        il_sum = np.sum(il, axis=0) if il.ndim == 4 else il
        angle_weight = 4.0 * np.pi / nra
        qr[:] = kfst4 - extcoe * il_sum * angle_weight
        return 0
    except Exception as e:
        print(f"CPU radiation failed: {e}")
        return -1


def test_kernels():
    print("Testing FDS-FireX Triton radiation kernels...")
    ibar, jbar, kbar, nra = 25, 25, 16, 48
    tmp = np.ones((kbar+1, jbar+1, ibar+1), dtype=np.float32) * 300.0
    tmp[kbar//2, jbar//2, ibar//2] = 1000.0
    kappa_gas = np.ones((kbar, jbar, ibar), dtype=np.float32) * 0.1
    il = np.zeros((nra, kbar, jbar, ibar), dtype=np.float32)
    qr = np.zeros((kbar, jbar, ibar), dtype=np.float32)
    extcoe = np.ones((kbar, jbar, ibar), dtype=np.float32) * 0.1
    scaeff = np.zeros((kbar, jbar, ibar), dtype=np.float32)
    status = compute_radiation_gpu_fp32(tmp, kappa_gas, il, qr, extcoe, scaeff, 0.1, 0.1, 0.1)
    print(f"FP32 Status: {status}, QR range: [{qr.min():.2e}, {qr.max():.2e}]")
    return status


if __name__ == "__main__":
    test_kernels()
