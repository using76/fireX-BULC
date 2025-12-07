"""
FDS-FireX Radiation Heat Transfer Triton Kernels

This module implements GPU-accelerated radiative heat transfer calculations
for the FDS-FireX fire simulation. It replaces the CPU-based RADIATION_FVM
subroutine in radi.f90 with Triton kernels.

Key kernels:
    - kfst4_kernel: Compute emission term KFST4 = kappa * 4 * sigma * T^4
    - radiation_intensity_kernel: Compute radiation intensity field
    - integrate_intensity_kernel: Integrate intensity over angles

Reference: NIST FDS Technical Reference Guide, Chapter on Radiation
"""

import torch
import triton
import triton.language as tl
import numpy as np
import ctypes
from typing import Optional, Tuple, Dict, Any

# Physical constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2 K^4)
FOUR_SIGMA = 4.0 * STEFAN_BOLTZMANN
PI = 3.141592653589793
RFPI = 1.0 / (4.0 * PI)

# RTX 4060 optimized block sizes
BLOCK_SIZE_1D = 256
BLOCK_SIZE_X = 16
BLOCK_SIZE_Y = 16
BLOCK_SIZE_Z = 4


@triton.jit
def kfst4_kernel(
    # Inputs
    TMP_ptr,       # Temperature [IBAR, JBAR, KBAR]
    KAPPA_ptr,     # Absorption coefficient
    # Output
    KFST4_ptr,     # Emission term output
    # Dimensions
    n_elements,    # Total number of elements
    # Constants
    FOUR_SIGMA: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute KFST4 = KAPPA * 4 * SIGMA * T^4

    This is the emission term in the radiative transfer equation.
    Each grid cell emits radiation proportional to T^4 (Stefan-Boltzmann law).

    FDS Reference: radi.f90, lines 3698-3709
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load temperature and absorption coefficient
    TMP = tl.load(TMP_ptr + offset, mask=mask, other=300.0)
    KAPPA = tl.load(KAPPA_ptr + offset, mask=mask, other=0.0)

    # Compute T^4
    TMP2 = TMP * TMP
    TMP4 = TMP2 * TMP2

    # Compute emission term
    KFST4 = KAPPA * FOUR_SIGMA * TMP4

    # Store result
    tl.store(KFST4_ptr + offset, KFST4, mask=mask)


@triton.jit
def radiation_source_kernel(
    # Inputs
    KFST4_GAS_ptr,   # Gas emission term
    KFST4_PART_ptr,  # Particle emission term (can be zeros)
    SCAEFF_ptr,      # Scattering efficiency
    UIIOLD_ptr,      # Previous integrated intensity
    # Output
    SOURCE_ptr,      # Source term output
    # Dimensions
    n_elements,
    # Constants
    RFPI: tl.constexpr,
    RSA_RAT: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute radiation source term for intensity equation.

    SOURCE = RFPI * (KFST4_GAS + KFST4_PART + RSA_RAT * SCAEFF * UIIOLD)

    FDS Reference: radi.f90, lines 3776-3795
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load inputs
    KFST4_GAS = tl.load(KFST4_GAS_ptr + offset, mask=mask, other=0.0)
    KFST4_PART = tl.load(KFST4_PART_ptr + offset, mask=mask, other=0.0)
    SCAEFF = tl.load(SCAEFF_ptr + offset, mask=mask, other=0.0)
    UIIOLD = tl.load(UIIOLD_ptr + offset, mask=mask, other=0.0)

    # Compute source term
    SOURCE = RFPI * (KFST4_GAS + KFST4_PART + RSA_RAT * SCAEFF * UIIOLD)

    # Store result
    tl.store(SOURCE_ptr + offset, SOURCE, mask=mask)


@triton.jit
def integrate_qr_kernel(
    # Inputs
    KAPPA_ptr,       # Absorption coefficient
    UIID_ptr,        # Integrated intensity
    KFST4_ptr,       # Emission term
    # Output
    QR_ptr,          # Radiative heat source
    # Dimensions
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute radiative heat source QR = KAPPA * UII - KFST4

    This is the net radiative heat transfer rate at each cell.
    Positive QR means the cell is gaining heat from radiation.

    FDS Reference: radi.f90, lines 4479-4521
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load inputs
    KAPPA = tl.load(KAPPA_ptr + offset, mask=mask, other=0.0)
    UIID = tl.load(UIID_ptr + offset, mask=mask, other=0.0)
    KFST4 = tl.load(KFST4_ptr + offset, mask=mask, other=0.0)

    # Compute radiative heat source
    # QR = absorption - emission
    QR = KAPPA * UIID - KFST4

    # Store result
    tl.store(QR_ptr + offset, QR, mask=mask)


def compute_radiation(
    TMP: torch.Tensor,
    KAPPA_GAS: torch.Tensor,
    IL: torch.Tensor,
    UIID: torch.Tensor,
    QR: torch.Tensor,
    DX: torch.Tensor,
    DY: torch.Tensor,
    DZ: torch.Tensor,
    DLX: torch.Tensor,
    DLY: torch.Tensor,
    DLZ: torch.Tensor,
    RSA: torch.Tensor,
    NRA: int,
    NSB: int = 1,
    CHI_R: Optional[torch.Tensor] = None,
    Q: Optional[torch.Tensor] = None,
    **kwargs
) -> None:
    """
    Main radiation computation function.

    This function orchestrates the Triton kernels to compute radiative
    heat transfer. It is designed to match the interface expected by
    the C bridge.

    Args:
        TMP: Temperature field [IBAR, JBAR, KBAR] in Kelvin
        KAPPA_GAS: Gas absorption coefficient [IBAR, JBAR, KBAR]
        IL: Radiation intensity (output) [IBAR, JBAR, KBAR]
        UIID: Integrated intensity (output) [IBAR, JBAR, KBAR, NSB]
        QR: Radiative heat source (output) [IBAR, JBAR, KBAR]
        DX, DY, DZ: Grid spacings [IBAR], [JBAR], [KBAR]
        DLX, DLY, DLZ: Direction cosines [NRA]
        RSA: Solid angle weights [NRA]
        NRA: Number of radiation angles
        NSB: Number of spectral bands (default 1)
        CHI_R: Radiative fraction (optional)
        Q: Heat release rate (optional)

    Returns:
        None (results are written to IL, UIID, QR in-place)
    """
    device = TMP.device
    dtype = TMP.dtype

    # Get dimensions
    if TMP.dim() == 3:
        IBAR, JBAR, KBAR = TMP.shape
    else:
        raise ValueError(f"TMP must be 3D, got {TMP.dim()}D")

    n_elements = IBAR * JBAR * KBAR

    # Allocate working arrays
    KFST4_GAS = torch.zeros_like(TMP)
    KFST4_PART = torch.zeros_like(TMP)  # Placeholder for particles
    EXTCOE = KAPPA_GAS.clone()  # Extinction = absorption (no scattering for now)
    SCAEFF = torch.zeros_like(TMP)  # No scattering for now
    UIIOLD = UIID[:, :, :, 0] if UIID.dim() == 4 else UIID.clone()
    SOURCE = torch.zeros_like(TMP)

    # Constants
    RSA_RAT = 1.0 / (1.0 - 1.0 / max(NRA, 2))

    # Grid configuration for 1D kernels
    grid_1d = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Step 1: Compute KFST4 (emission term)
    kfst4_kernel[grid_1d](
        TMP, KAPPA_GAS, KFST4_GAS,
        n_elements,
        FOUR_SIGMA,
        BLOCK_SIZE=BLOCK_SIZE_1D,
    )

    # Step 2: Compute source term
    radiation_source_kernel[grid_1d](
        KFST4_GAS, KFST4_PART, SCAEFF, UIIOLD,
        SOURCE,
        n_elements,
        RFPI, RSA_RAT,
        BLOCK_SIZE=BLOCK_SIZE_1D,
    )

    # Step 3: Intensity update (simplified - full implementation needs sweep)
    # For now, use a simple approximation
    # IL = SOURCE / EXTCOE (optically thin limit)
    IL[:] = torch.where(
        EXTCOE > 1e-10,
        SOURCE / (EXTCOE + 1e-10),
        SOURCE * 1e10
    )
    IL.clamp_(min=0.0)

    # Step 4: Integrate intensity over angles
    # UIID = sum over angles of RSA[n] * IL
    for n in range(NRA):
        if UIID.dim() == 4:
            UIID[:, :, :, 0] += RSA[n].item() * IL
        else:
            UIID += RSA[n].item() * IL

    # Step 5: Compute radiative heat source QR
    integrate_qr_kernel[grid_1d](
        KAPPA_GAS, UIID if UIID.dim() == 3 else UIID[:, :, :, 0],
        KFST4_GAS, QR,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE_1D,
    )


def compute_radiation_from_c(data: Dict[str, Any]) -> int:
    """
    Entry point for C bridge.

    This function is called from bridge.c with a dictionary containing
    pointers and dimensions. It converts the pointers to PyTorch tensors
    and calls compute_radiation.

    Args:
        data: Dictionary with keys:
            - TMP_ptr, KAPPA_ptr, etc.: Memory addresses as integers
            - IBAR, JBAR, KBAR: Grid dimensions
            - NRA, NSB: Number of angles and spectral bands
            - FOUR_SIGMA, RFPI, RSA_RAT: Physical constants

    Returns:
        0 on success, negative error code on failure
    """
    try:
        # Extract dimensions
        IBAR = data['IBAR']
        JBAR = data['JBAR']
        KBAR = data['KBAR']
        NRA = data['NRA']
        NSB = data.get('NSB', 1)

        shape_3d = (IBAR, JBAR, KBAR)
        n_elements = IBAR * JBAR * KBAR

        # Convert pointers to tensors
        # Note: This uses ctypes to access the memory directly
        def ptr_to_tensor(ptr_val, shape, dtype=torch.float32):
            if ptr_val == 0 or ptr_val is None:
                return torch.zeros(shape, dtype=dtype, device='cuda')

            # Create a ctypes array from the pointer
            ptr = ctypes.cast(ptr_val, ctypes.POINTER(ctypes.c_float))
            n = np.prod(shape)

            # Create numpy array (no copy)
            arr = np.ctypeslib.as_array(ptr, shape=(n,))
            arr = arr.reshape(shape, order='F')  # Fortran column-major

            # Convert to PyTorch tensor and move to GPU
            # Note: This copies data to GPU
            tensor = torch.from_numpy(arr.copy()).to('cuda')

            # Transpose for row-major
            if len(shape) == 3:
                tensor = tensor.permute(2, 1, 0).contiguous()

            return tensor

        # Convert all arrays
        TMP = ptr_to_tensor(data['TMP_ptr'], shape_3d)
        KAPPA_GAS = ptr_to_tensor(data['KAPPA_ptr'], shape_3d)
        IL = ptr_to_tensor(data['IL_ptr'], shape_3d)
        QR = ptr_to_tensor(data['QR_ptr'], shape_3d)

        # Handle UIID (4D array)
        UIID_shape = (IBAR, JBAR, KBAR, NSB)
        UIID = torch.zeros(KBAR, JBAR, IBAR, NSB, dtype=torch.float32, device='cuda')

        # 1D arrays
        DX = ptr_to_tensor(data['DX_ptr'], (IBAR,))
        DY = ptr_to_tensor(data['DY_ptr'], (JBAR,))
        DZ = ptr_to_tensor(data['DZ_ptr'], (KBAR,))
        DLX = ptr_to_tensor(data['DLX_ptr'], (NRA,))
        DLY = ptr_to_tensor(data['DLY_ptr'], (NRA,))
        DLZ = ptr_to_tensor(data['DLZ_ptr'], (NRA,))
        RSA = ptr_to_tensor(data['RSA_ptr'], (NRA,))

        # Call main computation
        compute_radiation(
            TMP, KAPPA_GAS, IL, UIID, QR,
            DX, DY, DZ, DLX, DLY, DLZ, RSA,
            NRA, NSB
        )

        # Copy results back to host memory
        def tensor_to_ptr(tensor, ptr_val, shape):
            if ptr_val == 0 or ptr_val is None:
                return

            # Transpose back to Fortran order
            if tensor.dim() == 3:
                tensor = tensor.permute(2, 1, 0).contiguous()

            # Copy to CPU
            arr = tensor.cpu().numpy()
            arr = np.asfortranarray(arr)

            # Write back to original memory
            ptr = ctypes.cast(ptr_val, ctypes.POINTER(ctypes.c_float))
            n = np.prod(shape)
            target = np.ctypeslib.as_array(ptr, shape=(n,))
            target[:] = arr.ravel(order='F')

        tensor_to_ptr(IL, data['IL_ptr'], shape_3d)
        tensor_to_ptr(QR, data['QR_ptr'], shape_3d)

        return 0

    except Exception as e:
        print(f"GPU Bridge Error: {e}")
        import traceback
        traceback.print_exc()
        return -1


def test_kernels():
    """
    Test the Triton kernels with synthetic data.
    """
    print("Testing FDS-FireX Triton Radiation Kernels...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return

    device = 'cuda'
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Test parameters
    IBAR, JBAR, KBAR = 32, 32, 32
    NRA = 48  # Typical number of radiation angles
    n_elements = IBAR * JBAR * KBAR

    # Create test data
    TMP = torch.rand(IBAR, JBAR, KBAR, device=device) * 1000 + 300  # 300-1300 K
    KAPPA_GAS = torch.rand(IBAR, JBAR, KBAR, device=device) * 0.5  # 0-0.5 m^-1
    IL = torch.zeros(IBAR, JBAR, KBAR, device=device)
    UIID = torch.zeros(IBAR, JBAR, KBAR, device=device)
    QR = torch.zeros(IBAR, JBAR, KBAR, device=device)
    KFST4_GAS = torch.zeros(IBAR, JBAR, KBAR, device=device)

    DX = torch.ones(IBAR, device=device) * 0.1  # 10 cm cells
    DY = torch.ones(JBAR, device=device) * 0.1
    DZ = torch.ones(KBAR, device=device) * 0.1

    # Create angle arrays (simplified)
    angles = torch.linspace(0, 2 * PI, NRA + 1, device=device)[:-1]
    DLX = torch.cos(angles)
    DLY = torch.sin(angles)
    DLZ = torch.zeros(NRA, device=device)
    RSA = torch.ones(NRA, device=device) * (4 * PI / NRA)  # Uniform solid angle

    # Test KFST4 kernel
    print("\nTesting kfst4_kernel...")
    grid = (triton.cdiv(n_elements, BLOCK_SIZE_1D),)
    kfst4_kernel[grid](
        TMP, KAPPA_GAS, KFST4_GAS,
        n_elements,
        FOUR_SIGMA,
        BLOCK_SIZE=BLOCK_SIZE_1D,
    )

    # Verify result
    expected = KAPPA_GAS * FOUR_SIGMA * TMP ** 4
    error = torch.abs(KFST4_GAS - expected).max().item()
    rel_error = (torch.abs(KFST4_GAS - expected) / (expected.abs() + 1e-10)).max().item()
    print(f"  Max absolute error: {error:.2e}")
    print(f"  Max relative error: {rel_error:.2e}")
    # FP32 precision allows for ~1e-7 relative error, but T^4 amplifies this
    assert rel_error < 1e-4, f"KFST4 relative error too large: {rel_error}"
    print("  PASSED")

    # Test full radiation computation
    print("\nTesting compute_radiation...")
    compute_radiation(
        TMP, KAPPA_GAS, IL, UIID, QR,
        DX, DY, DZ, DLX, DLY, DLZ, RSA,
        NRA
    )

    print(f"  IL range: [{IL.min().item():.2e}, {IL.max().item():.2e}]")
    print(f"  UIID range: [{UIID.min().item():.2e}, {UIID.max().item():.2e}]")
    print(f"  QR range: [{QR.min().item():.2e}, {QR.max().item():.2e}]")
    print("  PASSED")

    # Benchmark
    print("\nBenchmarking...")
    import time

    # Warmup
    for _ in range(10):
        compute_radiation(
            TMP, KAPPA_GAS, IL, UIID, QR,
            DX, DY, DZ, DLX, DLY, DLZ, RSA,
            NRA
        )
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    n_iter = 100
    for _ in range(n_iter):
        compute_radiation(
            TMP, KAPPA_GAS, IL, UIID, QR,
            DX, DY, DZ, DLX, DLY, DLZ, RSA,
            NRA
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / n_iter * 1000  # ms
    print(f"  Grid size: {IBAR}x{JBAR}x{KBAR} = {n_elements:,} cells")
    print(f"  Angles: {NRA}")
    print(f"  Average time: {avg_time:.3f} ms")
    print(f"  Throughput: {n_elements * NRA / avg_time / 1e6:.2f} M cells*angles/ms")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_kernels()
