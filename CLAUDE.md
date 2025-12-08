# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FireX** is a GPU-accelerated branch of NIST Fire Dynamics Simulator (FDS) v6.9.1. It integrates Triton GPU kernels with the Fortran simulation engine via a C-Python bridge for high-performance fire simulation.

**Architecture Stack:**
```
FDS Main Loop (Fortran 2018)
    ↓ [ISO_C_BINDING]
C-Bridge Layer (Source/gpu_bridge/)
    ↓ [Embedded Python]
Triton GPU Kernels (python_kernels/)
    ↓ [JIT Compilation]
CUDA/HIP/SYCL Backend
```

## Build Commands

### Standard Build (CMake)
```bash
cd fds-FireX
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### GPU-Accelerated Build (Triton)
```bash
cmake .. -DUSE_TRITON=ON
make -j4
```

### FP32 GPU Build (experimental)
```bash
cmake .. -DUSE_TRITON=ON -DUSE_FP32=ON
make -j4
```

### Windows Build (MSYS2/MinGW)
```bash
# From MSYS2 UCRT64 shell
cd /c/Users/ji/Documents/fireX
mkdir build_gpu && cd build_gpu
cmake ../fds-FireX -G "MinGW Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_TRITON=ON \
  -DPython3_EXECUTABLE="C:/Users/ji/AppData/Local/Programs/Python/Python311/python.exe"
mingw32-make -j4
```

### Run Simulation
```bash
# Set kernel path for GPU builds
export FDS_KERNEL_PATH="/path/to/fds-FireX/python_kernels"
mpiexec -n 1 ./fds case.fds
```

## Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_TRITON` | OFF | Enable Triton GPU kernels |
| `USE_FP32` | OFF | Single precision mode (GPU optimized) |
| `USE_HYPRE` | ON | HYPRE pressure solver |
| `USE_SUNDIALS` | ON | SUNDIALS chemistry (FP64 only) |
| `USE_OPENMP` | ON | OpenMP threading |

## Project Structure

```
fds-FireX/
├── Source/                 # Fortran simulation engine
│   ├── main.f90           # Entry point, GPU initialization
│   ├── prec.f90           # Precision parameters (EB type)
│   ├── radi.f90           # Radiation solver (GPU target)
│   └── gpu_bridge/        # C-Python bridge
│       ├── bridge.f90     # Fortran interface (ISO_C_BINDING)
│       ├── bridge.c       # Python embedding, NumPy arrays
│       └── bridge.h       # Shared type definitions
├── python_kernels/         # Triton GPU kernels
│   └── radiation.py       # Radiation compute kernels
├── Build/                  # Legacy build scripts
└── CMakeLists.txt         # Primary build configuration
```

## GPU Bridge Interface

**Key Functions (bridge.f90):**
- `gpu_bridge_init()` - Initialize Python runtime
- `gpu_bridge_finalize()` - Cleanup
- `gpu_bridge_check_gpu()` - Check CUDA availability
- `gpu_radiation_kernel(data)` - Execute radiation on GPU

**Data Structure:**
```fortran
TYPE, BIND(C) :: RadiationGPUData
   TYPE(C_PTR) :: tmp_ptr, kappa_gas_ptr, il_ptr, qr_ptr
   INTEGER(C_INT) :: ibar, jbar, kbar, nra
   REAL(C_FLOAT) :: dx, dy, dz, sigma
END TYPE
```

## Precision Modes

- **FP64 (default)**: Full precision, SUNDIALS compatible, FP64→FP32 conversion for GPU
- **FP32 (USE_FP32=ON)**: Native GPU precision, no SUNDIALS, zero conversion overhead

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `FDS_KERNEL_PATH` | Custom Triton kernel directory |
| `CUDA_HOME` | CUDA installation path |
| `Python3_EXECUTABLE` | Python interpreter for embedded runtime |

## Development Notes

- Triton kernels require PyTorch and triton packages
- SUNDIALS is incompatible with FP32 mode (auto-disabled)
- GPU data uses zero-copy strategy via C pointers
- Bridge supports automatic fallback to CPU if GPU unavailable
