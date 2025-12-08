/**
 * @file bridge.h
 * @brief C-Python Bridge Header for FDS-FireX Triton GPU Kernels
 */

#ifndef FDS_GPU_BRIDGE_H
#define FDS_GPU_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Define REAL_EB type based on precision mode */
#ifdef USE_FP32
typedef float REAL_EB;
#else
typedef double REAL_EB;
#endif

/**
 * @brief Radiation data structure matching Fortran BIND(C) type
 *
 * IMPORTANT: All pointer fields MUST be void* to match Fortran's C_PTR
 * which is always 8 bytes on 64-bit systems. Using REAL_EB* would cause
 * struct layout mismatch in FP32 mode (float* = 4 bytes on some compilers).
 *
 * FP32 mode: void* cast to float* in bridge.c (no conversion needed)
 * FP64 mode: void* cast to double* in bridge.c, then converted to FP32 for GPU
 */
typedef struct {
    void* tmp_ptr;           /* Temperature field (C_PTR from Fortran) */
    void* kappa_gas_ptr;     /* Gas absorption coefficient */
    void* il_ptr;            /* Radiation intensity (in/out) */
    void* qr_ptr;            /* Radiation source term (out) */
    void* extcoe_ptr;        /* Extinction coefficient */
    void* scaeff_ptr;        /* Scattering efficiency */
    int32_t ibar;            /* Grid cells in X */
    int32_t jbar;            /* Grid cells in Y */
    int32_t kbar;            /* Grid cells in Z */
    int32_t nra;             /* Number of radiation angles */
    int32_t nband;           /* Number of spectral bands */
    float dx;                /* Grid spacing X */
    float dy;                /* Grid spacing Y */
    float dz;                /* Grid spacing Z */
    float sigma;             /* Stefan-Boltzmann constant */
} RadiationGPUData;

/* GPU Bridge API */
int32_t gpu_bridge_init(void);
void gpu_bridge_finalize(void);
int32_t gpu_bridge_check_gpu(void);
int32_t gpu_radiation_kernel(const RadiationGPUData* data);
void gpu_bridge_sync(void);

#ifdef __cplusplus
}
#endif

#endif /* FDS_GPU_BRIDGE_H */
