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

/**
 * @brief Radiation data structure matching Fortran BIND(C) type
 * Note: Uses double (FP64) for array pointers to match FDS REAL(EB)
 *       Python/Triton converts to FP32 internally for GPU computation
 */
typedef struct {
    double* tmp_ptr;        /* Temperature field (FP64) */
    double* kappa_gas_ptr;  /* Gas absorption coefficient (FP64) */
    double* il_ptr;         /* Radiation intensity (in/out) (FP64) */
    double* qr_ptr;         /* Radiation source term (out) (FP64) */
    double* extcoe_ptr;     /* Extinction coefficient (FP64) */
    double* scaeff_ptr;     /* Scattering efficiency (FP64) */
    int32_t ibar;           /* Grid cells in X */
    int32_t jbar;           /* Grid cells in Y */
    int32_t kbar;           /* Grid cells in Z */
    int32_t nra;            /* Number of radiation angles */
    int32_t nband;          /* Number of spectral bands */
    float dx;               /* Grid spacing X */
    float dy;               /* Grid spacing Y */
    float dz;               /* Grid spacing Z */
    float sigma;            /* Stefan-Boltzmann constant */
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
