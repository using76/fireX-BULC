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
 */
typedef struct {
    void* tmp_ptr;           /* Temperature field */
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

/**
 * @brief Filter data structure for TEST_FILTER (3x3x3 box filter)
 */
typedef struct {
    void* orig_ptr;          /* Input field */
    void* hat_ptr;           /* Output filtered field */
    void* k3d_ptr;           /* 3x3x3 filter weights */
    int32_t ibar;
    int32_t jbar;
    int32_t kbar;
} FilterGPUData;

/**
 * @brief Velocity data structure for viscosity and strain rate
 */
typedef struct {
    void* u_ptr;             /* U velocity */
    void* v_ptr;             /* V velocity */
    void* w_ptr;             /* W velocity */
    void* tmp_ptr;           /* Temperature */
    void* mu_ptr;            /* Viscosity output */
    void* strain_ptr;        /* Strain rate output */
    int32_t ibar;
    int32_t jbar;
    int32_t kbar;
    float rdx;               /* 1/dx */
    float rdy;               /* 1/dy */
    float rdz;               /* 1/dz */
} VelocityGPUData;

/**
 * @brief Divergence data structure
 */
typedef struct {
    void* u_ptr;
    void* v_ptr;
    void* w_ptr;
    void* div_ptr;           /* Divergence output */
    void* rho_ptr;           /* Density */
    int32_t ibar;
    int32_t jbar;
    int32_t kbar;
    float rdx;
    float rdy;
    float rdz;
} DivergenceGPUData;

/* GPU Bridge API */
int32_t gpu_bridge_init(void);
void gpu_bridge_finalize(void);
int32_t gpu_bridge_check_gpu(void);
int32_t gpu_radiation_kernel(const RadiationGPUData* data);
int32_t gpu_filter_kernel(const FilterGPUData* data);
int32_t gpu_velocity_kernel(const VelocityGPUData* data);
int32_t gpu_divergence_kernel(const DivergenceGPUData* data);
void gpu_bridge_sync(void);

#ifdef __cplusplus
}
#endif

#endif /* FDS_GPU_BRIDGE_H */
