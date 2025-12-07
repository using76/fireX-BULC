/**
 * @file bridge.h
 * @brief C-Python Bridge for Triton GPU Kernels
 *
 * This header defines the interface between Fortran/C and Python
 * for GPU-accelerated computations using Triton kernels.
 */

#ifndef GPU_BRIDGE_H
#define GPU_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Radiation data structure matching Fortran BIND(C) type
 *
 * All pointers are to GPU-accessible memory (pinned or unified).
 * Arrays use Fortran column-major ordering.
 */
typedef struct {
    float *TMP;           /**< Temperature array [IBAR, JBAR, KBAR] */
    float *KAPPA_GAS;     /**< Gas absorption coefficient */
    float *KFST4_GAS;     /**< Emission term (kappa * 4 * sigma * T^4) */
    float *IL;            /**< Radiation intensity */
    float *UIID;          /**< Integrated intensity */
    float *QR;            /**< Radiative heat source */
    float *EXTCOE;        /**< Extinction coefficient */
    float *SCAEFF;        /**< Scattering efficiency */
    float *UIIOLD;        /**< Previous integrated intensity */
    float *DX;            /**< Grid spacing X [IBAR] */
    float *DY;            /**< Grid spacing Y [JBAR] */
    float *DZ;            /**< Grid spacing Z [KBAR] */
    float *DLX;           /**< Direction cosine X [NRA] */
    float *DLY;           /**< Direction cosine Y [NRA] */
    float *DLZ;           /**< Direction cosine Z [NRA] */
    float *RSA;           /**< Solid angle weights [NRA] */
    int32_t IBAR;         /**< Number of cells in X */
    int32_t JBAR;         /**< Number of cells in Y */
    int32_t KBAR;         /**< Number of cells in Z */
    int32_t NRA;          /**< Number of radiation angles */
    int32_t NSB;          /**< Number of spectral bands */
    float FOUR_SIGMA;     /**< 4 * Stefan-Boltzmann constant */
    float RFPI;           /**< 1 / (4 * PI) */
    float RSA_RAT;        /**< Solid angle ratio */
} RadiationGPUData;

/**
 * @brief Initialize Python runtime and load Triton modules
 * @return 0 on success, negative error code on failure
 *
 * Error codes:
 *  -1: Python interpreter initialization failed
 *  -2: Failed to import radiation module
 *  -3: Failed to get kernel function
 */
int init_python_runtime(void);

/**
 * @brief Finalize Python runtime
 *
 * Clean up Python interpreter and release all resources.
 * Should be called once at program termination.
 */
void finalize_python_runtime(void);

/**
 * @brief Call Triton radiation kernel
 * @param data Pointer to radiation data structure
 * @return 0 on success, negative error code on failure
 *
 * This function:
 * 1. Converts Fortran arrays to NumPy/PyTorch tensors (zero-copy when possible)
 * 2. Calls the Triton kernel through Python
 * 3. Results are written back to the same memory locations
 */
int call_radiation_kernel(const RadiationGPUData *data);

/**
 * @brief Allocate GPU-accessible memory
 * @param size_bytes Size in bytes to allocate
 * @return Pointer to allocated memory, NULL on failure
 *
 * Allocates pinned (page-locked) memory for optimal CPU-GPU transfer.
 */
void* allocate_gpu_array(size_t size_bytes);

/**
 * @brief Free GPU-accessible memory
 * @param ptr Pointer to memory to free
 */
void free_gpu_array(void *ptr);

/**
 * @brief Synchronize GPU operations
 *
 * Blocks until all GPU operations are complete.
 */
void gpu_sync(void);

/**
 * @brief Check if GPU/Triton is available
 * @return 1 if available, 0 otherwise
 */
int is_gpu_available(void);

/**
 * @brief Get GPU device information
 * @param device_name Buffer to store device name
 * @param max_len Maximum length of buffer
 * @return 0 on success, -1 on failure
 */
int get_gpu_device_info(char *device_name, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif /* GPU_BRIDGE_H */
