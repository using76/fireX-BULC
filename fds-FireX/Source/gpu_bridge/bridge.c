/**
 * @file bridge.c
 * @brief C-Python Bridge Implementation for FDS-FireX Triton GPU Kernels
 * @note Supports both FP32 and FP64 modes
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int python_initialized = 0;
static PyObject* triton_module = NULL;
static PyObject* radiation_func = NULL;
static PyThreadState* main_thread_state = NULL;

#ifndef USE_FP32
/* FP64 mode: Need conversion buffers */
static float* fp32_tmp_buf = NULL;
static float* fp32_kappa_buf = NULL;
static float* fp32_il_buf = NULL;
static float* fp32_qr_buf = NULL;
static float* fp32_extcoe_buf = NULL;
static float* fp32_scaeff_buf = NULL;
static npy_intp cached_size_3d = 0;
static npy_intp cached_size_4d = 0;
static npy_intp cached_size_tmp = 0;

static void convert_fp64_to_fp32(const double* src, float* dst, npy_intp total);
static void convert_fp32_to_fp64(const float* src, double* dst, npy_intp total);
static void free_fp32_buffers(void);
#endif

static void* init_numpy_internal(void);
static PyObject* create_numpy_array_copy_fp32(float* ptr, int ndim, npy_intp* dims, npy_intp total_size);

int32_t gpu_bridge_init(void) {
    PyObject* sys_path = NULL;
    PyObject* module_path = NULL;
    const char* kernel_path = NULL;

    if (python_initialized) return 0;

    Py_Initialize();
    if (!Py_IsInitialized()) {
        fprintf(stderr, "GPU Bridge: Failed to initialize Python\n");
        return -1;
    }

    if (init_numpy_internal() == NULL) {
        fprintf(stderr, "GPU Bridge: Failed to initialize NumPy\n");
        Py_Finalize();
        return -1;
    }

    sys_path = PySys_GetObject("path");
    if (sys_path == NULL) {
        fprintf(stderr, "GPU Bridge: Failed to get sys.path\n");
        Py_Finalize();
        return -1;
    }

    const char* paths[] = {"python_kernels", "../python_kernels", "../../python_kernels", NULL};
    for (int i = 0; paths[i] != NULL; i++) {
        module_path = PyUnicode_FromString(paths[i]);
        if (module_path) { PyList_Append(sys_path, module_path); Py_DECREF(module_path); }
    }

    kernel_path = getenv("FDS_KERNEL_PATH");
    if (kernel_path != NULL) {
        module_path = PyUnicode_FromString(kernel_path);
        if (module_path) { PyList_Insert(sys_path, 0, module_path); Py_DECREF(module_path); }
    }

    triton_module = PyImport_ImportModule("radiation");
    if (triton_module == NULL) {
        if (PyErr_Occurred()) PyErr_Print();
        fprintf(stderr, "GPU Bridge: Failed to import radiation module\n");
        Py_Finalize();
        return -1;
    }

    radiation_func = PyObject_GetAttrString(triton_module, "compute_radiation_gpu_fp32");
    if (radiation_func == NULL || !PyCallable_Check(radiation_func)) {
        if (PyErr_Occurred()) PyErr_Clear();
        radiation_func = PyObject_GetAttrString(triton_module, "compute_radiation_gpu");
        if (radiation_func == NULL || !PyCallable_Check(radiation_func)) {
            if (PyErr_Occurred()) PyErr_Print();
            fprintf(stderr, "GPU Bridge: Failed to get compute_radiation_gpu\n");
            Py_XDECREF(triton_module);
            Py_Finalize();
            return -1;
        }
    }

    python_initialized = 1;
#ifdef USE_FP32
    fprintf(stdout, "GPU Bridge: Initialized (Native FP32 - Zero Conversion)\n");
#else
    fprintf(stdout, "GPU Bridge: Initialized (FP64 with FP32 conversion)\n");
#endif

    /* Release the GIL so that PyGILState_Ensure() works from other threads */
    main_thread_state = PyEval_SaveThread();

    return 0;
}

void gpu_bridge_finalize(void) {
    if (!python_initialized) return;

    /* Restore the main thread state before calling any Python API */
    if (main_thread_state) {
        PyEval_RestoreThread(main_thread_state);
        main_thread_state = NULL;
    }

#ifndef USE_FP32
    free_fp32_buffers();
#endif
    Py_XDECREF(radiation_func);
    Py_XDECREF(triton_module);
    radiation_func = NULL;
    triton_module = NULL;
    Py_Finalize();
    python_initialized = 0;
    fprintf(stdout, "GPU Bridge: Finalized\n");
}

int32_t gpu_bridge_check_gpu(void) {
    PyObject *check_func, *result;
    int gpu_available = 0;
    PyGILState_STATE gstate;

    if (!python_initialized || !triton_module) return 0;

    gstate = PyGILState_Ensure();
    check_func = PyObject_GetAttrString(triton_module, "check_gpu_available");
    if (check_func && PyCallable_Check(check_func)) {
        result = PyObject_CallObject(check_func, NULL);
        if (result) {
            gpu_available = PyObject_IsTrue(result);
            Py_DECREF(result);
        } else {
            if (PyErr_Occurred()) PyErr_Print();
        }
        Py_DECREF(check_func);
    } else {
        if (PyErr_Occurred()) PyErr_Clear();
        fprintf(stdout, "GPU Bridge: check_gpu_available not found\n");
        /* Release the GIL before returning */
        PyGILState_Release(gstate);
        return 0;
    }

    PyGILState_Release(gstate);
    fprintf(stdout, "GPU Bridge: CUDA %s\n", gpu_available ? "available" : "not available");
    return gpu_available;
}

static void* init_numpy_internal(void) { import_array(); return (void*)1; }

/* Create NumPy array from Fortran-ordered data */
static PyObject* create_numpy_array_fp32_fortran(float* ptr, int ndim, npy_intp* dims) {
    /* Reverse dimensions for Fortran order */
    npy_intp f_dims[4];
    for (int i = 0; i < ndim; i++) {
        f_dims[i] = dims[ndim - 1 - i];
    }
    /* Create array with Fortran memory layout using reversed dimensions in C order */
    PyObject* arr = PyArray_SimpleNewFromData(ndim, f_dims, NPY_FLOAT32, (void*)ptr);
    if (arr) {
        /* Make a contiguous copy to avoid memory layout issues */
        PyObject* copy = PyArray_FROM_OTF(arr, NPY_FLOAT32, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY);
        Py_DECREF(arr);
        return copy;
    }
    return NULL;
}

/* Alternative: Create a fresh contiguous copy of data */
static PyObject* create_numpy_array_copy_fp32(float* ptr, int ndim, npy_intp* dims, npy_intp total_size) {
    npy_intp c_dims[4];
    for (int i = 0; i < ndim; i++) {
        c_dims[i] = dims[i];
    }
    PyObject* arr = PyArray_SimpleNew(ndim, c_dims, NPY_FLOAT32);
    if (arr) {
        float* data = (float*)PyArray_DATA((PyArrayObject*)arr);
        memcpy(data, ptr, total_size * sizeof(float));
    }
    return arr;
}

#ifndef USE_FP32
/* FP64 mode helper functions */
static void convert_fp64_to_fp32(const double* src, float* dst, npy_intp total) {
    for (npy_intp i = 0; i < total; i++) dst[i] = (float)src[i];
}

static void convert_fp32_to_fp64(const float* src, double* dst, npy_intp total) {
    for (npy_intp i = 0; i < total; i++) dst[i] = (double)src[i];
}

static void free_fp32_buffers(void) {
    if (fp32_tmp_buf) { free(fp32_tmp_buf); fp32_tmp_buf = NULL; }
    if (fp32_kappa_buf) { free(fp32_kappa_buf); fp32_kappa_buf = NULL; }
    if (fp32_il_buf) { free(fp32_il_buf); fp32_il_buf = NULL; }
    if (fp32_qr_buf) { free(fp32_qr_buf); fp32_qr_buf = NULL; }
    if (fp32_extcoe_buf) { free(fp32_extcoe_buf); fp32_extcoe_buf = NULL; }
    if (fp32_scaeff_buf) { free(fp32_scaeff_buf); fp32_scaeff_buf = NULL; }
    cached_size_3d = cached_size_4d = cached_size_tmp = 0;
}
#endif

int32_t gpu_radiation_kernel(const RadiationGPUData* data) {
    PyObject *args = NULL, *result = NULL;
    PyObject *tmp_arr = NULL, *kappa_arr = NULL, *il_arr = NULL;
    PyObject *qr_arr = NULL, *extcoe_arr = NULL, *scaeff_arr = NULL;
    int status = -1;
    PyGILState_STATE gstate;

    /* DEBUG: Return early to test if crash is in kernel */
    (void)data; /* Suppress unused warning */
    return 0;

    if (!python_initialized || !radiation_func || !data) return -1;

    /* Acquire the GIL for thread safety with OpenMP */
    gstate = PyGILState_Ensure();

    npy_intp size_3d = (npy_intp)data->kbar * data->jbar * data->ibar;
    npy_intp size_4d = (npy_intp)data->nra * size_3d;
    npy_intp size_tmp = (npy_intp)(data->kbar + 1) * (data->jbar + 1) * (data->ibar + 1);

    npy_intp d3[3] = {data->kbar, data->jbar, data->ibar};
    npy_intp d4[4] = {data->nra, data->kbar, data->jbar, data->ibar};
    npy_intp dt[3] = {data->kbar + 1, data->jbar + 1, data->ibar + 1};

#ifdef USE_FP32
    /* Native FP32 mode - use safe copy to avoid memory corruption */
    /* Create contiguous copies of all input arrays */
    tmp_arr = create_numpy_array_copy_fp32((float*)data->tmp_ptr, 3, dt, size_tmp);
    kappa_arr = create_numpy_array_copy_fp32((float*)data->kappa_gas_ptr, 3, d3, size_3d);
    il_arr = create_numpy_array_copy_fp32((float*)data->il_ptr, 4, d4, size_4d);
    qr_arr = create_numpy_array_copy_fp32((float*)data->qr_ptr, 3, d3, size_3d);
    extcoe_arr = create_numpy_array_copy_fp32((float*)data->extcoe_ptr, 3, d3, size_3d);
    scaeff_arr = create_numpy_array_copy_fp32((float*)data->scaeff_ptr, 3, d3, size_3d);
#else
    /* FP64 mode - convert to FP32 */
    if (size_3d != cached_size_3d || size_4d != cached_size_4d || size_tmp != cached_size_tmp) {
        free_fp32_buffers();
        fp32_tmp_buf = (float*)malloc(size_tmp * sizeof(float));
        fp32_kappa_buf = (float*)malloc(size_3d * sizeof(float));
        fp32_il_buf = (float*)malloc(size_4d * sizeof(float));
        fp32_qr_buf = (float*)malloc(size_3d * sizeof(float));
        fp32_extcoe_buf = (float*)malloc(size_3d * sizeof(float));
        fp32_scaeff_buf = (float*)malloc(size_3d * sizeof(float));
        if (!fp32_tmp_buf || !fp32_kappa_buf || !fp32_il_buf ||
            !fp32_qr_buf || !fp32_extcoe_buf || !fp32_scaeff_buf) {
            free_fp32_buffers();
            fprintf(stderr, "GPU Bridge: Failed to allocate FP32 buffers\n");
            PyGILState_Release(gstate);
            return -1;
        }
        cached_size_3d = size_3d;
        cached_size_4d = size_4d;
        cached_size_tmp = size_tmp;
    }

    /* Cast void* to double* for FP64 mode conversion */
    convert_fp64_to_fp32((const double*)data->tmp_ptr, fp32_tmp_buf, size_tmp);
    convert_fp64_to_fp32((const double*)data->kappa_gas_ptr, fp32_kappa_buf, size_3d);
    convert_fp64_to_fp32((const double*)data->il_ptr, fp32_il_buf, size_4d);
    convert_fp64_to_fp32((const double*)data->extcoe_ptr, fp32_extcoe_buf, size_3d);
    convert_fp64_to_fp32((const double*)data->scaeff_ptr, fp32_scaeff_buf, size_3d);

    tmp_arr = create_numpy_array_copy_fp32(fp32_tmp_buf, 3, dt, size_tmp);
    kappa_arr = create_numpy_array_copy_fp32(fp32_kappa_buf, 3, d3, size_3d);
    il_arr = create_numpy_array_copy_fp32(fp32_il_buf, 4, d4, size_4d);
    qr_arr = create_numpy_array_copy_fp32(fp32_qr_buf, 3, d3, size_3d);
    extcoe_arr = create_numpy_array_copy_fp32(fp32_extcoe_buf, 3, d3, size_3d);
    scaeff_arr = create_numpy_array_copy_fp32(fp32_scaeff_buf, 3, d3, size_3d);
#endif

    if (!tmp_arr || !kappa_arr || !il_arr || !qr_arr || !extcoe_arr || !scaeff_arr) goto cleanup;

    args = Py_BuildValue("(OOOOOOffff)", tmp_arr, kappa_arr, il_arr, qr_arr,
                         extcoe_arr, scaeff_arr, data->dx, data->dy, data->dz, data->sigma);
    if (!args) goto cleanup;

    result = PyObject_CallObject(radiation_func, args);
    if (!result) { if (PyErr_Occurred()) PyErr_Print(); goto cleanup; }

#ifdef USE_FP32
    /* FP32 mode - copy QR result back to Fortran array */
    if (qr_arr) {
        float* qr_data = (float*)PyArray_DATA((PyArrayObject*)qr_arr);
        memcpy((float*)data->qr_ptr, qr_data, size_3d * sizeof(float));
    }
#else
    /* FP64 mode - copy QR result from PyArray back to fp32_qr_buf, then convert to double* */
    if (qr_arr) {
        float* qr_data = (float*)PyArray_DATA((PyArrayObject*)qr_arr);
        memcpy(fp32_qr_buf, qr_data, size_3d * sizeof(float));
    }
    convert_fp32_to_fp64(fp32_qr_buf, (double*)data->qr_ptr, size_3d);
#endif

    status = PyLong_Check(result) ? (int)PyLong_AsLong(result) : 0;
    Py_DECREF(result);

cleanup:
    Py_XDECREF(args);
    Py_XDECREF(tmp_arr); Py_XDECREF(kappa_arr); Py_XDECREF(il_arr);
    Py_XDECREF(qr_arr); Py_XDECREF(extcoe_arr); Py_XDECREF(scaeff_arr);

    /* Release the GIL */
    PyGILState_Release(gstate);
    return status;
}

void gpu_bridge_sync(void) {
    PyObject *torch, *cuda, *sync_func, *result;
    PyGILState_STATE gstate;

    if (!python_initialized) return;

    gstate = PyGILState_Ensure();

    torch = PyImport_ImportModule("torch");
    if (!torch) {
        PyGILState_Release(gstate);
        return;
    }
    cuda = PyObject_GetAttrString(torch, "cuda");
    if (cuda) {
        sync_func = PyObject_GetAttrString(cuda, "synchronize");
        if (sync_func && PyCallable_Check(sync_func)) {
            result = PyObject_CallObject(sync_func, NULL);
            Py_XDECREF(result);
            Py_DECREF(sync_func);
        }
        Py_DECREF(cuda);
    }
    Py_DECREF(torch);

    PyGILState_Release(gstate);
}
