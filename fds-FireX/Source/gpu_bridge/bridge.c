/**
 * @file bridge.c
 * @brief C-Python Bridge Implementation for FDS-FireX Triton GPU Kernels
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

static void* init_numpy_internal(void);
static PyObject* create_numpy_array_from_ptr(double* ptr, int ndim, npy_intp* dims);

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
    
    radiation_func = PyObject_GetAttrString(triton_module, "compute_radiation_gpu");
    if (radiation_func == NULL || !PyCallable_Check(radiation_func)) {
        if (PyErr_Occurred()) PyErr_Print();
        fprintf(stderr, "GPU Bridge: Failed to get compute_radiation_gpu\n");
        Py_XDECREF(triton_module);
        Py_Finalize();
        return -1;
    }
    
    python_initialized = 1;
    fprintf(stdout, "GPU Bridge: Initialized successfully\n");
    return 0;
}

void gpu_bridge_finalize(void) {
    if (!python_initialized) return;
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

    if (!python_initialized || !triton_module) return 0;

    /* Use radiation module's check_gpu_available() instead of direct torch import */
    /* This avoids potential memory issues with embedded Python + torch initialization */
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
        /* Fallback: assume GPU not available if function not found */
        if (PyErr_Occurred()) PyErr_Clear();
        fprintf(stdout, "GPU Bridge: check_gpu_available not found, assuming CPU mode\n");
        return 0;
    }

    fprintf(stdout, "GPU Bridge: CUDA %s\n", gpu_available ? "available" : "not available");
    return gpu_available;
}

static void* init_numpy_internal(void) { import_array(); return (void*)1; }

static PyObject* create_numpy_array_from_ptr(double* ptr, int ndim, npy_intp* dims) {
    /* Use NPY_FLOAT64 to match Fortran REAL(EB) = FP64 */
    return PyArray_SimpleNewFromData(ndim, dims, NPY_FLOAT64, (void*)ptr);
}

int32_t gpu_radiation_kernel(const RadiationGPUData* data) {
    PyObject *args, *result, *tmp_arr, *kappa_arr, *il_arr, *qr_arr, *extcoe_arr, *scaeff_arr;
    int status = -1;
    
    if (!python_initialized || !radiation_func || !data) return -1;
    
    npy_intp d3[3] = {data->kbar, data->jbar, data->ibar};
    npy_intp d4[4] = {data->nra, data->kbar, data->jbar, data->ibar};
    npy_intp dt[3] = {data->kbar + 1, data->jbar + 1, data->ibar + 1};
    
    tmp_arr = create_numpy_array_from_ptr(data->tmp_ptr, 3, dt);
    kappa_arr = create_numpy_array_from_ptr(data->kappa_gas_ptr, 3, d3);
    il_arr = create_numpy_array_from_ptr(data->il_ptr, 4, d4);
    qr_arr = create_numpy_array_from_ptr(data->qr_ptr, 3, d3);
    extcoe_arr = create_numpy_array_from_ptr(data->extcoe_ptr, 3, d3);
    scaeff_arr = create_numpy_array_from_ptr(data->scaeff_ptr, 3, d3);
    
    if (!tmp_arr || !kappa_arr || !il_arr || !qr_arr || !extcoe_arr || !scaeff_arr) goto cleanup;
    
    args = Py_BuildValue("(OOOOOOffff)", tmp_arr, kappa_arr, il_arr, qr_arr,
                         extcoe_arr, scaeff_arr, data->dx, data->dy, data->dz, data->sigma);
    if (!args) goto cleanup;
    
    result = PyObject_CallObject(radiation_func, args);
    if (!result) { if (PyErr_Occurred()) PyErr_Print(); goto cleanup; }
    
    status = PyLong_Check(result) ? (int)PyLong_AsLong(result) : 0;
    Py_DECREF(result);
    
cleanup:
    Py_XDECREF(args);
    Py_XDECREF(tmp_arr); Py_XDECREF(kappa_arr); Py_XDECREF(il_arr);
    Py_XDECREF(qr_arr); Py_XDECREF(extcoe_arr); Py_XDECREF(scaeff_arr);
    return status;
}

void gpu_bridge_sync(void) {
    PyObject *torch, *cuda, *sync_func, *result;
    if (!python_initialized) return;
    torch = PyImport_ImportModule("torch");
    if (!torch) return;
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
}
