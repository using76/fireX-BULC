/**
 * @file bridge.c
 * @brief C-Python Bridge Implementation for Triton GPU Kernels
 *
 * This file implements the interface between Fortran/C and Python
 * for GPU-accelerated computations using Triton kernels.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* NumPy support - conditionally included */
#ifdef WITH_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bridge.h"

/* Global Python objects */
static PyObject *pRadiationModule = NULL;
static PyObject *pRadiationFunc = NULL;
static PyObject *pTorchModule = NULL;
static int python_initialized = 0;

/* Path to Python kernels directory */
static const char *KERNEL_PATH = "C:/Users/ji/Documents/fireX/fds-FireX/python_kernels";

/**
 * @brief Initialize Python runtime and load Triton modules
 */
int init_python_runtime(void) {
    PyObject *pSysPath, *pPath;

    if (python_initialized) {
        return 0;  /* Already initialized */
    }

    /* Initialize Python interpreter */
    Py_Initialize();
    if (!Py_IsInitialized()) {
        fprintf(stderr, "GPU Bridge: Failed to initialize Python interpreter\n");
        return -1;
    }

    /* Add kernel path to sys.path */
    pSysPath = PySys_GetObject("path");
    if (pSysPath == NULL) {
        fprintf(stderr, "GPU Bridge: Failed to get sys.path\n");
        Py_Finalize();
        return -1;
    }

    pPath = PyUnicode_FromString(KERNEL_PATH);
    if (pPath == NULL) {
        fprintf(stderr, "GPU Bridge: Failed to create path string\n");
        Py_Finalize();
        return -1;
    }

    if (PyList_Insert(pSysPath, 0, pPath) < 0) {
        fprintf(stderr, "GPU Bridge: Failed to add kernel path to sys.path\n");
        Py_DECREF(pPath);
        Py_Finalize();
        return -1;
    }
    Py_DECREF(pPath);

    /* Import torch module */
    pTorchModule = PyImport_ImportModule("torch");
    if (pTorchModule == NULL) {
        fprintf(stderr, "GPU Bridge: Failed to import torch module\n");
        PyErr_Print();
        Py_Finalize();
        return -2;
    }

    /* Import radiation module */
    pRadiationModule = PyImport_ImportModule("radiation");
    if (pRadiationModule == NULL) {
        fprintf(stderr, "GPU Bridge: Failed to import radiation module\n");
        PyErr_Print();
        Py_XDECREF(pTorchModule);
        Py_Finalize();
        return -2;
    }

    /* Get compute_radiation function */
    pRadiationFunc = PyObject_GetAttrString(pRadiationModule, "compute_radiation_from_c");
    if (pRadiationFunc == NULL || !PyCallable_Check(pRadiationFunc)) {
        fprintf(stderr, "GPU Bridge: Failed to get compute_radiation_from_c function\n");
        PyErr_Print();
        Py_XDECREF(pRadiationFunc);
        Py_XDECREF(pRadiationModule);
        Py_XDECREF(pTorchModule);
        Py_Finalize();
        return -3;
    }

    python_initialized = 1;
    printf("GPU Bridge: Python runtime initialized successfully\n");
    printf("GPU Bridge: Kernel path: %s\n", KERNEL_PATH);

    return 0;
}

/**
 * @brief Finalize Python runtime
 */
void finalize_python_runtime(void) {
    if (!python_initialized) {
        return;
    }

    Py_XDECREF(pRadiationFunc);
    Py_XDECREF(pRadiationModule);
    Py_XDECREF(pTorchModule);

    pRadiationFunc = NULL;
    pRadiationModule = NULL;
    pTorchModule = NULL;

    Py_Finalize();
    python_initialized = 0;

    printf("GPU Bridge: Python runtime finalized\n");
}

/**
 * @brief Call Triton radiation kernel
 */
int call_radiation_kernel(const RadiationGPUData *data) {
    PyObject *pArgs, *pResult;
    PyObject *pDict;

    if (!python_initialized) {
        fprintf(stderr, "GPU Bridge: Python not initialized\n");
        return -1;
    }

    if (data == NULL) {
        fprintf(stderr, "GPU Bridge: NULL data pointer\n");
        return -2;
    }

    /* Create dictionary with data pointers and dimensions */
    pDict = PyDict_New();
    if (pDict == NULL) {
        PyErr_Print();
        return -3;
    }

    /* Set pointer addresses as integers (will be converted to tensors in Python) */
    PyDict_SetItemString(pDict, "TMP_ptr", PyLong_FromVoidPtr(data->TMP));
    PyDict_SetItemString(pDict, "KAPPA_ptr", PyLong_FromVoidPtr(data->KAPPA_GAS));
    PyDict_SetItemString(pDict, "KFST4_ptr", PyLong_FromVoidPtr(data->KFST4_GAS));
    PyDict_SetItemString(pDict, "IL_ptr", PyLong_FromVoidPtr(data->IL));
    PyDict_SetItemString(pDict, "UIID_ptr", PyLong_FromVoidPtr(data->UIID));
    PyDict_SetItemString(pDict, "QR_ptr", PyLong_FromVoidPtr(data->QR));
    PyDict_SetItemString(pDict, "EXTCOE_ptr", PyLong_FromVoidPtr(data->EXTCOE));
    PyDict_SetItemString(pDict, "SCAEFF_ptr", PyLong_FromVoidPtr(data->SCAEFF));
    PyDict_SetItemString(pDict, "UIIOLD_ptr", PyLong_FromVoidPtr(data->UIIOLD));
    PyDict_SetItemString(pDict, "DX_ptr", PyLong_FromVoidPtr(data->DX));
    PyDict_SetItemString(pDict, "DY_ptr", PyLong_FromVoidPtr(data->DY));
    PyDict_SetItemString(pDict, "DZ_ptr", PyLong_FromVoidPtr(data->DZ));
    PyDict_SetItemString(pDict, "DLX_ptr", PyLong_FromVoidPtr(data->DLX));
    PyDict_SetItemString(pDict, "DLY_ptr", PyLong_FromVoidPtr(data->DLY));
    PyDict_SetItemString(pDict, "DLZ_ptr", PyLong_FromVoidPtr(data->DLZ));
    PyDict_SetItemString(pDict, "RSA_ptr", PyLong_FromVoidPtr(data->RSA));

    /* Set dimensions */
    PyDict_SetItemString(pDict, "IBAR", PyLong_FromLong(data->IBAR));
    PyDict_SetItemString(pDict, "JBAR", PyLong_FromLong(data->JBAR));
    PyDict_SetItemString(pDict, "KBAR", PyLong_FromLong(data->KBAR));
    PyDict_SetItemString(pDict, "NRA", PyLong_FromLong(data->NRA));
    PyDict_SetItemString(pDict, "NSB", PyLong_FromLong(data->NSB));

    /* Set constants */
    PyDict_SetItemString(pDict, "FOUR_SIGMA", PyFloat_FromDouble(data->FOUR_SIGMA));
    PyDict_SetItemString(pDict, "RFPI", PyFloat_FromDouble(data->RFPI));
    PyDict_SetItemString(pDict, "RSA_RAT", PyFloat_FromDouble(data->RSA_RAT));

    /* Create args tuple */
    pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pDict);  /* Steals reference to pDict */

    /* Call Python function */
    pResult = PyObject_CallObject(pRadiationFunc, pArgs);
    Py_DECREF(pArgs);

    if (pResult == NULL) {
        fprintf(stderr, "GPU Bridge: Radiation kernel call failed\n");
        PyErr_Print();
        return -4;
    }

    /* Check return value */
    int status = 0;
    if (PyLong_Check(pResult)) {
        status = (int)PyLong_AsLong(pResult);
    }
    Py_DECREF(pResult);

    return status;
}

/**
 * @brief Allocate GPU-accessible memory
 */
void* allocate_gpu_array(size_t size_bytes) {
    /* For now, use standard malloc with alignment */
    /* TODO: Use CUDA pinned memory or unified memory */
    void *ptr = NULL;

#ifdef _WIN32
    ptr = _aligned_malloc(size_bytes, 64);
#else
    if (posix_memalign(&ptr, 64, size_bytes) != 0) {
        ptr = NULL;
    }
#endif

    if (ptr != NULL) {
        memset(ptr, 0, size_bytes);
    }

    return ptr;
}

/**
 * @brief Free GPU-accessible memory
 */
void free_gpu_array(void *ptr) {
    if (ptr != NULL) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

/**
 * @brief Synchronize GPU operations
 */
void gpu_sync(void) {
    PyObject *pResult;

    if (!python_initialized || pTorchModule == NULL) {
        return;
    }

    /* Call torch.cuda.synchronize() */
    pResult = PyObject_CallMethod(pTorchModule, "cuda.synchronize", NULL);
    if (pResult != NULL) {
        Py_DECREF(pResult);
    } else {
        PyErr_Clear();  /* Ignore errors (e.g., no CUDA) */
    }
}

/**
 * @brief Check if GPU/Triton is available
 */
int is_gpu_available(void) {
    PyObject *pCuda, *pResult;
    int available = 0;

    if (!python_initialized || pTorchModule == NULL) {
        return 0;
    }

    /* Get torch.cuda */
    pCuda = PyObject_GetAttrString(pTorchModule, "cuda");
    if (pCuda == NULL) {
        PyErr_Clear();
        return 0;
    }

    /* Call is_available() */
    pResult = PyObject_CallMethod(pCuda, "is_available", NULL);
    if (pResult != NULL) {
        available = PyObject_IsTrue(pResult);
        Py_DECREF(pResult);
    }

    Py_DECREF(pCuda);
    return available;
}

/**
 * @brief Get GPU device information
 */
int get_gpu_device_info(char *device_name, size_t max_len) {
    PyObject *pCuda, *pResult;

    if (!python_initialized || pTorchModule == NULL) {
        return -1;
    }

    if (device_name == NULL || max_len == 0) {
        return -1;
    }

    /* Get torch.cuda */
    pCuda = PyObject_GetAttrString(pTorchModule, "cuda");
    if (pCuda == NULL) {
        PyErr_Clear();
        return -1;
    }

    /* Call get_device_name(0) */
    pResult = PyObject_CallMethod(pCuda, "get_device_name", "i", 0);
    if (pResult != NULL && PyUnicode_Check(pResult)) {
        const char *name = PyUnicode_AsUTF8(pResult);
        if (name != NULL) {
            strncpy(device_name, name, max_len - 1);
            device_name[max_len - 1] = '\0';
        }
        Py_DECREF(pResult);
    } else {
        PyErr_Clear();
        Py_DECREF(pCuda);
        return -1;
    }

    Py_DECREF(pCuda);
    return 0;
}
