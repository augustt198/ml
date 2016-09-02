#include "Python.h"
#include "numpy/arrayobject.h"

#define UTILS_MODULE_NAME "utils"

static PyObject*
convolution(PyObject *dummy, PyObject *args) {
    PyObject *volumeObj = NULL, *kernelObj = NULL;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &volumeObj,
        &PyArray_Type, &kernelObj)) return NULL;

    PyArrayObject *volume = NULL, *kernel = NULL, *out = NULL;

    volume = (PyArrayObject*) PyArray_FROM_OTF(volumeObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!volume) return NULL;

    kernel = (PyArrayObject*) PyArray_FROM_OTF(kernelObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!kernel) return NULL;


    if (volume->nd != 3) {
        PyErr_SetString(PyExc_TypeError, "Expected 3-dimensional volume");
        goto fail;
    } else if (kernel->nd != 4) {
        PyErr_SetString(PyExc_TypeError, "Expected 4-dimensional kernel");
        goto fail;
    }

    npy_intp vol_chns   = volume->dimensions[0],
        vol_h           = volume->dimensions[1],
        vol_w           = volume->dimensions[2],
        kernel_out_chns = kernel->dimensions[0],
        kernel_in_chns  = kernel->dimensions[1],
        kernel_h        = kernel->dimensions[2],
        kernel_w        = kernel->dimensions[3];

    if (vol_chns != kernel_in_chns) {
        PyErr_SetString(PyExc_TypeError, "Volume and kernel must have same number of input channels");
        return NULL;
    }

    // shape of output volume
    npy_intp out_shape[3] = {kernel_out_chns, vol_h - kernel_h + 1, vol_w - kernel_w + 1};
    PyObject *outObj = PyArray_SimpleNew(3, out_shape, NPY_DOUBLE);
    out = (PyArrayObject*) PyArray_FROM_OTF(outObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!out) return NULL;

    for (int n = 0; n < kernel_out_chns; n++) {
        for (int y = 0; y <= vol_h - kernel_h; y++) {
            for (int x = 0; x <= vol_w - kernel_w; x++) {

                double sum = 0.0;
                for (int chn = 0; chn < vol_chns; chn++) {
                    for (int ky = 0; ky < kernel_h; ky++) {
                        for (int kx = 0; kx < kernel_w; kx++) {
                            double *vol_item = (double*) PyArray_GETPTR3(volume, chn, y+ky, x+kx);
                            double *kernel_item = (double*) PyArray_GETPTR4(kernel, n, chn, ky, kx);
                            sum += (*vol_item) * (*kernel_item);
                        }
                    }
                }

                double *res = (double*) PyArray_GETPTR3(out, n, y, x);
                *res = sum;
            }
        }
    }

    Py_DECREF(volume);
    Py_DECREF(kernel);
    Py_DECREF(out);
    return outObj;

    fail:
    PyArray_XDECREF(volume);
    PyArray_XDECREF(kernel);
    PyArray_XDECREF(out);
    return NULL;
}

static struct PyMethodDef methods[] = {
    {"convolution", convolution, METH_VARARGS, "description"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initutils (void) {
    (void) Py_InitModule(UTILS_MODULE_NAME, methods);

    import_array();
}
