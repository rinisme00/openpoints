#include <Python.h>
#include <numpy/arrayobject.h>
#include "grid_subsampling/grid_subsampling.h"
#include <string>
#include <vector>
#include <cstring>

// docstrings for our module
static char module_docstring[] = "This module provides an interface for the subsampling of a pointcloud";
static char compute_docstring[] = "function subsampling a pointcloud";

// Declare the functions
static PyObject *grid_subsampling_compute(PyObject *self, PyObject *args, PyObject *keywds);

// Specify the members of the module
static PyMethodDef module_methods[] =
{
    {"compute", (PyCFunction)grid_subsampling_compute, METH_VARARGS | METH_KEYWORDS, compute_docstring},
    {NULL, NULL, 0, NULL}
};

// Initialize the module
static struct PyModuleDef moduledef =
{
    PyModuleDef_HEAD_INIT,
    "grid_subsampling",
    module_docstring,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_grid_subsampling(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}

// Actual wrapper
static PyObject *grid_subsampling_compute(PyObject *self, PyObject *args, PyObject *keywds)
{
    // Raw Python objects from parser
    PyObject *points_obj = NULL;
    PyObject *features_obj = NULL;
    PyObject *classes_obj = NULL;

    // Keywords containers
    static const char *kwlist[] = {"points", "features", "classes", "sampleDl", "method", "verbose", NULL};
    float sampleDl = 0.1f;
    const char *method_buffer = "barycenters";
    int verbose = 0;

    // Parse the input
    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywds,
            "O|OOfsi",
            const_cast<char **>(kwlist),
            &points_obj,
            &features_obj,
            &classes_obj,
            &sampleDl,
            &method_buffer,
            &verbose))
    {
        PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
        return NULL;
    }

    std::string method(method_buffer);

    // Interpret method
    if (method != "barycenters" && method != "voxelcenters")
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "Error parsing method. Valid method names are \"barycenters\" and \"voxelcenters\"");
        return NULL;
    }

    bool use_feature = (features_obj != NULL && features_obj != Py_None);
    bool use_classes = (classes_obj != NULL && classes_obj != Py_None);

    // Convert Python objects to NumPy arrays
    PyArrayObject *points_array =
        (PyArrayObject *)PyArray_FROM_OTF(points_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);

    PyArrayObject *features_array = NULL;
    PyArrayObject *classes_array = NULL;

    if (use_feature)
        features_array = (PyArrayObject *)PyArray_FROM_OTF(features_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);

    if (use_classes)
        classes_array = (PyArrayObject *)PyArray_FROM_OTF(classes_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);

    // Verify conversion
    if (points_array == NULL)
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Error converting input points to numpy arrays of type float32");
        return NULL;
    }

    if (use_feature && features_array == NULL)
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Error converting input features to numpy arrays of type float32");
        return NULL;
    }

    if (use_classes && classes_array == NULL)
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Error converting input classes to numpy arrays of type int32");
        return NULL;
    }

    // Shape checks
    if ((int)PyArray_NDIM(points_array) != 2 || (int)PyArray_DIM(points_array, 1) != 3)
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : points.shape is not (N, 3)");
        return NULL;
    }

    if (use_feature && (int)PyArray_NDIM(features_array) != 2)
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : features.shape is not (N, d)");
        return NULL;
    }

    if (use_classes && (int)PyArray_NDIM(classes_array) > 2)
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : classes.shape is not (N,) or (N, d)");
        return NULL;
    }

    int N = (int)PyArray_DIM(points_array, 0);

    int fdim = 0;
    if (use_feature)
        fdim = (int)PyArray_DIM(features_array, 1);

    int ldim = 1;
    if (use_classes && (int)PyArray_NDIM(classes_array) == 2)
        ldim = (int)PyArray_DIM(classes_array, 1);

    if (use_feature && (int)PyArray_DIM(features_array, 0) != N)
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : features.shape is not (N, d)");
        return NULL;
    }

    if (use_classes && (int)PyArray_DIM(classes_array, 0) != N)
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : classes.shape is not (N,) or (N, d)");
        return NULL;
    }

    if (verbose > 0)
        std::cout << "Computing cloud pyramid with support points: " << std::endl;

    // Convert NumPy arrays to C++ vectors
    std::vector<PointXYZ> original_points;
    std::vector<float> original_features;
    std::vector<int> original_classes;

    original_points = std::vector<PointXYZ>(
        (PointXYZ *)PyArray_DATA(points_array),
        (PointXYZ *)PyArray_DATA(points_array) + N);

    if (use_feature)
    {
        original_features = std::vector<float>(
            (float *)PyArray_DATA(features_array),
            (float *)PyArray_DATA(features_array) + N * fdim);
    }

    if (use_classes)
    {
        original_classes = std::vector<int>(
            (int *)PyArray_DATA(classes_array),
            (int *)PyArray_DATA(classes_array) + N * ldim);
    }

    // Subsample
    std::vector<PointXYZ> subsampled_points;
    std::vector<float> subsampled_features;
    std::vector<int> subsampled_classes;

    grid_subsampling(original_points,
                     subsampled_points,
                     original_features,
                     subsampled_features,
                     original_classes,
                     subsampled_classes,
                     sampleDl,
                     verbose);

    if (subsampled_points.empty())
    {
        Py_XDECREF(points_array);
        Py_XDECREF(features_array);
        Py_XDECREF(classes_array);
        PyErr_SetString(PyExc_RuntimeError, "Error");
        return NULL;
    }

    // Output dimensions
    npy_intp point_dims[2] = {(npy_intp)subsampled_points.size(), 3};
    npy_intp feature_dims[2] = {(npy_intp)subsampled_points.size(), (npy_intp)fdim};
    npy_intp classes_dims[2] = {(npy_intp)subsampled_points.size(), (npy_intp)ldim};

    PyArrayObject *res_points_obj =
        (PyArrayObject *)PyArray_SimpleNew(2, point_dims, NPY_FLOAT);

    PyArrayObject *res_features_obj = NULL;
    PyArrayObject *res_classes_obj = NULL;
    PyObject *ret = NULL;

    size_t size_in_bytes = subsampled_points.size() * 3 * sizeof(float);
    std::memcpy(PyArray_DATA(res_points_obj), subsampled_points.data(), size_in_bytes);

    if (use_feature)
    {
        size_in_bytes = subsampled_points.size() * fdim * sizeof(float);
        res_features_obj = (PyArrayObject *)PyArray_SimpleNew(2, feature_dims, NPY_FLOAT);
        std::memcpy(PyArray_DATA(res_features_obj), subsampled_features.data(), size_in_bytes);
    }

    if (use_classes)
    {
        size_in_bytes = subsampled_points.size() * ldim * sizeof(int);
        res_classes_obj = (PyArrayObject *)PyArray_SimpleNew(2, classes_dims, NPY_INT);
        std::memcpy(PyArray_DATA(res_classes_obj), subsampled_classes.data(), size_in_bytes);
    }

    if (use_feature && use_classes)
        ret = Py_BuildValue("NNN", (PyObject *)res_points_obj, (PyObject *)res_features_obj, (PyObject *)res_classes_obj);
    else if (use_feature)
        ret = Py_BuildValue("NN", (PyObject *)res_points_obj, (PyObject *)res_features_obj);
    else if (use_classes)
        ret = Py_BuildValue("NN", (PyObject *)res_points_obj, (PyObject *)res_classes_obj);
    else
        ret = Py_BuildValue("N", (PyObject *)res_points_obj);

    Py_DECREF(points_array);
    Py_XDECREF(features_array);
    Py_XDECREF(classes_array);

    return ret;
}