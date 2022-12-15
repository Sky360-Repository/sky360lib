#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/pybind11.h>
#include <Python.h>

#include "ndarray_converter.h"

#include "bgs.hpp"

namespace py = pybind11;
using namespace sky360lib::bgs;

PYBIND11_MODULE(pysky360, m)
{
    NDArrayConverter::init_numpy();

    m.doc() = "python wrapper for sky360lib using pybind11";
    py::object version = py::cast("1.0.0");
    m.attr("__version__") = version;

    py::class_<WeightedMovingVariance>(m, "WeightedMovingVariance")
        .def(py::init<>())
        .def("apply", &WeightedMovingVariance::applyRet)
        .def("getBackgroundImage", &WeightedMovingVariance::getBackgroundImage);
    py::class_<WeightedMovingVarianceHalide>(m, "WeightedMovingVarianceHalide")
        .def(py::init<>())
        .def("apply", &WeightedMovingVarianceHalide::applyRet)
        .def("getBackgroundImage", &WeightedMovingVarianceHalide::getBackgroundImage);
    // py::class_<WeightedMovingVarianceCuda>(m, "WeightedMovingVarianceCuda")
    //     .def(py::init<>())
    //     .def("apply", &WeightedMovingVarianceCuda::applyRet)
    //     .def("getBackgroundImage", &WeightedMovingVarianceCuda::getBackgroundImage);
}