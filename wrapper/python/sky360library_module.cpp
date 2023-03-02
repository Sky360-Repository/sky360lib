#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include <Python.h>

#include "ndarray_converter.h"

#include "bgs.hpp"
#include "connectedBlobDetection.hpp"

namespace py = pybind11;
using namespace sky360lib::bgs;
using namespace sky360lib::blobs;

PYBIND11_MODULE(pysky360, m)
{
    NDArrayConverter::init_numpy();

    m.doc() = "python wrapper for sky360lib using pybind11";
    py::object version = py::cast("1.0.0");
    m.attr("__version__") = version;

    py::class_<WMVParams>(m, "WMVParams")
        .def(py::init<>())
        .def("getThreshold", &WMVParams::getThreshold)
        .def("getWeights", &WMVParams::getWeights)
        .def("getEnableWeight", &WMVParams::getEnableWeight)
        .def("getEnableThreshold", &WMVParams::getEnableThreshold)
        .def("setEnableWeight", &WMVParams::setEnableWeight)
        .def("setEnableThreshold", &WMVParams::setEnableThreshold)
        .def("setWeights", &WMVParams::setWeights)
        .def("setThreshold", &WMVParams::setThreshold)
        ;
    py::class_<VibeParams>(m, "VibeParams")
        .def(py::init<>())
        .def("getThreshold", &VibeParams::getThreshold)
        .def("getBGSamples", &VibeParams::getBGSamples)
        .def("getRequiredBGSamples", &VibeParams::getRequiredBGSamples)
        .def("getLearningRate", &VibeParams::getLearningRate)
        .def("setThreshold", &VibeParams::setThreshold)
        .def("setBGSamples", &VibeParams::setBGSamples)
        .def("setRequiredBGSamples", &VibeParams::setRequiredBGSamples)
        .def("setLearningRate", &VibeParams::setLearningRate)
        ;

    py::class_<Vibe>(m, "Vibe")
        .def(py::init<>())
        .def("apply", &Vibe::applyRet)
        .def("getBackgroundImage", &Vibe::getBackgroundImage)
        .def("getParameters", &Vibe::getParameters, py::return_value_policy::reference);
    py::class_<WeightedMovingVariance>(m, "WeightedMovingVariance")
        .def(py::init<>())
        .def("apply", &WeightedMovingVariance::applyRet)
        .def("getBackgroundImage", &WeightedMovingVariance::getBackgroundImage)
        .def("getParameters", &WeightedMovingVariance::getParameters, py::return_value_policy::reference);

    py::class_<ConnectedBlobDetection>(m, "ConnectedBlobDetection")
        .def(py::init<>())
        .def("detect", &ConnectedBlobDetection::detectKP)
        .def("detectBB", &ConnectedBlobDetection::detectRet)
        .def("setSizeThreshold", &ConnectedBlobDetection::setSizeThreshold)
        .def("setAreaThreshold", &ConnectedBlobDetection::setAreaThreshold)
        .def("setMinDistance", &ConnectedBlobDetection::setMinDistance);
}