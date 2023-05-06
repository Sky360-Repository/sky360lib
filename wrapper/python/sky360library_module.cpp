#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include <Python.h>

#include "ndarray_converter.h"

#include "bgs.hpp"
#include "connectedBlobDetection.hpp"
#include "qhy_camera.hpp"

namespace py = pybind11;
using namespace sky360lib::bgs;
using namespace sky360lib::blobs;
using namespace sky360lib::camera;

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

    py::class_<QhyCamera>(m, "QHYCamera")
        .def(py::init<>())
        .def("setDebugInfo", &QhyCamera::set_debug_info)
        .def("getFrame", &QhyCamera::get_frame_ret)
        .def("debayerImage", &QhyCamera::debayer_image_ret)
        .def("getLastFrameCaptureTime", &QhyCamera::get_last_frame_capture_time)
        .def("getCameraInfo", &QhyCamera::get_camera_info)
        .def("getCameraParams", &QhyCamera::get_camera_params)
        .def("open", &QhyCamera::open)
        .def("close", &QhyCamera::close)
        .def("setControl", &QhyCamera::set_control)
        .def("setDebayer", &QhyCamera::set_debayer)
        .def("setBinMode", &QhyCamera::set_bin_mode)
        .def("setResolution", &QhyCamera::set_resolution)
        .def("setStreamMode", &QhyCamera::set_stream_mode);

    py::class_<QhyCamera::CameraInfo>(m, "QHYCamera.CameraInfo")
        .def(py::init<>())
        .def("bayerFormatToString", &QhyCamera::CameraInfo::bayer_format_to_string)
        .def("toString", &QhyCamera::CameraInfo::to_string);

    py::enum_<QhyCamera::ControlParam>(m, "ControlParam")
        .value("Brightness", QhyCamera::ControlParam::Brightness)
        .value("Exposure", QhyCamera::ControlParam::Exposure)
        .value("Contrast", QhyCamera::ControlParam::Contrast)
        .value("UsbTraffic", QhyCamera::ControlParam::UsbTraffic)
        .value("UsbSpeed", QhyCamera::ControlParam::UsbSpeed)
        .value("Gain", QhyCamera::ControlParam::Gain)
        .value("Offset", QhyCamera::ControlParam::Offset)
        .value("TransferBits", QhyCamera::ControlParam::TransferBits)
        .value("RedWB", QhyCamera::ControlParam::RedWB)
        .value("GreenWB", QhyCamera::ControlParam::GreenWB)
        .value("BlueWB", QhyCamera::ControlParam::BlueWB)
        .value("Gamma", QhyCamera::ControlParam::Gamma)
        .value("Channels", QhyCamera::ControlParam::Channels)
        .export_values();
}