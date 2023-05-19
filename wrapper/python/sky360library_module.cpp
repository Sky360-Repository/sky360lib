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
        .def("getThreshold", &WMVParams::get_threshold)
        .def("getWeights", &WMVParams::get_weights)
        .def("getEnableWeight", &WMVParams::get_enable_weight)
        .def("getEnableThreshold", &WMVParams::get_enable_threshold)
        .def("setEnableWeight", &WMVParams::set_enable_weight)
        .def("setEnableThreshold", &WMVParams::set_enable_threshold)
        .def("setWeights", &WMVParams::set_weights)
        .def("setThreshold", &WMVParams::set_threshold)
        ;
    py::class_<VibeParams>(m, "VibeParams")
        .def(py::init<>())
        .def("getThreshold", &VibeParams::get_threshold)
        .def("getBGSamples", &VibeParams::get_bg_samples)
        .def("getRequiredBGSamples", &VibeParams::get_required_bg_samples)
        .def("getLearningRate", &VibeParams::get_learning_rate)
        .def("setThreshold", &VibeParams::set_threshold)
        .def("setBGSamples", &VibeParams::set_bg_samples)
        .def("setRequiredBGSamples", &VibeParams::set_required_bg_samples)
        .def("setLearningRate", &VibeParams::set_learning_rate)
        ;

    py::class_<Vibe>(m, "Vibe")
        .def(py::init<>())
        .def("apply", &Vibe::apply_ret)
        .def("getBackgroundImage", &Vibe::get_background_image)
        .def("getParameters", &Vibe::get_parameters, py::return_value_policy::reference);
    py::class_<WeightedMovingVariance>(m, "WeightedMovingVariance")
        .def(py::init<>())
        .def("apply", &WeightedMovingVariance::apply_ret)
        .def("getBackgroundImage", &WeightedMovingVariance::get_background_image)
        .def("getParameters", &WeightedMovingVariance::get_parameters, py::return_value_policy::reference);

    py::class_<ConnectedBlobDetection>(m, "ConnectedBlobDetection")
        .def(py::init<>())
        .def("detect", &ConnectedBlobDetection::detect_kp)
        .def("detectBB", &ConnectedBlobDetection::detect_ret)
        .def("setSizeThreshold", &ConnectedBlobDetection::set_size_threshold)
        .def("setAreaThreshold", &ConnectedBlobDetection::set_area_threshold)
        .def("setMinDistance", &ConnectedBlobDetection::set_min_distance);

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