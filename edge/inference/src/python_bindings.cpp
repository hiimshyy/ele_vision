#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "face_detector.h"

namespace py = pybind11;

/**
 * Python bindings for Smart Cabin Inference Engine.
 *
 * Usage from Python:
 *     import cabin_inference_py as ci
 *
 *     detector = ci.FaceDetector()
 *     detector.load("model.param", "model.bin", 320, 320, 2)
 *
 *     # frame is numpy array (H, W, 3) uint8 BGR
 *     faces = detector.detect(frame, 0.7, 0.3)
 *     for face in faces:
 *         print(f"bbox: ({face.x1}, {face.y1}, {face.x2}, {face.y2}), score: {face.score}")
 *         print(f"landmarks: {face.landmarks}")
 */

PYBIND11_MODULE(cabin_inference_py, m) {
    m.doc() = "Smart Cabin C++ Inference Engine - Face Detection";

    // FaceInfo struct
    py::class_<cabin::FaceInfo>(m, "FaceInfo")
        .def(py::init<>())
        .def_readwrite("x1", &cabin::FaceInfo::x1)
        .def_readwrite("y1", &cabin::FaceInfo::y1)
        .def_readwrite("x2", &cabin::FaceInfo::x2)
        .def_readwrite("y2", &cabin::FaceInfo::y2)
        .def_readwrite("score", &cabin::FaceInfo::score)
        .def_property_readonly("landmarks", [](const cabin::FaceInfo& f) {
            return py::array_t<float>({10}, f.landmarks);
        })
        .def_property_readonly("bbox", [](const cabin::FaceInfo& f) {
            return py::make_tuple(f.x1, f.y1, f.x2, f.y2);
        })
        .def("__repr__", [](const cabin::FaceInfo& f) {
            return "<FaceInfo bbox=(" + std::to_string((int)f.x1) + "," +
                   std::to_string((int)f.y1) + "," +
                   std::to_string((int)f.x2) + "," +
                   std::to_string((int)f.y2) + ") score=" +
                   std::to_string(f.score) + ">";
        });

    // FaceDetector class
    py::class_<cabin::FaceDetector>(m, "FaceDetector")
        .def(py::init<>())
        .def("load", &cabin::FaceDetector::load,
             py::arg("param_path"),
             py::arg("bin_path"),
             py::arg("input_width") = 320,
             py::arg("input_height") = 320,
             py::arg("num_threads") = 2,
             "Load NCNN model from .param and .bin files")
        .def("detect", [](cabin::FaceDetector& self,
                          py::array_t<uint8_t, py::array::c_style> frame,
                          float conf_threshold,
                          float nms_threshold) {
            auto buf = frame.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::runtime_error("Expected BGR image with shape (H, W, 3)");
            }
            int h = buf.shape[0];
            int w = buf.shape[1];
            const unsigned char* data = static_cast<const unsigned char*>(buf.ptr);
            return self.detect(data, w, h, conf_threshold, nms_threshold);
        },
             py::arg("frame"),
             py::arg("conf_threshold") = 0.7f,
             py::arg("nms_threshold") = 0.3f,
             "Detect faces in BGR numpy array (H, W, 3)")
        .def("is_loaded", &cabin::FaceDetector::is_loaded)
        .def("get_inference_time_ms", &cabin::FaceDetector::get_inference_time_ms);
}
