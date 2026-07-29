#pragma once

#include <string>
#include <vector>

namespace cabin {

/**
 * Detected face with bounding box, confidence, and 5-point landmarks.
 */
struct FaceInfo {
    float x1, y1, x2, y2;  // Bounding box (top-left, bottom-right)
    float score;             // Detection confidence [0, 1]
    // 5 landmarks: left eye, right eye, nose, left mouth, right mouth
    float landmarks[10];     // [x0,y0, x1,y1, x2,y2, x3,y3, x4,y4]
};

/**
 * YuNet Face Detector using NCNN.
 *
 * Lightweight face detection (~90KB model), optimized for edge devices.
 * Produces bounding boxes + 5-point landmarks per detected face.
 */
class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();

    /**
     * Load model from .param and .bin files.
     *
     * @param param_path Path to .param file
     * @param bin_path   Path to .bin file
     * @param input_width  Model input width (default 320)
     * @param input_height Model input height (default 320)
     * @param num_threads  Number of threads for inference (default 2)
     * @return true if loaded successfully
     */
    bool load(const std::string& param_path,
              const std::string& bin_path,
              int input_width = 320,
              int input_height = 320,
              int num_threads = 2);

    /**
     * Detect faces in a BGR image.
     *
     * @param bgr_data    Pointer to BGR pixel data (HWC format)
     * @param img_width   Image width
     * @param img_height  Image height
     * @param conf_threshold Minimum confidence threshold (default 0.7)
     * @param nms_threshold  NMS IoU threshold (default 0.3)
     * @return Vector of detected faces
     */
    std::vector<FaceInfo> detect(const unsigned char* bgr_data,
                                  int img_width,
                                  int img_height,
                                  float conf_threshold = 0.7f,
                                  float nms_threshold = 0.3f);

    /**
     * Check if model is loaded.
     */
    bool is_loaded() const { return loaded_; }

    /**
     * Get last inference time in milliseconds.
     */
    float get_inference_time_ms() const { return inference_time_ms_; }

private:
    struct Impl;
    Impl* impl_;
    bool loaded_;
    float inference_time_ms_;
    int input_width_;
    int input_height_;
    int num_threads_;
};

}  // namespace cabin
