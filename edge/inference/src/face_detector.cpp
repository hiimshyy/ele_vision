#include "face_detector.h"

#include <net.h>
#include <mat.h>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace cabin {

// --- Impl (PIMPL pattern to hide ncnn headers) ---

struct FaceDetector::Impl {
    ncnn::Net net;
};

// --- Helper: NMS ---

static float iou(const FaceInfo& a, const FaceInfo& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

static std::vector<FaceInfo> nms(std::vector<FaceInfo>& faces, float threshold) {
    std::sort(faces.begin(), faces.end(),
              [](const FaceInfo& a, const FaceInfo& b) { return a.score > b.score; });

    std::vector<FaceInfo> result;
    std::vector<bool> suppressed(faces.size(), false);

    for (size_t i = 0; i < faces.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(faces[i]);
        for (size_t j = i + 1; j < faces.size(); ++j) {
            if (!suppressed[j] && iou(faces[i], faces[j]) > threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

// --- Helper: Generate priors for YuNet ---

struct PriorBox {
    float cx, cy, sx, sy;
};

static std::vector<PriorBox> generate_priors(int input_w, int input_h) {
    // YuNet prior box generation (strides: 8, 16, 32)
    std::vector<std::vector<int>> min_sizes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
    std::vector<int> strides = {8, 16, 32, 64};

    std::vector<PriorBox> priors;

    for (size_t i = 0; i < strides.size(); ++i) {
        int stride = strides[i];
        int feat_w = (input_w + stride - 1) / stride;
        int feat_h = (input_h + stride - 1) / stride;

        for (int row = 0; row < feat_h; ++row) {
            for (int col = 0; col < feat_w; ++col) {
                for (int min_size : min_sizes[i]) {
                    float cx = (col + 0.5f) * stride / input_w;
                    float cy = (row + 0.5f) * stride / input_h;
                    float sx = (float)min_size / input_w;
                    float sy = (float)min_size / input_h;
                    priors.push_back({cx, cy, sx, sy});
                }
            }
        }
    }
    return priors;
}

// --- FaceDetector Implementation ---

FaceDetector::FaceDetector()
    : impl_(new Impl()), loaded_(false), inference_time_ms_(0.0f),
      input_width_(320), input_height_(320), num_threads_(2) {}

FaceDetector::~FaceDetector() {
    delete impl_;
}

bool FaceDetector::load(const std::string& param_path,
                         const std::string& bin_path,
                         int input_width,
                         int input_height,
                         int num_threads) {
    input_width_ = input_width;
    input_height_ = input_height;
    num_threads_ = num_threads;

    impl_->net.opt.num_threads = num_threads;
    impl_->net.opt.use_vulkan_compute = false;
    impl_->net.opt.lightmode = true;

    int ret1 = impl_->net.load_param(param_path.c_str());
    int ret2 = impl_->net.load_model(bin_path.c_str());

    loaded_ = (ret1 == 0 && ret2 == 0);
    return loaded_;
}

std::vector<FaceInfo> FaceDetector::detect(const unsigned char* bgr_data,
                                            int img_width,
                                            int img_height,
                                            float conf_threshold,
                                            float nms_threshold) {
    std::vector<FaceInfo> faces;
    if (!loaded_) return faces;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Create ncnn::Mat from BGR data and resize to model input
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        bgr_data, ncnn::Mat::PIXEL_BGR, img_width, img_height,
        input_width_, input_height_
    );

    // Normalize (YuNet expects raw pixels, no normalization needed)
    // input is already in [0, 255] range

    // Run inference
    ncnn::Extractor ex = impl_->net.create_extractor();
    ex.set_num_threads(num_threads_);
    ex.input("input", input);

    // Get outputs
    ncnn::Mat cls_out, bbox_out, landmark_out;
    ex.extract("score", cls_out);
    ex.extract("bbox", bbox_out);
    ex.extract("kps", landmark_out);

    auto t_end = std::chrono::high_resolution_clock::now();
    inference_time_ms_ = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    // Generate prior boxes
    auto priors = generate_priors(input_width_, input_height_);

    // Decode detections
    float scale_w = (float)img_width;
    float scale_h = (float)img_height;

    for (int i = 0; i < (int)priors.size(); ++i) {
        float score = cls_out[i * 2 + 1];  // Positive class score
        if (score < conf_threshold) continue;

        const PriorBox& prior = priors[i];

        // Decode bbox
        float cx = prior.cx + bbox_out[i * 4 + 0] * prior.sx;
        float cy = prior.cy + bbox_out[i * 4 + 1] * prior.sy;
        float w = prior.sx * std::exp(bbox_out[i * 4 + 2]);
        float h = prior.sy * std::exp(bbox_out[i * 4 + 3]);

        FaceInfo face;
        face.x1 = (cx - w * 0.5f) * scale_w;
        face.y1 = (cy - h * 0.5f) * scale_h;
        face.x2 = (cx + w * 0.5f) * scale_w;
        face.y2 = (cy + h * 0.5f) * scale_h;
        face.score = score;

        // Decode landmarks
        for (int j = 0; j < 5; ++j) {
            face.landmarks[j * 2] = (prior.cx + landmark_out[i * 10 + j * 2] * prior.sx) * scale_w;
            face.landmarks[j * 2 + 1] = (prior.cy + landmark_out[i * 10 + j * 2 + 1] * prior.sy) * scale_h;
        }

        faces.push_back(face);
    }

    // NMS
    faces = nms(faces, nms_threshold);

    return faces;
}

}  // namespace cabin
