#include "face_detector.h"
#include <cstdio>
#include <cstdlib>

/**
 * Simple CLI test for face detector.
 * Usage: ./test_detector <param_path> <bin_path>
 *
 * Creates a dummy 640x480 image and runs detection to verify model loads.
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <param_path> <bin_path>\n", argv[0]);
        return 1;
    }

    cabin::FaceDetector detector;
    printf("Loading model: %s, %s\n", argv[1], argv[2]);

    bool ok = detector.load(argv[1], argv[2], 320, 320, 2);
    if (!ok) {
        printf("ERROR: Failed to load model\n");
        return 1;
    }
    printf("Model loaded successfully\n");

    // Create dummy image (640x480 BGR, all zeros)
    int w = 640, h = 480;
    std::vector<unsigned char> dummy(w * h * 3, 128);

    printf("Running detection on %dx%d dummy image...\n", w, h);
    auto faces = detector.detect(dummy.data(), w, h, 0.5f, 0.3f);

    printf("Detected %zu faces (on dummy image, expect 0)\n", faces.size());
    printf("Inference time: %.1f ms\n", detector.get_inference_time_ms());

    for (auto& f : faces) {
        printf("  Face: (%.0f, %.0f, %.0f, %.0f) score=%.3f\n",
               f.x1, f.y1, f.x2, f.y2, f.score);
    }

    printf("Test PASSED\n");
    return 0;
}
