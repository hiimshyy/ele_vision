#!/bin/bash
# Smart Cabin - Download face detection models (YuNet NCNN format)
#
# Usage: bash edge/inference/download_models.sh
#
# Downloads YuNet model converted to NCNN format.
# Model: YuNet (libfacedetection) - ~90KB, optimized for edge devices

set -e

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)/../plugins/face_recognition/models"
mkdir -p "$MODEL_DIR"

echo "=============================================="
echo "  Smart Cabin - Download Face Detection Models"
echo "=============================================="

# YuNet NCNN model from OpenCV Zoo / libfacedetection
# Using the ONNX model from OpenCV and converting, or using pre-converted NCNN
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_ONNX="${MODEL_DIR}/yunet.onnx"

echo "[1/2] Downloading YuNet ONNX model..."
if [ -f "$YUNET_ONNX" ]; then
    echo "  Already exists: $YUNET_ONNX"
else
    wget -q --show-progress -O "$YUNET_ONNX" "$YUNET_URL"
    echo "  Downloaded: $YUNET_ONNX ($(du -h $YUNET_ONNX | cut -f1))"
fi

echo "[2/2] Converting ONNX to NCNN format..."
# Check if ncnn tools are available (onnx2ncnn)
if command -v onnx2ncnn &> /dev/null; then
    onnx2ncnn "$YUNET_ONNX" "${MODEL_DIR}/yunet.param" "${MODEL_DIR}/yunet.bin"
    echo "  Converted: yunet.param + yunet.bin"
elif [ -f "edge/inference/build/install/bin/onnx2ncnn" ]; then
    edge/inference/build/install/bin/onnx2ncnn "$YUNET_ONNX" "${MODEL_DIR}/yunet.param" "${MODEL_DIR}/yunet.bin"
    echo "  Converted: yunet.param + yunet.bin"
else
    echo "  NOTE: onnx2ncnn not found. You have two options:"
    echo "    a) Build ncnn with tools: cmake -DNCNN_BUILD_TOOLS=ON .."
    echo "    b) Use OpenCV DNN directly with the ONNX file (Python fallback)"
    echo ""
    echo "  For now, we'll use OpenCV DNN as fallback (no conversion needed)."
    echo "  The Python wrapper will detect .onnx file and use cv2.FaceDetectorYN"
fi

echo ""
echo "=============================================="
echo "  Models downloaded to: $MODEL_DIR"
echo "=============================================="
ls -la "$MODEL_DIR"
