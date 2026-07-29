#!/bin/bash
# Smart Cabin - Download face detection models
#
# Usage: bash edge/inference/download_models.sh
#
# Models:
#   1. det_500m.onnx (primary) - SCRFD-500M from InsightFace buffalo_s pack
#      Source: https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_s.zip
#      Extract det_500m.onnx from the zip file.
#
#   2. yunet.onnx (fallback) - YuNet from OpenCV Zoo
#      Auto-downloaded below.

set -e

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)/../plugins/face_recognition/models"
mkdir -p "$MODEL_DIR"

echo "=============================================="
echo "  Smart Cabin - Download Face Detection Models"
echo "=============================================="

# --- SCRFD det_500m.onnx (Primary) ---
SCRFD_ONNX="${MODEL_DIR}/det_500m.onnx"

echo "[1/2] SCRFD-500M (det_500m.onnx)..."
if [ -f "$SCRFD_ONNX" ]; then
    echo "  Already exists: $SCRFD_ONNX"
else
    echo "  NOT FOUND: $SCRFD_ONNX"
    echo ""
    echo "  Manual download required:"
    echo "    1. Download buffalo_s.zip from:"
    echo "       https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_s.zip"
    echo "    2. Extract det_500m.onnx from the zip"
    echo "    3. Copy to: $SCRFD_ONNX"
    echo ""
fi

# --- YuNet (Fallback) ---
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_ONNX="${MODEL_DIR}/yunet.onnx"

echo "[2/2] YuNet (fallback)..."
if [ -f "$YUNET_ONNX" ]; then
    echo "  Already exists: $YUNET_ONNX"
else
    wget -q --show-progress -O "$YUNET_ONNX" "$YUNET_URL"
    echo "  Downloaded: $YUNET_ONNX"
fi

echo ""
echo "=============================================="
echo "  Models in: $MODEL_DIR"
echo "=============================================="
ls -la "$MODEL_DIR"/*.onnx 2>/dev/null || echo "  (no models found)"
echo ""
echo "  Priority: det_500m.onnx (SCRFD) > yunet.onnx (YuNet)"
echo ""
