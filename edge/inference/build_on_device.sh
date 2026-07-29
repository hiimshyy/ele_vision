#!/bin/bash
# Smart Cabin - Build inference engine on Orange Pi 4 Pro
# Run this ON the Orange Pi (Debian 12, aarch64)
#
# Usage: bash edge/inference/build_on_device.sh
#
# Prerequisites:
#   sudo apt install build-essential cmake git libprotobuf-dev protobuf-compiler libomp-dev python3-dev

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=============================================="
echo "  Smart Cabin - Build Inference Engine"
echo "  Platform: $(uname -m) $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)"
echo "=============================================="

# Check dependencies
echo "[1/4] Checking dependencies..."
for cmd in cmake g++ git python3; do
    if ! command -v $cmd &> /dev/null; then
        echo "ERROR: $cmd not found. Install with:"
        echo "  sudo apt install build-essential cmake git python3-dev"
        exit 1
    fi
done

# Check Python development headers
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
if [ ! -f "${PYTHON_INCLUDE}/Python.h" ]; then
    echo "ERROR: Python development headers not found. Install with:"
    echo "  sudo apt install python3-dev"
    exit 1
fi

echo "  All dependencies OK"

# Build
echo "[2/4] Configuring CMake..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DUSE_OPENMP=ON

echo "[3/4] Building (this may take 10-20 minutes on first build)..."
make -j$(nproc)

echo "[4/4] Installing Python module..."
# Copy .so to face_recognition plugin directory
PYTHON_MODULE=$(find "${BUILD_DIR}" -name "cabin_inference_py*.so" | head -1)
if [ -n "$PYTHON_MODULE" ]; then
    cp "$PYTHON_MODULE" "${SCRIPT_DIR}/../plugins/face_recognition/"
    echo "  Installed: $(basename $PYTHON_MODULE)"
    echo "  Location: edge/plugins/face_recognition/"
else
    echo "  WARNING: Python module not found in build output"
fi

echo ""
echo "=============================================="
echo "  Build complete!"
echo "=============================================="
echo ""
echo "  Test C++ binary:"
echo "    ${BUILD_DIR}/test_detector <param_path> <bin_path>"
echo ""
echo "  Test Python module:"
echo "    python3 -c \"import sys; sys.path.insert(0, 'edge/plugins/face_recognition'); import cabin_inference_py; print('OK')\""
echo ""
echo "  Download YuNet model:"
echo "    bash edge/inference/download_models.sh"
echo ""
