#!/bin/bash
# Smart Cabin - Edge Setup Script for Orange Pi 4 Pro (Ubuntu/Debian ARM64)
# Run this on the Orange Pi after cloning the repo
#
# Usage: bash deploy/setup_edge.sh

set -e

echo "=============================================="
echo "  Smart Cabin - Edge Device Setup"
echo "  Target: Orange Pi 4 Pro (RK3399)"
echo "=============================================="

# --- System dependencies ---
echo "[1/5] Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    libopencv-dev python3-opencv \
    libgstreamer1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    curl git

# --- Install uv (fast Python package manager) ---
echo "[2/5] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv version: $(uv --version)"

# --- Create venv and install Python dependencies ---
echo "[3/5] Creating Python virtual environment..."
uv venv --python 3.12 || uv venv  # Fallback to system python if 3.12 not available
uv pip install -e .
uv pip install opencv-python-headless numpy

# --- Create directories ---
echo "[4/5] Creating runtime directories..."
mkdir -p logs
mkdir -p data/faces

# --- Config ---
echo "[5/5] Setting up configuration..."
if [ ! -f config.yaml ]; then
    cp config.yaml.example config.yaml 2>/dev/null || true
    echo "  Created config.yaml - please edit with your RTSP URL"
else
    echo "  config.yaml already exists, skipping"
fi

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "  1. Edit config.yaml with your RTSP camera URL"
echo "  2. Test: uv run python edge/run_edge.py"
echo "  3. Service: sudo cp deploy/smart-cabin.service /etc/systemd/system/"
echo "             sudo systemctl enable smart-cabin"
echo "             sudo systemctl start smart-cabin"
echo ""
