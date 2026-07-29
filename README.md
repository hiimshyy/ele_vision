# Smart Cabin Platform

Nền tảng AI cho cabin thang máy, chạy trên Orange Pi 4 Pro với camera RTSP. Kiến trúc plugin-based cho phép mở rộng từ Face Recognition sang People Counting, Elevator Control, và các module khác.

## Hardware

| Component | Spec |
|-----------|------|
| SBC | Orange Pi 4 Pro (RK3399, 6-core ARM, 4GB RAM, Debian 12) |
| Camera | IP Camera with RTSP stream (1920x1080) |
| Inference | C++ NCNN (ARM NEON) + OpenCV DNN fallback |

## Features (Current)

- **Video Pipeline** — RTSP capture, latest frame buffer, per-callback FPS scheduling, auto-reconnect
- **Frame Scheduler** — Mỗi plugin nhận frames ở FPS riêng (Face: 5fps, Display: 15fps, Recorder: 1fps)
- **Plugin Manager** — BasePlugin lifecycle (init → running → stopped), config-driven loading, auto-disable on crash
- **Event Bus** — Thread-safe pub/sub, Pydantic validation, wildcard subscribe, event history
- **Face Detection** — SCRFD-500M (det_500m.onnx, InsightFace) + YuNet fallback, ~10-15ms/frame
- **Structured Logging** — Loguru, key-value format, module-based files (camera/scheduler/plugin/system)
- **Per-plugin Metrics** — actual_fps, avg_process_ms, missed_deadlines, errors, disabled status

## Quick Start

```bash
# Clone
git clone https://github.com/hiimshyy/ele_vision.git
cd ele_vision

# Setup (requires Python 3.12+)
uv venv --python 3.12
uv pip install -e .
uv pip install opencv-python numpy loguru psutil

# Download face detection model
bash edge/inference/download_models.sh

# Run examples
python examples/run_camera.py --url "rtsp://USER:PASS@IP:554/stream" --scale 0.5
python examples/run_face_detection.py --url "rtsp://..." --det-fps 5 --scale 0.5
python examples/run_stress_test.py --url "rtsp://..." --duration 60

# Run tests
pytest edge/tests/ -v
```

## Build C++ Inference (Optional, for max performance)

```bash
# Install build dependencies
sudo apt install build-essential cmake git python3-dev libprotobuf-dev protobuf-compiler libomp-dev

# Build (on Orange Pi, ~15 minutes first time)
cd edge/inference
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python)
make -j$(nproc)

# Copy module
cp cabin_inference_py.cpython-312*.so ../../plugins/face_recognition/
```

## Project Structure

```
edge/
├── core/
│   ├── config.py             # YAML config + env override
│   ├── video_pipeline.py     # RTSP capture + Frame Scheduler
│   ├── event_bus.py          # Thread-safe pub/sub event system
│   ├── plugin_manager.py     # Plugin lifecycle management
│   └── logging_setup.py      # Loguru module-based logging
├── plugins/
│   ├── face_recognition/
│   │   ├── detector.py       # FaceDetector (SCRFD primary + YuNet fallback)
│   │   └── models/           # det_500m.onnx (SCRFD), yunet.onnx
│   └── dummy/
│       └── plugin.py         # Test plugin (frame counter)
├── inference/
│   ├── CMakeLists.txt        # C++ build (NCNN + pybind11)
│   ├── src/
│   │   ├── face_detector.cpp
│   │   └── python_bindings.cpp
│   ├── include/face_detector.h
│   ├── build_on_device.sh    # Build script for Orange Pi
│   └── download_models.sh    # Model downloader
├── tests/                    # 92+ tests
└── config.yaml
examples/
├── run_camera.py             # Camera only + stats overlay
├── run_face_detection.py     # Camera + realtime face detection
└── run_stress_test.py        # Plugin isolation stress test
docs/
├── implementation_plan.md    # Full task breakdown
├── video_pipeline_metrics.md # Metric definitions & troubleshooting
└── log_inspection_guide.md   # How to read and analyze logs
```

## Configuration

Edit `config.yaml`:

```yaml
camera:
  url: "rtsp://admin:password@192.168.1.100:554/stream"
  capture_fps: 25
  process_fps: 15
  reconnect_interval: 5.0
  connection_timeout: 10.0

plugins:
  modules:
    - name: "face_recognition"
      enabled: true
      config:
        detection_threshold: 0.7
        min_face_size: 80

mqtt:
  broker_host: "localhost"
  broker_port: 1883
  device_id: "cabin-001"

logging:
  level: "INFO"
```

Override with environment variables: `SC_CAMERA_URL=rtsp://...`

## Logging

Log files in `logs/`:

| File | Content |
|------|---------|
| `camera.log` | Video pipeline: connect, decode, reconnect, periodic stats |
| `scheduler.log` | Frame scheduler: plugin FPS, errors, auto-disable, plugin_stats |
| `plugin.log` | Plugin processing: init, ticks, shutdown |
| `system.log` | Event bus, pipeline start/stop |
| `all.log` | Everything combined |

Format (key-value):
```
2026-07-28 10:00:30.456 | INFO     | camera     | event=periodic_stats | uptime_s=30 | capture_fps=24.8 | distribute_fps=14.9 | resolution=1920x1080 | decode_ms=8.3 | buffer_latency_ms=12.5 | reconnects=0 | cpu_percent=23.5 | ram_used_mb=512 | ram_percent=12.8
2026-07-28 10:00:30.457 | INFO     | scheduler  | event=plugin_stats | plugin=on_frame | target_fps=15 | actual_fps=14.9 | avg_process_ms=4.5 | missed_deadlines=0 | errors=0 | disabled=False
```

See [docs/log_inspection_guide.md](docs/log_inspection_guide.md) for full analysis guide.

## Examples

| Script | Purpose |
|--------|---------|
| `run_camera.py` | Camera stream + overlay (FPS, uptime, latency, reconnects) |
| `run_face_detection.py` | Realtime face detection with bbox + landmarks |
| `run_stress_test.py` | Plugin isolation: slow (300ms), crashing, dummy plugins |

## Roadmap

- [x] Task 1: Project structure & Config system
- [x] Task 2: Video Pipeline (RTSP, latest frame buffer, per-callback FPS scheduler)
- [x] Task 3: Event Bus (pub/sub, validation, wildcard, history)
- [x] Task 4: Plugin Manager (BasePlugin, lifecycle, config-driven, auto-disable)
- [x] Task 5: Face Detection (C++ NCNN + OpenCV DNN fallback)
- [ ] Task 6: Face Embedding (MobileFaceNet)
- [ ] Task 7: Face Recognition Plugin
- [ ] Task 8: Data Collection & Auto-Snapshot
- [ ] Task 9: Face Enrollment CLI
- [ ] Task 10: MQTT Client & Cloud Sync
- [ ] Task 11: Edge REST API
- [ ] Task 12: API Documentation
- [ ] Task 13: End-to-End Integration

See [docs/implementation_plan.md](docs/implementation_plan.md) for full details.

## License

Private — DATGROUP Internal Use
