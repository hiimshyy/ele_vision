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
- **Face Embedding** — MobileFaceNet (w600k_mbf.onnx), 512-dim vectors, cosine similarity matching
- **Face Alignment** — 5-point landmark similarity transform → 112×112 canonical pose
- **Face Tracking** — IoU + centroid distance tracker, reduces embedding calls 80-90%, handles fast movement
- **Face Database** — SQLite face storage, cosine similarity matching, multi-embedding per person
- **Data Collection** — Video recorder, periodic snapshots, auto face crop on detection (PNG lossless)
- **OpenCV Display** — Realtime UI with bbox, name, confidence, face size, landmarks, stats bar
- **MQTT Cloud Sync** — paho-mqtt, event bridge, heartbeat, offline buffer, command handling
- **Edge REST API** — FastAPI, face enrollment upload, system status, auto-generated OpenAPI docs
- **Structured Logging** — Loguru, key-value format, module-based files (camera/scheduler/plugin/system)
- **Per-plugin Metrics** — actual_fps, avg_process_ms, missed_deadlines, errors, disabled status

## Quick Start

```bash
# Clone
git clone https://github.com/hiimshyy/ele_vision.git
cd ele_vision

# Setup (requires Python 3.12+)
uv venv --python 3.12
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

uv pip install -e .
uv pip install opencv-python numpy loguru psutil

# Download face detection model
bash edge/inference/download_models.sh
# Also download embedding model (w600k_mbf.onnx from InsightFace buffalo_s)
# Place in: edge/plugins/face_recognition/models/w600k_mbf.onnx

# Run face recognition (enroll + recognize)
python examples/run_recognition.py enroll --image face.jpg --name "Alice" --id p001
python examples/run_recognition.py run --url 0 --snapshot

# Run other examples
python examples/run_camera.py --url "rtsp://USER:PASS@IP:554/stream" --scale 0.5
python examples/run_face_detection.py --url 0 --det-fps 5 --scale 0.5
python examples/run_face_embedding.py --compare face1.jpg face2.jpg
python examples/run_stress_test.py --url "rtsp://..." --duration 60

# Data collection
python -m edge.tools.data_recorder --mode video --url 0 --duration 60
python -m edge.tools.data_recorder --mode snapshot --url 0 --interval 5

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
│   │   ├── alignment.py      # Face alignment (5-point → 112×112)
│   │   ├── embedder.py       # FaceEmbedder (MobileFaceNet) + cosine similarity
│   │   ├── tracker.py        # IoU + centroid distance face tracker
│   │   ├── database.py       # SQLite face database (CRUD + matching)
│   │   ├── plugin.py         # Face Recognition plugin (full pipeline)
│   │   └── models/           # det_500m.onnx, w600k_mbf.onnx
│   └── dummy/
│       └── plugin.py         # Test plugin (frame counter)
├── tools/
│   ├── data_recorder.py      # Video/snapshot CLI recorder
│   ├── storage_manager.py    # Disk usage auto-cleanup
│   └── face_snapshot.py      # Auto face crop on detection (PNG lossless)
├── inference/
│   ├── CMakeLists.txt        # C++ build (NCNN + pybind11)
│   ├── src/
│   │   ├── face_detector.cpp
│   │   └── python_bindings.cpp
│   ├── include/face_detector.h
│   ├── build_on_device.sh    # Build script for Orange Pi
│   └── download_models.sh    # Model downloader
├── tests/                    # 222+ tests
└── config.yaml
examples/
├── run_camera.py             # Camera only + stats overlay
├── run_face_detection.py     # Camera + realtime face detection (with face size display)
├── run_face_embedding.py     # Face embedding extraction + comparison
├── run_recognition.py        # Full pipeline: enroll, list, test, run realtime, snapshot
└── run_stress_test.py        # Plugin isolation stress test
docs/
├── implementation_plan.md    # Full task breakdown
├── video_pipeline_metrics.md # Metric definitions & troubleshooting
└── log_inspection_guide.md   # How to read and analyze logs
```

## Configuration

Edit `edge/config.yaml`:

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
        embedding_model: "w600k_mbf.onnx"
        embedding_threshold: 0.4
        min_face_size: 60
        min_face_quality: 50.0
        database_path: "data/db/faces.db"
        # Tracker
        tracker_iou_threshold: 0.4
        tracker_max_lost: 15
        tracker_max_tracks: 10
        tracker_reverify_interval: 15
        # Auto-snapshot (PNG lossless)
        snapshot_enabled: false
        snapshot_dir: "data/snapshots"
        snapshot_max_per_person_per_day: 10
        snapshot_save_full_frame: true

mqtt:
  broker_host: "localhost"
  broker_port: 1883
  device_id: "cabin-001"

logging:
  level: "INFO"
```

Override with environment variables: `SC_CAMERA_URL=rtsp://...`

## Data Directory

```
data/
├── db/
│   └── faces.db              # SQLite database (embeddings + metadata)
├── faces/
│   └── {person_id}/          # Enrolled face images (aligned 112×112)
├── snapshots/
│   ├── faces/                # Auto-captured face crops (PNG)
│   └── full/                 # Full frames (PNG, clean - no annotation)
├── videos/                   # Recorded video segments
└── frames/                   # Periodic snapshot captures
```

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

## Face Recognition Usage

```bash
# 1. Enroll faces (from images)
python examples/run_recognition.py enroll --image photo1.jpg --name "Nguyen Van A" --id p001
python examples/run_recognition.py enroll --image photo2.jpg --name "Nguyen Van A" --id p001  # multiple angles

# 2. List enrolled faces
python examples/run_recognition.py list

# 3. Test on single image
python examples/run_recognition.py test --image test.jpg

# 4. Run realtime recognition (OpenCV window)
python examples/run_recognition.py run --url 0                    # webcam
python examples/run_recognition.py run --url 0 --snapshot         # with auto-snapshot
python examples/run_recognition.py run --url 0 --min-size 40      # detect smaller faces
python examples/run_recognition.py run --url "rtsp://..." --scale 0.5  # RTSP camera

# 5. Remove enrolled face
python examples/run_recognition.py remove --id p001
```

Display overlay shows: bounding box (green=recognized, red=unknown), name + confidence + face size, 5-point landmarks, stats bar (display FPS, process FPS, tracks, inference times).

Controls: `[q/ESC]` quit | `[s]` screenshot

## Examples

| Script | Purpose |
|--------|---------|
| `run_camera.py` | Camera stream + overlay (FPS, uptime, latency, reconnects) |
| `run_face_detection.py` | Realtime face detection with bbox + landmarks |
| `run_face_embedding.py` | Face embedding extraction, alignment, comparison |
| `run_recognition.py` | Full pipeline: enroll, test, realtime recognition |
| `run_stress_test.py` | Plugin isolation: slow (300ms), crashing, dummy plugins |

## Roadmap

- [x] Task 1: Project structure & Config system
- [x] Task 2: Video Pipeline (RTSP, latest frame buffer, per-callback FPS scheduler)
- [x] Task 3: Event Bus (pub/sub, validation, wildcard, history)
- [x] Task 4: Plugin Manager (BasePlugin, lifecycle, config-driven, auto-disable)
- [x] Task 5: Face Detection (C++ NCNN + OpenCV DNN fallback)
- [x] Task 6: Face Embedding (MobileFaceNet w600k_mbf, 512-dim, alignment + cosine similarity)
- [x] Task 7: Face Recognition Plugin (tracker + database + full pipeline integration)
- [x] Task 8: Data Collection & Auto-Snapshot
- [x] Task 9: Face Enrollment CLI
- [x] Task 10: MQTT Client & Cloud Sync
- [x] Task 11: Edge REST API
- [ ] Task 12: API Documentation
- [ ] Task 13: End-to-End Integration

See [docs/implementation_plan.md](docs/implementation_plan.md) for full details.

## License

Private — DATGROUP Internal Use
