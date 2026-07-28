# Smart Cabin Platform

Nền tảng AI cho cabin thang máy, chạy trên Orange Pi 4 Pro với camera RTSP. Kiến trúc plugin-based cho phép mở rộng từ Face Recognition sang People Counting, Elevator Control, và các module khác.

## Hardware

| Component | Spec |
|-----------|------|
| SBC | Orange Pi 4 Pro (RK3399, 6-core ARM, 4GB RAM) |
| Camera | IP Camera with RTSP stream |
| Inference | CPU (ARM NEON) via NCNN |

## Features (Current)

- **Video Pipeline** — RTSP capture with auto-reconnect, FPS throttling, ring buffer
- **Plugin Architecture** — Extensible module system for AI tasks
- **Structured Logging** — Loguru, key-value format, module-based log files, periodic system stats
- **Edge REST API** — FastAPI for remote management
- **MQTT** — Realtime events for cloud integration

## Quick Start

```bash
# Clone
git clone https://github.com/hiimshyy/ele_vision.git
cd ele_vision

# Setup (requires Python 3.12+)
uv venv --python 3.12
uv pip install -e .
uv pip install opencv-python numpy loguru psutil

# Run demo stream
uv run python demo_stream.py --url "rtsp://USER:PASS@IP:554/stream" --scale 0.5

# Run with webcam
uv run python demo_stream.py --url 0 --fps 15

# Run tests
uv run pytest edge/tests/ -v
```

## Project Structure

```
edge/
├── core/
│   ├── config.py             # YAML config + env override
│   ├── video_pipeline.py     # RTSP capture & frame distribution
│   ├── logging_setup.py      # Loguru structured logging
│   ├── event_bus.py          # (planned) Internal pub/sub
│   ├── plugin_manager.py     # (planned) Plugin lifecycle
│   └── cloud_sync.py         # (planned) MQTT client
├── plugins/
│   └── face_recognition/     # (planned) FR module
├── api/                      # (planned) Edge REST API
├── tools/                    # (planned) CLI utilities
└── tests/
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
| `camera.log` | Video pipeline events, FPS stats |
| `system.log` | General system events |
| `all.log` | Everything combined |

Format (key-value, parseable):
```
2026-07-28 10:00:30.456 | INFO     | camera     | event=periodic_stats | uptime_s=30 | capture_fps=24.8 | distribute_fps=14.9 | cpu_percent=23.5 | ram_used_mb=512 | ram_percent=12.8
```

## Demo Stream Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `s` | Print stats to console |

Overlay displays: FPS, Resolution, Uptime, Latency, Reconnects

## Roadmap

- [x] Task 1: Project structure & Config system
- [x] Task 2: Video Pipeline (RTSP, auto-reconnect, FPS throttle)
- [ ] Task 3: Event Bus
- [ ] Task 4: Plugin Manager
- [ ] Task 5: Face Detection (C++ NCNN)
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
