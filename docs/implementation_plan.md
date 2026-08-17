# Implementation Plan — Phase 1: Camera AI Platform

> Phase 1 tập trung vào nền tảng core + Face Recognition.
> Phase 2 (Display + Elevator) xem: `docs/display_module_plan.md`

---

## Problem Statement

Xây dựng nền tảng Smart Cabin có kiến trúc plugin-based cho thang máy, chạy trên Orange Pi 4 Pro (RK3399) với camera RTSP. Phase 1 bao gồm core platform + Face Recognition (nhận diện cư dân, ghi log, publish events qua MQTT).

## Requirements

- **Edge device**: Orange Pi 4 Pro (RK3399, 6-core ARM, Mali-T860 GPU, 4GB RAM, Debian 12)
- **Camera**: RTSP stream có sẵn
- **Architecture**: Edge + Cloud (self-hosted), giao tiếp qua MQTT
- **Stack**: Python orchestration + OpenCV DNN inference (C++ NCNN optional)
- **Edge API**: REST API (FastAPI) cho admin operations + MQTT cho realtime events
- **Cloud**: Do đội dev khác phát triển — cung cấp API spec/protocol
- **MQTT Broker**: Tự host, single-topic mode (`embody/w` write, `embody/r` read)
- **Demo module**: Face Recognition — nhận diện < 50 cư dân, ghi log, auto floor call
- **Face Enrollment**: CLI trên Edge (`examples/run_recognition.py enroll`)
- **Extensible**: Plugin architecture (FramePlugin + ServicePlugin) cho modules tương lai

## Background (Hardware Constraints)

**Orange Pi 4 Pro dùng RK3399** (không phải RK3399**Pro**) → **KHÔNG có NPU**:

- Không dùng RKNN Toolkit
- Inference chạy trên CPU (ARM NEON) via OpenCV DNN
- Model lightweight, tối ưu cho ARM

**Model stack**:

| Thành phần   | Model                          | Performance (Orange Pi) |
| -------------- | ------------------------------ | ----------------------- |
| Face Detection | SCRFD-500M (det_500m.onnx)     | ~10-15ms/frame          |
| Face Embedding | MobileFaceNet (w600k_mbf.onnx) | ~20-30ms/frame          |
| Inference      | OpenCV DNN (ARM NEON)          | CPU-only, đủ nhanh    |

**Communication**: MQTT single-topic mode (`embody/w` / `embody/r`) + REST API planned.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Edge — Orange Pi 4 Pro                         │
│                                                             │
│  Camera RTSP → Video Pipeline → Frame Scheduler             │
│                                     │                       │
│                            ┌────────┴────────┐              │
│                            │                 │              │
│                     FramePlugins      ServicePlugin         │
│                     (5fps each)       (event-driven)        │
│                            │                 │              │
│                            └────────┬────────┘              │
│                                     │                       │
│                              Event Bus (pub/sub)            │
│                                     │                       │
│                    ┌────────────────┼────────────────┐      │
│                    │                │                │      │
│              Cloud Sync       Local Logger     REST API     │
│              (MQTT)           (SQLite)         (FastAPI)    │
└─────────────────────────────────────────────────────────────┘
                         │ MQTT
                         ▼
              ┌─────────────────────┐
              │   MQTT Broker       │
              │   (Mosquitto)       │
              └──────┬──────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
  Elevator Controller     Cloud Backend
  (sensor data,           (dashboard,
   floor call)             content mgmt)
```

### Plugin Architecture

| Plugin Type   | Trigger                      | Interface                              | Ví dụ                           |
| ------------- | ---------------------------- | -------------------------------------- | --------------------------------- |
| FramePlugin   | Video frames (FPS-scheduled) | `process_frame(frame, frame_id, ts)` | face_recognition, people_counter  |
| ServicePlugin | EventBus events              | `handle_event(event)`                | display, elevator, sensors, audio |

### MQTT Topics

**Hiện tại (single-topic mode)**:

| Topic        | Direction     | Mô tả                                              |
| ------------ | ------------- | ---------------------------------------------------- |
| `embody/w` | Edge → Cloud | Tất cả events (face recognized, heartbeat, system) |
| `embody/r` | Cloud → Edge | Commands (sync, restart, config update)              |

**Per-topic mode (supported, disabled by default)**:

| Topic                                  | Direction          | Payload                                             |
| -------------------------------------- | ------------------ | --------------------------------------------------- |
| `cabin/{device_id}/face/recognized`  | Edge → Cloud      | `{person_id, person_name, confidence, timestamp}` |
| `cabin/{device_id}/face/unknown`     | Edge → Cloud      | `{confidence, timestamp, bbox}`                   |
| `cabin/{device_id}/status/heartbeat` | Edge → Cloud      | `{uptime, cpu, memory, fps}`                      |
| `cabin/{device_id}/system/error`     | Edge → Cloud      | `{error_message, error_type}`                     |
| `elevator/floor_call`                | Edge → Controller | `{person_id, floor, confidence}`                  |
| `elevator/sensor_data`               | Controller → Edge | `{type, value, unit, ts}`                         |

---

## Project Structure (Actual)

```
ele_vision/
├── edge/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # YAML + Pydantic + env override (SC_*)
│   │   ├── video_pipeline.py      # RTSP capture, frame scheduler, per-callback FPS
│   │   ├── event_bus.py           # Thread-safe pub/sub, Pydantic validation, wildcard
│   │   ├── plugin_manager.py      # BasePlugin ABC, PluginManager, PluginWrapper
│   │   ├── cloud_sync.py          # paho-mqtt v2, offline buffer, heartbeat, commands
│   │   └── logging_setup.py       # Loguru: camera.log, scheduler.log, plugin.log, system.log, all.log
│   │
│   ├── plugins/
│   │   ├── face_recognition/
│   │   │   ├── __init__.py
│   │   │   ├── plugin.py          # Full pipeline: detect → track → embed → match → publish
│   │   │   ├── detector.py        # FaceDetector: SCRFD-500M primary + YuNet fallback
│   │   │   ├── alignment.py       # align_face(): 5-point → 112×112 (Umeyama transform)
│   │   │   ├── embedder.py        # FaceEmbedder: w600k_mbf, 512-dim, cosine_similarity
│   │   │   ├── tracker.py         # FaceTracker: IoU + centroid, track states, re-verify
│   │   │   ├── database.py        # FaceDatabase: SQLite CRUD, find_match (cosine)
│   │   │   └── models/            # det_500m.onnx, w600k_mbf.onnx
│   │   └── dummy/
│   │       ├── __init__.py
│   │       └── plugin.py          # Test plugin (frame counter)
│   │
│   ├── inference/                  # C++ NCNN (optional, for max performance on ARM)
│   │   ├── CMakeLists.txt
│   │   ├── src/
│   │   │   ├── face_detector.cpp
│   │   │   └── python_bindings.cpp
│   │   ├── include/face_detector.h
│   │   ├── build_on_device.sh
│   │   └── download_models.sh
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── enroll_face.py         # Dedicated enrollment (image/camera/batch modes + validator)
│   │   ├── data_recorder.py       # Video/snapshot CLI recorder
│   │   ├── storage_manager.py     # Disk usage auto-cleanup
│   │   └── face_snapshot.py       # Auto face crop on detection (PNG lossless)
│   │
│   ├── tests/                      # 12 test files, 222+ test cases
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_video_pipeline.py
│   │   ├── test_event_bus.py
│   │   ├── test_plugin_manager.py
│   │   ├── test_cloud_sync.py
│   │   ├── test_face_detector.py
│   │   ├── test_face_embedder.py
│   │   ├── test_face_tracker.py
│   │   ├── test_face_database.py
│   │   ├── test_face_recognition_plugin.py
│   │   ├── test_enroll_face.py
│   │   └── test_data_collection.py
│   │
│   ├── config.yaml                 # Edge configuration (camera, mqtt, plugins, logging)
│   └── requirements.txt
│
├── shared/
│   ├── __init__.py
│   ├── event_schemas.py            # EventType enum, BaseEvent, Face events, MQTT_TOPIC_MAP
│   └── mqtt_topics.py             # Topic constants (TOPIC_FACE_RECOGNIZED, etc.)
│
├── examples/
│   ├── run_camera.py              # Camera only + stats overlay
│   ├── run_face_detection.py      # Realtime face detection + bbox + landmarks
│   ├── run_face_embedding.py      # Embedding extraction + pair comparison
│   ├── run_recognition.py         # Full pipeline: enroll, list, test, run, remove
│   └── run_stress_test.py         # Plugin isolation stress test (slow/crash plugins)
│
├── data/
│   ├── db/
│   │   └── faces.db               # SQLite: face embeddings + metadata
│   ├── faces/
│   │   └── {person_id}/           # Enrolled face images (aligned 112×112)
│   └── snapshots/
│       ├── faces/                  # Auto-captured face crops (PNG)
│       └── full/                   # Full frames at recognition time (PNG)
│
├── docs/
│   ├── implementation_plan.md     # This file (Phase 1)
│   ├── vision.md                  # Technical vision & system overview
│   ├── architecture_v2.md         # Architecture v2 (Display + Elevator + Sensors)
│   ├── display_module_plan.md     # Implementation plan Phase 2 (10 tasks)
│   ├── video_pipeline_metrics.md  # Metric definitions & troubleshooting
│   └── log_inspection_guide.md    # How to read and analyze logs
│
├── deploy/
│   ├── setup_edge.sh              # Edge environment setup script
│   ├── mosquitto.conf             # MQTT broker config
│   └── smart-cabin.service        # Systemd service (future)
│
├── config.yaml                    # Root config (same as edge/config.yaml)
├── .env                           # Environment variables (gitignored)
├── .env.example                   # Example env vars
└── .gitignore
```

> **Note**: `edge/api/` directory chưa tồn tại — sẽ tạo khi implement Task 11.

---

## Task Breakdown

### Task 1: Project Structure & Configuration System ✅

**Objective**: Skeleton project + config management.

**Implemented**:

- `edge/core/config.py`: YAML loading, Pydantic models (`SmartCabinConfig`), env var override (`SC_*`)
- Config sections: camera, mqtt, plugins, logging
- `edge/config.yaml`: full working config
- Python venv (uv), `requirements.txt`

---

### Task 2: Video Pipeline — RTSP Capture & Frame Distribution ✅

**Objective**: RTSP capture, latest frame buffer, per-callback FPS scheduling.

**Implemented**:

- `edge/core/video_pipeline.py`: OpenCV VideoCapture, RTSP
- Latest frame buffer (1 frame, atomic swap)
- Per-callback FPS scheduling (mỗi plugin register target FPS riêng)
- Auto-reconnect khi stream ngắt
- Connection timeout
- Callback-based frame distribution (observer pattern)

---

### Task 3: Event Bus ✅

**Objective**: Thread-safe pub/sub cho inter-plugin communication.

**Implemented**:

- `edge/core/event_bus.py`: `EventBus` class, `ThreadPoolExecutor` dispatch
  - Subscribe by `EventType` hoặc wildcard `"*"`
  - Non-blocking publish, error isolation per handler
  - Event history (bounded deque, 100 events)
  - Stats tracking (published, delivered, errors)
- `shared/event_schemas.py`:
  - `EventType` enum (face.detected, face.recognized, face.unknown, person.*, elevator.*, system.*)
  - `BaseEvent` model (event_type, timestamp, source, device_id, metadata)
  - Concrete events: `FaceRecognizedEvent`, `FaceUnknownEvent`, `FaceDetectedEvent`, `SystemErrorEvent`
  - `MQTT_TOPIC_MAP`: event type → MQTT topic mapping
  - `get_mqtt_topic(event)`: resolve topic with device_id
- `shared/mqtt_topics.py`: topic constants (TOPIC_FACE_RECOGNIZED, TOPIC_HEARTBEAT, etc.)

---

### Task 4: Plugin Manager ✅

**Objective**: Plugin lifecycle, config-driven loading, frame routing.

**Implemented**:

- `edge/core/plugin_manager.py`:
  - `BasePlugin` ABC: `initialize()`, `process_frame()`, `shutdown()`, `name`, `version`, `default_fps`
  - `PluginManager`: load from config, lifecycle management
  - `PluginWrapper`: state tracking (UNLOADED → INITIALIZED → RUNNING → STOPPED / ERROR)
- Plugin discovery: `importlib.import_module(f"edge.plugins.{name}.plugin")` → expects `Plugin` class
- Frame routing via `VideoPipeline.register_callback(plugin.process_frame, fps)`
- Auto-disable after 5 consecutive errors
- `edge/plugins/dummy/plugin.py`: test plugin (frame counter, event publishing)

> **Note**: Hiện tại chỉ support `FramePlugin` (camera-driven). `ServicePlugin` (event-driven) sẽ thêm trong Phase 2 Task 1.

---

### Task 5: Face Detection (SCRFD + YuNet) ✅

**Objective**: Face detection engine, Python-based (OpenCV DNN).

**Implemented**:

- `edge/plugins/face_recognition/detector.py`: FaceDetector class
- Primary: SCRFD-500M (det_500m.onnx), 5-point landmarks
- Fallback: YuNet (khi SCRFD unavailable)
- Output: bounding boxes + confidence + landmarks
- Performance: ~10-15ms/frame on ARM

---

### Task 6: Face Embedding (MobileFaceNet) ✅

**Objective**: Face alignment + embedding extraction.

**Implemented**:

- `edge/plugins/face_recognition/alignment.py`: Umeyama similarity transform, ArcFace 5-point reference
- `edge/plugins/face_recognition/embedder.py`: FaceEmbedder (w600k_mbf.onnx)
  - Input: 112×112 aligned BGR → 512-dim L2-normalized vector
  - Thread-safe (threading.Lock)
  - `cosine_similarity()`, `cosine_similarity_batch()`
- `examples/run_face_embedding.py`: CLI demo

---

### Task 7: Face Recognition Plugin (Full Pipeline) ✅

**Objective**: Detection + Tracking + Embedding + Matching integrated.

**Implemented**:

- `edge/plugins/face_recognition/tracker.py`: IoU + centroid distance tracker
  - Track states, embedding skip for identified tracks (80-90% CPU savings)
  - Stale track cleanup, re-verify interval
- `edge/plugins/face_recognition/plugin.py`: Full pipeline
  - detect → filter (min size, blur) → track → embed (new tracks only) → match → publish events
  - 1 event per track entry (no spam)
- `edge/plugins/face_recognition/database.py`: SQLite face DB
  - CRUD, multi-embedding per person, cosine matching

---

### Task 8: Data Collection & Auto-Snapshot ✅

**Objective**: Video recording + auto face crop on detection.

**Implemented**:

- `edge/tools/data_recorder.py`: CLI tool (video/snapshot/continuous modes)
- `edge/tools/storage_manager.py`: disk usage auto-cleanup
- `edge/tools/face_snapshot.py`: auto face crop (PNG lossless) + full frame
- Config: `snapshot_enabled`, `snapshot_dir`, `snapshot_max_per_person_per_day`

---

### Task 9: Face Enrollment Tool ✅

**Objective**: CLI enrollment trực tiếp trên Edge.

**Implemented**:

2 tools enrollment (unified CLI + dedicated tool):

**`examples/run_recognition.py`** — Unified demo/operation CLI:

- `enroll --image <path> --name <name> --id <id>` (multi-image per person)
- `list` (show all enrolled faces)
- `remove --id <id>`
- `test --image <path>` (test matching trên single image)
- `run --url <camera> [--snapshot]` (realtime recognition + display)

**`edge/tools/enroll_face.py`** — Dedicated enrollment tool (feature-rich):

- `image --path <files...> --id <id> --name <name>` (single/multi image)
- `camera --id <id> --name <name> --url 0` (live preview, SPACE capture, multi-shot)
- `batch --folder <dir> [--name-from-file]` (bulk enrollment from folder)
- `list` / `remove --id <id>`
- `EnrollmentValidator`: face size check, blur detection (Laplacian), duplicate detection
- `FaceEnroller`: full workflow class (load models → validate → embed → store)
- Saves aligned face PNG to `data/faces/{person_id}/`

**Cần bổ sung (Phase 2 — Elevator Plugin dependency)**:

- `enroll` command: thêm `--floor <int>` option (tầng làm việc / tầng mặc định)
- Database schema: thêm cột `default_floor INTEGER DEFAULT NULL` vào bảng `faces`
- `database.py`: thêm methods `update_person_floor()`, `get_person_floor()`
- `list` command: hiển thị thêm cột floor
- Migration: auto-detect nếu cột chưa tồn tại → ALTER TABLE ADD COLUMN

---

### Task 10: MQTT Client & Cloud Sync ✅

**Objective**: MQTT publish events, subscribe commands, offline buffer.

**Implemented**:

- `edge/core/cloud_sync.py`: `CloudSync` class (paho-mqtt v2, MQTTv311)
  - **Single-topic mode** (hiện tại): `embody/w` (publish), `embody/r` (subscribe)
  - **Per-topic mode**: supported (sử dụng `MQTT_TOPIC_MAP` từ event_schemas.py), disabled in config
  - `OfflineBuffer`: SQLite queue (`data/db/mqtt_buffer.db`), auto-flush on reconnect
  - Heartbeat: system stats every 30s (CPU, RAM, uptime, buffer count)
  - Last Will: broker publishes stop event on unexpected disconnect
  - Event Bus bridge (`bridge_event_bus`): auto-forward FACE_RECOGNIZED, FACE_UNKNOWN, SYSTEM_ERROR
  - Command handlers: `register_command(name, handler_fn)` for cloud → edge commands
  - `publish_event(event)`: serialize BaseEvent → JSON → MQTT (or buffer if disconnected)
  - `publish_raw(topic, payload)`: raw dict → JSON publish
- `deploy/mosquitto.conf`: broker config
- `shared/mqtt_topics.py`: topic constants (TOPIC_FACE_RECOGNIZED, TOPIC_HEARTBEAT, etc.)

---

### Task 11: Edge REST API ⬜ (Not started)

**Objective**: FastAPI REST API cho admin operations + đội cloud.

**Implementation guidance**:

- `edge/api/main.py` — FastAPI app, port 8080
- Endpoints:
  - `GET /api/status` — system status, uptime, plugin states
  - `GET /api/faces` — list registered faces
  - `POST /api/faces/enroll` — đăng ký face (upload ảnh)
  - `DELETE /api/faces/{id}` — xóa face
  - `GET /api/logs` — recognition logs (filter by time, person)
  - `GET /api/plugins` — plugin status
  - `POST /api/plugins/{name}/restart` — restart plugin
  - `GET /api/stats` — pipeline stats (FPS, frames, latency)
- Auto-generated OpenAPI docs (`/docs`)
- CORS enabled

**Test requirements**:

- Test all endpoints (happy path + error cases)
- Test face enrollment qua API (upload image)
- Test query logs với filters
- Test plugin restart

**Demo**: Swagger UI test + cung cấp OpenAPI spec cho đội cloud.

---

### Task 12: API Documentation & Protocol Spec ⬜ (Not started)

**Objective**: Tài liệu API/Protocol cho đội cloud integrate.

**Implementation guidance**:

- `docs/api_spec.md`:
  - REST API specification (endpoints, request/response schemas)
  - MQTT topics & message formats (JSON schemas)
  - Error codes & handling
  - Example requests/responses
- Auto-export `openapi.json` từ FastAPI
- Sequence diagrams cho main flows

**Demo**: Deliver `docs/api_spec.md` + `openapi.json`.

---

### Task 13: End-to-End Integration & System Testing ⬜ (Not started)

**Objective**: Full integration test, performance tuning, deployment.

**Implementation guidance**:

- Integration test: Camera → Face Recognition → MQTT → broker
- Performance profiling trên Orange Pi: CPU, memory, latency
- Error handling hardening: network failures, camera disconnects, disk full
- Systemd service setup (`deploy/smart-cabin.service`)
- 24h stability run

**Test requirements**:

- End-to-end: enroll → recognize → MQTT event
- Stress test: 24h continuous, no memory leak
- Performance: face appear → MQTT event < 1s
- Failure recovery: simulate network/camera failures

**Demo**: Full flow trên Orange Pi, đội cloud subscribe thành công.

---

## Progress Summary

| Task                       | Status            | Note                                                |
| -------------------------- | ----------------- | --------------------------------------------------- |
| 1. Config System           | ✅ Done           |                                                     |
| 2. Video Pipeline          | ✅ Done           |                                                     |
| 3. Event Bus               | ✅ Done           |                                                     |
| 4. Plugin Manager          | ✅ Done           | Chỉ support FramePlugin (ServicePlugin ở Phase 2) |
| 5. Face Detection          | ✅ Done           |                                                     |
| 6. Face Embedding          | ✅ Done           |                                                     |
| 7. Face Recognition Plugin | ✅ Done           |                                                     |
| 8. Data Collection         | ✅ Done           |                                                     |
| 9. Face Enrollment         | ✅ Done (partial) | Core done. Thiếu`--floor` option cho Phase 2     |
| 10. MQTT Cloud Sync        | ✅ Done           |                                                     |
| 11. Edge REST API          | ⬜ Not started    | `edge/api/` chưa tồn tại                       |
| 12. API Documentation      | ⬜ Not started    |                                                     |
| 13. E2E Integration        | ⬜ Not started    |                                                     |

**Phase 1 completion**: 10/13 tasks done (77%). Task 9 cần bổ sung nhỏ cho Phase 2.

**Remaining work**:

- Task 9 bổ sung `--floor`: ~0.5 day
- Task 11 REST API: ~2 days
- Task 12 API docs: ~1 day
- Task 13 E2E: ~2-3 days
- **Total remaining**: ~5-7 days

---

## Technology Summary

| Component      | Technology                     | Lý do                                 |
| -------------- | ------------------------------ | -------------------------------------- |
| OS             | Debian 12 (ARM64)              | Official support for Orange Pi         |
| Language       | Python 3.12+                   | Fast dev, ML ecosystem                 |
| Inference      | OpenCV DNN (ARM NEON)          | CPU-only, đủ nhanh, no external deps |
| Face Detection | SCRFD-500M (det_500m.onnx)     | Best accuracy/speed on ARM             |
| Face Embedding | MobileFaceNet (w600k_mbf.onnx) | 1M params, 512-dim, 99.4% LFW          |
| Face Tracking  | IoU + centroid tracker         | 80-90% CPU savings                     |
| Database       | SQLite                         | Embedded, lightweight                  |
| REST API       | FastAPI                        | Async, auto-docs, lightweight          |
| MQTT           | paho-mqtt + Mosquitto          | Standard IoT, offline buffer           |
| Logging        | Loguru                         | Structured, rotation, module-based     |
| Config         | YAML + Pydantic                | Validation, env override               |
| Service        | systemd                        | Native Linux, auto-restart             |

---

## Phân chia trách nhiệm

| Phần                           | Ai làm         | Giao tiếp                          |
| ------------------------------- | --------------- | ----------------------------------- |
| Edge Platform (Phase 1 + 2)     | R&D (bạn)      | —                                  |
| MQTT Broker setup               | R&D (bạn)      | Cung cấp broker address            |
| Elevator Controller integration | Đội elevator  | MQTT topics, payload format TBD     |
| Cloud Backend                   | Đội cloud dev | Subscribe MQTT + call Edge REST API |
| Web Dashboard                   | Đội cloud dev | Dùng Cloud Backend API             |
| API Spec & Protocol docs        | R&D (bạn)      | `docs/api_spec.md` + OpenAPI JSON |

---

## What's Next (Phase 2)

Xem `docs/display_module_plan.md` cho chi tiết. Tóm tắt:

| Task                     | Mô tả                            | Priority |
| ------------------------ | ---------------------------------- | -------- |
| ServicePlugin base class | Event-driven plugin type (blocker) | P0       |
| Elevator Plugin          | face → floor → MQTT command      | P0       |
| Content Model + Storage  | Display content SQLite             | P0       |
| Personalization Engine   | Person → content resolution       | P0       |
| Web Display Engine       | FastAPI + WebSocket                | P0       |
| Display Plugin           | Wire events → display             | P0       |
| Content API              | CRUD REST endpoints                | P1       |
| Cloud Content Sync       | MQTT push content                  | P1       |
| Display Frontend         | HTML/CSS/JS client                 | P1       |
| E2E Testing              | Full flow validation               | P0       |

---

*Document version: 2.0*
*Updated: 2026-08-14*
*Author: DATGROUP — Smart Cabin R&D*
