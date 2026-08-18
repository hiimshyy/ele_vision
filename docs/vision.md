# Smart Cabin Platform — Vision & Technical Overview

> Biến cabin thang máy thành không gian thông minh, cá nhân hóa, kết nối.

---

## 1. Tổng quan hệ thống

**Smart Cabin** là nền tảng edge AI chạy trên máy tính nhúng (Orange Pi 4 Pro), đặt trong cabin thang máy, kết nối với bộ điều khiển thang máy qua MQTT. Hệ thống thực hiện:

1. **Nhận diện cư dân** — Face recognition real-time (SCRFD + MobileFaceNet)
2. **Hiển thị cá nhân hóa** — Nội dung riêng cho từng người trên display (web-based)
3. **Tự động gọi tầng** — Nhận diện → lookup floor → MQTT command đến elevator controller
4. **Thu thập dữ liệu** — Sensor data từ elevator controller (qua MQTT), traffic patterns
5. **Giao tiếp giọng nói** — TTS chào hỏi, thông báo (speaker trực tiếp)

### Kiến trúc tổng quan

```
┌───────────────────────────────────────────────────────────────────┐
│                      Smart Cabin Edge Device                      │
│                                                                   │
│  ┌────────────────────┐   ┌────────────────────────────────────┐  │
│  │   Hardware I/O     │   │         Software Platform          │  │
│  │                    │   │                                    │  │
│  │  Camera (RTSP) ────────→ Video Pipeline → Face Recognition  │  │
│  │  Speaker (USB) ←────────── Audio Manager (TTS)              │  │
│  │  Microphone (USB)──────→ (Future: voice commands)           │  │
│  │  Display (HDMI/Web)←────── Display Engine (WebSocket)       │  │
│  └────────────────────┘   │                                    │  │
│                           │  Event Bus (pub/sub)               │  │
│                           │       ↕                            │  │
│                           │  Plugin Manager                    │  │
│                           │   ├── face_recognition (Frame)     │  │
│                           │   ├── display (Service)            │  │
│                           │   ├── elevator (Service)           │  │
│                           │   ├── sensors (Service)            │  │
│                           │   └── audio (Service)              │  │
│                           │                                    │  │
│                           │  MQTT Client (Cloud Sync)          │  │
│                           └────────────────────────────────────┘  │
└───────────────────────────────────┬───────────────────────────────┘
                                    │ MQTT
                                    ▼
                ┌───────────────────────────────────────┐
                │         MQTT Broker (Mosquitto)       │
                └──────┬─────────────────┬──────────────┘
                       │                 │
                       ▼                 ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  Elevator Controller │  │    Cloud Backend     │
        │  (sensor data, floor │  │  (dashboard, content │
        │   call response)     │  │   management, logs)  │
        └──────────────────────┘  └──────────────────────┘
```

### Kết nối phần cứng

| Kết nối trực tiếp (Orange Pi) | Nhận qua MQTT (từ elevator controller) |
| --------------------------------- | ---------------------------------------- |
| Camera IP (RTSP stream)           | Nhiệt độ / Độ ẩm cabin             |
| Speaker (USB/3.5mm)               | Rung lắc (vibration)                    |
| Microphone (USB)                  | Trọng tải cabin                        |
| Display (HDMI hoặc Web)          | Trạng thái cửa (mở/đóng)           |
|                                   | Tầng hiện tại                         |
|                                   | Mã lỗi / cảnh báo                    |

---

## 2. Khả năng hệ thống (System Capabilities)

### Phase 1 — Camera AI (DONE)

| Capability      | Mô tả                                                        | Status |
| --------------- | -------------------------------------------------------------- | ------ |
| Face Detection  | SCRFD-500M, 5-point landmarks, ~15ms/frame                     | Done   |
| Face Embedding  | MobileFaceNet w600k_mbf, 512-dim, cosine matching              | Done   |
| Face Tracking   | IoU + centroid tracker, giảm 80-90% embedding calls           | Done   |
| Face Database   | SQLite, multi-embedding per person, CRUD                       | Done   |
| Face Enrollment | CLI tool + ảnh, support nhiều ảnh/người                   | Done   |
| Video Pipeline  | RTSP capture, per-callback FPS scheduler, auto-reconnect       | Done   |
| Plugin System   | BasePlugin (frame-driven), config-driven loading, auto-disable | Done   |
| Event Bus       | Thread-safe pub/sub, Pydantic validation, wildcard             | Done   |
| MQTT Sync       | paho-mqtt, event bridge, offline buffer, heartbeat             | Done   |
| Data Collection | Video recorder, snapshots, auto face crop (PNG)                | Done   |
| Logging         | Loguru, module-based files, key-value structured format        | Done   |

### Phase 2 — Display + Auto Floor Call (NEXT)

| Capability           | Mô tả                                                 | Dependencies                    |
| -------------------- | ------------------------------------------------------- | ------------------------------- |
| ServicePlugin        | Event-driven plugin base (không cần video frames)     | Plugin Manager update           |
| Personalized Display | Nhận diện → nội dung cá nhân hóa trên screen    | ServicePlugin, Face Recognition |
| Content Management   | CRUD content, zones, rules, scheduling (SQLite)         | —                              |
| Web Display Engine   | FastAPI + WebSocket, real-time push, HTML/CSS/JS client | FastAPI, uvicorn                |
| Auto Floor Call      | face.recognized → lookup floor → MQTT publish         | ServicePlugin, Face DB          |
| Display REST API     | Content CRUD, rules, preview, status                    | FastAPI                         |
| Cloud Content Sync   | Cloud push content/rules via MQTT → local store        | CloudSync                       |

### Phase 3 — Sensors + Voice (Future)

| Capability         | Mô tả                                            | Dependencies        |
| ------------------ | -------------------------------------------------- | ------------------- |
| Sensor Plugin      | Subscribe MQTT sensor data từ elevator controller | ServicePlugin, MQTT |
| TTS Voice Greeting | edge-tts / pyttsx3, chào hỏi khi nhận diện     | Speaker hardware    |
| People Counting    | YOLO-based, cabin occupancy                        | Video Pipeline      |
| Alert System       | Threshold alerts (nhiệt, quá tải, rung)         | Sensor Plugin       |

### Phase 4 — Building Integration (Future)

| Capability             | Mô tả                                       |
| ---------------------- | --------------------------------------------- |
| BMS Protocol           | BACnet/Modbus gateway                         |
| Multi-cabin            | Quản lý nhiều cabin từ 1 cloud            |
| Advanced Floor Routing | Phân luồng tối ưu giờ cao điểm         |
| Predictive Maintenance | Phân tích vibration data → dự đoán lỗi |

---

## 3. Hardware Platform

### Edge Device: Orange Pi 4 Pro

| Spec         | Value                                             |
| ------------ | ------------------------------------------------- |
| SoC          | Rockchip RK3399 (2× Cortex-A72 + 4× Cortex-A53) |
| RAM          | 4GB LPDDR4                                        |
| GPU          | Mali-T860 (unused — CPU inference đủ nhanh)    |
| NPU          | Không có (RK3399, không phải RK3399Pro)       |
| Storage      | 32GB microSD (OS) + 128GB USB (data)              |
| Connectivity | Gigabit Ethernet, WiFi, USB 3.0, HDMI 2.0         |
| OS           | Debian 12 / Ubuntu (ARM64)                        |
| Power        | 5V/4A, ~10-15W under load                         |

### Hardware BOM (PoC)

| Component             | Model                                 | Giá (USD)          | Ghi chú                             |
| --------------------- | ------------------------------------- | ------------------- | ------------------------------------ |
| SBC                   | Orange Pi 4 Pro                       | ~$55                |                                      |
| Camera                | IP Camera 2MP RTSP                    | ~$30                | Hikvision/Dahua                      |
| Display               | 10" IPS (hoặc laptop/tablet cho PoC) | ~$0-120             | Web display — dùng device có sẵn |
| Speaker               | 3W USB/3.5mm                          | ~$10                |                                      |
| Storage               | microSD 32GB + USB 128GB              | ~$20                |                                      |
| PSU                   | 5V/4A                                 | ~$8                 |                                      |
| Enclosure             | Metal/3D print                        | ~$15                |                                      |
| **Total (PoC)** |                                       | **~$140-260** | Tùy có display hay dùng sẵn      |

> **PoC approach**: Display dùng web-based — bất kỳ laptop/tablet mở browser là display client. Không cần mua display riêng.

### Inference Performance (trên Orange Pi 4 Pro)

| Model                          | Task                     | Latency  | Note                      |
| ------------------------------ | ------------------------ | -------- | ------------------------- |
| SCRFD-500M (det_500m.onnx)     | Face Detection           | ~10-15ms | OpenCV DNN, ARM NEON      |
| MobileFaceNet (w600k_mbf.onnx) | Face Embedding           | ~20-30ms | 512-dim vector            |
| Full pipeline (5fps)           | Detect+Track+Embed+Match | ~50-80ms | Chỉ embed khi track mới |

---

## 4. Software Stack

| Layer          | Technology                                  | Lý do chọn                         |
| -------------- | ------------------------------------------- | ------------------------------------ |
| Language       | Python 3.12+                                | Fast prototyping, ML ecosystem       |
| Inference      | OpenCV DNN (fallback) + C++ NCNN (optional) | ARM NEON optimized                   |
| Face Detection | SCRFD-500M (InsightFace buffalo_s)          | Best accuracy/speed tradeoff cho ARM |
| Face Embedding | MobileFaceNet w600k_mbf (InsightFace)       | 1M params, 99.4% LFW                 |
| Database       | SQLite                                      | Lightweight, embedded, no server     |
| MQTT           | paho-mqtt                                   | Standard IoT protocol                |
| Display Server | FastAPI + WebSocket + uvicorn               | Async, auto-docs, lightweight        |
| Display Client | HTML + CSS + Vanilla JS                     | No build step, fast load             |
| TTS            | edge-tts (online) / pyttsx3 (offline)       | Vietnamese support                   |
| Logging        | Loguru                                      | Structured, rotation, module-based   |
| Config         | YAML + Pydantic                             | Validation, env var override         |
| Process        | systemd service                             | Native Linux, auto-restart           |

---

## 5. Plugin Architecture

Hệ thống sử dụng 2 loại plugin:

### FramePlugin (camera-driven)

- Nhận video frames từ pipeline
- Xử lý frame-by-frame (face detection, people counting...)
- Interface: `initialize()`, `process_frame(frame, frame_id, ts)`, `shutdown()`

### ServicePlugin (event-driven)

- Không nhận frames, chỉ nhận events từ EventBus
- Quản lý subsystem (display, elevator, sensors, audio)
- Interface: `start()`, `handle_event(event)`, `stop()`, `health_check()`

### Danh sách plugins

| Plugin               | Type          | Phase    | Mô tả                                             |
| -------------------- | ------------- | -------- | --------------------------------------------------- |
| `face_recognition` | FramePlugin   | 1 (done) | Detect → track → embed → match → publish events |
| `display`          | ServicePlugin | 2 (next) | Personalized content trên web display              |
| `elevator`         | ServicePlugin | 2 (next) | Auto floor call via MQTT                            |
| `sensors`          | ServicePlugin | 3        | MQTT subscriber cho sensor data từ controller      |
| `audio`            | ServicePlugin | 3        | TTS greeting, notification sounds                   |
| `people_counter`   | FramePlugin   | 3        | YOLO-based counting                                 |

---

## 6. Communication Protocol

### MQTT Topic Structure

```
Smart Cabin → Cloud/Controller:
  cabin/{device_id}/face/recognized       # Person identified
  cabin/{device_id}/face/unknown          # Unknown face
  cabin/{device_id}/status/heartbeat      # System stats (30s interval)
  cabin/{device_id}/display/status        # Current display state
  cabin/{device_id}/system/start|stop|error

Smart Cabin → Elevator Controller:
  elevator/floor_call                     # Auto floor call command
  
Elevator Controller → Smart Cabin:
  elevator/sensor_data                    # Sensor readings (temp, vibration, load...)

Cloud → Smart Cabin:
  cabin/{device_id}/display/content/push  # Push display content
  cabin/{device_id}/display/rules/push    # Push personalization rules
  cabin/{device_id}/command/+             # Commands (restart, sync, config)
```

### Giao tiếp với Elevator Controller

| Direction          | Topic                    | Payload         | Ghi chú        |
| ------------------ | ------------------------ | --------------- | --------------- |
| Edge → Controller | `elevator/floor_call`  | TBD (chờ spec) | Gọi tầng      |
| Controller → Edge | `elevator/sensor_data` | TBD (chờ spec) | Sensor readings |

> **Note**: Payload format cho cả 2 chiều sẽ được xác định khi có spec từ đội elevator controller.

---

## 7. Use Cases (Kịch bản kỹ thuật)

### 7.1 Auto Floor Call + Personalized Display

```
Timeline:
  T+0ms     Camera capture frame
  T+15ms    SCRFD detect face, extract landmarks
  T+20ms    Tracker: new face → need embedding
  T+50ms    MobileFaceNet extract 512-dim vector
  T+55ms    Database: cosine match → person_id="0820", name="Sy", floor=8
  T+60ms    EventBus publish: face.recognized {person_id, name, confidence, floor}
  T+65ms    Elevator Plugin: MQTT publish floor_call {floor: 8}
  T+70ms    Display Plugin: resolve content for person "0820"
  T+100ms   WebSocket push: greeting "Xin chào anh Sy — Tầng 8"
  T+600ms   Browser renders with fade-in animation

Total latency: ~100ms (face → MQTT + display push)
```

### 7.2 Multi-person

```
Person A vào (t=0):
  → Recognized: Sy (floor 8)
  → MQTT: floor_call {floor: 8}
  → Display: "Xin chào anh Sy — Tầng 8"

Person B vào (t=2s):
  → Recognized: Ngọc Cần (floor 6)
  → MQTT: floor_call {floor: 6}
  → Display: "Tầng 8 (Sy), Tầng 6 (Ngọc Cần)"

Person C vào (t=4s):
  → Unknown
  → Không gọi tầng
  → Display giữ nguyên (không hiện info cá nhân khi có stranger)
```

### 7.3 Offline Operation

```
Cloud mất kết nối:
  → MQTT buffer messages locally (SQLite)
  → Display vẫn hoạt động (content cached)
  → Face recognition vẫn hoạt động (database local)
  → Floor call vẫn hoạt động (MQTT đến controller qua local broker)
  → Khi cloud reconnect → flush buffer
```

---

## 8. Technical Constraints & Decisions

| Constraint                             | Impact                             | Decision                                     |
| -------------------------------------- | ---------------------------------- | -------------------------------------------- |
| RK3399 không có NPU                  | Không dùng RKNN acceleration     | OpenCV DNN (CPU, ARM NEON) đủ nhanh        |
| 4GB RAM                                | Giới hạn model size + data cache | Lightweight models, bounded buffers          |
| No GPU compute                         | Không dùng CUDA/OpenCL           | CPU-only inference, tối ưu NEON            |
| Cabin environment (rung, nhiệt)       | Hardware reliability               | Industrial-grade enclosure, watchdog         |
| Network không stable                  | MQTT disconnect possible           | Offline-first: local buffer, local DB        |
| Privacy requirement                    | Face data không lên cloud        | Embeddings + matching hoàn toàn on-device  |
| Elevator controller protocol chưa rõ | Payload format TBD                 | Abstract qua MQTT topic, plugin handle parse |
| Display hardware chưa chốt           | Không biết screen nào           | Web-based: bất kỳ browser nào đều work  |

---

## 9. Technical Roadmap (R&D Focus)

### Phase 2 — Immediate (Q4 2026)

```
Week 1-2:
  ├── ServicePlugin base class
  ├── Plugin Manager v2 (load both types)
  └── Elevator Plugin (face → floor → MQTT)

Week 3-4:
  ├── Content model + SQLite storage
  ├── Personalization engine (person → content resolution)
  └── Web Display Engine (FastAPI + WebSocket)

Week 5-6:
  ├── Display Plugin integration (events → personalization → display)
  ├── Display frontend (HTML/CSS/JS)
  └── Content Management REST API

Week 7-8:
  ├── Cloud content sync (MQTT push)
  ├── Face DB update (add default_floor)
  ├── Enrollment tool update (--floor option)
  └── End-to-end testing + performance tuning
```

### PoC Deliverables

| Deliverable          | Mô tả                                               | Success Criteria                        |
| -------------------- | ----------------------------------------------------- | --------------------------------------- |
| Auto Floor Call      | Nhận diện → MQTT floor command                     | Latency < 300ms, multi-person works     |
| Personalized Display | Web page hiển thị nội dung theo người            | Latency < 500ms, smooth transitions     |
| Content API          | REST CRUD cho content + rules                         | Swagger docs, all endpoints work        |
| Multi-person         | Nhiều người → nhiều tầng, display update đúng | 3+ people tested                        |
| 24h stability        | Chạy liên tục không crash/leak                    | Uptime > 99%, no memory growth          |
| Offline resilience   | Mất cloud → vẫn hoạt động                       | Floor call + display work without cloud |

---

## 10. PoC Success Criteria (Technical)

| Metric                         | Target                      | How to Measure                           |
| ------------------------------ | --------------------------- | ---------------------------------------- |
| Face recognition accuracy      | >95%                        | 20+ enrolled, 100+ test passes           |
| Face → Floor Call latency     | <300ms                      | Timestamp diff (event → MQTT publish)   |
| Face → Display update latency | <500ms                      | Timestamp diff (event → WebSocket push) |
| System uptime                  | >99%                        | 30-day continuous run                    |
| Memory stability               | No growth over 24h          | RSS monitoring, no leak                  |
| CPU usage (all plugins)        | <50% sustained              | psutil monitoring                        |
| Concurrent display clients     | 5+                          | WebSocket stress test                    |
| Offline operation              | Full function without cloud | Disconnect cloud, verify all features    |
| Face DB capacity               | 50+ persons, <100ms match   | Benchmark cosine search                  |

---

## 11. Risks (Technical)

| Risk                                        | Impact                   | Mitigation                                            |
| ------------------------------------------- | ------------------------ | ----------------------------------------------------- |
| Elevator controller MQTT format chưa rõ   | Block floor call feature | Abstract payload builder, mock controller for testing |
| Display latency > 500ms                     | Bad UX                   | Profile bottleneck, optimize WebSocket push           |
| WebSocket drops on unstable network         | Display freezes          | Auto-reconnect with exponential backoff               |
| SQLite lock contention (multi-plugin write) | Data corruption          | WAL mode, separate DB per plugin                      |
| Memory leak trong long-running service      | OOM after days           | Bounded buffers, periodic monitoring, restart policy  |
| ARM thermal throttling                      | Performance drop         | Monitor CPU temp, optimize duty cycle                 |
| Face recognition flicker (multi-face)       | Wrong floor call         | Tracker cooldown, confidence threshold for elevator   |

---

## 12. Reference

| Document                        | Mô tả                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `docs/architecture_v2.md`     | Chi tiết kiến trúc: plugin system, display, sensors, elevator, MQTT topics |
| `docs/display_module_plan.md` | Implementation plan chi tiết cho Display + Elevator (10 tasks)               |
| `docs/implementation_plan.md` | Plan gốc Phase 1 (face recognition, 13 tasks)                                |

---

## Appendix: Business Context (Tóm tắt)

> Phần này giữ lại ở mức overview để team R&D hiểu context tại sao build.

**Target market**: Platform mở cho mọi loại tòa nhà (chung cư, văn phòng, bệnh viện, thương mại). Khách hàng chọn module phù hợp.

**Revenue model**: Hardware (one-time) + SaaS subscription (monthly per cabin) + Advertising revenue share.

**Competitive edge**: Edge-first (privacy), modular (plugin-based), affordable ($140-260 BOM vs $2000+ enterprise), open platform (REST/MQTT/WebSocket).

**GTM**: Internal pilot (DATGROUP building) → 3-5 early adopters → channel partners.

---

*Document version: 2.0*
*Created: 2026-08-14*
*Author: DATGROUP — Smart Cabin R&D*
