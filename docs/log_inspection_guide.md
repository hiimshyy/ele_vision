# Smart Cabin - Hướng dẫn kiểm tra Logs

## Cấu trúc Log Files

```
logs/
├── all.log           # Tất cả events combined (dùng khi cần xem timeline đầy đủ)
├── camera.log        # Video pipeline: connect, decode, reconnect, periodic stats
├── scheduler.log     # Frame scheduler: plugin FPS, errors, auto-disable
├── plugin.log        # Plugin processing: init, tick, shutdown
└── system.log        # System events: event bus subscribe/publish, pipeline start/stop
```

## Log Format

```
{timestamp} | {level} | {module} | {key=value pairs}
```

Ví dụ:
```
2026-07-28 16:40:14.369 | INFO     | camera     | event=periodic_stats | uptime_s=30 | capture_fps=24.8 | ...
```

---

## 1. Camera & Kết nối

### Xem trạng thái kết nối

```bash
# Timeline connect/disconnect đầy đủ
grep "stream_lost\|connected\|connecting\|connection_failed" logs/camera.log

# Đếm số lần mất kết nối
grep "stream_lost" logs/camera.log | wc -l

# Chi tiết lần mất kết nối gần nhất (uptime tại thời điểm đó)
grep "stream_lost" logs/camera.log | tail -3

# Xem thời gian giữa các lần reconnect
grep "stream_lost" logs/camera.log | awk -F'|' '{print $1}'
```

### Phân tích reconnect pattern

```bash
# Xem toàn bộ quá trình reconnect (mất → retry → thành công)
grep "stream_lost\|connecting\|connected" logs/camera.log
```

Output mẫu:
```
16:45:12 | event=stream_lost | uptime_s=327.0 | frames_captured=8175
16:45:17 | event=connecting | attempt=1
16:45:18 | event=connected
```
→ Mất kết nối sau 5.5 phút, recover trong 6 giây.

### Đánh giá

| Tần suất reconnect | Đánh giá | Hành động |
|---------------------|----------|-----------|
| 0 lần/giờ | Tốt | Không cần |
| 1-3 lần/giờ | Bình thường | Kiểm tra cable nếu muốn cải thiện |
| >5 lần/giờ | Có vấn đề | Kiểm tra network, tăng connection_timeout |
| Liên tục | Critical | Camera/network hỏng |

---

## 2. Performance (FPS, CPU, RAM)

### Xem stats gần nhất

```bash
# Stats mới nhất
grep "periodic_stats" logs/camera.log | tail -1

# Stats 5 bản ghi gần nhất
grep "periodic_stats" logs/camera.log | tail -5
```

### Theo dõi FPS theo thời gian

```bash
# Chỉ lấy timestamp + capture_fps + distribute_fps
grep "periodic_stats" logs/camera.log | awk -F'|' '{print $1, $4, $5}'
```

### Tìm peak CPU

```bash
# CPU usage cao nhất (top 5)
grep "periodic_stats" logs/camera.log | grep -oP 'cpu_percent=\K[0-9.]+' | sort -n | tail -5

# RAM usage cao nhất
grep "periodic_stats" logs/camera.log | grep -oP 'ram_used_mb=\K[0-9.]+' | sort -n | tail -5
```

### Decode time (camera/network bottleneck)

```bash
# Decode time cao nhất
grep "periodic_stats" logs/camera.log | grep -oP 'decode_ms=\K[0-9.]+' | sort -n | tail -5

# Buffer latency cao nhất
grep "periodic_stats" logs/camera.log | grep -oP 'buffer_latency_ms=\K[0-9.]+' | sort -n | tail -5
```

### Đánh giá performance

| Metric | Bình thường | Cảnh báo | Critical |
|--------|-------------|----------|----------|
| capture_fps | 23-25 | 15-22 | <15 |
| distribute_fps | ≈ target | target - 3 | < target/2 |
| cpu_percent | <50% | 50-80% | >80% |
| ram_percent | <50% | 50-80% | >80% |
| decode_ms | <20ms | 20-40ms | >40ms |
| buffer_latency_ms | <50ms | 50-100ms | >100ms |

---

## 3. Plugin Health

### Xem trạng thái tất cả plugins

```bash
# Plugin stats gần nhất (FPS, process time, errors)
grep "plugin_stats" logs/scheduler.log | tail -10

# Chỉ xem plugin cụ thể
grep "plugin_stats.*dummy" logs/scheduler.log | tail -3
grep "plugin_stats.*on_frame" logs/scheduler.log | tail -3
```

### Plugin bị disable (crash quá nhiều)

```bash
# Plugin nào bị disable
grep "callback_disabled" logs/scheduler.log

# Xem lỗi trước khi bị disable
grep "callback_error" logs/scheduler.log | tail -10
```

### Plugin lifecycle

```bash
# Plugin load/start/stop events
grep "plugin_loaded\|plugin_started\|plugin_stopped\|plugin_shutdown" logs/plugin.log
```

### Đánh giá plugin

| Metric | Ý nghĩa | Cảnh báo |
|--------|---------|----------|
| actual_fps ≈ target_fps | Plugin nhận đủ frames | actual < target * 0.8 |
| avg_process_ms < interval | Plugin xử lý kịp | process > interval (missed deadline) |
| missed_deadlines = 0 | Không bị trễ | > 0 thường xuyên → plugin quá chậm |
| errors = 0 | Không lỗi | > 0 → kiểm tra logic plugin |
| disabled = False | Đang hoạt động | True → plugin crash liên tiếp 5 lần |

---

## 4. Event Bus

### Xem events được publish

```bash
# Tất cả events nhận được
grep "bus_received" logs/system.log | tail -10

# Đếm theo loại event
grep "bus_received" logs/system.log | grep -oP 'type=\K[^ |]+' | sort | uniq -c

# Xem event bus subscribe
grep "bus_subscribe" logs/system.log
```

---

## 5. Face Recognition

### Xem recognition events

```bash
# Ai được nhận diện?
grep "face_recognized" logs/plugin.log | tail -10

# Mặt lạ (unknown)?
grep "face_unknown" logs/plugin.log | tail -10

# Timeline nhận diện (cả recognized + unknown)
grep "face_recognized\|face_unknown" logs/plugin.log | tail -20
```

### Tracker: track lifecycle

```bash
# Xem tracks được tạo/xóa
grep "track_created\|track_removed" logs/plugin.log | tail -20

# Đếm tracks tạo trong 1 phiên
grep "track_created" logs/plugin.log | wc -l

# Track bị remove vì lost quá lâu (người rời cabin)
grep "track_removed" logs/plugin.log | tail -5
```

### Recognition stats (periodic, mỗi 5s)

```bash
# Stats gần nhất
grep "recognition_stats" logs/plugin.log | tail -5

# Theo dõi embedding ratio (lower = tracker tiết kiệm CPU tốt hơn)
grep "recognition_stats" logs/plugin.log | grep -oP 'embeddings_extracted=\K[0-9]+'

# Detection inference time
grep "recognition_stats" logs/plugin.log | grep -oP 'det_ms=\K[0-9.]+'

# Embedding inference time
grep "recognition_stats" logs/plugin.log | grep -oP 'emb_ms=\K[0-9.]+'
```

### Database events

```bash
# Face enrollment/removal
grep "face_added\|face_removed" logs/plugin.log

# Database init
grep "database_initialized\|database_closed" logs/plugin.log
```

### Model loading

```bash
# Detector + embedder load status
grep "detector_loaded\|embedder_loaded\|load_failed" logs/plugin.log
```

### Đánh giá Face Recognition

| Metric | Bình thường | Cảnh báo | Hành động |
|--------|-------------|----------|-----------|
| det_ms | <15ms (PC), <150ms (OPi) | >200ms | Giảm input_size hoặc process_fps |
| emb_ms | <5ms (PC), <100ms (OPi) | >150ms | Bình thường nếu tracker giảm calls |
| embeddings/frame | <0.5 (tracker hoạt động) | ~1.0 | Tracker không match — kiểm tra IoU threshold |
| active_tracks | 0-6 | >8 | max_tracks đang bị hit |
| face_quality_rejected | Occasional | Liên tục | Camera mờ hoặc min_face_quality quá cao |

---

## 6. Troubleshooting Scenarios

### Scenario: Video bị giật / FPS thấp

```bash
# 1. Kiểm tra capture FPS (camera có gửi đủ frame không?)
grep "periodic_stats" logs/camera.log | tail -3 | grep -oP 'capture_fps=\K[0-9.]+'

# 2. Kiểm tra CPU (overload?)
grep "periodic_stats" logs/camera.log | tail -3 | grep -oP 'cpu_percent=\K[0-9.]+'

# 3. Kiểm tra plugin nào chậm
grep "plugin_stats" logs/scheduler.log | tail -10 | grep "missed_deadlines"

# 4. Kiểm tra decode time (network chậm?)
grep "periodic_stats" logs/camera.log | tail -3 | grep -oP 'decode_ms=\K[0-9.]+'
```

### Scenario: Plugin không nhận frames

```bash
# 1. Plugin có được load không?
grep "plugin_loaded\|plugin_load_error" logs/plugin.log

# 2. Plugin có bị disable không?
grep "callback_disabled" logs/scheduler.log

# 3. Pipeline có đang chạy không?
grep "pipeline_started\|pipeline_stopped" logs/camera.log | tail -3
```

### Scenario: Reconnect liên tục

```bash
# 1. Xem pattern mất kết nối
grep "stream_lost" logs/camera.log | awk -F'|' '{print $1, $3}'

# 2. Kiểm tra decode time trước khi mất (nếu cao = network lag trước khi drop)
grep "periodic_stats" logs/camera.log | grep -oP 'decode_ms=\K[0-9.]+' | tail -10

# 3. Kiểm tra connection timeout hiện tại
grep "connecting" logs/camera.log | tail -3
```

### Scenario: RAM tăng liên tục (memory leak)

```bash
# Xem RAM trend (cột ram_used_mb)
grep "periodic_stats" logs/camera.log | grep -oP 'ram_used_mb=\K[0-9.]+' | tail -20

# So sánh đầu vs cuối
grep "periodic_stats" logs/camera.log | grep -oP 'ram_used_mb=\K[0-9.]+' | head -1
grep "periodic_stats" logs/camera.log | grep -oP 'ram_used_mb=\K[0-9.]+' | tail -1
```

### Scenario: Không nhận diện được (luôn UNKNOWN)

```bash
# 1. Database có faces không?
grep "database_initialized" logs/plugin.log | tail -1
grep "recognition_stats" logs/plugin.log | tail -1 | grep -oP 'db_persons=\K[0-9]+'

# 2. Embedding có extract được không?
grep "recognition_stats" logs/plugin.log | tail -3 | grep -oP 'embeddings_extracted=\K[0-9]+'

# 3. Face bị reject vì quality?
grep "face_quality_rejected\|face_align_failed" logs/plugin.log | tail -5

# 4. Threshold quá cao?
# Kiểm tra embedding_threshold trong config.yaml (default 0.4, thử giảm 0.3)
grep "emb_threshold" logs/plugin.log
```

### Scenario: Recognition event spam (quá nhiều events)

```bash
# 1. Kiểm tra số events
grep "face_recognized\|face_unknown" logs/plugin.log | wc -l

# 2. Tracks bị tạo/xóa liên tục? (tracker IoU threshold quá cao)
grep "track_created\|track_removed" logs/plugin.log | tail -20

# 3. So sánh track count vs recognition events (should be 1:1)
grep "track_created" logs/plugin.log | wc -l
grep "face_recognized\|face_unknown" logs/plugin.log | wc -l
```

---

## 7. Real-time Monitoring

```bash
# Follow camera log (xem realtime)
tail -f logs/camera.log

# Follow scheduler log
tail -f logs/scheduler.log

# Follow all logs
tail -f logs/all.log

# Follow chỉ errors
tail -f logs/all.log | grep "ERROR\|WARNING"
```

---

## 8. Log Rotation

- Mỗi file log tự rotate khi đạt **10 MB**
- Giữ tối đa **5 backup files** (vd: camera.log.1, camera.log.2, ...)
- Tổng max disk usage: ~300 MB (6 files × 10MB × 5 backups)
- Xóa logs cũ: `rm logs/*.log.*`
