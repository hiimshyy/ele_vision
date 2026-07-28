# Video Pipeline - Metrics & Benchmark Documentation

## Pipeline Data Flow

```
Camera (RTSP) → [decode] → Latest Frame (1 frame) → Frame Scheduler → Per-plugin callbacks
                  │                                        │
                  ├─ decode_time_ms                         ├─ buffer_latency_ms
                  │                                        │
              t_capture                                t_schedule
```

## Metric Definitions

### capture_fps
- **Đo từ**: Số frame decode thành công trong 1 giây
- **Điểm đo**: Capture thread, sau `capture.read()` thành công
- **Ý nghĩa**: Tốc độ thực tế nhận frame từ camera (phụ thuộc camera + network)
- **Bình thường**: Xấp xỉ native FPS của camera (25fps)

### distribute_fps
- **Đo từ**: Số frame gửi cho plugins trong 1 giây
- **Điểm đo**: Distribute thread, sau khi invoke callbacks
- **Ý nghĩa**: Tốc độ xử lý thực tế mà plugins nhận được
- **Bình thường**: Xấp xỉ `process_fps` trong config (15fps)

### decode_time_ms
- **Đo từ**: Trước `capture.read()` → Sau `capture.read()` return
- **Điểm đo**: Capture thread
- **Bao gồm**: Network receive + RTSP demux + H.264/H.265 decode + color convert
- **Ý nghĩa**: Chi phí decode 1 frame. Nếu cao → network chậm hoặc CPU decode bottleneck
- **Bình thường**: 5-20ms (1080p, LAN, software decode trên RK3399)
- **Cảnh báo**: >40ms nghĩa là decode không kịp native FPS

### buffer_latency_ms
- **Đo từ**: Thời điểm frame được lưu vào latest buffer (`t_capture`) → Thời điểm scheduler lấy frame ra gửi cho plugins (`t_schedule`)
- **Điểm đo**: Scheduler thread, tính bằng `t_schedule - frame.timestamp`
- **Bao gồm**: Thời gian chờ trong buffer + scheduler tick delay
- **KHÔNG bao gồm**: Decode time (đã tính riêng), plugin processing time
- **Ý nghĩa**: Độ "cũ" của frame khi plugin nhận được. Ảnh hưởng trực tiếp đến realtime response
- **Bình thường**: 5-50ms (latest frame buffer nên rất thấp)
- **Cảnh báo**: >100ms nghĩa là scheduler bị chậm

### queue_length
- **Đo từ**: Luôn là 0 hoặc 1 (latest frame buffer, không phải ring buffer)
- **Điểm đo**: Scheduler thread
- **Ý nghĩa**: 1 = có frame mới sẵn sàng, 0 = đang chờ frame

### resolution
- **Đo từ**: `frame.shape` sau decode
- **Điểm đo**: Capture thread, chỉ log khi phát hiện thay đổi
- **Ý nghĩa**: Resolution thực tế từ camera (có thể khác config nếu camera auto-switch)

### cpu_percent
- **Đo từ**: `psutil.cpu_percent()` - toàn hệ thống (all cores)
- **Điểm đo**: Stats thread, mỗi 30s
- **Ý nghĩa**: Tổng CPU usage, bao gồm cả process khác

### ram_used_mb / ram_percent
- **Đo từ**: `psutil.virtual_memory()` - toàn hệ thống
- **Điểm đo**: Stats thread, mỗi 30s
- **Ý nghĩa**: RAM usage tổng. Orange Pi 4 Pro có 4GB → giám sát để tránh OOM

## Timeline Diagram

```
Time ──────────────────────────────────────────────────────►

Camera frame arrives
│
├── t0: capture.read() starts
│      ↓
│   [network + decode]  ← decode_time_ms
│      ↓
├── t1: capture.read() returns, frame stored as latest (frame.timestamp = t1)
│      ↓
│   [scheduler checks timing per callback]  ← buffer_latency_ms
│      ↓
├── t2: scheduler invokes callback (when interval elapsed)
│      ↓
│   buffer_latency_ms = t2 - t1
│      ↓
│   [plugin processing]  ← NOT measured here (plugin's responsibility)
│      ↓
└── t3: callback returns
```

## End-to-End Latency (tham khảo)

Tổng latency từ thực tế → hiển thị/phản ứng:

```
Total = camera_encoding + network + decode_time + buffer_latency + plugin_processing
      ≈ 30-50ms        + 1-5ms   + 5-20ms     + 5-50ms        + (varies)
      ≈ 40-130ms typical (without plugin processing)
```

## Periodic Stats Log Format

```
event=periodic_stats | uptime_s=3600 | capture_fps=24.8 | distribute_fps=14.9 | resolution=1920x1080 | decode_ms=8.3 | latency_ms=45.2 | queue_len=1 | reconnects=0 | cpu_percent=23.5 | ram_used_mb=512 | ram_percent=12.8
```

## Troubleshooting Guide

| Triệu chứng | Metric bất thường | Nguyên nhân có thể | Giải pháp |
|-------------|-------------------|--------------------|-----------| 
| Video giật | distribute_fps thấp | CPU overload | Giảm process_fps hoặc resolution |
| Video trễ | buffer_latency_ms cao | Scheduler bị block | Optimize plugin processing time |
| Frame drop nhiều | N/A (latest buffer) | Không áp dụng | Latest buffer luôn có frame mới nhất |
| Reconnect liên tục | reconnects tăng | Network không ổn định | Kiểm tra cable/WiFi, tăng connection_timeout |
| CPU 100% | cpu_percent >90 | Decode + plugins quá nặng | Giảm resolution, giảm FPS, optimize inference |
| RAM tăng liên tục | ram_used_mb tăng | Memory leak | Kiểm tra plugins, restart service |
