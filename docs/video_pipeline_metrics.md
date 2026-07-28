# Video Pipeline - Metrics & Benchmark Documentation

## Pipeline Data Flow

```
Camera (RTSP) → [decode] → Ring Buffer → [throttle] → Callbacks (Plugins)
                  │                          │
                  ├─ decode_time_ms           ├─ latency_ms
                  │                          │
              t_capture                  t_distribute
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

### latency_ms
- **Đo từ**: Thời điểm frame được đặt vào ring buffer (`t_capture`) → Thời điểm frame được lấy ra và bắt đầu gửi cho plugins (`t_distribute`)
- **Điểm đo**: Distribute thread, tính bằng `t_distribute - frame.timestamp`
- **Bao gồm**: Thời gian chờ trong buffer + FPS throttling delay
- **KHÔNG bao gồm**: Decode time (đã tính riêng), plugin processing time
- **Ý nghĩa**: Độ "cũ" của frame khi plugin nhận được. Ảnh hưởng trực tiếp đến realtime response
- **Bình thường**: 10-70ms (phụ thuộc process_fps: 1000/fps = frame interval)
- **Cảnh báo**: >200ms nghĩa là buffer đang backlog hoặc plugins xử lý chậm

### queue_length
- **Đo từ**: Số frame hiện có trong ring buffer tại thời điểm distribute lấy frame
- **Điểm đo**: Distribute thread, trong buffer lock
- **Ý nghĩa**: Nếu queue luôn cao → capture nhanh hơn distribute xử lý kịp
- **Bình thường**: 1-3
- **Cảnh báo**: >10 liên tục → cần giảm process_fps hoặc optimize plugins

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
├── t1: capture.read() returns, frame placed into buffer (frame.timestamp = t1)
│      ↓
│   [waiting in ring buffer]  ← queue wait
│      ↓
│   [FPS throttling sleep]    ← throttle wait
│      ↓
├── t2: distribute thread picks frame, sends to callbacks
│      ↓
│   latency_ms = t2 - t1
│      ↓
│   [plugin processing]  ← NOT measured here (plugin's responsibility)
│      ↓
└── t3: callback returns
```

## End-to-End Latency (tham khảo)

Tổng latency từ thực tế → hiển thị/phản ứng:

```
Total = camera_encoding + network + decode_time + buffer_latency + plugin_processing
      ≈ 30-50ms        + 1-5ms   + 5-20ms     + 10-70ms       + (varies)
      ≈ 50-150ms typical (without plugin processing)
```

## Periodic Stats Log Format

```
event=periodic_stats | uptime_s=3600 | capture_fps=24.8 | distribute_fps=14.9 | resolution=1920x1080 | decode_ms=8.3 | latency_ms=45.2 | queue_len=1 | reconnects=0 | cpu_percent=23.5 | ram_used_mb=512 | ram_percent=12.8
```

## Troubleshooting Guide

| Triệu chứng | Metric bất thường | Nguyên nhân có thể | Giải pháp |
|-------------|-------------------|--------------------|-----------| 
| Video giật | distribute_fps thấp | CPU overload | Giảm process_fps hoặc resolution |
| Video trễ | latency_ms cao | Queue backlog | Giảm process_fps, optimize plugins |
| Frame drop nhiều | queue_len = 30 (max) | Distribute không kịp | Giảm capture_fps hoặc tăng process_fps |
| Reconnect liên tục | reconnects tăng | Network không ổn định | Kiểm tra cable/WiFi, tăng connection_timeout |
| CPU 100% | cpu_percent >90 | Decode + plugins quá nặng | Giảm resolution, giảm FPS, optimize inference |
| RAM tăng liên tục | ram_used_mb tăng | Memory leak | Kiểm tra plugins, restart service |
