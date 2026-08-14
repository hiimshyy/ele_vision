# Smart Cabin Platform — Vision Document

> Biến cabin thang máy thành không gian thông minh, cá nhân hóa, kết nối.

---

## 1. Vision Statement

**Smart Cabin** là nền tảng AI/IoT mở, biến cabin thang máy — một không gian mà hàng trăm người đi qua mỗi ngày — từ "dead space" thành một điểm chạm thông minh: nhận biết cư dân, hiển thị nội dung cá nhân hóa, giao tiếp bằng giọng nói, thu thập dữ liệu vận hành, và kết nối với hệ thống quản lý tòa nhà (BMS).

Platform được thiết kế theo kiến trúc **plugin-based**, cho phép khách hàng chọn và kết hợp các module phù hợp với nhu cầu: từ chung cư cao cấp đến văn phòng, bệnh viện, hay trung tâm thương mại.

---

## 2. Problem & Opportunity

### Thực trạng

| Vấn đề | Chi tiết |
|--------|----------|
| Dead space | Cabin thang máy chiếm ~2m2, hàng trăm lượt/ngày, nhưng không khai thác giá trị |
| Thông tin một chiều | Bảng thông báo giấy, poster cũ, không cập nhật |
| Không nhận biết | Thang máy không biết ai đang đi, không thể cá nhân hóa |
| Vận hành mù | Không có dữ liệu: bao nhiêu người, giờ cao điểm, tình trạng cabin |
| An ninh hạn chế | Camera CCTV record nhưng không phân tích real-time |

### Cơ hội

- **50-200 lượt đi/ngày** mỗi cabin → touchpoint tiếp cận cư dân lớn
- **Dwell time 30-60s** → đủ thời gian truyền tải thông tin
- **Captive audience** → tỷ lệ chú ý cao hơn quảng cáo ngoài trời
- **Data goldmine** → traffic patterns, occupancy, behavioral insights
- **Xu hướng smart building** → BMS integration, ESG reporting, proptech investment tăng

---

## 3. Value Proposition

### Theo stakeholder

| Stakeholder | Giá trị nhận được |
|-------------|-------------------|
| **Cư dân / Nhân viên** | Trải nghiệm cá nhân hóa (chào hỏi, nhắc lịch, thông báo), tiện lợi (tự gọi tầng), an toàn |
| **Chủ đầu tư / BMS** | Data vận hành (occupancy, peak hours), tiết kiệm năng lượng, tăng giá trị bất động sản, doanh thu quảng cáo |
| **Ban quản lý tòa nhà** | Thông báo tức thời (sự cố, bảo trì), giám sát an ninh, kiểm soát truy cập |
| **Nhà quảng cáo** | Targeted ads (theo demographics, tầng, thời điểm), đo lường impression chính xác (face detection = viewability) |
| **Đơn vị bảo trì thang máy** | Predictive maintenance data (rung lắc, nhiệt, tần suất), giảm downtime |

### Unique Selling Points

1. **Edge AI First** — Xử lý hoàn toàn tại chỗ, không phụ thuộc cloud, privacy-compliant (face data không rời device)
2. **Plugin Architecture** — Modular, khách hàng chọn features cần, upgrade dần
3. **Personalization at Scale** — Nhận diện → nội dung riêng cho từng người, không phải broadcast chung
4. **Platform Play** — Không bán hardware thuần, bán platform + ecosystem (content marketplace, ad network, BMS integration)
5. **Hardware Agnostic Display** — Support HDMI/Tablet/Web, khách hàng chọn display phù hợp budget

---

## 4. Use Cases by Vertical

### 4.1 Chung cư cao cấp (Luxury Residential)

| Feature | Mô tả |
|---------|--------|
| Chào hỏi cá nhân | "Xin chào anh Minh, chúc buổi sáng tốt lành" |
| Thông báo cá nhân | "Anh có bưu kiện tại sảnh", "Phí quản lý tháng 8 đã đến hạn" |
| Thời tiết + giao thông | Hiển thị thời tiết, tình trạng giao thông theo tuyến đường thường đi |
| Quảng cáo targeted | Spa, gym, nhà hàng trong tòa nhà — theo demographics |
| Access log | Ghi nhận ai vào/ra, thời điểm (cho BMS dashboard) |
| Emergency broadcast | Thông báo cháy, động đất, sơ tán — override mọi content |

**Kịch bản mẫu:**
> Anh Minh (P0820) bước vào thang máy lúc 7:45 sáng. Camera nhận diện → Display hiện: "Chào anh Minh! Hôm nay 32°C, có mưa chiều. Bưu kiện #2847 đang chờ tại sảnh." Speaker: "Xin chào anh Minh". Thang tự gọi tầng 8 qua MQTT → elevator controller chọn tầng.

### 4.2 Tòa nhà văn phòng (Office Building)

| Feature | Mô tả |
|---------|--------|
| Access Control | Chỉ cho phép nhân viên đã đăng ký sử dụng thang |
| Auto Floor Call | Nhận diện → tự gọi tầng làm việc (MQTT → elevator controller) |
| Floor routing | Nhiều người cùng cabin → gọi nhiều tầng tương ứng |
| Meeting reminder | "Cuộc họp 9:00 tại phòng 15A" (sync calendar) |
| Phân luồng giờ cao điểm | Analytics → suggest thời điểm tránh đông |
| Quảng cáo nội bộ | HR announcements, company events, new hire welcome |
| Visitor management | Khách chưa đăng ký → hiển thị hướng dẫn, QR check-in |

### 4.3 Bệnh viện (Healthcare)

| Feature | Mô tả |
|---------|--------|
| Strict access | Chỉ nhân viên có badge/face mới dùng thang chuyên dụng |
| Hygiene monitoring | Cảm biến nhiệt (phát hiện sốt), nhắc rửa tay |
| Emergency priority | Bác sĩ/y tá → ưu tiên thang, bỏ qua queue |
| Wayfinding | "Khoa Tim mạch: Tầng 7, rẽ trái" cho bệnh nhân/khách |
| Load monitoring | Giám sát trọng tải (cáng, xe đẩy) |
| Sterilization alert | Nhắc lịch vệ sinh cabin |

### 4.4 Trung tâm thương mại (Commercial / Retail)

| Feature | Mô tả |
|---------|--------|
| Quảng cáo CPM | Tính impression dựa face detection (viewability metric) |
| Demographic targeting | Tuổi, giới tính → quảng cáo phù hợp (future: age/gender estimation) |
| Wayfinding interactive | "Tầng 3: Thời trang, Tầng 5: Ẩm thực" |
| Event promotion | Flash sale, sự kiện đang diễn ra |
| Traffic analytics | Footfall counting, peak hours, dwell time per floor |
| Revenue reporting | Dashboard cho landlord: impressions, engagement, revenue |

---

## 5. Feature Matrix & Roadmap

### Feature Matrix

| Feature | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---------|:-------:|:-------:|:-------:|:-------:|
| Face Recognition | Done | Done | Done | Done |
| Face Tracking (IoU) | Done | Done | Done | Done |
| Event Bus + MQTT Sync | Done | Done | Done | Done |
| Structured Logging | Done | Done | Done | Done |
| **Personalized Display** | - | **Next** | Done | Done |
| Content Management | - | Next | Done | Done |
| Ad Scheduling | - | Next | Done | Done |
| **Auto Floor Call (MQTT)** | - | **Next** | Done | Done |
| **People Counting** | - | - | Next | Done |
| **TTS / Voice Greeting** | - | - | Next | Done |
| **Sensor Suite** | - | - | Next | Done |
| Occupancy Analytics | - | - | Next | Done |
| BMS Integration | - | - | - | Next |
| Predictive Maintenance | - | - | - | Next |
| Multi-cabin Orchestration | - | - | - | Next |

### Roadmap

```
Phase 1: Camera AI (DONE)                    Phase 2: Smart Display + Floor Call (NEXT)
Q3 2026                                       Q4 2026
┌─────────────────────────┐                  ┌─────────────────────────┐
│ Face Detection (SCRFD)  │                  │ Personalized Display    │
│ Face Embedding (MFNet)  │                  │ Content Management      │
│ Face Tracking (IoU)     │                  │ Web Display Engine      │
│ Face Database (SQLite)  │                  │ Ad Scheduling           │
│ MQTT Cloud Sync         │                  │ Display REST API        │
│ Plugin Architecture     │                  │ Cloud Content Sync      │
│ Data Collection         │                  │ ServicePlugin Base      │
└─────────────────────────┘                  │ Auto Floor Call (MQTT)  │
                                              └─────────────────────────┘

Phase 3: Sensor & Voice                      Phase 4: Building Integration
Q1-Q2 2027                                    Q3-Q4 2027
┌─────────────────────────┐                  ┌─────────────────────────┐
│ People Counting (YOLO)  │                  │ BMS Protocol (BACnet)   │
│ TTS Voice Greeting      │                  │ Multi-cabin Management  │
│ Temperature Sensor      │                  │ Predictive Maintenance  │
│ Vibration Sensor        │                  │ Energy Optimization     │
│ Air Quality Sensor      │                  │ Visitor Management      │
│ Load Cell (weight)      │                  │ Mobile App Integration  │
│ Occupancy Analytics     │                  │ Advanced Floor Routing  │
└─────────────────────────┘                  └─────────────────────────┘
```

---

## 6. Hardware BOM (Bill of Materials)

### Tier 1: Basic (Camera AI Only) — Phase 1

| Component | Model | Giá (USD) | Ghi chú |
|-----------|-------|-----------|---------|
| SBC | Orange Pi 4 Pro (RK3399, 4GB) | ~$55 | Edge compute |
| Camera | IP Camera 2MP RTSP | ~$25-40 | Hikvision/Dahua mini |
| Storage | 32GB microSD + 128GB USB | ~$20 | OS + data |
| PSU | 5V/4A adapter | ~$8 | |
| Enclosure | 3D printed / metal box | ~$15 | Gắn trần cabin |
| **Total** | | **~$125-140** | |

### Tier 2: Standard (+ Display + Speaker) — Phase 2

| Component | Model | Giá (USD) | Ghi chú |
|-----------|-------|-----------|---------|
| Tier 1 (above) | | ~$130 | |
| Display | 10.1" IPS (HDMI hoặc Android Tablet) | ~$80-150 | Touch optional |
| Speaker | 3W mini speaker + amplifier | ~$10 | Cho TTS/notifications |
| HDMI cable | Flat HDMI 2.0 | ~$5 | Nếu dùng HDMI display |
| Mount | Display bracket (cabin wall) | ~$15 | |
| **Total** | | **~$240-310** | |

### Tier 3: Premium (Full Kit) — Phase 3-4

| Component | Model | Giá (USD) | Ghi chú |
|-----------|-------|-----------|---------|
| Tier 2 (above) | | ~$275 | |
| Temp/Humidity | DHT22 / SHT30 | ~$5 | I2C |
| Vibration | ADXL345 3-axis accelerometer | ~$8 | I2C, predictive maintenance |
| Air Quality | SGP30 / CCS811 | ~$12 | VOC, CO2 equivalent |
| Load Cell | HX711 + 50kg cell | ~$10 | Trọng tải cabin |
| Relay Module | 2-channel 5V relay | ~$5 | Elevator control |
| Microphone | USB mini mic | ~$8 | Voice commands (future) |
| GPIO Breakout | Header + wiring | ~$5 | |
| **Total** | | **~$330-400** | |

### Display Options (Chi tiết)

| Option | Ưu điểm | Nhược điểm | Giá | Phù hợp |
|--------|----------|------------|-----|----------|
| **HDMI Display 10"** | Đơn giản, Orange Pi render trực tiếp | Cần HDMI cable, Orange Pi chịu tải render | $80-120 | PoC, small deploy |
| **Android Tablet 10"** | Touch, có WiFi, chạy app riêng | Cần develop Android app hoặc dùng browser | $100-150 | Mid-range |
| **Web Display (any screen)** | Flexible, bất kỳ device nào có browser | Cần WiFi local stable | $0 (dùng device có sẵn) | Flexible |
| **Commercial display 15-21"** | Chuyên dụng, độ sáng cao, 24/7 | Đắt | $200-500 | Premium deploy |

**Recommendation cho PoC**: Dùng **Web Display** approach — Orange Pi host local web server, bất kỳ tablet/TV nào mở browser đều là display. Không cần mua hardware mới, test trên laptop/tablet có sẵn.

---

## 7. Competitive Analysis

### Existing Solutions

| Giải pháp | Mô tả | Hạn chế | Smart Cabin khác biệt |
|-----------|--------|---------|----------------------|
| **Captivate (USA)** | Digital signage trong thang máy, content broadcast | Không AI, không cá nhân hóa, chỉ quảng cáo | Personalization + AI |
| **Schindler DoorShow** | Màn hình trên cửa thang, tin tức/thời tiết | Tied to Schindler hardware, không mở | Platform mở, hardware agnostic |
| **KovaiTech** | IoT sensors cho thang máy, predictive maintenance | Không display, không AI vision | Full experience (vision + display + sensors) |
| **Vertical Impression** | Quảng cáo thang máy truyền thống | Poster/TV đơn giản, không targeting | AI targeting, measurable ROI |
| **Kone DX** | Connected elevator, cloud monitoring | Enterprise only, đắt, closed ecosystem | Affordable, modular, open |

### Competitive Advantages

1. **Edge-first**: Data không rời device → privacy compliance (GDPR, PDPA), low latency
2. **Open platform**: Không lock-in vendor, khách hàng own data
3. **Modular pricing**: Trả cho what you use, không phải full suite
4. **Developer-friendly**: Plugin SDK, REST API, MQTT — dễ integrate
5. **Cost-effective**: Hardware BOM ~$130-400 vs enterprise solutions $2000+

---

## 8. Business Model

### Revenue Streams

```
┌─────────────────────────────────────────────────────────┐
│                   Revenue Model                          │
├──────────────────┬──────────────────────────────────────┤
│ Hardware         │ One-time: Sell cabin kit (margin 30-40%)│
│                  │ Tier 1: ~$200  Tier 2: ~$450          │
│                  │ Tier 3: ~$600                          │
├──────────────────┼──────────────────────────────────────┤
│ SaaS Platform    │ Monthly subscription per cabin:       │
│                  │ Basic: $15/cabin/month (cloud dashboard)│
│                  │ Pro: $35/cabin/month (+analytics, API) │
│                  │ Enterprise: Custom pricing             │
├──────────────────┼──────────────────────────────────────┤
│ Advertising      │ Revenue share model:                  │
│                  │ Platform takes 30% of ad revenue       │
│                  │ CPM model: $5-15 per 1000 impressions │
│                  │ (verified by face detection)           │
├──────────────────┼──────────────────────────────────────┤
│ Integration      │ BMS integration fee: one-time setup   │
│                  │ Custom plugin development              │
│                  │ API access for 3rd party developers    │
├──────────────────┼──────────────────────────────────────┤
│ Data & Analytics │ Anonymized traffic data licensing     │
│                  │ (aggregated, GDPR compliant)           │
│                  │ Building intelligence reports          │
└──────────────────┴──────────────────────────────────────┘
```

### Unit Economics (per cabin, monthly)

| Item | Revenue | Cost | Margin |
|------|---------|------|--------|
| SaaS subscription | $35 | $5 (cloud hosting) | $30 |
| Advertising (avg) | $50 | $0 | $50 (×70% after rev share = $35) |
| **Monthly margin/cabin** | | | **~$65** |
| **Break-even** | | | **~4-6 months** (after hardware cost) |

### Pricing Strategy

| Package | Target | Hardware | Monthly | Features |
|---------|--------|----------|---------|----------|
| **Starter** | Small residential | Tier 1 (camera only) | $15/mo | Face recognition, access log, cloud dashboard |
| **Smart** | Luxury residential | Tier 2 (+ display) | $35/mo | + Personalized display, notifications, ads |
| **Premium** | Office/Commercial | Tier 3 (full kit) | $60/mo | + Sensors, voice, elevator control, analytics |
| **Enterprise** | Hospital/Large | Custom | Negotiable | + BMS integration, SLA, on-premise option |

---

## 9. Technical Principles

| Principle | Rationale |
|-----------|-----------|
| **Edge-first processing** | Privacy (face data stays on device), low latency (<500ms), works offline |
| **Plugin architecture** | Modular features, independent development, customer-configurable |
| **Event-driven communication** | Loose coupling, extensible, real-time responsiveness |
| **Web-based display** | Hardware agnostic, easy to update, no app store dependency |
| **MQTT for cloud sync** | Lightweight IoT protocol, offline buffer, pub/sub fits our event model |
| **Open standards** | REST API, WebSocket, MQTT, SQLite — no proprietary lock-in |
| **Fail-safe design** | Each module degrades gracefully: no display? still log. No cloud? still work locally |

---

## 10. Privacy & Compliance

| Concern | Mitigation |
|---------|-----------|
| Face data | Embeddings stored on-device only. Raw images never uploaded to cloud. Option to disable face storage. |
| GDPR/PDPA | Consent via building registration. Opt-out available. Data deletion on request. |
| Camera in cabin | Clear signage "AI Camera Active". Compliant with local surveillance laws. |
| Data retention | Configurable retention period. Auto-purge old logs. |
| Third-party access | API access requires auth. Anonymized data only for analytics. |
| Children | No face enrollment for minors. Age estimation (future) excludes children from targeting. |

---

## 11. Go-to-Market Strategy

### Phase 1: PoC / Pilot (Current → Q4 2026)

1. **Internal demo** — Full working prototype trên 1 cabin thật (DATGROUP office/building)
2. **Refine** — Thu thập feedback từ cư dân/BMS team, iterate
3. **Case study** — Document kết quả: adoption rate, engagement, issues

### Phase 2: Early Adopters (Q1 2027)

1. **Target**: 3-5 tòa chung cư cao cấp tại TP.HCM
2. **Model**: Free hardware installation + discounted SaaS 6 tháng
3. **Goal**: Validate product-market fit, collect testimonials

### Phase 3: Scale (Q3 2027+)

1. **Channel partners**: Đại lý thang máy, BMS integrators
2. **Self-service**: Web portal đặt hàng + cấu hình
3. **Content marketplace**: 3rd party content providers (tin tức, thời tiết, quảng cáo)

---

## 12. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Hardware theft/damage | High | Medium | Secure mounting, tamper detection, remote disable |
| Privacy backlash | High | Low | Transparent policy, opt-out, no cloud face data |
| Network instability | Medium | High | Offline-first design, local buffer, graceful degradation |
| Low adoption (cư dân không care) | High | Medium | Focus on utility (thông báo, bưu kiện) not just ads |
| Competition from elevator OEMs | Medium | Low | Open platform, price advantage, faster iteration |
| Regulatory changes | Medium | Low | Modular compliance (disable features per region) |
| Orange Pi EOL / supply chain | Low | Low | Abstract hardware layer, support multiple SBCs |

---

## 13. Success Metrics

### PoC Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Face recognition accuracy | >95% cho enrolled faces | Test với 20+ người, 100+ lượt |
| Display update latency | <500ms (face → content change) | End-to-end timing |
| System uptime | >99% (24/7 operation) | 30-day continuous run |
| Content engagement | Cư dân nhìn display >3s | Eye gaze estimation (future) hoặc survey |
| System cost (BOM) | <$300 cho Tier 2 | Actual procurement |
| Monthly operating cost | <$10/cabin (cloud) | Server bills |

### Scale Success Criteria (Post-pilot)

| Metric | Target |
|--------|--------|
| Net Promoter Score (cư dân) | >40 |
| Cabin deployments | 50+ trong 12 tháng |
| Monthly recurring revenue | $5000+ |
| Ad fill rate | >60% available slots filled |
| Hardware failure rate | <2%/year |

---

## Appendix: Glossary

| Term | Definition |
|------|-----------|
| Smart Cabin | Cabin thang máy được trang bị hệ thống AI/IoT |
| Edge Device | Thiết bị xử lý tại chỗ (Orange Pi) |
| BMS | Building Management System — hệ thống quản lý tòa nhà |
| PoC | Proof of Concept — prototype chứng minh khả thi |
| Dwell Time | Thời gian người ở trong cabin (30-60s trung bình) |
| Impression | Một lượt hiển thị quảng cáo được xác nhận (có người nhìn) |
| CPM | Cost Per Mille — giá per 1000 impressions |
| Plugin | Module tính năng có thể bật/tắt độc lập |
| ServicePlugin | Plugin event-driven (không cần video frames) |
| FramePlugin | Plugin xử lý video frames (camera-driven) |

---

*Document version: 1.0*
*Created: 2026-08-14*
*Author: DATGROUP — Smart Cabin Team*
