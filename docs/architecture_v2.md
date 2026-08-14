# Smart Cabin Platform — Architecture v2

> Kiến trúc mở rộng: từ Camera AI Box → Full Smart Cabin Platform (Display + Sensors + Audio + Relay)

---

## 1. Tổng quan thay đổi

### v1 (Hiện tại) — Camera AI Only

```
Camera RTSP → VideoPipeline → FramePlugin (Face Recognition) → EventBus → MQTT → Cloud
```

- Plugin system: chỉ hỗ trợ `BasePlugin` với `process_frame()` (frame-driven)
- Mọi plugin đều nhận video frames
- Không có abstraction cho display, sensors, audio, relay

### v2 (Mới) — Full Platform

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Smart Cabin Edge Platform                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌─────────┐  │
│  │ Camera  │  │ Display  │  │ Sensors  │  │ Audio │  │  Relay  │  │
│  │ (RTSP)  │  │ (Web/HDMI)│ │(I2C/GPIO)│  │ (TTS) │  │ (GPIO)  │  │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └───┬───┘  └────┬────┘  │
│       │             │             │             │            │        │
│       ▼             ▼             ▼             ▼            ▼        │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌─────────┐  │
│  │ Video   │  │ Display  │  │ Sensor   │  │ Audio │  │ Relay   │  │
│  │Pipeline │  │ Engine   │  │ Manager  │  │Manager│  │Controller│ │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └───┬───┘  └────┬────┘  │
│       │             │             │             │            │        │
│       ▼             ▼             ▼             ▼            ▼        │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌─────────┐  │
│  │Frame    │  │ Display  │  │ Sensor   │  │ Audio │  │Elevator │  │
│  │Plugins  │  │ Plugin   │  │ Plugin   │  │Plugin │  │ Plugin  │  │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └───┬───┘  └────┬────┘  │
│       │             │             │             │            │        │
│       └─────────────┴─────────────┴─────────────┴────────────┘        │
│                                   │                                    │
│                            ┌──────▼──────┐                            │
│                            │  Event Bus  │                            │
│                            └──────┬──────┘                            │
│                                   │                                    │
│                    ┌──────────────┼──────────────┐                    │
│                    ▼              ▼              ▼                     │
│              ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│              │Cloud Sync│  │Local Log │  │  REST    │                │
│              │  (MQTT)  │  │ (SQLite) │  │   API   │                │
│              └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### Thay đổi chính

| Thành phần | v1 | v2 |
|-----------|----|----|
| Plugin base class | `BasePlugin` (frame-driven only) | `FramePlugin` + `ServicePlugin` |
| Plugin communication | EventBus (face events only) | EventBus (full event taxonomy) |
| Hardware support | Camera only | Camera + Display + Sensors + Audio + Relay |
| Display | Không có | DisplayEngine + WebSocket + Content Management |
| Sensors | Không có | SensorManager (I2C, GPIO, UART) |
| Audio | Không có | AudioManager (TTS, notification sounds) |
| Relay | Không có | RelayController (GPIO, safety interlocks) |
| Config | camera + mqtt + plugins + logging | + display + sensors + audio + relay |

---

## 2. Plugin System v2

### 2.1 Plugin Type Hierarchy

```python
# edge/core/plugin_base.py (NEW — shared base)

class PluginType(str, Enum):
    FRAME = "frame"       # Camera-driven (receives video frames)
    SERVICE = "service"   # Event-driven (reacts to events, manages subsystem)


# edge/core/plugin_manager.py (UPDATED — BasePlugin remains for backward compat)

class BasePlugin(ABC):
    """Original frame-driven plugin. KEPT for backward compatibility."""
    # ... unchanged interface ...


# edge/core/service_plugin.py (NEW)

class ServicePlugin(ABC):
    """
    Event-driven plugin base class.
    
    Does NOT receive video frames. Instead:
    - Subscribes to EventBus events
    - Manages a subsystem (display, sensors, audio, relay)
    - Has its own lifecycle (start/stop) independent of video pipeline
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.SERVICE

    @property
    @abstractmethod
    def subscribed_events(self) -> list[EventType]:
        """Event types this plugin wants to receive."""
        ...

    @abstractmethod
    def start(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        """Initialize and start the service. Return True on success."""
        ...

    @abstractmethod
    def handle_event(self, event: BaseEvent) -> None:
        """Handle an event from the EventBus."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Shutdown the service, cleanup resources."""
        ...

    def health_check(self) -> bool:
        """Return True if service is healthy. Override for custom checks."""
        return True
```

### 2.2 Plugin Manager v2 — Unified Loading

```python
# Updated PluginManager logic (pseudocode)

class PluginManager:
    def _load_plugin(self, entry: PluginEntry) -> None:
        module = import_module(f"edge.plugins.{entry.name}.plugin")
        plugin_class = module.Plugin

        if issubclass(plugin_class, BasePlugin):
            # Frame plugin — register with video pipeline (existing behavior)
            self._load_frame_plugin(plugin_class, entry)

        elif issubclass(plugin_class, ServicePlugin):
            # Service plugin — subscribe to EventBus, no video frames
            self._load_service_plugin(plugin_class, entry)

    def _load_service_plugin(self, cls, entry):
        instance = cls()
        success = instance.start(entry.config, self._event_bus)
        if success:
            # Auto-subscribe to declared events
            for event_type in instance.subscribed_events:
                self._event_bus.subscribe(event_type, instance.handle_event)
            # Also subscribe wildcard if plugin declares "*"
```

### 2.3 Backward Compatibility

| Concern | Solution |
|---------|----------|
| Existing `face_recognition` plugin | Unchanged. Still subclasses `BasePlugin`, still receives frames. |
| Config format | `PluginEntry` unchanged. Plugin type auto-detected from class hierarchy. |
| Event schemas | New event types ADDED, existing ones unchanged. |
| Tests | All 222+ existing tests pass without modification. |

---

## 3. Display Subsystem

### 3.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Display Subsystem                           │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  EventBus ──→ DisplayPlugin (ServicePlugin)                   │
│                     │                                          │
│                     ├── PersonalizationEngine                  │
│                     │       │                                  │
│                     │       ├── Resolve person → content rules │
│                     │       ├── Priority: personal > scheduled │
│                     │       └── Multi-person handling          │
│                     │                                          │
│                     ├── ContentManager                         │
│                     │       │                                  │
│                     │       ├── Content CRUD (SQLite)          │
│                     │       ├── Content scheduling             │
│                     │       └── Zone layout management         │
│                     │                                          │
│                     └── DisplayEngine (abstract)               │
│                             │                                  │
│                             ├── WebDisplay (FastAPI+WebSocket) │
│                             ├── HDMIDisplay (framebuffer)      │
│                             └── RemoteDisplay (API push)       │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Display Engine Interface

```python
# edge/plugins/display/engine.py

class DisplayEngine(ABC):
    """Abstract display output backend."""

    @abstractmethod
    def start(self) -> bool:
        """Start the display backend (web server, framebuffer, etc.)"""
        ...

    @abstractmethod
    def push_content(self, zone: ContentZone, content: ContentItem) -> bool:
        """Push content to a specific zone on the display."""
        ...

    @abstractmethod
    def clear_zone(self, zone: ContentZone) -> bool:
        """Clear a zone (show empty/default)."""
        ...

    @abstractmethod
    def set_layout(self, layout: DisplayLayout) -> bool:
        """Change the display layout (zone arrangement)."""
        ...

    @abstractmethod
    def get_status(self) -> DisplayStatus:
        """Get current display status (connected clients, current content)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Shutdown display backend."""
        ...
```

### 3.3 Web Display Implementation

```python
# edge/plugins/display/web_display.py

class WebDisplay(DisplayEngine):
    """
    Web-based display backend.
    
    Runs a local FastAPI server with WebSocket for real-time content push.
    Any device with a browser can be a display client.
    
    Architecture:
        FastAPI (port 8081)
        ├── GET /              → Display HTML page
        ├── GET /status        → JSON status
        ├── WS  /ws/display    → Real-time content push
        └── Static files       → CSS, JS, images
    
    WebSocket Protocol:
        Server → Client:
            {"action": "update_zone", "zone": "main", "content": {...}}
            {"action": "clear_zone", "zone": "top_bar"}
            {"action": "set_layout", "layout": "standard_3zone"}
            {"action": "transition", "zone": "main", "effect": "fade", "duration_ms": 500}
        
        Client → Server:
            {"action": "ready"}           # Client loaded, ready for content
            {"action": "heartbeat"}       # Keep-alive
            {"action": "interaction", "type": "touch", "zone": "main"}  # Future: touch events
    """
```

### 3.4 Content Model

```python
# edge/plugins/display/models.py

class ContentType(str, Enum):
    GREETING = "greeting"           # "Xin chào anh Minh!"
    NOTIFICATION = "notification"   # "Bưu kiện chờ tại sảnh"
    ADVERTISEMENT = "advertisement" # Banner quảng cáo
    WEATHER = "weather"             # Widget thời tiết
    CLOCK = "clock"                 # Đồng hồ + ngày
    IMAGE = "image"                 # Static image
    VIDEO = "video"                 # Video loop
    HTML = "html"                   # Custom HTML content
    NEWS = "news"                   # News ticker

class ContentZone(str, Enum):
    TOP_BAR = "top_bar"             # Narrow bar: clock, weather, logo
    MAIN = "main"                   # Large area: greeting, ads, video
    SIDEBAR = "sidebar"            # Side panel: notifications, info
    BOTTOM_TICKER = "bottom_ticker" # Scrolling text: news, alerts

class ContentItem(BaseModel):
    id: str
    type: ContentType
    zone: ContentZone
    title: str = ""
    body: str = ""                  # Text content or HTML
    media_url: str = ""             # Path to image/video
    duration_seconds: int = 10      # How long to show (0 = forever)
    priority: int = 0               # Higher = more important
    style: dict = {}                # CSS overrides

class ContentRule(BaseModel):
    id: str
    person_id: str | None = None    # None = default/everyone
    zone: ContentZone
    content_id: str
    schedule: str = "*"             # Cron-like schedule ("* * 8-18 * *" = office hours)
    priority: int = 0
    active: bool = True

class DisplayLayout(BaseModel):
    name: str                       # "standard_3zone", "fullscreen", "split_2"
    zones: list[ZoneConfig]

class ZoneConfig(BaseModel):
    zone: ContentZone
    x: float                        # % position
    y: float
    width: float                    # % size
    height: float
    visible: bool = True
```

### 3.5 Personalization Flow

```
face.recognized event
       │
       ▼
PersonalizationEngine.resolve(person_id="0820", person_name="Sy")
       │
       ├── Query ContentRules WHERE person_id="0820" ORDER BY priority DESC
       │   → Rule: zone=MAIN, content="greeting_sy", priority=100
       │   → Rule: zone=SIDEBAR, content="notification_package", priority=50
       │
       ├── Query ContentRules WHERE person_id=NULL (default content)
       │   → Rule: zone=TOP_BAR, content="weather_widget", priority=10
       │   → Rule: zone=BOTTOM_TICKER, content="building_news", priority=10
       │
       ├── Merge: personal rules override default for same zone
       │
       └── Output: DisplayContent
               ├── TOP_BAR: weather_widget (default, no personal override)
               ├── MAIN: "Xin chào anh Sy! 🌤️ 32°C" (personal greeting)
               ├── SIDEBAR: "Bưu kiện #2847 chờ tại sảnh" (personal notification)
               └── BOTTOM_TICKER: "Bảo trì thang B ngày 20/8..." (default news)
```

### 3.6 Display State Machine

```
                    ┌──────────────┐
                    │  SCREENSAVER │ ◄──── Cabin trống > timeout
                    │  (default)   │
                    └──────┬───────┘
                           │ face detected
                           ▼
                    ┌──────────────┐
                    │  IDENTIFYING │ ◄──── Face detected, chờ recognition
                    │  (show logo) │
                    └──────┬───────┘
                           │ face recognized / unknown
                           ▼
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ PERSONALIZED │          │   DEFAULT    │
    │ (known face) │          │(unknown face)│
    └──────┬───────┘          └──────┬───────┘
           │                         │
           │ person exits            │ person exits
           ▼                         ▼
    ┌──────────────┐
    │  TRANSITION  │ ──→ Back to SCREENSAVER after timeout
    │  (fade out)  │
    └──────────────┘
```

---

## 4. Sensor Subsystem

### 4.1 Architecture

```python
# edge/core/sensor_manager.py (NEW)

class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    VIBRATION = "vibration"
    AIR_QUALITY = "air_quality"
    LOAD_CELL = "load_cell"         # Cabin weight
    DOOR_SENSOR = "door_sensor"     # Open/closed
    MOTION = "motion"               # PIR motion detect

class SensorReading(BaseModel):
    sensor_type: SensorType
    value: float
    unit: str                       # "°C", "%", "g", "kg", "ppm"
    timestamp: datetime
    metadata: dict = {}

class SensorManager:
    """
    Manages hardware sensors (I2C, GPIO, UART).
    
    Reads sensors periodically, publishes events to EventBus.
    Supports threshold alerts (e.g., temperature > 40°C → alarm).
    
    Polling model:
    - Fast sensors (vibration): 100Hz in dedicated thread
    - Slow sensors (temp, humidity): every 5-30s
    - Event sensors (door, motion): interrupt-based (GPIO callback)
    """

    def __init__(self, config: SensorConfig, event_bus: EventBus):
        self._sensors: dict[str, BaseSensor] = {}
        self._polling_thread: threading.Thread
        self._alerts: list[AlertRule] = []

    def register_sensor(self, sensor: BaseSensor) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_latest(self, sensor_type: SensorType) -> SensorReading | None: ...
```

### 4.2 Sensor Plugin

```python
# edge/plugins/sensors/plugin.py

class Plugin(ServicePlugin):
    """
    Sensor monitoring service plugin.
    
    - Reads configured sensors periodically
    - Publishes sensor.reading events to EventBus
    - Publishes sensor.alert events when thresholds exceeded
    - Reports to cloud via MQTT (aggregated, not every reading)
    """

    @property
    def name(self) -> str:
        return "sensors"

    @property
    def subscribed_events(self) -> list[EventType]:
        return []  # Sensors produce events, don't consume them

    def start(self, config, event_bus) -> bool:
        self._manager = SensorManager(config, event_bus)
        self._manager.start()
        return True
```

### 4.3 Sensor Events

```python
# Thêm vào shared/event_schemas.py

class EventType(str, Enum):
    # ... existing events ...
    
    # Sensor events (Phase 3)
    SENSOR_READING = "sensor.reading"
    SENSOR_ALERT = "sensor.alert"
    CABIN_WEIGHT_CHANGED = "sensor.weight_changed"
    CABIN_DOOR_OPENED = "sensor.door_opened"
    CABIN_DOOR_CLOSED = "sensor.door_closed"
    CABIN_MOTION_DETECTED = "sensor.motion_detected"
```

### 4.4 MQTT Topics — Sensors

```
cabin/{device_id}/sensor/temperature     → {"value": 28.5, "unit": "°C"}
cabin/{device_id}/sensor/humidity        → {"value": 65, "unit": "%"}
cabin/{device_id}/sensor/vibration       → {"rms": 0.02, "peak": 0.1, "unit": "g"}
cabin/{device_id}/sensor/air_quality     → {"co2_eq": 450, "tvoc": 12, "unit": "ppm"}
cabin/{device_id}/sensor/weight          → {"value": 320, "unit": "kg", "persons_est": 4}
cabin/{device_id}/sensor/alert           → {"sensor": "temperature", "value": 42, "threshold": 40}
```

---

## 5. Audio Subsystem

### 5.1 Architecture

```python
# edge/core/audio_manager.py (NEW)

class AudioManager:
    """
    Manages audio output: TTS greetings, notification sounds, alarms.
    
    Components:
    - TTS Engine: pyttsx3 (offline) or edge-tts (Microsoft, online)
    - Sound Player: pygame.mixer or aplay
    - Volume Control: ALSA mixer
    - Queue: Priority-based audio queue (alarm > greeting > notification)
    
    Design:
    - Non-blocking: audio plays in background thread
    - Priority queue: urgent sounds interrupt lower-priority
    - Cooldown: same greeting not repeated within 60s
    - Configurable: enable/disable per sound type
    """

    def speak(self, text: str, priority: int = 0) -> None: ...
    def play_sound(self, sound_name: str, priority: int = 0) -> None: ...
    def set_volume(self, volume: float) -> None: ...  # 0.0 - 1.0
    def stop(self) -> None: ...


class AudioPlugin(ServicePlugin):
    """
    Audio service plugin.
    
    Subscribes to:
    - face.recognized → "Xin chào {person_name}"
    - sensor.alert → alarm sound
    - system.error → error beep
    - display.notification → chime sound
    """

    @property
    def subscribed_events(self) -> list[EventType]:
        return [
            EventType.FACE_RECOGNIZED,
            EventType.SENSOR_ALERT,
            EventType.SYSTEM_ERROR,
        ]

    def handle_event(self, event: BaseEvent) -> None:
        if event.event_type == EventType.FACE_RECOGNIZED:
            name = event.person_name or "quý khách"
            self._audio.speak(f"Xin chào {name}")
```

### 5.2 TTS Options

| Engine | Offline | Quality | Latency | Vietnamese |
|--------|---------|---------|---------|-----------|
| pyttsx3 (espeak) | Yes | Low-Medium | <100ms | OK |
| edge-tts (Microsoft) | No (online) | High | 200-500ms | Excellent |
| Piper TTS | Yes | High | 100-300ms | Community models |
| gTTS | No (online) | Medium | 300-500ms | Good |

**Recommendation**: `edge-tts` cho chất lượng tốt nhất (Vietnamese), fallback `pyttsx3` khi offline.

---

## 6. Elevator Control Subsystem (Auto Floor Call)

### 6.1 Concept

Khi người dùng bước vào cabin thang máy:
1. Camera nhận diện khuôn mặt → xác định person_id
2. Lookup `default_floor` từ face database
3. Gửi lệnh chọn tầng qua **MQTT** đến bộ điều khiển thang máy
4. Hỗ trợ **nhiều người** cùng lúc → gọi nhiều tầng

### 6.2 Architecture

```python
# edge/plugins/elevator/plugin.py (NEW)

class ElevatorPlugin(ServicePlugin):
    """
    Auto Floor Call plugin.
    
    Flow:
        face.recognized → lookup default_floor → publish MQTT floor call
    
    Multi-person support:
        - Track all recognized persons in cabin (via face tracker)
        - Call floor for each person as they are identified
        - Dedup: same person → same floor, don't call twice
    
    Communication:
        - Floor call sent via MQTT (payload format configurable)
        - Topic: configured in plugin config (e.g., "elevator/floor_call")
        - Payload: JSON (format provided by elevator controller team)
    
    Safety:
        - Only calls if person has registered default_floor
        - Confidence must exceed elevator_threshold (0.75)
        - Cooldown per person per floor (no duplicate calls within 30s)
        - Logging: every call logged (person_id, floor, timestamp, confidence)
    """

    @property
    def name(self) -> str:
        return "elevator"

    @property
    def subscribed_events(self) -> list[EventType]:
        return [EventType.FACE_RECOGNIZED]

    def handle_event(self, event: BaseEvent) -> None:
        if event.event_type == EventType.FACE_RECOGNIZED:
            self._handle_floor_call(event)

    def _handle_floor_call(self, event: FaceRecognizedEvent) -> None:
        person_id = event.person_id
        confidence = event.confidence

        # Safety check: confidence threshold
        if confidence < self._elevator_threshold:
            return

        # Lookup floor from face database
        floor = self._get_person_floor(person_id)
        if floor is None:
            return  # Person has no floor registered

        # Cooldown check (same person, same floor)
        if self._is_in_cooldown(person_id, floor):
            return

        # Send floor call via MQTT
        self._publish_floor_call(person_id, floor, confidence)

        # Update cooldown
        self._set_cooldown(person_id, floor)

        # Publish internal event
        self._event_bus.publish(FloorRequestedEvent(
            source=self.name,
            person_id=person_id,
            floor=floor,
            confidence=confidence,
        ))

    def _get_person_floor(self, person_id: str) -> int | None:
        """Lookup person's default floor from face database."""
        # Query face database: SELECT default_floor FROM persons WHERE person_id = ?
        ...

    def _publish_floor_call(self, person_id: str, floor: int, confidence: float) -> None:
        """Publish floor call command via MQTT."""
        topic = self._config.get("mqtt_topic", "elevator/floor_call")
        # Payload format will be provided by elevator controller team
        payload = self._build_payload(person_id, floor, confidence)
        self._cloud_sync.publish_raw(topic, payload)
```

### 6.3 Face Database Schema Update

```sql
-- Thêm cột default_floor vào bảng persons (face database)
ALTER TABLE persons ADD COLUMN default_floor INTEGER DEFAULT NULL;
-- NULL = chưa đăng ký tầng → elevator plugin bỏ qua
```

```python
# Update: edge/plugins/face_recognition/database.py

class FaceDatabase:
    def add_person(self, person_id: str, name: str, embedding, default_floor: int | None = None): ...
    def update_floor(self, person_id: str, floor: int) -> bool: ...
    def get_person_floor(self, person_id: str) -> int | None: ...
```

### 6.4 Enrollment Update

```python
# Update: edge/tools/enroll_face.py — thêm option --floor

# Enrollment CLI:
#   python examples/run_recognition.py enroll --image face.jpg --name "Sy" --id 0820 --floor 8
#
# REST API (future):
#   POST /api/faces/enroll  body: {image, name, id, default_floor: 8}
```

### 6.5 Multi-Person Flow

```
Person A bước vào cabin (recognized → floor 8)
    → MQTT: floor_call {floor: 8, person_id: "0820"}

Person B bước vào cabin (recognized → floor 3)
    → MQTT: floor_call {floor: 3, person_id: "0681"}

Person C bước vào cabin (unknown → no floor registered)
    → Không gọi tầng, chỉ hiển thị default content

Kết quả: Thang máy nhận 2 lệnh gọi tầng 8 và tầng 3
Display hiện: "Tầng 8 (Sy), Tầng 3 (Ngọc Cần)"
```

### 6.6 Safety & Dedup Logic

```
Face Recognized
       │
       ├── Confidence > elevator_threshold (0.75)? ─── NO → Skip
       │
       ├── Person has default_floor? ─── NO → Skip (no floor registered)
       │
       ├── Cooldown expired? (>30s since last call for this person) ─── NO → Skip
       │
       ▼
Publish MQTT Floor Call
       │
       ├── Topic: elevator/floor_call (configurable)
       │
       ├── Payload: JSON (format TBD by elevator controller team)
       │
       ├── Log: {person_id, person_name, floor, timestamp, confidence}
       │
       └── EventBus: FLOOR_REQUESTED (for display plugin to show)
```

### 6.7 MQTT Floor Call

```
# Topic (configurable)
elevator/floor_call

# Payload format: TBD (sẽ được cung cấp bởi đội elevator controller)
# Placeholder structure:
{
    "device_id": "cabin-001",
    "person_id": "0820",
    "person_name": "Sy",
    "floor": 8,
    "confidence": 0.92,
    "timestamp": "2026-08-14T10:30:00Z"
}
```

### 6.8 Config

```yaml
# edge/config.yaml
- name: "elevator"
  enabled: true
  config:
    mqtt_topic: "elevator/floor_call"     # Topic gửi lệnh gọi tầng
    confidence_threshold: 0.75            # Ngưỡng confidence cao hơn display
    cooldown_seconds: 30                  # Không gọi trùng trong 30s
    # Payload format sẽ config sau khi có spec từ elevator controller team
```

### 6.9 Display Integration

Khi Elevator Plugin gọi tầng, nó cũng publish `FLOOR_REQUESTED` event → Display Plugin subscribe và hiển thị:
- Single person: "Xin chào anh Sy — Tầng 8"
- Multi-person: "Tầng 8 (Sy), Tầng 3 (Ngọc Cần)"

---

## 7. Extended Event Taxonomy

### 7.1 Full Event Types (v2)

```python
class EventType(str, Enum):
    # === Face Recognition (Phase 1, existing) ===
    FACE_DETECTED = "face.detected"
    FACE_RECOGNIZED = "face.recognized"
    FACE_UNKNOWN = "face.unknown"

    # === People Counter (Phase 3) ===
    PERSON_ENTERED = "person.entered"
    PERSON_EXITED = "person.exited"
    PERSON_COUNT_UPDATED = "person.count_updated"

    # === Display (Phase 2, NEW) ===
    DISPLAY_CONTENT_UPDATED = "display.content_updated"
    DISPLAY_ZONE_CLEARED = "display.zone_cleared"
    DISPLAY_CLIENT_CONNECTED = "display.client_connected"
    DISPLAY_CLIENT_DISCONNECTED = "display.client_disconnected"
    DISPLAY_INTERACTION = "display.interaction"          # Touch event (future)

    # === Sensors (Phase 3, NEW) ===
    SENSOR_READING = "sensor.reading"
    SENSOR_ALERT = "sensor.alert"
    CABIN_WEIGHT_CHANGED = "sensor.weight_changed"
    CABIN_DOOR_OPENED = "sensor.door_opened"
    CABIN_DOOR_CLOSED = "sensor.door_closed"
    CABIN_MOTION_DETECTED = "sensor.motion_detected"

    # === Audio (Phase 3, NEW) ===
    AUDIO_PLAYING = "audio.playing"
    AUDIO_FINISHED = "audio.finished"
    AUDIO_ERROR = "audio.error"

    # === Elevator Control / Auto Floor Call (Phase 2, NEW) ===
    FLOOR_REQUESTED = "elevator.floor_requested"
    FLOOR_ARRIVED = "elevator.floor_arrived"       # Future: feedback from controller
    ELEVATOR_ERROR = "elevator.error"

    # === System (existing + extended) ===
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    PLUGIN_LOADED = "system.plugin_loaded"
    PLUGIN_ERROR = "system.plugin_error"
    PLUGIN_HEALTH_CHECK = "system.plugin_health"        # NEW
    CONFIG_UPDATED = "system.config_updated"            # NEW
```

### 7.2 Event Flow Diagram (Phase 2 — Display)

```
Camera ──→ Face Recognition Plugin
                    │
                    ├── face.recognized {person_id: "0820", person_name: "Sy", confidence: 0.92}
                    │
                    ▼
           Display Plugin (ServicePlugin)
                    │
                    ├── PersonalizationEngine.resolve("0820")
                    │         │
                    │         ▼
                    │   ContentManager.get_content(person_id="0820")
                    │         │
                    │         ▼
                    ├── DisplayEngine.push_content(MAIN, greeting_content)
                    ├── DisplayEngine.push_content(SIDEBAR, notification_content)
                    │
                    ├── Publish: display.content_updated {zone: "main", person_id: "0820"}
                    │
                    ▼
           WebSocket ──→ Browser Display Client
                              │
                              └── Renders: "Xin chào anh Sy! 🌤️ 32°C"
```

---

## 8. Extended MQTT Topics (v2)

### 8.1 Topic Hierarchy

```
cabin/{device_id}/
├── face/
│   ├── recognized          # Edge → Cloud: face matched
│   ├── unknown             # Edge → Cloud: unknown face
│   └── enrolled            # Edge → Cloud: new face enrolled
├── display/
│   ├── status              # Edge → Cloud: current display state
│   ├── content/push        # Cloud → Edge: push new content
│   ├── content/update      # Cloud → Edge: update existing content
│   ├── content/delete      # Cloud → Edge: remove content
│   ├── rules/push          # Cloud → Edge: push personalization rules
│   └── schedule/update     # Cloud → Edge: update content schedule
├── sensor/
│   ├── temperature         # Edge → Cloud: periodic readings
│   ├── humidity            # Edge → Cloud
│   ├── vibration           # Edge → Cloud
│   ├── air_quality         # Edge → Cloud
│   ├── weight              # Edge → Cloud: cabin load
│   └── alert               # Edge → Cloud: threshold exceeded
├── audio/
│   ├── status              # Edge → Cloud: playing/idle
│   └── command/speak       # Cloud → Edge: TTS command
├── elevator/
│   ├── floor_call              # Edge → Elevator Controller: auto floor call (MQTT)
│   ├── floor_requested         # Edge → Cloud: log floor request
│   ├── status                  # Elevator Controller → Edge: current state (future)
│   └── command/activate        # Cloud → Edge: remote floor call (manual)
├── status/
│   └── heartbeat           # Edge → Cloud: system stats (every 30s)
├── system/
│   ├── start               # Edge → Cloud: system booted
│   ├── stop                # Edge → Cloud: system shutdown
│   └── error               # Edge → Cloud: error occurred
└── command/
    ├── restart_plugin      # Cloud → Edge: restart a plugin
    ├── update_config       # Cloud → Edge: config update
    ├── sync_faces          # Cloud → Edge: sync face database
    └── reboot              # Cloud → Edge: reboot device
```

### 8.2 Cloud → Edge Content Push (Display)

```json
// Topic: cabin/cabin-001/display/content/push
{
    "action": "add",
    "content": {
        "id": "ad_gym_summer_2026",
        "type": "advertisement",
        "zone": "main",
        "title": "Summer Fitness Deal",
        "body": "<div class='ad-banner'>...</div>",
        "media_url": "https://cdn.example.com/gym_banner.jpg",
        "duration_seconds": 15,
        "priority": 20,
        "schedule": "* * 6-22 * *",
        "target": {
            "person_ids": null,
            "demographics": {"age_range": "25-45"},
            "floors": [3, 5, 7]
        }
    }
}
```

### 8.3 Edge → Cloud Display Status

```json
// Topic: cabin/cabin-001/display/status
{
    "timestamp": "2026-08-14T10:30:00Z",
    "state": "personalized",
    "current_person": "0820",
    "zones": {
        "top_bar": {"content_id": "weather_widget", "since": "2026-08-14T10:29:55Z"},
        "main": {"content_id": "greeting_sy", "since": "2026-08-14T10:29:58Z"},
        "sidebar": {"content_id": "notification_package", "since": "2026-08-14T10:29:58Z"},
        "bottom_ticker": {"content_id": "building_news", "since": "2026-08-14T10:00:00Z"}
    },
    "clients_connected": 1,
    "uptime_seconds": 86400
}
```

---

## 9. Configuration v2

### 9.1 Extended config.yaml

```yaml
# edge/config.yaml (v2)

camera:
  url: "rtsp://192.168.1.100:554/stream1"
  capture_fps: 25
  process_fps: 10
  reconnect_interval: 5.0
  connection_timeout: 10.0

mqtt:
  broker_host: "localhost"
  broker_port: 1883
  device_id: "cabin-001"
  topic_publish: "embody/w"
  topic_subscribe: "embody/r"

plugins:
  directory: "edge/plugins"
  modules:
    # --- Frame Plugins (camera-driven) ---
    - name: "face_recognition"
      enabled: true
      config:
        detection_threshold: 0.7
        embedding_threshold: 0.4
        min_face_size: 60
        tracker_iou_threshold: 0.4
        tracker_max_lost: 15
        database_path: "data/db/faces.db"
        snapshot_enabled: true

    # --- Service Plugins (event-driven) ---
    - name: "display"
      enabled: true
      config:
        port: 8081
        layout: "standard_3zone"
        default_greeting: "Xin chào"
        screensaver_timeout: 30
        transition_effect: "fade"
        transition_duration_ms: 500
        content_db_path: "data/db/display.db"
        media_dir: "data/display/media"

    - name: "audio"
      enabled: false
      config:
        engine: "edge-tts"              # "edge-tts", "pyttsx3", "piper"
        voice: "vi-VN-HoaiMyNeural"     # Vietnamese female voice
        volume: 0.7
        greeting_template: "Xin chào {name}"
        cooldown_seconds: 60
        fallback_engine: "pyttsx3"      # Offline fallback

    - name: "sensors"
      enabled: false
      config:
        poll_interval: 10               # Seconds between readings
        sensors:
          - type: "temperature"
            bus: "i2c"
            address: 0x44               # SHT30
            alert_threshold: 40
          - type: "vibration"
            bus: "i2c"
            address: 0x53               # ADXL345
            alert_threshold: 0.5        # g-force
          - type: "air_quality"
            bus: "i2c"
            address: 0x58               # SGP30
            alert_threshold: 1000       # CO2 eq ppm

    - name: "elevator"
      enabled: false
      config:
        mode: "mqtt"                    # "mqtt" (primary), "relay", "serial", "api"
        mqtt_topic: "elevator/floor_call"  # Topic gửi lệnh chọn tầng
        confidence_threshold: 0.75      # Higher than face recognition display
        cooldown_seconds: 30            # Dedup: same person, same floor
        # Floor mapping: person_id → default floor
        # (Also stored in face database, this is override/fallback)
        floor_mapping:
          "0820": 8                     # Sy → tầng 8
          "0681": 6                     # Ngọc Cần → tầng 6

# --- Display (global settings) ---
display:
  enabled: true
  backend: "web"                        # "web", "hdmi", "remote"
  host: "0.0.0.0"
  port: 8081
  layout: "standard_3zone"

# --- Logging ---
logging:
  level: "INFO"
```

### 9.2 Config Models (Updated)

```python
# Additions to edge/core/config.py

class DisplayConfig(BaseModel):
    """Display subsystem configuration."""
    enabled: bool = False
    backend: str = "web"            # "web", "hdmi", "remote"
    host: str = "0.0.0.0"
    port: int = 8081
    layout: str = "standard_3zone"

class SensorEntry(BaseModel):
    type: str
    bus: str = "i2c"
    address: int = 0
    alert_threshold: float = 0

class SensorsConfig(BaseModel):
    enabled: bool = False
    poll_interval: int = 10
    sensors: list[SensorEntry] = []

class AudioConfig(BaseModel):
    enabled: bool = False
    engine: str = "edge-tts"
    voice: str = "vi-VN-HoaiMyNeural"
    volume: float = 0.7

class ElevatorConfig(BaseModel):
    enabled: bool = False
    mode: str = "mqtt"              # "mqtt", "relay", "serial", "api"
    mqtt_topic: str = "elevator/floor_call"
    confidence_threshold: float = 0.75
    cooldown_seconds: int = 30

class SmartCabinConfig(BaseModel):
    """Root config — v2 (backward compatible)."""
    camera: CameraConfig
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # New in v2 (optional, all disabled by default)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    sensors: SensorsConfig = Field(default_factory=SensorsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    relay: ElevatorConfig = Field(default_factory=ElevatorConfig)
```

---

## 10. Project Structure v2

```
smart-cabin/
├── edge/
│   ├── core/
│   │   ├── config.py                 # Updated: + DisplayConfig, SensorsConfig, etc.
│   │   ├── video_pipeline.py         # Unchanged
│   │   ├── event_bus.py              # Unchanged (generic enough)
│   │   ├── plugin_manager.py         # Updated: support ServicePlugin loading
│   │   ├── service_plugin.py         # NEW: ServicePlugin abstract base
│   │   ├── cloud_sync.py             # Updated: forward new event types
│   │   ├── logging_setup.py          # Unchanged
│   │   ├── sensor_manager.py         # NEW: hardware sensor abstraction
│   │   ├── audio_manager.py          # NEW: TTS + sound playback
│   │   └── relay_controller.py       # NEW: GPIO relay control
│   │
│   ├── plugins/
│   │   ├── face_recognition/         # Unchanged (FramePlugin)
│   │   │   ├── plugin.py
│   │   │   ├── detector.py
│   │   │   ├── embedder.py
│   │   │   ├── tracker.py
│   │   │   ├── database.py
│   │   │   └── alignment.py
│   │   │
│   │   ├── display/                  # NEW (ServicePlugin)
│   │   │   ├── plugin.py            # DisplayPlugin(ServicePlugin)
│   │   │   ├── engine.py            # DisplayEngine abstract
│   │   │   ├── web_display.py       # WebDisplay implementation
│   │   │   ├── content_store.py     # SQLite content storage
│   │   │   ├── personalization.py   # Content resolution logic
│   │   │   ├── models.py            # ContentItem, ContentZone, etc.
│   │   │   ├── api.py               # REST API for content management
│   │   │   └── static/              # HTML/CSS/JS display client
│   │   │       ├── index.html
│   │   │       ├── display.js
│   │   │       └── display.css
│   │   │
│   │   ├── audio/                    # NEW (ServicePlugin, Phase 3)
│   │   │   ├── plugin.py
│   │   │   └── tts.py
│   │   │
│   │   ├── sensors/                  # NEW (ServicePlugin, Phase 3)
│   │   │   ├── plugin.py
│   │   │   └── drivers/
│   │   │       ├── dht22.py
│   │   │       ├── adxl345.py
│   │   │       └── sgp30.py
│   │   │
│   │   ├── elevator/                 # NEW (ServicePlugin, Phase 4)
│   │   │   └── plugin.py
│   │   │
│   │   └── dummy/                    # Unchanged
│   │       └── plugin.py
│   │
│   ├── api/                          # Edge REST API (Task 11)
│   │   ├── main.py
│   │   └── routers/
│   │       ├── faces.py
│   │       ├── display.py            # NEW
│   │       ├── sensors.py            # NEW
│   │       └── status.py
│   │
│   ├── tools/                        # Unchanged
│   ├── inference/                    # Unchanged
│   ├── tests/
│   │   ├── test_service_plugin.py    # NEW
│   │   ├── test_display_plugin.py    # NEW
│   │   ├── test_content_store.py     # NEW
│   │   ├── test_personalization.py   # NEW
│   │   └── ... (existing tests)
│   └── config.yaml
│
├── shared/
│   ├── event_schemas.py              # Updated: new EventTypes + event models
│   └── mqtt_topics.py                # Updated: new topic definitions
│
├── data/
│   ├── db/
│   │   ├── faces.db                  # Existing
│   │   ├── display.db                # NEW: content storage
│   │   └── mqtt_buffer.db            # Existing
│   ├── display/                      # NEW
│   │   └── media/                    # Uploaded images/videos for display
│   ├── faces/                        # Existing
│   └── snapshots/                    # Existing
│
├── docs/
│   ├── implementation_plan.md        # Existing (Phase 1)
│   ├── vision.md                     # NEW
│   ├── architecture_v2.md            # NEW (this document)
│   ├── display_module_plan.md        # NEW
│   └── api_spec.md                   # Future (Task 12)
│
└── deploy/
    ├── setup_edge.sh
    ├── mosquitto.conf
    └── smart-cabin.service
```

---

## 11. Migration Path (v1 → v2)

### Step 1: Add ServicePlugin (Non-breaking)

```
- Thêm edge/core/service_plugin.py
- Update PluginManager: detect plugin type, load accordingly
- Không thay đổi BasePlugin interface
- Tất cả existing tests pass
```

### Step 2: Add Display Plugin (Additive)

```
- Tạo edge/plugins/display/ directory
- Add "display" entry to config.yaml (enabled: true)
- DisplayPlugin subscribes to face events via EventBus
- Zero changes to face_recognition plugin
```

### Step 3: Extend Event Schema (Additive)

```
- Thêm EventType values (new enum members)
- Thêm event models (new Pydantic classes)
- Existing events unchanged
- CloudSync: add new events to bridge (opt-in)
```

### Step 4: Add Sensor/Audio/Relay (Phase 3-4, Additive)

```
- Mỗi subsystem là một ServicePlugin riêng
- Enable/disable via config
- Không ảnh hưởng existing modules
```

### Zero Breaking Changes Guarantee

| What | Why it's safe |
|------|---------------|
| BasePlugin interface | Untouched. face_recognition vẫn dùng nó. |
| EventBus | Generic. Accepts any BaseEvent subclass. New events just work. |
| Config loading | New sections optional (default_factory). Old configs valid. |
| MQTT topics | Additive. Existing topics unchanged. |
| Plugin loading | Auto-detect type from class hierarchy. Old plugins = FramePlugin. |

---

## 12. Security Considerations

### 12.1 Display Security

| Threat | Mitigation |
|--------|-----------|
| XSS in content | Sanitize HTML content before rendering. CSP headers on web display. |
| Unauthorized content push | MQTT auth required. Content validation before display. |
| Display hijacking | WebSocket auth token. Display client on local network only. |
| Inappropriate content | Content moderation rules. Admin approval for user-uploaded content. |

### 12.2 Relay/Elevator Security

| Threat | Mitigation |
|--------|-----------|
| Unauthorized floor access | High confidence threshold (0.75). Person must have floor mapping. |
| Replay attack | Timestamp validation. Fresh face detection required. |
| Relay stuck ON | Hardware timeout (5s max). Watchdog timer. |
| GPIO manipulation | Physical enclosure. Tamper detection sensor. |

### 12.3 Network Security

| Threat | Mitigation |
|--------|-----------|
| MQTT sniffing | TLS encryption. Username/password auth. |
| REST API abuse | API key/token auth. Rate limiting. |
| Local network attacks | Display on isolated VLAN. Firewall rules. |

---

## 13. Performance Budget (Orange Pi 4 Pro)

| Resource | Budget | Current Usage | After v2 (Display) |
|----------|--------|---------------|---------------------|
| CPU (6 cores) | 100% | ~25% (camera + face) | ~35% (+display server) |
| RAM (4GB) | 4096 MB | ~512 MB | ~700 MB (+content cache) |
| Storage | 128GB USB | ~2 GB (models + data) | ~5 GB (+media content) |
| Network (local) | 100 Mbps | ~2 Mbps (RTSP) | ~5 Mbps (+display WebSocket) |
| GPU (Mali-T860) | Available | Unused | Unused (CPU sufficient) |

### Per-Subsystem CPU Estimate

| Subsystem | CPU Usage | Notes |
|-----------|-----------|-------|
| Video Pipeline | 5-8% | RTSP decode + distribute |
| Face Recognition | 15-20% | Detection + embedding (~200ms/frame at 5fps) |
| Display Server | 3-5% | FastAPI + WebSocket (mostly idle, spike on content push) |
| Sensor Polling | 1-2% | I2C reads every 10s |
| Audio (TTS) | 5-10% (burst) | Only when speaking, idle otherwise |
| MQTT + EventBus | 1-2% | Lightweight messaging |
| **Total** | **~35-45%** | **Comfortable headroom** |

---

## 14. Sequence Diagrams

### 14.1 Face → Personalized Display (Main Flow)

```
┌──────┐     ┌─────────┐     ┌────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│Camera│     │FaceReco │     │EventBus│     │DisplayPl│     │WebDispl │     │ Browser │
└──┬───┘     └────┬────┘     └───┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
   │              │              │              │              │              │
   │─frame──────→│              │              │              │              │
   │              │─detect+track │              │              │              │
   │              │─embed+match  │              │              │              │
   │              │              │              │              │              │
   │              │──publish─────→│              │              │              │
   │              │  face.recognized             │              │              │
   │              │              │──dispatch─────→│              │              │
   │              │              │              │              │              │
   │              │              │              │─resolve───→  │              │
   │              │              │              │  person      │              │
   │              │              │              │←─content──── │              │
   │              │              │              │              │              │
   │              │              │              │─push_content─→│              │
   │              │              │              │              │─WebSocket───→│
   │              │              │              │              │ {zone, data} │
   │              │              │              │              │              │─render
   │              │              │              │              │              │
   │              │              │              │──publish─────→│              │
   │              │              │              │ display.      │              │
   │              │              │              │ content_      │              │
   │              │              │              │ updated       │              │
```

### 14.2 Cloud Content Push → Display

```
┌─────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐
│Cloud│     │CloudSync │     │ContentMgr│     │DisplayPl│     │ Browser │
└──┬──┘     └─────┬────┘     └─────┬────┘     └────┬────┘     └────┬────┘
   │              │              │              │              │
   │─MQTT push───→│              │              │              │
   │ content/push │              │              │              │
   │              │─parse+validate│              │              │
   │              │──────────────→│              │              │
   │              │              │─store content │              │
   │              │              │─check if active│             │
   │              │              │──────────────→│              │
   │              │              │ content_ready │              │
   │              │              │              │─push_content─→│
   │              │              │              │              │─render
   │              │              │              │              │
```

---

## 15. Decision Records

### ADR-001: ServicePlugin vs extending BasePlugin

**Decision**: Create separate `ServicePlugin` class instead of making `process_frame()` optional in `BasePlugin`.

**Rationale**:
- Clear separation of concerns (frame-driven vs event-driven)
- No confusion about when process_frame is called or not
- Each type has appropriate lifecycle methods
- Existing BasePlugin users (face_recognition) unaffected
- Type checking catches misuse at load time

### ADR-002: Web-based Display (not HDMI/Android)

**Decision**: Primary display backend is web-based (FastAPI + WebSocket + Browser).

**Rationale**:
- Hardware agnostic: any device with browser works (tablet, TV, laptop for PoC)
- Easy to develop and iterate (HTML/CSS/JS, hot-reload friendly)
- Multiple displays from one server (multi-screen support free)
- Remote preview for debugging
- Can add HDMI backend later (same DisplayEngine interface)

### ADR-003: Content stored on Edge (not cloud-only)

**Decision**: Content stored locally on edge device (SQLite + file system), synced from cloud.

**Rationale**:
- Offline-first: display works without cloud connection
- Low latency: no network round-trip for content resolution
- Privacy: personal content rules stored locally
- Cloud is source of truth, edge is cache + runtime

### ADR-004: Personalization per-person (not per-demographic)

**Decision**: Phase 2 personalizes by `person_id` (exact match from face recognition). Demographic-based targeting (age, gender) deferred to Phase 3+.

**Rationale**:
- person_id matching is already working (face recognition done)
- Demographic estimation requires additional AI models (not built yet)
- Simpler to implement and validate in PoC
- Still valuable: personal greetings, personal notifications

---

*Document version: 1.0*
*Created: 2026-08-14*
*Author: DATGROUP — Smart Cabin Team*
