# Implementation Plan — Personalized Display Module

> Module hiển thị cá nhân hóa cho Smart Cabin: nhận diện cư dân → hiển thị nội dung riêng (chào hỏi, thông báo, quảng cáo targeted).

---

## Problem Statement

Xây dựng module Display cho Smart Cabin Platform, biến cabin thang máy thành điểm chạm cá nhân hóa. Khi camera nhận diện được cư dân, display tự động hiển thị nội dung phù hợp (chào hỏi, thông báo cá nhân, quảng cáo targeted). Khi không nhận diện ai, display hiển thị default content (do khách hàng config).

## Requirements

- **Display backend**: Web-based (FastAPI + WebSocket), bất kỳ device có browser đều là display client
- **Content zones**: Layout chia vùng (top_bar, main, sidebar, bottom_ticker)
- **Personalization**: person_id → content rules (greeting + notification + ads)
- **Default content**: Khi cabin trống hoặc unknown → content do khách hàng config
- **Multi-person**: Khi nhiều người trong cabin → neutral/common content
- **Real-time**: Face recognized → display update < 500ms
- **Offline**: Display vẫn hoạt động khi mất cloud (content cached locally)
- **Plugin type**: `ServicePlugin` (event-driven, không cần video frames)
- **Backward compatible**: Không ảnh hưởng face_recognition plugin hiện có

## Dependencies

- Face Recognition plugin (Phase 1, done) — provides `face.recognized` events
- EventBus (done) — inter-plugin communication
- CloudSync MQTT (done) — content sync from cloud
- Python 3.12+, FastAPI, uvicorn, websockets

## Architecture Reference

Xem `docs/architecture_v2.md` sections:
- Section 2: Plugin System v2 (ServicePlugin)
- Section 3: Display Subsystem
- Section 7: Extended Event Taxonomy
- Section 8: Extended MQTT Topics
- Section 14: Sequence Diagrams

---

## Task Breakdown

### Task 1: ServicePlugin Base Class & Plugin Manager Update

**Objective**: Tạo `ServicePlugin` abstract base class và update `PluginManager` để support cả FramePlugin và ServicePlugin.

**Implementation guidance**:

**File: `edge/core/service_plugin.py` (NEW)**
- Abstract base class `ServicePlugin`:
  ```python
  class ServicePlugin(ABC):
      @property
      @abstractmethod
      def name(self) -> str: ...

      @property
      def version(self) -> str:
          return "0.1.0"

      @property
      @abstractmethod
      def subscribed_events(self) -> list[EventType]:
          """Event types this plugin subscribes to."""
          ...

      @abstractmethod
      def start(self, config: dict[str, Any], event_bus: EventBus) -> bool:
          """Initialize and start service. Return True on success."""
          ...

      @abstractmethod
      def handle_event(self, event: BaseEvent) -> None:
          """Process an event from EventBus."""
          ...

      @abstractmethod
      def stop(self) -> None:
          """Shutdown service, cleanup resources."""
          ...

      def health_check(self) -> bool:
          """Override for custom health monitoring."""
          return True
  ```

**File: `edge/core/plugin_manager.py` (UPDATE)**
- Import `ServicePlugin`
- `_load_plugin()`: detect if plugin class inherits `BasePlugin` or `ServicePlugin`
- `_load_service_plugin(cls, entry)`:
  - Instantiate plugin
  - Call `plugin.start(config, event_bus)`
  - Auto-subscribe `plugin.handle_event` to each event in `plugin.subscribed_events`
  - Track in `self._service_plugins: dict[str, ServicePluginWrapper]`
- `ServicePluginWrapper`: state tracking (STARTED, RUNNING, STOPPED, ERROR)
- `stop_all()`: also stops service plugins
- `get_all_plugins()`: unified status for both types

**File: `edge/core/plugin_manager.py` — `ServicePluginWrapper` class**
```python
class ServicePluginWrapper:
    def __init__(self, plugin: ServicePlugin, config: dict[str, Any]):
        self.plugin = plugin
        self.config = config
        self.state = PluginState.UNLOADED
        self.error_message: str = ""
        self.started_at: float = 0.0
        self.events_received: int = 0
        self.events_errors: int = 0
```

**Test requirements** (`edge/tests/test_service_plugin.py`):
- Test ServicePlugin abstract interface (cannot instantiate directly)
- Test PluginManager loads ServicePlugin correctly
- Test PluginManager loads FramePlugin unchanged (backward compat)
- Test mixed loading: FramePlugin + ServicePlugin from same config
- Test ServicePlugin receives events via EventBus after start
- Test ServicePlugin.stop() unsubscribes from EventBus
- Test ServicePlugin health_check reporting
- Test ServicePlugin error isolation (one crash doesn't affect others)
- Test PluginManager unified status (both types in get_all_plugins)

**Demo**: Create `edge/plugins/echo/plugin.py` — minimal ServicePlugin that logs all received events. Load via config, verify events from face_recognition arrive.

---

### Task 2: Content Data Model & Storage

**Objective**: Define content types, zones, rules, và SQLite storage cho display content.

**Implementation guidance**:

**File: `edge/plugins/display/models.py` (NEW)**
```python
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class ContentType(str, Enum):
    GREETING = "greeting"
    NOTIFICATION = "notification"
    ADVERTISEMENT = "advertisement"
    WEATHER = "weather"
    CLOCK = "clock"
    IMAGE = "image"
    VIDEO = "video"
    HTML = "html"
    NEWS = "news"

class ContentZone(str, Enum):
    TOP_BAR = "top_bar"
    MAIN = "main"
    SIDEBAR = "sidebar"
    BOTTOM_TICKER = "bottom_ticker"

class ContentItem(BaseModel):
    id: str
    type: ContentType
    zone: ContentZone
    title: str = ""
    body: str = ""                          # Text or HTML content
    media_url: str = ""                     # Relative path to media file
    duration_seconds: int = 10              # Display duration (0 = indefinite)
    priority: int = 0                       # Higher = more important
    style: dict = Field(default_factory=dict)  # CSS overrides
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ContentRule(BaseModel):
    id: str
    person_id: str | None = None            # None = default (applies to everyone)
    zone: ContentZone
    content_id: str                         # Reference to ContentItem.id
    schedule: str = "*"                     # Cron-like: "* * 8-18 * *" = 8AM-6PM
    priority: int = 0                       # Rule priority (higher wins)
    active: bool = True

class DisplayLayout(BaseModel):
    name: str                               # "standard_3zone", "fullscreen", "split_2"
    zones: list["ZoneConfig"]

class ZoneConfig(BaseModel):
    zone: ContentZone
    x_percent: float                        # Position X (0-100)
    y_percent: float                        # Position Y (0-100)
    width_percent: float                    # Width (0-100)
    height_percent: float                   # Height (0-100)
    visible: bool = True
    background: str = ""                    # CSS background

# Predefined layouts
LAYOUTS = {
    "standard_3zone": DisplayLayout(
        name="standard_3zone",
        zones=[
            ZoneConfig(zone=ContentZone.TOP_BAR, x_percent=0, y_percent=0, width_percent=100, height_percent=10),
            ZoneConfig(zone=ContentZone.MAIN, x_percent=0, y_percent=10, width_percent=70, height_percent=80),
            ZoneConfig(zone=ContentZone.SIDEBAR, x_percent=70, y_percent=10, width_percent=30, height_percent=80),
            ZoneConfig(zone=ContentZone.BOTTOM_TICKER, x_percent=0, y_percent=90, width_percent=100, height_percent=10),
        ],
    ),
    "fullscreen": DisplayLayout(
        name="fullscreen",
        zones=[
            ZoneConfig(zone=ContentZone.MAIN, x_percent=0, y_percent=0, width_percent=100, height_percent=100),
        ],
    ),
    "split_2": DisplayLayout(
        name="split_2",
        zones=[
            ZoneConfig(zone=ContentZone.TOP_BAR, x_percent=0, y_percent=0, width_percent=100, height_percent=8),
            ZoneConfig(zone=ContentZone.MAIN, x_percent=0, y_percent=8, width_percent=50, height_percent=84),
            ZoneConfig(zone=ContentZone.SIDEBAR, x_percent=50, y_percent=8, width_percent=50, height_percent=84),
            ZoneConfig(zone=ContentZone.BOTTOM_TICKER, x_percent=0, y_percent=92, width_percent=100, height_percent=8),
        ],
    ),
}
```

**File: `edge/plugins/display/content_store.py` (NEW)**
- `ContentStore` class wrapping SQLite database
- Tables:
  - `content_items`: id, type, zone, title, body, media_url, duration_seconds, priority, style_json, active, created_at, updated_at
  - `content_rules`: id, person_id, zone, content_id, schedule, priority, active
- Methods:
  ```python
  class ContentStore:
      def __init__(self, db_path: str = "data/db/display.db"): ...
      def initialize(self) -> bool: ...

      # Content CRUD
      def add_content(self, item: ContentItem) -> bool: ...
      def get_content(self, content_id: str) -> ContentItem | None: ...
      def list_content(self, zone: ContentZone | None = None, content_type: ContentType | None = None) -> list[ContentItem]: ...
      def update_content(self, content_id: str, updates: dict) -> bool: ...
      def delete_content(self, content_id: str) -> bool: ...

      # Rules CRUD
      def add_rule(self, rule: ContentRule) -> bool: ...
      def get_rules_for_person(self, person_id: str) -> list[ContentRule]: ...
      def get_default_rules(self) -> list[ContentRule]: ...
      def get_rules_for_zone(self, zone: ContentZone, person_id: str | None = None) -> list[ContentRule]: ...
      def update_rule(self, rule_id: str, updates: dict) -> bool: ...
      def delete_rule(self, rule_id: str) -> bool: ...

      # Query (used by PersonalizationEngine)
      def resolve_content_for_person(self, person_id: str | None, zone: ContentZone) -> ContentItem | None: ...

      def close(self) -> None: ...
  ```
- File storage: media files in `data/display/media/` (images, videos)
- `resolve_content_for_person` logic:
  1. Query rules WHERE person_id = given AND zone = given AND active = True
  2. If no person-specific rules, query WHERE person_id IS NULL (default)
  3. Filter by schedule (check if current time matches cron pattern)
  4. Sort by priority DESC
  5. Return first match's ContentItem

**Test requirements** (`edge/tests/test_content_store.py`):
- Test database initialization (tables created)
- Test add/get/update/delete content items
- Test add/get/update/delete rules
- Test list_content with filters (zone, type)
- Test resolve_content_for_person (person-specific rule)
- Test resolve_content_for_person (fallback to default when no personal rule)
- Test resolve_content_for_person (priority ordering)
- Test schedule filtering (time-based rules)
- Test content with media_url (file existence check)
- Test concurrent access (thread safety)
- Test database close and reopen

**Demo**: Seed database với sample content (greeting cho person "0820", default ads), query → verify correct prioritized results.

---

### Task 3: Personalization Engine

**Objective**: Logic engine quyết định nội dung nào hiển thị cho ai, khi nào, xử lý transitions.

**Implementation guidance**:

**File: `edge/plugins/display/personalization.py` (NEW)**
```python
class CabinState(str, Enum):
    EMPTY = "empty"                 # Không ai trong cabin
    IDENTIFYING = "identifying"     # Face detected, chờ recognition
    PERSONALIZED = "personalized"   # Known person → personal content
    DEFAULT = "default"             # Unknown person → default content
    MULTI_PERSON = "multi_person"   # Multiple people → neutral content

class DisplayContent(BaseModel):
    """Resolved content for all zones at a given moment."""
    state: CabinState
    person_id: str | None = None
    person_name: str = ""
    zones: dict[ContentZone, ContentItem | None]
    resolved_at: datetime

class PersonalizationEngine:
    """
    Resolves what content to show based on cabin state.

    Input: face events (recognized/unknown) + cabin occupancy
    Output: DisplayContent (resolved content per zone)

    Logic:
    1. Person recognized → personal greeting + personal rules + default fills
    2. Unknown person → default content for all zones
    3. Multiple people → neutral/common content (no personal info)
    4. Empty cabin (timeout) → screensaver content
    5. Same person re-entry within cooldown → skip greeting, show info only
    """

    def __init__(self, content_store: ContentStore, config: dict):
        self._store = content_store
        self._cooldown_seconds = config.get("greeting_cooldown", 300)  # 5 min
        self._screensaver_timeout = config.get("screensaver_timeout", 30)  # 30s
        self._current_state = CabinState.EMPTY
        self._current_person: str | None = None
        self._last_greeting: dict[str, float] = {}  # person_id → timestamp
        self._persons_in_cabin: list[str] = []       # Track multiple people

    def on_person_recognized(self, person_id: str, person_name: str, confidence: float) -> DisplayContent:
        """Handle face.recognized event. Returns resolved display content."""
        ...

    def on_person_unknown(self) -> DisplayContent:
        """Handle face.unknown event. Returns default content."""
        ...

    def on_person_exited(self, person_id: str) -> DisplayContent:
        """Handle person leaving cabin (track lost). Returns updated content."""
        ...

    def on_cabin_empty(self) -> DisplayContent:
        """Handle cabin empty (all tracks lost). Returns screensaver."""
        ...

    def get_current_content(self) -> DisplayContent:
        """Get current display state (for status API)."""
        ...

    @property
    def current_state(self) -> CabinState:
        return self._current_state
```

**Resolution logic details:**

```python
def on_person_recognized(self, person_id, person_name, confidence):
    # Check cooldown (same person within 5 min → skip greeting)
    show_greeting = True
    now = time.time()
    if person_id in self._last_greeting:
        if now - self._last_greeting[person_id] < self._cooldown_seconds:
            show_greeting = False

    # Track person in cabin
    if person_id not in self._persons_in_cabin:
        self._persons_in_cabin.append(person_id)

    # Multi-person check
    if len(self._persons_in_cabin) > 1:
        return self._resolve_multi_person()

    # Single person → personalized
    self._current_state = CabinState.PERSONALIZED
    self._current_person = person_id

    zones = {}
    for zone in ContentZone:
        if zone == ContentZone.MAIN and show_greeting:
            # Personal greeting (generated dynamically)
            zones[zone] = self._build_greeting(person_id, person_name)
            self._last_greeting[person_id] = now
        else:
            # Resolve from content rules
            zones[zone] = self._store.resolve_content_for_person(person_id, zone)
            # Fallback to default if no personal rule
            if zones[zone] is None:
                zones[zone] = self._store.resolve_content_for_person(None, zone)

    return DisplayContent(
        state=CabinState.PERSONALIZED,
        person_id=person_id,
        person_name=person_name,
        zones=zones,
        resolved_at=datetime.utcnow(),
    )
```

**Test requirements** (`edge/tests/test_personalization.py`):
- Test single recognized person → personalized content
- Test unknown person → default content for all zones
- Test multi-person → neutral content (no personal info)
- Test cabin empty → screensaver state
- Test person exits → content transitions
- Test greeting cooldown (same person within 5 min → no re-greeting)
- Test greeting reset (same person after 5 min → greeting again)
- Test priority ordering (personal rule priority > default)
- Test zone fallback (no personal rule for zone → use default)
- Test state transitions: EMPTY → PERSONALIZED → EMPTY
- Test state transitions: PERSONALIZED(A) → MULTI_PERSON → PERSONALIZED(B)
- Test dynamic greeting generation (includes person name, time of day)

**Demo**: Simulate sequence of face events → PersonalizationEngine outputs correct DisplayContent at each step, log state transitions.

---

### Task 4: Display Engine — Web Backend

**Objective**: FastAPI + WebSocket server làm display backend, real-time content push.

**Implementation guidance**:

**File: `edge/plugins/display/engine.py` (NEW)** — Abstract interface
```python
from abc import ABC, abstractmethod

class DisplayEngine(ABC):
    """Abstract display output interface."""

    @abstractmethod
    def start(self) -> bool: ...

    @abstractmethod
    def push_content(self, zone: ContentZone, content: ContentItem) -> bool: ...

    @abstractmethod
    def clear_zone(self, zone: ContentZone) -> bool: ...

    @abstractmethod
    def push_full_update(self, display_content: DisplayContent) -> bool:
        """Push all zones at once (used on state change)."""
        ...

    @abstractmethod
    def set_layout(self, layout: DisplayLayout) -> bool: ...

    @abstractmethod
    def get_status(self) -> dict: ...

    @abstractmethod
    def get_connected_clients(self) -> int: ...

    @abstractmethod
    def stop(self) -> None: ...
```

**File: `edge/plugins/display/web_display.py` (NEW)** — Web implementation
```python
import asyncio
import json
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

class WebDisplay(DisplayEngine):
    """
    Web-based display backend.

    Runs FastAPI server in background thread.
    Pushes content to connected browser clients via WebSocket.

    Endpoints:
        GET /                   → Display HTML page (full-screen kiosk mode)
        GET /status             → JSON: current content, clients, uptime
        GET /preview            → Display page in windowed mode (for testing)
        WS  /ws/display         → Real-time content push channel

    WebSocket Protocol (Server → Client):
        {"action": "update_zone", "zone": "main", "content": {...}, "transition": "fade", "duration_ms": 500}
        {"action": "clear_zone", "zone": "main"}
        {"action": "full_update", "zones": {"main": {...}, "sidebar": {...}}, "state": "personalized"}
        {"action": "set_layout", "layout": {...}}

    WebSocket Protocol (Client → Server):
        {"action": "ready"}                 → Client loaded, request current state
        {"action": "heartbeat"}             → Keep-alive ping
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8081,
                 layout: str = "standard_3zone",
                 transition_effect: str = "fade",
                 transition_duration_ms: int = 500):
        self._host = host
        self._port = port
        self._layout_name = layout
        self._transition_effect = transition_effect
        self._transition_duration_ms = transition_duration_ms

        self._app: FastAPI | None = None
        self._server_thread: threading.Thread | None = None
        self._clients: list[WebSocket] = []
        self._current_content: dict[str, dict] = {}  # zone → content dict
        self._current_state: str = "empty"
        self._started_at: float = 0.0

    def start(self) -> bool:
        self._app = self._create_app()
        self._server_thread = threading.Thread(
            target=self._run_server, daemon=True, name="display-server"
        )
        self._server_thread.start()
        self._started_at = time.time()
        return True

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Smart Cabin Display")
        static_dir = Path(__file__).parent / "static"
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def display_page():
            return (static_dir / "index.html").read_text()

        @app.get("/status")
        async def status():
            return {
                "state": self._current_state,
                "zones": self._current_content,
                "clients_connected": len(self._clients),
                "uptime_seconds": time.time() - self._started_at,
                "layout": self._layout_name,
            }

        @app.websocket("/ws/display")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._clients.append(websocket)
            try:
                # Send current state on connect
                await websocket.send_json({
                    "action": "full_update",
                    "zones": self._current_content,
                    "state": self._current_state,
                    "layout": self._layout_name,
                })
                while True:
                    data = await websocket.receive_text()
                    msg = json.loads(data)
                    if msg.get("action") == "heartbeat":
                        await websocket.send_json({"action": "pong"})
            except WebSocketDisconnect:
                self._clients.remove(websocket)

        return app

    def _run_server(self):
        uvicorn.run(self._app, host=self._host, port=self._port, log_level="warning")

    def push_content(self, zone: ContentZone, content: ContentItem) -> bool:
        """Push content to a zone, notify all WebSocket clients."""
        content_dict = content.model_dump() if content else None
        self._current_content[zone.value] = content_dict
        message = {
            "action": "update_zone",
            "zone": zone.value,
            "content": content_dict,
            "transition": self._transition_effect,
            "duration_ms": self._transition_duration_ms,
        }
        self._broadcast(message)
        return True

    def push_full_update(self, display_content: DisplayContent) -> bool:
        """Push all zones at once."""
        self._current_state = display_content.state.value
        zones_dict = {}
        for zone, item in display_content.zones.items():
            zones_dict[zone.value] = item.model_dump() if item else None
        self._current_content = zones_dict
        message = {
            "action": "full_update",
            "zones": zones_dict,
            "state": display_content.state.value,
            "person_name": display_content.person_name,
            "transition": self._transition_effect,
            "duration_ms": self._transition_duration_ms,
        }
        self._broadcast(message)
        return True

    def _broadcast(self, message: dict):
        """Send message to all connected WebSocket clients."""
        # Uses asyncio to send from sync context
        ...
```

**File: `edge/plugins/display/static/index.html` (NEW)**
- Full-screen display page
- CSS Grid layout matching zone configuration
- WebSocket client with auto-reconnect
- Smooth transitions (CSS animations: fade, slide)
- Responsive: adapts to screen size
- Dark theme by default (cabin environment)
- Clock widget, weather placeholder
- Greeting animation (text fade-in with scale)

**File: `edge/plugins/display/static/display.js` (NEW)**
- WebSocket connection management (connect, reconnect with backoff)
- Zone renderer: update DOM elements based on content type
- Transition engine: fade-in/out, slide, cross-dissolve
- Content type handlers: greeting (text), image, video, HTML, ticker
- Layout manager: apply zone positions from layout config
- Heartbeat: ping server every 30s

**File: `edge/plugins/display/static/display.css` (NEW)**
- CSS Grid layout for zones
- Animation keyframes (fadeIn, fadeOut, slideIn, slideOut)
- Typography: large greeting text, notification cards, ticker scroll
- Responsive breakpoints (7", 10", 15", 21")
- Dark theme: dark background, light text, ambient feel
- Zone styling: subtle borders, rounded corners, shadows

**Test requirements** (`edge/tests/test_web_display.py`):
- Test server starts on configured port
- Test GET / returns HTML page
- Test GET /status returns JSON
- Test WebSocket connection accepted
- Test WebSocket receives full_update on connect
- Test push_content → all clients receive update_zone message
- Test push_full_update → all clients receive full_update
- Test clear_zone → clients receive clear message
- Test multiple clients connected simultaneously
- Test client disconnect → removed from client list
- Test server stop → graceful shutdown

**Demo**: Start WebDisplay, open `http://localhost:8081` in browser, push content via Python → verify real-time updates with smooth transitions.

---

### Task 5: Display Plugin — Full Integration

**Objective**: `DisplayPlugin(ServicePlugin)` kết nối mọi thứ: face events → personalization → content → display.

**Implementation guidance**:

**File: `edge/plugins/display/__init__.py` (NEW)**
```python
# Display plugin package
```

**File: `edge/plugins/display/plugin.py` (NEW)**
```python
from typing import Any
from edge.core.service_plugin import ServicePlugin
from edge.core.event_bus import EventBus
from edge.core.logging_setup import get_logger
from edge.plugins.display.models import ContentZone, LAYOUTS
from edge.plugins.display.content_store import ContentStore
from edge.plugins.display.personalization import PersonalizationEngine, CabinState
from edge.plugins.display.web_display import WebDisplay
from shared.event_schemas import BaseEvent, EventType, FaceRecognizedEvent, FaceUnknownEvent

logger = get_logger("plugin")


class Plugin(ServicePlugin):
    """
    Personalized Display plugin.

    Subscribes to face recognition events, resolves personalized content,
    and pushes to display via WebSocket.

    Flow:
        face.recognized → PersonalizationEngine → ContentStore → WebDisplay → Browser
    """

    @property
    def name(self) -> str:
        return "display"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def subscribed_events(self) -> list[EventType]:
        return [
            EventType.FACE_RECOGNIZED,
            EventType.FACE_UNKNOWN,
            # Future: PERSON_EXITED for track lost
        ]

    def start(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        self._config = config
        self._event_bus = event_bus

        # Initialize content store
        db_path = config.get("content_db_path", "data/db/display.db")
        self._store = ContentStore(db_path)
        if not self._store.initialize():
            logger.error("event=display_init_failed | reason=content store init failed")
            return False

        # Initialize personalization engine
        self._personalization = PersonalizationEngine(self._store, config)

        # Initialize display engine
        self._display = WebDisplay(
            host=config.get("host", "0.0.0.0"),
            port=config.get("port", 8081),
            layout=config.get("layout", "standard_3zone"),
            transition_effect=config.get("transition_effect", "fade"),
            transition_duration_ms=config.get("transition_duration_ms", 500),
        )
        if not self._display.start():
            logger.error("event=display_init_failed | reason=display engine start failed")
            return False

        # Push initial screensaver content
        initial_content = self._personalization.on_cabin_empty()
        self._display.push_full_update(initial_content)

        # Start empty-cabin timeout checker
        self._start_empty_checker()

        logger.info(
            "event=display_plugin_started | port={port} | layout={layout} | "
            "content_items={items} | content_rules={rules}",
            port=config.get("port", 8081),
            layout=config.get("layout", "standard_3zone"),
            items=self._store.count_content(),
            rules=self._store.count_rules(),
        )
        return True

    def handle_event(self, event: BaseEvent) -> None:
        """Process face events and update display."""
        if event.event_type == EventType.FACE_RECOGNIZED:
            self._handle_recognized(event)
        elif event.event_type == EventType.FACE_UNKNOWN:
            self._handle_unknown(event)

    def _handle_recognized(self, event: FaceRecognizedEvent) -> None:
        content = self._personalization.on_person_recognized(
            person_id=event.person_id,
            person_name=event.person_name,
            confidence=event.confidence,
        )
        self._display.push_full_update(content)
        logger.info(
            "event=display_personalized | person_id={pid} | person_name={name} | state={state}",
            pid=event.person_id, name=event.person_name, state=content.state.value,
        )

    def _handle_unknown(self, event: FaceUnknownEvent) -> None:
        content = self._personalization.on_person_unknown()
        self._display.push_full_update(content)
        logger.debug("event=display_default | state={state}", state=content.state.value)

    def _start_empty_checker(self):
        """Background thread: check if cabin is empty (no events for timeout period)."""
        # Timer-based: if no face event for `screensaver_timeout` seconds,
        # trigger on_cabin_empty()
        ...

    def stop(self) -> None:
        self._display.stop()
        self._store.close()
        logger.info("event=display_plugin_stopped")

    def health_check(self) -> bool:
        return self._display.get_connected_clients() >= 0  # Server running
```

**Config entry** (`edge/config.yaml`):
```yaml
- name: "display"
  enabled: true
  config:
    host: "0.0.0.0"
    port: 8081
    layout: "standard_3zone"
    content_db_path: "data/db/display.db"
    media_dir: "data/display/media"
    default_greeting: "Xin chào"
    greeting_cooldown: 300          # 5 minutes
    screensaver_timeout: 30         # 30 seconds → empty state
    transition_effect: "fade"
    transition_duration_ms: 500
```

**Test requirements** (`edge/tests/test_display_plugin.py`):
- Test plugin starts successfully with valid config
- Test plugin subscribes to FACE_RECOGNIZED and FACE_UNKNOWN events
- Test face.recognized event → display receives personalized content
- Test face.unknown event → display receives default content
- Test no events for timeout → display switches to screensaver
- Test plugin stop → display server stops
- Test plugin health_check → True when running
- Test multiple face events in sequence (A → B → unknown → empty)
- Test integration: face recognition plugin → EventBus → display plugin → WebDisplay

**Demo**: Run full system:
1. Start face recognition + display plugins
2. Enroll person "0820" with name "Sy"
3. Add content rule: person "0820" → greeting "Xin chào anh Sy!"
4. Person walks in front of camera
5. Display updates with personalized greeting
6. Person leaves → after 30s → screensaver

---

### Task 6: Content Management REST API

**Objective**: REST API cho quản lý content (CRUD), upload media, configure rules.

**Implementation guidance**:

**File: `edge/plugins/display/api.py` (NEW)**
```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from edge.plugins.display.models import ContentItem, ContentRule, ContentZone, ContentType
from edge.plugins.display.content_store import ContentStore

router = APIRouter(prefix="/api/display", tags=["display"])

# Inject content_store reference at startup
_store: ContentStore | None = None

def set_store(store: ContentStore):
    global _store
    _store = store

# --- Content CRUD ---

@router.get("/content")
def list_content(zone: ContentZone | None = None, type: ContentType | None = None):
    """List all content items, optionally filtered by zone or type."""
    return _store.list_content(zone=zone, content_type=type)

@router.get("/content/{content_id}")
def get_content(content_id: str):
    """Get a specific content item."""
    item = _store.get_content(content_id)
    if item is None:
        raise HTTPException(404, "Content not found")
    return item

@router.post("/content", status_code=201)
def create_content(item: ContentItem):
    """Create a new content item."""
    if not _store.add_content(item):
        raise HTTPException(400, "Failed to create content")
    return {"id": item.id, "status": "created"}

@router.put("/content/{content_id}")
def update_content(content_id: str, updates: dict):
    """Update a content item."""
    if not _store.update_content(content_id, updates):
        raise HTTPException(404, "Content not found or update failed")
    return {"id": content_id, "status": "updated"}

@router.delete("/content/{content_id}")
def delete_content(content_id: str):
    """Delete a content item."""
    if not _store.delete_content(content_id):
        raise HTTPException(404, "Content not found")
    return {"id": content_id, "status": "deleted"}

# --- Rules CRUD ---

@router.get("/rules")
def list_rules(person_id: str | None = None, zone: ContentZone | None = None):
    """List personalization rules."""
    ...

@router.post("/rules", status_code=201)
def create_rule(rule: ContentRule):
    """Create a new personalization rule."""
    ...

@router.delete("/rules/{rule_id}")
def delete_rule(rule_id: str):
    """Delete a rule."""
    ...

# --- Media Upload ---

@router.post("/media/upload")
async def upload_media(file: UploadFile = File(...)):
    """Upload image/video for display content."""
    # Save to data/display/media/{filename}
    # Return relative path
    ...

# --- Display Control ---

@router.get("/status")
def display_status():
    """Current display state, connected clients, active content."""
    ...

@router.post("/preview")
def preview_content(content: ContentItem):
    """Preview content on display immediately (temporary, doesn't persist)."""
    ...
```

**Integration**: 
- Mount router vào WebDisplay FastAPI app (cùng port 8081) hoặc vào main Edge API (port 8080)
- Recommendation: mount vào WebDisplay app (port 8081) — giữ display self-contained

**Test requirements** (`edge/tests/test_display_api.py`):
- Test GET /api/display/content → list items
- Test POST /api/display/content → create item
- Test GET /api/display/content/{id} → get item
- Test PUT /api/display/content/{id} → update item
- Test DELETE /api/display/content/{id} → delete item
- Test POST /api/display/rules → create rule
- Test GET /api/display/rules?person_id=0820 → filter rules
- Test DELETE /api/display/rules/{id} → delete rule
- Test POST /api/display/media/upload → upload file
- Test POST /api/display/preview → pushes to display
- Test GET /api/display/status → returns current state
- Test error cases (404, invalid data, duplicate ID)

**Demo**: Dùng Swagger UI (`http://localhost:8081/docs`):
1. Create content items (greeting, notification, ad)
2. Create rules (person "0820" → greeting on MAIN zone)
3. Upload image for ad
4. Preview on display
5. Verify display updates

---

### Task 7: Cloud Content Sync via MQTT

**Objective**: Cloud push content/rules → Edge via MQTT, offline resilience.

**Implementation guidance**:

**File: `edge/plugins/display/cloud_content_sync.py` (NEW)**
```python
class DisplayCloudSync:
    """
    Bridges cloud content management to local ContentStore via MQTT.

    Subscribes to:
        cabin/{device_id}/display/content/push    → Add/update content
        cabin/{device_id}/display/content/delete   → Remove content
        cabin/{device_id}/display/rules/push       → Add/update rules
        cabin/{device_id}/display/rules/delete     → Remove rules
        cabin/{device_id}/display/schedule/update  → Update content schedule

    Publishes:
        cabin/{device_id}/display/status           → Current display state (periodic)
        cabin/{device_id}/display/content/ack      → Acknowledge content received

    Usage:
        Called by DisplayPlugin during start() to register MQTT handlers
        with the CloudSync module.
    """

    def __init__(self, content_store: ContentStore, cloud_sync: CloudSync, device_id: str):
        self._store = content_store
        self._cloud = cloud_sync
        self._device_id = device_id

    def register_handlers(self) -> None:
        """Register MQTT command handlers with CloudSync."""
        self._cloud.register_command("display_content_push", self._handle_content_push)
        self._cloud.register_command("display_content_delete", self._handle_content_delete)
        self._cloud.register_command("display_rules_push", self._handle_rules_push)
        self._cloud.register_command("display_rules_delete", self._handle_rules_delete)

    def _handle_content_push(self, payload: dict) -> None:
        """Cloud pushes new/updated content → store locally."""
        item = ContentItem(**payload.get("content", {}))
        existing = self._store.get_content(item.id)
        if existing:
            self._store.update_content(item.id, item.model_dump())
        else:
            self._store.add_content(item)
        # ACK back to cloud
        self._cloud.publish_raw("", {
            "action": "display_content_ack",
            "content_id": item.id,
            "status": "received",
        })

    def _handle_content_delete(self, payload: dict) -> None:
        content_id = payload.get("content_id")
        self._store.delete_content(content_id)

    def _handle_rules_push(self, payload: dict) -> None:
        rule = ContentRule(**payload.get("rule", {}))
        self._store.add_rule(rule)

    def _handle_rules_delete(self, payload: dict) -> None:
        rule_id = payload.get("rule_id")
        self._store.delete_rule(rule_id)

    def publish_status(self, display_content: DisplayContent) -> None:
        """Publish current display state to cloud."""
        self._cloud.publish_raw("", {
            "action": "display_status",
            "device_id": self._device_id,
            "state": display_content.state.value,
            "person_id": display_content.person_id,
            "zones": {z.value: c.id if c else None for z, c in display_content.zones.items()},
        })
```

**Integration with DisplayPlugin**:
```python
# In DisplayPlugin.start():
if cloud_sync_available:
    self._cloud_content = DisplayCloudSync(self._store, cloud_sync, device_id)
    self._cloud_content.register_handlers()
```

**Offline resilience**:
- Content stored locally (SQLite) → display works without cloud
- Cloud pushes treated as "source of truth sync" → local cache
- If cloud unreachable, display uses last-known content
- On reconnect, cloud can push full content refresh

**Test requirements** (`edge/tests/test_display_cloud_sync.py`):
- Test content push from cloud → stored locally
- Test content delete from cloud → removed locally
- Test rules push from cloud → stored locally
- Test rules delete from cloud → removed locally
- Test ACK published back to cloud after content received
- Test display status published periodically
- Test offline: cloud disconnect → display still works with cached content
- Test reconnect: cloud reconnect → can push new content
- Test invalid payload → error handled gracefully (no crash)
- Test duplicate content push → update (not duplicate)

**Demo**: 
1. Start system with MQTT broker
2. Use `mosquitto_pub` to push content to display topic
3. Verify content appears in local store and on display
4. Disconnect broker → display keeps working
5. Reconnect → push new content → verify update

---

### Task 8: Display Frontend (HTML/CSS/JS)

**Objective**: Complete, polished display client UI chạy trong browser.

**Implementation guidance**:

**File: `edge/plugins/display/static/index.html`**
```html
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Cabin Display</title>
    <link rel="stylesheet" href="/static/display.css">
</head>
<body class="dark-theme">
    <div id="display-container">
        <div id="zone-top_bar" class="zone top-bar">
            <span class="clock" id="clock"></span>
            <span class="weather" id="weather"></span>
            <span class="logo">Smart Cabin</span>
        </div>
        <div id="zone-main" class="zone main">
            <div class="content-wrapper" id="main-content">
                <!-- Dynamic content rendered here -->
            </div>
        </div>
        <div id="zone-sidebar" class="zone sidebar">
            <div class="content-wrapper" id="sidebar-content">
                <!-- Notifications, info -->
            </div>
        </div>
        <div id="zone-bottom_ticker" class="zone bottom-ticker">
            <div class="ticker-content" id="ticker-content">
                <!-- Scrolling text -->
            </div>
        </div>
    </div>
    <div id="connection-status" class="connection-indicator"></div>
    <script src="/static/display.js"></script>
</body>
</html>
```

**File: `edge/plugins/display/static/display.css`**
- Layout: CSS Grid matching zone positions
- Themes: Dark (default), customizable via CSS variables
- Typography: Large greeting (3-4rem), notification cards, ticker
- Animations:
  - `@keyframes fadeIn` / `fadeOut`
  - `@keyframes slideInFromRight` / `slideOutToLeft`
  - `@keyframes tickerScroll` (infinite horizontal scroll)
  - `@keyframes pulse` (greeting emphasis)
- Responsive: media queries cho 7", 10", 15", 21"
- Connection indicator: green dot (connected), red dot (disconnected)
- Content type styling: greeting (centered, large), notification (card), ad (image fill), ticker (scroll)

**File: `edge/plugins/display/static/display.js`**
```javascript
class SmartCabinDisplay {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectDelay = 10000;
        this.heartbeatInterval = null;
        this.currentState = "empty";
    }

    connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws/display`);
        this.ws.onopen = () => this.onConnected();
        this.ws.onclose = () => this.onDisconnected();
        this.ws.onmessage = (event) => this.onMessage(JSON.parse(event.data));
    }

    onConnected() {
        this.reconnectAttempts = 0;
        this.updateConnectionStatus(true);
        this.startHeartbeat();
    }

    onDisconnected() {
        this.updateConnectionStatus(false);
        this.stopHeartbeat();
        this.scheduleReconnect();
    }

    onMessage(msg) {
        switch (msg.action) {
            case "full_update":
                this.handleFullUpdate(msg);
                break;
            case "update_zone":
                this.handleZoneUpdate(msg);
                break;
            case "clear_zone":
                this.handleZoneClear(msg);
                break;
            case "set_layout":
                this.handleLayoutChange(msg);
                break;
        }
    }

    handleFullUpdate(msg) {
        this.currentState = msg.state;
        for (const [zone, content] of Object.entries(msg.zones)) {
            this.renderZone(zone, content, msg.transition, msg.duration_ms);
        }
    }

    handleZoneUpdate(msg) {
        this.renderZone(msg.zone, msg.content, msg.transition, msg.duration_ms);
    }

    renderZone(zone, content, transition = "fade", durationMs = 500) {
        const element = document.getElementById(`zone-${zone}`);
        if (!element) return;

        const wrapper = element.querySelector('.content-wrapper') || element;

        // Apply transition out
        wrapper.style.transition = `opacity ${durationMs}ms ease`;
        wrapper.style.opacity = "0";

        setTimeout(() => {
            // Render new content
            wrapper.innerHTML = this.renderContent(content);
            // Transition in
            wrapper.style.opacity = "1";
        }, durationMs);
    }

    renderContent(content) {
        if (!content) return "";
        switch (content.type) {
            case "greeting":
                return `<div class="greeting"><h1>${content.body}</h1></div>`;
            case "notification":
                return `<div class="notification"><h3>${content.title}</h3><p>${content.body}</p></div>`;
            case "advertisement":
                if (content.media_url) {
                    return `<div class="advertisement"><img src="${content.media_url}" alt="${content.title}"></div>`;
                }
                return `<div class="advertisement"><h2>${content.title}</h2><p>${content.body}</p></div>`;
            case "image":
                return `<div class="image-content"><img src="${content.media_url}" alt=""></div>`;
            case "weather":
                return `<div class="weather-widget">${content.body}</div>`;
            case "clock":
                return `<div class="clock-widget">${this.getCurrentTime()}</div>`;
            case "news":
                return `<div class="news-ticker">${content.body}</div>`;
            case "html":
                return content.body;
            default:
                return `<div class="generic">${content.body || content.title || ""}</div>`;
        }
    }

    scheduleReconnect() {
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay);
        this.reconnectAttempts++;
        setTimeout(() => this.connect(), delay);
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ action: "heartbeat" }));
            }
        }, 30000);
    }

    // ... clock update, connection status, etc.
}

// Initialize
const display = new SmartCabinDisplay();
display.connect();
```

**Test requirements** (`edge/tests/test_display_frontend.py` — integration):
- Test HTML page loads (GET / returns 200, contains expected elements)
- Test WebSocket connects and receives full_update
- Test content rendering (each content type produces expected HTML structure)
- Test transition timing (opacity changes)
- Test reconnection after disconnect
- Test multiple content updates in sequence
- Test screensaver → personalized → screensaver transitions

**Demo**: Full visual demo:
1. Open `http://localhost:8081` in browser (full screen / kiosk mode)
2. Initially: dark screensaver with clock
3. Person recognized → smooth fade to personalized greeting
4. After 10s → show notification in sidebar
5. Person leaves → fade back to default/screensaver
6. Test on multiple screen sizes (resize browser)

---

### Task 9: End-to-End Integration & Testing

**Objective**: Full integration test, performance validation, documentation.

**Implementation guidance**:

**Integration Tests** (`edge/tests/test_display_integration.py`):
```python
class TestDisplayIntegration:
    """End-to-end tests for display module."""

    def test_full_flow_recognized(self):
        """Camera → Face Recognition → EventBus → Display → WebSocket → Content"""
        # Setup: start all components
        # Action: simulate face.recognized event
        # Assert: WebSocket client receives personalized content within 500ms

    def test_full_flow_unknown(self):
        """Unknown face → default content displayed."""

    def test_full_flow_empty_cabin(self):
        """No events for 30s → screensaver displayed."""

    def test_person_transition(self):
        """Person A → Person B → content changes correctly."""

    def test_content_update_while_running(self):
        """Update content via API → next display cycle uses new content."""

    def test_cloud_push_content(self):
        """Cloud pushes content via MQTT → appears on display."""

    def test_offline_operation(self):
        """Disconnect cloud → display still works with cached content."""

    def test_multi_client(self):
        """Multiple browser clients connected → all receive updates."""
```

**Performance Tests**:
```python
class TestDisplayPerformance:
    def test_latency_face_to_display(self):
        """face.recognized → WebSocket message sent < 500ms"""

    def test_memory_usage(self):
        """Display server + content cache < 200MB after 1000 events"""

    def test_websocket_throughput(self):
        """Handle 10 content updates/second without drops"""
```

**Example Script**: `examples/run_display_demo.py`
```python
"""
Demo: Personalized Display Module

Usage:
    # Start display only (no camera needed)
    python examples/run_display_demo.py demo

    # Start with face recognition (needs camera)
    python examples/run_display_demo.py full --url 0

    # Seed sample content
    python examples/run_display_demo.py seed

Commands:
    demo  - Start display with simulated face events
    full  - Full pipeline (camera + face + display)
    seed  - Add sample content to database
"""
```

**Documentation updates**:
- Update `README.md`: add Display module section
- Update `edge/config.yaml`: add display plugin config (enabled: false by default)

**Test requirements**:
- All integration tests pass
- Latency test: face event → WebSocket push < 500ms (measured)
- Memory test: stable after 1000+ events (no leak)
- 24-hour continuous run: no crashes, no memory growth
- Content API: all endpoints respond < 100ms
- WebSocket: supports 5+ concurrent clients

**Demo** (final):
Full demo video/live showing:
1. Start system: `python -m edge.main` (or equivalent entry point)
2. Open display in browser: `http://{orange_pi_ip}:8081`
3. Initially shows: screensaver (clock, building logo)
4. Person "Sy" (0820, floor 8) walks to camera
5. Within 500ms: display shows "Xin chào anh Sy! Chúc buổi sáng tốt lành — Tầng 8"
6. Within 300ms: MQTT publishes floor call `{floor: 8}` to elevator controller
7. Person "Ngọc Cần" (0681, floor 6) also enters
8. Display updates: "Tầng 8 (Sy), Tầng 6 (Ngọc Cần)"
9. MQTT publishes second floor call `{floor: 6}`
10. Person leaves camera view
11. After 30s: display fades back to default ads
12. Cloud admin pushes new ad via MQTT → appears on display immediately
13. New person (unknown) enters → display shows generic content, no floor call

---

### Task 10: Auto Floor Call — Elevator Plugin

**Objective**: Khi nhận diện được người, tự động gọi tầng qua MQTT. Hỗ trợ nhiều người cùng lúc.

**Implementation guidance**:

**File: `edge/plugins/face_recognition/database.py` (UPDATE)**
- Thêm cột `default_floor` vào schema `persons`:
  ```sql
  ALTER TABLE persons ADD COLUMN default_floor INTEGER DEFAULT NULL;
  ```
- Thêm methods:
  ```python
  def update_person_floor(self, person_id: str, floor: int) -> bool: ...
  def get_person_floor(self, person_id: str) -> int | None: ...
  ```
- Migration: auto-detect nếu cột chưa tồn tại → add column

**File: `edge/tools/enroll_face.py` (UPDATE)**
- Thêm `--floor` argument:
  ```
  python examples/run_recognition.py enroll --image face.jpg --name "Sy" --id 0820 --floor 8
  ```
- Validate floor: positive integer, optional

**File: `examples/run_recognition.py` (UPDATE)**
- `enroll` command: thêm `--floor` option
- `list` command: hiển thị thêm cột floor

**File: `edge/plugins/elevator/__init__.py` (NEW)**
**File: `edge/plugins/elevator/plugin.py` (NEW)**
```python
from typing import Any
from edge.core.service_plugin import ServicePlugin
from edge.core.event_bus import EventBus
from edge.core.logging_setup import get_logger
from shared.event_schemas import BaseEvent, EventType, FaceRecognizedEvent

logger = get_logger("plugin")


class Plugin(ServicePlugin):
    """
    Auto Floor Call plugin.
    
    Listens to face.recognized events, looks up person's default_floor,
    and publishes floor call command via MQTT.
    
    Multi-person: each recognized person triggers their own floor call.
    Dedup: same person → same floor not called again within cooldown.
    """

    @property
    def name(self) -> str:
        return "elevator"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def subscribed_events(self) -> list[EventType]:
        return [EventType.FACE_RECOGNIZED]

    def start(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        self._config = config
        self._event_bus = event_bus
        self._confidence_threshold = config.get("confidence_threshold", 0.75)
        self._cooldown_seconds = config.get("cooldown_seconds", 30)
        self._mqtt_topic = config.get("mqtt_topic", "elevator/floor_call")
        
        # Cooldown tracker: {person_id: last_call_timestamp}
        self._last_call: dict[str, float] = {}
        
        # Floor mapping from config (fallback if not in database)
        self._floor_mapping: dict[str, int] = config.get("floor_mapping", {})
        
        # Reference to face database (for floor lookup)
        self._face_db = None  # Set during integration
        
        # Reference to cloud_sync (for MQTT publish)
        self._cloud_sync = None  # Set during integration
        
        logger.info(
            "event=elevator_plugin_started | topic={topic} | threshold={t} | cooldown={c}s",
            topic=self._mqtt_topic, t=self._confidence_threshold, c=self._cooldown_seconds,
        )
        return True

    def handle_event(self, event: BaseEvent) -> None:
        if event.event_type == EventType.FACE_RECOGNIZED:
            self._handle_floor_call(event)

    def _handle_floor_call(self, event: FaceRecognizedEvent) -> None:
        person_id = event.person_id
        confidence = event.confidence

        # Safety: confidence check
        if confidence < self._confidence_threshold:
            return

        # Lookup floor (database first, then config fallback)
        floor = self._get_floor(person_id)
        if floor is None:
            return  # No floor registered for this person

        # Cooldown check
        import time
        now = time.time()
        if person_id in self._last_call:
            if now - self._last_call[person_id] < self._cooldown_seconds:
                return  # Already called recently

        # Publish MQTT floor call
        self._publish_floor_call(person_id, event.person_name, floor, confidence)
        self._last_call[person_id] = now

        logger.info(
            "event=floor_call_sent | person_id={pid} | person_name={name} | "
            "floor={floor} | confidence={conf:.3f} | topic={topic}",
            pid=person_id, name=event.person_name, floor=floor,
            conf=confidence, topic=self._mqtt_topic,
        )

    def _get_floor(self, person_id: str) -> int | None:
        """Lookup person's default floor. Database first, config fallback."""
        # Try face database
        if self._face_db:
            floor = self._face_db.get_person_floor(person_id)
            if floor is not None:
                return floor
        # Fallback to config floor_mapping
        return self._floor_mapping.get(person_id)

    def _publish_floor_call(self, person_id: str, person_name: str, 
                            floor: int, confidence: float) -> None:
        """Publish floor call via MQTT. Payload format TBD."""
        from datetime import datetime, timezone
        payload = {
            "person_id": person_id,
            "person_name": person_name,
            "floor": floor,
            "confidence": round(confidence, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_id": self._config.get("device_id", "cabin-001"),
        }
        # Payload format will be finalized when elevator controller spec is provided
        if self._cloud_sync:
            self._cloud_sync.publish_raw(self._mqtt_topic, payload)

    def stop(self) -> None:
        self._last_call.clear()
        logger.info("event=elevator_plugin_stopped")

    def health_check(self) -> bool:
        return True
```

**Config** (`edge/config.yaml`):
```yaml
- name: "elevator"
  enabled: true
  config:
    mqtt_topic: "elevator/floor_call"
    confidence_threshold: 0.75
    cooldown_seconds: 30
    floor_mapping:              # Fallback (primary source is face database)
      "0820": 8
      "0681": 6
```

**Test requirements** (`edge/tests/test_elevator_plugin.py`):
- Test floor lookup from database (person has floor → returns floor)
- Test floor lookup from config fallback (person not in DB, in config)
- Test floor lookup returns None (person has no floor → no call)
- Test confidence threshold (below → skip, above → call)
- Test cooldown (same person within 30s → skip, after 30s → call again)
- Test multi-person (A recognized → call floor 8, B recognized → call floor 3)
- Test MQTT publish (correct topic, correct payload structure)
- Test enrollment with floor (enroll --floor 8 → stored in DB)
- Test update floor (change person's floor → new calls use new floor)
- Test integration: face.recognized event → elevator plugin → MQTT publish

**Demo**:
1. Enroll person "0820" (Sy) with `--floor 8`
2. Enroll person "0681" (Ngọc Cần) with `--floor 6`
3. Start system with elevator plugin enabled
4. Person Sy walks in → MQTT message published to `elevator/floor_call` with `{floor: 8}`
5. Person Ngọc Cần walks in → MQTT message `{floor: 6}`
6. Subscribe `mosquitto_sub -t elevator/floor_call` → verify both messages received
7. Sy walks out and back in within 30s → no duplicate call (cooldown)
8. Display shows: "Tầng 8 (Sy), Tầng 6 (Ngọc Cần)"

---

## Technology Summary

| Component | Technology | Lý do |
|-----------|-----------|-------|
| ServicePlugin base | Python ABC | Clean contract, type-safe, consistent with existing BasePlugin |
| Content Storage | SQLite | Lightweight, no server, same as face DB |
| Display Server | FastAPI + uvicorn | Async, fast, auto-docs, same stack as planned Edge API |
| Real-time Push | WebSocket | Bidirectional, low latency, browser-native |
| Display Client | HTML + CSS + Vanilla JS | No build step, no framework dependency, fast load |
| Content Sync | MQTT (existing CloudSync) | Already built, offline buffer, reliable |
| Layout System | CSS Grid | Flexible, responsive, native browser |
| Transitions | CSS Animations | GPU-accelerated, smooth, no JS overhead |

## Dependencies (New)

```
# requirements.txt additions
fastapi>=0.100.0      # (may already exist for Edge API)
uvicorn>=0.23.0       # ASGI server
websockets>=11.0      # WebSocket support for FastAPI
```

## File Structure (Display + Elevator Modules)

```
edge/plugins/display/
├── __init__.py
├── plugin.py              # DisplayPlugin(ServicePlugin) — main entry
├── engine.py              # DisplayEngine ABC
├── web_display.py         # WebDisplay(DisplayEngine) — FastAPI + WS
├── content_store.py       # SQLite content/rules storage
├── personalization.py     # PersonalizationEngine — content resolution
├── models.py              # ContentItem, ContentZone, ContentRule, etc.
├── api.py                 # REST API router for content management
├── cloud_content_sync.py  # MQTT ↔ ContentStore bridge
└── static/
    ├── index.html         # Display client page
    ├── display.js         # WebSocket client + renderer
    └── display.css        # Layout, theme, animations

edge/plugins/elevator/
├── __init__.py
└── plugin.py              # ElevatorPlugin(ServicePlugin) — auto floor call via MQTT
```

## Implementation Order & Dependencies

```
Task 1: ServicePlugin ──────────────────┐
                                         │
Task 2: Content Model & Storage ────────┤
                                         ├──→ Task 5: Display Plugin (Integration)
Task 3: Personalization Engine ─────────┤         │
                                         │         ├──→ Task 6: Content API
Task 4: Display Engine (Web) ───────────┘         │
                                                   ├──→ Task 7: Cloud Sync
Task 8: Frontend (HTML/CSS/JS) ───────────────────┘
                                                   │
Task 10: Elevator Plugin ──────────────────────────┤ (parallel, needs Task 1 only)
                                                   │
                                                   └──→ Task 9: E2E Testing
```

**Critical path**: Task 1 → Task 5 → Task 9
**Parallel work**: Tasks 2, 3, 4, 10 can be developed in parallel after Task 1.

## Estimated Effort

| Task | Effort | Priority |
|------|--------|----------|
| Task 1: ServicePlugin | 1 day | P0 (blocker) |
| Task 2: Content Model | 1 day | P0 |
| Task 3: Personalization | 1-2 days | P0 |
| Task 4: Web Display Engine | 2 days | P0 |
| Task 5: Plugin Integration | 1 day | P0 |
| Task 6: Content API | 1 day | P1 |
| Task 7: Cloud Sync | 1 day | P1 |
| Task 8: Frontend Polish | 1-2 days | P1 |
| Task 9: E2E + Performance | 1-2 days | P0 |
| Task 10: Elevator Plugin | 1-2 days | P0 |
| **Total** | **~12-15 days** | |

## Success Criteria

| Metric | Target |
|--------|--------|
| Face → Display latency | < 500ms |
| Face → Floor Call latency | < 300ms |
| Display uptime | > 99% (24/7 continuous) |
| Content API response time | < 100ms |
| WebSocket concurrent clients | 5+ |
| Memory usage (display + elevator) | < 250MB |
| Floor call accuracy | 100% (correct floor for recognized person) |
| Multi-person floor call | All recognized persons get floor called |
| Cooldown dedup | 0 duplicate calls within cooldown period |
| Existing tests regression | 0 failures |
| New test coverage | > 80% cho display + elevator modules |

---

*Document version: 1.0*
*Created: 2026-08-14*
*Author: DATGROUP — Smart Cabin Team*
*Reference: [architecture_v2.md](architecture_v2.md) | [vision.md](vision.md)*
