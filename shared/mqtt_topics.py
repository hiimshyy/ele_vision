"""
Smart Cabin Platform - MQTT Topic Definitions

Centralizes all MQTT topic patterns used for Edge-Cloud communication.
Topics follow the pattern: cabin/{device_id}/{category}/{action}
"""

# --- Edge → Cloud (Publish from edge) ---

# Face Recognition
TOPIC_FACE_RECOGNIZED = "cabin/{device_id}/face/recognized"
TOPIC_FACE_UNKNOWN = "cabin/{device_id}/face/unknown"

# People Counter
TOPIC_PEOPLE_COUNT = "cabin/{device_id}/people/count"

# System Status
TOPIC_HEARTBEAT = "cabin/{device_id}/status/heartbeat"
TOPIC_SYSTEM_ERROR = "cabin/{device_id}/system/error"
TOPIC_SYSTEM_START = "cabin/{device_id}/system/start"
TOPIC_SYSTEM_STOP = "cabin/{device_id}/system/stop"

# --- Cloud → Edge (Subscribe on edge) ---

# Commands
TOPIC_CMD_SYNC_FACES = "cabin/{device_id}/command/sync_faces"
TOPIC_CMD_UPDATE_CONFIG = "cabin/{device_id}/command/update_config"
TOPIC_CMD_RESTART_PLUGIN = "cabin/{device_id}/command/restart_plugin"

# --- Wildcards (for cloud subscriber) ---

# Subscribe to all events from all devices
TOPIC_ALL_FACE_EVENTS = "cabin/+/face/#"
TOPIC_ALL_STATUS = "cabin/+/status/#"
TOPIC_ALL_SYSTEM = "cabin/+/system/#"
