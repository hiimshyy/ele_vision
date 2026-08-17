"""Schemas for system status API."""

from pydantic import BaseModel, Field


class SystemStatusResponse(BaseModel):
    """System status overview."""
    status: str = "running"
    device_id: str = ""
    uptime_s: float = 0.0
    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_percent: float = 0.0
    mqtt_connected: bool = False
    db_persons: int = 0
    db_embeddings: int = 0


class PipelineStatsResponse(BaseModel):
    """Video pipeline statistics."""
    capture_fps: float = 0.0
    distribute_fps: float = 0.0
    resolution: str = ""
    reconnects: int = 0
    uptime_s: float = 0.0


class PluginStatusItem(BaseModel):
    """Single plugin status."""
    name: str
    version: str = ""
    state: str = ""
    fps: float = 0.0
    uptime_s: float = 0.0
    error: str = ""


class PluginStatusResponse(BaseModel):
    """Response for plugin list."""
    plugins: list[PluginStatusItem]
