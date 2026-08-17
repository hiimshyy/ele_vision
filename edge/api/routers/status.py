"""
Status & Stats API routes.

GET /api/status  - System status (CPU, RAM, uptime, MQTT)
GET /api/stats   - Pipeline stats (FPS, resolution)
GET /api/plugins - Plugin status list
POST /api/plugins/{name}/restart - Restart a plugin
"""

import time

import psutil
from fastapi import APIRouter

from edge.api.schemas.status import (
    SystemStatusResponse,
    PipelineStatsResponse,
    PluginStatusItem,
    PluginStatusResponse,
)

router = APIRouter(prefix="/api", tags=["status"])

# These will be set by main.py at startup
_start_time: float = 0.0
_device_id: str = "cabin-001"
_database = None
_cloud_sync = None


def configure(device_id: str, database=None, cloud_sync=None):
    """Configure router dependencies (called from main.py)."""
    global _start_time, _device_id, _database, _cloud_sync
    _start_time = time.time()
    _device_id = device_id
    _database = database
    _cloud_sync = cloud_sync


@router.get("/status", response_model=SystemStatusResponse)
async def get_status():
    """Get system status: CPU, RAM, uptime, MQTT, database."""
    uptime = time.time() - _start_time if _start_time > 0 else 0
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()

    db_persons = 0
    db_embeddings = 0
    if _database:
        db_persons = _database.count_persons()
        db_embeddings = _database.count()

    mqtt_connected = False
    if _cloud_sync:
        mqtt_connected = _cloud_sync.is_connected

    return SystemStatusResponse(
        status="running",
        device_id=_device_id,
        uptime_s=round(uptime, 1),
        cpu_percent=cpu,
        ram_used_mb=round(mem.used / (1024 * 1024), 1),
        ram_percent=mem.percent,
        mqtt_connected=mqtt_connected,
        db_persons=db_persons,
        db_embeddings=db_embeddings,
    )


@router.get("/stats", response_model=PipelineStatsResponse)
async def get_stats():
    """Get pipeline stats (placeholder — full stats when pipeline running)."""
    uptime = time.time() - _start_time if _start_time > 0 else 0
    return PipelineStatsResponse(
        uptime_s=round(uptime, 1),
    )


@router.get("/plugins", response_model=PluginStatusResponse)
async def get_plugins():
    """Get status of all loaded plugins."""
    # Placeholder — will be connected to PluginManager in production
    return PluginStatusResponse(plugins=[])
