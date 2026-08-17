"""
Smart Cabin Platform - Edge REST API

FastAPI application exposing face enrollment, system status, and plugin management.
Designed for cloud team integration and admin operations.

Usage:
    # Run standalone (development)
    python -m edge.api.main

    # Or with uvicorn directly
    uvicorn edge.api.main:app --host 0.0.0.0 --port 8080 --reload

    # Access docs
    http://localhost:8080/docs   (Swagger UI)
    http://localhost:8080/redoc  (ReDoc)
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from edge.core.logging_setup import setup_logging, get_logger
from edge.plugins.face_recognition.detector import FaceDetector
from edge.plugins.face_recognition.embedder import FaceEmbedder
from edge.plugins.face_recognition.database import FaceDatabase
from edge.api.routers import status, faces

logger = get_logger("system")

# --- Load .env ---
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# --- Config ---
MODEL_DIR = Path("edge/plugins/face_recognition/models")
DB_PATH = Path("data/db/faces.db")
DATA_FACES_DIR = Path("data/faces")
DEVICE_ID = os.environ.get("SC_DEVICE_ID", "cabin-001")
API_HOST = os.environ.get("SC_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("SC_API_PORT", "8080"))

# --- App ---
app = FastAPI(
    title="Smart Cabin - Edge API",
    description="Face Recognition & System Management API for Smart Cabin Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (allow all for development, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Shared resources ---
_database: FaceDatabase | None = None
_detector: FaceDetector | None = None
_embedder: FaceEmbedder | None = None


@app.on_event("startup")
async def startup():
    """Initialize models and database on startup."""
    global _database, _detector, _embedder

    setup_logging("INFO")
    logger.info("event=api_starting | host={h} | port={p}", h=API_HOST, p=API_PORT)

    # Load detector
    _detector = FaceDetector()
    if not _detector.load():
        logger.error("event=api_init_failed | reason=detector load failed")
    else:
        logger.info("event=api_detector_loaded | model={m}", m=_detector.model_name)

    # Load embedder
    _embedder = FaceEmbedder()
    if not _embedder.load(MODEL_DIR / "w600k_mbf.onnx"):
        logger.error("event=api_init_failed | reason=embedder load failed")
    else:
        logger.info("event=api_embedder_loaded | model={m}", m=_embedder.model_name)

    # Initialize database
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _database = FaceDatabase(DB_PATH)
    if not _database.initialize():
        logger.error("event=api_init_failed | reason=database init failed")
    else:
        logger.info(
            "event=api_database_ready | persons={p} | embeddings={e}",
            p=_database.count_persons(), e=_database.count(),
        )

    # Configure routers with shared resources
    status.configure(device_id=DEVICE_ID, database=_database)
    faces.configure(
        database=_database,
        detector=_detector,
        embedder=_embedder,
        data_faces_dir=DATA_FACES_DIR,
    )

    logger.info("event=api_started | device_id={d}", d=DEVICE_ID)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    if _database:
        _database.close()
    logger.info("event=api_stopped")


# --- Include routers ---
app.include_router(status.router)
app.include_router(faces.router)


# --- Root endpoint ---
@app.get("/")
async def root():
    """API root — health check."""
    return {
        "service": "Smart Cabin Edge API",
        "version": "1.0.0",
        "device_id": DEVICE_ID,
        "docs": "/docs",
    }


# --- Run standalone ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("edge.api.main:app", host=API_HOST, port=API_PORT, reload=False)
