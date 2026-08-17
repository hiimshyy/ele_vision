"""
Face Enrollment API routes.

GET    /api/faces          - List all enrolled faces
GET    /api/faces/{id}     - Get specific person details
POST   /api/faces/enroll   - Enroll a face (multipart: image + metadata)
DELETE /api/faces/{id}     - Remove a person
PUT    /api/faces/{id}/floor - Update person's default floor
"""

import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from edge.api.schemas.faces import (
    FaceEnrollResponse,
    FaceDetailResponse,
    FaceListResponse,
)
from edge.core.logging_setup import get_logger

logger = get_logger("system")

router = APIRouter(prefix="/api/faces", tags=["faces"])

# These will be set by main.py at startup
_database = None
_detector = None
_embedder = None
_data_faces_dir = Path("data/faces")


def configure(database, detector, embedder, data_faces_dir: Path = Path("data/faces")):
    """Configure router dependencies (called from main.py)."""
    global _database, _detector, _embedder, _data_faces_dir
    _database = database
    _detector = detector
    _embedder = embedder
    _data_faces_dir = data_faces_dir


@router.get("", response_model=FaceListResponse)
async def list_faces():
    """List all enrolled persons with embedding counts."""
    if _database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    records = _database.get_all()
    persons_map = {}
    for r in records:
        if r.person_id not in persons_map:
            persons_map[r.person_id] = {
                "person_id": r.person_id,
                "name": r.name,
                "default_floor": r.default_floor,
                "embedding_count": 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.created_at)),
            }
        persons_map[r.person_id]["embedding_count"] += 1

    persons = [FaceDetailResponse(**p) for p in persons_map.values()]

    return FaceListResponse(
        total_embeddings=_database.count(),
        total_persons=_database.count_persons(),
        persons=persons,
    )


@router.get("/{person_id}", response_model=FaceDetailResponse)
async def get_face(person_id: str):
    """Get details for a specific person."""
    if _database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    records = _database.get_person(person_id)
    if not records:
        raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")

    r = records[0]
    return FaceDetailResponse(
        person_id=r.person_id,
        name=r.name,
        default_floor=r.default_floor,
        embedding_count=len(records),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.created_at)),
    )


@router.post("/enroll", response_model=FaceEnrollResponse)
async def enroll_face(
    image: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    person_id: str = Form(..., description="Person ID"),
    name: str = Form(..., description="Person name"),
    default_floor: int | None = Form(default=None, description="Default floor"),
):
    """
    Enroll a face from uploaded image.

    Validates: single face, minimum size (60px), quality check.
    """
    if _database is None or _detector is None or _embedder is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Detect faces
    faces = _detector.detect(frame, conf_threshold=0.5)

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in image")
    if len(faces) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple faces detected ({len(faces)}). Use image with single face.",
        )

    face = faces[0]

    # Size check
    if face.width < 60 or face.height < 60:
        raise HTTPException(
            status_code=400,
            detail=f"Face too small ({int(face.width)}x{int(face.height)}px). Minimum 60x60.",
        )

    # Align
    from edge.plugins.face_recognition.alignment import align_face
    aligned = align_face(frame, face.landmarks)
    if aligned is None:
        raise HTTPException(status_code=400, detail="Face alignment failed")

    # Extract embedding
    embedding = _embedder.extract(aligned)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding extraction failed")

    # Save to database
    _database.add_face(person_id, name, embedding, default_floor=default_floor)

    # Save aligned face image
    face_dir = _data_faces_dir / person_id
    face_dir.mkdir(parents=True, exist_ok=True)
    face_filename = f"{int(time.time())}_{image.filename or 'upload'}.png"
    cv2.imwrite(str(face_dir / face_filename), aligned)

    logger.info(
        "event=api_face_enrolled | person_id={pid} | name={n} | floor={f} | "
        "face_size={w}x{h}",
        pid=person_id, n=name, f=default_floor,
        w=int(face.width), h=int(face.height),
    )

    return FaceEnrollResponse(
        success=True,
        person_id=person_id,
        name=name,
        default_floor=default_floor,
        embedding_dim=embedding.shape[0],
        face_size=f"{int(face.width)}x{int(face.height)}",
        message="Face enrolled successfully",
    )


@router.delete("/{person_id}")
async def delete_face(person_id: str):
    """Remove a person and all their embeddings."""
    if _database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    removed = _database.remove_face(person_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")

    # Remove face images
    import shutil
    face_dir = _data_faces_dir / person_id
    if face_dir.exists():
        shutil.rmtree(face_dir)

    logger.info("event=api_face_deleted | person_id={pid} | removed={n}", pid=person_id, n=removed)

    return {"success": True, "person_id": person_id, "removed": removed}


@router.put("/{person_id}/floor")
async def update_floor(person_id: str, floor: int | None = None):
    """Update default floor for a person."""
    if _database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    records = _database.get_person(person_id)
    if not records:
        raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")

    _database.update_person_floor(person_id, floor)

    return {"success": True, "person_id": person_id, "default_floor": floor}
