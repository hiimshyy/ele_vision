"""Schemas for face enrollment API."""

from pydantic import BaseModel, Field


class FaceEnrollRequest(BaseModel):
    """Request body for face enrollment (JSON metadata, image via form)."""
    person_id: str = Field(description="Unique person identifier")
    name: str = Field(description="Person display name")
    default_floor: int | None = Field(default=None, description="Default floor for elevator")


class FaceEnrollResponse(BaseModel):
    """Response after successful enrollment."""
    success: bool
    person_id: str
    name: str
    default_floor: int | None = None
    embedding_dim: int = 512
    face_size: str = ""
    message: str = ""


class FaceDetailResponse(BaseModel):
    """Single face record."""
    person_id: str
    name: str
    default_floor: int | None = None
    embedding_count: int = 1
    created_at: str = ""


class FaceListResponse(BaseModel):
    """Response for listing all faces."""
    total_embeddings: int
    total_persons: int
    persons: list[FaceDetailResponse]
