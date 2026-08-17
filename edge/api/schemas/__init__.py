"""API request/response schemas."""

from edge.api.schemas.faces import (
    FaceEnrollRequest,
    FaceEnrollResponse,
    FaceListResponse,
    FaceDetailResponse,
)
from edge.api.schemas.status import (
    SystemStatusResponse,
    PipelineStatsResponse,
    PluginStatusResponse,
)
