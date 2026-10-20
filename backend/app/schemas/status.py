from typing import Dict, Optional
from pydantic import BaseModel, Field


class DatabaseStatus(BaseModel):
    status: str = Field(description="'connected' or 'disconnected'")
    engine: str = Field(default="MongoDB Atlas", description="Database engine name")
    details: str = Field(description="Connection diagnostic or fallback status")


class HealthResponse(BaseModel):
    status: str = Field(default="online", description="API operational status")
    version: str = Field(description="MediVision system version")
    timestamp: str = Field(description="ISO-8601 server timestamp")
    database: DatabaseStatus
    environment: str = Field(description="Current deployment environment")


class ModelItemStatus(BaseModel):
    name: str
    modality: str
    architecture: str
    framework: str
    status: str = Field(default="unconfigured", description="'unconfigured', 'training', 'ready'")
    trained: bool = Field(default=False, description="Strictly false in Stage 1")
    weights_path: Optional[str] = None
    next_stage: str
    description: str


class ModelsStatusResponse(BaseModel):
    stage: int = Field(default=1, description="Current development milestone stage")
    system_status: str = Field(default="foundation_ready")
    total_models: int
    active_trained_count: int = Field(default=0, description="Count of trained models (0 in Stage 1)")
    models: Dict[str, ModelItemStatus]
    disclaimer: str = Field(
        default="Stage 1 foundation only: No clinical prediction models are trained or executing inferences."
    )
