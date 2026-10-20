from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.config import get_settings
from app.db.mongodb import check_mongo_connection
from app.schemas.status import DatabaseStatus, HealthResponse

router = APIRouter(tags=["System Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health_status():
    settings = get_settings()
    is_connected, details = await check_mongo_connection()

    db_status = DatabaseStatus(
        status="connected" if is_connected else "disconnected",
        engine="MongoDB Atlas",
        details=details,
    )

    return HealthResponse(
        status="online",
        version=settings.version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        database=db_status,
        environment=settings.environment,
    )
