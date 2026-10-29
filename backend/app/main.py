from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import brain_mri, clinical_ml, health, imaging, models, skin_lesion, xai, yolo
from app.core.config import get_settings
from app.db.mongodb import close_mongo_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    settings = get_settings()
    print(f"[MediVision AI] Starting Stage 2 Backend on {settings.host}:{settings.port}")

    # Load Chest X-ray ResNet-50 model once at application startup
    from app.services.chest_xray import chest_xray_service
    try:
        loaded = chest_xray_service.load_model()
        if loaded:
            print(f"[MediVision AI] ResNet-50 model pre-loaded successfully from {chest_xray_service.checkpoint_path}")
        else:
            print("[MediVision AI] ResNet-50 checkpoint unconfigured; continuing in safe not_trained mode.")
    except Exception as exc:
        print(f"[MediVision AI] Note: Model pre-load deferred ({exc})")

    # Verify Skin Lesion model status at application startup
    from app.services.skin_lesion import skin_lesion_service
    try:
        if skin_lesion_service.is_ready():
            print(f"[MediVision AI] Skin Lesion model loaded successfully from {skin_lesion_service.checkpoint_path}")
        else:
            print("[MediVision AI] Skin Lesion checkpoint unconfigured; continuing in safe not_trained mode.")
    except Exception as exc:
        print(f"[MediVision AI] Note: Skin Lesion pre-load deferred ({exc})")

    # Verify MongoDB Atlas connectivity at startup
    from app.db.mongodb import check_mongo_connection
    try:
        is_connected, details = await check_mongo_connection()
        if is_connected:
            print(f"[MediVision AI] MongoDB Atlas initialized: {details}")
        else:
            print(f"[MediVision AI] MongoDB Atlas status: {details}")
    except Exception as exc:
        print(f"[MediVision AI] Note: MongoDB Atlas check deferred ({exc})")

    yield
    # Shutdown actions
    await close_mongo_connection()
    print("[MediVision AI] Backend shut down cleanly.")


def create_application() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description="MediVision AI Research Platform - Stage 2 Multi-Disease AI API",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routers
    app.include_router(health.router, prefix=settings.api_v1_prefix)
    app.include_router(models.router, prefix=settings.api_v1_prefix)
    app.include_router(imaging.router, prefix=settings.api_v1_prefix)
    app.include_router(brain_mri.router, prefix=settings.api_v1_prefix)
    app.include_router(clinical_ml.router, prefix=settings.api_v1_prefix)
    app.include_router(xai.router, prefix=settings.api_v1_prefix)
    app.include_router(yolo.router, prefix=settings.api_v1_prefix)
    app.include_router(skin_lesion.router, prefix=settings.api_v1_prefix)

    @app.get(
        f"{settings.api_v1_prefix}/reports/{{record_id}}",
        summary="Download PDF Report Alias",
        tags=["Reports"],
    )
    async def download_report_alias(record_id: str):
        return await imaging.get_history_pdf_report(record_id)

    @app.post(
        f"{settings.api_v1_prefix}/reports/generate",
        summary="Generate PDF Report Direct Alias",
        tags=["Reports"],
    )
    async def generate_report_alias(payload: dict):
        return await imaging.generate_direct_pdf_report_endpoint(payload)

    @app.get("/")
    async def root():
        return {
            "name": settings.project_name,
            "version": settings.version,
            "stage": 1,
            "status": "online",
            "docs_url": "/docs",
            "health_endpoint": f"{settings.api_v1_prefix}/health",
            "models_endpoint": f"{settings.api_v1_prefix}/models/status",
        }

    return app


app = create_application()
