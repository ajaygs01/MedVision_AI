from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import health, imaging, models
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
