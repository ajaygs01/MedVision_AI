import json
import os
from functools import lru_cache
from typing import List, Union
from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    has_pydantic_settings = True
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = None
    has_pydantic_settings = False


from pathlib import Path
_backend_env = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    project_name: str = "MediVision AI Research Platform"
    version: str = "1.0.0-stage1"
    api_v1_prefix: str = "/api"
    port: int = 8000
    host: str = "0.0.0.0"
    environment: str = "development"
    cors_origins: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # MongoDB Atlas
    mongodb_uri: str = Field(default="", description="MongoDB Atlas Connection URI")
    mongodb_db_name: str = "medivision_ai"
    mongodb_connect_timeout_ms: int = 2500

    # Brain MRI Model Configuration
    brain_mri_experimental_mode: bool = Field(
        default=False,
        description="Explicit opt-in to load experimental 10p/3e Brain MRI prototype weights",
    )
    brain_mri_custom_checkpoint_path: str = Field(
        default="",
        description="Path to experimental checkpoint if overriding default custom_10p_3e/unet3d_best.pth",
    )

    if has_pydantic_settings and SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=(_backend_env, ".env", "backend/.env"),
            env_file_encoding="utf-8",
            extra="ignore",
        )
    else:
        class Config:
            env_file = str(_backend_env)
            extra = "ignore"

    def get_cors_origins(self) -> List[str]:
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        if isinstance(self.cors_origins, str):
            try:
                parsed = json.loads(self.cors_origins)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return ["*"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
