"""Application settings (env-driven, Docker-friendly)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    port: int = 8000
    upload_dir: Path = Path("/app/data/uploads")
    results_dir: Path = Path("/app/data/results")
    max_upload_mb: int = 50
    # stub | full — full enables optional YOLO for openings/OCR assist;
    # geometry itself prefers drawing-faithful exact-trace vectorization.
    ai_mode: str = "stub"
    vectorization_profile: str = "default"
    vectorization_auto_profile: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
