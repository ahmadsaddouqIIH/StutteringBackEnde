from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment (and optional `.env`)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    MODEL_PATH: str = Field(
        default="",
        description="Filesystem path to model artifact or local cache path for S3-backed models.",
    )
    MODEL_SOURCE: Literal["local", "s3"] = "local"
    DEVICE: Literal["cpu", "cuda"] = "cpu"
    MAX_AUDIO_SIZE_MB: int = 10
    # Stored as str so env / .env can use comma-separated values (pydantic-settings JSON-decodes list fields).
    allowed_origins_raw: str = Field(
        default="",
        validation_alias="ALLOWED_ORIGINS",
    )
    PRODUCTION_MODE: bool = False
    DB_URL: str = Field(
        default="postgresql://user:password@localhost:5432/stuttering_ai",
        description="Database URL (placeholder default for local dev).",
    )
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    @field_validator("allowed_origins_raw", mode="before")
    @classmethod
    def coerce_allowed_origins_raw(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ",".join(str(x).strip() for x in value if str(x).strip())
        if isinstance(value, str):
            return value.strip()
        raise TypeError("ALLOWED_ORIGINS must be a list or string")

    @staticmethod
    def _split_origins(raw: str) -> list[str]:
        if not raw.strip():
            return []
        if raw.strip().startswith("["):
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("ALLOWED_ORIGINS JSON must be a list of strings")
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [part.strip() for part in raw.split(",") if part.strip()]

    @computed_field
    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return self._split_origins(self.allowed_origins_raw)

    @field_validator("LOG_LEVEL")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        upper = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return upper

    @model_validator(mode="after")
    def validate_cors_wildcard(self) -> Settings:
        if self.PRODUCTION_MODE and "*" in self.ALLOWED_ORIGINS:
            raise ValueError(
                "ALLOWED_ORIGINS must not contain '*' when PRODUCTION_MODE is true"
            )
        return self

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Origins passed to CORSMiddleware (never `['*']` in production)."""
        if self.PRODUCTION_MODE:
            return [o for o in self.ALLOWED_ORIGINS if o != "*"]
        return list(self.ALLOWED_ORIGINS)

    @property
    def max_audio_size_bytes(self) -> int:
        return self.MAX_AUDIO_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
