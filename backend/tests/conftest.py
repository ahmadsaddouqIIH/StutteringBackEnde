from __future__ import annotations

import io
import struct
import sys
import wave
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.routes import router
from backend.app.config import Settings
from backend.app.middleware import RequestLoggingMiddleware, register_exception_handlers
from backend.services.model_service import InvalidAudioError


class MockModelService:
    """Fast mock service: fixed output, validates WAV decode."""

    def is_loaded(self) -> bool:
        return True

    def predict(self, audio_bytes: bytes) -> dict[str, Any]:
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                wav_file.getparams()
                wav_file.readframes(min(32, wav_file.getnframes()))
        except wave.Error as exc:
            raise InvalidAudioError(f"Corrupt or undecodable WAV: {exc}") from exc
        return {
            "predicted_class": "Fluent",
            "confidence_scores": {
                "Fluent": 0.91,
                "Blocks": 0.02,
                "Prolongations": 0.02,
                "Repetitions": 0.03,
                "Interjections": 0.02,
            },
            "processing_time_ms": 5,
            "model_version": "mock-v1",
        }


@pytest.fixture
def test_app() -> FastAPI:
    app = FastAPI(title="Stuttering AI API (integration tests)")
    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.state.model_service = MockModelService()
    app.state.db_service = None
    app.state.service_version = "0.1.0-test"
    app.state.settings = Settings(
        DB_URL="postgresql://test:test@localhost:5432/test",
        PRODUCTION_MODE=False,
        MAX_AUDIO_SIZE_MB=1,
    )
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    with TestClient(test_app) as tc:
        yield tc


@pytest.fixture
def fixture_wav_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "silence_1s_16k_mono.wav"
