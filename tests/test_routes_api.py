from __future__ import annotations

import io
import math
import struct
import wave
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import router
from backend.app.config import Settings
from backend.app.middleware import RequestLoggingMiddleware, register_exception_handlers
from backend.services.model_service import InvalidAudioError, ModelNotLoadedError, PredictionError


def _wav_bytes(sr: int = 16000, hz: float = 440.0, sec: float = 0.2) -> bytes:
    count = int(sr * sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = []
        for i in range(count):
            value = int(12000 * math.sin(2 * math.pi * hz * (i / sr)))
            frames.append(struct.pack("<h", value))
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


class _DBRecorder:
    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []

    def log_prediction(self, row: dict[str, Any]) -> None:
        self.logged.append(row)


class _ModelOK:
    def is_loaded(self) -> bool:
        return True

    def predict(self, audio_bytes: bytes) -> dict[str, Any]:
        _ = audio_bytes
        return {
            "predicted_class": "Fluent",
            "confidence_scores": {
                "Fluent": 0.9,
                "Blocks": 0.02,
                "Prolongations": 0.02,
                "Repetitions": 0.03,
                "Interjections": 0.03,
            },
            "processing_time_ms": 1,
            "model_version": "m1",
        }


def _make_app(model_service: Any, db_service: Any | None = None, max_mb: int = 1) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.state.model_service = model_service
    app.state.db_service = db_service
    app.state.service_version = "0.1.0-test"
    app.state.settings = Settings(DB_URL="postgresql://x", MAX_AUDIO_SIZE_MB=max_mb)
    return app


def test_health_and_classes_endpoints() -> None:
    app = _make_app(_ModelOK())
    with TestClient(app) as c:
        h = c.get("/api/v1/health")
        assert h.status_code == 200
        body = h.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["version"] == "0.1.0-test"
        cls = c.get("/api/v1/classes")
        assert cls.status_code == 200
        cls_body = cls.json()
        assert cls_body["label_to_id"]["Fluent"] == 0
        assert cls_body["id_to_label"]["0"] == "Fluent"


def test_predict_valid_wav_returns_prediction_and_logs_db() -> None:
    db = _DBRecorder()
    app = _make_app(_ModelOK(), db_service=db)
    audio = _wav_bytes()
    with TestClient(app) as c:
        r = c.post(
            "/api/v1/predict",
            files={"audio_file": ("sample.wav", audio, "audio/wav")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "Fluent"
    assert "request_id" in body and body["request_id"]
    assert isinstance(body["processing_time_ms"], int)
    assert db.logged and db.logged[0]["predicted_class"] == "Fluent"


def test_predict_invalid_file_type_returns_400() -> None:
    app = _make_app(_ModelOK())
    with TestClient(app) as c:
        r = c.post(
            "/api/v1/predict",
            files={"audio_file": ("bad.mp3", b"abc", "audio/mpeg")},
        )
    assert r.status_code == 400
    detail = r.json()
    assert detail["error_code"] == "INVALID_REQUEST"


def test_predict_oversized_file_returns_413() -> None:
    app = _make_app(_ModelOK(), max_mb=1)
    huge = _wav_bytes(sec=70.0)
    with TestClient(app) as c:
        r = c.post(
            "/api/v1/predict",
            files={"audio_file": ("sample.wav", huge, "audio/wav")},
        )
    assert r.status_code == 413
    assert r.json()["error_code"] == "FILE_TOO_LARGE"


def test_predict_exception_mapping() -> None:
    class _ModelBad:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        def is_loaded(self) -> bool:
            return True

        def predict(self, audio_bytes: bytes) -> dict[str, Any]:
            _ = audio_bytes
            raise self.exc

    cases = [
        (InvalidAudioError("bad"), 422, "UNPROCESSABLE_AUDIO"),
        (ModelNotLoadedError("down"), 503, "SERVICE_UNAVAILABLE"),
        (PredictionError("oops"), 500, "MODEL_ERROR"),
    ]
    for exc, code, err in cases:
        app = _make_app(_ModelBad(exc))
        with TestClient(app) as c:
            r = c.post(
                "/api/v1/predict",
                files={"audio_file": ("sample.wav", _wav_bytes(), "audio/wav")},
            )
        assert r.status_code == code
        assert r.json()["error_code"] == err
