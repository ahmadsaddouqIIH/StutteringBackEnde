from __future__ import annotations

import io
import math
import struct
import wave
from tempfile import SpooledTemporaryFile
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

from backend.api.routes import router
from backend.app.config import Settings
from backend.app.middleware import (
    InvalidAudioMagicBytesError,
    InvalidAudioMimeTypeError,
    RequestLoggingMiddleware,
    RequestSizeLimitExceeded,
    register_exception_handlers,
    validate_audio_upload,
)


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


def _upload_file(data: bytes, content_type: str, content_length: str | None = None) -> UploadFile:
    fp = SpooledTemporaryFile()
    fp.write(data)
    fp.seek(0)
    hdr = {"content-type": content_type}
    if content_length is not None:
        hdr["content-length"] = content_length
    return UploadFile(file=fp, filename="sample.wav", headers=Headers(hdr))


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


def _make_app(production: bool = False, max_mb: int = 1) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    settings = Settings(DB_URL="postgresql://x", PRODUCTION_MODE=production, MAX_AUDIO_SIZE_MB=max_mb)
    app.state.model_service = _ModelOK()
    app.state.service_version = "0.1.0-test"
    app.state.settings = settings
    return app


@pytest.mark.anyio
async def test_validate_audio_upload_rejects_bad_mime() -> None:
    upload = _upload_file(_wav_bytes(), "audio/mpeg")
    with pytest.raises(InvalidAudioMimeTypeError):
        await validate_audio_upload(upload, max_size_mb=1)


@pytest.mark.anyio
async def test_validate_audio_upload_rejects_bad_magic_bytes_disguised_wav() -> None:
    fake_wav = b"NOTW" + (b"\x00" * 64)
    upload = _upload_file(fake_wav, "audio/wav")
    with pytest.raises(InvalidAudioMagicBytesError):
        await validate_audio_upload(upload, max_size_mb=1)


@pytest.mark.anyio
async def test_validate_audio_upload_rejects_declared_oversize() -> None:
    upload = _upload_file(_wav_bytes(), "audio/wav", content_length=str(99 * 1024 * 1024))
    with pytest.raises(RequestSizeLimitExceeded):
        await validate_audio_upload(upload, max_size_mb=1)


def test_request_size_limit_handler_maps_to_413() -> None:
    app = _make_app(production=False, max_mb=1)
    huge = _wav_bytes(sec=70.0)
    with TestClient(app) as c:
        r = c.post("/api/v1/predict", files={"audio_file": ("sample.wav", huge, "audio/wav")})
    assert r.status_code == 413
    assert r.json()["error_code"] == "FILE_TOO_LARGE"


def test_production_error_response_hides_internal_detail() -> None:
    app = _make_app(production=True)
    with TestClient(app) as c:
        r = c.post("/api/v1/predict", files={"audio_file": ("bad.mp3", b"abc", "audio/mpeg")})
    assert r.status_code == 400
    body = r.json()
    assert body["error_code"] == "INVALID_REQUEST"
    assert "detail" not in body


def test_request_logging_middleware_emits_log_line(caplog) -> None:
    import logging

    caplog.set_level(logging.INFO)
    app = _make_app(production=False)
    with TestClient(app) as c:
        c.get("/api/v1/health", headers={"x-request-id": "req-123"})
    records = [r for r in caplog.records if r.name == "backend.app.middleware"]
    assert records
    structured = [r.msg for r in records if isinstance(r.msg, dict)]
    assert any(m.get("event") == "request_processed" and m.get("request_id") == "req-123" for m in structured)
