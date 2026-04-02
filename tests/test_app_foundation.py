"""Acceptance and requirement tests for FastAPI foundation (main + config)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.config import Settings, clear_settings_cache, get_settings
from backend.app.main import create_app
from backend.services.model_service import ModelNotLoadedError, ModelService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _minimal_dev_settings(**overrides) -> Settings:
    base = dict(
        MODEL_PATH="",
        MODEL_SOURCE="local",
        DEVICE="cpu",
        MAX_AUDIO_SIZE_MB=10,
        ALLOWED_ORIGINS=["http://localhost:3000"],
        PRODUCTION_MODE=False,
        DB_URL="postgresql://test:test@localhost:5432/test",
        SERVICE_VERSION="9.9.9-test",
        LOG_LEVEL="INFO",
    )
    base.update(overrides)
    return Settings(**base)


def _write_minimal_artifacts(base_dir: Path) -> Path:
    artifact_dir = base_dir / "foundation_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": "facebook/wav2vec2-base-960h",
                "model_version": "0.0.0-test",
                "sample_rate": 16000,
                "max_samples": 16000,
                "class_names": [
                    "Fluent",
                    "Blocks",
                    "Prolongations",
                    "Repetitions",
                    "Interjections",
                ],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model_inference.pt").write_bytes(b"placeholder")
    return artifact_dir


@pytest.fixture(autouse=True)
def reset_settings_cache(monkeypatch):
    """Avoid cross-test pollution of cached settings when env is patched."""
    clear_settings_cache()
    yield
    clear_settings_cache()


class TestRequirementsChecklist:
    """Maps to task description and acceptance criteria."""

    def test_fastapi_title_version_description(self) -> None:
        settings = _minimal_dev_settings(SERVICE_VERSION="1.2.3")
        app = create_app(settings)
        assert app.title == "Stuttering AI API"
        assert app.version == "1.2.3"
        assert "api_contract" in (app.description or "").lower()

    def test_router_mounted_at_api_v1(self) -> None:
        settings = _minimal_dev_settings()
        app = create_app(settings)
        routes = {getattr(r, "path", None) for r in app.routes}
        assert "/api/v1/health" in routes
        assert "/health" in routes

    def test_get_health_200_json_contract_shape(self) -> None:
        settings = _minimal_dev_settings()
        app = create_app(settings)
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["model_loaded"] is True
            assert body["version"] == settings.SERVICE_VERSION
            assert isinstance(body["uptime_seconds"], int)
            assert body["uptime_seconds"] >= 0

    def test_get_api_v1_health_matches_root(self) -> None:
        settings = _minimal_dev_settings()
        app = create_app(settings)
        with TestClient(app) as client:
            a = client.get("/health").json()
            b = client.get("/api/v1/health").json()
            assert a == b

    def test_model_loaded_in_lifespan_and_shutdown_releases(self) -> None:
        settings = _minimal_dev_settings()
        app = create_app(settings)
        with TestClient(app) as client:
            assert client.app.state.model_service is not None
            assert client.app.state.model_service.is_loaded() is True
            client.get("/health")
        assert app.state.model_service is None

    def test_runtime_error_returns_500_error_schema(self) -> None:
        settings = _minimal_dev_settings()
        app = create_app(settings)

        @app.get("/trigger-runtime-error")
        def trigger() -> None:
            raise RuntimeError("deliberate")

        with TestClient(app) as client:
            r = client.get("/trigger-runtime-error")
        assert r.status_code == 500
        data = r.json()
        assert data["error_code"] == "MODEL_ERROR"
        assert "message" in data
        assert "detail" in data
        assert data["detail"] == "deliberate"

    def test_runtime_error_hides_detail_in_production_mode(self, monkeypatch, tmp_path) -> None:
        artifact_dir = _write_minimal_artifacts(tmp_path)
        monkeypatch.setattr(ModelService, "_load_processor", lambda self, cfg: object())
        monkeypatch.setattr(ModelService, "_load_model", lambda self, model_path, cfg: object())
        settings = _minimal_dev_settings(PRODUCTION_MODE=True, MODEL_PATH=str(artifact_dir))
        app = create_app(settings)

        @app.get("/trigger-runtime-error")
        def trigger() -> None:
            raise RuntimeError("secret")

        monkeypatch.setenv("PRODUCTION_MODE", "true")
        clear_settings_cache()

        with TestClient(app) as client:
            r = client.get("/trigger-runtime-error")
        assert r.status_code == 500
        assert "detail" not in r.json()

    def test_cors_uses_config_origins(self) -> None:
        settings = _minimal_dev_settings(
            ALLOWED_ORIGINS=["http://allowed.example", "http://other.example"],
        )
        app = create_app(settings)
        with TestClient(app) as client:
            r = client.options(
                "/health",
                headers={
                    "Origin": "http://allowed.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert r.status_code in (200, 400)
        assert r.headers.get("access-control-allow-origin") == "http://allowed.example"

    def test_cors_no_wildcard_in_production_settings(self) -> None:
        with pytest.raises(ValidationError):
            Settings(
                MODEL_PATH="",
                MODEL_SOURCE="local",
                DEVICE="cpu",
                ALLOWED_ORIGINS=["*"],
                PRODUCTION_MODE=True,
                DB_URL="postgresql://x",
            )

    def test_cors_wildcard_allowed_in_dev(self) -> None:
        s = Settings(
            MODEL_PATH="",
            MODEL_SOURCE="local",
            DEVICE="cpu",
            ALLOWED_ORIGINS=["*"],
            PRODUCTION_MODE=False,
            DB_URL="postgresql://x",
        )
        assert "*" in s.cors_allowed_origins

    def test_settings_reads_allow_origins_comma_separated(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "ALLOWED_ORIGINS",
            "http://a.example, http://b.example",
        )
        clear_settings_cache()
        s = get_settings()
        assert s.ALLOWED_ORIGINS == ["http://a.example", "http://b.example"]

    def test_settings_reads_allow_origins_json_list(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "ALLOWED_ORIGINS",
            '["http://one.example", "http://two.example"]',
        )
        clear_settings_cache()
        s = get_settings()
        assert s.ALLOWED_ORIGINS == ["http://one.example", "http://two.example"]

    def test_production_local_missing_model_startup_fails(self) -> None:
        settings = _minimal_dev_settings(
            PRODUCTION_MODE=True,
            MODEL_PATH=str(PROJECT_ROOT / "nonexistent_model_artifact.bin"),
        )
        app = create_app(settings)
        with pytest.raises(ModelNotLoadedError):
            with TestClient(app):
                pass

    def test_env_example_committed_no_real_secrets(self) -> None:
        example = PROJECT_ROOT / ".env.example"
        assert example.is_file()
        text = example.read_text(encoding="utf-8")
        assert "CHANGE_ME" in text or "example" in text.lower()
        assert "sk-" not in text
        assert "AKIA" not in text


class TestLogging:
    def test_structlog_records_contain_event_and_timestamp(self, caplog) -> None:
        """structlog + ProcessorFormatter attach structured fields (JSON on real stdout)."""
        import logging

        caplog.set_level(logging.INFO)
        settings = _minimal_dev_settings(LOG_LEVEL="INFO")
        app = create_app(settings)
        with TestClient(app) as client:
            client.get("/health")
        ours = [r for r in caplog.records if r.name == "backend.app.main"]
        assert ours, "expected structlog-backed records from backend.app.main"
        payloads = [r.msg for r in ours if isinstance(r.msg, dict)]
        assert payloads, "expected dict payloads on LogRecord.msg from structlog"
        assert all("event" in p and "timestamp" in p for p in payloads)


class TestUvicornLaunch:
    def test_uvicorn_import_string_resolves_app(self) -> None:
        """`uvicorn backend.app.main:app` requires `app` importable from repo root."""
        code = (
            "from backend.app.main import app; "
            "assert app.title == 'Stuttering AI API'; "
            "assert app.version"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
        )


class TestConfigFields:
    def test_max_audio_size_mb_default_and_bytes(self) -> None:
        s = Settings(
            MODEL_PATH="",
            DB_URL="postgresql://x",
            MAX_AUDIO_SIZE_MB=10,
        )
        assert s.MAX_AUDIO_SIZE_MB == 10
        assert s.max_audio_size_bytes == 10 * 1024 * 1024

    def test_model_source_device_literal(self) -> None:
        s = Settings(MODEL_PATH="", DB_URL="postgresql://x", MODEL_SOURCE="local", DEVICE="cpu")
        assert s.MODEL_SOURCE == "local"
        assert s.DEVICE == "cpu"
