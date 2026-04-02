from __future__ import annotations

import json
import math
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.services.model_service import (
    InvalidAudioError,
    ModelNotLoadedError,
    ModelService,
    PredictionError,
)


def _settings_for_model(model_path: str) -> Settings:
    return Settings(
        MODEL_PATH=model_path,
        MODEL_SOURCE="local",
        DEVICE="cpu",
        PRODUCTION_MODE=False,
        DB_URL="postgresql://test:test@localhost:5432/test",
        SERVICE_VERSION="0.2.0-test",
    )


def _write_artifacts(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact_export"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_name": "facebook/wav2vec2-base-960h",
        "model_version": "adn04-export-v1",
        "sample_rate": 16000,
        "max_samples": 32000,
        "class_names": ["Fluent", "Blocks", "Prolongations", "Repetitions", "Interjections"],
    }
    (artifact_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (artifact_dir / "model_inference.pt").write_bytes(b"placeholder-for-tests")
    return artifact_dir


def _make_wav_bytes(*, sample_rate: int, frequency: float, seconds: float) -> bytes:
    frames = int(sample_rate * seconds)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = []
        for i in range(frames):
            t = i / float(sample_rate)
            value = int(16000 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack("<h", value))
        wf.writeframes(b"".join(samples))
    return buffer.getvalue()


class _FakeProcessor:
    def __call__(self, values, *, sampling_rate: int, return_tensors: str = "pt", padding: bool = True):
        _ = (sampling_rate, return_tensors, padding)
        return {"input_values": values}


class _FakeModel:
    def to(self, device: str) -> _FakeModel:
        _ = device
        return self

    def eval(self) -> _FakeModel:
        return self

    def __call__(self, **inputs):
        values = inputs.get("input_values")
        energy = 0.0
        if isinstance(values, list) and values:
            energy = sum(abs(float(v)) for v in values[:1000]) / min(1000, len(values))
        base = 1.0 + energy
        return {"logits": [base, base + 0.1, base + 0.2, base + 0.3, base + 0.4]}


@pytest.fixture
def patched_model_stack(monkeypatch):
    monkeypatch.setattr(ModelService, "_load_processor", lambda self, cfg: _FakeProcessor())
    monkeypatch.setattr(ModelService, "_load_model", lambda self, model_path, cfg: _FakeModel())


def test_custom_exceptions_importable() -> None:
    assert issubclass(InvalidAudioError, Exception)
    assert issubclass(ModelNotLoadedError, Exception)
    assert issubclass(PredictionError, Exception)


def test_model_service_loads_from_local_artifact_path(tmp_path: Path, patched_model_stack) -> None:
    artifact_dir = _write_artifacts(tmp_path)
    service = ModelService(_settings_for_model(str(artifact_dir)))
    assert service.is_loaded() is True
    assert service.model_version == "adn04-export-v1"


def test_predict_schema_with_three_different_wav_files(tmp_path: Path, patched_model_stack) -> None:
    artifact_dir = _write_artifacts(tmp_path)
    service = ModelService(_settings_for_model(str(artifact_dir)))
    wav_inputs = [
        _make_wav_bytes(sample_rate=16000, frequency=220.0, seconds=0.40),
        _make_wav_bytes(sample_rate=8000, frequency=440.0, seconds=0.65),
        _make_wav_bytes(sample_rate=22050, frequency=880.0, seconds=0.55),
    ]
    for audio in wav_inputs:
        out = service.predict(audio)
        assert set(out.keys()) == {
            "predicted_class",
            "confidence_scores",
            "processing_time_ms",
            "model_version",
        }
        assert out["predicted_class"] in {
            "Fluent",
            "Blocks",
            "Prolongations",
            "Repetitions",
            "Interjections",
        }
        assert isinstance(out["confidence_scores"], dict)
        assert len(out["confidence_scores"]) == 5
        assert out["processing_time_ms"] >= 0
        assert out["model_version"] == "adn04-export-v1"


def test_predict_invalid_audio_raises_invalid_audio_error(tmp_path: Path, patched_model_stack) -> None:
    artifact_dir = _write_artifacts(tmp_path)
    service = ModelService(_settings_for_model(str(artifact_dir)))
    with pytest.raises(InvalidAudioError):
        service.predict(b"not-a-valid-wav")


def test_predict_raises_model_not_loaded_after_shutdown(tmp_path: Path, patched_model_stack) -> None:
    artifact_dir = _write_artifacts(tmp_path)
    service = ModelService(_settings_for_model(str(artifact_dir)))
    service.shutdown()
    wav_bytes = _make_wav_bytes(sample_rate=16000, frequency=350.0, seconds=0.20)
    with pytest.raises(ModelNotLoadedError):
        service.predict(wav_bytes)


def test_predict_is_thread_safe_under_parallel_calls(tmp_path: Path, patched_model_stack) -> None:
    artifact_dir = _write_artifacts(tmp_path)
    service = ModelService(_settings_for_model(str(artifact_dir)))
    wav_bytes = _make_wav_bytes(sample_rate=16000, frequency=500.0, seconds=0.30)

    def _call() -> dict:
        return service.predict(wav_bytes)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _call(), range(20)))

    assert len(results) == 20
    assert all("predicted_class" in r for r in results)


def test_runtime_failure_is_wrapped_as_prediction_error(
    tmp_path: Path, patched_model_stack, monkeypatch
) -> None:
    artifact_dir = _write_artifacts(tmp_path)
    service = ModelService(_settings_for_model(str(artifact_dir)))

    def _boom(self, inputs):
        _ = inputs
        raise RuntimeError("inference exploded")

    monkeypatch.setattr(ModelService, "_forward", _boom)
    wav_bytes = _make_wav_bytes(sample_rate=16000, frequency=600.0, seconds=0.25)
    with pytest.raises(PredictionError):
        service.predict(wav_bytes)
