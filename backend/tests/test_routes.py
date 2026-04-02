from __future__ import annotations

from pathlib import Path


def test_health_returns_200(client) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_classes_returns_5_labels(client) -> None:
    response = client.get("/api/v1/classes")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["classes"]) == 5
    assert payload["label_to_id"]["Fluent"] == 0
    assert payload["id_to_label"]["0"] == "Fluent"


def test_predict_valid_wav_returns_prediction(client, fixture_wav_path: Path) -> None:
    with fixture_wav_path.open("rb") as fh:
        response = client.post(
            "/api/v1/predict",
            files={"audio_file": ("silence.wav", fh.read(), "audio/wav")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "Fluent"
    assert "confidence_scores" in payload
    assert "request_id" in payload and payload["request_id"]
    assert payload["model_version"] == "mock-v1"


def test_predict_invalid_mime_type_returns_400(client, fixture_wav_path: Path) -> None:
    with fixture_wav_path.open("rb") as fh:
        response = client.post(
            "/api/v1/predict",
            files={"audio_file": ("silence.wav", fh.read(), "audio/mpeg")},
        )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_REQUEST"


def test_predict_missing_file_returns_422(client) -> None:
    response = client.post("/api/v1/predict", files={})
    assert response.status_code == 422


def test_predict_corrupt_audio_returns_422(client) -> None:
    # RIFF prefix passes magic-byte validation, but invalid body fails WAV decode in mock model.
    corrupt = b"RIFF" + b"this-is-not-a-real-wav"
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("corrupt.wav", corrupt, "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "UNPROCESSABLE_AUDIO"


def test_predict_oversized_file_returns_413(client) -> None:
    # Build an oversized-but-WAV-shaped payload to trigger size check instead of magic-byte check.
    oversized = b"RIFF" + (b"\x00" * (2 * 1024 * 1024))
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("big.wav", oversized, "audio/wav")},
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "FILE_TOO_LARGE"
