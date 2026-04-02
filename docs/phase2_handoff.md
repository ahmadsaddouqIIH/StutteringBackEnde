# Phase 2 Backend Handoff

This document summarizes the current backend implementation and the recommended Phase 2 roadmap.

## 1) Local Setup

### Prerequisites
- Python 3.12+
- Virtual environment (`.venv`)
- Dependencies installed from:
  - `requirements.txt` (runtime)
  - `requirements-dev.txt` (tests)

### Required environment variables
Create `.env` from `.env.example` and set the following:
- `MODEL_PATH` - path to model artifact directory (expects `model_inference.pt` and `config.json`)
- `MODEL_SOURCE` - currently `local` supported for production path
- `DEVICE` - `cpu` or `cuda`
- `MAX_AUDIO_SIZE_MB` - upload size limit
- `ALLOWED_ORIGINS` - CORS origins list
- `PRODUCTION_MODE` - `true` or `false`
- `DB_URL` - database URL
- `SERVICE_VERSION` - API/service version string
- `LOG_LEVEL` - `DEBUG|INFO|WARNING|ERROR|CRITICAL`

### Startup command
From repository root:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Expected output
- Uvicorn startup logs
- App startup log (`app_startup`)
- Model load log (`model_loaded`)
- Ready log (`model_ready`)
- `GET /health` returns `200` with:
  - `status: "ok"`
  - `model_loaded: true`
  - `version`
  - `uptime_seconds`

## 2) Model Artifact Swap (No Code Changes)

Use this process to replace the deployed model without editing Python code:

1. Prepare a new export artifact directory containing:
   - `model_inference.pt`
   - `config.json`
2. Ensure `config.json` includes at minimum:
   - `model_name` (for `Wav2Vec2Processor`)
   - class labels (`class_names` or equivalent)
   - optional `model_version`, `sample_rate`, `max_samples`
3. Copy the directory to the deployment host (for example: `/opt/stuttering/model_v2/`).
4. Update `.env`:
   - `MODEL_PATH=/opt/stuttering/model_v2/`
   - optionally update `SERVICE_VERSION`
5. Restart the backend process.
6. Validate:
   - `GET /health` is `200`
   - `POST /api/v1/predict` returns valid response schema
   - logs show model loaded successfully.

## 3) Known Limitations

- Single file inference only (no batch input payloads).
- Synchronous inference path (request thread is blocked until inference completes).
- No server-side streaming responses for prediction progress.
- No dedicated model version registry endpoint.
- No async queue/worker for long-running inference workloads.

## 4) Planned Phase 2 Improvements

### Async inference execution
- Introduce task-based inference with either:
  - Celery + worker queue, or
  - FastAPI `BackgroundTasks` (lightweight option)
- Add task status endpoints and timeout/retry policy.

### Model versioning endpoint
- Add `GET /models` to expose:
  - active model version
  - available versions
  - metadata (created date, checksum, source)

### Batch prediction endpoint
- Add batch upload interface (multiple files per request).
- Return per-file `request_id`, prediction, and failure details.

### Mobile authentication integration
- Integrate with mobile app auth layer (token validation/JWT middleware).
- Associate predictions to authenticated user context.

## 5) API Endpoint Reference Summary

### `GET /health`
- Health probe for app and model readiness.
- Response schema: `HealthResponse` (`status`, `model_loaded`, `version`, `uptime_seconds`).

### `GET /api/v1/classes`
- Returns class labels and bidirectional label/id mapping.
- Response schema: `ClassesResponse` (`classes`, `label_to_id`, `id_to_label`).

### `POST /api/v1/predict`
- Accepts one WAV file and returns model prediction.
- Response schema: `PredictionResponse` (`predicted_class`, `confidence_scores`, `processing_time_ms`, `model_version`, `request_id`).

## 6) Open Risks and Questions for Phase 2

- Cold-start latency for model load after deploy/restart.
- Memory pressure under concurrent requests (especially with larger models and `cuda`).
- Throughput bottleneck from synchronous inference path.
- Audio format diversity from mobile clients (header mismatch, malformed WAV, sample-rate variance).
- DB persistence growth strategy for high-volume prediction logs.
- Error response consistency if new background task architecture is introduced.

---

**Review requirement before merge:** Wael approval via PR comment is required.
