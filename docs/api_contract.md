# Stuttering Inference Service — REST API Contract

**Document version:** 1.0  
**Status:** Draft — Pending review (Adan: output format alignment; Wael: DB schema alignment)  
**Related task:** SDQ-02 (implementation begins after this contract is approved)

This document defines the complete REST API contract for the stuttering inference service. No implementation code should be written until this contract is reviewed and approved by the designated reviewers.

---

## 1. Base URL and Conventions

- **Base path:** `/` (all endpoints are relative to the service root).
- **Content negotiation:** Success responses use `Content-Type: application/json`. Error responses use `Content-Type: application/json`.
- **Character encoding:** UTF-8.

---

## 2. Endpoints

### 2.1 POST /predict

Submit an audio file for stuttering classification. The model returns a predicted class and per-class confidence scores.

#### Request

| Aspect | Specification |
|--------|----------------|
| **Method** | `POST` |
| **Content-Type** | `multipart/form-data` |
| **Body** | Form data (see below) |

**Form fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_file` | binary (file) | **Yes** | The audio file to classify. Must be WAV. |
| `sample_rate_hint` | integer (form field) | **Yes** | Expected sample rate in Hz (e.g. `16000`). Server may use this for validation or resampling. |

**Request constraints:**

| Constraint | Rule | Default / Notes |
|------------|------|------------------|
| **Accepted formats** | File must be WAV. Accepted: filename extension `.wav` and `Content-Type: audio/wav` (or `audio/wave`). | Server may reject other types with 415. |
| **Max file size** | Configurable maximum size in bytes. | Default: **10 MB** (10,485,760 bytes). Larger files → 413. |
| **Sample rate hint** | Must be present and represent a valid positive integer (Hz). | Invalid or missing → 400. |

**Example request (conceptual):**

```http
POST /predict HTTP/1.1
Host: inference.example.com
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="audio_file"; filename="recording.wav"
Content-Type: audio/wav

<binary data>
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="sample_rate_hint"

16000
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

#### Success response (200 OK)

**Body schema:**

```json
{
  "predicted_class": "string",
  "confidence_scores": {
    "Fluent": 0.0,
    "Blocks": 0.0,
    "Prolongations": 0.0,
    "Repetitions": 0.0,
    "Interjections": 0.0
  },
  "processing_time_ms": 0,
  "model_version": "string",
  "request_id": "string"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted_class` | string | One of: `"Fluent"`, `"Blocks"`, `"Prolongations"`, `"Repetitions"`, `"Interjections"`. |
| `confidence_scores` | object | Maps each class name to a probability in [0.0, 1.0]. Sum should be 1.0. |
| `processing_time_ms` | integer | Time spent in model inference (and any mandatory preprocessing), in milliseconds. ≥ 0. |
| `model_version` | string | Identifier of the model version that produced the prediction (e.g. semver or commit hash). |
| `request_id` | string | Unique id for this request (e.g. UUID). For logging and correlation. |

**Example:**

```json
{
  "predicted_class": "Fluent",
  "confidence_scores": {
    "Fluent": 0.92,
    "Blocks": 0.02,
    "Prolongations": 0.01,
    "Repetitions": 0.03,
    "Interjections": 0.02
  },
  "processing_time_ms": 45,
  "model_version": "1.0.0",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**HTTP status:** `200 OK`

#### Error responses (POST /predict)

All error responses use the [Standard error response schema](#4-standard-error-response-schema).

| HTTP status | error_code (example) | When to use |
|-------------|----------------------|-------------|
| **400 Bad Request** | `INVALID_REQUEST` | Missing `audio_file` or `sample_rate_hint`; invalid or non-numeric `sample_rate_hint`; malformed form data. |
| **413 Payload Too Large** | `FILE_TOO_LARGE` | File size exceeds configured max (e.g. default 10 MB). |
| **415 Unsupported Media Type** | `UNSUPPORTED_MEDIA_TYPE` | File is not WAV (wrong extension or Content-Type). |
| **422 Unprocessable Entity** | `UNPROCESSABLE_AUDIO` | File is valid WAV but cannot be processed (e.g. duration too short/long, corrupt frames, unsupported sample rate after validation). |
| **500 Internal Server Error** | `MODEL_ERROR` | Inference or model loading failed (e.g. runtime error, corrupted model). |

---

### 2.2 GET /health

Health check for the service and the loaded model.

#### Request

| Aspect | Specification |
|--------|----------------|
| **Method** | `GET` |
| **Body** | None |

#### Success response (200 OK)

**Body schema:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "string",
  "uptime_seconds": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Literal `"ok"` when the service is healthy. |
| `model_loaded` | boolean | `true` if the inference model is loaded and ready; `false` otherwise. |
| `version` | string | Service or app version (e.g. semver or git tag). |
| `uptime_seconds` | integer | Seconds since the process started. ≥ 0. |

**Example:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

**HTTP status:** `200 OK`

#### Error responses (GET /health)

All error responses use the [Standard error response schema](#4-standard-error-response-schema).

| HTTP status | error_code (example) | When to use |
|-------------|----------------------|-------------|
| **503 Service Unavailable** | `SERVICE_UNAVAILABLE` | Service is not ready (e.g. model not loaded, dependency down). |

---

### 2.3 GET /classes

Returns the set of output classes and their optional integer IDs (for DB or downstream alignment).

#### Request

| Aspect | Specification |
|--------|----------------|
| **Method** | `GET` |
| **Body** | None |

#### Success response (200 OK)

**Body schema:**

```json
{
  "classes": ["string"],
  "label_to_id": {
    "string": 0
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `classes` | array of strings | Ordered list of class names. Same order as used in `confidence_scores` and `predicted_class`. |
| `label_to_id` | object | Map from class label (string) to a stable integer ID (e.g. for database schema). Keys = `classes`; IDs are non-negative and unique. |

**Example:**

```json
{
  "classes": ["Fluent", "Blocks", "Prolongations", "Repetitions", "Interjections"],
  "label_to_id": {
    "Fluent": 0,
    "Blocks": 1,
    "Prolongations": 2,
    "Repetitions": 3,
    "Interjections": 4
  }
}
```

**HTTP status:** `200 OK`

#### Error responses (GET /classes)

All error responses use the [Standard error response schema](#4-standard-error-response-schema).

| HTTP status | error_code (example) | When to use |
|-------------|----------------------|-------------|
| **500 Internal Server Error** | `MODEL_ERROR` or `INTERNAL_ERROR` | Server failed to build or load class metadata. |

---

## 3. HTTP Status Code Summary

| Code | Meaning | Used in |
|------|---------|--------|
| **200** | Success | POST /predict, GET /health, GET /classes |
| **400** | Bad Request — invalid format, missing/invalid parameters, malformed form | POST /predict |
| **413** | Payload Too Large — file exceeds max size | POST /predict |
| **415** | Unsupported Media Type — not WAV | POST /predict |
| **422** | Unprocessable Entity — audio not processable | POST /predict |
| **500** | Internal Server Error — model or server error | POST /predict, GET /classes |
| **503** | Service Unavailable — health check failing | GET /health |

All five required error cases from the contract are covered: **400** (invalid format), **413** (file too large), **415** (unsupported media type), **422** (unprocessable audio), **500** (model error); **503** is used for unhealthy service on GET /health.

---

## 4. Standard Error Response Schema

All 4xx and 5xx responses MUST use this JSON schema.

**Body schema:**

```json
{
  "error_code": "string",
  "message": "string",
  "detail": "string"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `error_code` | string | Machine-readable code (e.g. `INVALID_REQUEST`, `FILE_TOO_LARGE`, `UNSUPPORTED_MEDIA_TYPE`, `UNPROCESSABLE_AUDIO`, `MODEL_ERROR`). |
| `message` | string | Short human-readable summary. |
| `detail` | string | Optional extra context (e.g. validation failure reason, max size in bytes). Can be empty string if not applicable. |

**Content-Type:** `application/json`

**Example (413):**

```json
{
  "error_code": "FILE_TOO_LARGE",
  "message": "Audio file exceeds maximum allowed size",
  "detail": "Max size: 10485760 bytes (10 MB)"
}
```

**Example (422):**

```json
{
  "error_code": "UNPROCESSABLE_AUDIO",
  "message": "Audio could not be processed",
  "detail": "Duration too short: 0.2 s (minimum 0.5 s)"
}
```

**Example (503 — GET /health when unhealthy):**

```json
{
  "error_code": "SERVICE_UNAVAILABLE",
  "message": "Service is not ready",
  "detail": "Model not loaded"
}
```

---

## 5. Edge Cases and Validation

Implementations MUST handle the following in line with this contract:

| Case | Handling | Status |
|------|-----------|--------|
| Missing `audio_file` | Reject with 400, `error_code` e.g. `INVALID_REQUEST`. | 400 |
| Empty file (0 bytes) | Reject with 400 or 422; prefer 422 `UNPROCESSABLE_AUDIO` if format is valid. | 400 / 422 |
| Missing `sample_rate_hint` | Reject with 400, `INVALID_REQUEST`. | 400 |
| Non-integer or invalid `sample_rate_hint` | Reject with 400, `INVALID_REQUEST`. | 400 |
| File size > configured max | Reject with 413, `FILE_TOO_LARGE`. | 413 |
| Wrong file type (e.g. .mp3, wrong Content-Type) | Reject with 415, `UNSUPPORTED_MEDIA_TYPE`. | 415 |
| Valid WAV but unprocessable (corrupt, bad duration, etc.) | Reject with 422, `UNPROCESSABLE_AUDIO`. | 422 |
| Model crash or load failure during inference | Reject with 500, `MODEL_ERROR`. | 500 |
| Request not `multipart/form-data` for POST /predict | Reject with 400 or 415; recommend 415 if body is not form-data. | 400 / 415 |
| Multiple `audio_file` parts | Server may use first part only; contract does not require multiple-file support. | — |

---

## 6. Configuration

| Setting | Description | Default |
|--------|-------------|---------|
| `max_file_size_bytes` | Maximum allowed size for `audio_file` (bytes). | 10,485,760 (10 MB) |
| Accepted MIME types for audio | Allowed for `audio_file`. | `audio/wav`, `audio/wave` |
| Accepted filename extension | Allowed for upload. | `.wav` |

---

## 7. Review Checklist (Before SDQ-02)

- [ ] **Adan (AI / output format):** Output format alignment — `predicted_class`, `confidence_scores` keys and order, and `GET /classes` match the trained model’s labels and IDs. Approve via PR comment.
- [ ] **Wael (Backend & DB):** DB schema alignment — `label_to_id` and class names align with database enums/tables; `request_id` and response shape support persistence. Approve via PR comment.
- [ ] Contract covers all three endpoints: POST /predict, GET /health, GET /classes.
- [ ] Error schema and status codes (400, 413, 415, 422, 500, 503) are defined and accepted.
- [ ] Document committed to the repo (e.g. in `docs/api_contract.md`).

---

**End of contract.** Implementation (SDQ-02) should start only after both reviewers have approved this document.
