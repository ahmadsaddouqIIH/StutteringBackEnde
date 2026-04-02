from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from backend.app.middleware import (
    RequestSizeLimitExceeded,
    validate_audio_upload,
)
from backend.services.model_service import (
    InvalidAudioError,
    ModelNotLoadedError,
    ModelService,
    PredictionError,
)

LABEL2ID: dict[str, int] = {
    "Fluent": 0,
    "Blocks": 1,
    "Prolongations": 2,
    "Repetitions": 3,
    "Interjections": 4,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence_scores: dict[str, float]
    processing_time_ms: int
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: int | None = None


class ClassesResponse(BaseModel):
    classes: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[str, str]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: str


def get_model_service(request: Request) -> ModelService:
    model_service = getattr(request.app.state, "model_service", None)
    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error_code="SERVICE_UNAVAILABLE",
                message="Service is not ready",
                detail="Model service is not initialized",
            ).model_dump(),
        )
    return model_service


def get_db_service(request: Request) -> Any:
    return getattr(request.app.state, "db_service", None)


router = APIRouter()

_PROCESS_START_WALL = time.time()


def _uptime_seconds() -> int:
    return int(time.time() - _PROCESS_START_WALL)


async def _log_prediction(db_service: Any, record: dict[str, Any]) -> None:
    if db_service is None:
        return
    for name in ("log_prediction", "create_prediction", "save_prediction"):
        fn = getattr(db_service, name, None)
        if callable(fn):
            result = fn(record)
            if isinstance(result, Awaitable):
                await result
            return


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(
    request: Request,
    audio_file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service),
    db_service: Any = Depends(get_db_service),
) -> PredictionResponse:
    started_at = time.perf_counter()
    max_size_mb = getattr(request.app.state.settings, "MAX_AUDIO_SIZE_MB", 10)
    audio_bytes = await validate_audio_upload(audio_file, max_size_mb)
    request_id = str(uuid.uuid4())
    try:
        prediction = model_service.predict(audio_bytes)
    except InvalidAudioError as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error_code="UNPROCESSABLE_AUDIO",
                message="Audio could not be processed",
                detail=str(exc),
            ).model_dump(),
        ) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error_code="SERVICE_UNAVAILABLE",
                message="Service is not ready",
                detail=str(exc),
            ).model_dump(),
        ) from exc
    except RequestSizeLimitExceeded:
        raise
    except PredictionError as exc:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="MODEL_ERROR",
                message="Inference failed",
                detail=str(exc),
            ).model_dump(),
        ) from exc

    response = PredictionResponse(
        predicted_class=str(prediction["predicted_class"]),
        confidence_scores={k: float(v) for k, v in prediction["confidence_scores"].items()},
        processing_time_ms=int((time.perf_counter() - started_at) * 1000),
        model_version=str(prediction["model_version"]),
        request_id=request_id,
    )
    await _log_prediction(
        db_service,
        {
            "request_id": request_id,
            "predicted_class": response.predicted_class,
            "confidence_scores": response.confidence_scores,
            "processing_time_ms": response.processing_time_ms,
            "model_version": response.model_version,
            "filename": audio_file.filename,
        },
    )
    return response


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    model_service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    model_loaded = bool(model_service.is_loaded())
    version = getattr(request.app.state, "service_version", "0.0.0")
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        version=version,
        uptime_seconds=_uptime_seconds(),
    )


@router.get("/classes", response_model=ClassesResponse)
async def get_classes() -> ClassesResponse:
    return ClassesResponse(
        classes=list(LABEL2ID.keys()),
        label_to_id=LABEL2ID,
        id_to_label={str(k): v for k, v in ID2LABEL.items()},
    )
