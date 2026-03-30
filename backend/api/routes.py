from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter()

_PROCESS_START_WALL = time.time()


def _uptime_seconds() -> int:
    return int(time.time() - _PROCESS_START_WALL)


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    model_service = getattr(request.app.state, "model_service", None)
    model_loaded = bool(model_service and model_service.is_loaded)
    version = getattr(request.app.state, "service_version", "0.0.0")
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "version": version,
        "uptime_seconds": _uptime_seconds(),
    }
