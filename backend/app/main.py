from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import health, router
from backend.app.config import Settings, get_settings
from backend.services.model_service import ModelService

logger = structlog.get_logger(__name__)


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level, logging.INFO)
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: Sequence[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
    ]
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = getattr(app.state, "settings", None) or get_settings()
    log = structlog.get_logger(__name__)
    log.info("app_startup", production_mode=settings.PRODUCTION_MODE)
    model_service = ModelService(settings)
    try:
        await model_service.load()
    except Exception:
        log.exception("model_load_failed")
        raise
    app.state.model_service = model_service
    app.state.service_version = settings.SERVICE_VERSION
    log.info("model_ready", loaded=model_service.is_loaded)
    yield
    model_service.shutdown()
    app.state.model_service = None
    log.info("app_shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    app = FastAPI(
        title="Stuttering AI API",
        version=resolved.SERVICE_VERSION,
        description=(
            "REST API for stuttering classification inference. "
            "See docs/api_contract.md for request and response schemas."
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved
    origins = resolved.cors_allowed_origins
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    app.add_api_route("/health", health, methods=["GET"], include_in_schema=True)

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        log = structlog.get_logger(__name__)
        log.exception("runtime_error", path=request.url.path)
        current = get_settings()
        detail = "" if current.PRODUCTION_MODE else str(exc)
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "MODEL_ERROR",
                "message": "Inference or model operation failed",
                "detail": detail,
            },
        )

    return app


_bootstrap_settings = get_settings()
configure_logging(_bootstrap_settings.LOG_LEVEL)
app = create_app(_bootstrap_settings)
