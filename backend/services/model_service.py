from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from backend.app.config import Settings


class ModelService:
    """Loads and holds the inference model (SDQ-03). Release via `shutdown()`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._artifact: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        if not self._loaded:
            return "unknown"
        if isinstance(self._artifact, dict) and "version" in self._artifact:
            return str(self._artifact["version"])
        return self._settings.SERVICE_VERSION

    async def load(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        if self._settings.MODEL_SOURCE == "s3":
            self._load_s3()
        else:
            self._load_local()

    def _load_local(self) -> None:
        path = Path(self._settings.MODEL_PATH).expanduser()
        if path.is_file():
            self._artifact = {
                "source": "local",
                "path": str(path.resolve()),
                "device": self._settings.DEVICE,
                "version": self._settings.SERVICE_VERSION,
            }
            self._loaded = True
            return
        if self._settings.PRODUCTION_MODE:
            raise FileNotFoundError(f"Model file not found: {path}")
        self._artifact = {
            "source": "local-stub",
            "requested_path": self._settings.MODEL_PATH or "(empty)",
            "device": self._settings.DEVICE,
            "version": self._settings.SERVICE_VERSION,
        }
        self._loaded = True

    def _load_s3(self) -> None:
        if self._settings.PRODUCTION_MODE:
            raise RuntimeError(
                "S3 model loading is not implemented; use MODEL_SOURCE=local or extend ModelService"
            )
        self._artifact = {
            "source": "s3-stub",
            "path": self._settings.MODEL_PATH or "(empty)",
            "device": self._settings.DEVICE,
            "version": self._settings.SERVICE_VERSION,
        }
        self._loaded = True

    def shutdown(self) -> None:
        self._artifact = None
        self._loaded = False
