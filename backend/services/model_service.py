from __future__ import annotations

import asyncio
import importlib
import io
import json
import logging
import math
import struct
import threading
import time
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from backend.app.config import Settings

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for local dev
    torch = None  # type: ignore[assignment]

try:
    from transformers import Wav2Vec2Processor
except Exception:  # pragma: no cover - optional dependency for local dev
    Wav2Vec2Processor = None  # type: ignore[assignment]


class InvalidAudioError(Exception):
    """Raised when uploaded audio is corrupt or cannot be decoded."""


class ModelNotLoadedError(Exception):
    """Raised when model initialization fails or predict() is called before load."""


class PredictionError(Exception):
    """Raised when an inference-time failure occurs."""


class _IdentityProcessor:
    """Fallback processor used only in development when transformers is unavailable."""

    def __call__(
        self,
        waveform: Any,
        *,
        sampling_rate: int,
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> dict[str, Any]:
        _ = (sampling_rate, return_tensors, padding)
        if torch is not None and not torch.is_tensor(waveform):
            return {"input_values": torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)}
        return {"input_values": waveform}


class _FallbackClassifier:
    """Deterministic fallback model for non-production environments."""

    def __init__(self, num_classes: int) -> None:
        self._num_classes = max(1, num_classes)
        self._is_eval = False
        self._device = "cpu"

    def to(self, device: str) -> _FallbackClassifier:
        self._device = device
        return self

    def eval(self) -> _FallbackClassifier:
        self._is_eval = True
        return self

    def __call__(self, **_: Any) -> Any:
        logits = [float(i) for i in range(self._num_classes)]
        return {"logits": logits}


class ModelService:
    """Model loader and thread-safe predictor for stuttering inference."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._predict_lock = threading.Lock()
        self._model_version = settings.SERVICE_VERSION
        self._model_config: dict[str, Any] = {}
        self._class_names: list[str] = [
            "Fluent",
            "Blocks",
            "Prolongations",
            "Repetitions",
            "Interjections",
        ]
        self._target_sample_rate = 16000
        self._max_samples = self._target_sample_rate * 5
        self._load_sync()

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version if self._loaded else "unknown"

    async def load(self) -> None:
        """Kept for app lifespan compatibility; __init__ already loads synchronously."""
        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        if self._loaded:
            return
        started_at = time.perf_counter()
        if self._settings.MODEL_SOURCE != "local":
            raise ModelNotLoadedError("Only MODEL_SOURCE=local is implemented for ModelService")

        artifact_dir, model_path, config_path = self._resolve_artifact_paths(self._settings.MODEL_PATH)
        if not config_path.exists() or not model_path.exists():
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError(
                    f"Missing artifact files under {artifact_dir} (need model_inference.pt and config.json)"
                )
            self._use_fallback_model("missing_artifacts")
            self._log_load_time(started_at, source="fallback")
            return

        self._model_config = self._read_json_config(config_path)
        self._class_names = self._extract_class_names(self._model_config)
        self._target_sample_rate = int(
            self._model_config.get("target_sample_rate", self._model_config.get("sample_rate", 16000))
        )
        self._max_samples = int(
            self._model_config.get("max_samples", self._target_sample_rate * 5)
        )
        self._model_version = str(self._model_config.get("model_version", self._settings.SERVICE_VERSION))
        self._processor = self._load_processor(self._model_config)
        self._model = self._load_model(model_path, self._model_config)
        self._loaded = True
        self._log_load_time(started_at, source="artifact")

    def _resolve_artifact_paths(self, model_path_value: str) -> tuple[Path, Path, Path]:
        model_path = Path(model_path_value or "").expanduser()
        if model_path.is_file():
            artifact_dir = model_path.parent
        else:
            artifact_dir = model_path if model_path_value else Path.cwd()
        return artifact_dir, artifact_dir / "model_inference.pt", artifact_dir / "config.json"

    def _read_json_config(self, config_path: Path) -> dict[str, Any]:
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelNotLoadedError(f"Failed to parse config.json: {exc}") from exc

    def _extract_class_names(self, cfg: dict[str, Any]) -> list[str]:
        keys = ("class_names", "labels", "classes")
        for key in keys:
            value = cfg.get(key)
            if isinstance(value, list) and value:
                return [str(v) for v in value]
        return list(self._class_names)

    def _load_processor(self, cfg: dict[str, Any]) -> Any:
        model_name = str(cfg.get("model_name", "")).strip()
        if not model_name:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError("config.json is missing required field 'model_name'")
            return _IdentityProcessor()
        if Wav2Vec2Processor is None:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError("transformers is not installed; cannot load Wav2Vec2Processor")
            return _IdentityProcessor()
        try:
            return Wav2Vec2Processor.from_pretrained(model_name)
        except Exception as exc:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError(f"Failed to load Wav2Vec2Processor: {exc}") from exc
            return _IdentityProcessor()

    def _load_model(self, model_path: Path, cfg: dict[str, Any]) -> Any:
        if torch is None:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError("torch is not installed; cannot load model_inference.pt")
            return _FallbackClassifier(num_classes=len(self._class_names)).eval()

        model: Any = None
        classifier_class = cfg.get("classifier_class")
        if isinstance(classifier_class, str) and classifier_class.strip():
            model = self._rebuild_classifier(classifier_class, cfg, model_path)

        if model is None:
            model = self._load_torchscript(model_path)
        if model is None:
            if self._settings.PRODUCTION_MODE:
                raise ModelNotLoadedError(f"Unable to reconstruct model from {model_path}")
            model = _FallbackClassifier(num_classes=len(self._class_names))

        if hasattr(model, "to"):
            model = model.to(self._settings.DEVICE)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def _rebuild_classifier(self, dotted_path: str, cfg: dict[str, Any], model_path: Path) -> Any:
        if torch is None:
            return None
        try:
            module_name, class_name = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            classifier_cls = getattr(module, class_name)
            init_args = cfg.get("model_init_args", {})
            model = classifier_cls(**init_args)
            checkpoint = torch.load(model_path, map_location=self._settings.DEVICE)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(state_dict)
            return model
        except Exception:
            return None

    def _load_torchscript(self, model_path: Path) -> Any:
        if torch is None:
            return None
        try:
            return torch.jit.load(str(model_path), map_location=self._settings.DEVICE)
        except Exception:
            return None

    def _use_fallback_model(self, reason: str) -> None:
        self._model_config = {"fallback_reason": reason}
        self._processor = _IdentityProcessor()
        self._model = _FallbackClassifier(num_classes=len(self._class_names)).eval()
        self._loaded = True
        self._model_version = self._settings.SERVICE_VERSION

    def _log_load_time(self, started_at: float, *, source: str) -> None:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "model_loaded",
            extra={
                "source": source,
                "device": self._settings.DEVICE,
                "model_version": self._model_version,
                "load_time_ms": elapsed_ms,
            },
        )

    def predict(self, audio_bytes: bytes) -> dict[str, Any]:
        if not self._loaded:
            raise ModelNotLoadedError("ModelService is not loaded")
        started_at = time.perf_counter()
        with self._predict_lock:
            try:
                waveform, sample_rate = self._decode_wav_bytes(audio_bytes)
                processed = self._preprocess_waveform(waveform, sample_rate)
                inputs = self._prepare_inputs(processed)
                outputs = self._forward(inputs)
                probs = self._softmax(outputs)
                predicted_idx = max(range(len(probs)), key=lambda i: probs[i])
                confidence_scores = {
                    label: float(probs[i]) for i, label in enumerate(self._class_names[: len(probs)])
                }
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                return {
                    "predicted_class": self._class_names[predicted_idx],
                    "confidence_scores": confidence_scores,
                    "processing_time_ms": elapsed_ms,
                    "model_version": self._model_version,
                }
            except (InvalidAudioError, ModelNotLoadedError):
                raise
            except Exception as exc:
                raise PredictionError(f"Inference failed: {exc}") from exc

    def _decode_wav_bytes(self, audio_bytes: bytes) -> tuple[Any, int]:
        if not audio_bytes:
            raise InvalidAudioError("Audio payload is empty")
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                raw = wav_file.readframes(num_frames)
        except wave.Error as exc:
            raise InvalidAudioError(f"Invalid WAV audio: {exc}") from exc

        samples = self._pcm_to_floats(raw, sample_width)
        if channels > 1:
            mono: list[float] = []
            for i in range(0, len(samples), channels):
                frame = samples[i : i + channels]
                if not frame:
                    continue
                mono.append(sum(frame) / float(len(frame)))
            samples = mono

        if torch is not None:
            tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
            return tensor, sample_rate
        return samples, sample_rate

    def _pcm_to_floats(self, raw: bytes, sample_width: int) -> list[float]:
        if sample_width == 1:
            return [((b - 128) / 128.0) for b in raw]
        if sample_width == 2:
            count = len(raw) // 2
            values = struct.unpack("<" + ("h" * count), raw)
            return [v / 32768.0 for v in values]
        if sample_width == 4:
            count = len(raw) // 4
            values = struct.unpack("<" + ("i" * count), raw)
            return [v / 2147483648.0 for v in values]
        raise InvalidAudioError(f"Unsupported WAV sample width: {sample_width}")

    def _preprocess_waveform(self, waveform: Any, sample_rate: int) -> Any:
        resampled = self._resample(waveform, sample_rate, self._target_sample_rate)
        normalized = self._normalize(resampled)
        return self._pad_or_truncate(normalized, self._max_samples)

    def _resample(self, waveform: Any, source_rate: int, target_rate: int) -> Any:
        if source_rate == target_rate:
            return waveform
        if torch is not None and torch.is_tensor(waveform):
            try:
                import torchaudio.functional as TAF

                return TAF.resample(waveform, source_rate, target_rate)
            except Exception:
                # Fallback linear interpolation if torchaudio isn't available.
                import torch.nn.functional as F

                x = waveform.unsqueeze(0)
                target_len = max(1, int(waveform.shape[-1] * (target_rate / float(source_rate))))
                y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
                return y.squeeze(0)
        target_len = max(1, int(len(waveform) * (target_rate / float(source_rate))))
        if target_len == len(waveform):
            return waveform
        output: list[float] = []
        scale = (len(waveform) - 1) / float(max(1, target_len - 1))
        for i in range(target_len):
            src_pos = i * scale
            left = int(math.floor(src_pos))
            right = min(left + 1, len(waveform) - 1)
            alpha = src_pos - left
            output.append((1 - alpha) * waveform[left] + alpha * waveform[right])
        return output

    def _normalize(self, waveform: Any) -> Any:
        if torch is not None and torch.is_tensor(waveform):
            peak = torch.max(torch.abs(waveform))
            if float(peak) > 0:
                return waveform / peak
            return waveform
        peak = max((abs(x) for x in waveform), default=0.0)
        if peak <= 0:
            return waveform
        return [x / peak for x in waveform]

    def _pad_or_truncate(self, waveform: Any, max_samples: int) -> Any:
        if torch is not None and torch.is_tensor(waveform):
            curr = waveform.shape[-1]
            if curr > max_samples:
                return waveform[..., :max_samples]
            if curr < max_samples:
                pad = max_samples - curr
                return torch.nn.functional.pad(waveform, (0, pad))
            return waveform
        curr = len(waveform)
        if curr > max_samples:
            return waveform[:max_samples]
        if curr < max_samples:
            return waveform + ([0.0] * (max_samples - curr))
        return waveform

    def _prepare_inputs(self, waveform: Any) -> dict[str, Any]:
        processor = self._processor
        if processor is None:
            raise ModelNotLoadedError("Wav2Vec2Processor is not initialized")
        if torch is not None and torch.is_tensor(waveform):
            values = waveform.squeeze(0).detach().cpu().tolist()
        else:
            values = waveform
        inputs = processor(
            values,
            sampling_rate=self._target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        if not isinstance(inputs, dict):
            raise PredictionError("Processor returned invalid payload")
        if torch is not None:
            for key, value in list(inputs.items()):
                if torch.is_tensor(value):
                    inputs[key] = value.to(self._settings.DEVICE)
        return inputs

    def _forward(self, inputs: dict[str, Any]) -> Any:
        if self._model is None:
            raise ModelNotLoadedError("Model is not initialized")
        no_grad = torch.no_grad if torch is not None else nullcontext
        with no_grad():
            output = self._model(**inputs)
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, dict):
            logits = output.get("logits")
        if logits is None:
            logits = output
        return logits

    def _softmax(self, logits: Any) -> list[float]:
        if torch is not None and torch.is_tensor(logits):
            tensor = logits
            if tensor.dim() > 1:
                tensor = tensor[0]
            probs = torch.softmax(tensor, dim=-1).detach().cpu().tolist()
            return [float(p) for p in probs]
        if isinstance(logits, dict) and "logits" in logits:
            logits = logits["logits"]
        if isinstance(logits, (list, tuple)) and logits and isinstance(logits[0], (list, tuple)):
            logits = logits[0]
        arr = [float(x) for x in logits]
        if not arr:
            raise PredictionError("Model returned empty logits")
        max_v = max(arr)
        exps = [math.exp(v - max_v) for v in arr]
        total = sum(exps)
        return [v / total for v in exps]

    def shutdown(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
