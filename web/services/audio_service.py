"""
Audio generation service for ShadowBridge.

Provides local audio generation using audiocraft (MusicGen / AudioGen) when installed.
Falls back with helpful errors if dependencies are missing.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / ".shadowai" / "audio_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_setup_lock = threading.Lock()
_setup_status: Dict[str, Any] = {
    "status": "idle",
    "stage": "idle",
    "progress": 0,
    "message": "Audio setup not started",
    "error": None,
}

try:
    from audiocraft.models import MusicGen, AudioGen
    from audiocraft.data.audio import audio_write

    AUDIOCRAFT_AVAILABLE = True
except Exception:
    AUDIOCRAFT_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


DEFAULT_MODELS = {
    "music": [
        {"id": "musicgen-small", "name": "MusicGen Small", "max_duration": 30},
        {"id": "musicgen-medium", "name": "MusicGen Medium", "max_duration": 30},
        {"id": "musicgen-large", "name": "MusicGen Large", "max_duration": 30},
    ],
    "sfx": [
        {"id": "audiogen-medium", "name": "AudioGen Medium (SFX)", "max_duration": 10},
    ],
}


@dataclass
class AudioGenerationResult:
    success: bool
    audio_path: Optional[str] = None
    duration_seconds: float = 0.0
    sample_rate: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "audio_path": self.audio_path,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "error": self.error,
        }


class AudioGenerationService:
    """Audio generation service using audiocraft when available."""

    def __init__(self) -> None:
        if not AUDIOCRAFT_AVAILABLE:
            raise RuntimeError(
                "audiocraft is not installed. Install dependencies via /audio/setup."
            )

        self._music_model = None
        self._music_model_id: Optional[str] = None
        self._sfx_model = None
        self._sfx_model_id: Optional[str] = None

    def _load_model(self, model_id: str, mode: str):
        if mode == "music":
            if self._music_model is None or self._music_model_id != model_id:
                self._music_model = MusicGen.get_pretrained(model_id)
                self._music_model_id = model_id
            return self._music_model
        if mode == "sfx":
            if self._sfx_model is None or self._sfx_model_id != model_id:
                self._sfx_model = AudioGen.get_pretrained(model_id)
                self._sfx_model_id = model_id
            return self._sfx_model
        raise ValueError(f"Unknown mode: {mode}")

    def generate_audio(
        self,
        prompt: str,
        duration_seconds: float,
        model_id: str,
        mode: str,
        output_name: str,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
    ) -> AudioGenerationResult:
        if not prompt:
            return AudioGenerationResult(success=False, error="Prompt is required.")

        if duration_seconds <= 0:
            return AudioGenerationResult(
                success=False, error="Duration must be greater than 0."
            )

        try:
            if TORCH_AVAILABLE and seed is not None:
                torch.manual_seed(seed)

            model = self._load_model(model_id, mode)
            model.set_generation_params(
                duration=duration_seconds,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

            wav = model.generate([prompt])[0]
            sample_rate = getattr(model, "sample_rate", 32000)

            output_base = OUTPUT_DIR / output_name
            audio_write(
                str(output_base),
                wav.cpu(),
                sample_rate,
                strategy="loudness",
            )
            final_path = str(output_base) + ".wav"

            return AudioGenerationResult(
                success=True,
                audio_path=final_path,
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            logger.error(f"Audio generation failed: {exc}")
            return AudioGenerationResult(success=False, error=str(exc))


_audio_service: Optional[AudioGenerationService] = None


def get_audio_generation_service() -> AudioGenerationService:
    global _audio_service
    if _audio_service is None:
        _audio_service = AudioGenerationService()
    return _audio_service


def get_audio_service_status() -> Dict[str, Any]:
    status = {
        "audiocraft_available": AUDIOCRAFT_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "gpu_available": False,
        "gpu_name": None,
    }

    if TORCH_AVAILABLE:
        try:
            status["gpu_available"] = bool(torch.cuda.is_available())
            if status["gpu_available"]:
                status["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    status["ready"] = AUDIOCRAFT_AVAILABLE and TORCH_AVAILABLE
    return status


def _set_setup_status(status: str, stage: str, progress: int, message: str, error: Optional[str] = None) -> None:
    with _setup_lock:
        _setup_status.update(
            {
                "status": status,
                "stage": stage,
                "progress": progress,
                "message": message,
                "error": error,
            }
        )


def _run_audio_setup() -> None:
    try:
        _set_setup_status("running", "checking", 5, "Checking audio dependencies...")

        if AUDIOCRAFT_AVAILABLE and TORCH_AVAILABLE:
            _set_setup_status("ready", "ready", 100, "Audio generation ready")
            return

        _set_setup_status("running", "installing", 20, "Installing audio dependencies...")

        packages = [
            "torch",
            "torchaudio",
            "audiocraft",
        ]

        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "pip install failed")

        _set_setup_status("ready", "ready", 100, "Audio generation ready")
    except Exception as exc:
        logger.error(f"Audio setup failed: {exc}")
        _set_setup_status("error", "error", 100, "Audio setup failed", error=str(exc))


def start_audio_setup() -> Dict[str, Any]:
    with _setup_lock:
        if _setup_status.get("status") == "running":
            return dict(_setup_status)

        _set_setup_status("running", "queued", 0, "Audio setup queued")
        threading.Thread(target=_run_audio_setup, daemon=True).start()
        return dict(_setup_status)


def get_audio_setup_status() -> Dict[str, Any]:
    with _setup_lock:
        return dict(_setup_status)
