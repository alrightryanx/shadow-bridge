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

# Detect if running from PyInstaller bundle
IS_FROZEN = getattr(sys, 'frozen', False)

AUDIO_INSTALL_COMMANDS = [
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "pip install audiocraft diffusers transformers accelerate rembg",
]
AUDIO_INSTALL_GUIDE_URL = (
    "https://github.com/ryancartwright/shadow-bridge/blob/main/docs/windows-installer.md#audio-dependencies"
)
AUDIO_INSTALL_SIZE_GB = 3.0

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
        {
            "id": "musicgen-small",
            "name": "MusicGen Small",
            "max_duration": 30,
            "description": "Fast generation, good for melodies and ideas.",
            "estimated_multiplier": 4
        },
        {
            "id": "musicgen-medium",
            "name": "MusicGen Medium",
            "max_duration": 30,
            "description": "Balanced speed and quality for full tracks.",
            "estimated_multiplier": 6
        },
        {
            "id": "musicgen-large",
            "name": "MusicGen Large",
            "max_duration": 30,
            "description": "Highest quality, slower generation.",
            "estimated_multiplier": 10
        },
    ],
    "sfx": [
        {
            "id": "audiogen-medium",
            "name": "AudioGen Medium (SFX)",
            "max_duration": 10,
            "description": "High fidelity sound effects and foley.",
            "estimated_multiplier": 5
        },
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

    def unload_all_models(self):
        """Unload all loaded models to free GPU memory."""
        self._music_model = None
        self._music_model_id = None
        self._sfx_model = None
        self._sfx_model_id = None
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {"success": True, "message": "All audio models unloaded and VRAM cleared."}

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
        progress_callback: Optional[callable] = None,
    ) -> AudioGenerationResult:
        if not prompt:
            return AudioGenerationResult(success=False, error="Prompt is required.")

        if duration_seconds <= 0:
            return AudioGenerationResult(
                success=False, error="Duration must be greater than 0."
            )

        try:
            if progress_callback:
                progress_callback(5, "Initializing...")

            if TORCH_AVAILABLE and seed is not None:
                torch.manual_seed(seed)

            if progress_callback:
                progress_callback(10, "Loading model...")

            model = self._load_model(model_id, mode)
            model.set_generation_params(
                duration=duration_seconds,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

            if progress_callback:
                progress_callback(30, "Generating audio (this may take a while)...")

            wav = model.generate([prompt], progress=True)[0]
            
            if progress_callback:
                progress_callback(90, "Processing output...")

            sample_rate = getattr(model, "sample_rate", 32000)

            output_base = OUTPUT_DIR / output_name
            audio_write(
                str(output_base),
                wav.cpu(),
                sample_rate,
                strategy="loudness",
            )
            final_path = str(output_base) + ".wav"
            
            if progress_callback:
                progress_callback(100, "Complete!")

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
        "is_frozen": IS_FROZEN,
        "install_size_gb": AUDIO_INSTALL_SIZE_GB,
    }

    if TORCH_AVAILABLE:
        try:
            status["gpu_available"] = bool(torch.cuda.is_available())
            if status["gpu_available"]:
                status["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    status["ready"] = AUDIOCRAFT_AVAILABLE and TORCH_AVAILABLE
    
    # Add helpful message if not ready
    if not status["ready"]:
        # Check if we can actually install (need system python if frozen)
        has_python = True
        if IS_FROZEN:
            import shutil
            has_python = any(shutil.which(cmd) for cmd in ["python", "python3", "py"])
        
        status["install_commands"] = list(AUDIO_INSTALL_COMMANDS)
        status["install_manual_url"] = AUDIO_INSTALL_GUIDE_URL
        
        if IS_FROZEN:
            if has_python:
                status["install_note"] = (
                    "AI dependencies were not installed during setup. "
                    "You can install them now using the system Python."
                )
                status["can_install"] = True
            else:
                status["install_note"] = (
                    "AI dependencies missing. System Python not found. "
                    "Please install Python 3.10+ manually to use these features."
                )
                status["can_install"] = False
        else:
            status["install_note"] = (
                "Click 'Install' to download AI dependencies "
                f"(~{AUDIO_INSTALL_SIZE_GB}GB) or run the commands below."
            )
            status["can_install"] = True
    
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
        _set_setup_status("running", "checking", 5, "Checking disk space and dependencies...")

        # Check disk space (need ~3.0GB + some buffer = 4GB)
        import shutil
        total, used, free = shutil.disk_usage(Path.home())
        free_gb = free / (1024**3)
        if free_gb < 4.0:
            raise RuntimeError(f"Insufficient disk space. Need at least 4GB, but only {free_gb:.1f}GB available.")

        if AUDIOCRAFT_AVAILABLE and TORCH_AVAILABLE:
            _set_setup_status("ready", "ready", 100, "AI generation dependencies ready")
            return

        # When frozen, we need to find a system python to run pip
        python_exe = sys.executable
        if IS_FROZEN:
            import shutil
            for cmd in ["python", "python3", "py"]:
                path = shutil.which(cmd)
                if path:
                    python_exe = path
                    # Check if pip is available
                    try:
                        subprocess.run([path, "-m", "pip", "--version"], capture_output=True, check=True)
                        break
                    except Exception:
                        continue
            else:
                _set_setup_status(
                    "error",
                    "python_not_found",
                    100,
                    (
                        "System Python not found or pip is missing. "
                        "Please install Python 3.10+ to use AI generation features."
                    ),
                    error="Python not found"
                )
                return

        _set_setup_status("running", "installing", 20, "Installing AI dependencies (~3.0GB download)...")

        # Install PyTorch with CUDA support first
        _set_setup_status("running", "installing_torch", 30, "Installing PyTorch + Vision/Audio (~2GB)...")
        torch_cmd = [python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
        result = subprocess.run(torch_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode != 0:
            raise RuntimeError(f"PyTorch install failed: {result.stderr.strip()}")

        _set_setup_status("running", "installing_ai_libs", 70, "Installing AudioCraft, Diffusers, etc (~1GB)...")
        ac_cmd = [python_exe, "-m", "pip", "install", "audiocraft", "diffusers", "transformers", "accelerate", "rembg"]
        result = subprocess.run(ac_cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout
        if result.returncode != 0:
            raise RuntimeError(f"AI libraries install failed: {result.stderr.strip()}")

        _set_setup_status("ready", "ready", 100, "AI dependencies installed! Please restart ShadowBridge to use them.")
    except subprocess.TimeoutExpired:
        logger.error("Audio setup timed out")
        _set_setup_status("error", "timeout", 100, "Installation timed out. Try running manually: pip install torch torchaudio audiocraft", error="Timeout")
    except Exception as exc:
        logger.error(f"Audio setup failed: {exc}")
        _set_setup_status("error", "error", 100, f"Audio setup failed: {exc}", error=str(exc))


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
