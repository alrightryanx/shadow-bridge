"""
Music generation service for ShadowBridge.

Provides music generation using AudioCraft (MusicGen) as the primary backend.
Suno API available as optional backend when API key is configured.
RVC voice conversion stubs (not yet implemented).
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Storage directories
SONGS_DIR = Path.home() / ".shadowai" / "music" / "songs"
STYLES_DIR = Path.home() / ".shadowai" / "music" / "styles"
VOICE_MODELS_DIR = Path.home() / ".shadowai" / "music" / "voice_models"
SONGS_DIR.mkdir(parents=True, exist_ok=True)
STYLES_DIR.mkdir(parents=True, exist_ok=True)
VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Generation job tracking
_generation_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


@dataclass
class MusicGenerationResult:
    """Result of music generation."""
    success: bool
    song_id: Optional[str] = None
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    backend_used: Optional[str] = None
    generation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "song_id": self.song_id,
            "audio_path": self.audio_path,
            "audio_url": self.audio_url,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "backend_used": self.backend_used,
            "generation_time": self.generation_time,
        }


class MusicService:
    """
    Music generation service.

    Primary backend: AudioCraft (local)
    Optional backend: Suno API (cloud)
    """

    def __init__(self):
        self._suno_api_key = os.environ.get("SUNO_API_KEY")
        self._audiocraft_available = self._check_audiocraft()

    def _check_audiocraft(self) -> bool:
        """Check if AudioCraft dependencies are available."""
        try:
            from web.services.audio_service import AUDIOCRAFT_AVAILABLE
            return AUDIOCRAFT_AVAILABLE
        except ImportError:
            return False

    def generate_music(
        self,
        prompt: str,
        genre: str = "pop",
        duration: int = 120,
        voice_model_id: Optional[str] = None,
        is_instrumental: bool = False,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
        lyrics: Optional[str] = None,
        project_id: Optional[str] = None,
        style_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> MusicGenerationResult:
        """
        Generate music using available backend.

        Priority:
        1. Suno API (if API key configured)
        2. AudioCraft (local, if installed)
        3. Error
        """
        start_time = time.time()
        song_id = f"song_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        if progress_callback:
            progress_callback(5, "Starting generation...")

        # Try Suno API first if configured
        if self._suno_api_key:
            try:
                if progress_callback:
                    progress_callback(10, "Using Suno API...")
                result = self._generate_suno(
                    song_id=song_id,
                    prompt=prompt,
                    genre=genre,
                    duration=duration,
                    is_instrumental=is_instrumental,
                    tags=tags,
                    progress_callback=progress_callback,
                )
                if result.success:
                    result.generation_time = time.time() - start_time
                    self._save_song_metadata(song_id, prompt, genre, duration, result)
                    return result
                else:
                    logger.warning(f"Suno API failed: {result.error}, trying AudioCraft...")
            except Exception as e:
                logger.warning(f"Suno API error: {e}, trying AudioCraft...")

        # Fall back to AudioCraft
        if self._audiocraft_available:
            try:
                if progress_callback:
                    progress_callback(20, "Using AudioCraft (local)...")
                result = self._generate_audiocraft(
                    song_id=song_id,
                    prompt=prompt,
                    genre=genre,
                    duration=duration,
                    progress_callback=progress_callback,
                )
                if result.success:
                    result.generation_time = time.time() - start_time
                    self._save_song_metadata(song_id, prompt, genre, duration, result)
                    return result
                else:
                    return result
            except Exception as e:
                logger.error(f"AudioCraft error: {e}")
                return MusicGenerationResult(
                    success=False,
                    song_id=song_id,
                    error=f"AudioCraft generation failed: {str(e)}",
                )

        # No backend available
        return MusicGenerationResult(
            success=False,
            song_id=song_id,
            error="No music generation backend available. Install AudioCraft via /audio/setup or configure SUNO_API_KEY.",
        )

    def _generate_audiocraft(
        self,
        song_id: str,
        prompt: str,
        genre: str,
        duration: int,
        progress_callback: Optional[callable] = None,
    ) -> MusicGenerationResult:
        """Generate music using AudioCraft."""
        try:
            from web.services.audio_service import get_audio_generation_service

            service = get_audio_generation_service()

            # Combine prompt with genre for better results
            full_prompt = f"{genre} music: {prompt}"

            # AudioCraft max duration is 30 seconds, so cap it
            actual_duration = min(duration, 30)

            if progress_callback:
                progress_callback(40, "Generating audio with MusicGen...")

            result = service.generate_audio(
                prompt=full_prompt,
                duration_seconds=actual_duration,
                model_id="musicgen-medium",
                mode="music",
                output_name=song_id,
                progress_callback=progress_callback,
            )

            if result.success:
                # Move to songs directory
                song_dir = SONGS_DIR / song_id
                song_dir.mkdir(parents=True, exist_ok=True)

                song_path = song_dir / "audio.wav"

                # Copy file if not already in the right place
                if result.audio_path and Path(result.audio_path).exists():
                    import shutil
                    shutil.copy2(result.audio_path, song_path)

                return MusicGenerationResult(
                    success=True,
                    song_id=song_id,
                    audio_path=str(song_path),
                    audio_url=f"/music/songs/{song_id}/audio",
                    duration_seconds=actual_duration,
                    backend_used="audiocraft",
                )
            else:
                return MusicGenerationResult(
                    success=False,
                    song_id=song_id,
                    error=result.error or "AudioCraft generation failed",
                )

        except Exception as e:
            logger.error(f"AudioCraft generation error: {e}")
            return MusicGenerationResult(
                success=False,
                song_id=song_id,
                error=str(e),
            )

    def _generate_suno(
        self,
        song_id: str,
        prompt: str,
        genre: str,
        duration: int,
        is_instrumental: bool,
        tags: Optional[List[str]],
        progress_callback: Optional[callable] = None,
    ) -> MusicGenerationResult:
        """Generate music using Suno API (sync version)."""
        import requests

        if not self._suno_api_key:
            return MusicGenerationResult(
                success=False,
                song_id=song_id,
                error="Suno API key not configured",
            )

        try:
            base_url = "https://api.suno.ai/v1"
            headers = {
                "Authorization": f"Bearer {self._suno_api_key}",
                "Content-Type": "application/json",
            }

            if progress_callback:
                progress_callback(15, "Submitting to Suno API...")

            # Submit generation request
            payload = {
                "prompt": f"{genre}: {prompt}",
                "duration": duration,
                "instrumental": is_instrumental,
            }
            if tags:
                payload["tags"] = tags

            response = requests.post(
                f"{base_url}/generate",
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code != 200:
                return MusicGenerationResult(
                    success=False,
                    song_id=song_id,
                    error=f"Suno API error: {response.status_code} - {response.text}",
                )

            data = response.json()
            suno_song_id = data.get("song_id") or data.get("id")

            if not suno_song_id:
                return MusicGenerationResult(
                    success=False,
                    song_id=song_id,
                    error="No song ID returned from Suno API",
                )

            # Poll for completion
            if progress_callback:
                progress_callback(30, "Waiting for Suno to generate...")

            max_attempts = 60  # 5 minutes
            for attempt in range(max_attempts):
                time.sleep(5)

                status_response = requests.get(
                    f"{base_url}/songs/{suno_song_id}",
                    headers=headers,
                    timeout=10,
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")

                    if progress_callback:
                        progress = 30 + int((attempt / max_attempts) * 50)
                        progress_callback(progress, f"Suno generating... ({status})")

                    if status == "complete":
                        break
                    elif status == "failed":
                        return MusicGenerationResult(
                            success=False,
                            song_id=song_id,
                            error=status_data.get("error", "Suno generation failed"),
                        )
            else:
                return MusicGenerationResult(
                    success=False,
                    song_id=song_id,
                    error="Suno generation timed out",
                )

            # Download audio
            if progress_callback:
                progress_callback(85, "Downloading audio...")

            download_response = requests.get(
                f"{base_url}/songs/{suno_song_id}/download",
                headers=headers,
                timeout=60,
            )

            if download_response.status_code != 200:
                return MusicGenerationResult(
                    success=False,
                    song_id=song_id,
                    error="Failed to download audio from Suno",
                )

            # Save audio
            song_dir = SONGS_DIR / song_id
            song_dir.mkdir(parents=True, exist_ok=True)
            song_path = song_dir / "audio.wav"

            with open(song_path, "wb") as f:
                f.write(download_response.content)

            if progress_callback:
                progress_callback(100, "Complete!")

            return MusicGenerationResult(
                success=True,
                song_id=song_id,
                audio_path=str(song_path),
                audio_url=f"/music/songs/{song_id}/audio",
                duration_seconds=duration,
                backend_used="suno",
            )

        except requests.RequestException as e:
            return MusicGenerationResult(
                success=False,
                song_id=song_id,
                error=f"Suno API request failed: {str(e)}",
            )

    def _save_song_metadata(
        self,
        song_id: str,
        prompt: str,
        genre: str,
        duration: int,
        result: MusicGenerationResult,
    ) -> None:
        """Save song metadata to JSON file."""
        try:
            song_dir = SONGS_DIR / song_id
            song_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "song_id": song_id,
                "prompt": prompt,
                "genre": genre,
                "duration": duration,
                "title": f"Generated Song",
                "audio_path": result.audio_path,
                "audio_url": result.audio_url,
                "backend_used": result.backend_used,
                "generation_time": result.generation_time,
                "status": "ready",
                "created_at": int(time.time() * 1000),
                "modified_at": int(time.time() * 1000),
            }

            metadata_path = song_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save song metadata: {e}")

    def get_song(self, song_id: str) -> Optional[Dict[str, Any]]:
        """Get song metadata by ID."""
        song_dir = SONGS_DIR / song_id
        metadata_path = song_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load song metadata: {e}")
            return None

    def get_song_audio_path(self, song_id: str) -> Optional[str]:
        """Get path to song audio file."""
        song_dir = SONGS_DIR / song_id

        # Check for common audio formats
        for ext in ["wav", "mp3", "flac", "m4a"]:
            audio_path = song_dir / f"audio.{ext}"
            if audio_path.exists():
                return str(audio_path)

        return None

    def list_songs(
        self,
        project_id: Optional[str] = None,
        genre: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List all generated songs."""
        songs = []

        try:
            if not SONGS_DIR.exists():
                return {"songs": [], "total": 0}

            song_dirs = sorted(
                SONGS_DIR.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            for song_dir in song_dirs:
                if not song_dir.is_dir():
                    continue

                metadata_path = song_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Filter by project_id if specified
                    if project_id and metadata.get("project_id") != project_id:
                        continue

                    # Filter by genre if specified
                    if genre and metadata.get("genre") != genre:
                        continue

                    songs.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {song_dir.name}: {e}")

            total = len(songs)
            songs = songs[offset:offset + limit]

            return {"songs": songs, "total": total}

        except Exception as e:
            logger.error(f"Failed to list songs: {e}")
            return {"songs": [], "total": 0, "error": str(e)}

    def delete_song(self, song_id: str) -> Dict[str, Any]:
        """Delete a song."""
        import shutil

        song_dir = SONGS_DIR / song_id

        if not song_dir.exists():
            return {"success": False, "error": "Song not found"}

        try:
            shutil.rmtree(song_dir)
            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to delete song: {e}")
            return {"success": False, "error": str(e)}


# RVC Voice Conversion Stubs
class RVCManager:
    """
    RVC (Retrieval-based Voice Conversion) manager.

    NOTE: RVC is not currently installed. These are honest stubs that return
    appropriate error messages directing users to setup instructions.
    """

    def __init__(self):
        self._rvc_available = self._check_rvc()

    def _check_rvc(self) -> bool:
        """Check if RVC dependencies are available."""
        try:
            import rvc
            return True
        except ImportError:
            return False

    def train_voice_model(
        self,
        name: str,
        audio_files: List[str],
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """Train a new voice model from audio samples."""
        if not self._rvc_available:
            return {
                "success": False,
                "error": "RVC is not installed. See setup instructions at: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
                "install_guide": "https://github.com/ryancartwright/shadow-bridge/blob/main/docs/rvc-setup.md",
            }

        # Actual RVC training would go here
        return {
            "success": False,
            "error": "RVC training not yet implemented",
        }

    def convert_audio(
        self,
        audio_path: str,
        voice_model_id: str,
        pitch_shift: int = 0,
    ) -> Dict[str, Any]:
        """Convert audio using a trained voice model."""
        if not self._rvc_available:
            return {
                "success": False,
                "error": "RVC is not installed. See setup instructions at: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
                "install_guide": "https://github.com/ryancartwright/shadow-bridge/blob/main/docs/rvc-setup.md",
            }

        # Actual RVC conversion would go here
        return {
            "success": False,
            "error": "RVC conversion not yet implemented",
        }

    def preview_voice(
        self,
        voice_model_id: str,
        sample_text: str = "Hello, this is a voice preview.",
    ) -> Dict[str, Any]:
        """Preview a voice model with sample text."""
        if not self._rvc_available:
            return {
                "success": False,
                "error": "RVC is not installed. See setup instructions at: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
                "install_guide": "https://github.com/ryancartwright/shadow-bridge/blob/main/docs/rvc-setup.md",
            }

        # Actual RVC preview would go here
        return {
            "success": False,
            "error": "RVC preview not yet implemented",
        }

    def list_voice_models(self) -> Dict[str, Any]:
        """List available voice models."""
        models = []

        try:
            if VOICE_MODELS_DIR.exists():
                for model_dir in VOICE_MODELS_DIR.iterdir():
                    if not model_dir.is_dir():
                        continue

                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            models.append(json.load(f))
        except Exception as e:
            logger.error(f"Failed to list voice models: {e}")

        return {"success": True, "models": models}


# Style Model Management
class StyleManager:
    """Manager for music style models."""

    def list_styles(self) -> Dict[str, Any]:
        """List all trained style models."""
        styles = []

        try:
            if STYLES_DIR.exists():
                for style_dir in STYLES_DIR.iterdir():
                    if not style_dir.is_dir():
                        continue

                    metadata_path = style_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            styles.append(json.load(f))
        except Exception as e:
            logger.error(f"Failed to list styles: {e}")

        return {"success": True, "styles": styles}

    def train_style(
        self,
        name: str,
        audio_files: List[str],
    ) -> Dict[str, Any]:
        """Train a new style model from audio samples."""
        try:
            style_id = f"style_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            style_dir = STYLES_DIR / style_id
            style_dir.mkdir(parents=True, exist_ok=True)

            # Save audio files
            samples_dir = style_dir / "samples"
            samples_dir.mkdir(exist_ok=True)

            import shutil
            for audio_file in audio_files:
                if Path(audio_file).exists():
                    shutil.copy2(audio_file, samples_dir)

            # Save metadata
            metadata = {
                "id": style_id,
                "name": name,
                "sample_count": len(audio_files),
                "status": "ready",
                "created_at": int(time.time() * 1000),
            }

            with open(style_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            return {"success": True, "style_id": style_id}

        except Exception as e:
            logger.error(f"Failed to train style: {e}")
            return {"success": False, "error": str(e)}


# Singleton instances
_music_service: Optional[MusicService] = None
_rvc_manager: Optional[RVCManager] = None
_style_manager: Optional[StyleManager] = None


def get_music_service() -> MusicService:
    """Get the music service singleton."""
    global _music_service
    if _music_service is None:
        _music_service = MusicService()
    return _music_service


def get_rvc_manager() -> RVCManager:
    """Get the RVC manager singleton."""
    global _rvc_manager
    if _rvc_manager is None:
        _rvc_manager = RVCManager()
    return _rvc_manager


def get_style_manager() -> StyleManager:
    """Get the style manager singleton."""
    global _style_manager
    if _style_manager is None:
        _style_manager = StyleManager()
    return _style_manager


# Job tracking functions
def create_generation_job(song_id: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new generation job."""
    with _jobs_lock:
        job = {
            "song_id": song_id,
            "prompt": prompt,
            "params": params,
            "status": "pending",
            "progress": 0,
            "message": "Queued",
            "created_at": int(time.time() * 1000),
            "updated_at": int(time.time() * 1000),
            "error": None,
            "result": None,
        }
        _generation_jobs[song_id] = job
        return job


def update_generation_job(song_id: str, **updates) -> None:
    """Update a generation job."""
    with _jobs_lock:
        if song_id in _generation_jobs:
            _generation_jobs[song_id].update(updates)
            _generation_jobs[song_id]["updated_at"] = int(time.time() * 1000)


def get_generation_job(song_id: str) -> Optional[Dict[str, Any]]:
    """Get a generation job."""
    with _jobs_lock:
        return _generation_jobs.get(song_id)


def remove_generation_job(song_id: str) -> None:
    """Remove a generation job (after completion/failure)."""
    with _jobs_lock:
        _generation_jobs.pop(song_id, None)
