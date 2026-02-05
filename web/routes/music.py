"""
Music Studio routes for Shadow Web Dashboard.

Provides REST API for music generation, voice cloning (RVC), and style training.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

music_bp = Blueprint("music", __name__)

# Allowed audio extensions for uploads
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


def _allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


# =============================================================================
# Music Generation Routes
# =============================================================================


@music_bp.route("/generate", methods=["POST"])
def api_music_generate():
    """Start music generation."""
    from web.services.music_service import (
        get_music_service,
        create_generation_job,
        update_generation_job,
    )

    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"success": False, "error": "Prompt is required"}), 400

    # Extract parameters
    genre = data.get("genre", "pop")
    duration = int(data.get("duration", 120))
    voice_model_id = data.get("voice_model_id")
    is_instrumental = data.get("is_instrumental", False)
    bpm = data.get("bpm")
    key = data.get("key")
    lyrics = data.get("lyrics")
    project_id = data.get("project_id")
    style_id = data.get("style_id")
    tags = data.get("tags")

    # Create song ID and job
    song_id = f"song_{int(time.time() * 1000)}"
    job = create_generation_job(song_id, prompt, data)

    def _run_generation():
        """Background generation task."""
        try:
            update_generation_job(song_id, status="running", progress=10, message="Starting generation...")

            service = get_music_service()

            def progress_callback(progress: int, message: str):
                update_generation_job(song_id, progress=progress, message=message)

            result = service.generate_music(
                prompt=prompt,
                genre=genre,
                duration=duration,
                voice_model_id=voice_model_id,
                is_instrumental=is_instrumental,
                bpm=bpm,
                key=key,
                lyrics=lyrics,
                project_id=project_id,
                style_id=style_id,
                tags=tags,
                progress_callback=progress_callback,
            )

            if result.success:
                update_generation_job(
                    song_id,
                    status="completed",
                    progress=100,
                    message="Complete!",
                    result=result.to_dict(),
                )
            else:
                update_generation_job(
                    song_id,
                    status="failed",
                    progress=100,
                    message=result.error or "Generation failed",
                    error=result.error,
                )

        except Exception as e:
            logger.error(f"Music generation error: {e}")
            update_generation_job(
                song_id,
                status="failed",
                progress=100,
                message=str(e),
                error=str(e),
            )

    # Start background thread
    threading.Thread(target=_run_generation, daemon=True).start()

    return jsonify({
        "success": True,
        "song_id": song_id,
        "status": "pending",
        "message": "Generation started",
    })


@music_bp.route("/status/<song_id>")
def api_music_status(song_id: str):
    """Get generation status for a song."""
    from web.services.music_service import get_generation_job, get_music_service

    # Check job status first (for in-progress generations)
    job = get_generation_job(song_id)
    if job:
        response = {
            "success": True,
            "song_id": song_id,
            "status": job.get("status", "pending"),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
        }

        # If completed, include result
        if job.get("status") == "completed" and job.get("result"):
            response.update(job["result"])
            response["audio_url"] = f"/music/songs/{song_id}/audio"

        # If failed, include error
        if job.get("status") == "failed":
            response["error"] = job.get("error")

        return jsonify(response)

    # Check if song already exists (completed previously)
    service = get_music_service()
    song = service.get_song(song_id)

    if song:
        return jsonify({
            "success": True,
            "song_id": song_id,
            "status": "completed",
            "progress": 100,
            "audio_url": f"/music/songs/{song_id}/audio",
            **song,
        })

    return jsonify({
        "success": False,
        "error": "Song not found",
    }), 404


@music_bp.route("/songs")
def api_music_songs():
    """List all generated songs."""
    from web.services.music_service import get_music_service

    project_id = request.args.get("project_id")
    genre = request.args.get("genre")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    service = get_music_service()
    result = service.list_songs(
        project_id=project_id,
        genre=genre,
        limit=limit,
        offset=offset,
    )

    return jsonify({
        "success": True,
        "songs": result.get("songs", []),
        "total": result.get("total", 0),
    })


@music_bp.route("/songs/<song_id>", methods=["DELETE"])
def api_music_delete(song_id: str):
    """Delete a generated song."""
    from web.services.music_service import get_music_service

    service = get_music_service()
    result = service.delete_song(song_id)

    if result.get("success"):
        return jsonify({"success": True})
    else:
        return jsonify(result), 404


@music_bp.route("/songs/<song_id>/audio")
def api_music_audio(song_id: str):
    """Stream audio for a song."""
    from web.services.music_service import get_music_service

    service = get_music_service()
    audio_path = service.get_song_audio_path(song_id)

    if not audio_path or not Path(audio_path).exists():
        return jsonify({"error": "Audio file not found"}), 404

    # Determine mimetype based on extension
    ext = Path(audio_path).suffix.lower()
    mimetypes = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
    }
    mimetype = mimetypes.get(ext, "audio/wav")

    return send_file(audio_path, mimetype=mimetype, as_attachment=False)


# =============================================================================
# Style Routes
# =============================================================================


@music_bp.route("/styles")
def api_music_styles():
    """List all trained style models."""
    from web.services.music_service import get_style_manager

    manager = get_style_manager()
    result = manager.list_styles()

    return jsonify({
        "success": True,
        "styles": result.get("styles", []),
    })


@music_bp.route("/styles/train", methods=["POST"])
def api_music_styles_train():
    """Train a new style model from uploaded audio files."""
    from web.services.music_service import get_style_manager

    # Check for name
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400

    # Check for files
    if "files" not in request.files:
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    # Validate and save files
    from web.services.music_service import STYLES_DIR
    temp_dir = STYLES_DIR / "temp_upload" / str(int(time.time() * 1000))
    temp_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    try:
        for file in files:
            if file.filename:
                # Validate extension
                if not _allowed_file(file.filename):
                    return jsonify({
                        "success": False,
                        "error": f"Invalid file type: {file.filename}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                    }), 400

                # Validate size
                file.seek(0, 2)  # Seek to end
                size = file.tell()
                file.seek(0)  # Reset to start

                if size > MAX_UPLOAD_SIZE:
                    return jsonify({
                        "success": False,
                        "error": f"File too large: {file.filename}. Max size: 50MB",
                    }), 400

                # Sanitize filename and save
                safe_name = secure_filename(file.filename)
                save_path = temp_dir / safe_name
                file.save(save_path)
                saved_files.append(str(save_path))

        # Train style
        manager = get_style_manager()
        result = manager.train_style(name, saved_files)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Style training error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        # Cleanup temp files
        import shutil
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception:
            pass


# =============================================================================
# Voice Model Routes (RVC)
# =============================================================================


@music_bp.route("/voice/models")
def api_voice_models():
    """List available voice models."""
    from web.services.music_service import get_rvc_manager

    manager = get_rvc_manager()
    result = manager.list_voice_models()

    return jsonify(result)


@music_bp.route("/voice/train", methods=["POST"])
def api_voice_train():
    """Train a new voice model (RVC)."""
    from web.services.music_service import get_rvc_manager

    data = request.get_json() or {}
    name = data.get("name", "").strip()
    audio_files = data.get("audio_files", [])
    epochs = int(data.get("epochs", 100))

    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400

    if not audio_files:
        return jsonify({"success": False, "error": "Audio files are required"}), 400

    manager = get_rvc_manager()
    result = manager.train_voice_model(name, audio_files, epochs)

    return jsonify(result)


@music_bp.route("/voice/convert", methods=["POST"])
def api_voice_convert():
    """Convert audio using a voice model (RVC)."""
    from web.services.music_service import get_rvc_manager

    data = request.get_json() or {}
    audio_path = data.get("audio_path", "").strip()
    voice_model_id = data.get("voice_model_id", "").strip()
    pitch_shift = int(data.get("pitch_shift", 0))

    if not audio_path:
        return jsonify({"success": False, "error": "Audio path is required"}), 400

    if not voice_model_id:
        return jsonify({"success": False, "error": "Voice model ID is required"}), 400

    manager = get_rvc_manager()
    result = manager.convert_audio(audio_path, voice_model_id, pitch_shift)

    return jsonify(result)


@music_bp.route("/voice/preview", methods=["POST"])
def api_voice_preview():
    """Preview a voice model with sample text."""
    from web.services.music_service import get_rvc_manager

    data = request.get_json() or {}
    voice_model_id = data.get("voice_model_id", "").strip()
    sample_text = data.get("sample_text", "Hello, this is a voice preview.")

    if not voice_model_id:
        return jsonify({"success": False, "error": "Voice model ID is required"}), 400

    manager = get_rvc_manager()
    result = manager.preview_voice(voice_model_id, sample_text)

    return jsonify(result)


# =============================================================================
# Audio Upload Route
# =============================================================================


@music_bp.route("/upload", methods=["POST"])
def api_music_upload():
    """Upload audio file for processing."""
    from web.services.music_service import SONGS_DIR

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400

    # Validate extension
    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        }), 400

    # Validate size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)

    if size > MAX_UPLOAD_SIZE:
        return jsonify({
            "success": False,
            "error": f"File too large. Max size: 50MB",
        }), 400

    try:
        # Create upload directory
        upload_id = f"upload_{int(time.time() * 1000)}"
        upload_dir = SONGS_DIR / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize and save
        safe_name = secure_filename(file.filename)
        ext = Path(safe_name).suffix.lower()
        save_path = upload_dir / f"audio{ext}"
        file.save(save_path)

        return jsonify({
            "success": True,
            "upload_id": upload_id,
            "audio_path": str(save_path),
            "audio_url": f"/music/songs/{upload_id}/audio",
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Automation Routes (for background/scheduled generation)
# =============================================================================


@music_bp.route("/automation/generate", methods=["POST"])
def api_automation_generate():
    """Queue music generation for automation."""
    # Same as regular generate, but could add scheduling logic
    return api_music_generate()


@music_bp.route("/automation/status/<job_id>")
def api_automation_status(job_id: str):
    """Get automation job status."""
    return api_music_status(job_id)
