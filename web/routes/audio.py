"""
Audio generation routes for Shadow Web Dashboard.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request, send_file

from web.services.audio_service import (
    DEFAULT_MODELS,
    get_audio_generation_service,
    get_audio_service_status,
    start_audio_setup,
    get_audio_setup_status,
)

logger = logging.getLogger(__name__)

audio_bp = Blueprint("audio", __name__)

AUDIO_GENERATIONS_FILE = Path.home() / ".shadowai" / "audio_generations.json"


def _load_generations() -> Dict[str, Any]:
    try:
        if AUDIO_GENERATIONS_FILE.exists():
            with open(AUDIO_GENERATIONS_FILE, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception as exc:
        logger.warning(f"Failed to load audio generations: {exc}")
    return {"generations": []}


def _save_generations(data: Dict[str, Any]) -> None:
    try:
        AUDIO_GENERATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIO_GENERATIONS_FILE, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    except Exception as exc:
        logger.error(f"Failed to save audio generations: {exc}")


def _find_generation(generation_id: str) -> Optional[Dict[str, Any]]:
    data = _load_generations()
    for generation in data.get("generations", []):
        if generation.get("id") == generation_id:
            return generation
    return None


def _update_generation(generation_id: str, updates: Dict[str, Any]) -> None:
    data = _load_generations()
    updated = False
    for generation in data.get("generations", []):
        if generation.get("id") == generation_id:
            generation.update(updates)
            generation["updated_at"] = int(time.time() * 1000)
            updated = True
            break
    if updated:
        _save_generations(data)


@audio_bp.route("/models")
def api_audio_models():
    """List audio generation models."""
    return jsonify({"data": DEFAULT_MODELS})


@audio_bp.route("/status")
def api_audio_service_status():
    """Return audio service availability."""
    return jsonify(
        {
            "service": get_audio_service_status(),
            "setup": get_audio_setup_status(),
        }
    )


@audio_bp.route("/setup", methods=["POST"])
def api_audio_setup():
    """Trigger audio dependency setup."""
    return jsonify(start_audio_setup())


@audio_bp.route("/setup/status")
def api_audio_setup_status():
    """Get audio setup status."""
    return jsonify(get_audio_setup_status())


@audio_bp.route("/generate", methods=["POST"])
def api_audio_generate():
    """Start audio generation."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    mode = data.get("mode", "music")
    model_id = data.get("model", "musicgen-small")
    duration = float(data.get("duration", 8))
    seed = data.get("seed")
    temperature = float(data.get("temperature", 1.0))
    top_k = int(data.get("top_k", 250))
    top_p = float(data.get("top_p", 0.0))

    if not prompt:
        return jsonify({"success": False, "error": "Prompt is required"}), 400

    model_options = DEFAULT_MODELS.get(mode, [])
    model_ids = {model["id"] for model in model_options}
    if model_id not in model_ids:
        return jsonify({"success": False, "error": "Unknown model"}), 400

    max_duration = next(
        (model["max_duration"] for model in model_options if model["id"] == model_id),
        30,
    )
    if duration > max_duration:
        duration = max_duration

    generation_id = f"aud_{int(time.time() * 1000)}"
    generation = {
        "id": generation_id,
        "prompt": prompt,
        "mode": mode,
        "model": model_id,
        "duration": duration,
        "status": "pending",
        "progress": 0,
        "created_at": int(time.time() * 1000),
        "updated_at": int(time.time() * 1000),
        "audio_path": None,
        "error": None,
    }

    data_store = _load_generations()
    data_store.setdefault("generations", []).append(generation)
    _save_generations(data_store)

    def _run():
        try:
            _update_generation(
                generation_id,
                {"status": "running", "progress": 10, "message": "Loading model"},
            )

            service = get_audio_generation_service()
            output_name = f"{generation_id}"

            _update_generation(
                generation_id,
                {"progress": 40, "message": "Generating audio"},
            )

            result = service.generate_audio(
                prompt=prompt,
                duration_seconds=duration,
                model_id=model_id,
                mode=mode,
                output_name=output_name,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if result.success:
                _update_generation(
                    generation_id,
                    {
                        "status": "completed",
                        "progress": 100,
                        "audio_path": result.audio_path,
                        "sample_rate": result.sample_rate,
                        "duration": result.duration_seconds,
                        "message": "Completed",
                    },
                )
            else:
                _update_generation(
                    generation_id,
                    {
                        "status": "failed",
                        "progress": 100,
                        "error": result.error,
                        "message": "Failed",
                    },
                )
        except Exception as exc:
            logger.error(f"Audio generation failed: {exc}")
            _update_generation(
                generation_id,
                {"status": "failed", "progress": 100, "error": str(exc)},
            )

    threading.Thread(target=_run, daemon=True).start()

    return jsonify({"success": True, "generation_id": generation_id})


@audio_bp.route("/status/<generation_id>")
def api_audio_status(generation_id: str):
    generation = _find_generation(generation_id)
    if not generation:
        return jsonify({"error": "Generation not found"}), 404

    generation = dict(generation)
    if generation.get("audio_path"):
        generation["stream_url"] = f"/audio/file/{generation_id}"
    return jsonify(generation)


@audio_bp.route("/history")
def api_audio_history():
    data = _load_generations()
    generations = sorted(
        data.get("generations", []),
        key=lambda g: g.get("created_at", 0),
        reverse=True,
    )

    for generation in generations:
        if generation.get("audio_path"):
            generation["stream_url"] = f"/audio/file/{generation.get('id')}"

    return jsonify({"data": generations})


@audio_bp.route("/file/<generation_id>")
def api_audio_file(generation_id: str):
    generation = _find_generation(generation_id)
    if not generation:
        return jsonify({"error": "Generation not found"}), 404

    audio_path = generation.get("audio_path")
    if not audio_path:
        return jsonify({"error": "Audio file not available"}), 404

    path = Path(audio_path)
    if not path.exists():
        return jsonify({"error": "Audio file missing"}), 404

    return send_file(path, mimetype="audio/wav", as_attachment=False)
