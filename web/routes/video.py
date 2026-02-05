"Video Generation Routes for Shadow Web Dashboard"

from flask import Blueprint, jsonify, request, send_file
import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import json
import re
import time
import threading
import queue
import hashlib
import ctypes

# Try to import advanced services
try:
    from web.services.video_progress import get_progress_tracker, ProgressStage

    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Video progress monitoring not available")

try:
    from web.services.video_error_handling import (
        get_robust_generator,
        RobustVideoGenerator,
    )

    ROBUST_GENERATOR_AVAILABLE = True
except ImportError:
    ROBUST_GENERATOR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Robust error handling not available")

# Try to import metrics service
try:
    from web.services.metrics_service import get_metrics_service

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Metrics service not available")

# Try to import downloader service
try:
    from web.services.downloader_service import get_downloader_service

    DOWNLOADER_AVAILABLE = True
except ImportError:
    DOWNLOADER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Downloader service not available")

# Try to import ComfyUI executor
try:
    from web.services.comfyui_executor import get_comfyui_executor, ComfyUIExecutor

    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ComfyUI executor not available")

logger = logging.getLogger(__name__)

video_bp = Blueprint("video", __name__)

# ============ Configuration ============

# Base models directory
MODELS_DIR = os.path.join(os.path.expanduser("~"), ".shadowai", "video_models")

# Concurrency control (Optimization: Limit to 1 concurrent generation)
generation_lock = threading.Semaphore(1)

# Process tracking (for cancellation)
active_processes = {}  # generation_id -> subprocess.Popen
active_processes_lock = threading.Lock()


def start_artifact_cleanup_thread():
    """Start a background thread to clean up old video files."""

    def cleanup_loop():
        while True:
            try:
                output_dir = os.path.join(MODELS_DIR, "outputs")
                if os.path.exists(output_dir):
                    now = time.time()
                    # 24 hours in seconds
                    retention_period = 24 * 3600

                    for f in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, f)
                        if os.path.isfile(file_path):
                            if now - os.path.getmtime(file_path) > retention_period:
                                try:
                                    os.remove(file_path)
                                    logger.info(f"Cleaned up old video artifact: {f}")
                                except Exception as e:
                                    logger.warning(f"Failed to delete {f}: {e}")
            except Exception as e:
                logger.error(f"Error in artifact cleanup loop: {e}")

            # Run every hour - use smaller chunks for better responsiveness
            for _ in range(60):  # 60 x 60 seconds = 1 hour
                time.sleep(60)  # Sleep 1 minute at a time to stay responsive

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    logger.info("Artifact cleanup thread started")


# Initialize cleanup
start_artifact_cleanup_thread()

# Model configuration
MODELS = {
    # High-end GPU models (16GB+ VRAM)
    "svi-pro-wan22": {
        "name": "SVI Pro 2.0 + Wan 2.2",
        "tier": "high_end",
        "vram_required": 16,
        "repo": "https://github.com/comfyanonymous/ComfyUI.git",
        "script": "generate.py",
        "requirements": "requirements.txt",
        "path": os.path.join(MODELS_DIR, "comfyui", ""),
        "max_duration": 120,
        "cost": 0,
        "speed_estimate_10s": 120,  # seconds
    },
    "hunyuan-15": {
        "name": "HunyuanVideo 1.5 (8.3B)",
        "tier": "high_end",
        "vram_required": 12,
        "repo": "https://github.com/Tencent-Hunyuan/HunyuanVideo.git",
        "script": "generate.py",
        "requirements": "requirements.txt",
        "path": os.path.join(MODELS_DIR, "hunyuan", ""),
        "max_duration": 30,
        "cost": 0,
        "speed_estimate_10s": 180,
    },
    # Mid-range GPU models (8-12GB VRAM)
    "ltx-video": {
        "name": "LTX Video (Fast)",
        "tier": "mid_range",
        "vram_required": 8,
        "repo": "https://github.com/Lightricks/LTX-Video.git",
        "script": "generate.py",
        "requirements": "requirements.txt",
        "path": os.path.join(MODELS_DIR, "ltx", ""),
        "max_duration": 15,
        "cost": 0,
        "speed_estimate_10s": 50,
    },
    "wan-21": {
        "name": "Wan 2.1 (1.3B)",
        "tier": "mid_range",
        "vram_required": 8,
        "repo": "https://github.com/Wan-Video/Wan2.1.git",
        "script": "generate.py",
        "requirements": "requirements.txt",
        "path": os.path.join(MODELS_DIR, "wan", ""),
        "max_duration": 30,
        "cost": 0,
        "speed_estimate_10s": 60,
    },
    # Cloud options (any hardware)
    "cloud-fal": {
        "name": "FAL.ai Cloud",
        "tier": "cloud",
        "vram_required": 0,
        "max_duration": 60,
        "cost": 0.50,
        "speed_estimate_10s": 40,
        "api_endpoint": "https://fal.run/fal-ai/fast-svd",
    },
    "cloud-runpod": {
        "name": "RunPod Cloud (H100)",
        "tier": "cloud",
        "vram_required": 0,
        "max_duration": 120,
        "cost": 2.50,
        "speed_estimate_10s": 60,
        "api_endpoint": "https://api.runpod.ai/v2",
    },
}

# In-memory storage for video generations (simple JSON file)
GENERATIONS_FILE = os.path.join(
    os.path.expanduser("~"), ".shadowai", "video_generations.json"
)

# Cache mapping for prompts
CACHE_FILE = os.path.join(os.path.expanduser("~"), ".shadowai", "video_cache.json")

# Process registry for recovery
PROCESS_REGISTRY_FILE = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")),
    ".shadowai",
    "active_video_processes.json",
)


def _save_process_registry(registry):
    """Save active process IDs to disk for recovery."""
    try:
        # Popen objects aren't JSON serializable, so we only save PIDs
        serializable = {gid: p.pid for gid, p in registry.items() if p.poll() is None}
        with open(PROCESS_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
    except Exception as e:
        logger.warning(f"Failed to save process registry: {e}")


def _load_process_registry():
    """Load process registry from disk."""
    try:
        if os.path.exists(PROCESS_REGISTRY_FILE):
            with open(PROCESS_REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load process registry: {e}")
    return {}


class WindowsKeepAwake:
    """Context manager to prevent Windows from sleeping during long operations."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def __enter__(self):
        if os.name == "nt":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
                )
                logger.info("Windows Keep-Awake enabled")
            except Exception as e:
                logger.warning(f"Failed to set thread execution state: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.name == "nt":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
                logger.info("Windows Keep-Awake disabled")
            except Exception as e:
                logger.warning(f"Failed to reset thread execution state: {e}")


def generate_request_hash(options):
    """Generate a stable MD5 hash for a video generation request."""
    # Only include parameters that affect the visual output
    key_params = {
        "prompt": options.get("prompt"),
        "model": options.get("model"),
        "duration": options.get("duration"),
        "aspect_ratio": options.get("aspect_ratio"),
        "negative_prompt": options.get("negative_prompt"),
        "seed": options.get("seed"),
        "precision": options.get("precision"),
    }
    dump = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(dump.encode("utf-8")).hexdigest()


def _load_cache():
    """Load prompt cache from file."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
    return {}


def _save_cache(cache):
    """Save prompt cache to file."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


def _load_generations():
    """Load video generations from file."""
    try:
        if os.path.exists(GENERATIONS_FILE):
            with open(GENERATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load generations: {e}")
        return {"generations": []}


def _save_generations(data):
    """Save video generations to file."""
    try:
        os.makedirs(os.path.dirname(GENERATIONS_FILE), exist_ok=True)
        with open(GENERATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save generations: {e}")
        return False


def _find_generation(generation_id):
    data = _load_generations() or {}
    for generation in data.get("generations", []):
        if generation.get("id") == generation_id:
            return generation
    return None


def _attach_stream_url(generation):
    if not generation:
        return None
    generation = dict(generation)
    if generation.get("video_path"):
        generation["stream_url"] = f"/video/file/{generation.get('id')}"
    return generation


# ============ Helper Functions ============


def build_model_command(
    model,
    script_path,
    prompt,
    duration,
    aspect_ratio,
    negative_prompt,
    seed,
    output_path,
    progress_callback,
    precision="bf16",
):
    """Build model-specific command for video generation."""

    if model == "hunyuan-15":
        return (
            [
                "python",
                script_path,
                "--prompt",
                prompt,
                "--duration",
                str(min(duration, 30)),  # Hunyuan max 30s
                "--aspect_ratio",
                aspect_ratio,
                "--output_path",
                output_path,
                "--precision",
                precision,
            ]
            + (["--negative_prompt", negative_prompt] if negative_prompt else [])
            + (["--seed", str(seed)] if seed else [])
        )

    elif model == "ltx-video":
        # LTX specific parameters
        width, height = get_resolution_for_aspect_ratio(aspect_ratio, model)
        return (
            [
                "python",
                script_path,
                "--prompt",
                prompt,
                "--duration",
                str(min(duration, 15)),  # LTX max 15s
                "--width",
                str(width),
                "--height",
                str(height),
                "--output_path",
                output_path,
                "--precision",
                precision,
            ]
            + (["--negative_prompt", negative_prompt] if negative_prompt else [])
            + (["--seed", str(seed)] if seed else [])
        )

    elif model == "wan-21":
        return (
            [
                "python",
                script_path,
                "--prompt",
                prompt,
                "--duration",
                str(min(duration, 30)),  # Wan max 30s
                "--aspect_ratio",
                aspect_ratio,
                "--output_path",
                output_path,
                "--precision",
                precision,
            ]
            + (["--negative_prompt", negative_prompt] if negative_prompt else [])
            + (["--seed", str(seed)] if seed else [])
        )

    else:
        raise Exception(f"Unknown model: {model}")


def get_resolution_for_aspect_ratio(aspect_ratio, model):
    """Get width/height for aspect ratio and model."""
    if model == "ltx-video":
        if aspect_ratio == "16:9":
            return 1024, 576
        elif aspect_ratio == "9:16":
            return 576, 1024
        else:  # 1:1
            return 768, 768
    else:
        if aspect_ratio == "16:9":
            return 1280, 720
        elif aspect_ratio == "9:16":
            return 720, 1280
        else:  # 1:1
            return 1024, 1024


def execute_video_generation_enhanced(cmd, monitor, model_type, generation_id):
    """Execute video generation with enhanced progress monitoring."""

    process = None
    try:
        monitor.update_progress(
            ProgressStage.PROCESSING_FRAMES, "Starting generation process...", 25
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=os.path.dirname(cmd[1]),  # Run in model directory
        )

        # Register process for potential cancellation
        with active_processes_lock:
            active_processes[generation_id] = process
            _save_process_registry(active_processes)
            logger.info(f"Registered process for generation {generation_id}")

        output_lines = []

        # Monitor process output with enhanced parsing
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                line = line.strip()
                if line:
                    output_lines.append(line)

                    # Use enhanced progress parsing
                    progress_info = monitor.parse_model_output(line, model_type)
                    if progress_info:
                        info_type = progress_info.get("type")

                        if info_type == "progress":
                            progress_value = progress_info.get("value", 0)
                            stage = ProgressStage.PROCESSING_FRAMES
                            monitor.update_progress(
                                stage,
                                progress_info.get("message", "Processing..."),
                                progress_value,
                            )

                        elif info_type == "stage_update":
                            stage_name = progress_info.get("stage")
                            stage_map = {
                                "loading_model": ProgressStage.LOADING_MODEL,
                                "processing_frames": ProgressStage.PROCESSING_FRAMES,
                                "encoding": ProgressStage.ENCODING,
                                "finalizing": ProgressStage.FINALIZING,
                                "completed": ProgressStage.COMPLETED,
                            }
                            stage = stage_map.get(
                                stage_name, ProgressStage.PROCESSING_FRAMES
                            )
                            progress_value = progress_info.get("value", 0)
                            monitor.update_progress(
                                stage,
                                progress_info.get("message", "Processing..."),
                                progress_value,
                            )

                        elif info_type == "performance":
                            # Performance update (FPS, etc.)
                            message = progress_info.get("message", "Processing...")
                            monitor.update_progress(
                                ProgressStage.PROCESSING_FRAMES,
                                message,
                                monitor.last_progress,
                            )

        # Wait for completion
        return_code = process.wait()

        if return_code == 0:
            monitor.update_progress(ProgressStage.FINALIZING, "Finalizing video...", 95)
            return {"success": True, "output": "\n".join(output_lines)}
        else:
            error_output = "\n".join(
                output_lines[-10:]
            )  # Last 10 lines for error context
            monitor.error(f"Process failed with code {return_code}: {error_output}")
            return {
                "success": False,
                "error": f"Process failed with code {return_code}: {error_output}",
            }

    except subprocess.TimeoutExpired:
        if process:
            process.kill()
        monitor.error("Generation timed out")
        return {"success": False, "error": "Generation timed out"}
    except Exception as e:
        monitor.error(f"Execution error: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        # Unregister process
        with active_processes_lock:
            if generation_id in active_processes:
                del active_processes[generation_id]
                _save_process_registry(active_processes)
                logger.info(f"Unregistered process for generation {generation_id}")


def get_video_duration(video_path):
    """Get video duration using ffprobe or fallback."""
    try:
        # Try ffprobe first
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            duration_str = result.stdout.strip()
            return float(duration_str) if duration_str else 0.0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: estimate based on file size and model
    try:
        file_size = os.path.getsize(video_path)
        # Rough estimate: 1MB per second of 720p video
        estimated_duration = max(1, file_size / (1024 * 1024))
        return min(estimated_duration, 30)  # Cap at 30s
    except Exception:
        return 10.0  # Default fallback


def is_model_installed(model_key):
    """Check if model is installed."""
    model = MODELS.get(model_key)
    if not model:
        return False

    try:
        # Check if model directory exists and has required files
        model_path = model["path"]
        return os.path.exists(model_path) and os.path.isdir(model_path)
    except Exception:
        return False


def install_model(model_key, progress_callback):
    """Install model automatically with progress callbacks."""
    model = MODELS[model_key]

    progress_callback(
        {
            "status": "Installing model",
            "message": f"Setting up {model['name']}...",
            "progress": 10,
        }
    )

    try:
        # Create base models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Clone repository
        install_path = model["path"]

        progress_callback(
            {
                "status": "Downloading",
                "message": f"Cloning {model['name']} repository...",
                "progress": 20,
            }
        )

        if os.path.exists(install_path):
            progress_callback(
                {
                    "status": "Installing",
                    "message": "Repository cloned successfully",
                    "progress": 40,
                }
            )
        else:
            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", model["repo"], install_path],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    progress_callback(
                        {
                            "status": "Installing",
                            "message": "Repository cloned successfully",
                            "progress": 40,
                        }
                    )
                else:
                    raise Exception(f"Git clone failed with code {result.returncode}")
            except Exception as e:
                logger.error(f"Git clone failed: {e}")
                # Continue if directory exists
                pass

        # Install Python dependencies
        requirements_path = os.path.join(install_path, model["requirements"])
        python_path = "python"

        progress_callback(
            {
                "status": "Installing dependencies",
                "message": "Installing Python packages...",
                "progress": 50,
            }
        )

        try:
            if os.path.exists(requirements_path):
                subprocess.run(
                    [
                        python_path,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "-r",
                        requirements_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                progress_callback(
                    {
                        "status": "Installing",
                        "message": "Dependencies installed successfully",
                        "progress": 70,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to install requirements: {e}")
            progress_callback(
                {
                    "status": "Installing",
                    "message": "No requirements.txt found",
                    "progress": 70,
                }
            )

        # Download Model Weights
        if DOWNLOADER_AVAILABLE:
            downloader = get_downloader_service()
            if not downloader.is_weight_present(model_key):
                progress_callback(
                    {
                        "status": "Downloading Weights",
                        "message": f"Downloading weights for {model['name']} (~15GB)...",
                        "progress": 75,
                    }
                )
                downloader.download_model_weights(model_key, progress_callback)
            else:
                progress_callback(
                    {
                        "status": "Installing",
                        "message": "Weights already present",
                        "progress": 90,
                    }
                )

        progress_callback(
            {
                "status": "Installing",
                "message": "Model installed successfully",
                "progress": 100,
            }
        )

        return {"success": True, "modelPath": install_path}

    except Exception as e:
        raise Exception(f"Failed to install model: {str(e)}")


def _run_generation_internal(context, progress_callback):
    """Internal generation function used by robust generator."""
    prompt = context.get("prompt", "")
    model = context.get("model", "hunyuan-15")
    duration = context.get("duration", 10)
    aspect_ratio = context.get("aspect_ratio", "16:9")
    negative_prompt = context.get("negative_prompt", "")
    seed = context.get("seed")
    precision = context.get("precision", "bf16")
    generation_id = context.get("generation_id", f"gen_{int(time.time())}")
    input_type = context.get("input_type", "text")
    input_path = context.get("input_path")
    motion_strength = context.get("motion_strength", 0.7)

    # Use tracker if available
    tracker = get_progress_tracker()
    monitor = tracker.get_monitor(generation_id)
    if not monitor:
        monitor = tracker.create_monitor(generation_id, progress_callback)

    try:
        monitor.update_progress(
            ProgressStage.INITIALIZING, "Starting video generation...", 0
        )

        # Use ComfyUI for SVI Pro or I2V/V2V workflows
        if model == "svi-pro-wan22" or input_type in ("image", "video"):
            if COMFYUI_AVAILABLE:
                monitor.update_progress(
                    ProgressStage.LOADING_MODEL, "Using ComfyUI for generation...", 10
                )

                executor = get_comfyui_executor()

                # Progress adapter
                def comfyui_progress(percent, message):
                    if percent >= 0:
                        stage = (
                            ProgressStage.LOADING_MODEL
                            if percent < 20
                            else ProgressStage.PROCESSING_FRAMES
                            if percent < 90
                            else ProgressStage.ENCODING
                        )
                        monitor.update_progress(stage, message, percent)

                result = executor.generate_video(
                    prompt=prompt,
                    duration=duration,
                    aspect_ratio=aspect_ratio,
                    input_type=input_type,
                    input_path=input_path,
                    progress_callback=comfyui_progress,
                    motion_strength=motion_strength,
                    seed=seed if seed else -1,
                )

                if result.get("success"):
                    video_path = result.get("video_path")
                    if video_path and os.path.exists(video_path):
                        file_size = os.path.getsize(video_path)
                        actual_duration = get_video_duration(video_path)

                        monitor.complete("Video generation complete!")

                        return {
                            "success": True,
                            "videoUrl": f"file://{video_path}",
                            "videoPath": video_path,
                            "model": MODELS.get(model, {}).get("name", "ComfyUI"),
                            "duration": actual_duration,
                            "fileSize": file_size,
                            "seed": seed,
                            "error": None,
                            "generation_id": generation_id,
                        }
                else:
                    raise Exception(result.get("error", "ComfyUI generation failed"))
            else:
                raise Exception("ComfyUI not available. Please install dependencies.")

        # Check if model is installed
        installed = is_model_installed(model)

        if not installed:
            monitor.update_progress(
                ProgressStage.LOADING_MODEL,
                "Model not installed. Installing now...",
                10,
            )
            try:
                install_model(model, progress_callback)
            except Exception as install_error:
                monitor.error(f"Failed to install model: {str(install_error)}")
                raise Exception(f"Failed to install model: {str(install_error)}")

        # Get model configuration
        model_config = MODELS[model]
        model_path = model_config["path"]
        script_path = os.path.join(model_path, model_config["script"])

        # Check if script exists
        if not os.path.exists(script_path):
            monitor.error(f"Model script not found: {script_path}")
            raise Exception(f"Model script not found: {script_path}")

        # Generate output path
        timestamp = int(time.time())
        output_filename = f"{model}_{timestamp}.mp4"
        output_path = os.path.join(MODELS_DIR, "outputs", output_filename)

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build model-specific command
        cmd = build_model_command(
            model,
            script_path,
            prompt,
            duration,
            aspect_ratio,
            negative_prompt,
            seed,
            output_path,
            progress_callback,
            precision,
        )

        monitor.update_progress(
            ProgressStage.LOADING_MODEL, "Initializing model...", 20
        )

        # Execute generation with enhanced progress monitoring
        result = execute_video_generation_enhanced(cmd, monitor, model, generation_id)

        if result["success"]:
            # Verify output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    # Get video duration from file
                    actual_duration = get_video_duration(output_path)

                    monitor.complete("Video generation complete!")

                    return {
                        "success": True,
                        "videoUrl": f"file://{output_path}",
                        "videoPath": output_path,
                        "model": model_config["name"],
                        "duration": actual_duration,
                        "fileSize": file_size,
                        "seed": seed,
                        "error": None,
                        "generation_id": generation_id,
                    }
                else:
                    monitor.error("Generated video file is empty")
                    raise Exception("Generated video file is empty")
            else:
                monitor.error("Video generation completed but output file not found")
                raise Exception("Video generation completed but output file not found")
        else:
            monitor.error(result.get("error", "Video generation failed"))
            raise Exception(result.get("error", "Video generation failed"))

    finally:
        # Don't clean up monitor here, let caller handle it or let it persist for status checks
        pass


def unload_all_models():
    """Unload all video models and stop backend servers to free VRAM."""
    success = True
    messages = []

    # Stop ComfyUI
    if COMFYUI_AVAILABLE:
        try:
            executor = get_comfyui_executor()
            if executor.stop_server():
                messages.append("ComfyUI server stopped")
            else:
                success = False
                messages.append("Failed to stop ComfyUI server")
        except Exception as e:
            success = False
            messages.append(f"ComfyUI cleanup error: {e}")

    # No specific cleanup needed for script-based models (run in subprocess)
    # But clean up any tracked processes if we want to be aggressive?
    # For now, just ComfyUI is the big memory hog.

    return {
        "success": success,
        "message": "; ".join(messages) or "No active models to unload",
    }


def generate_video_local(options, progress_callback):
    """Generate video locally with enhanced progress monitoring and error recovery."""

    generation_id = f"gen_{int(time.time())}"
    options["generation_id"] = generation_id

    # Use robust generator if available
    if ROBUST_GENERATOR_AVAILABLE:
        robust_gen = get_robust_generator()
        result_wrapper = robust_gen.generate_with_retry(
            _run_generation_internal, options, progress_callback
        )

        if result_wrapper["success"]:
            return result_wrapper["result"]
        else:
            return {
                "success": False,
                "error": result_wrapper.get("error", "Unknown error"),
                "videoUrl": None,
                "attempts": result_wrapper.get("attempts", 1),
                "recovery_suggestions": result_wrapper.get("recovery_suggestions", {}),
            }
    else:
        # Fallback to direct execution without robust features
        try:
            return _run_generation_internal(options, progress_callback)
        except Exception as e:
            return {"success": False, "error": str(e), "videoUrl": None}


def get_time_ago(timestamp):
    """Get human-readable time ago string."""
    if not timestamp:
        return ""

    try:
        ts = (
            datetime.fromisoformat(timestamp)
            if isinstance(timestamp, str)
            else datetime.fromtimestamp(timestamp / 1000.0)
        )
        now = datetime.now()
        diff = now - ts

        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60}m ago"
        elif diff.seconds < 86400:
            return f"{diff.seconds // 3600}h ago"
        else:
            return f"{diff.seconds // 86400}d ago"
    except Exception:
        return ""


def is_model_installed_sync(model_key):
    """Synchronous check if model is installed."""
    model = MODELS.get(model_key)
    if not model:
        return False

    try:
        model_path = model.get("path")
        if not model_path:
            return False
        return os.path.exists(model_path)
    except Exception:
        return False


# ============ Routes ============


@video_bp.route("/models")
def api_get_models():
    """Get available video generation models."""
    try:
        models = {
            "free": [
                {
                    "id": "hunyuan-15",
                    "name": MODELS["hunyuan-15"]["name"],
                    "mode": "free",
                    "cost_per_video": 0,
                    "is_local": True,
                    "description": "Lightweight model, good quality, runs locally",
                    "max_duration": 30,
                    "max_resolution": "1080p",
                    "installed": is_model_installed_sync("hunyuan-15"),
                },
                {
                    "id": "wan-21",
                    "name": MODELS["wan-21"]["name"],
                    "mode": "free",
                    "cost_per_video": 0,
                    "is_local": True,
                    "description": "Very lightweight, fast inference",
                    "max_duration": 30,
                    "max_resolution": "720p",
                    "installed": is_model_installed_sync("wan-21"),
                },
                {
                    "id": "ltx-video",
                    "name": MODELS["ltx-video"]["name"],
                    "mode": "free",
                    "cost_per_video": 0,
                    "is_local": True,
                    "description": "Real-time generation, ~30 FPS",
                    "max_duration": 15,
                    "max_resolution": "704p",
                    "installed": is_model_installed_sync("ltx-video"),
                },
            ],
            "quality": [
                {
                    "id": "kling-26",
                    "name": "Kling 2.6 Pro",
                    "mode": "quality",
                    "cost_per_video": 2.50,
                    "is_local": False,
                    "description": "Best quality, native audio, cinematic visuals",
                    "max_duration": 10,
                    "max_resolution": "1080p",
                    "coming_soon": True,
                },
                {
                    "id": "veo-31",
                    "name": "Veo 3.1",
                    "mode": "quality",
                    "cost_per_video": 3.50,
                    "is_local": False,
                    "description": "Google's latest, realistic motion",
                    "max_duration": 15,
                    "max_resolution": "1080p",
                    "coming_soon": True,
                },
                {
                    "id": "sora-2",
                    "name": "Sora 2",
                    "mode": "quality",
                    "cost_per_video": 5.00,
                    "is_local": False,
                    "description": "OpenAI's video model, long-form generation",
                    "max_duration": 25,
                    "max_resolution": "1080p",
                    "coming_soon": True,
                },
            ],
        }

        return jsonify({"data": models})
    except Exception as e:
        logger.error(f"Get models error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/generate", methods=["POST"])
def api_generate():
    """Start video generation."""
    try:
        data = request.get_json()
        mode = data.get("mode", "free")
        prompt = data.get("prompt", "")
        model = data.get("model", "hunyuan-15")
        input_type = data.get("input_type", "text")
        duration = data.get("duration", 10)
        aspect_ratio = data.get("aspect_ratio", "16:9")
        negative_prompt = data.get("negative_prompt", "")
        seed = data.get("seed", None)
        precision = data.get("precision", "bf16")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Optimization: Check for existing concurrent generations
        # Use acquire(blocking=False) to quickly fail or queue if needed
        # But here we will wait in the background thread, so just checking lock is enough for UI feedback if we wanted
        # For now, we will handle queueing in the thread.

        # Generate unique ID
        generation_id = f"gen_{int(time.time())}"

        # Store generation request
        generations = _load_generations()
        if not generations or "generations" not in generations:
            generations = {"generations": []}

        generations["generations"].append(
            {
                "id": generation_id,
                "prompt": prompt,
                "mode": mode,
                "model": model,
                "input_type": input_type,
                "status": "pending",
                "progress": 0,
                "created_at": datetime.now().isoformat(),
                "video_url": None,
                "video_path": None,
            }
        )
        _save_generations(generations)

        # Progress callback function
        def progress_callback(progress_data):
            generations = _load_generations()
            if generations and "generations" in generations:
                for gen in generations["generations"]:
                    if gen.get("id") == generation_id:
                        gen.update(progress_data)
                _save_generations(generations)

        # Start generation
        if mode == "quality":
            return jsonify(
                {
                    "error": "Best Quality mode coming soon! Kling 2.6, Veo 3.1, and Sora 2 integration will be added in a future update.",
                    "message": "Use Free mode for now or wait for quality mode release",
                }
            ), 501

        # Free mode - local generation
        def run_generation():
            start_time = time.time()
            success = False
            error_msg = None

            # 1. Check Cache first (Optimization)
            req_hash = generate_request_hash(data)
            cache = _load_cache()

            if req_hash in cache:
                cached_result = cache[req_hash]
                cached_path = cached_result.get("video_path")
                if cached_path and os.path.exists(cached_path):
                    logger.info(f"Returning cached video for hash {req_hash}")
                    progress_callback(
                        {
                            "status": "Complete",
                            "message": "Using cached result",
                            "progress": 100,
                        }
                    )

                    # Update generations record
                    gens = _load_generations()
                    for gen in gens["generations"]:
                        if gen.get("id") == generation_id:
                            gen.update(
                                {
                                    "status": "completed",
                                    "video_url": cached_result.get("video_url"),
                                    "video_path": cached_path,
                                    "duration": cached_result.get("duration"),
                                    "completed_at": datetime.now().isoformat(),
                                    "cached": True,
                                }
                            )
                    _save_generations(gens)
                    return

            # 2. Optimization: Acquire lock to prevent concurrent generations
            progress_callback(
                {"status": "Queued", "message": "Waiting for GPU...", "progress": 0}
            )

            with WindowsKeepAwake():  # Prevent sleep during generation
                with generation_lock:
                    try:
                        result = generate_video_local(
                            {
                                "prompt": prompt,
                                "model": model,
                                "input_type": input_type,
                                "duration": duration,
                                "aspect_ratio": aspect_ratio,
                                "negative_prompt": negative_prompt,
                                "seed": seed,
                                "precision": precision,
                            },
                            progress_callback,
                        )

                        success = result.get("success", False)
                        error_msg = result.get("error")

                        if success:
                            # Update Cache
                            cache[req_hash] = {
                                "video_url": result.get("videoUrl"),
                                "video_path": result.get("videoPath"),
                                "duration": result.get("duration"),
                                "timestamp": time.time(),
                            }
                            _save_cache(cache)

                        # Update result in database
                        generations = _load_generations()
                        if generations and "generations" in generations:
                            for gen in generations["generations"]:
                                if gen.get("id") == generation_id:
                                    gen["status"] = "completed" if success else "failed"
                                    gen["video_url"] = result.get("videoUrl")
                                    gen["video_path"] = result.get("videoPath")
                                    gen["duration"] = result.get("duration")
                                    gen["cost"] = 0
                                    gen["error"] = error_msg
                                    gen["completed_at"] = (
                                        datetime.now().isoformat() if success else None
                                    )

                            _save_generations(generations)

                    except Exception as e:
                        logger.error(f"Generation error: {e}")
                        error_msg = str(e)
                    finally:
                        # Metrics: Record stats
                        if METRICS_AVAILABLE:
                            elapsed_ms = (time.time() - start_time) * 1000
                            get_metrics_service().record_generation(
                                model, elapsed_ms, success, error_msg
                            )

        # Run in thread to not block
        threading.Thread(target=run_generation).start()

        return jsonify(
            {
                "success": True,
                "generation_id": generation_id,
                "status": "pending",
                "message": "Video generation started",
            }
        )

    except Exception as e:
        logger.error(f"Generate video error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/image-to-video", methods=["POST"])
def api_image_to_video():
    """
    Generate video from a still image.

    Request body:
        - image: Base64 encoded image or file path
        - prompt: Text description of desired motion/animation
        - duration: Video duration in seconds (default: 5)
        - aspect_ratio: Output aspect ratio (default: from image)
        - motion_strength: How much motion to add (0.0-1.0, default: 0.7)
        - model: Model to use (default: svi-pro-wan22)
    """
    try:
        data = request.get_json()
        image_data = data.get("image")
        prompt = data.get("prompt", "")
        duration = data.get("duration", 5)
        aspect_ratio = data.get("aspect_ratio", "16:9")
        motion_strength = data.get("motion_strength", 0.7)
        model = data.get("model", "svi-pro-wan22")

        if not image_data:
            return jsonify({"error": "Image is required"}), 400

        if not prompt:
            return jsonify({"error": "Prompt is required for image-to-video"}), 400

        # Handle base64 image
        input_path = None
        if image_data.startswith("data:image"):
            import base64

            # Extract base64 data
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)

            # Save to temp file
            temp_dir = os.path.join(MODELS_DIR, "temp_inputs")
            os.makedirs(temp_dir, exist_ok=True)
            input_path = os.path.join(temp_dir, f"i2v_{int(time.time())}.png")

            with open(input_path, "wb") as f:
                f.write(image_bytes)
        elif os.path.exists(image_data):
            input_path = image_data
        else:
            return jsonify(
                {"error": "Invalid image: must be base64 data URL or file path"}
            ), 400

        generation_id = f"i2v_{int(time.time())}"

        # Store generation request
        generations = _load_generations()
        if not generations or "generations" not in generations:
            generations = {"generations": []}

        generations["generations"].append(
            {
                "id": generation_id,
                "prompt": prompt,
                "mode": "free",
                "model": model,
                "input_type": "image",
                "input_path": input_path,
                "status": "pending",
                "progress": 0,
                "created_at": datetime.now().isoformat(),
                "video_url": None,
            }
        )
        _save_generations(generations)

        def progress_callback(progress_data):
            generations = _load_generations()
            if generations and "generations" in generations:
                for gen in generations["generations"]:
                    if gen.get("id") == generation_id:
                        gen.update(progress_data)
                _save_generations(generations)

        def run_i2v_generation():
            try:
                with WindowsKeepAwake():
                    with generation_lock:
                        result = generate_video_local(
                            {
                                "prompt": prompt,
                                "model": model,
                                "input_type": "image",
                                "input_path": input_path,
                                "duration": duration,
                                "aspect_ratio": aspect_ratio,
                                "motion_strength": motion_strength,
                            },
                            progress_callback,
                        )

                        generations = _load_generations()
                        if generations and "generations" in generations:
                            for gen in generations["generations"]:
                                if gen.get("id") == generation_id:
                                    gen["status"] = (
                                        "completed"
                                        if result.get("success")
                                        else "failed"
                                    )
                                    gen["video_url"] = result.get("videoUrl")
                                    gen["video_path"] = result.get("videoPath")
                                    gen["error"] = result.get("error")
                                    gen["completed_at"] = (
                                        datetime.now().isoformat()
                                        if result.get("success")
                                        else None
                                    )
                            _save_generations(generations)

            except Exception as e:
                logger.error(f"I2V generation error: {e}")

        threading.Thread(target=run_i2v_generation).start()

        return jsonify(
            {
                "success": True,
                "generation_id": generation_id,
                "status": "pending",
                "message": "Image-to-video generation started",
            }
        )

    except Exception as e:
        logger.error(f"I2V error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/video-to-video", methods=["POST"])
def api_video_to_video():
    """
    Transform an existing video with style transfer or enhancement.

    Request body:
        - video: File path to source video
        - prompt: Text description of desired transformation
        - denoise: How much to modify (0.0-1.0, default: 0.6)
        - model: Model to use (default: svi-pro-wan22)
    """
    try:
        data = request.get_json()
        video_path = data.get("video")
        prompt = data.get("prompt", "")
        denoise = data.get("denoise", 0.6)
        model = data.get("model", "svi-pro-wan22")
        aspect_ratio = data.get("aspect_ratio", "16:9")

        if not video_path:
            return jsonify({"error": "Video path is required"}), 400

        if not os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {video_path}"}), 400

        if not prompt:
            return jsonify({"error": "Prompt is required for video-to-video"}), 400

        generation_id = f"v2v_{int(time.time())}"

        # Get source video duration
        source_duration = get_video_duration(video_path) or 5

        # Store generation request
        generations = _load_generations()
        if not generations or "generations" not in generations:
            generations = {"generations": []}

        generations["generations"].append(
            {
                "id": generation_id,
                "prompt": prompt,
                "mode": "free",
                "model": model,
                "input_type": "video",
                "input_path": video_path,
                "status": "pending",
                "progress": 0,
                "created_at": datetime.now().isoformat(),
                "video_url": None,
            }
        )
        _save_generations(generations)

        def progress_callback(progress_data):
            generations = _load_generations()
            if generations and "generations" in generations:
                for gen in generations["generations"]:
                    if gen.get("id") == generation_id:
                        gen.update(progress_data)
                _save_generations(generations)

        def run_v2v_generation():
            try:
                with WindowsKeepAwake():
                    with generation_lock:
                        result = generate_video_local(
                            {
                                "prompt": prompt,
                                "model": model,
                                "input_type": "video",
                                "input_path": video_path,
                                "duration": source_duration,
                                "aspect_ratio": aspect_ratio,
                                "denoise": denoise,
                            },
                            progress_callback,
                        )

                        generations = _load_generations()
                        if generations and "generations" in generations:
                            for gen in generations["generations"]:
                                if gen.get("id") == generation_id:
                                    gen["status"] = (
                                        "completed"
                                        if result.get("success")
                                        else "failed"
                                    )
                                    gen["video_url"] = result.get("videoUrl")
                                    gen["video_path"] = result.get("videoPath")
                                    gen["error"] = result.get("error")
                                    gen["completed_at"] = (
                                        datetime.now().isoformat()
                                        if result.get("success")
                                        else None
                                    )
                            _save_generations(generations)

            except Exception as e:
                logger.error(f"V2V generation error: {e}")

        threading.Thread(target=run_v2v_generation).start()

        return jsonify(
            {
                "success": True,
                "generation_id": generation_id,
                "status": "pending",
                "message": "Video-to-video generation started",
            }
        )

    except Exception as e:
        logger.error(f"V2V error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/status/<generation_id>")
def api_get_status(generation_id):
    """Get generation status."""
    try:
        generation = _find_generation(generation_id)

        # Check active tracker first
        if PROGRESS_AVAILABLE:
            tracker = get_progress_tracker()
            monitor = tracker.get_monitor(generation_id)
            if monitor:
                response = {
                    "id": generation_id,
                    "status": monitor.current_stage.value,
                    "progress": monitor.last_progress,
                    "message": "Processing...",
                }
                if generation:
                    response.update(
                        _attach_stream_url(
                            {
                                "prompt": generation.get("prompt"),
                                "model": generation.get("model"),
                                "created_at": generation.get("created_at"),
                                "completed_at": generation.get("completed_at"),
                                "video_url": generation.get("video_url"),
                                "video_path": generation.get("video_path"),
                                "duration": generation.get("duration"),
                                "error": generation.get("error"),
                                "id": generation.get("id"),
                            }
                        )
                    )
                return jsonify(response)

        # Fallback to stored generations
        if generation:
            return jsonify(
                _attach_stream_url(
                    {
                        "id": generation.get("id"),
                        "status": generation.get("status", "unknown"),
                        "progress": generation.get("progress", 0),
                        "video_url": generation.get("video_url"),
                        "video_path": generation.get("video_path"),
                        "model": generation.get("model"),
                        "created_at": generation.get("created_at"),
                        "completed_at": generation.get("completed_at"),
                        "duration": generation.get("duration"),
                        "error": generation.get("error"),
                        "prompt": generation.get("prompt"),
                    }
                )
            )

        return jsonify({"error": "Generation not found"}), 404

    except Exception as e:
        logger.error(f"Get status error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/result/<generation_id>")
def api_get_result(generation_id):
    """Get generation result."""
    try:
        generation = _find_generation(generation_id)

        if generation:
            return jsonify(
                _attach_stream_url(
                    {
                        "id": generation.get("id"),
                        "prompt": generation.get("prompt"),
                        "mode": generation.get("mode"),
                        "status": generation.get("status"),
                        "video_url": generation.get("video_url"),
                        "video_path": generation.get("video_path"),
                        "duration_seconds": generation.get("duration"),
                        "cost": generation.get("cost", 0),
                        "created_at": generation.get("created_at"),
                        "completed_at": generation.get("completed_at"),
                        "time_ago": get_time_ago(generation.get("completed_at")),
                    }
                )
            )

        return jsonify({"error": "Generation not found"}), 404

    except Exception as e:
        logger.error(f"Get result error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/cancel/<generation_id>", methods=["DELETE"])
def api_cancel_generation(generation_id):
    """Cancel generation."""
    try:
        # 1. Kill the actual process if it exists
        with active_processes_lock:
            if generation_id in active_processes:
                process = active_processes[generation_id]
                try:
                    import psutil

                    parent = psutil.Process(process.pid)
                    # Kill children first (important for GPU worker processes)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                    logger.info(f"Killed process tree for generation {generation_id}")
                except Exception as e:
                    logger.warning(f"Failed to kill process tree via psutil: {e}")
                    # Fallback to simple kill
                    process.kill()

                del active_processes[generation_id]

        # 2. Stop monitoring
        if PROGRESS_AVAILABLE:
            tracker = get_progress_tracker()
            tracker.remove_monitor(generation_id)

        generations = _load_generations()

        if generations and "generations" in generations:
            for gen in generations["generations"]:
                if gen.get("id") == generation_id:
                    gen["status"] = "cancelled"
                    gen["completed_at"] = datetime.now().isoformat()
                    _save_generations(generations)

                    return jsonify(
                        {
                            "success": True,
                            "message": "Generation cancelled and process terminated",
                        }
                    )

        return jsonify({"error": "Generation not found"}), 404

    except Exception as e:
        logger.error(f"Cancel generation error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/file/<generation_id>")
def api_video_file(generation_id):
    """Serve generated video files."""
    generation = _find_generation(generation_id)
    if not generation:
        return jsonify({"error": "Generation not found"}), 404

    video_path = generation.get("video_path")
    if not video_path:
        return jsonify({"error": "Video file not available"}), 404

    path = Path(video_path)
    if not path.exists():
        return jsonify({"error": "Video file missing"}), 404

    return send_file(path, mimetype="video/mp4", as_attachment=False)


@video_bp.route("/history")
def api_get_history():
    """Get generation history."""
    try:
        generations = _load_generations()
        all_generations = (generations or {}).get("generations", [])
        history = [
            gen
            for gen in all_generations
            if gen.get("status") in ["completed", "failed"]
        ]

        return jsonify(
            {
                "data": sorted(
                    [_attach_stream_url(item) for item in history],
                    key=lambda x: x.get("created_at", ""),
                    reverse=True,
                )
            }
        )

    except Exception as e:
        logger.error(f"Get history error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/clear-history", methods=["POST"])
def api_clear_history():
    """Clear generation history."""
    try:
        _save_generations({"generations": []})
        return jsonify({"success": True, "message": "History cleared"})

    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/metrics")
def api_get_metrics():
    """Get performance metrics."""
    if not METRICS_AVAILABLE:
        return jsonify({"error": "Metrics service not available"}), 501

    try:
        return jsonify(get_metrics_service().get_metrics())
    except Exception as e:
        logger.error(f"Get metrics error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/gpu-info")
def api_gpu_info():
    """Detect GPU capabilities and recommend model."""
    try:
        # Try to detect NVIDIA GPU via nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            # Parse GPU info
            gpu_line = result.stdout.strip().split("\n")[0]  # Get first GPU
            parts = gpu_line.split(", ")

            if len(parts) >= 2:
                gpu_name = parts[0].strip()
                vram_str = parts[1].strip()

                # Extract VRAM in GB (format: "XXXX MiB" or "XX GB")
                if "MiB" in vram_str:
                    vram_mb = int(vram_str.split()[0])
                    vram_gb = round(vram_mb / 1024, 1)
                else:
                    vram_gb = float(vram_str.split()[0])

                # Determine tier and recommended model
                if vram_gb >= 16:
                    tier = "high_end"
                    recommended_model = "svi-pro-wan22"
                    can_run_local = True
                    estimated_time = "2-3 min"
                elif vram_gb >= 8:
                    tier = "mid_range"
                    recommended_model = "ltx-video"
                    can_run_local = True
                    estimated_time = "30-60 sec"
                else:
                    tier = "low_end"
                    recommended_model = "cloud-fal"
                    can_run_local = False
                    estimated_time = "Use cloud (30-60 sec)"

                return jsonify(
                    {
                        "gpu_name": gpu_name,
                        "vram_gb": vram_gb,
                        "tier": tier,
                        "recommended_model": recommended_model,
                        "can_run_local": can_run_local,
                        "estimated_time_10s": estimated_time,
                    }
                )

        # No NVIDIA GPU detected
        return jsonify(
            {
                "gpu_name": "None (CPU only)",
                "vram_gb": 0,
                "tier": "no_gpu",
                "recommended_model": "cloud-fal",
                "can_run_local": False,
                "estimated_time_10s": "Cloud generation recommended",
            }
        )

    except FileNotFoundError:
        # nvidia-smi not found
        return jsonify(
            {
                "gpu_name": "None (nvidia-smi not found)",
                "vram_gb": 0,
                "tier": "no_gpu",
                "recommended_model": "cloud-fal",
                "can_run_local": False,
                "estimated_time_10s": "Cloud generation recommended",
            }
        )
    except Exception as e:
        logger.error(f"GPU detection error: {e}")
        return jsonify(
            {
                "gpu_name": "Unknown",
                "vram_gb": 0,
                "tier": "unknown",
                "recommended_model": "cloud-fal",
                "can_run_local": False,
                "error": str(e),
            }
        )


@video_bp.route("/installation-status")
def api_installation_status():
    """Check if ComfyUI/local models are installed."""
    try:
        # Check which models are installed
        installed_models = {}

        for model_id, model_config in MODELS.items():
            # Skip cloud models
            if model_config.get("tier") == "cloud":
                continue

            model_path = model_config.get("path")
            if model_path and os.path.exists(model_path) and os.path.isdir(model_path):
                installed_models[model_id] = True
            else:
                installed_models[model_id] = False

        # Check if any local model is installed
        any_installed = any(installed_models.values())

        return jsonify(
            {"installed": any_installed, "models": installed_models, "status": "idle"}
        )

    except Exception as e:
        logger.error(f"Installation status error: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/cloud-generate", methods=["POST"])
def api_cloud_generate():
    """Generate video via cloud API (FAL.ai or RunPod)."""
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        provider = data.get("provider", "cloud-fal")  # Default to FAL.ai
        duration = data.get("duration", 10)
        aspect_ratio = data.get("aspect_ratio", "16:9")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Validate provider
        if provider not in MODELS or MODELS[provider].get("tier") != "cloud":
            return jsonify({"error": f"Invalid cloud provider: {provider}"}), 400

        # Generate unique job ID
        generation_id = str(uuid.uuid4())

        # Initialize generation tracking
        active_generations[generation_id] = {
            "status": "initializing",
            "progress": 0,
            "prompt": prompt,
            "provider": provider,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "created_at": datetime.now().isoformat(),
            "stage": "cloud_init",
        }

        # Start cloud generation in background thread
        def run_cloud_generation():
            try:
                if provider == "cloud-fal":
                    result = generate_video_fal(
                        prompt, duration, aspect_ratio, generation_id
                    )
                elif provider == "cloud-runpod":
                    result = generate_video_runpod(
                        prompt, duration, aspect_ratio, generation_id
                    )
                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                active_generations[generation_id].update(
                    {
                        "status": "completed",
                        "progress": 100,
                        "video_path": result["video_path"],
                        "completed_at": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                logger.error(f"Cloud generation failed: {e}")
                active_generations[generation_id].update(
                    {
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat(),
                    }
                )

        threading.Thread(target=run_cloud_generation, daemon=True).start()

        return jsonify(
            {
                "success": True,
                "generation_id": generation_id,
                "message": f"Cloud generation started via {MODELS[provider]['name']}",
            }
        )

    except Exception as e:
        logger.error(f"Cloud generation error: {e}")
        return jsonify({"error": str(e)}), 500


def generate_video_fal(
    prompt: str, duration: int, aspect_ratio: str, generation_id: str
) -> dict:
    """Generate video via FAL.ai API."""
    import requests

    # Update progress
    active_generations[generation_id].update(
        {
            "stage": "cloud_submitting",
            "progress": 10,
            "message": "Submitting to FAL.ai...",
        }
    )

    # FAL.ai API endpoint (using fast-svd model as example)
    api_url = "https://fal.run/fal-ai/fast-svd"

    # Get API key from environment
    api_key = os.environ.get("FAL_API_KEY")
    if not api_key:
        raise Exception("FAL_API_KEY not configured in environment")

    # Prepare request
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}

    payload = {
        "prompt": prompt,
        "video_size": aspect_ratio,
        "num_frames": duration * 8,  # ~8 fps
    }

    # Submit generation request
    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    request_id = result.get("request_id")

    if not request_id:
        # Synchronous response with video URL
        video_url = result.get("video", {}).get("url")
        if not video_url:
            raise Exception("No video URL in FAL.ai response")
    else:
        # Poll for completion
        active_generations[generation_id].update(
            {
                "stage": "cloud_processing",
                "progress": 30,
                "message": "Generating video on FAL.ai...",
            }
        )

        status_url = f"https://fal.run/fal-ai/fast-svd/requests/{request_id}"

        while True:
            status_response = requests.get(status_url, headers=headers, timeout=30)
            status_response.raise_for_status()
            status_data = status_response.json()

            status = status_data.get("status")

            if status == "completed":
                video_url = status_data.get("output", {}).get("video", {}).get("url")
                if not video_url:
                    raise Exception("No video URL in completed response")
                break
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                raise Exception(f"FAL.ai generation failed: {error}")

            # Update progress based on status
            progress = 30 + (status_data.get("progress", 0) * 0.5)  # 30-80%
            active_generations[generation_id].update(
                {
                    "progress": int(progress),
                    "message": f"Processing... {status_data.get('progress', 0)}%",
                }
            )

            time.sleep(2)

    # Download video
    active_generations[generation_id].update(
        {
            "stage": "cloud_downloading",
            "progress": 85,
            "message": "Downloading video...",
        }
    )

    video_response = requests.get(video_url, timeout=120)
    video_response.raise_for_status()

    # Save video to local storage
    output_dir = os.path.join(OUTPUT_DIR, "cloud")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{generation_id}.mp4")

    with open(output_path, "wb") as f:
        f.write(video_response.content)

    active_generations[generation_id].update(
        {"stage": "finalizing", "progress": 95, "message": "Finalizing..."}
    )

    return {
        "video_path": output_path,
        "cost": 0.50,  # FAL.ai typical cost
    }


def generate_video_runpod(
    prompt: str, duration: int, aspect_ratio: str, generation_id: str
) -> dict:
    """Generate video via RunPod serverless API."""
    import requests

    # Update progress
    active_generations[generation_id].update(
        {
            "stage": "cloud_submitting",
            "progress": 10,
            "message": "Submitting to RunPod...",
        }
    )

    # Get API key and endpoint from environment
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")

    if not api_key or not endpoint_id:
        raise Exception("RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not configured")

    # RunPod serverless endpoint
    api_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "input": {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "model": "svi-pro-wan22",
        }
    }

    # Submit job
    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    job_id = result.get("id")

    if not job_id:
        raise Exception("No job ID in RunPod response")

    # Poll for completion
    active_generations[generation_id].update(
        {
            "stage": "cloud_processing",
            "progress": 30,
            "message": "Generating video on RunPod...",
        }
    )

    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    while True:
        status_response = requests.get(status_url, headers=headers, timeout=30)
        status_response.raise_for_status()
        status_data = status_response.json()

        status = status_data.get("status")

        if status == "COMPLETED":
            output = status_data.get("output")
            video_url = output.get("video_url") if output else None
            if not video_url:
                raise Exception("No video URL in completed response")
            break
        elif status == "FAILED":
            error = status_data.get("error", "Unknown error")
            raise Exception(f"RunPod generation failed: {error}")

        # Update progress
        progress = 30 + (status_data.get("progress", 0) * 0.5)  # 30-80%
        active_generations[generation_id].update(
            {
                "progress": int(progress),
                "message": f"Processing on RunPod... {status_data.get('progress', 0)}%",
            }
        )

        time.sleep(3)

    # Download video
    active_generations[generation_id].update(
        {
            "stage": "cloud_downloading",
            "progress": 85,
            "message": "Downloading video...",
        }
    )

    video_response = requests.get(video_url, timeout=120)
    video_response.raise_for_status()

    # Save video
    output_dir = os.path.join(OUTPUT_DIR, "cloud")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{generation_id}.mp4")

    with open(output_path, "wb") as f:
        f.write(video_response.content)

    active_generations[generation_id].update(
        {"stage": "finalizing", "progress": 95, "message": "Finalizing..."}
    )

    # Calculate cost (RunPod charges by GPU time)
    generation_time = status_data.get("executionTime", 120)  # seconds
    cost_per_second = 0.002  # ~$2/1000s for H100
    cost = generation_time * cost_per_second

    return {"video_path": output_path, "cost": round(cost, 2)}


@video_bp.route("/disk-space")
def api_disk_space():
    """Check available disk space for model downloads."""
    try:
        if os.name == "nt":  # Windows
            import ctypes

            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p("C:\\"),
                None,
                ctypes.pointer(total_bytes),
                ctypes.pointer(free_bytes),
            )
            free_gb = free_bytes.value / (1024**3)
            total_gb = total_bytes.value / (1024**3)
        else:  # Unix/Linux
            stat = os.statvfs(os.path.expanduser("~"))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)

        # SVI Pro 2.0 + Wan 2.2 + ComfyUI requires ~100GB
        min_required_gb = 100

        return jsonify(
            {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "sufficient": free_gb >= min_required_gb,
                "min_required_gb": min_required_gb,
                "message": f"{round(free_gb, 2)}GB free of {round(total_gb, 2)}GB total",
            }
        )

    except Exception as e:
        logger.error(f"Disk space check error: {e}")
        return jsonify({"error": str(e)}), 500


# Global installation tracker
installation_status = {}


@video_bp.route("/install-comfyui", methods=["POST"])
def api_install_comfyui():
    """Trigger ComfyUI installation with progress tracking."""
    try:
        data = request.get_json() or {}
        model_id = data.get("model", "svi-pro-wan22")

        if model_id not in MODELS:
            return jsonify({"error": f"Unknown model: {model_id}"}), 400

        model_config = MODELS[model_id]

        if model_config.get("tier") == "cloud":
            return jsonify({"error": "Cloud models don't require installation"}), 400

        # Check if already installing
        if installation_status.get("status") == "installing":
            return jsonify(
                {
                    "success": False,
                    "message": "Installation already in progress",
                    "installation_id": installation_status.get("installation_id"),
                }
            )

        # Generate installation ID
        installation_id = str(uuid.uuid4())

        # Initialize installation tracking
        installation_status.update(
            {
                "installation_id": installation_id,
                "status": "installing",
                "progress": 0,
                "stage": "initializing",
                "message": "Starting installation...",
                "model": model_id,
                "started_at": datetime.now().isoformat(),
            }
        )

        # Start installation in background
        def run_installation():
            try:
                install_comfyui_model(model_id)
                installation_status.update(
                    {
                        "status": "completed",
                        "progress": 100,
                        "stage": "complete",
                        "message": "Installation complete!",
                        "completed_at": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Installation failed: {e}")
                installation_status.update(
                    {
                        "status": "failed",
                        "progress": 0,
                        "stage": "failed",
                        "error": str(e),
                        "message": f"Installation failed: {str(e)}",
                        "failed_at": datetime.now().isoformat(),
                    }
                )

        threading.Thread(target=run_installation, daemon=True).start()

        return jsonify(
            {
                "success": True,
                "installation_id": installation_id,
                "message": "Installation started",
            }
        )

    except Exception as e:
        logger.error(f"Install ComfyUI error: {e}")
        return jsonify({"error": str(e)}), 500


def install_comfyui_model(model_id: str):
    """Install ComfyUI and download model weights."""
    import subprocess
    import sys

    model_config = MODELS[model_id]
    comfyui_path = os.path.join(MODELS_DIR, "comfyui")

    # Stage 1: Clone ComfyUI (0-20%)
    if not os.path.exists(comfyui_path):
        installation_status.update(
            {
                "stage": "cloning_comfyui",
                "progress": 5,
                "message": "Cloning ComfyUI repository...",
            }
        )

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/comfyanonymous/ComfyUI.git",
                comfyui_path,
            ],
            check=True,
            timeout=300,
        )

        installation_status.update(
            {"progress": 20, "message": "ComfyUI cloned successfully"}
        )

    # Stage 2: Install Python dependencies (20-35%)
    installation_status.update(
        {
            "stage": "dependencies",
            "progress": 20,
            "message": "Installing Python dependencies...",
        }
    )

    requirements_path = os.path.join(comfyui_path, "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_path],
            check=True,
            timeout=600,
        )

    installation_status.update({"progress": 35, "message": "Dependencies installed"})

    # Stage 3: Download model weights (35-95%)
    if model_id == "svi-pro-wan22":
        # Download SVI Pro 2.0 (35-65%)
        installation_status.update(
            {
                "stage": "downloading_svi",
                "progress": 35,
                "message": "Downloading SVI Pro 2.0 weights (12.5GB)...",
            }
        )

        from huggingface_hub import snapshot_download

        svi_path = os.path.join(comfyui_path, "models", "checkpoints", "svi-pro-20")
        snapshot_download(
            repo_id="vita-video-gen/svi-model",
            local_dir=svi_path,
            allow_patterns=["version-2.0/*"],
            resume_download=True,
        )

        installation_status.update(
            {"progress": 65, "message": "SVI Pro 2.0 downloaded"}
        )

        # Download Wan 2.2 (65-95%)
        installation_status.update(
            {
                "stage": "downloading_wan",
                "progress": 65,
                "message": "Downloading Wan 2.2 weights (3.8GB)...",
            }
        )

        wan_path = os.path.join(comfyui_path, "models", "checkpoints", "wan22")
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-T2V-A14B", local_dir=wan_path, resume_download=True
        )

        installation_status.update({"progress": 95, "message": "Wan 2.2 downloaded"})

    elif model_id == "ltx-video":
        # Download LTX Video (35-95%)
        installation_status.update(
            {
                "stage": "downloading_ltx",
                "progress": 35,
                "message": "Downloading LTX Video weights (8GB)...",
            }
        )

        from huggingface_hub import snapshot_download

        ltx_path = os.path.join(comfyui_path, "models", "checkpoints", "ltx-video")
        snapshot_download(
            repo_id="Lightricks/LTX-Video", local_dir=ltx_path, resume_download=True
        )

        installation_status.update({"progress": 95, "message": "LTX Video downloaded"})

    # Stage 4: Finalize (95-100%)
    installation_status.update(
        {"stage": "finalizing", "progress": 98, "message": "Finalizing installation..."}
    )

    # Mark installation complete in models config
    time.sleep(1)  # Brief pause for finalization

    installation_status.update({"progress": 100, "message": "Installation complete!"})


@video_bp.route("/generate-with-fallback", methods=["POST"])
def api_generate_with_fallback():
    """Generate video with automatic cloud fallback if local fails."""
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        preferred_model = data.get("model", "svi-pro-wan22")
        duration = data.get("duration", 10)
        aspect_ratio = data.get("aspect_ratio", "16:9")
        auto_fallback = data.get("auto_fallback", True)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Generate unique job ID
        generation_id = str(uuid.uuid4())

        # Initialize generation tracking
        active_generations[generation_id] = {
            "status": "initializing",
            "progress": 0,
            "prompt": prompt,
            "preferred_model": preferred_model,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "created_at": datetime.now().isoformat(),
            "stage": "initializing",
            "auto_fallback": auto_fallback,
        }

        # Start generation with fallback in background
        def run_generation_with_fallback():
            try:
                # Try local generation first if it's a local model
                if MODELS.get(preferred_model, {}).get("tier") != "cloud":
                    try:
                        active_generations[generation_id].update(
                            {
                                "stage": "local_attempt",
                                "progress": 5,
                                "message": f"Attempting local generation with {MODELS[preferred_model]['name']}...",
                            }
                        )

                        # TODO: Call actual local generation function here
                        # For now, simulate local generation attempt
                        result = generate_local_video(
                            preferred_model,
                            prompt,
                            duration,
                            aspect_ratio,
                            generation_id,
                        )

                        active_generations[generation_id].update(
                            {
                                "status": "completed",
                                "progress": 100,
                                "video_path": result["video_path"],
                                "model_used": preferred_model,
                                "cost": 0,
                                "completed_at": datetime.now().isoformat(),
                            }
                        )
                        return

                    except Exception as local_error:
                        logger.warning(f"Local generation failed: {local_error}")

                        if not auto_fallback:
                            # Don't fallback, just fail
                            raise local_error

                        # Fallback to cloud
                        active_generations[generation_id].update(
                            {
                                "stage": "cloud_fallback",
                                "progress": 10,
                                "message": "Local generation failed. Switching to cloud (free this time)...",
                                "local_error": str(local_error),
                            }
                        )

                        # Use FAL.ai as fallback
                        result = generate_video_fal(
                            prompt, duration, aspect_ratio, generation_id
                        )

                        active_generations[generation_id].update(
                            {
                                "status": "completed",
                                "progress": 100,
                                "video_path": result["video_path"],
                                "model_used": "cloud-fal",
                                "cost": 0,  # Free for automatic fallback
                                "fallback_used": True,
                                "completed_at": datetime.now().isoformat(),
                            }
                        )

                else:
                    # Cloud model requested directly
                    provider = preferred_model
                    if provider == "cloud-fal":
                        result = generate_video_fal(
                            prompt, duration, aspect_ratio, generation_id
                        )
                    elif provider == "cloud-runpod":
                        result = generate_video_runpod(
                            prompt, duration, aspect_ratio, generation_id
                        )
                    else:
                        raise ValueError(f"Unknown cloud provider: {provider}")

                    active_generations[generation_id].update(
                        {
                            "status": "completed",
                            "progress": 100,
                            "video_path": result["video_path"],
                            "model_used": provider,
                            "cost": result["cost"],
                            "completed_at": datetime.now().isoformat(),
                        }
                    )

            except Exception as e:
                logger.error(f"Generation with fallback failed: {e}")
                active_generations[generation_id].update(
                    {
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat(),
                    }
                )

        threading.Thread(target=run_generation_with_fallback, daemon=True).start()

        return jsonify(
            {
                "success": True,
                "generation_id": generation_id,
                "message": "Generation started with automatic fallback enabled",
            }
        )

    except Exception as e:
        logger.error(f"Generate with fallback error: {e}")
        return jsonify({"error": str(e)}), 500


def generate_local_video(
    model_id: str,
    prompt: str,
    duration: int,
    aspect_ratio: str,
    generation_id: str,
    input_type: str = "text",
    input_path: str = None,
    motion_strength: float = 0.7,
    **kwargs,
) -> dict:
    """
    Generate video using local ComfyUI model.

    Args:
        model_id: Model identifier (svi-pro-wan22, etc.)
        prompt: Text description of the video
        duration: Video duration in seconds
        aspect_ratio: Output aspect ratio (16:9, 9:16, 1:1)
        generation_id: Unique generation identifier
        input_type: "text", "image", or "video"
        input_path: Path to input image/video (for I2V/V2V)
        motion_strength: Motion intensity for I2V (0.0-1.0)

    Returns:
        Dict with video_path and cost
    """
    if not COMFYUI_AVAILABLE:
        raise Exception("ComfyUI executor not available. Please install dependencies.")

    comfyui_path = os.path.join(MODELS_DIR, "comfyui")

    if not os.path.exists(comfyui_path):
        raise Exception(
            "ComfyUI not installed. Please install first via /api/video/install-comfyui"
        )

    # Get the executor
    executor = get_comfyui_executor()

    # Progress callback to update active_generations
    def progress_callback(percent: int, message: str):
        if percent < 0:
            # Status message without percent
            active_generations[generation_id].update({"message": message})
        else:
            stage = (
                "loading_model"
                if percent < 20
                else "processing"
                if percent < 90
                else "encoding"
            )
            active_generations[generation_id].update(
                {"stage": stage, "progress": percent, "message": message}
            )

    # Update initial progress
    active_generations[generation_id].update(
        {
            "stage": "initializing",
            "progress": 5,
            "message": f"Initializing {MODELS[model_id]['name']}...",
        }
    )

    # Handle input upload for I2V/V2V
    uploaded_input = None
    if input_type in ("image", "video") and input_path:
        if os.path.exists(input_path):
            progress_callback(8, f"Uploading input {input_type}...")
            uploaded_input = executor.upload_input(input_path, input_type)
        else:
            raise Exception(f"Input file not found: {input_path}")

    # Generate video using ComfyUI
    result = executor.generate_video(
        prompt=prompt,
        duration=duration,
        aspect_ratio=aspect_ratio,
        input_type=input_type,
        input_path=uploaded_input,
        progress_callback=progress_callback,
        motion_strength=motion_strength,
        **kwargs,
    )

    if not result.get("success"):
        raise Exception(result.get("error", "Video generation failed"))

    video_path = result.get("video_path")

    # Copy to output directory with consistent naming
    output_dir = os.path.join(OUTPUT_DIR, "local")
    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, f"{generation_id}.mp4")

    if video_path and os.path.exists(video_path):
        import shutil

        shutil.copy2(video_path, final_output_path)
    else:
        raise Exception("Video generation completed but output file not found")

    # Final progress update
    active_generations[generation_id].update(
        {"stage": "completed", "progress": 100, "message": "Video generation complete!"}
    )

    return {"video_path": final_output_path, "cost": 0}
