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
    from web.services.video_error_handling import get_robust_generator, RobustVideoGenerator
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
            
            # Run every hour
            time.sleep(3600)

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    logger.info("Artifact cleanup thread started")

# Initialize cleanup
start_artifact_cleanup_thread()

# Model configuration
MODELS = {
    "hunyuan-15": {
        "name": "HunyuanVideo 1.5 (8.3B)",
        "repo": "https://github.com/Tencent-Hunyuan/HunyuanVideo.git",
        "script": "generate.py",
        "requirements": "requirements.txt",
        "path": os.path.join(MODELS_DIR, "hunyuan", ""),
    },
    "wan-21": {
        "name": "Wan 2.1 (1.3B)",
        "repo": "https://github.com/Wan-Video/Wan2.1.git",
        "script": "generate.py",
        "requirements": "requirements.txt",
        "path": os.path.join(MODELS_DIR, "wan", ""),
    },
    "ltx-video": {
        "name": "LTX Video",
        "repo": "https://github.com/Lightricks/LTX-Video.git",
        "script": "generate.py",
        "requirements": "requirements.txt",
        "path": os.path.join(MODELS_DIR, "ltx", ""),
    },
}

# In-memory storage for video generations (simple JSON file)
GENERATIONS_FILE = os.path.join(
    os.path.expanduser("~"), ".shadowai", "video_generations.json"
)

# Cache mapping for prompts
CACHE_FILE = os.path.join(
    os.path.expanduser("~"), ".shadowai", "video_cache.json"
)

# Process registry for recovery
PROCESS_REGISTRY_FILE = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")), ".shadowai", "active_video_processes.json"
)

def _save_process_registry(registry):
    """Save active process IDs to disk for recovery."""
    try:
        # Popen objects aren't JSON serializable, so we only save PIDs
        serializable = {gid: p.pid for gid, p in registry.items() if p.poll() is None}
        with open(PROCESS_REGISTRY_FILE, "w") as f:
            json.dump(serializable, f)
    except Exception as e:
        logger.warning(f"Failed to save process registry: {e}")

def _load_process_registry():
    """Load process registry from disk."""
    try:
        if os.path.exists(PROCESS_REGISTRY_FILE):
            with open(PROCESS_REGISTRY_FILE, "r") as f:
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
        if os.name == 'nt':
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
                )
                logger.info("Windows Keep-Awake enabled")
            except Exception as e:
                logger.warning(f"Failed to set thread execution state: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.name == 'nt':
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
        "precision": options.get("precision")
    }
    dump = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(dump.encode('utf-8')).hexdigest()

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
        monitor.update_progress(ProgressStage.PROCESSING_FRAMES, 'Starting generation process...', 25)
        
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
                                progress_value
                            )
                            
                        elif info_type == "stage_update":
                            stage_name = progress_info.get("stage")
                            stage_map = {
                                "loading_model": ProgressStage.LOADING_MODEL,
                                "processing_frames": ProgressStage.PROCESSING_FRAMES,
                                "encoding": ProgressStage.ENCODING,
                                "finalizing": ProgressStage.FINALIZING,
                                "completed": ProgressStage.COMPLETED
                            }
                            stage = stage_map.get(stage_name, ProgressStage.PROCESSING_FRAMES)
                            progress_value = progress_info.get("value", 0)
                            monitor.update_progress(
                                stage,
                                progress_info.get("message", "Processing..."),
                                progress_value
                            )
                            
                        elif info_type == "performance":
                            # Performance update (FPS, etc.)
                            message = progress_info.get("message", "Processing...")
                            monitor.update_progress(
                                ProgressStage.PROCESSING_FRAMES,
                                message,
                                monitor.last_progress
                            )

        # Wait for completion
        return_code = process.wait()

        if return_code == 0:
            monitor.update_progress(ProgressStage.FINALIZING, 'Finalizing video...', 95)
            return {"success": True, "output": "\n".join(output_lines)}
        else:
            error_output = "\n".join(output_lines[-10:])  # Last 10 lines for error context
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
    prompt = context.get('prompt', '')
    model = context.get('model', 'hunyuan-15')
    duration = context.get('duration', 10)
    aspect_ratio = context.get('aspect_ratio', '16:9')
    negative_prompt = context.get('negative_prompt', '')
    seed = context.get('seed')
    precision = context.get('precision', 'bf16')
    generation_id = context.get('generation_id', f"gen_{int(time.time())}")

    # Use tracker if available
    tracker = get_progress_tracker()
    monitor = tracker.get_monitor(generation_id)
    if not monitor:
        monitor = tracker.create_monitor(generation_id, progress_callback)

    try:
        monitor.update_progress(ProgressStage.INITIALIZING, 'Starting video generation...', 0)

        # Check if model is installed
        installed = is_model_installed(model)

        if not installed:
            monitor.update_progress(ProgressStage.LOADING_MODEL, 'Model not installed. Installing now...', 10)
            try:
                install_model(model, progress_callback)
            except Exception as install_error:
                monitor.error(f"Failed to install model: {str(install_error)}")
                raise Exception(f"Failed to install model: {str(install_error)}")

        # Get model configuration
        model_config = MODELS[model]
        model_path = model_config['path']
        script_path = os.path.join(model_path, model_config['script'])

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
        cmd = build_model_command(model, script_path, prompt, duration, aspect_ratio, 
                               negative_prompt, seed, output_path, progress_callback, precision)

        monitor.update_progress(ProgressStage.LOADING_MODEL, 'Initializing model...', 20)

        # Execute generation with enhanced progress monitoring
        result = execute_video_generation_enhanced(cmd, monitor, model, generation_id)

        if result['success']:
            # Verify output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    # Get video duration from file
                    actual_duration = get_video_duration(output_path)
                    
                    monitor.complete('Video generation complete!')
                    
                    return {
                        'success': True,
                        'videoUrl': f"file://{output_path}",
                        'videoPath': output_path,
                        'model': model_config['name'],
                        'duration': actual_duration,
                        'fileSize': file_size,
                        'seed': seed,
                        'error': None,
                        'generation_id': generation_id
                    }
                else:
                    monitor.error("Generated video file is empty")
                    raise Exception("Generated video file is empty")
            else:
                monitor.error("Video generation completed but output file not found")
                raise Exception("Video generation completed but output file not found")
        else:
            monitor.error(result.get('error', 'Video generation failed'))
            raise Exception(result.get('error', 'Video generation failed'))
            
    finally:
        # Don't clean up monitor here, let caller handle it or let it persist for status checks
        pass


def generate_video_local(options, progress_callback):
    """Generate video locally with enhanced progress monitoring and error recovery."""
    
    generation_id = f"gen_{int(time.time())}"
    options['generation_id'] = generation_id
    
    # Use robust generator if available
    if ROBUST_GENERATOR_AVAILABLE:
        robust_gen = get_robust_generator()
        result_wrapper = robust_gen.generate_with_retry(
            _run_generation_internal,
            options,
            progress_callback
        )
        
        if result_wrapper['success']:
            return result_wrapper['result']
        else:
            return {
                'success': False,
                'error': result_wrapper.get('error', 'Unknown error'),
                'videoUrl': None,
                'attempts': result_wrapper.get('attempts', 1),
                'recovery_suggestions': result_wrapper.get('recovery_suggestions', {})
            }
    else:
        # Fallback to direct execution without robust features
        try:
            return _run_generation_internal(options, progress_callback)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'videoUrl': None
            }


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
                    progress_callback({"status": "Complete", "message": "Using cached result", "progress": 100})
                    
                    # Update generations record
                    gens = _load_generations()
                    for gen in gens["generations"]:
                        if gen.get("id") == generation_id:
                            gen.update({
                                "status": "completed",
                                "video_url": cached_result.get("video_url"),
                                "video_path": cached_path,
                                "duration": cached_result.get("duration"),
                                "completed_at": datetime.now().isoformat(),
                                "cached": True
                            })
                    _save_generations(gens)
                    return

            # 2. Optimization: Acquire lock to prevent concurrent generations
            progress_callback({"status": "Queued", "message": "Waiting for GPU...", "progress": 0})
            
            with WindowsKeepAwake(): # Prevent sleep during generation
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
                                "timestamp": time.time()
                            }
                            _save_cache(cache)

                        # Update result in database
                        generations = _load_generations()
                        if generations and "generations" in generations:
                            for gen in generations["generations"]:
                                if gen.get("id") == generation_id:
                                    gen["status"] = (
                                        "completed" if success else "failed"
                                    )
                                    gen["video_url"] = result.get("videoUrl")
                                    gen["video_path"] = result.get("videoPath")
                                    gen["duration"] = result.get("duration")
                                    gen["cost"] = 0
                                    gen["error"] = error_msg
                                    gen["completed_at"] = (
                                        datetime.now().isoformat()
                                        if success
                                        else None
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
        
        return jsonify({
            "success": True, 
            "generation_id": generation_id,
            "status": "pending",
            "message": "Video generation started"
        })

    except Exception as e:
        logger.error(f"Generate video error: {e}")
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

                    return jsonify({"success": True, "message": "Generation cancelled and process terminated"})

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
