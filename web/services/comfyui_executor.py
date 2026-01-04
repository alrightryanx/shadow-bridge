"""
ComfyUI Executor Service

Handles communication with ComfyUI server for video generation.
Supports text-to-video, image-to-video, and video-to-video workflows.
"""

import os
import json
import uuid
import time
import logging
import requests
import websocket
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# ComfyUI server configuration
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Workflow templates directory
WORKFLOWS_DIR = Path(__file__).parent.parent / "comfyui_workflows"

# Resolution presets
ASPECT_RATIOS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:3": (960, 720),
    "3:4": (720, 960),
    "21:9": (1680, 720),
}


class ComfyUIExecutor:
    """Executes ComfyUI workflows for video generation."""

    def __init__(self, comfyui_path: str = None):
        self.comfyui_path = comfyui_path or os.path.join(os.environ.get("MODELS_DIR", "C:/models"), "comfyui")
        self.client_id = str(uuid.uuid4())
        self._progress_callbacks: Dict[str, Callable] = {}
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

    def is_server_running(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False

    def start_server(self) -> bool:
        """Start ComfyUI server if not running."""
        if self.is_server_running():
            logger.info("ComfyUI server already running")
            return True

        import subprocess
        import sys

        main_py = os.path.join(self.comfyui_path, "main.py")
        if not os.path.exists(main_py):
            raise FileNotFoundError(f"ComfyUI not found at {self.comfyui_path}")

        logger.info("Starting ComfyUI server...")

        # Start server in background
        subprocess.Popen(
            [sys.executable, main_py, "--listen", COMFYUI_HOST, "--port", str(COMFYUI_PORT)],
            cwd=self.comfyui_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Wait for server to start
        for _ in range(30):
            time.sleep(1)
            if self.is_server_running():
                logger.info("ComfyUI server started successfully")
                return True

        raise TimeoutError("ComfyUI server failed to start within 30 seconds")

    def load_workflow(self, workflow_name: str) -> Dict:
        """Load a workflow template from disk."""
        workflow_path = WORKFLOWS_DIR / f"{workflow_name}.json"
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_name}")

        with open(workflow_path, "r") as f:
            return json.load(f)

    def prepare_workflow(
        self,
        workflow_name: str,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        input_image: str = None,
        input_video: str = None,
        motion_strength: float = 0.7,
        seed: int = -1,
        **kwargs
    ) -> Dict:
        """Prepare a workflow with the given parameters."""
        template = self.load_workflow(workflow_name)

        # Calculate dimensions from aspect ratio
        width, height = ASPECT_RATIOS.get(aspect_ratio, (1280, 720))

        # Calculate frames from duration (24 FPS)
        fps = template.get("defaults", {}).get("FPS", 24)
        frames = duration * fps

        # Generate output prefix
        output_prefix = f"shadowai_{int(time.time())}"

        # Replacement values
        replacements = {
            "PROMPT": prompt,
            "NEGATIVE_PROMPT": template.get("defaults", {}).get("NEGATIVE_PROMPT", ""),
            "WIDTH": width,
            "HEIGHT": height,
            "FRAMES": frames,
            "FPS": fps,
            "SEED": seed if seed >= 0 else int(time.time()) % 2**32,
            "OUTPUT_PREFIX": output_prefix,
            "MOTION_STRENGTH": motion_strength,
            "DENOISE": kwargs.get("denoise", 0.6),
            "MAX_FRAMES": kwargs.get("max_frames", 120),
        }

        if input_image:
            replacements["INPUT_IMAGE"] = input_image
        if input_video:
            replacements["INPUT_VIDEO"] = input_video

        # Deep copy and replace placeholders
        workflow = self._replace_placeholders(template["nodes"], replacements)

        return {
            "prompt": workflow,
            "client_id": self.client_id,
            "extra_data": {
                "output_prefix": output_prefix,
                "expected_frames": frames
            }
        }

    def _replace_placeholders(self, obj: Any, replacements: Dict) -> Any:
        """Recursively replace {{PLACEHOLDER}} in workflow."""
        if isinstance(obj, str):
            for key, value in replacements.items():
                obj = obj.replace(f"{{{{{key}}}}}", str(value))
            return obj
        elif isinstance(obj, dict):
            return {k: self._replace_placeholders(v, replacements) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_placeholders(item, replacements) for item in obj]
        else:
            return obj

    def queue_prompt(self, workflow: Dict) -> str:
        """Queue a workflow for execution."""
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json=workflow,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Failed to queue prompt: {response.text}")

        result = response.json()
        return result.get("prompt_id")

    def get_history(self, prompt_id: str) -> Dict:
        """Get execution history for a prompt."""
        response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}

    def get_output_path(self, prompt_id: str) -> Optional[str]:
        """Get the output video path from execution history."""
        history = self.get_history(prompt_id)

        if prompt_id not in history:
            return None

        outputs = history[prompt_id].get("outputs", {})
        for node_id, node_output in outputs.items():
            if "gifs" in node_output:
                # VHS_VideoCombine outputs to "gifs" key
                for video in node_output["gifs"]:
                    filename = video.get("filename")
                    subfolder = video.get("subfolder", "")
                    if filename:
                        return os.path.join(self.comfyui_path, "output", subfolder, filename)
            elif "videos" in node_output:
                for video in node_output["videos"]:
                    filename = video.get("filename")
                    subfolder = video.get("subfolder", "")
                    if filename:
                        return os.path.join(self.comfyui_path, "output", subfolder, filename)

        return None

    def start_progress_monitor(self, prompt_id: str, callback: Callable[[int, str], None]):
        """Start monitoring progress via WebSocket."""
        self._progress_callbacks[prompt_id] = callback

        if not self._running:
            self._running = True
            self._ws_thread = threading.Thread(target=self._ws_listener, daemon=True)
            self._ws_thread.start()

    def _ws_listener(self):
        """WebSocket listener for progress updates."""
        ws_url = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws?clientId={self.client_id}"

        def on_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "progress":
                    progress_data = data.get("data", {})
                    value = progress_data.get("value", 0)
                    max_val = progress_data.get("max", 100)
                    percent = int((value / max_val) * 100) if max_val > 0 else 0

                    # Notify all callbacks
                    for prompt_id, callback in list(self._progress_callbacks.items()):
                        callback(percent, f"Processing frame {value}/{max_val}")

                elif msg_type == "executing":
                    node = data.get("data", {}).get("node")
                    if node:
                        for prompt_id, callback in list(self._progress_callbacks.items()):
                            callback(-1, f"Executing node: {node}")

                elif msg_type == "execution_cached":
                    for prompt_id, callback in list(self._progress_callbacks.items()):
                        callback(50, "Using cached results...")

            except Exception as e:
                logger.error(f"WebSocket message error: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status, close_msg):
            logger.info("WebSocket connection closed")

        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

    def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        input_type: str = "text",
        input_path: str = None,
        progress_callback: Callable[[int, str], None] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a video using ComfyUI.

        Args:
            prompt: Text description of the video
            duration: Video duration in seconds
            aspect_ratio: Output aspect ratio (16:9, 9:16, 1:1, etc.)
            input_type: "text", "image", or "video"
            input_path: Path to input image/video (required for I2V/V2V)
            progress_callback: Function(percent, message) for progress updates
            **kwargs: Additional parameters (motion_strength, denoise, seed, etc.)

        Returns:
            Dict with "success", "video_path", "error" keys
        """
        try:
            # Ensure server is running
            if not self.is_server_running():
                if progress_callback:
                    progress_callback(5, "Starting ComfyUI server...")
                self.start_server()

            # Select workflow based on input type
            if input_type == "image":
                if not input_path:
                    raise ValueError("input_path required for image-to-video")
                workflow_name = "svi_wan22_i2v"
                kwargs["input_image"] = input_path
            elif input_type == "video":
                if not input_path:
                    raise ValueError("input_path required for video-to-video")
                workflow_name = "wan22_v2v"
                kwargs["input_video"] = input_path
            else:
                workflow_name = "svi_wan22_t2v"

            if progress_callback:
                progress_callback(10, f"Loading {workflow_name} workflow...")

            # Prepare workflow
            workflow = self.prepare_workflow(
                workflow_name,
                prompt=prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                **kwargs
            )

            if progress_callback:
                progress_callback(15, "Queueing generation...")

            # Queue the prompt
            prompt_id = self.queue_prompt(workflow)

            if progress_callback:
                self.start_progress_monitor(prompt_id, progress_callback)
                progress_callback(20, "Generation started...")

            # Poll for completion
            max_wait = duration * 30  # 30 seconds per second of video
            start_time = time.time()

            while time.time() - start_time < max_wait:
                history = self.get_history(prompt_id)

                if prompt_id in history:
                    status = history[prompt_id].get("status", {})

                    if status.get("completed"):
                        output_path = self.get_output_path(prompt_id)

                        if output_path and os.path.exists(output_path):
                            if progress_callback:
                                progress_callback(100, "Video generation complete!")

                            # Cleanup
                            if prompt_id in self._progress_callbacks:
                                del self._progress_callbacks[prompt_id]

                            return {
                                "success": True,
                                "video_path": output_path,
                                "prompt_id": prompt_id
                            }
                        else:
                            raise Exception("Generation completed but output not found")

                    elif status.get("status_str") == "error":
                        error_msg = status.get("messages", [["", "Unknown error"]])[-1][-1]
                        raise Exception(f"Generation failed: {error_msg}")

                time.sleep(2)

            raise TimeoutError(f"Generation timed out after {max_wait} seconds")

        except Exception as e:
            logger.error(f"Video generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "video_path": None
            }

    def upload_input(self, file_path: str, input_type: str = "image") -> str:
        """Upload an input file to ComfyUI."""
        filename = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            files = {
                "image": (filename, f, "image/png" if input_type == "image" else "video/mp4")
            }
            data = {
                "type": "input",
                "subfolder": "",
                "overwrite": "true"
            }

            response = requests.post(
                f"{COMFYUI_URL}/upload/image",
                files=files,
                data=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("name", filename)
            else:
                raise Exception(f"Upload failed: {response.text}")


# Singleton instance
_executor: Optional[ComfyUIExecutor] = None


def get_comfyui_executor() -> ComfyUIExecutor:
    """Get the singleton ComfyUI executor instance."""
    global _executor
    if _executor is None:
        _executor = ComfyUIExecutor()
    return _executor
