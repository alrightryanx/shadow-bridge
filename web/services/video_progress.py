"""
Video Generation Progress Monitoring
Provides real-time progress tracking and status updates for video generation
"""

import re
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ProgressStage(Enum):
    """Video generation progress stages."""

    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    PROCESSING_FRAMES = "processing_frames"
    ENCODING = "encoding"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"


class VideoProgressMonitor:
    """Monitors and tracks video generation progress."""

    def __init__(
        self, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.progress_callback = progress_callback
        self.start_time = time.time()
        self.last_progress = 0
        self.stage_start_time = time.time()
        self.current_stage = ProgressStage.INITIALIZING
        self.total_frames = 0
        self.processed_frames = 0

    def update_progress(
        self,
        stage: ProgressStage,
        message: str,
        progress: float,
        frame_info: Optional[Dict[str, Any]] = None,
    ):
        """Update progress with stage-specific information."""
        # Calculate stage duration
        stage_duration = time.time() - self.stage_start_time
        total_duration = time.time() - self.start_time

        # Only update if significant progress change
        if progress > self.last_progress + 1.0:
            progress_data = {
                "stage": stage.value,
                "status": stage.value.title(),
                "message": message,
                "progress": int(progress),
                "stage_progress": self._calculate_stage_progress(stage, progress),
                "total_duration_ms": int(total_duration * 1000),
                "stage_duration_ms": int(stage_duration * 1000),
                "eta_ms": self._calculate_eta(progress),
            }

            # Add frame-specific information
            if frame_info:
                progress_data.update(frame_info)

            # Send callback
            if self.progress_callback:
                self.progress_callback(progress_data)

            self.last_progress = progress

        # Update stage if changed
        if stage != self.current_stage:
            self.current_stage = stage
            self.stage_start_time = time.time()
            logger.info(f"Progress stage changed to: {stage.value}")

    def _calculate_stage_progress(
        self, stage: ProgressStage, overall_progress: float
    ) -> float:
        """Calculate progress within current stage."""
        stage_ranges = {
            ProgressStage.INITIALIZING: (0, 10),
            ProgressStage.LOADING_MODEL: (10, 20),
            ProgressStage.PROCESSING_FRAMES: (20, 80),
            ProgressStage.ENCODING: (80, 95),
            ProgressStage.FINALIZING: (95, 100),
        }

        if stage not in stage_ranges:
            return 0.0

        start, end = stage_ranges[stage]
        if overall_progress <= start:
            return 0.0
        elif overall_progress >= end:
            return 100.0
        else:
            return ((overall_progress - start) / (end - start)) * 100

    def _calculate_eta(self, progress: float) -> int:
        """Calculate estimated time remaining in milliseconds."""
        if progress <= 0:
            return 0

        elapsed = time.time() - self.start_time
        if progress >= 100:
            return 0

        eta_seconds = (elapsed * (100 - progress)) / progress
        return int(eta_seconds * 1000)

    def parse_model_output(
        self, line: str, model_type: str
    ) -> Optional[Dict[str, Any]]:
        """Parse model output for progress information."""
        line = line.strip().lower()

        if not line:
            return None

        # Common progress patterns
        if any(keyword in line for keyword in ["step", "frame", "%"]):
            progress_info = self._parse_common_progress(line)
            if progress_info:
                return self._create_progress_update(progress_info, model_type)

        # Model-specific patterns
        if model_type == "hunyuan-15":
            return self._parse_hunyuan_output(line)
        elif model_type == "ltx-video":
            return self._parse_ltx_output(line)
        elif model_type == "wan-21":
            return self._parse_wan_output(line)

        return None

    def _parse_common_progress(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse common progress patterns."""
        # Percentage pattern: "75%" or "75.0%"
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%?", line)
        if percent_match:
            progress = float(percent_match.group(1))
            return {"type": "percentage", "value": progress}

        # Step pattern: "step 50/100" or "50/100"
        step_match = re.search(r"step\s*(\d+)/(\d+)|(\d+)/(\d+)", line)
        if step_match:
            if step_match.group(1):  # "step 50/100" format
                current = int(step_match.group(1))
                total = int(step_match.group(2))
            else:  # "50/100" format
                current = int(step_match.group(3))
                total = int(step_match.group(4))

            if total > 0:
                progress = (current / total) * 100
                return {
                    "type": "step",
                    "value": progress,
                    "current": current,
                    "total": total,
                }

        # Frame pattern: "frame 50/100" or "processing frame 50"
        frame_match = re.search(r"frame\s*(\d+)/(\d+)|frame\s*(\d+)", line)
        if frame_match:
            if frame_match.group(1):  # "frame 50/100" format
                current = int(frame_match.group(1))
                total = int(frame_match.group(2))
                progress = (current / total) * 100 if total > 0 else 0
                return {
                    "type": "frame",
                    "value": progress,
                    "current": current,
                    "total": total,
                }
            else:  # "processing frame 50" format
                current = int(frame_match.group(3))
                return {"type": "frame", "value": None, "current": current}

        return None

    def _parse_hunyuan_output(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse HunyuanVideo specific output."""
        if "loading" in line or "initializing" in line:
            return {
                "type": "stage",
                "stage": "loading_model",
                "message": "Loading HunyuanVideo model...",
            }

        if "generating" in line:
            return {
                "type": "stage",
                "stage": "processing_frames",
                "message": "Generating video frames...",
            }

        if "encoding" in line:
            return {
                "type": "stage",
                "stage": "encoding",
                "message": "Encoding video...",
            }

        if "complete" in line or "done" in line:
            return {
                "type": "stage",
                "stage": "completed",
                "message": "Generation complete!",
            }

        return None

    def _parse_ltx_output(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse LTX Video specific output."""
        if "real-time" in line or "realtime" in line:
            return {
                "type": "stage",
                "stage": "processing_frames",
                "message": "Real-time generation...",
            }

        if "fps" in line:
            fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
            if fps_match:
                fps = float(fps_match.group(1))
                return {
                    "type": "fps",
                    "value": fps,
                    "message": f"Generating at {fps:.1f} FPS",
                }

        if "finalizing" in line or "saving" in line:
            return {
                "type": "stage",
                "stage": "finalizing",
                "message": "Finalizing video...",
            }

        return None

    def _parse_wan_output(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse Wan 2.1 specific output."""
        if "inference" in line:
            return {
                "type": "stage",
                "stage": "processing_frames",
                "message": "Running inference...",
            }

        if "post-processing" in line:
            return {
                "type": "stage",
                "stage": "encoding",
                "message": "Post-processing video...",
            }

        if "optimizing" in line:
            return {
                "type": "stage",
                "stage": "finalizing",
                "message": "Optimizing output...",
            }

        return None

    def _create_progress_update(
        self, progress_info: Dict[str, Any], model_type: str
    ) -> Dict[str, Any]:
        """Create a progress update from parsed information."""
        info_type = progress_info.get("type")

        if info_type in ["percentage", "step", "frame"]:
            progress_value = progress_info.get("value", 0)
            if progress_value is not None:
                # Map to 20-80% range for actual generation
                mapped_progress = 20 + (progress_value * 0.6)

                message_parts = []
                if progress_info.get("current") and progress_info.get("total"):
                    message_parts.append(
                        f"{progress_info['current']}/{progress_info['total']}"
                    )

                if info_type == "frame" and progress_info.get("fps"):
                    message_parts.append(f"{progress_info['fps']:.1f} FPS")

                return {
                    "type": "progress",
                    "value": mapped_progress,
                    "message": " ".join(message_parts)
                    if message_parts
                    else f"Processing... {int(progress_value)}%",
                }

        elif info_type == "stage":
            stage_name = progress_info.get("stage")
            stage_progress_map = {
                "loading_model": 15,
                "processing_frames": 50,
                "encoding": 85,
                "finalizing": 95,
                "completed": 100,
            }

            return {
                "type": "stage_update",
                "stage": stage_name,
                "value": stage_progress_map.get(stage_name, 0),
                "message": progress_info.get("message", "Processing..."),
            }

        elif info_type == "fps":
            # FPS update during generation
            fps = progress_info.get("value", 0)
            return {
                "type": "performance",
                "fps": fps,
                "message": f"Generating at {fps:.1f} FPS",
            }

        return {"type": "unknown", "message": "Processing..."}

    def complete(self, final_message: str = "Video generation complete!"):
        """Mark generation as completed."""
        total_duration = time.time() - self.start_time
        self.update_progress(
            ProgressStage.COMPLETED,
            final_message,
            100.0,
            {
                "total_duration_ms": int(total_duration * 1000),
                "completed_at": int(time.time() * 1000),
            },
        )

    def error(self, error_message: str):
        """Mark generation as failed."""
        total_duration = time.time() - self.start_time
        self.update_progress(
            ProgressStage.ERROR,
            f"Error: {error_message}",
            self.last_progress,
            {"error": error_message, "total_duration_ms": int(total_duration * 1000)},
        )


class ProgressTracker:
    """Thread-safe progress tracking for multiple concurrent generations."""

    def __init__(self):
        self.active_monitors = {}
        self.lock = threading.Lock()

    def create_monitor(
        self,
        generation_id: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> VideoProgressMonitor:
        """Create a new progress monitor for a generation."""
        with self.lock:
            monitor = VideoProgressMonitor(progress_callback)
            self.active_monitors[generation_id] = monitor
            return monitor

    def get_monitor(self, generation_id: str) -> Optional[VideoProgressMonitor]:
        """Get an existing progress monitor."""
        with self.lock:
            return self.active_monitors.get(generation_id)

    def remove_monitor(self, generation_id: str):
        """Remove a progress monitor."""
        with self.lock:
            self.active_monitors.pop(generation_id, None)

    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get progress for all active generations."""
        with self.lock:
            result = {}
            for gen_id, monitor in self.active_monitors.items():
                result[gen_id] = {
                    "stage": monitor.current_stage.value,
                    "progress": monitor.last_progress,
                    "duration_ms": int((time.time() - monitor.start_time) * 1000),
                }
            return result


# Global progress tracker
_global_tracker = ProgressTracker()


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker."""
    return _global_tracker
