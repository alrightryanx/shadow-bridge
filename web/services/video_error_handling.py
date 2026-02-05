"""
Video Generation Error Handling and Recovery System
Provides robust error handling, retry logic, and recovery mechanisms
"""

import time
import logging
import subprocess
import os
import psutil
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class VideoGenerationError(Exception):
    """Custom exception for video generation errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.recoverable = recoverable


class ErrorType(Enum):
    """Types of video generation errors."""

    MODEL_NOT_FOUND = "model_not_found"
    SCRIPT_NOT_FOUND = "script_not_found"
    PERMISSION_DENIED = "permission_denied"
    OUT_OF_MEMORY = "out_of_memory"
    GPU_ERROR = "gpu_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    PROCESS_FAILED = "process_failed"
    DISK_SPACE = "disk_space"
    PYTHON_ERROR = "python_error"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_attempts: int = 3
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    backoff_multiplier: float = 2.0
    jitter_ms: int = 500
    retryable_errors: List[str] = None

    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = [
                ErrorType.TIMEOUT.value,
                ErrorType.OUT_OF_MEMORY.value,
                ErrorType.GPU_ERROR.value,
                ErrorType.PROCESS_FAILED.value,
            ]


class VideoErrorRecovery:
    """Handles error detection, classification, and recovery strategies."""

    def __init__(self):
        self.system_checks = SystemResourceChecker()

    def classify_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> VideoGenerationError:
        """Classify and wrap exceptions with detailed error information."""
        error_message = str(error).lower()

        # Permission errors
        if any(
            keyword in error_message
            for keyword in ["permission denied", "access denied", "unauthorized"]
        ):
            return VideoGenerationError(
                f"Permission denied: {str(error)}",
                ErrorType.PERMISSION_DENIED.value,
                {
                    "suggestion": "Check file/directory permissions",
                    "recoverable": False,
                },
            )

        # Out of memory errors
        if any(
            keyword in error_message
            for keyword in ["out of memory", "memory", "ram", "cuda out of memory"]
        ):
            memory_info = self.system_checks.get_memory_info()
            return VideoGenerationError(
                f"Out of memory: {str(error)}",
                ErrorType.OUT_OF_MEMORY.value,
                {
                    "available_memory_gb": memory_info["available_gb"],
                    "required_memory_gb": context.get("estimated_memory_gb", 0),
                    "suggestion": "Close other applications or reduce video quality",
                },
            )

        # GPU errors
        if any(
            keyword in error_message
            for keyword in ["gpu", "cuda", "device error", "driver"]
        ):
            gpu_info = self.system_checks.get_gpu_info()
            return VideoGenerationError(
                f"GPU error: {str(error)}",
                ErrorType.GPU_ERROR.value,
                {
                    "gpu_available": gpu_info["available"],
                    "gpu_memory_gb": gpu_info["memory_gb"],
                    "suggestion": "Update GPU drivers or reduce video resolution",
                },
            )

        # Disk space errors
        if any(
            keyword in error_message
            for keyword in ["disk space", "no space", "storage full"]
        ):
            disk_info = self.system_checks.get_disk_info()
            return VideoGenerationError(
                f"Disk space error: {str(error)}",
                ErrorType.DISK_SPACE.value,
                {
                    "available_gb": disk_info["available_gb"],
                    "required_gb": context.get("estimated_disk_gb", 1),
                    "suggestion": "Free up disk space or change output location",
                },
            )

        # Timeout errors
        if any(keyword in error_message for keyword in ["timeout", "timed out"]):
            return VideoGenerationError(
                f"Operation timeout: {str(error)}",
                ErrorType.TIMEOUT.value,
                {"suggestion": "Increase timeout or reduce video complexity"},
            )

        # Python/dependency errors
        if any(
            keyword in error_message
            for keyword in ["module", "import", "pip", "python"]
        ):
            return VideoGenerationError(
                f"Python/dependency error: {str(error)}",
                ErrorType.DEPENDENCY_ERROR.value,
                {
                    "suggestion": "Check Python environment and install missing dependencies"
                },
            )

        # Process execution errors
        if isinstance(error, subprocess.CalledProcessError):
            return VideoGenerationError(
                f"Process failed: {error}",
                ErrorType.PROCESS_FAILED.value,
                {
                    "return_code": error.returncode,
                    "stdout": error.stdout,
                    "stderr": error.stderr,
                    "suggestion": "Check model installation and dependencies",
                },
            )

        # Unknown error
        return VideoGenerationError(
            f"Unknown error: {str(error)}",
            ErrorType.UNKNOWN_ERROR.value,
            {"original_exception": str(error), "recoverable": True},
        )

    def suggest_recovery_action(self, error: VideoGenerationError) -> Dict[str, Any]:
        """Suggest recovery actions based on error type."""
        suggestions = {
            "can_retry": error.recoverable,
            "auto_recovery": False,
            "user_action": "",
            "technical_details": {},
        }

        if error.error_code == ErrorType.OUT_OF_MEMORY.value:
            suggestions.update(
                {
                    "auto_recovery": True,
                    "user_action": "Reducing video quality to use less memory",
                    "technical_details": {
                        "action": "reduce_quality",
                        "new_duration": min(
                            30, error.details.get("original_duration", 10)
                        ),
                        "new_resolution": "720p",
                    },
                }
            )

        elif error.error_code == ErrorType.GPU_ERROR.value:
            if error.details.get("gpu_available", False):
                suggestions.update(
                    {
                        "auto_recovery": True,
                        "user_action": "Switching to CPU mode (slower but more compatible)",
                        "technical_details": {"action": "use_cpu"},
                    }
                )
            else:
                suggestions.update(
                    {
                        "user_action": "GPU drivers need to be updated or reinstalled",
                        "technical_details": {"action": "check_drivers"},
                    }
                )

        elif error.error_code == ErrorType.DISK_SPACE.value:
            suggestions.update(
                {
                    "user_action": f"Free up at least {error.details.get('required_gb', 1)}GB of disk space",
                    "technical_details": {"action": "cleanup_disk"},
                }
            )

        elif error.error_code == ErrorType.PERMISSION_DENIED.value:
            suggestions.update(
                {
                    "user_action": "Run as administrator or check folder permissions",
                    "technical_details": {"action": "fix_permissions"},
                }
            )

        elif error.error_code == ErrorType.DEPENDENCY_ERROR.value:
            suggestions.update(
                {
                    "auto_recovery": True,
                    "user_action": "Attempting to install missing dependencies",
                    "technical_details": {"action": "install_deps"},
                }
            )

        return suggestions



def parse_nvidia_memory_mb(value_str: str) -> int:
    """Parse memory string like '16384 MiB' to MB as int."""
    # Remove common suffixes and extract numeric part
    numeric = ''.join(c for c in value_str.split()[0] if c.isdigit())
    return int(numeric) if numeric else 0


class SystemResourceChecker:
    """Checks system resources and constraints."""

    def __init__(self):
        self.last_check = 0
        self.cached_memory = None
        self.cached_gpu = None
        self.cached_disk = None
        self.check_interval = 30  # Cache for 30 seconds

    def get_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval and self.cached_memory:
            return self.cached_memory

        try:
            memory = psutil.virtual_memory()
            info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent_used": memory.percent,
            }
            self.cached_memory = info
            self.last_check = current_time
            return info
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent_used": 100}

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval and self.cached_gpu:
            return self.cached_gpu

        info = {
            "available": False,
            "memory_gb": 0,
            "name": "Unknown",
            "driver_version": "Unknown",
        }

        try:
            # Try to detect GPU using nvidia-smi
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines and lines[0].strip():
                    parts = [p.strip() for p in lines[0].split(",")]
                    if len(parts) >= 3:
                        info.update(
                            {
                                "available": True,
                                "name": parts[0],
                                "memory_gb": parse_nvidia_memory_mb(parts[1]) / 1024,  # Convert MiB to GB
                                "free_memory_gb": parse_nvidia_memory_mb(parts[2]) / 1024,
                            }
                        )

            # Try to get driver version
            driver_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if driver_result.returncode == 0:
                info["driver_version"] = driver_result.stdout.strip()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # nvidia-smi not available, try other methods
            try:
                import torch

                if torch.cuda.is_available():
                    info.update(
                        {
                            "available": True,
                            "name": torch.cuda.get_device_name(0),
                            "memory_gb": torch.cuda.get_device_properties(
                                0
                            ).total_memory
                            / (1024**3),
                        }
                    )
            except ImportError:
                pass

        self.cached_gpu = info
        self.last_check = current_time
        return info

    def get_disk_info(self, path: str = None) -> Dict[str, float]:
        """Get disk space information."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval and self.cached_disk:
            return self.cached_disk

        try:
            # Use current working directory or specified path
            check_path = path or os.getcwd()
            disk = psutil.disk_usage(check_path)
            info = {
                "total_gb": disk.total / (1024**3),
                "available_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3),
                "percent_used": (disk.used / disk.total) * 100,
            }
            self.cached_disk = info
            self.last_check = current_time
            return info
        except Exception as e:
            logger.warning(f"Failed to get disk info: {e}")
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent_used": 100}

    def check_requirements(
        self, estimated_memory_gb: float = 0, estimated_disk_gb: float = 0
    ) -> Dict[str, Any]:
        """Check if system meets requirements."""
        memory = self.get_memory_info()
        gpu = self.get_gpu_info()
        disk = self.get_disk_info()

        checks = {
            "memory_ok": memory["available_gb"] >= estimated_memory_gb,
            "gpu_ok": gpu["available"],
            "disk_ok": disk["available_gb"] >= estimated_disk_gb,
            "recommendations": [],
        }

        # Generate recommendations
        if not checks["memory_ok"]:
            memory_short = estimated_memory_gb - memory["available_gb"]
            checks["recommendations"].append(f"Need {memory_short:.1f}GB more RAM")

        if not checks["gpu_ok"]:
            checks["recommendations"].append(
                "GPU not available - generation will be slower"
            )

        if not checks["disk_ok"]:
            disk_short = estimated_disk_gb - disk["available_gb"]
            checks["recommendations"].append(f"Need {disk_short:.1f}GB more disk space")

        return checks


class RobustVideoGenerator:
    """Robust video generation with error handling and recovery."""

    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.error_recovery = VideoErrorRecovery()
        self.system_checker = SystemResourceChecker()

    def generate_with_retry(
        self,
        generation_func: Callable,
        context: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Generate video with comprehensive error handling and retry logic."""
        last_error = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                # Pre-flight checks
                if attempt > 0:
                    # Recover from previous error
                    if last_error and last_error.recoverable:
                        recovery = self.error_recovery.suggest_recovery_action(
                            last_error
                        )
                        if recovery["auto_recovery"]:
                            context = self._apply_recovery(context, recovery)
                            if progress_callback:
                                progress_callback(
                                    {
                                        "status": "Recovering",
                                        "message": recovery["user_action"],
                                        "progress": 0,
                                    }
                                )

                # Check system resources
                resource_checks = self.system_checker.check_requirements(
                    context.get("estimated_memory_gb", 8),
                    context.get("estimated_disk_gb", 2),
                )

                if not all([resource_checks["memory_ok"], resource_checks["disk_ok"]]):
                    raise VideoGenerationError(
                        "System resources insufficient",
                        ErrorType.OUT_OF_MEMORY.value,
                        {
                            "checks": resource_checks,
                            "recommendations": resource_checks["recommendations"],
                        },
                    )

                # Attempt generation
                if progress_callback and attempt > 0:
                    progress_callback(
                        {
                            "status": "Retrying",
                            "message": f"Attempt {attempt + 1}/{self.retry_config.max_attempts}",
                            "progress": 0,
                        }
                    )

                result = generation_func(context, progress_callback)

                # Success - return result
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1,
                    "warnings": []
                    if attempt == 0
                    else [f"Recovered from error on attempt {attempt + 1}"],
                }

            except Exception as e:
                # Classify error
                video_error = self.error_recovery.classify_error(e, context)
                last_error = video_error

                logger.error(f"Generation attempt {attempt + 1} failed: {video_error}")

                # Check if error is retryable
                if (
                    not video_error.recoverable
                    or video_error.error_code not in self.retry_config.retryable_errors
                ):
                    logger.error(f"Non-retryable error: {video_error.error_code}")
                    break

                # Check if we have more attempts
                if attempt < self.retry_config.max_attempts - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        self.retry_config.base_delay_ms
                        * (self.retry_config.backoff_multiplier**attempt),
                        self.retry_config.max_delay_ms,
                    )

                    # Add jitter to avoid thundering herd
                    import random

                    jitter = random.randint(0, self.retry_config.jitter_ms)
                    total_delay = (delay + jitter) / 1000

                    if progress_callback:
                        progress_callback(
                            {
                                "status": "Waiting",
                                "message": f"Retrying in {total_delay:.1f}s...",
                                "progress": 0,
                            }
                        )

                    logger.info(
                        f"Retrying in {total_delay:.1f}s (attempt {attempt + 2})"
                    )
                    time.sleep(total_delay)
                else:
                    logger.error(
                        f"All {self.retry_config.max_attempts} attempts exhausted"
                    )

        # All attempts failed
        recovery = (
            self.error_recovery.suggest_recovery_action(last_error)
            if last_error
            else {}
        )

        return {
            "success": False,
            "error": str(last_error) if last_error else "All attempts failed",
            "error_code": last_error.error_code if last_error else None,
            "attempts": self.retry_config.max_attempts,
            "recovery_suggestions": recovery,
            "last_error_details": last_error.details if last_error else {},
        }

    def _apply_recovery(
        self, context: Dict[str, Any], recovery: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply automatic recovery actions."""
        new_context = context.copy()
        technical = recovery.get("technical_details", {})
        action = technical.get("action")

        if action == "reduce_quality":
            new_context["duration"] = technical.get("new_duration", 10)
            new_context["resolution"] = technical.get("new_resolution", "720p")
            new_context["quality"] = "low"

        elif action == "use_cpu":
            new_context["use_gpu"] = False
            new_context["device"] = "cpu"

        elif action == "install_deps":
            # This would trigger dependency installation
            new_context["auto_install_deps"] = True

        return new_context


# Global robust generator instance
_global_generator = RobustVideoGenerator()


def get_robust_generator(
    retry_config: Optional[RetryConfig] = None,
) -> RobustVideoGenerator:
    """Get the global robust video generator."""
    return _global_generator
