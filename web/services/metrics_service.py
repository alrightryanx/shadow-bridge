"""
Video Generation Metrics Service
Tracks performance statistics and usage metrics for video generation.
"""

import json
import os
import time
import logging
import threading
from typing import Dict, Any, List
from datetime import datetime

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)

METRICS_FILE = os.path.join(os.path.expanduser("~"), ".shadowai", "video_metrics.json")

class MetricsService:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = self._load_metrics()
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
            except Exception as e:
                logger.warning(f"NVML init failed: {e}")

    def get_gpu_telemetry(self) -> Dict[str, Any]:
        """Get real-time GPU statistics."""
        if not NVML_AVAILABLE:
            return {"available": False}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                "available": True,
                "used_vram_gb": round(info.used / (1024**3), 2),
                "total_vram_gb": round(info.total / (1024**3), 2),
                "free_vram_gb": round(info.free / (1024**3), 2),
                "gpu_util_percent": util.gpu,
                "temp_c": temp
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def _load_metrics(self):
        """Load metrics from disk."""
        try:
            if os.path.exists(METRICS_FILE):
                with open(METRICS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

        return {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_duration_seconds": 0,
            "models": {}
        }

    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
            with open(METRICS_FILE, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def record_generation(self, model: str, duration_ms: float, success: bool, error: str = None):
        """Record a single generation event."""
        with self._lock:
            self._cache["total_generations"] += 1
            
            if success:
                self._cache["successful_generations"] += 1
                self._cache["total_duration_seconds"] += (duration_ms / 1000.0)
            else:
                self._cache["failed_generations"] += 1

            # Model specific stats
            if model not in self._cache["models"]:
                self._cache["models"][model] = {
                    "count": 0,
                    "success": 0,
                    "total_duration_ms": 0,
                    "avg_duration_ms": 0,
                    "last_used": None
                }
            
            m_stats = self._cache["models"][model]
            m_stats["count"] += 1
            m_stats["last_used"] = datetime.now().isoformat()
            
            if success:
                m_stats["success"] += 1
                m_stats["total_duration_ms"] += duration_ms
                # Update moving average
                if m_stats["success"] > 0:
                    m_stats["avg_duration_ms"] = m_stats["total_duration_ms"] / m_stats["success"]

            self._save_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            metrics = dict(self._cache)
            metrics["gpu_current"] = self.get_gpu_telemetry()
            return metrics

    def get_model_stats(self, model: str) -> Dict[str, Any]:
        """Get stats for a specific model."""
        with self._lock:
            return self._cache["models"].get(model, {})

# Global instance
_metrics_service = MetricsService()

def get_metrics_service():
    return _metrics_service
