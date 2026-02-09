"""
Routine ROI Measurement & Auto-Promotion

Tracks the value of detected routines and auto-promotes those with proven ROI.
Integrates with routine_detector.py and predictive_engine.py.
"""

import os
import json
import time
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

ROI_STORE_PATH = os.path.join(os.path.expanduser("~"), ".shadowai", "routine_roi.json")

# Auto-promote thresholds
MIN_TRIGGERS_FOR_PROMOTION = 5
MIN_CONFIDENCE_FOR_PROMOTION = 0.8
ESTIMATED_MANUAL_TIME_SECONDS = 120  # Assumed time saved per automated trigger


class RoutineROI:
    """Track routine trigger outcomes and calculate ROI for auto-promotion."""

    def __init__(self):
        self._data: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load ROI data from disk."""
        try:
            if os.path.exists(ROI_STORE_PATH):
                with open(ROI_STORE_PATH, "r") as f:
                    self._data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load routine ROI data: {e}")
            self._data = {}

    def _save(self):
        """Persist ROI data to disk."""
        try:
            os.makedirs(os.path.dirname(ROI_STORE_PATH), exist_ok=True)
            with open(ROI_STORE_PATH, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save routine ROI data: {e}")

    def record_trigger(self, routine_id: str, trigger_type: str, success: bool, duration_ms: int = 0):
        """Record a routine trigger outcome."""
        if routine_id not in self._data:
            self._data[routine_id] = {
                "trigger_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_duration_ms": 0,
                "triggers": [],
            }

        entry = self._data[routine_id]
        entry["trigger_count"] += 1
        if success:
            entry["success_count"] += 1
        else:
            entry["failure_count"] += 1
        entry["total_duration_ms"] += duration_ms

        # Keep last 50 triggers for analysis
        entry["triggers"].append({
            "timestamp": time.time(),
            "trigger_type": trigger_type,
            "success": success,
            "duration_ms": duration_ms,
        })
        entry["triggers"] = entry["triggers"][-50:]

        self._save()
        logger.debug(f"Recorded trigger for routine {routine_id}: success={success}")

    def calculate_roi(self, routine_id: str) -> Dict:
        """Calculate ROI metrics for a routine."""
        entry = self._data.get(routine_id)
        if not entry or entry["trigger_count"] == 0:
            return {
                "time_saved_seconds": 0,
                "error_reduction": 0.0,
                "trigger_count": 0,
                "success_rate": 0.0,
                "roi_positive": False,
            }

        trigger_count = entry["trigger_count"]
        success_count = entry["success_count"]
        success_rate = success_count / max(trigger_count, 1)

        # Time saved = successful triggers * estimated manual time
        time_saved = success_count * ESTIMATED_MANUAL_TIME_SECONDS

        # Error reduction = success rate improvement over baseline (50%)
        error_reduction = max(0, success_rate - 0.5) * 2  # normalized 0-1

        # ROI is positive if more successes than failures and time saved > 0
        roi_positive = success_rate > 0.5 and time_saved > 0

        return {
            "time_saved_seconds": time_saved,
            "error_reduction": round(error_reduction, 3),
            "trigger_count": trigger_count,
            "success_rate": round(success_rate, 3),
            "roi_positive": roi_positive,
        }

    def should_auto_promote(self, routine_id: str, confidence: float = 0.0) -> bool:
        """Check if a routine should be auto-promoted to active status."""
        roi = self.calculate_roi(routine_id)

        return (
            roi["roi_positive"]
            and confidence >= MIN_CONFIDENCE_FOR_PROMOTION
            and roi["trigger_count"] >= MIN_TRIGGERS_FOR_PROMOTION
        )


# Global singleton
_routine_roi: Optional[RoutineROI] = None


def get_routine_roi() -> RoutineROI:
    """Get or create the global RoutineROI instance."""
    global _routine_roi
    if _routine_roi is None:
        _routine_roi = RoutineROI()
    return _routine_roi
