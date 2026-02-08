"""Predictive intelligence data store with JSON file persistence.

Stores temporal patterns, detected routines, and prediction logs.
Thread-safe with singleton access via get_predictive_store().
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.path.join(str(Path.home()), ".shadowai")
PREDICTIVE_STORE_FILE = "predictive_store.json"


class PredictiveStore:
    """Thread-safe in-memory predictive data store backed by JSON file."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.file_path = os.path.join(self.data_dir, PREDICTIVE_STORE_FILE)
        self._lock = threading.Lock()
        self.temporal_patterns: Dict[str, dict] = {}
        self.detected_routines: Dict[str, dict] = {}
        self.prediction_log: Dict[str, dict] = {}
        self.action_history: List[dict] = []
        self._load()

    def _load(self):
        """Load state from JSON file."""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.temporal_patterns = data.get("temporal_patterns", {})
                self.detected_routines = data.get("detected_routines", {})
                self.prediction_log = data.get("prediction_log", {})
                self.action_history = data.get("action_history", [])
                log.info(
                    f"PredictiveStore loaded: {len(self.temporal_patterns)} patterns, "
                    f"{len(self.detected_routines)} routines, "
                    f"{len(self.prediction_log)} predictions"
                )
        except Exception as e:
            log.warning(f"Failed to load predictive store: {e}")

    def _save(self):
        """Persist state to JSON file."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            payload = {
                "temporal_patterns": self.temporal_patterns,
                "detected_routines": self.detected_routines,
                "prediction_log": self.prediction_log,
                "action_history": self.action_history[-10000:],  # Cap history
                "updated": time.time(),
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save predictive store: {e}")

    # ---- Action History ----

    def record_action(self, action_type: str, project_id: str = "",
                      metadata: Optional[Dict[str, Any]] = None) -> dict:
        """Record a user action for temporal pattern analysis."""
        now = time.time()
        lt = time.localtime(now)
        action = {
            "id": str(uuid.uuid4()),
            "action_type": action_type,
            "project_id": project_id,
            "metadata": metadata or {},
            "timestamp": now,
            "hour_of_day": lt.tm_hour,
            "day_of_week": lt.tm_wday,  # 0=Monday, 6=Sunday
        }
        with self._lock:
            self.action_history.append(action)
            self._update_temporal_pattern(action)
            self._save()
        return action

    def _update_temporal_pattern(self, action: dict):
        """Update temporal pattern counts for this action type + time slot."""
        key = f"{action['action_type']}:{action['hour_of_day']}:{action['day_of_week']}"
        if key in self.temporal_patterns:
            self.temporal_patterns[key]["count"] += 1
            self.temporal_patterns[key]["last_seen"] = action["timestamp"]
        else:
            self.temporal_patterns[key] = {
                "id": key,
                "action_type": action["action_type"],
                "project_id": action.get("project_id", ""),
                "hour_of_day": action["hour_of_day"],
                "day_of_week": action["day_of_week"],
                "count": 1,
                "first_seen": action["timestamp"],
                "last_seen": action["timestamp"],
            }

    def get_actions_since(self, since_timestamp: float) -> List[dict]:
        """Get all actions recorded after a timestamp."""
        with self._lock:
            return [a for a in self.action_history if a["timestamp"] > since_timestamp]

    # ---- Temporal Patterns ----

    def get_temporal_patterns(self, min_count: int = 1) -> List[dict]:
        """Get temporal patterns with at least min_count occurrences."""
        with self._lock:
            return [p for p in self.temporal_patterns.values()
                    if p["count"] >= min_count]

    def get_patterns_for_time(self, hour: int, day_of_week: int,
                              min_count: int = 3) -> List[dict]:
        """Get patterns matching a specific time slot."""
        with self._lock:
            return [
                p for p in self.temporal_patterns.values()
                if p["hour_of_day"] == hour
                and p["day_of_week"] == day_of_week
                and p["count"] >= min_count
            ]

    # ---- Detected Routines ----

    def add_routine(self, pattern_hash: str, actions: List[dict],
                    frequency: str, confidence: float) -> dict:
        """Add a detected routine."""
        routine_id = str(uuid.uuid4())
        routine = {
            "id": routine_id,
            "pattern_hash": pattern_hash,
            "actions": actions,
            "frequency": frequency,
            "confidence": confidence,
            "status": "detected",  # detected, approved, active, dismissed
            "created_at": time.time(),
            "last_triggered": None,
            "trigger_count": 0,
        }
        with self._lock:
            self.detected_routines[routine_id] = routine
            self._save()
        return routine

    def get_routines(self, status: Optional[str] = None) -> List[dict]:
        """Get all routines, optionally filtered by status."""
        with self._lock:
            routines = list(self.detected_routines.values())
        if status:
            routines = [r for r in routines if r.get("status") == status]
        return routines

    def update_routine(self, routine_id: str, updates: dict) -> Optional[dict]:
        """Update a routine's fields."""
        with self._lock:
            if routine_id not in self.detected_routines:
                return None
            self.detected_routines[routine_id].update(updates)
            self._save()
            return self.detected_routines[routine_id]

    # ---- Prediction Log ----

    def add_prediction(self, signal_type: str, predicted_action: str,
                       confidence: float, context: Optional[dict] = None) -> dict:
        """Log a prediction."""
        pred_id = str(uuid.uuid4())
        prediction = {
            "id": pred_id,
            "signal_type": signal_type,
            "predicted_action": predicted_action,
            "confidence": confidence,
            "context": context or {},
            "outcome": "pending",  # pending, accepted, rejected, expired
            "prediction_time": time.time(),
            "resolution_time": None,
            "feedback": None,
        }
        with self._lock:
            self.prediction_log[pred_id] = prediction
            self._save()
        return prediction

    def get_predictions(self, limit: int = 10,
                        min_confidence: float = 0.0,
                        outcome: Optional[str] = None) -> List[dict]:
        """Get predictions, optionally filtered by confidence and outcome."""
        with self._lock:
            preds = list(self.prediction_log.values())
        preds = [p for p in preds if p["confidence"] >= min_confidence]
        if outcome:
            preds = [p for p in preds if p["outcome"] == outcome]
        preds.sort(key=lambda p: p["prediction_time"], reverse=True)
        return preds[:limit]

    def resolve_prediction(self, prediction_id: str, outcome: str,
                           feedback: Optional[str] = None) -> Optional[dict]:
        """Resolve a prediction with outcome and optional feedback."""
        with self._lock:
            if prediction_id not in self.prediction_log:
                return None
            pred = self.prediction_log[prediction_id]
            pred["outcome"] = outcome
            pred["resolution_time"] = time.time()
            if feedback:
                pred["feedback"] = feedback
            self._save()
            return pred

    def get_prediction_stats(self) -> dict:
        """Get aggregate prediction statistics."""
        with self._lock:
            preds = list(self.prediction_log.values())
        total = len(preds)
        accepted = sum(1 for p in preds if p["outcome"] == "accepted")
        rejected = sum(1 for p in preds if p["outcome"] == "rejected")
        pending = sum(1 for p in preds if p["outcome"] == "pending")
        accuracy = accepted / (accepted + rejected) if (accepted + rejected) > 0 else 0.0
        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "pending": pending,
            "accuracy": round(accuracy, 3),
        }


# ---- Singleton ----

_store: Optional[PredictiveStore] = None
_store_lock = threading.Lock()


def get_predictive_store() -> PredictiveStore:
    """Return the singleton PredictiveStore instance."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = PredictiveStore()
    return _store
