"""Predictive intelligence engine for ShadowBridge.

Analyzes temporal patterns, detects routines, and generates predictions
for proactive task creation. Called by the predictive API routes.
"""

import hashlib
import logging
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

from web.services.predictive_store import get_predictive_store

log = logging.getLogger(__name__)

# Minimum occurrences to flag a temporal pattern
MIN_PATTERN_OCCURRENCES = 7
# Minimum sequence occurrences to flag a routine
MIN_ROUTINE_OCCURRENCES = 3
# Default lookback for routine detection (14 days)
DEFAULT_LOOKBACK_DAYS = 14
# Prediction confidence thresholds
HIGH_CONFIDENCE = 0.9
MEDIUM_CONFIDENCE = 0.7


class PredictiveEngine:
    """Analyzes user behavior patterns and generates predictions."""

    def __init__(self):
        self._store = get_predictive_store()

    def record_user_action(self, action_type: str, project_id: str = "",
                           metadata: Optional[Dict[str, Any]] = None) -> dict:
        """Record a user action for learning. Called when Android reports events."""
        action = self._store.record_action(action_type, project_id, metadata)
        log.info(f"Recorded action: {action_type} for project {project_id}")
        return action

    def analyze_temporal_patterns(self) -> List[dict]:
        """Analyze recorded actions for time-based patterns.

        Groups actions by hour-of-day/day-of-week and flags patterns
        with MIN_PATTERN_OCCURRENCES+ occurrences.
        """
        patterns = self._store.get_temporal_patterns(min_count=MIN_PATTERN_OCCURRENCES)
        log.info(f"Found {len(patterns)} temporal patterns with {MIN_PATTERN_OCCURRENCES}+ occurrences")
        return patterns

    def detect_routines(self, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> List[dict]:
        """Scan action history for repeating sequences (3+ occurrences).

        Looks for sequences of 2-5 actions that repeat within the lookback window.
        Returns newly detected routines.
        """
        since = time.time() - (lookback_days * 86400)
        actions = self._store.get_actions_since(since)

        if len(actions) < MIN_ROUTINE_OCCURRENCES * 2:
            return []

        # Group actions by project
        by_project: Dict[str, List[dict]] = defaultdict(list)
        for action in actions:
            pid = action.get("project_id", "global")
            by_project[pid].append(action)

        new_routines = []
        existing_hashes = {r["pattern_hash"] for r in self._store.get_routines()}

        for project_id, proj_actions in by_project.items():
            # Sort by timestamp
            proj_actions.sort(key=lambda a: a["timestamp"])
            action_types = [a["action_type"] for a in proj_actions]

            # Scan for repeating sequences of length 2-5
            for seq_len in range(2, min(6, len(action_types))):
                sequence_counts: Counter = Counter()
                for i in range(len(action_types) - seq_len + 1):
                    seq = tuple(action_types[i:i + seq_len])
                    sequence_counts[seq] += 1

                for seq, count in sequence_counts.items():
                    if count >= MIN_ROUTINE_OCCURRENCES:
                        pattern_hash = hashlib.md5(
                            f"{project_id}:{':'.join(seq)}".encode()
                        ).hexdigest()

                        if pattern_hash not in existing_hashes:
                            # Estimate frequency
                            frequency = self._estimate_frequency(proj_actions, seq)
                            confidence = min(0.5 + (count * 0.05), 0.95)

                            routine = self._store.add_routine(
                                pattern_hash=pattern_hash,
                                actions=[{"type": t, "project_id": project_id} for t in seq],
                                frequency=frequency,
                                confidence=confidence,
                            )
                            new_routines.append(routine)
                            existing_hashes.add(pattern_hash)

        log.info(f"Detected {len(new_routines)} new routines from {len(actions)} actions")
        return new_routines

    def _estimate_frequency(self, actions: List[dict], sequence: tuple) -> str:
        """Estimate how often a sequence occurs (daily, weekly, etc.)."""
        action_types = [a["action_type"] for a in actions]
        timestamps = []

        for i in range(len(action_types) - len(sequence) + 1):
            if tuple(action_types[i:i + len(sequence)]) == sequence:
                timestamps.append(actions[i]["timestamp"])

        if len(timestamps) < 2:
            return "unknown"

        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        avg_gap = sum(gaps) / len(gaps)

        if avg_gap < 7200:  # < 2 hours
            return "hourly"
        elif avg_gap < 86400:  # < 1 day
            return "daily"
        elif avg_gap < 604800:  # < 1 week
            return "weekly"
        else:
            return "monthly"

    def predict_next_actions(self, context: Optional[dict] = None) -> List[dict]:
        """Given current context, return high-confidence predictions.

        Uses temporal patterns and routine history to predict what
        actions the user is likely to perform next.
        """
        now = time.time()
        lt = time.localtime(now)
        current_hour = lt.tm_hour
        current_dow = lt.tm_wday

        predictions = []

        # 1. Temporal pattern predictions
        time_patterns = self._store.get_patterns_for_time(
            current_hour, current_dow, min_count=MIN_PATTERN_OCCURRENCES
        )
        for pattern in time_patterns:
            base_confidence = min(0.5 + (pattern["count"] * 0.03), 0.95)
            # Adjust confidence based on historical prediction accuracy
            accuracy = pattern.get("prediction_accuracy")
            if accuracy is not None and (pattern.get("prediction_hits", 0) +
                                          pattern.get("prediction_misses", 0)) >= 3:
                confidence = base_confidence * (0.5 + 0.5 * accuracy)
            else:
                confidence = base_confidence
            pred = self._store.add_prediction(
                signal_type="temporal_pattern",
                predicted_action=pattern["action_type"],
                confidence=confidence,
                context={
                    "hour": current_hour,
                    "day_of_week": current_dow,
                    "pattern_count": pattern["count"],
                    "project_id": pattern.get("project_id", ""),
                },
            )
            predictions.append(pred)

        # 2. Active routine predictions
        active_routines = self._store.get_routines(status="active")
        for routine in active_routines:
            last_triggered = routine.get("last_triggered") or 0
            frequency = routine.get("frequency", "daily")

            # Check if routine is due
            is_due = False
            if frequency == "hourly" and (now - last_triggered) > 3600:
                is_due = True
            elif frequency == "daily" and (now - last_triggered) > 82800:  # 23 hours
                is_due = True
            elif frequency == "weekly" and (now - last_triggered) > 590400:  # 6.8 days
                is_due = True

            if is_due:
                actions = routine.get("actions", [])
                action_desc = " -> ".join(a.get("type", "?") for a in actions)
                pred = self._store.add_prediction(
                    signal_type="routine",
                    predicted_action=action_desc,
                    confidence=routine.get("confidence", 0.7),
                    context={
                        "routine_id": routine["id"],
                        "frequency": frequency,
                    },
                )
                predictions.append(pred)

        # 3. Recent momentum predictions (if user was active recently)
        recent = self._store.get_actions_since(now - 3600)
        if recent:
            recent_types = Counter(a["action_type"] for a in recent)
            most_common = recent_types.most_common(1)
            if most_common and most_common[0][1] >= 3:
                action_type = most_common[0][0]
                pred = self._store.add_prediction(
                    signal_type="momentum",
                    predicted_action=action_type,
                    confidence=0.6,
                    context={"recent_count": most_common[0][1]},
                )
                predictions.append(pred)

        log.info(f"Generated {len(predictions)} predictions")
        return predictions

    def get_predictions(self, limit: int = 10,
                        min_confidence: float = 0.0) -> List[dict]:
        """Get recent predictions. Used by /api/predictive/predictions."""
        return self._store.get_predictions(
            limit=limit, min_confidence=min_confidence
        )

    def get_status(self) -> dict:
        """Get engine status. Matches BriefingEngine's expected response shape:
        {loops: {total, completed, running}, patterns: {...}}
        """
        pred_stats = self._store.get_prediction_stats()
        patterns = self._store.get_temporal_patterns(min_count=1)
        routines = self._store.get_routines()

        active_routines = [r for r in routines if r.get("status") == "active"]
        detected_routines = [r for r in routines if r.get("status") == "detected"]

        return {
            "loops": {
                "total": pred_stats["total"],
                "completed": pred_stats["accepted"] + pred_stats["rejected"],
                "running": pred_stats["pending"],
            },
            "patterns": {
                "temporal_count": len(patterns),
                "strong_patterns": len([p for p in patterns
                                        if p["count"] >= MIN_PATTERN_OCCURRENCES]),
                "routines_detected": len(detected_routines),
                "routines_active": len(active_routines),
            },
            "accuracy": pred_stats["accuracy"],
            "predictions": pred_stats,
        }

    def resolve_prediction(self, prediction_id: str, outcome: str,
                           feedback: Optional[str] = None) -> Optional[dict]:
        """Resolve a prediction with outcome feedback.

        Also adjusts confidence for the originating signal using Bayesian updating:
        - Accepted outcomes increase confidence of similar future predictions
        - Rejected outcomes decrease confidence
        """
        result = self._store.resolve_prediction(prediction_id, outcome, feedback)
        if result is None:
            return None

        # Bayesian confidence adjustment based on outcome
        signal_type = result.get("signal_type", "")
        action = result.get("predicted_action", "")
        old_confidence = result.get("confidence", 0.5)
        learning_rate = 0.05

        if outcome in ("accepted", "resolved"):
            # Increase confidence for this signal pattern
            adjustment = learning_rate * (1.0 - old_confidence)
        elif outcome in ("rejected", "failed"):
            # Decrease confidence for this signal pattern
            adjustment = -learning_rate * old_confidence
        else:
            return result

        # Apply adjustment to the originating routine's confidence
        if signal_type == "routine":
            routine_id = result.get("context", {}).get("routine_id")
            if routine_id:
                routine = self._store.detected_routines.get(routine_id)
                if routine:
                    new_conf = max(0.1, min(0.99,
                                            routine["confidence"] + adjustment))
                    self._store.update_routine(routine_id,
                                               {"confidence": new_conf})
                    log.info(f"Routine {routine_id} confidence: "
                             f"{routine['confidence']:.2f} -> {new_conf:.2f}")

        # Update accuracy stats in the temporal pattern if applicable
        if signal_type == "temporal_pattern":
            ctx = result.get("context", {})
            hour = ctx.get("hour")
            dow = ctx.get("day_of_week")
            if hour is not None and dow is not None:
                pattern_key = f"{action}:{hour}:{dow}"
                pattern = self._store.temporal_patterns.get(pattern_key)
                if pattern:
                    hits = pattern.get("prediction_hits", 0)
                    misses = pattern.get("prediction_misses", 0)
                    if outcome in ("accepted", "resolved"):
                        pattern["prediction_hits"] = hits + 1
                    else:
                        pattern["prediction_misses"] = misses + 1
                    total = pattern["prediction_hits"] + pattern["prediction_misses"]
                    if total > 0:
                        pattern["prediction_accuracy"] = round(
                            pattern["prediction_hits"] / total, 3)

        return result


# ---- Singleton ----

_engine: Optional[PredictiveEngine] = None


def get_predictive_engine() -> PredictiveEngine:
    """Return the singleton PredictiveEngine instance."""
    global _engine
    if _engine is None:
        _engine = PredictiveEngine()
    return _engine
