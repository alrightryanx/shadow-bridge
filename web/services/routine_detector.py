"""Routine detector service for ShadowBridge.

Detects repeating manual action patterns and offers to automate them.
Integrates with PredictiveEngine for pattern data and PredictiveStore
for persistence.
"""

import logging
import time
from typing import List, Optional

from web.services.predictive_engine import get_predictive_engine
from web.services.predictive_store import get_predictive_store

log = logging.getLogger(__name__)


class RoutineDetector:
    """Detects and manages automated routines from user behavior patterns."""

    def __init__(self):
        self._engine = get_predictive_engine()
        self._store = get_predictive_store()

    def scan_for_routines(self, lookback_days: int = 14) -> List[dict]:
        """Scan action history for repeating patterns.

        Delegates to PredictiveEngine.detect_routines() which does the
        heavy lifting of sequence analysis.
        """
        new_routines = self._engine.detect_routines(lookback_days)
        log.info(f"Routine scan found {len(new_routines)} new routines")
        return new_routines

    def approve_routine(self, routine_id: str) -> Optional[dict]:
        """Convert a detected routine to active automation.

        Changes status from 'detected' to 'active', making it eligible
        for automatic execution.
        """
        routine = self._store.update_routine(routine_id, {
            "status": "active",
            "approved_at": time.time(),
        })
        if routine:
            log.info(f"Routine {routine_id} approved and activated")
        return routine

    def dismiss_routine(self, routine_id: str) -> Optional[dict]:
        """Dismiss a detected routine."""
        return self._store.update_routine(routine_id, {
            "status": "dismissed",
            "dismissed_at": time.time(),
        })

    def execute_active_routines(self) -> List[dict]:
        """Check all active routines and fire those that are due.

        Called periodically from the AgentDaemon poll loop.
        Returns list of routines that were triggered.
        """
        now = time.time()
        active_routines = self._store.get_routines(status="active")
        triggered = []

        for routine in active_routines:
            last_triggered = routine.get("last_triggered") or 0
            frequency = routine.get("frequency", "daily")

            # Determine if routine is due based on frequency
            min_gap = {
                "hourly": 3600,
                "daily": 82800,      # 23 hours (allows for drift)
                "weekly": 590400,    # 6.8 days
                "monthly": 2505600,  # 29 days
            }.get(frequency, 86400)

            if (now - last_triggered) < min_gap:
                continue

            # Check if current time matches the routine's typical trigger time
            # (within a 2-hour window)
            actions = routine.get("actions", [])
            if not actions:
                continue

            # Fire the routine
            log.info(f"Triggering routine {routine['id']}: "
                     f"{' -> '.join(a.get('type', '?') for a in actions)}")

            self._store.update_routine(routine["id"], {
                "last_triggered": now,
                "trigger_count": routine.get("trigger_count", 0) + 1,
            })

            # Create a prediction entry for tracking
            action_desc = " -> ".join(a.get("type", "?") for a in actions)
            self._store.add_prediction(
                signal_type="routine_trigger",
                predicted_action=action_desc,
                confidence=routine.get("confidence", 0.7),
                context={
                    "routine_id": routine["id"],
                    "frequency": frequency,
                    "trigger_count": routine.get("trigger_count", 0) + 1,
                },
            )

            triggered.append(routine)

        if triggered:
            log.info(f"Triggered {len(triggered)} active routines")
        return triggered

    def get_routine_summary(self) -> dict:
        """Get a summary of routine detection status."""
        all_routines = self._store.get_routines()
        return {
            "total": len(all_routines),
            "detected": sum(1 for r in all_routines if r.get("status") == "detected"),
            "active": sum(1 for r in all_routines if r.get("status") == "active"),
            "dismissed": sum(1 for r in all_routines if r.get("status") == "dismissed"),
        }


# ---- Singleton ----

_detector: Optional[RoutineDetector] = None


def get_routine_detector() -> RoutineDetector:
    """Return the singleton RoutineDetector instance."""
    global _detector
    if _detector is None:
        _detector = RoutineDetector()
    return _detector
