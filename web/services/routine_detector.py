"""Routine detector service for ShadowBridge.

Detects repeating manual action patterns and offers to automate them.
Integrates with PredictiveEngine for pattern data and PredictiveStore
for persistence. When routines fire, creates tasks in TaskStore for
the AgentDaemon to execute.
"""

import logging
import time
import uuid
from typing import List, Optional

from web.services.predictive_engine import get_predictive_engine
from web.services.predictive_store import get_predictive_store

log = logging.getLogger(__name__)

# Map routine action types to task descriptions for the daemon
_ACTION_TASK_TEMPLATES = {
    "generate_briefing": {
        "title": "Generate project briefing",
        "description": "Analyze recent activity and produce a briefing summary.",
    },
    "run_tests": {
        "title": "Run project test suite",
        "description": "Execute the project's test suite and report results.",
    },
    "update_dependencies": {
        "title": "Check and update dependencies",
        "description": "Review project dependencies for security updates and staleness.",
    },
    "code_review": {
        "title": "Review recent code changes",
        "description": "Audit recent commits for quality, security, and style issues.",
    },
    "git_cleanup": {
        "title": "Git repository cleanup",
        "description": "Commit uncommitted files, clean stale branches, verify repo health.",
    },
    "generate_todos": {
        "title": "Generate TODO tasks from project analysis",
        "description": "Analyze project state and generate actionable improvement tasks.",
    },
    "health_check": {
        "title": "Run project health check",
        "description": "Analyze project health: build, deps, docs, git status.",
    },
}


class RoutineDetector:
    """Detects and manages automated routines from user behavior patterns."""

    def __init__(self):
        self._engine = get_predictive_engine()
        self._store = get_predictive_store()
        self._task_store = None  # Lazy-loaded to avoid circular imports

    def _get_task_store(self):
        """Lazy-load TaskStore to avoid circular imports."""
        if self._task_store is None:
            from web.services.task_store import get_task_store
            self._task_store = get_task_store()
        return self._task_store

    def _create_tasks_for_routine(self, routine: dict, trigger_type: str = "time") -> List[dict]:
        """Convert a triggered routine's actions into TaskStore tasks.

        Each action in the routine becomes a real task that the AgentDaemon
        will pick up and execute via CLI tools.
        """
        store = self._get_task_store()
        actions = routine.get("actions", [])
        created_tasks = []

        for action in actions:
            action_type = action.get("type", "unknown")
            template = _ACTION_TASK_TEMPLATES.get(action_type, {})

            project_id = (action.get("project_id")
                          or routine.get("project_id", ""))
            project_dir = action.get("project_dir", "")

            task_data = {
                "title": template.get("title", f"Routine action: {action_type}"),
                "description": (
                    template.get("description", f"Execute routine action: {action_type}")
                    + f"\n\nProject: {project_id}" if project_id else ""
                ),
                "priority": "NORMAL",
                "tags": ["routine", f"trigger:{trigger_type}",
                         f"routine_id:{routine.get('id', '?')}"],
                "input": {
                    "action_type": action_type,
                    "project_id": project_id,
                    "routine_id": routine.get("id"),
                    "routine_name": routine.get("name", ""),
                    "source": "routine_detector",
                    **{k: v for k, v in action.items()
                       if k not in ("type", "project_id", "project_dir")},
                },
                "created_by": "routine_detector",
            }
            if project_dir:
                task_data["input"]["working_directory"] = project_dir

            created = store.create_task(task_data)
            created_tasks.append(created)
            log.info(f"Created task {created['id']} for routine "
                     f"{routine.get('id', '?')} action: {action_type}")

        return created_tasks

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

    def check_event_triggers(self, event_type: str, event_data: dict = None) -> List[dict]:
        """Check if any routines should be triggered by an event.

        Supports event-based triggers in addition to time-based frequency.
        Event types: 'project_activated', 'build_completed', 'commit_pushed',
                     'agent_idle', 'review_submitted', 'error_detected'

        Returns list of routines that were triggered by this event.
        """
        if event_data is None:
            event_data = {}
        now = time.time()
        active_routines = self._store.get_routines(status="active")
        triggered = []

        for routine in active_routines:
            trigger_events = routine.get("trigger_events", [])
            if not trigger_events or event_type not in trigger_events:
                continue

            # Cooldown: don't re-trigger within 5 minutes of last trigger
            last_triggered = routine.get("last_triggered") or 0
            if (now - last_triggered) < 300:
                continue

            actions = routine.get("actions", [])
            if not actions:
                continue

            # Optional: check event filter conditions
            conditions = routine.get("trigger_conditions", {})
            if conditions:
                match = all(
                    event_data.get(k) == v
                    for k, v in conditions.items()
                )
                if not match:
                    continue

            log.info(f"Event '{event_type}' triggered routine {routine['id']}: "
                     f"{' -> '.join(a.get('type', '?') for a in actions)}")

            self._store.update_routine(routine["id"], {
                "last_triggered": now,
                "trigger_count": routine.get("trigger_count", 0) + 1,
            })

            action_desc = " -> ".join(a.get("type", "?") for a in actions)
            self._store.add_prediction(
                signal_type="event_trigger",
                predicted_action=action_desc,
                confidence=routine.get("confidence", 0.8),
                context={
                    "routine_id": routine["id"],
                    "event_type": event_type,
                    "trigger_count": routine.get("trigger_count", 0) + 1,
                },
            )

            # Create real tasks for the daemon to execute
            self._create_tasks_for_routine(routine, trigger_type=f"event:{event_type}")
            triggered.append(routine)

        if triggered:
            log.info(f"Event '{event_type}' triggered {len(triggered)} routines, "
                     f"created tasks for execution")
        return triggered

    def execute_active_routines(self) -> List[dict]:
        """Check all active routines and fire those that are due.

        Called periodically from the AgentDaemon poll loop.
        Returns list of routines that were triggered.
        """
        now = time.time()
        active_routines = self._store.get_routines(status="active")
        triggered = []

        for routine in active_routines:
            # Skip event-only routines (no time-based frequency)
            if routine.get("trigger_events") and not routine.get("frequency"):
                continue

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

            # Create real tasks for the daemon to execute
            self._create_tasks_for_routine(routine, trigger_type=f"schedule:{frequency}")
            triggered.append(routine)

        if triggered:
            log.info(f"Triggered {len(triggered)} active routines, "
                     f"created tasks for execution")
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
