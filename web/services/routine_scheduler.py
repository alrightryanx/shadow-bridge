"""
Routine Task Scheduler

Manages scheduled routines that generate agent goals on a cron-like schedule.
Routines are stored in the database and can be managed via the web dashboard.

Built-in routines are seeded on first run:
- Dependency Audit (daily 9am)
- Test Coverage Check (weekly Monday)
- Security Scan (weekly Friday)
- Code Quality Report (bi-weekly)
- Briefing Generation (daily 8am)
- Stale Task Cleanup (daily 11pm)

Usage:
    scheduler = RoutineScheduler(db_path)
    due = scheduler.check_due_routines()
    for routine in due:
        scheduler.execute_routine(routine)
"""

import os
import json
import uuid
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = os.path.join("C:", os.sep, "shadow", "backend", "data", "shadow_ai.db")

# Built-in routine definitions seeded on first run
BUILTIN_ROUTINES = [
    {
        "name": "Dependency Audit",
        "description": "Scan project dependencies for vulnerabilities and outdated packages",
        "cron_expression": "0 9 * * *",  # Daily 9am
        "task_template": json.dumps({
            "type": "audit",
            "prompt": "Scan all project dependencies for known vulnerabilities and outdated packages. Report findings and suggest updates. Check package.json, build.gradle.kts, and requirements.txt.",
            "category": "security",
            "priority": 6
        }),
    },
    {
        "name": "Test Coverage Check",
        "description": "Run tests and report coverage delta since last check",
        "cron_expression": "0 9 * * 1",  # Weekly Monday 9am
        "task_template": json.dumps({
            "type": "testing",
            "prompt": "Run the full test suite across all repos. Report test pass/fail counts and coverage percentages. Compare with previous run and highlight coverage regressions.",
            "category": "testing",
            "priority": 5
        }),
    },
    {
        "name": "Security Scan",
        "description": "Run security audit on active projects",
        "cron_expression": "0 9 * * 5",  # Weekly Friday 9am
        "task_template": json.dumps({
            "type": "security",
            "prompt": "Perform a security audit: check for hardcoded credentials, insecure API endpoints, SQL injection risks, XSS vulnerabilities, and OWASP top 10 issues in the codebase.",
            "category": "security",
            "priority": 8
        }),
    },
    {
        "name": "Code Quality Report",
        "description": "Lint and complexity analysis across the codebase",
        "cron_expression": "0 10 1,15 * *",  # Bi-weekly (1st and 15th)
        "task_template": json.dumps({
            "type": "quality",
            "prompt": "Run lint checks and code complexity analysis. Identify files with highest cyclomatic complexity, duplicate code blocks, and style violations. Generate a summary report.",
            "category": "code_review",
            "priority": 4
        }),
    },
    {
        "name": "Briefing Generation",
        "description": "Compile overnight activity into a briefing document",
        "cron_expression": "0 8 * * *",  # Daily 8am
        "task_template": json.dumps({
            "type": "briefing",
            "prompt": "Compile a daily briefing: summarize agent activity from the last 24 hours, list completed goals, failed tasks, new predictions, and any alerts or anomalies detected.",
            "category": "reporting",
            "priority": 3
        }),
    },
    {
        "name": "Stale Task Cleanup",
        "description": "Archive tasks that have been idle for more than 7 days",
        "cron_expression": "0 23 * * *",  # Daily 11pm
        "task_template": json.dumps({
            "type": "cleanup",
            "prompt": "Review all active goals and intentions. Archive goals that have had no progress in 7+ days. Mark stale intentions as failed. Generate a cleanup report.",
            "category": "maintenance",
            "priority": 2
        }),
    },
]


class RoutineScheduler:
    """Manages scheduled routines and converts them into agent goals."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._seeded = False

    def _get_db(self) -> Optional[sqlite3.Connection]:
        """Get a database connection."""
        try:
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                return conn
            return None
        except Exception as e:
            logger.warning(f"Failed to connect to DB at {self.db_path}: {e}")
            return None

    def seed_defaults(self):
        """Seed built-in routines if the table is empty."""
        if self._seeded:
            return

        conn = self._get_db()
        if conn is None:
            return

        try:
            cursor = conn.cursor()
            count = cursor.execute("SELECT COUNT(*) FROM routines").fetchone()[0]

            if count == 0:
                logger.info("Seeding default routines...")
                for routine in BUILTIN_ROUTINES:
                    next_run = self._calculate_next_run(routine["cron_expression"])
                    cursor.execute(
                        """INSERT INTO routines (name, description, cron_expression, task_template, enabled, next_run_at)
                           VALUES (?, ?, ?, ?, 1, ?)""",
                        [
                            routine["name"],
                            routine["description"],
                            routine["cron_expression"],
                            routine["task_template"],
                            next_run.isoformat() if next_run else None,
                        ],
                    )
                conn.commit()
                logger.info(f"Seeded {len(BUILTIN_ROUTINES)} default routines")

            self._seeded = True
        except Exception as e:
            logger.error(f"Failed to seed defaults: {e}")
        finally:
            conn.close()

    def check_due_routines(self) -> List[Dict[str, Any]]:
        """
        Check for routines whose next_run_at <= now.
        Called by autonomous_loop every cycle.
        """
        self.seed_defaults()

        conn = self._get_db()
        if conn is None:
            return []

        try:
            now = datetime.utcnow().isoformat()
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT * FROM routines WHERE enabled = 1 AND (next_run_at IS NULL OR next_run_at <= ?)",
                [now],
            ).fetchall()

            routines = [dict(row) for row in rows]
            conn.close()
            return routines
        except Exception as e:
            logger.error(f"Failed to check due routines: {e}")
            conn.close()
            return []

    def execute_routine(self, routine: Dict[str, Any]) -> Optional[str]:
        """
        Convert a routine into a goal and mark it as run.
        Returns the created goal_id, or None on failure.
        """
        conn = self._get_db()
        if conn is None:
            return None

        try:
            template = json.loads(routine["task_template"])
            cursor = conn.cursor()

            # Find a user to assign the goal to
            user_row = cursor.execute("SELECT id FROM subscribers LIMIT 1").fetchone()
            if not user_row:
                logger.warning("No users found to assign routine goal")
                conn.close()
                return None

            user_id = user_row[0]
            goal_id = str(uuid.uuid4())
            description = template.get("prompt", routine.get("description", routine["name"]))
            priority = template.get("priority", 5)

            metadata = json.dumps({
                "source": "routine",
                "routine_id": routine["id"],
                "routine_name": routine["name"],
                "category": template.get("category", "general"),
            })

            cursor.execute(
                "INSERT INTO goals (id, user_id, description, priority, status, metadata) VALUES (?, ?, ?, ?, 'active', ?)",
                [goal_id, user_id, description, priority, metadata],
            )

            # Update routine timing
            next_run = self._calculate_next_run(routine["cron_expression"])
            cursor.execute(
                "UPDATE routines SET last_run_at = datetime('now'), next_run_at = ? WHERE id = ?",
                [next_run.isoformat() if next_run else None, routine["id"]],
            )

            conn.commit()
            conn.close()

            logger.info(f"Routine '{routine['name']}' executed -> goal {goal_id}")
            return goal_id

        except Exception as e:
            logger.error(f"Failed to execute routine '{routine.get('name')}': {e}")
            conn.close()
            return None

    def get_all_routines(self) -> List[Dict[str, Any]]:
        """Get all routines for dashboard display."""
        self.seed_defaults()

        conn = self._get_db()
        if conn is None:
            return []

        try:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT * FROM routines ORDER BY id").fetchall()
            routines = [dict(row) for row in rows]
            conn.close()
            return routines
        except Exception as e:
            logger.error(f"Failed to get routines: {e}")
            conn.close()
            return []

    def toggle_routine(self, routine_id: int, enabled: bool) -> bool:
        """Enable or disable a routine."""
        conn = self._get_db()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE routines SET enabled = ? WHERE id = ?",
                [1 if enabled else 0, routine_id],
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to toggle routine {routine_id}: {e}")
            conn.close()
            return False

    def create_routine(
        self,
        name: str,
        description: str,
        cron_expression: str,
        task_template: Dict[str, Any],
        project_id: Optional[int] = None,
    ) -> Optional[int]:
        """Create a new custom routine."""
        conn = self._get_db()
        if conn is None:
            return None

        try:
            cursor = conn.cursor()
            next_run = self._calculate_next_run(cron_expression)
            cursor.execute(
                """INSERT INTO routines (name, description, cron_expression, task_template, enabled, next_run_at, project_id)
                   VALUES (?, ?, ?, ?, 1, ?, ?)""",
                [
                    name,
                    description,
                    cron_expression,
                    json.dumps(task_template),
                    next_run.isoformat() if next_run else None,
                    project_id,
                ],
            )
            routine_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return routine_id
        except Exception as e:
            logger.error(f"Failed to create routine: {e}")
            conn.close()
            return None

    def delete_routine(self, routine_id: int) -> bool:
        """Delete a routine by ID."""
        conn = self._get_db()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM routines WHERE id = ?", [routine_id])
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to delete routine {routine_id}: {e}")
            conn.close()
            return False

    def _calculate_next_run(self, cron_expression: str) -> Optional[datetime]:
        """
        Simple cron-like next-run calculator.
        Supports common patterns: hourly, daily at hour, weekly on day at hour, bi-weekly.
        """
        now = datetime.utcnow()
        parts = cron_expression.strip().split()

        if len(parts) != 5:
            logger.warning(f"Invalid cron expression: {cron_expression}")
            return now + timedelta(hours=24)

        minute, hour, day_of_month, month, day_of_week = parts

        try:
            # Hourly: "0 * * * *"
            if hour == "*" and day_of_month == "*" and day_of_week == "*":
                m = int(minute)
                next_run = now.replace(minute=m, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(hours=1)
                return next_run

            # Daily: "0 9 * * *"
            if day_of_month == "*" and day_of_week == "*" and hour != "*":
                h = int(hour)
                m = int(minute)
                next_run = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                return next_run

            # Weekly: "0 9 * * 1" (Monday=1)
            if day_of_month == "*" and day_of_week != "*" and hour != "*":
                h = int(hour)
                m = int(minute)
                target_dow = int(day_of_week)
                next_run = now.replace(hour=h, minute=m, second=0, microsecond=0)
                days_ahead = (target_dow - now.weekday()) % 7
                if days_ahead == 0 and next_run <= now:
                    days_ahead = 7
                next_run += timedelta(days=days_ahead)
                return next_run

            # Bi-weekly (1st and 15th): "0 10 1,15 * *"
            if "," in day_of_month and day_of_week == "*":
                h = int(hour)
                m = int(minute)
                days = sorted([int(d) for d in day_of_month.split(",")])
                next_run = now.replace(hour=h, minute=m, second=0, microsecond=0)

                for d in days:
                    candidate = next_run.replace(day=d)
                    if candidate > now:
                        return candidate

                # Next month, first day in list
                if next_run.month == 12:
                    next_run = next_run.replace(year=next_run.year + 1, month=1, day=days[0])
                else:
                    next_run = next_run.replace(month=next_run.month + 1, day=days[0])
                return next_run

        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing cron '{cron_expression}': {e}")

        # Fallback: 24 hours from now
        return now + timedelta(hours=24)


# Module-level singleton
_scheduler_instance: Optional[RoutineScheduler] = None


def get_routine_scheduler(db_path: Optional[str] = None) -> RoutineScheduler:
    """Get or create the singleton RoutineScheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = RoutineScheduler(db_path)
    return _scheduler_instance
