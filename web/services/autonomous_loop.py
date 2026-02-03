"""
Autonomous Loop Controller

Main orchestrator tying scanner -> agent assignment -> build -> deploy.
Runs a continuous loop where agents autonomously find and fix code issues.

Usage:
    loop = AutonomousLoop()
    loop.start(agent_count=5, focus="backend-polish")
    # ... later ...
    loop.stop()
"""

import os
import time
import uuid
import json
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

from web.services.autonomous_scanner import AutonomousScanner
from web.services.data_service import get_tasks, sync_agents_from_device
from web.services.build_automation import BuildAutomation, GateResult
from web.services.deployment_automation import DeploymentAutomation
from web.services.workspace_manager import WorkspaceManager
from web.services.agent_orchestrator import (
    spawn_agent,
    assign_task,
    stop_agent,
    get_all_agents,
    active_agents,
    agent_task_events,
    agent_task_states,
    broadcast_agent_event,
)

logger = logging.getLogger(__name__)

# Persistence
STATUS_FILE = os.path.join(os.path.expanduser("~"), ".shadowai", "autonomous_status.json")

# Default agent roles per focus area
AGENT_ROLES = {
    "backend-polish": [
        {"name": "Backend Agent", "specialty": "refactoring", "repo": "shadow-bridge"},
        {"name": "Android Core Agent", "specialty": "debugging", "repo": "shadow-android"},
        {"name": "QA Agent", "specialty": "testing", "repo": "shadow-android"},
        {"name": "Performance Agent", "specialty": "performance", "repo": "shadow-android"},
        {"name": "Cleanup Agent", "specialty": "code_review", "repo": "shadow-bridge"},
    ],
    "android-focus": [
        {"name": "Android UI Agent", "specialty": "refactoring", "repo": "shadow-android"},
        {"name": "Android Backend Agent", "specialty": "debugging", "repo": "shadow-android"},
        {"name": "Android Tests Agent", "specialty": "testing", "repo": "shadow-android"},
        {"name": "Android Perf Agent", "specialty": "performance", "repo": "shadow-android"},
        {"name": "Android Review Agent", "specialty": "code_review", "repo": "shadow-android"},
    ],
    "bridge-focus": [
        {"name": "Bridge Core Agent", "specialty": "refactoring", "repo": "shadow-bridge"},
        {"name": "Bridge API Agent", "specialty": "debugging", "repo": "shadow-bridge"},
        {"name": "Bridge Tests Agent", "specialty": "testing", "repo": "shadow-bridge"},
        {"name": "Bridge Perf Agent", "specialty": "performance", "repo": "shadow-bridge"},
        {"name": "Bridge Review Agent", "specialty": "code_review", "repo": "shadow-bridge"},
    ],
}

# Config resolution
def get_config_value(key: str, default: str = None) -> Optional[str]:
    """Read a value from backend/.env or environment variables."""
    # 1. Try environment variable
    val = os.environ.get(key)
    if val: return val
    
    # 2. Try backend/.env
    env_path = "C:/shadow/backend/.env"
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith(f"{key}="):
                        return line.split("=", 1)[1].strip()
        except: pass
    return default

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROVIDER = "gemini"

AUTO_DEPLOY_ENABLED = (get_config_value("AUTO_DEPLOY_ENABLED", "false") or "").lower() in ("1", "true", "yes", "on")
AUTO_DEPLOY_ACTIVITY = get_config_value("AUTO_DEPLOY_ACTIVITY")
DEPLOY_COOLDOWN_SECONDS = int(get_config_value("DEPLOY_COOLDOWN_SECONDS", "1800") or 1800)
BUILD_COOLDOWN_SECONDS = int(get_config_value("BUILD_COOLDOWN_SECONDS", "900") or 900)
FAILURE_BACKOFF_THRESHOLD = int(get_config_value("FAILURE_BACKOFF_THRESHOLD", "3") or 3)
FAILURE_BACKOFF_BASE_SECONDS = int(get_config_value("FAILURE_BACKOFF_BASE_SECONDS", "300") or 300)
FAILURE_BACKOFF_MAX_SECONDS = int(get_config_value("FAILURE_BACKOFF_MAX_SECONDS", "3600") or 3600)
DAILY_TASK_LIMIT = int(get_config_value("DAILY_TASK_LIMIT", "500") or 500)

# Estimated cost per task by provider (rough averages in USD)
COST_PER_TASK_ESTIMATE = {
    "gemini": 0.002,       # Gemini Flash is very cheap
    "claude": 0.025,       # Claude Sonnet ~$3/MTok input, $15/MTok output
    "codex": 0.015,        # o4-mini moderate pricing
}

PULSE_POLICY_FILE = get_config_value("PULSE_POLICY_FILE", "C:/shadow/.aidev/pulse_policy.json")
PULSE_POLICY_START = "[PULSE_POLICY]"
PULSE_POLICY_END = "[/PULSE_POLICY]"

# Scope constraints injected into every task prompt
SCOPE_CONSTRAINTS = """
SCOPE CONSTRAINTS (MANDATORY):
ALLOWED: Fix TODOs, improve error handling, refactor methods,
         optimize performance, add tests, clean up code smells,
         add minor features within existing patterns.
FORBIDDEN: Database migrations, new Room entities, version bumps
           (pipeline handles), breaking API changes, adding new
           dependencies, modifying build.gradle.kts signing config,
           modifying AndroidManifest permissions,
           using fallbackToDestructiveMigration(),
           using adb uninstall or pm clear.

After completing your change:
1. Run a quick verification (compile check or syntax check)
2. Stage and commit with a descriptive message
3. Push to remote
4. Follow the protocol header at the top of this task for completion markers
"""

# Build trigger config
BUILD_COMPLETIONS_THRESHOLD = 3
BUILD_PRIORITY_THRESHOLD = 2  # Priority 1-2 triggers immediate build

# Loop timing
CYCLE_INTERVAL_SECONDS = 10
SCAN_STALE_MINUTES = 30

PULSE_FILE = "C:/shadow/.aidev/pulse.md"


class AutonomousLoop:
    """Main autonomous loop controller."""

    def __init__(self):
        self.scanner = AutonomousScanner()
        self.build_automation = BuildAutomation()
        self.deployment_automation = DeploymentAutomation()
        self.workspace_manager = WorkspaceManager()

        self.running = False
        self.paused = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Agent tracking
        self.agents: Dict[str, dict] = {}  # agent_id -> {role, provider, model, repo, current_task}
        self.agent_configs: List[dict] = []

        # Task tracking
        self.task_queue: List[dict] = []
        self.completed_tasks: List[dict] = []
        self.failed_tasks: List[dict] = []
        self._repo_completions: Dict[str, int] = {}  # repo -> completions since last build
        self._repo_workspaces: Dict[str, str] = {}  # repo -> workspace root
        self.efficiency_log: List[dict] = [] # Track task efficiency: {agent_id, task_id, duration_ms}
        self.estimated_cost_usd: float = 0.0  # Running cost estimate
        self._cost_by_provider: Dict[str, float] = {}  # provider -> cost
        self._last_build_time: float = 0.0
        self._last_deploy_time: float = 0.0
        self._last_build_skip_notice: float = 0.0
        self._last_deploy_skip_notice: float = 0.0
        self._failure_streak: int = 0
        self._backoff_until: float = 0.0
        self._requires_manual_resume: bool = False
        self._pulse_policy: Dict[str, object] = {}

        # Session stats
        self.cycle_count = 0
        self.started_at: Optional[str] = None
        self.last_scan_time: Optional[float] = None

    def _get_pulse_nudge(self) -> str:
        """Read current time nudges from pulse.md."""
        if os.path.exists(PULSE_FILE):
            try:
                with open(PULSE_FILE, "r") as f:
                    return f.read()
            except:
                pass
        return "No specific temporal nudges active."

    def _get_pulse_policy(self) -> Dict[str, object]:
        """Load pulse policy config (optional) for concurrency and task filtering."""
        policy = {}

        if PULSE_POLICY_FILE and os.path.exists(PULSE_POLICY_FILE):
            try:
                with open(PULSE_POLICY_FILE, "r", encoding="utf-8") as f:
                    policy = json.load(f) or {}
                    self._pulse_policy = policy
                    return policy
            except (OSError, json.JSONDecodeError):
                logger.warning("Failed to parse pulse policy file")

        if os.path.exists(PULSE_FILE):
            try:
                with open(PULSE_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                if PULSE_POLICY_START in content and PULSE_POLICY_END in content:
                    start = content.index(PULSE_POLICY_START) + len(PULSE_POLICY_START)
                    end = content.index(PULSE_POLICY_END)
                    block = content[start:end].strip()
                    if block:
                        policy = json.loads(block)
                else:
                    stripped = content.strip()
                    if stripped.startswith("{") and stripped.endswith("}"):
                        policy = json.loads(stripped)
            except (OSError, ValueError, json.JSONDecodeError):
                logger.warning("Failed to parse pulse policy from pulse.md")

        self._pulse_policy = policy or {}
        return self._pulse_policy

    def _resolve_active_pulse_policy(self) -> Dict[str, object]:
        """Resolve the active policy based on time windows."""
        policy = self._get_pulse_policy()
        if not policy:
            return {}

        base = policy.get("default", policy) if isinstance(policy, dict) else {}
        windows = policy.get("windows", []) if isinstance(policy, dict) else []
        if not windows:
            return self._normalize_pulse_policy(base or {})

        now = datetime.now().time()
        for window in windows:
            start_raw = window.get("start")
            end_raw = window.get("end")
            if not start_raw or not end_raw:
                continue
            try:
                start_parts = [int(part) for part in start_raw.split(":")]
                end_parts = [int(part) for part in end_raw.split(":")]
                start_time = datetime.now().replace(hour=start_parts[0], minute=start_parts[1], second=0, microsecond=0).time()
                end_time = datetime.now().replace(hour=end_parts[0], minute=end_parts[1], second=0, microsecond=0).time()
            except (ValueError, IndexError):
                continue

            if start_time <= end_time:
                in_window = start_time <= now <= end_time
            else:
                in_window = now >= start_time or now <= end_time

            if in_window:
                merged = dict(base) if isinstance(base, dict) else {}
                merged.update(window)
                merged.pop("start", None)
                merged.pop("end", None)
                return self._normalize_pulse_policy(merged)

        return self._normalize_pulse_policy(base or {})

    def _normalize_pulse_policy(self, policy: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(policy, dict):
            return {}
        normalized = dict(policy)

        if "max_agents" in normalized:
            try:
                normalized["max_agents"] = int(normalized["max_agents"])
            except (TypeError, ValueError):
                normalized.pop("max_agents", None)

        if "allowed_categories" in normalized and isinstance(normalized["allowed_categories"], str):
            normalized["allowed_categories"] = [
                item.strip() for item in normalized["allowed_categories"].split(",") if item.strip()
            ]

        return normalized

    def _backoff_active(self) -> bool:
        return self._backoff_until and time.time() < self._backoff_until

    def _build_cooldown_remaining(self) -> int:
        if not self._last_build_time:
            return 0
        remaining = int((self._last_build_time + BUILD_COOLDOWN_SECONDS) - time.time())
        return max(0, remaining)

    def _deploy_cooldown_remaining(self) -> int:
        if not self._last_deploy_time:
            return 0
        remaining = int((self._last_deploy_time + DEPLOY_COOLDOWN_SECONDS) - time.time())
        return max(0, remaining)

    def _record_success(self):
        if self._failure_streak:
            self._failure_streak = 0
        self._backoff_until = 0.0

    def _record_failure(self, reason: str):
        self._failure_streak += 1
        backoff_seconds = min(FAILURE_BACKOFF_BASE_SECONDS * self._failure_streak, FAILURE_BACKOFF_MAX_SECONDS)
        self._backoff_until = time.time() + backoff_seconds

        self._broadcast_event("backoff_started", {
            "reason": reason,
            "failure_streak": self._failure_streak,
            "backoff_seconds": backoff_seconds
        })

        if self._failure_streak >= FAILURE_BACKOFF_THRESHOLD:
            self.paused = True
            self._requires_manual_resume = True
            self._broadcast_event("halted", {
                "reason": reason,
                "failure_streak": self._failure_streak
            })

    def start(self, agent_count: int = 5, focus: str = "backend-polish",
              provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL,
              agent_configs: Optional[List[dict]] = None):
        """
        Start the autonomous loop.

        Args:
            agent_count: Number of agents to spawn
            focus: Focus area (backend-polish, android-focus, bridge-focus)
            provider: Default CLI provider for agents
            model: Default model for agents
            agent_configs: Optional per-agent config overrides
        """
        if self.running:
            # If already running, stop first to apply new model/key
            logger.info("Restarting autonomous loop to apply configuration changes...")
            self.stop()

        policy = self._resolve_active_pulse_policy()
        max_agents = policy.get("max_agents") if isinstance(policy, dict) else None
        if isinstance(max_agents, int) and agent_count > max_agents:
            agent_count = max_agents

        logger.info(f"Starting autonomous loop: {agent_count} agents, focus={focus} using {model}")

        self.running = True
        self.paused = False
        self.started_at = datetime.utcnow().isoformat()
        self.cycle_count = 0
        self._stop_event.clear()
        self._repo_completions = {}
        self._failure_streak = 0
        self._backoff_until = 0.0
        self._requires_manual_resume = False

        # Resolve agent configs
        self.agent_configs = self._resolve_configs(
            agent_count, focus, provider, model, agent_configs
        )

        # Spawn agent team
        self._spawn_team()

        # Start main loop thread
        self._thread = threading.Thread(
            target=self._run_loop,
            name="autonomous-loop",
            daemon=True,
        )
        self._thread.start()

        self._broadcast_status()
        self._sync_agents_to_disk()
        self._save_status()

    def stop(self):
        """Graceful shutdown of the autonomous loop and all agents."""
        if not self.running:
            return

        logger.info("Stopping autonomous loop...")
        self.running = False
        self._stop_event.set()

        # Stop all managed agents
        for agent_id in list(self.agents.keys()):
            try:
                stop_agent(agent_id, graceful=True)
            except Exception as e:
                logger.error(f"Error stopping agent {agent_id}: {e}")

        self.agents.clear()

        if self._thread:
            self._thread.join(timeout=10)

        self._broadcast_status()
        self._sync_agents_to_disk()
        self._save_status()
        logger.info("Autonomous loop stopped")

    def pause(self):
        """Pause task assignment. Agents stay alive but get no new tasks."""
        self.paused = True
        self._broadcast_status()
        logger.info("Autonomous loop paused")

    def resume(self):
        """Resume task assignment."""
        self.paused = False
        self._requires_manual_resume = False
        self._failure_streak = 0
        self._backoff_until = 0.0
        self._broadcast_status()
        logger.info("Autonomous loop resumed")

    def get_status(self) -> dict:
        """Get full status for dashboard."""
        uptime = None
        if self.started_at:
            started = datetime.fromisoformat(self.started_at)
            uptime = str(datetime.utcnow() - started).split(".")[0]

        # Sync data for counts
        sync_tasks = self.get_tasks()
        sync_completed = self.get_completed_tasks()
        
        # Build agents map from internal and SQLite sources
        all_agents = {}
        # Internal agents first
        for aid, info in self.agents.items():
            state = agent_task_states.get(aid, {})
            all_agents[aid] = {
                "name": info.get("name"),
                "role": info.get("specialty"),
                "provider": info.get("provider"),
                "model": info.get("model"),
                "repo": info.get("repo"),
                "current_task": info.get("current_task"),
                "current_task_id": state.get("task_id") or active_agents.get(aid, {}).get("current_task_id"),
                "current_thread_id": state.get("thread_id") or active_agents.get(aid, {}).get("current_thread_id"),
                "tasks_completed": info.get("tasks_completed", 0),
                "status": info.get("status", "unknown"),
            }
            
        # SQLite agents second
        try:
            import sqlite3
            SQLITE_DB_PATH = r"C:\shadow\backend\data\shadow_ai.db"
            if os.path.exists(SQLITE_DB_PATH):
                # Use context manager for sqlite connection
                with sqlite3.connect(SQLITE_DB_PATH, timeout=5.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, name, status, role, current_goal_id FROM agents")
                    rows = cursor.fetchall()
                    for row in rows:
                        aid = row['id']
                        if aid not in all_agents:
                            # Map SQLite fields to dashboard expected fields
                            # Ensure 'specialty' is mapped from 'role' for UI consistency
                            all_agents[aid] = {
                                "name": row['name'],
                                "role": row['role'],
                                "specialty": row['role'],
                                "provider": "remote", 
                                "model": "remote",
                                "repo": "remote",
                                "current_task": row['current_goal_id'],
                                "tasks_completed": 0,
                                "status": row['status'].lower() if row['status'] else "offline",
                            }
        except Exception as e:
            logger.error(f"Error merging SQLite agents in get_status: {e}")

        return {
            "running": self.running or len([a for a in all_agents.values() if a['status'] != 'offline']) > 0,
            "paused": self.paused,
            "requires_manual_resume": self._requires_manual_resume,
            "started_at": self.started_at,
            "uptime": uptime,
            "cycle_count": self.cycle_count,
            "tasks_completed": len(sync_completed),
            "task_queue": len(sync_tasks),
            "agents": all_agents,
            "tasks": sync_tasks, 
            "completed_tasks": sync_completed,
            "tasks_pending": sum(1 for t in sync_tasks if t.get("status") in ("pending", "active", "awaiting_approval")),
            "tasks_in_progress": sum(1 for t in sync_tasks if t.get("status") == "in_progress"),
            "tasks_failed": len(self.failed_tasks),
            "build_history": self.build_automation.get_build_history(),
            "repo_completions": dict(self._repo_completions),
            "failure_streak": self._failure_streak,
            "backoff_until": self._backoff_until,
            "build_cooldown_remaining": self._build_cooldown_remaining(),
            "deploy_cooldown_remaining": self._deploy_cooldown_remaining(),
            "pulse_policy": self._pulse_policy,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "cost_by_provider": {k: round(v, 4) for k, v in self._cost_by_provider.items()},
            "daily_task_limit": DAILY_TASK_LIMIT,
        }

    def get_tasks(self) -> List[dict]:
        """Get all tasks (pending + in-progress) including synced ones from SQLite."""
        tasks = list(self.task_queue)
        
        # Pull from SQLite ShadowAI DB
        try:
            import sqlite3
            SQLITE_DB_PATH = r"C:\shadow\backend\data\shadow_ai.db"
            if os.path.exists(SQLITE_DB_PATH):
                with sqlite3.connect(SQLITE_DB_PATH, timeout=5.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, description as title, status, priority, metadata FROM goals WHERE status != 'completed'")
                    rows = cursor.fetchall()
                    for row in rows:
                        status = row['status'].lower()
                        if status == 'active': status = 'pending' # Map for dashboard

                        # Deduplicate
                        if not any(t.get("id") == str(row['id']) for t in tasks):
                            tasks.append({
                                "id": str(row['id']),
                                "title": row['title'],
                                "status": status,
                                "priority": row['priority'],
                                "repo": "shadow-ai" # Default
                            })
        except Exception as e:
            logger.error(f"Error pulling tasks from SQLite: {e}")

        try:
            from .data_service import _read_json_file, Path
            SHADOWAI_DIR = Path.home() / ".shadowai"
            TASKS_FILE = SHADOWAI_DIR / "tasks.json"
            if TASKS_FILE.exists():
                with open(TASKS_FILE, "r", encoding="utf-8") as f:
                    import json
                    sync_tasks = json.load(f)
                    for st in sync_tasks:
                        if st.get("status") in ["PENDING", "IN_PROGRESS", "in_progress", "pending"]:
                            # Map sync status to loop status
                            s = st.get("status").lower()
                            st["status"] = s
                            if not any(t.get("id") == st.get("id") for t in tasks):
                                tasks.append(st)
        except Exception as e:
            logger.error(f'Error syncing tasks: {e}')
        return tasks

    def get_completed_tasks(self) -> List[dict]:
        """Get completed tasks including synced ones via SQLite and data service."""
        tasks = list(self.completed_tasks)

        # Pull from SQLite ShadowAI DB
        try:
            import sqlite3
            SQLITE_DB_PATH = r"C:\shadow\backend\data\shadow_ai.db"
            if os.path.exists(SQLITE_DB_PATH):
                with sqlite3.connect(SQLITE_DB_PATH, timeout=5.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, description as title, status, priority FROM goals WHERE status = 'completed'")
                    rows = cursor.fetchall()
                    for row in rows:
                        if not any(t.get("id") == str(row['id']) for t in tasks):
                            tasks.append({
                                "id": str(row['id']),
                                "title": row['title'],
                                "status": "completed",
                                "priority": row['priority']
                            })
        except Exception as e:
            logger.error(f"Error pulling completed tasks from SQLite: {e}")

        try:
            # Use cached data service to get all tasks
            from .data_service import get_tasks as get_sync_tasks
            synced_tasks = get_sync_tasks()
            
            for st in synced_tasks:
                if st.get("status") in ["COMPLETED", "completed"]:
                    # Normalize status
                    st["status"] = "completed"
                    # Deduplicate by ID
                    if not any(t.get("id") == st.get("id") for t in tasks):
                        tasks.append(st)
                        
        except Exception as e:
            logger.error(f"Error syncing completed tasks: {e}")
            
        return tasks

    def add_tasks(self, tasks: List[dict]) -> int:
        """Add tasks to the in-memory queue with simple dedupe."""
        if not tasks:
            return 0

        existing_ids = {str(t.get("id")) for t in self.task_queue}
        existing_keys = {(t.get("repo", ""), t.get("title", "")) for t in self.task_queue}
        added = 0

        for task in tasks:
            task_id = str(task.get("id"))
            key = (task.get("repo", ""), task.get("title", ""))
            if task_id in existing_ids or key in existing_keys:
                continue
            self.task_queue.append(task)
            existing_ids.add(task_id)
            existing_keys.add(key)
            added += 1

        return added

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue or SQLite database."""
        # 1. Check in-memory queue
        for i, task in enumerate(self.task_queue):
            if str(task.get("id")) == str(task_id):
                self.task_queue.pop(i)
                logger.info(f"Removed task {task_id} from in-memory queue")
                return True

        # 2. Check SQLite database (goals table)
        try:
            import sqlite3
            SQLITE_DB_PATH = r"C:\shadow\backend\data\shadow_ai.db"
            if os.path.exists(SQLITE_DB_PATH):
                with sqlite3.connect(SQLITE_DB_PATH, timeout=5.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM goals WHERE id = ?", (task_id,))
                    if cursor.fetchone():
                        cursor.execute("DELETE FROM goals WHERE id = ?", (task_id,))
                        conn.commit()
                        logger.info(f"Deleted task {task_id} from SQLite goals table")
                        return True
        except Exception as e:
            logger.error(f"Error deleting task {task_id} from SQLite: {e}")

        return False

    def get_builds(self) -> List[dict]:
        """Get build history."""
        return self.build_automation.get_build_history()

    # ---- Main Loop ----

    def _run_loop(self):
        """Main autonomous loop with task seeking and generation."""
        logger.info("Autonomous loop thread started")

        while self.running and not self._stop_event.is_set():
            try:
                active_policy = self._resolve_active_pulse_policy()
                self._pulse_policy = active_policy
                max_agents = active_policy.get("max_agents")
                allowed_categories = active_policy.get("allowed_categories")

                if not self.paused and not self._requires_manual_resume:
                    self.cycle_count += 1
                    logger.debug(f"Autonomous cycle #{self.cycle_count}")

                    # 1. Scan if queue empty or stale
                    if not self._backoff_active() and self._should_scan():
                        self._do_scan()
                        
                    # 2. Seek for tasks: Generate new ones if queue still empty
                    if not self._backoff_active() and not any(t.get("status") == "pending" for t in self.task_queue):
                        self._generate_tasks()

                    # 3. Assign tasks to idle agents
                    if not self._backoff_active():
                        self._assign_tasks(max_agents=max_agents, allowed_categories=allowed_categories)

                    # 4. Check for task completions
                    self._check_completions()

                    # 5. Check if build should trigger
                    if not self._backoff_active():
                        self._check_build_trigger()

                    # 6. Periodic workspace garbage collection (every 100 cycles)
                    if self.cycle_count % 100 == 0:
                        try:
                            gc_result = self.workspace_manager.gc(max_age_hours=24)
                            if gc_result["removed"] > 0:
                                logger.info(f"Workspace GC: removed {gc_result['removed']} stale workspaces")
                        except Exception as e:
                            logger.warning(f"Workspace GC failed: {e}")

                    # Broadcast status
                    self._broadcast_status()
                    self._sync_agents_to_disk()
                else:
                    # Still check for completions and update status while paused/backing off
                    self._check_completions()
                    self._broadcast_status()
                    self._sync_agents_to_disk()

            except Exception as e:
                logger.exception(f"Error in autonomous cycle #{self.cycle_count}: {e}")

            # Sleep between cycles
            self._stop_event.wait(CYCLE_INTERVAL_SECONDS)

        logger.info("Autonomous loop thread exiting")

    def _generate_tasks(self):
        """Proactively generate new tasks if the queue is empty."""
        logger.info("Task queue empty. Seeking new tasks...")
        policy = self._resolve_active_pulse_policy()
        allowed_categories = policy.get("allowed_categories") if isinstance(policy, dict) else None
        category_hint = ""
        if allowed_categories:
            category_hint = f"\nOnly use these categories: {', '.join(allowed_categories)}.\n"
        
        # Select an idle agent to generate tasks for the team
        idle_agents = [aid for aid, info in self.agents.items() if not info.get("current_task")]
        if not idle_agents:
            # Fallback to specialized agents from DB if loop agents are all busy
            # For now, let's just use loop agents to avoid complexity
            return
            
        agent_id = idle_agents[0]
        agent_info = self.agents[agent_id]
        
        # Use an AI query to brainstorm tasks based on the current repo
        repo = agent_info.get("repo", "shadow-bridge")
        
        # We'll trigger a deep scan first
        self._broadcast_event("seeking_tasks", {"agent": agent_info.get("name"), "repo": repo})
        self._do_scan()
        
        # If still empty, use AI to brainstorm
        if not any(t.get("status") == "pending" for t in self.task_queue):
            logger.info(f"Scan found nothing. Brainstorming with AI agent {agent_id}...")
            
            prompt = f"""
[TASK GENERATION MODE]
Analyze the {repo} repository context. 
Your goal is to identify 3-5 high-value technical tasks that improve the codebase.
Focus on: 1. Code quality, 2. Performance bottlenecks, 3. Unimplemented TODOs, 4. Test gaps.
{category_hint}

After the protocol markers, respond with a JSON list of tasks:
[
  {{"title": "Task Title", "description": "Full details", "category": "refactoring", "priority": 3, "scope": "small", "repo": "{repo}"}},
  ...
]
"""
            # Assign the task to the agent
            assign_task(agent_id, prompt, task_id=f"brainstorm-{uuid.uuid4().hex[:8]}")
            agent_info["status"] = "brainstorming"
            agent_info["current_task"] = "Generating new tasks"

    def _assign_tasks(self, max_agents: Optional[int] = None, allowed_categories: Optional[List[str]] = None):
        """Assign pending tasks to idle agents, enforcing approvals for risky tasks."""
        # Daily task limit safety check
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_completed = sum(
            1 for t in self.completed_tasks
            if (t.get("completed_at") or t.get("created_at", "")) >= today_start
        )
        if today_completed >= DAILY_TASK_LIMIT:
            if not getattr(self, '_daily_limit_warned', False):
                logger.warning(f"Daily task limit reached ({DAILY_TASK_LIMIT}). Pausing assignments.")
                self._daily_limit_warned = True
            return
        self._daily_limit_warned = False

        pulse_nudge = self._get_pulse_nudge()
        active_count = sum(1 for info in self.agents.values() if info.get("current_task"))
        slots = None
        if isinstance(max_agents, int):
            slots = max_agents - active_count
            if slots <= 0:
                return
        assigned = 0
        
        for agent_id, agent_info in list(self.agents.items()):
            if agent_info.get("current_task"):
                continue  # Agent is busy

            # Check if agent process is still alive
            if agent_id not in active_agents:
                agent_info["status"] = "dead"
                continue

            agent_status = active_agents.get(agent_id, {}).get("status")
            if agent_status not in ("idle", None):
                continue

            # Find a matching task
            task = self._find_task_for_agent(agent_info, allowed_categories=allowed_categories)
            if not task:
                continue
                
            # ENFORCE APPROVAL: Risky tasks (large scope) require manual approval
            if task.get("scope") == "large" and task.get("status") != "approved":
                if task.get("status") != "awaiting_approval":
                    task["status"] = "awaiting_approval"
                    logger.info(f"Task '{task['title']}' requires approval (LARGE scope)")
                    self._broadcast_event("approval_required", {
                        "task_id": task["id"],
                        "task_title": task["title"],
                        "reason": "Large scope task detected"
                    })
                continue

            workspace_allocation = self.workspace_manager.allocate(
                task.get("repo", ""),
                task.get("id", ""),
                agent_id
            )
            if not workspace_allocation.success:
                logger.warning(
                    f"Workspace allocation failed for task {task.get('id')}: {workspace_allocation.error}"
                )
                self._broadcast_event("task_blocked", {
                    "task_id": task.get("id"),
                    "task_title": task.get("title"),
                    "agent_id": agent_id,
                    "reason": workspace_allocation.error or "workspace_allocation_failed",
                    "repo": task.get("repo")
                })
                continue

            task["workspace_root"] = workspace_allocation.workspace_root
            task["workspace_branch"] = workspace_allocation.branch
            task["workspace"] = workspace_allocation.workspace_root

            # Build prompt and assign
            prompt = self._build_task_prompt(task, pulse_nudge)
            lock_paths = task.get("lock_paths") or []
            if not lock_paths and task.get("file_path"):
                lock_paths = [task.get("file_path")]
            success = assign_task(
                agent_id,
                prompt,
                task_id=task["id"],
                repo=task.get("repo"),
                lock_paths=lock_paths,
                workspace_root=task.get("workspace_root")
            )

            if success:
                task["status"] = "in_progress"
                task["assigned_to"] = agent_id
                task["assigned_at"] = datetime.utcnow().isoformat()
                task["started_at_ts"] = time.time() # For efficiency tracking
                agent_info["current_task"] = task["id"]
                agent_info["status"] = "working"
                assigned += 1

                logger.info(f"Assigned task '{task['title']}' to agent {agent_info.get('name')}")

                self._broadcast_event("task_assigned", {
                    "task_id": task["id"],
                    "task_title": task["title"],
                    "agent_id": agent_id,
                    "agent_name": agent_info.get("name"),
                    "repo": task.get("repo"),
                    "workspace": task.get("workspace_root")
                })

                if slots is not None and assigned >= slots:
                    break

    def _should_scan(self) -> bool:
        """Check if we need to rescan repos."""
        pending = sum(1 for t in self.task_queue if t.get("status") == "pending")
        if pending == 0:
            return True

        if self.last_scan_time is None:
            return True

        age_minutes = (time.time() - self.last_scan_time) / 60
        return age_minutes > SCAN_STALE_MINUTES

    def _do_scan(self):
        """Run the code scanner and update task queue."""
        logger.info("Running code scanner...")
        try:
            new_tasks = self.scanner.scan_all()

            # Merge with existing queue (keep in-progress tasks)
            in_progress = [t for t in self.task_queue if t.get("status") == "in_progress"]
            existing_ids = {t["id"] for t in in_progress}

            # Add new tasks that aren't already being worked on
            for task in new_tasks:
                if task["id"] not in existing_ids:
                    in_progress.append(task)

            self.task_queue = in_progress
            self.last_scan_time = time.time()
            logger.info(f"Scanner found {len(new_tasks)} tasks, queue now has {len(self.task_queue)}")

            self._broadcast_event("scan_complete", {
                "tasks_found": len(new_tasks),
                "queue_size": len(self.task_queue),
            })

        except Exception as e:
            logger.exception(f"Scanner error: {e}")

    def _find_task_for_agent(self, agent_info: dict, allowed_categories: Optional[List[str]] = None) -> Optional[dict]:
        """Find the best pending task for an agent based on its role and repo."""
        agent_repo = agent_info.get("repo", "")
        agent_specialty = agent_info.get("specialty", "general")

        # Category mapping from specialty
        specialty_categories = {
            "refactoring": ["smell", "todo"],
            "debugging": ["error_handling", "todo"],
            "testing": ["test_gap"],
            "performance": ["performance"],
            "code_review": ["smell", "error_handling", "todo"],
            "general": ["todo", "smell", "error_handling", "performance", "test_gap"],
        }
        preferred = specialty_categories.get(agent_specialty, ["todo"])

        # Sort pending tasks: matching repo + category first, then by score
        pending = [t for t in self.task_queue if t.get("status") == "pending"]
        if allowed_categories:
            pending = [t for t in pending if t.get("category") in allowed_categories]

        def task_sort_key(t):
            repo_match = 1 if agent_repo and t.get("repo", "") in agent_repo else 0
            cat_match = 1 if t.get("category") in preferred else 0
            score = t.get("score", 0)
            return (repo_match, cat_match, score)

        pending.sort(key=task_sort_key, reverse=True)
        return pending[0] if pending else None

    def _check_completions(self):
        """Check if any agents have completed their tasks."""
        for agent_id, agent_info in list(self.agents.items()):
            task_id = agent_info.get("current_task")
            if not task_id:
                continue

            # Check completion via event system
            event = agent_task_events.get(agent_id)
            if event and event.is_set():
                state = agent_task_states.get(agent_id, {})
                status = state.get("done_status") or state.get("result_status")
                if not status and isinstance(state.get("result"), dict):
                    status = state["result"].get("status")

                success = status == "success"
                if status is None:
                    logger.warning(f"Task {task_id} completed without status for agent {agent_id}")

                self._handle_task_completion(agent_id, agent_info, task_id, success=success)
                event.clear()
                continue

            # Check if agent process died
            if agent_id not in active_agents:
                logger.warning(f"Agent {agent_id} process died during task {task_id}")
                self._handle_task_completion(agent_id, agent_info, task_id, success=False)

    def _handle_task_completion(self, agent_id: str, agent_info: dict,
                                task_id: str, success: bool):
        """Handle a task being completed or failed."""
        
        # 1. Handle brainstorming results (New tasks from AI)
        if agent_info.get("status") == "brainstorming":
            if success:
                logger.info(f"Agent {agent_id} successfully brainstormed new tasks.")
                # We need to extract the JSON from the agent logs
                try:
                    from web.services.agent_orchestrator import agent_logs
                    logs = agent_logs.get(agent_id, [])
                    # Find last output lines
                    output_text = "\n".join([l["line"] for l in logs[-20:]])
                    
                    # Extract JSON array
                    import re
                    match = re.search(r"\[\s*\{.*\}\s*\]", output_text, re.DOTALL)
                    if match:
                        new_tasks = json.loads(match.group(0))
                        added_count = 0
                        for nt in new_tasks:
                            nt["id"] = nt.get("id") or str(uuid.uuid4())
                            nt["status"] = "pending"
                            nt["created_at"] = datetime.utcnow().isoformat()
                            if nt["id"] not in [t["id"] for t in self.task_queue]:
                                self.task_queue.append(nt)
                                added_count += 1
                        
                        logger.info(f"Added {added_count} AI-generated tasks to queue.")
                        self._broadcast_event("tasks_generated", {"count": added_count, "agent": agent_info.get("name")})
                except Exception as e:
                    logger.error(f"Failed to parse AI-generated tasks: {e}")
            
            agent_info["status"] = "idle"
            agent_info["current_task"] = None
            return

        # 2. Standard task completion
        task = next((t for t in self.task_queue if t.get("id") == task_id), None)

        if task:
            # Efficiency tracking
            duration_ms = 0
            if "started_at_ts" in task:
                duration_ms = int((time.time() - task["started_at_ts"]) * 1000)
                self.efficiency_log.append({
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "duration_ms": duration_ms,
                    "timestamp": datetime.utcnow().isoformat()
                })
                # Keep last 100 entries
                if len(self.efficiency_log) > 100:
                    self.efficiency_log.pop(0)

            if success:
                task["status"] = "completed"
                task["completed_at"] = datetime.utcnow().isoformat()
                task["efficiency_ms"] = duration_ms
                self.completed_tasks.append(task)
                self.task_queue.remove(task)
                self._record_success()

                # Track repo completions for build trigger
                repo = task.get("repo", "")
                self._repo_completions[repo] = self._repo_completions.get(repo, 0) + 1
                workspace_root = task.get("workspace") or task.get("workspace_root")
                if workspace_root:
                    self._repo_workspaces[repo] = workspace_root

                # Mark in scanner so it won't regenerate
                self.scanner.mark_completed(task)

                agent_info["tasks_completed"] = agent_info.get("tasks_completed", 0) + 1

                # Track estimated cost
                provider = agent_info.get("provider", "gemini").lower()
                task_cost = COST_PER_TASK_ESTIMATE.get(provider, 0.005)
                self.estimated_cost_usd += task_cost
                self._cost_by_provider[provider] = self._cost_by_provider.get(provider, 0.0) + task_cost

                logger.info(f"Task completed: {task.get('title')} in {duration_ms}ms (est. cost: ${task_cost:.4f})")
                self._broadcast_event("task_completed", {
                    "task_id": task_id,
                    "task_title": task.get("title"),
                    "agent_id": agent_id,
                    "efficiency_ms": duration_ms,
                    "repo": task.get("repo")
                })

                # Cleanup workspace after successful completion
                self._cleanup_task_workspace(task, agent_id)
            else:
                task["status"] = "failed"
                task["failed_at"] = datetime.utcnow().isoformat()
                self.failed_tasks.append(task)
                self.task_queue.remove(task)

                # Cleanup workspace after failure too
                self._cleanup_task_workspace(task, agent_id)
                self._record_failure(f"task_failed:{task.get('title')}")

                logger.warning(f"Task failed: {task.get('title')}")
                self._broadcast_event("task_failed", {
                    "task_id": task_id,
                    "task_title": task.get("title"),
                    "agent_id": agent_id,
                    "repo": task.get("repo")
                })

        # Reset agent
        agent_info["current_task"] = None
        agent_info["status"] = "idle"

    def _cleanup_task_workspace(self, task: dict, agent_id: str):
        """Clean up workspace after task completion/failure."""
        try:
            repo = task.get("repo", "")
            task_id = task.get("id", "")
            thread_id = agent_id
            if repo and task_id:
                self.workspace_manager.cleanup_workspace(repo, task_id, thread_id)
        except Exception as e:
            logger.warning(f"Workspace cleanup failed for task {task.get('id')}: {e}")

    def _check_build_trigger(self):
        """Check if we should trigger a build."""
        if self._backoff_active() or self._requires_manual_resume:
            return

        cooldown_remaining = self._build_cooldown_remaining()
        if cooldown_remaining > 0:
            now = time.time()
            if now - self._last_build_skip_notice > 60:
                self._broadcast_event("build_skipped", {
                    "reason": "build_cooldown",
                    "cooldown_remaining": cooldown_remaining
                })
                self._last_build_skip_notice = now
            return

        for repo, count in self._repo_completions.items():
            if count >= BUILD_COMPLETIONS_THRESHOLD:
                logger.info(f"Build trigger: {count} completions in {repo}")
                self._trigger_build(repo)
                self._repo_completions[repo] = 0

    def _trigger_build(self, repo: str):
        """Trigger a build gate for a repo."""
        logger.info(f"Triggering build gate for {repo}")
        self._last_build_time = time.time()

        workspace_root = self._repo_workspaces.get(repo) or self._get_working_dir(repo)
        self._broadcast_event("build_started", {"repo": repo, "workspace": workspace_root})

        try:
            result: GateResult = self.build_automation.run_gate(repo, workspace_root)

            event_type = "build_success" if result.success else "build_failed"
            steps_payload = [
                {
                    "name": step.name,
                    "success": step.success,
                    "duration_seconds": step.duration_seconds,
                    "error": step.error,
                    "artifacts": step.artifacts,
                }
                for step in result.steps
            ]
            self._broadcast_event(event_type, {
                "repo": repo,
                "success": result.success,
                "error": result.error,
                "duration": result.duration_seconds,
                "workspace": result.workspace,
                "steps": steps_payload,
            })

            if result.success:
                self._record_success()
            else:
                self._record_failure(f"build_failed:{repo}")

            if result.success and AUTO_DEPLOY_ENABLED and "android" in (repo or "").lower():
                deploy_cooldown = self._deploy_cooldown_remaining()
                if deploy_cooldown > 0:
                    now = time.time()
                    if now - self._last_deploy_skip_notice > 60:
                        self._broadcast_event("deploy_skipped", {
                            "repo": repo,
                            "reason": "deploy_cooldown",
                            "cooldown_remaining": deploy_cooldown
                        })
                        self._last_deploy_skip_notice = now
                    return

                self._last_deploy_time = time.time()
                self._broadcast_event("deploy_started", {
                    "repo": repo,
                    "workspace": workspace_root
                })
                deploy_result = self.deployment_automation.deploy_android(
                    workspace_root,
                    activity=AUTO_DEPLOY_ACTIVITY
                )
                deploy_event = "deploy_success" if deploy_result.success else "deploy_failed"
                self._broadcast_event(deploy_event, {
                    "repo": repo,
                    "workspace": deploy_result.workspace,
                    "apk_path": deploy_result.apk_path,
                    "package_name": deploy_result.package_name,
                    "devices": [
                        {
                            "device_id": r.device_id,
                            "success": r.success,
                            "error": r.error
                        }
                        for r in deploy_result.devices
                    ],
                    "error": deploy_result.error
                })
                if deploy_result.success:
                    self._record_success()
                else:
                    self._record_failure(f"deploy_failed:{repo}")
                if deploy_result.rollback_attempted:
                    self._broadcast_event("deploy_rollback", {
                        "repo": repo,
                        "workspace": deploy_result.workspace,
                        "devices": [
                            {
                                "device_id": r.device_id,
                                "success": r.success,
                                "error": r.error
                            }
                            for r in deploy_result.rollback_results
                        ]
                    })

        except Exception as e:
            logger.exception(f"Build gate error: {e}")
            self._broadcast_event("build_failed", {
                "repo": repo,
                "error": str(e),
            })

    # ---- Agent Management ----

    def _spawn_team(self):
        """Spawn the agent team based on configs."""
        google_api_key = get_config_value("GOOGLE_API_KEY")
        anthropic_api_key = get_config_value("ANTHROPIC_API_KEY")
        openai_api_key = get_config_value("OPENAI_API_KEY")

        for config in self.agent_configs:
            try:
                # Prepare environment for CLI authentication
                env = {}
                provider = config.get("provider", "").lower()
                if "gemini" in provider:
                    if google_api_key:
                        env["GOOGLE_API_KEY"] = google_api_key
                elif "claude" in provider:
                    if anthropic_api_key:
                        env["ANTHROPIC_API_KEY"] = anthropic_api_key
                elif "codex" in provider:
                    if openai_api_key:
                        env["OPENAI_API_KEY"] = openai_api_key

                agent_info = spawn_agent(
                    device_id="autonomous",
                    name=config["name"],
                    specialty=config["specialty"],
                    cli_provider=config.get("provider", "gemini"),
                    model=config.get("model", DEFAULT_MODEL),
                    working_directory=self._get_working_dir(config.get("repo", "")),
                    auto_accept_edits=True,
                    env=env
                )

                self.agents[agent_info["id"]] = {
                    **config,
                    "provider": config.get("provider", "gemini"),
                    "model": config.get("model", DEFAULT_MODEL),
                    "current_task": None,
                    "tasks_completed": 0,
                    "status": "idle",
                }

                logger.info(f"Spawned agent: {config['name']} ({agent_info['id']}) using {config.get('model', DEFAULT_MODEL)}")

            except Exception as e:
                logger.error(f"Failed to spawn agent {config['name']}: {e}")

    def _resolve_configs(self, count: int, focus: str, provider: str,
                         model: str, overrides: Optional[List[dict]]) -> List[dict]:
        """Resolve agent configurations from focus area and overrides."""
        if overrides:
            return overrides[:count]

        roles = AGENT_ROLES.get(focus, AGENT_ROLES["backend-polish"])

        configs = []
        for i in range(min(count, len(roles))):
            config = dict(roles[i])
            config["provider"] = provider
            config["model"] = model
            configs.append(config)

        # If requesting more agents than roles, duplicate with suffix
        while len(configs) < count:
            base = roles[len(configs) % len(roles)]
            config = dict(base)
            config["name"] = f"{base['name']} #{len(configs) + 1}"
            config["provider"] = provider
            config["model"] = model
            configs.append(config)

        return configs

    def _get_working_dir(self, repo: str) -> str:
        """Get working directory for a repo."""
        if "android" in repo.lower():
            return "C:/shadow/shadow-android"
        elif "bridge" in repo.lower():
            return "C:/shadow/shadow-bridge"
        return "C:/shadow"

    # ---- Task Prompt Builder ----

    def _build_task_prompt(self, task: dict, pulse_nudge: str = "") -> str:
        """Build a complete task prompt for an agent, including temporal pulse nudges."""
        return f"""[AUTONOMOUS MODE - TEMPORAL PULSE ACTIVE]

You are an autonomous agent working on the ShadowAI codebase.
Current Time: {datetime.now().strftime('%H:%M:%S')}

[TEMPORAL NUDGE]
{pulse_nudge}

TASK: {task.get('title', 'Unknown task')}

DESCRIPTION:
{task.get('description', 'No description')}

CATEGORY: {task.get('category', 'unknown')}
PRIORITY: {task.get('priority', 3)} (1=critical, 5=nice-to-have)
REPO: {task.get('repo', 'unknown')}
WORKSPACE_ROOT: {task.get('workspace_root') or task.get('workspace') or 'unassigned'}
WORKSPACE_BRANCH: {task.get('workspace_branch') or 'unassigned'}

MANDATORY: Run `cd {task.get('workspace_root') or task.get('workspace') or '.'}` before any commands.
Do not edit files outside the workspace root.

{SCOPE_CONSTRAINTS}

INSTRUCTIONS:
1. Read the file mentioned in the task
2. Understand the surrounding code context
3. Make the minimal change needed to fix the issue
4. Verify your change doesn't break anything
5. Stage, commit, and push with: git add <files> && git commit -m "<type>: <description>" && git push
6. Use the protocol header at the top of this task for completion markers
"""

    # ---- Broadcasting ----

    def _broadcast_status(self):
        """Broadcast current status via WebSocket."""
        # CRITICAL: Import orchestrator to get the real broadcast function
        try:
            from web.services.agent_orchestrator import broadcast_agent_event
            
            status_data = self.get_status() # Get the unified status
            
            # Update the dashboard real-time list
            for agent_id, agent_info in status_data.get("agents", {}).items():
                broadcast_agent_event("agent_status_changed", {
                    "id": agent_id,
                    "status": agent_info.get("status", "idle"),
                    "name": agent_info.get("name"),
                    "specialty": agent_info.get("role"), 
                    "current_task": agent_info.get("current_task"),
                    "current_task_id": agent_info.get("current_task_id"),
                    "current_thread_id": agent_info.get("current_thread_id"),
                    "tasks_completed": agent_info.get("tasks_completed", 0)
                })
                
                # Broadcast activity for ANY agent doing something
                if agent_info.get("status") in ("busy", "working", "active"):
                    task = agent_info.get("current_task") or "Executing sub-task"
                    
                    # For agent terminal box
                    broadcast_agent_event("agent_output_line", {
                        "agent_id": agent_id,
                        "line": f"> [{agent_info.get('name')}] Working on: {task}",
                        "stream": "stdout"
                    })
                    
                    # For global activity monitor (activity-feed)
                    broadcast_agent_event("autonomous_activity", {
                        "message": f"{agent_info.get('name')} is active: {task}",
                        "type": "info",
                        "agent_id": agent_id
                    })
                elif agent_info.get("status") == "idle" and self.cycle_count % 5 == 0:
                    # Occasional idle pulse to show life
                    broadcast_agent_event("autonomous_activity", {
                        "message": f"{agent_info.get('name')} is seeking tasks...",
                        "type": "debug",
                        "agent_id": agent_id
                    })
                    
            # Also broadcast the global autonomous status
            broadcast_agent_event("autonomous_status", status_data)
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")

    def _broadcast_event(self, event_type: str, data: dict):
        """Broadcast a specific event."""
        try:
            broadcast_agent_event(f"autonomous_{event_type}", data)
        except Exception as e:
            logger.error(f'Error syncing tasks: {e}')

    # ---- Persistence ----


    def _sync_agents_to_disk(self):
        """Sync current autonomous agents to agents.json for dashboard visibility."""
        try:
            status = self.get_status()
            agents_list = []
            
            for aid, info in status.get("agents", {}).items():
                agents_list.append({
                    "id": aid,
                    "name": info.get("name"),
                    "type": "AUTONOMOUS_PC",
                    "role": info.get("role"),
                    "specialty": info.get("role"),
                    "status": info.get("status", "offline").upper(),
                    "current_task": info.get("current_task"),
                    "tasks_completed": info.get("tasks_completed", 0),
                    "model": info.get("model"),
                    "provider": info.get("provider")
                })
            
            from web.services.data_service import sync_agents_from_device
            sync_agents_from_device("autonomous-team", agents_list)
            
        except Exception as e:
            logger.error(f"Failed to sync agents to disk: {e}")

    def _save_status(self):
        """Save status to disk."""
        try:
            os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
            status = self.get_status()
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save status: {e}")


# Global singleton
_autonomous_loop: Optional[AutonomousLoop] = None


def get_autonomous_loop() -> AutonomousLoop:
    """Get or create the global AutonomousLoop instance."""
    global _autonomous_loop
    if _autonomous_loop is None:
        _autonomous_loop = AutonomousLoop()
    return _autonomous_loop
