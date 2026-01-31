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

from .autonomous_scanner import AutonomousScanner
from .build_pipeline import BuildPipeline, PipelineResult
from .agent_orchestrator import (
    spawn_agent,
    assign_task,
    stop_agent,
    get_all_agents,
    active_agents,
    agent_task_events,
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
4. Output [TASK:COMPLETE] when done
5. If you cannot complete, output [TASK:ERROR] with reason
"""

# Build trigger config
BUILD_COMPLETIONS_THRESHOLD = 3
BUILD_PRIORITY_THRESHOLD = 2  # Priority 1-2 triggers immediate build

# Loop timing
CYCLE_INTERVAL_SECONDS = 60
SCAN_STALE_MINUTES = 30
TASK_TIMEOUT_SECONDS = 1800  # 30 min per task


class AutonomousLoop:
    """Main autonomous loop controller."""

    def __init__(self):
        self.scanner = AutonomousScanner()
        self.pipeline = BuildPipeline()

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

        # Session stats
        self.cycle_count = 0
        self.started_at: Optional[str] = None
        self.last_scan_time: Optional[float] = None

    def start(self, agent_count: int = 5, focus: str = "backend-polish",
              provider: str = "claude", model: str = "claude-sonnet-4-20250514",
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
            logger.warning("Autonomous loop already running")
            return

        logger.info(f"Starting autonomous loop: {agent_count} agents, focus={focus}")

        self.running = True
        self.paused = False
        self.started_at = datetime.utcnow().isoformat()
        self.cycle_count = 0
        self._stop_event.clear()
        self._repo_completions = {}

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
        self._broadcast_status()
        logger.info("Autonomous loop resumed")

    def get_status(self) -> dict:
        """Get full status for dashboard."""
        uptime = None
        if self.started_at:
            started = datetime.fromisoformat(self.started_at)
            uptime = str(datetime.utcnow() - started).split(".")[0]

        return {
            "running": self.running,
            "paused": self.paused,
            "started_at": self.started_at,
            "uptime": uptime,
            "cycle_count": self.cycle_count,
            "agents": {
                aid: {
                    "name": info.get("name"),
                    "role": info.get("specialty"),
                    "provider": info.get("provider"),
                    "model": info.get("model"),
                    "repo": info.get("repo"),
                    "current_task": info.get("current_task"),
                    "tasks_completed": info.get("tasks_completed", 0),
                    "status": info.get("status", "unknown"),
                }
                for aid, info in self.agents.items()
            },
            "task_queue": len(self.task_queue),
            "tasks_pending": sum(1 for t in self.task_queue if t.get("status") == "pending"),
            "tasks_in_progress": sum(1 for t in self.task_queue if t.get("status") == "in_progress"),
            "tasks_completed": len(self.completed_tasks),
            "tasks_failed": len(self.failed_tasks),
            "build_history": self.pipeline.get_build_history(),
            "repo_completions": dict(self._repo_completions),
        }

    def get_tasks(self) -> List[dict]:
        """Get all tasks (pending + in-progress)."""
        return list(self.task_queue)

    def get_completed_tasks(self) -> List[dict]:
        """Get completed tasks."""
        return list(self.completed_tasks)

    def get_builds(self) -> List[dict]:
        """Get build history."""
        return self.pipeline.get_build_history()

    # ---- Main Loop ----

    def _run_loop(self):
        """Main autonomous loop."""
        logger.info("Autonomous loop thread started")

        while self.running and not self._stop_event.is_set():
            try:
                if not self.paused:
                    self.cycle_count += 1
                    logger.debug(f"Autonomous cycle #{self.cycle_count}")

                    # 1. Scan if queue empty or stale
                    if self._should_scan():
                        self._do_scan()

                    # 2. Assign tasks to idle agents
                    self._assign_tasks()

                    # 3. Check for task completions
                    self._check_completions()

                    # 4. Check if build should trigger
                    self._check_build_trigger()

                    # Broadcast status
                    self._broadcast_status()

            except Exception as e:
                logger.exception(f"Error in autonomous cycle #{self.cycle_count}: {e}")

            # Sleep between cycles
            self._stop_event.wait(CYCLE_INTERVAL_SECONDS)

        logger.info("Autonomous loop thread exiting")

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

    def _assign_tasks(self):
        """Assign pending tasks to idle agents."""
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
            task = self._find_task_for_agent(agent_info)
            if not task:
                continue

            # Build prompt and assign
            prompt = self._build_task_prompt(task)
            success = assign_task(agent_id, prompt)

            if success:
                task["status"] = "in_progress"
                task["assigned_to"] = agent_id
                task["assigned_at"] = datetime.utcnow().isoformat()
                agent_info["current_task"] = task["id"]
                agent_info["status"] = "working"

                logger.info(f"Assigned task '{task['title']}' to agent {agent_info.get('name')}")

                self._broadcast_event("task_assigned", {
                    "task_id": task["id"],
                    "task_title": task["title"],
                    "agent_id": agent_id,
                    "agent_name": agent_info.get("name"),
                })

    def _find_task_for_agent(self, agent_info: dict) -> Optional[dict]:
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
                self._handle_task_completion(agent_id, agent_info, task_id, success=True)
                event.clear()
                continue

            # Check for timeout
            task = next((t for t in self.task_queue if t.get("id") == task_id), None)
            if task and task.get("assigned_at"):
                assigned = datetime.fromisoformat(task["assigned_at"])
                if (datetime.utcnow() - assigned).total_seconds() > TASK_TIMEOUT_SECONDS:
                    logger.warning(f"Task {task_id} timed out for agent {agent_id}")
                    self._handle_task_completion(agent_id, agent_info, task_id, success=False)

            # Check if agent process died
            if agent_id not in active_agents:
                logger.warning(f"Agent {agent_id} process died during task {task_id}")
                self._handle_task_completion(agent_id, agent_info, task_id, success=False)

    def _handle_task_completion(self, agent_id: str, agent_info: dict,
                                task_id: str, success: bool):
        """Handle a task being completed or failed."""
        task = next((t for t in self.task_queue if t.get("id") == task_id), None)

        if task:
            if success:
                task["status"] = "completed"
                task["completed_at"] = datetime.utcnow().isoformat()
                self.completed_tasks.append(task)
                self.task_queue.remove(task)

                # Track repo completions for build trigger
                repo = task.get("repo", "")
                self._repo_completions[repo] = self._repo_completions.get(repo, 0) + 1

                # Mark in scanner so it won't regenerate
                self.scanner.mark_completed(task)

                agent_info["tasks_completed"] = agent_info.get("tasks_completed", 0) + 1

                logger.info(f"Task completed: {task.get('title')}")
                self._broadcast_event("task_completed", {
                    "task_id": task_id,
                    "task_title": task.get("title"),
                    "agent_id": agent_id,
                })
            else:
                task["status"] = "failed"
                task["failed_at"] = datetime.utcnow().isoformat()
                self.failed_tasks.append(task)
                self.task_queue.remove(task)

                logger.warning(f"Task failed: {task.get('title')}")
                self._broadcast_event("task_failed", {
                    "task_id": task_id,
                    "task_title": task.get("title"),
                    "agent_id": agent_id,
                })

        # Reset agent
        agent_info["current_task"] = None
        agent_info["status"] = "idle"

    def _check_build_trigger(self):
        """Check if we should trigger a build."""
        for repo, count in self._repo_completions.items():
            if count >= BUILD_COMPLETIONS_THRESHOLD:
                logger.info(f"Build trigger: {count} completions in {repo}")
                self._trigger_build(repo)
                self._repo_completions[repo] = 0

    def _trigger_build(self, repo: str):
        """Trigger a build pipeline for a repo."""
        logger.info(f"Triggering build pipeline for {repo}")

        self._broadcast_event("build_started", {"repo": repo})

        try:
            if "android" in repo.lower():
                result = self.pipeline.run_pipeline("shadow-android")
            elif "bridge" in repo.lower():
                result = self.pipeline.run_pipeline("shadow-bridge", deploy=False)
            else:
                logger.warning(f"Unknown repo for build: {repo}")
                return

            event_type = "build_success" if result.success else "build_failed"
            self._broadcast_event(event_type, {
                "repo": repo,
                "success": result.success,
                "error": result.error,
                "build_type": result.build.build_type if result.build else None,
                "duration": result.build.duration_seconds if result.build else None,
            })

        except Exception as e:
            logger.exception(f"Build pipeline error: {e}")
            self._broadcast_event("build_failed", {
                "repo": repo,
                "error": str(e),
            })

    # ---- Agent Management ----

    def _spawn_team(self):
        """Spawn the agent team based on configs."""
        for config in self.agent_configs:
            try:
                agent_info = spawn_agent(
                    device_id="autonomous",
                    name=config["name"],
                    specialty=config["specialty"],
                    cli_provider=config.get("provider", "claude"),
                    model=config.get("model", "claude-sonnet-4-20250514"),
                    working_directory=self._get_working_dir(config.get("repo", "")),
                    auto_accept_edits=True,
                )

                self.agents[agent_info["id"]] = {
                    **config,
                    "provider": config.get("provider", "claude"),
                    "model": config.get("model", "claude-sonnet-4-20250514"),
                    "current_task": None,
                    "tasks_completed": 0,
                    "status": "idle",
                }

                logger.info(f"Spawned agent: {config['name']} ({agent_info['id']})")

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

    def _build_task_prompt(self, task: dict) -> str:
        """Build a complete task prompt for an agent."""
        return f"""[AUTONOMOUS MODE]

You are an autonomous agent working on the ShadowAI codebase.

TASK: {task.get('title', 'Unknown task')}

DESCRIPTION:
{task.get('description', 'No description')}

CATEGORY: {task.get('category', 'unknown')}
PRIORITY: {task.get('priority', 3)} (1=critical, 5=nice-to-have)
REPO: {task.get('repo', 'unknown')}

{SCOPE_CONSTRAINTS}

INSTRUCTIONS:
1. Read the file mentioned in the task
2. Understand the surrounding code context
3. Make the minimal change needed to fix the issue
4. Verify your change doesn't break anything
5. Stage, commit, and push with: git add <files> && git commit -m "<type>: <description>" && git push
6. Output [TASK:COMPLETE] when done, or [TASK:ERROR] if you cannot complete

[TASK:START]
"""

    # ---- Broadcasting ----

    def _broadcast_status(self):
        """Broadcast current status via WebSocket."""
        try:
            broadcast_agent_event("autonomous_status", self.get_status())
        except Exception:
            pass

    def _broadcast_event(self, event_type: str, data: dict):
        """Broadcast a specific event."""
        try:
            broadcast_agent_event(f"autonomous_{event_type}", data)
        except Exception:
            pass

    # ---- Persistence ----

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
