"""In-memory task store with JSON file persistence for the continuity system."""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.path.join(str(Path.home()), ".shadowai")
TASK_STORE_FILE = "task_store.json"


class TaskStore:
    """Thread-safe in-memory task store backed by JSON file."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.file_path = os.path.join(self.data_dir, TASK_STORE_FILE)
        self._lock = threading.Lock()
        self.tasks: Dict[str, dict] = {}
        self.agents: Dict[str, dict] = {}
        self.teams: Dict[str, dict] = {}
        self.events: Dict[str, List[dict]] = {}  # task_id -> list of events
        self._start_time = time.time()
        self._load()

    def _load(self):
        """Load state from JSON file."""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tasks = data.get("tasks", {})
                self.agents = data.get("agents", {})
                self.teams = data.get("teams", {})
                self.events = data.get("events", {})
                log.info(f"TaskStore loaded: {len(self.tasks)} tasks, "
                         f"{len(self.agents)} agents, {len(self.teams)} teams")
        except Exception as e:
            log.warning(f"Failed to load task store: {e}")

    def _save(self):
        """Persist state to JSON file."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            payload = {
                "tasks": self.tasks,
                "agents": self.agents,
                "teams": self.teams,
                "events": self.events,
                "updated": time.time(),
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save task store: {e}")

    # ---- Tasks ----

    def create_task(self, task_data: dict) -> dict:
        """Create a new task. Returns the created task with generated id."""
        with self._lock:
            task_id = task_data.get("id") or str(uuid.uuid4())[:8]
            now = time.time()
            task = {
                "id": task_id,
                "title": task_data.get("title", "Untitled"),
                "description": task_data.get("description", ""),
                "status": task_data.get("status", "PENDING"),
                "priority": task_data.get("priority", "NORMAL"),
                "executor": task_data.get("executor"),
                "agent_id": task_data.get("agent_id"),
                "team_id": task_data.get("team_id"),
                "input": task_data.get("input", {}),
                "output": task_data.get("output"),
                "checkpoint": task_data.get("checkpoint"),
                "created_at": task_data.get("created_at", now),
                "updated_at": now,
                "claimed_at": None,
                "completed_at": None,
                "tags": task_data.get("tags", []),
            }
            self.tasks[task_id] = task
            self.events[task_id] = []
            self._save()
            return task

    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def update_task(self, task_id: str, updates: dict) -> Optional[dict]:
        """Update task fields. Returns updated task or None."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            for key, value in updates.items():
                if key != "id":  # Don't allow changing ID
                    task[key] = value
            task["updated_at"] = time.time()
            self._save()
            return task

    def list_tasks(self, status: Optional[str] = None,
                   executor: Optional[str] = None) -> List[dict]:
        """List tasks with optional filters."""
        results = list(self.tasks.values())
        if status:
            results = [t for t in results if t.get("status") == status]
        if executor:
            results = [t for t in results if t.get("executor") == executor]
        # Sort by created_at descending
        results.sort(key=lambda t: t.get("created_at", 0), reverse=True)
        return results

    def get_pending_tasks(self, executor: Optional[str] = None) -> List[dict]:
        """Get pending tasks, optionally filtered by executor preference."""
        with self._lock:
            results = [t for t in self.tasks.values()
                       if t.get("status") == "PENDING"]
            if executor:
                # Return tasks not yet claimed or assigned to this executor
                results = [t for t in results if not t.get("executor")
                           or t.get("executor") == executor]
            results.sort(key=lambda t: t.get("created_at", 0))
            return results

    def claim_task(self, task_id: str, executor: str) -> Optional[dict]:
        """Claim a pending task. Sets executor and status to IN_PROGRESS."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            if task["status"] != "PENDING":
                return None  # Already claimed or completed
            now = time.time()
            task["executor"] = executor
            task["status"] = "IN_PROGRESS"
            task["claimed_at"] = now
            task["updated_at"] = now
            self._save()
            return task

    def complete_task(self, task_id: str, output: dict = None) -> Optional[dict]:
        """Mark task as completed with optional output."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            now = time.time()
            task["status"] = "COMPLETED"
            task["completed_at"] = now
            task["updated_at"] = now
            if output is not None:
                task["output"] = output
            self._save()
            return task

    def save_checkpoint(self, task_id: str, checkpoint: dict) -> Optional[dict]:
        """Save a checkpoint for a task."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            task["checkpoint"] = checkpoint
            task["updated_at"] = time.time()
            self._save()
            return task

    # ---- Events ----

    def add_event(self, task_id: str, event_data: dict) -> dict:
        """Add an execution event to a task."""
        with self._lock:
            if task_id not in self.events:
                self.events[task_id] = []
            event = {
                "id": str(uuid.uuid4())[:8],
                "task_id": task_id,
                "type": event_data.get("type", "LOG"),
                "message": event_data.get("message", ""),
                "data": event_data.get("data"),
                "timestamp": time.time(),
            }
            self.events[task_id].append(event)
            self._save()
            return event

    def get_events(self, task_id: str, since: float = 0) -> List[dict]:
        """Get events for a task, optionally filtered by timestamp."""
        events = self.events.get(task_id, [])
        if since:
            events = [e for e in events if e.get("timestamp", 0) > since]
        return events

    # ---- Agents ----

    def upsert_agent(self, agent_data: dict) -> dict:
        """Create or update an agent."""
        with self._lock:
            agent_id = agent_data.get("id") or str(uuid.uuid4())[:8]
            existing = self.agents.get(agent_id, {})
            agent = {**existing, **agent_data, "id": agent_id,
                     "updated_at": time.time()}
            self.agents[agent_id] = agent
            self._save()
            return agent

    def get_agent(self, agent_id: str) -> Optional[dict]:
        return self.agents.get(agent_id)

    def list_agents(self) -> List[dict]:
        return list(self.agents.values())

    # ---- Teams ----

    def upsert_team(self, team_data: dict) -> dict:
        """Create or update a team."""
        with self._lock:
            team_id = team_data.get("id") or str(uuid.uuid4())[:8]
            existing = self.teams.get(team_id, {})
            team = {**existing, **team_data, "id": team_id,
                    "updated_at": time.time()}
            self.teams[team_id] = team
            self._save()
            return team

    def list_teams(self) -> List[dict]:
        return list(self.teams.values())

    # ---- Stats ----

    def get_stats(self) -> dict:
        """Get system stats."""
        return {
            "total_tasks": len(self.tasks),
            "pending": len([t for t in self.tasks.values()
                           if t.get("status") == "PENDING"]),
            "in_progress": len([t for t in self.tasks.values()
                               if t.get("status") == "IN_PROGRESS"]),
            "completed": len([t for t in self.tasks.values()
                             if t.get("status") == "COMPLETED"]),
            "total_agents": len(self.agents),
            "total_teams": len(self.teams),
            "uptime_seconds": time.time() - self._start_time,
        }


# Singleton instance
_store: Optional[TaskStore] = None
_store_lock = threading.Lock()


def get_task_store() -> TaskStore:
    """Get or create the singleton TaskStore instance."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = TaskStore()
    return _store
