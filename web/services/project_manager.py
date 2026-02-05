"""
Project Manager

Manages multiple simultaneous project deployments, each with its own
autonomous loop instance, budget tracking, and project-scoped isolation.

Usage:
    pm = get_project_manager()
    dep = pm.deploy_project(config)
    pm.stop_project(project_id)
    projects = pm.get_all_projects()
"""

import os
import time
import uuid
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from web.services.autonomous_loop import AutonomousLoop
from web.services.rules_engine import get_rules_engine, _project_id_from_path

logger = logging.getLogger(__name__)

MAX_ACTIVE_PROJECTS = int(os.environ.get("MAX_ACTIVE_PROJECTS", "10"))


@dataclass
class ProjectConfig:
    project_path: str
    display_name: str = ""
    agent_count: int = 5
    provider: str = "gemini"
    model: str = ""
    focus: str = "general"
    preset: str = "general"
    custom_rules: List[str] = field(default_factory=list)
    budget_limit_usd: float = 0.0
    max_runtime_hours: int = 0
    agent_configs: Optional[List[dict]] = None


@dataclass
class ProjectDeployment:
    project_id: str
    config: ProjectConfig
    loop: AutonomousLoop
    status: str = "starting"
    started_at: str = ""
    stopped_at: str = ""
    error: str = ""
    _runtime_timer: Optional[threading.Timer] = field(default=None, repr=False)


class ProjectManager:
    """Manages multiple project deployments with isolated autonomous loops."""

    def __init__(self):
        self._lock = threading.Lock()
        self._projects: Dict[str, ProjectDeployment] = {}

    def deploy_project(self, config: ProjectConfig) -> Dict:
        project_id = _project_id_from_path(config.project_path)

        with self._lock:
            existing = self._projects.get(project_id)
            if existing and existing.status in ("starting", "running", "paused"):
                return {"success": False, "error": f"Project already active ({existing.status})", "project_id": project_id}

            active_count = sum(1 for p in self._projects.values() if p.status in ("starting", "running", "paused"))
            if active_count >= MAX_ACTIVE_PROJECTS:
                return {"success": False, "error": f"Max active projects ({MAX_ACTIVE_PROJECTS}) reached"}

        engine = get_rules_engine()
        engine.get_or_create_rules(config.project_path, config.display_name or os.path.basename(config.project_path))

        loop = AutonomousLoop()
        deployment = ProjectDeployment(
            project_id=project_id, config=config, loop=loop,
            status="starting", started_at=datetime.utcnow().isoformat(),
        )

        with self._lock:
            self._projects[project_id] = deployment

        try:
            focus = config.focus or "backend-polish"
            if "android" in config.project_path.lower():
                focus = "android-focus"
            elif "bridge" in config.project_path.lower():
                focus = "bridge-focus"

            loop.start(
                agent_count=config.agent_count, focus=focus,
                provider=config.provider, model=config.model,
                agent_configs=config.agent_configs, budget_limit_usd=config.budget_limit_usd,
            )
            deployment.status = "running"

            if config.max_runtime_hours > 0:
                timer = threading.Timer(config.max_runtime_hours * 3600, self._runtime_expired, args=(project_id,))
                timer.daemon = True
                timer.start()
                deployment._runtime_timer = timer

            logger.info(f"Project {project_id} deployed: {config.agent_count} agents on {config.project_path}")
            return {"success": True, "project_id": project_id, "agent_count": config.agent_count, "status": "running"}

        except Exception as e:
            deployment.status = "failed"
            deployment.error = str(e)
            logger.error(f"Project {project_id} deploy failed: {e}")
            return {"success": False, "error": str(e), "project_id": project_id}

    def stop_project(self, project_id: str) -> Dict:
        with self._lock:
            dep = self._projects.get(project_id)
            if not dep:
                return {"success": False, "error": "Project not found"}
            if dep.status in ("stopped", "failed"):
                return {"success": False, "error": f"Project already {dep.status}"}
            dep.status = "stopping"

        if dep._runtime_timer:
            dep._runtime_timer.cancel()
            dep._runtime_timer = None

        try:
            if dep.loop.running:
                dep.loop.stop()
            with self._lock:
                dep.status = "stopped"
                dep.stopped_at = datetime.utcnow().isoformat()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def pause_project(self, project_id: str) -> Dict:
        with self._lock:
            dep = self._projects.get(project_id)
            if not dep or dep.status != "running":
                return {"success": False, "error": "Cannot pause"}
        dep.loop.paused = True
        dep.status = "paused"
        return {"success": True}

    def resume_project(self, project_id: str) -> Dict:
        with self._lock:
            dep = self._projects.get(project_id)
            if not dep or dep.status != "paused":
                return {"success": False, "error": "Cannot resume"}
        dep.loop.paused = False
        dep.status = "running"
        return {"success": True}

    def get_project_status(self, project_id: str) -> Optional[Dict]:
        with self._lock:
            dep = self._projects.get(project_id)
            if not dep:
                return None
        return self._to_dict(dep)

    def get_all_projects(self) -> List[Dict]:
        with self._lock:
            deps = list(self._projects.values())
        return [self._to_dict(d) for d in deps]

    def get_active_projects(self) -> List[Dict]:
        with self._lock:
            deps = [d for d in self._projects.values() if d.status in ("starting", "running", "paused")]
        return [self._to_dict(d) for d in deps]

    def get_aggregate_status(self) -> Dict:
        with self._lock:
            deps = list(self._projects.values())
        active = [d for d in deps if d.status in ("starting", "running", "paused")]
        return {
            "active_projects": len(active),
            "total_projects": len(deps),
            "total_agents": sum(d.config.agent_count for d in active),
            "total_tasks_queued": sum(len(d.loop.task_queue) for d in active),
            "total_tasks_completed": sum(len(d.loop.completed_tasks) for d in active),
            "total_estimated_cost_usd": round(sum(getattr(d.loop, 'estimated_cost_usd', 0.0) for d in active), 4),
        }

    def _runtime_expired(self, project_id: str):
        logger.warning(f"Project {project_id} runtime limit reached")
        self.stop_project(project_id)

    def _to_dict(self, dep: ProjectDeployment) -> Dict:
        return {
            "project_id": dep.project_id,
            "project_path": dep.config.project_path,
            "display_name": dep.config.display_name or os.path.basename(dep.config.project_path),
            "status": dep.status,
            "agent_count": dep.config.agent_count,
            "provider": dep.config.provider,
            "model": dep.config.model,
            "budget_limit_usd": dep.config.budget_limit_usd,
            "started_at": dep.started_at,
            "stopped_at": dep.stopped_at,
            "error": dep.error,
            "tasks_queued": len(dep.loop.task_queue) if dep.loop else 0,
            "tasks_completed": len(dep.loop.completed_tasks) if dep.loop else 0,
            "estimated_cost_usd": round(getattr(dep.loop, 'estimated_cost_usd', 0.0), 4),
        }


_project_manager: Optional[ProjectManager] = None
_pm_lock = threading.Lock()


def get_project_manager() -> ProjectManager:
    global _project_manager
    if _project_manager is None:
        with _pm_lock:
            if _project_manager is None:
                _project_manager = ProjectManager()
    return _project_manager
