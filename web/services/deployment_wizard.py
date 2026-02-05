"""
Deployment Wizard Service

Unified service for deploying agent teams to projects.
Sits above the autonomous loop and orchestrator, providing a single entry point
for all deployment paths (web dashboard, Android, CLI).

Supports all providers: Claude, Gemini, Codex, Ollama.
Manages deployment lifecycle, presets, cost estimation, and active deployments.
"""

import os
import uuid
import time
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from .autonomous_loop import get_autonomous_loop, get_config_value, COST_PER_TASK_ESTIMATE
from .agent_orchestrator import check_system_resources, get_all_agents

logger = logging.getLogger(__name__)


# ---- Provider / Model Catalog ----

PROVIDER_MODELS = {
    "claude": {
        "display": "Claude (Anthropic)",
        "models": [
            {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "cost_per_task": 0.025},
            {"id": "claude-opus-4-20250514", "name": "Claude Opus 4", "cost_per_task": 0.08},
            {"id": "claude-haiku-3-20250514", "name": "Claude Haiku 3.5", "cost_per_task": 0.005},
        ],
        "cli": "claude",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "display": "Gemini (Google)",
        "models": [
            {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash", "cost_per_task": 0.002},
            {"id": "gemini-2.5-pro-preview-06-05", "name": "Gemini 2.5 Pro", "cost_per_task": 0.015},
            {"id": "gemini-2.5-flash-preview-05-20", "name": "Gemini 2.5 Flash", "cost_per_task": 0.003},
        ],
        "cli": "gemini",
        "env_key": "GOOGLE_API_KEY",
    },
    "codex": {
        "display": "Codex (OpenAI)",
        "models": [
            {"id": "o4-mini", "name": "o4-mini", "cost_per_task": 0.015},
            {"id": "gpt-4.1", "name": "GPT-4.1", "cost_per_task": 0.03},
        ],
        "cli": "codex",
        "env_key": "OPENAI_API_KEY",
    },
    "ollama": {
        "display": "Ollama (Local)",
        "models": [
            {"id": "llama3.1:8b", "name": "Llama 3.1 8B", "cost_per_task": 0.0},
            {"id": "codellama:13b", "name": "Code Llama 13B", "cost_per_task": 0.0},
            {"id": "deepseek-coder:6.7b", "name": "DeepSeek Coder 6.7B", "cost_per_task": 0.0},
        ],
        "cli": "ollama",
        "env_key": None,
    },
}

# ---- Focus Presets ----

DEPLOYMENT_PRESETS = {
    "code_review": {
        "name": "Code Review",
        "description": "Agents review code for quality, bugs, and best practices",
        "icon": "rate_review",
        "default_rules": [
            "Focus on code quality, readability, and maintainability",
            "Flag potential bugs, security issues, and performance problems",
            "Suggest improvements but do not make changes directly",
        ],
        "focus": "backend-polish",
        "specialty_mix": {"code_review": 0.6, "debugging": 0.2, "performance": 0.2},
    },
    "testing": {
        "name": "Testing",
        "description": "Agents write and improve test coverage",
        "icon": "science",
        "default_rules": [
            "Write unit tests for uncovered functions and classes",
            "Add integration tests for critical paths",
            "Ensure tests are deterministic and fast",
        ],
        "focus": "backend-polish",
        "specialty_mix": {"testing": 0.7, "debugging": 0.2, "code_review": 0.1},
    },
    "refactoring": {
        "name": "Refactoring",
        "description": "Agents clean up and modernize code",
        "icon": "construction",
        "default_rules": [
            "Improve code structure without changing behavior",
            "Remove dead code and reduce duplication",
            "Follow existing patterns and conventions",
        ],
        "focus": "backend-polish",
        "specialty_mix": {"refactoring": 0.6, "code_review": 0.2, "performance": 0.2},
    },
    "bug_fixing": {
        "name": "Bug Fixing",
        "description": "Agents find and fix bugs from issues/TODOs",
        "icon": "bug_report",
        "default_rules": [
            "Fix bugs identified in TODOs and issue trackers",
            "Add regression tests for each fix",
            "Verify fixes do not break existing functionality",
        ],
        "focus": "backend-polish",
        "specialty_mix": {"debugging": 0.6, "testing": 0.2, "code_review": 0.2},
    },
    "performance": {
        "name": "Performance",
        "description": "Agents optimize slow paths and resource usage",
        "icon": "speed",
        "default_rules": [
            "Profile and identify performance bottlenecks",
            "Optimize hot paths and reduce memory usage",
            "Add benchmarks to verify improvements",
        ],
        "focus": "backend-polish",
        "specialty_mix": {"performance": 0.6, "refactoring": 0.2, "testing": 0.2},
    },
    "general": {
        "name": "General",
        "description": "Balanced agent team for mixed tasks",
        "icon": "auto_awesome",
        "default_rules": [
            "Work on the most impactful tasks first",
            "Balance between fixing bugs, improving code, and adding tests",
        ],
        "focus": "backend-polish",
        "specialty_mix": {"refactoring": 0.2, "debugging": 0.2, "testing": 0.2, "code_review": 0.2, "performance": 0.2},
    },
}

# Known project paths to scan for repos
PROJECT_SCAN_PATHS = [
    "C:/shadow/shadow-android",
    "C:/shadow/shadow-bridge",
    "C:/shadow",
]


@dataclass
class DeploymentConfig:
    """Configuration for a deployment."""
    project_path: str
    agent_count: int = 5
    provider: str = "gemini"
    model: str = "gemini-3-flash-preview"
    preset: str = "general"
    custom_rules: List[str] = field(default_factory=list)
    budget_limit_usd: float = 0.0  # 0 = unlimited
    max_runtime_hours: int = 0  # 0 = unlimited
    scan_tasks: bool = True  # auto-scan for tasks on start


@dataclass
class Deployment:
    """Tracks an active deployment."""
    id: str
    config: DeploymentConfig
    status: str = "starting"  # starting, running, paused, stopping, stopped, failed
    started_at: str = ""
    stopped_at: str = ""
    agent_ids: List[str] = field(default_factory=list)
    tasks_completed: int = 0
    tasks_failed: int = 0
    estimated_cost_usd: float = 0.0
    error: str = ""


class DeploymentWizard:
    """Manages deployment lifecycle for agent teams."""

    def __init__(self):
        self._lock = threading.Lock()
        self._deployments: Dict[str, Deployment] = {}
        self._runtime_watchers: Dict[str, threading.Timer] = {}

    def get_providers(self) -> Dict:
        """Get available providers and their models with API key status."""
        result = {}
        for provider_id, info in PROVIDER_MODELS.items():
            has_key = True
            if info["env_key"]:
                key_val = get_config_value(info["env_key"])
                has_key = bool(key_val and key_val.strip())

            result[provider_id] = {
                "display": info["display"],
                "models": info["models"],
                "cli": info["cli"],
                "has_api_key": has_key,
            }
        return result

    def get_presets(self) -> Dict:
        """Get available deployment presets."""
        return {
            k: {
                "name": v["name"],
                "description": v["description"],
                "icon": v["icon"],
                "default_rules": v["default_rules"],
            }
            for k, v in DEPLOYMENT_PRESETS.items()
        }

    def discover_projects(self) -> List[Dict]:
        """Discover git repos in known paths."""
        projects = []
        seen = set()

        for path in PROJECT_SCAN_PATHS:
            norm = os.path.normpath(path)
            if norm in seen:
                continue
            seen.add(norm)

            if os.path.isdir(os.path.join(norm, ".git")):
                name = os.path.basename(norm)
                projects.append({
                    "path": norm.replace("\\", "/"),
                    "name": name,
                    "has_git": True,
                })

        # Also scan user-configured paths
        extra_paths = get_config_value("DEPLOY_PROJECT_PATHS", "")
        if extra_paths:
            for p in extra_paths.split(";"):
                p = p.strip()
                if not p:
                    continue
                norm = os.path.normpath(p)
                if norm in seen:
                    continue
                seen.add(norm)
                if os.path.isdir(norm):
                    projects.append({
                        "path": norm.replace("\\", "/"),
                        "name": os.path.basename(norm),
                        "has_git": os.path.isdir(os.path.join(norm, ".git")),
                    })

        return projects

    def estimate(self, config: DeploymentConfig) -> Dict:
        """Estimate resource usage and cost for a deployment."""
        resources = check_system_resources()

        # Estimate tasks per hour based on provider speed
        tasks_per_agent_per_hour = {
            "gemini": 12,
            "claude": 6,
            "codex": 8,
            "ollama": 4,
        }.get(config.provider, 6)

        # Cost per task for this model
        cost_per_task = 0.0
        provider_info = PROVIDER_MODELS.get(config.provider, {})
        for m in provider_info.get("models", []):
            if m["id"] == config.model:
                cost_per_task = m["cost_per_task"]
                break
        if cost_per_task == 0.0:
            cost_per_task = COST_PER_TASK_ESTIMATE.get(config.provider, 0.01)

        total_tasks_per_hour = tasks_per_agent_per_hour * config.agent_count
        cost_per_hour = total_tasks_per_hour * cost_per_task

        # RAM estimate: CLI agents use ~200-400MB each
        ram_per_agent_mb = {
            "claude": 300,
            "gemini": 250,
            "codex": 250,
            "ollama": 500,
        }.get(config.provider, 300)

        total_ram_mb = ram_per_agent_mb * config.agent_count

        # Budget runtime estimate
        hours_until_budget = None
        if config.budget_limit_usd > 0 and cost_per_hour > 0:
            hours_until_budget = round(config.budget_limit_usd / cost_per_hour, 1)

        return {
            "can_deploy": resources["can_spawn"],
            "resource_warning": resources.get("reason", ""),
            "system_metrics": resources.get("metrics", {}),
            "estimated_tasks_per_hour": total_tasks_per_hour,
            "estimated_cost_per_hour": round(cost_per_hour, 4),
            "estimated_ram_mb": total_ram_mb,
            "ram_per_agent_mb": ram_per_agent_mb,
            "hours_until_budget": hours_until_budget,
            "max_recommended_agents": resources.get("metrics", {}).get("estimated_capacity", 50),
            "cost_per_task": cost_per_task,
        }

    def deploy(self, config: DeploymentConfig) -> Dict:
        """Deploy an agent team to a project."""
        deployment_id = f"dep-{uuid.uuid4().hex[:12]}"

        # Validate
        if config.agent_count < 1:
            return {"success": False, "error": "Agent count must be at least 1"}
        if config.agent_count > 200:
            return {"success": False, "error": "Maximum 200 agents per deployment"}

        # Check resources
        resources = check_system_resources()
        if not resources["can_spawn"]:
            return {
                "success": False,
                "error": f"Insufficient resources: {resources['reason']}",
                "metrics": resources.get("metrics", {}),
            }

        # Check provider API key
        provider_info = PROVIDER_MODELS.get(config.provider)
        if not provider_info:
            return {"success": False, "error": f"Unknown provider: {config.provider}"}

        if provider_info["env_key"]:
            key_val = get_config_value(provider_info["env_key"])
            if not key_val or not key_val.strip():
                return {
                    "success": False,
                    "error": f"No API key configured for {provider_info['display']}. "
                             f"Set {provider_info['env_key']} in backend/.env",
                }

        # Build agent configs from preset
        preset = DEPLOYMENT_PRESETS.get(config.preset, DEPLOYMENT_PRESETS["general"])
        specialty_mix = preset["specialty_mix"]

        agent_configs = []
        specialties = list(specialty_mix.keys())
        weights = list(specialty_mix.values())

        for i in range(config.agent_count):
            # Distribute specialties by weight
            weight_sum = sum(weights)
            cumulative = 0
            chosen_specialty = specialties[0]
            target = (i % 100) / 100.0  # Spread evenly

            for s, w in zip(specialties, weights):
                cumulative += w / weight_sum
                if target < cumulative:
                    chosen_specialty = s
                    break

            agent_configs.append({
                "name": f"{preset['name']} Agent #{i + 1}",
                "specialty": chosen_specialty,
                "repo": os.path.basename(config.project_path),
                "provider": config.provider,
                "model": config.model,
            })

        # Create deployment record
        deployment = Deployment(
            id=deployment_id,
            config=config,
            status="starting",
            started_at=datetime.utcnow().isoformat(),
        )

        with self._lock:
            self._deployments[deployment_id] = deployment

        # Build custom rules string for task prompts
        rules_text = ""
        all_rules = preset.get("default_rules", []) + (config.custom_rules or [])
        if all_rules:
            rules_text = "\n".join(f"- {r}" for r in all_rules)

        # Start the autonomous loop with these configs
        try:
            loop = get_autonomous_loop()

            # If already running, stop first
            if loop.running:
                loop.stop()
                time.sleep(1)

            # Determine focus area from project path
            focus = preset.get("focus", "backend-polish")
            if "android" in config.project_path.lower():
                focus = "android-focus"
            elif "bridge" in config.project_path.lower():
                focus = "bridge-focus"

            loop.start(
                agent_count=config.agent_count,
                focus=focus,
                provider=config.provider,
                model=config.model,
                agent_configs=agent_configs,
                budget_limit_usd=config.budget_limit_usd,
            )

            # Track agent IDs from the loop
            deployment.agent_ids = list(loop.agents.keys())
            deployment.status = "running"

            # Set up runtime limit watcher
            if config.max_runtime_hours > 0:
                timer = threading.Timer(
                    config.max_runtime_hours * 3600,
                    self._runtime_expired,
                    args=(deployment_id,),
                )
                timer.daemon = True
                timer.start()
                self._runtime_watchers[deployment_id] = timer

            logger.info(
                f"Deployment {deployment_id} started: "
                f"{config.agent_count} agents on {config.project_path} "
                f"using {config.provider}/{config.model}"
            )

            return {
                "success": True,
                "deployment_id": deployment_id,
                "agent_count": len(deployment.agent_ids),
                "status": "running",
            }

        except Exception as e:
            deployment.status = "failed"
            deployment.error = str(e)
            logger.error(f"Deployment {deployment_id} failed: {e}")
            return {"success": False, "error": str(e)}

    def stop_deployment(self, deployment_id: str) -> Dict:
        """Stop an active deployment."""
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return {"success": False, "error": "Deployment not found"}

            if deployment.status in ("stopped", "failed"):
                return {"success": False, "error": f"Deployment already {deployment.status}"}

            deployment.status = "stopping"

        # Cancel runtime watcher
        timer = self._runtime_watchers.pop(deployment_id, None)
        if timer:
            timer.cancel()

        try:
            loop = get_autonomous_loop()
            if loop.running:
                loop.stop()

            with self._lock:
                deployment.status = "stopped"
                deployment.stopped_at = datetime.utcnow().isoformat()

            logger.info(f"Deployment {deployment_id} stopped")
            return {"success": True, "status": "stopped"}

        except Exception as e:
            logger.error(f"Error stopping deployment {deployment_id}: {e}")
            return {"success": False, "error": str(e)}

    def pause_deployment(self, deployment_id: str) -> Dict:
        """Pause an active deployment (agents stay alive)."""
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return {"success": False, "error": "Deployment not found"}

            if deployment.status != "running":
                return {"success": False, "error": f"Can only pause running deployments (current: {deployment.status})"}

        loop = get_autonomous_loop()
        if loop.running:
            loop.pause()

        with self._lock:
            deployment.status = "paused"

        return {"success": True, "status": "paused"}

    def resume_deployment(self, deployment_id: str) -> Dict:
        """Resume a paused deployment."""
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return {"success": False, "error": "Deployment not found"}

            if deployment.status != "paused":
                return {"success": False, "error": f"Can only resume paused deployments (current: {deployment.status})"}

        loop = get_autonomous_loop()
        if loop.running:
            loop.resume()

        with self._lock:
            deployment.status = "running"

        return {"success": True, "status": "running"}

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict]:
        """Get detailed status for a deployment."""
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return None

        # Sync live data from the autonomous loop
        loop = get_autonomous_loop()
        loop_status = loop.get_status() if loop.running else {}

        # Calculate uptime
        uptime_str = ""
        if deployment.started_at:
            try:
                started = datetime.fromisoformat(deployment.started_at)
                uptime = datetime.utcnow() - started
                uptime_str = str(uptime).split(".")[0]
            except (ValueError, TypeError):
                pass

        return {
            "id": deployment.id,
            "status": deployment.status,
            "config": asdict(deployment.config),
            "started_at": deployment.started_at,
            "stopped_at": deployment.stopped_at,
            "uptime": uptime_str,
            "agent_count": len(deployment.agent_ids),
            "agent_ids": deployment.agent_ids,
            "tasks_completed": loop_status.get("completed_count", deployment.tasks_completed),
            "tasks_failed": loop_status.get("failed_count", deployment.tasks_failed),
            "tasks_queued": loop_status.get("queue_length", 0),
            "estimated_cost_usd": loop_status.get("estimated_cost_usd", deployment.estimated_cost_usd),
            "cycle_count": loop_status.get("cycle_count", 0),
            "error": deployment.error,
        }

    def get_active_deployments(self) -> List[Dict]:
        """Get all active (non-stopped) deployments."""
        with self._lock:
            active = []
            for dep in self._deployments.values():
                status = self.get_deployment_status(dep.id)
                if status:
                    active.append(status)
            return active

    def get_all_deployments(self) -> List[Dict]:
        """Get all deployments including stopped ones."""
        with self._lock:
            all_deps = []
            for dep in self._deployments.values():
                status = self.get_deployment_status(dep.id)
                if status:
                    all_deps.append(status)
            return all_deps

    def _runtime_expired(self, deployment_id: str):
        """Called when a deployment's max runtime is reached."""
        logger.info(f"Deployment {deployment_id} runtime limit reached, stopping...")
        self.stop_deployment(deployment_id)


# ---- Singleton ----

_wizard: Optional[DeploymentWizard] = None


def get_deployment_wizard() -> DeploymentWizard:
    global _wizard
    if _wizard is None:
        _wizard = DeploymentWizard()
    return _wizard
