"""
Per-Project Rules Engine

Validates tasks and file changes against project-specific rules.
Generates prompt addenda injected into agent system prompts so agents
respect forbidden files, coding standards, and operational limits.

Persisted in SQLite via state_store's project_rules table.

Usage:
    engine = get_rules_engine()
    ok, reason = engine.validate_task("proj-abc", task_dict)
    ok, reason = engine.validate_file_change("proj-abc", "src/db/migrate.py")
    addendum = engine.get_prompt_addendum("proj-abc")
"""

import os
import json
import hashlib
import fnmatch
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from web.services.state_store import get_state_store

logger = logging.getLogger(__name__)

# Default forbidden operations (always dangerous for autonomous agents)
DEFAULT_FORBIDDEN_OPS = [
    "database_migration",
    "dependency_add",
    "dependency_remove",
    "build_config_change",
    "signing_config_change",
    "manifest_permission_change",
    "destructive_data_operation",
]

# Default forbidden file patterns
DEFAULT_FORBIDDEN_FILES = [
    "*.env",
    "*.env.*",
    "**/credentials*",
    "**/secrets*",
    "**/.git/**",
    "**/signing*",
    "**/keystore*",
    "**/*.jks",
    "**/*.p12",
]


@dataclass
class ProjectRules:
    """Rules configuration for a project."""
    project_id: str
    project_path: str
    display_name: str = ""
    forbidden_files: List[str] = field(default_factory=list)
    allowed_files: List[str] = field(default_factory=list)
    forbidden_operations: List[str] = field(default_factory=list)
    coding_standards: str = ""
    max_file_changes_per_task: int = 0  # 0 = unlimited
    require_tests: bool = False
    custom_prompt_addendum: str = ""

    def to_store_kwargs(self) -> Dict:
        """Convert to kwargs for state_store.upsert_project_rules()."""
        return {
            "display_name": self.display_name,
            "forbidden_files_json": json.dumps(self.forbidden_files),
            "allowed_files_json": json.dumps(self.allowed_files),
            "forbidden_operations_json": json.dumps(self.forbidden_operations),
            "coding_standards": self.coding_standards,
            "max_file_changes_per_task": self.max_file_changes_per_task,
            "require_tests": 1 if self.require_tests else 0,
            "custom_prompt_addendum": self.custom_prompt_addendum,
        }


def _project_id_from_path(project_path: str) -> str:
    """Generate a stable project ID from a path."""
    normalized = os.path.normpath(project_path).lower().replace("\\", "/")
    return "proj-" + hashlib.sha256(normalized.encode()).hexdigest()[:12]


class RulesEngine:
    """Validates agent actions against per-project rules."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[str, ProjectRules] = {}

    def get_or_create_rules(self, project_path: str, display_name: str = "") -> ProjectRules:
        """Get rules for a project, creating defaults if none exist."""
        project_id = _project_id_from_path(project_path)

        with self._lock:
            if project_id in self._cache:
                return self._cache[project_id]

        store = get_state_store()
        row = store.get_project_rules(project_id)

        if row:
            rules = ProjectRules(
                project_id=project_id,
                project_path=row["project_path"],
                display_name=row.get("display_name", ""),
                forbidden_files=row.get("forbidden_files", []),
                allowed_files=row.get("allowed_files", []),
                forbidden_operations=row.get("forbidden_operations", []),
                coding_standards=row.get("coding_standards", ""),
                max_file_changes_per_task=row.get("max_file_changes_per_task", 0),
                require_tests=bool(row.get("require_tests", 0)),
                custom_prompt_addendum=row.get("custom_prompt_addendum", ""),
            )
        else:
            # Create with sensible defaults
            rules = ProjectRules(
                project_id=project_id,
                project_path=project_path,
                display_name=display_name or os.path.basename(project_path),
                forbidden_files=list(DEFAULT_FORBIDDEN_FILES),
                forbidden_operations=list(DEFAULT_FORBIDDEN_OPS),
            )
            store.upsert_project_rules(
                project_id, project_path, **rules.to_store_kwargs()
            )

        with self._lock:
            self._cache[project_id] = rules
        return rules

    def save_rules(self, rules: ProjectRules) -> bool:
        """Save updated rules to the store."""
        store = get_state_store()
        store.upsert_project_rules(
            rules.project_id, rules.project_path, **rules.to_store_kwargs()
        )
        with self._lock:
            self._cache[rules.project_id] = rules
        logger.info(f"Rules saved for {rules.display_name} ({rules.project_id})")
        return True

    def delete_rules(self, project_path: str) -> bool:
        """Delete rules for a project."""
        project_id = _project_id_from_path(project_path)
        store = get_state_store()
        store.delete_project_rules(project_id)
        with self._lock:
            self._cache.pop(project_id, None)
        return True

    def get_all_rules(self) -> List[ProjectRules]:
        """Get all configured project rules."""
        store = get_state_store()
        rows = store.get_all_project_rules()
        result = []
        for row in rows:
            rules = ProjectRules(
                project_id=row["project_id"],
                project_path=row["project_path"],
                display_name=row.get("display_name", ""),
                forbidden_files=row.get("forbidden_files", []),
                allowed_files=row.get("allowed_files", []),
                forbidden_operations=row.get("forbidden_operations", []),
                coding_standards=row.get("coding_standards", ""),
                max_file_changes_per_task=row.get("max_file_changes_per_task", 0),
                require_tests=bool(row.get("require_tests", 0)),
                custom_prompt_addendum=row.get("custom_prompt_addendum", ""),
            )
            result.append(rules)
        return result

    # ---- Validation ----

    def validate_task(self, project_path: str, task: Dict) -> Tuple[bool, str]:
        """
        Validate whether a task is allowed under project rules.

        Returns (True, "") if OK, or (False, reason) if blocked.
        """
        rules = self.get_or_create_rules(project_path)

        # Check forbidden operations
        task_category = (task.get("category") or "").lower()
        task_title = (task.get("title") or "").lower()
        task_desc = (task.get("description") or "").lower()

        for op in rules.forbidden_operations:
            op_lower = op.lower().replace("_", " ")
            # Check if the forbidden operation appears in task metadata
            if op_lower in task_category or op_lower in task_title or op_lower in task_desc:
                return False, f"Forbidden operation '{op}' detected in task"

            # Specific keyword matching for common operations
            op_keywords = {
                "database_migration": ["migration", "migrate", "alter table", "schema change"],
                "dependency_add": ["add dependency", "new dependency", "install package", "npm install", "pip install"],
                "dependency_remove": ["remove dependency", "uninstall"],
                "build_config_change": ["build.gradle", "build config", "signing config"],
                "signing_config_change": ["keystore", "signing", "jks"],
                "manifest_permission_change": ["manifest", "permission", "uses-permission"],
                "destructive_data_operation": ["drop table", "truncate", "delete all", "clear data", "uninstall"],
            }
            for keyword in op_keywords.get(op, []):
                if keyword in task_title or keyword in task_desc:
                    return False, f"Forbidden operation '{op}' matches keyword '{keyword}'"

        # Check forbidden files in task target
        task_file = task.get("file_path") or task.get("file") or ""
        if task_file:
            ok, reason = self.validate_file_change(project_path, task_file)
            if not ok:
                return False, reason

        return True, ""

    def validate_file_change(self, project_path: str, file_path: str) -> Tuple[bool, str]:
        """
        Validate whether an agent is allowed to modify a specific file.

        Returns (True, "") if OK, or (False, reason) if blocked.
        """
        rules = self.get_or_create_rules(project_path)

        # Normalize path for matching
        rel_path = file_path
        if os.path.isabs(file_path):
            try:
                rel_path = os.path.relpath(file_path, project_path)
            except ValueError:
                rel_path = file_path
        rel_path = rel_path.replace("\\", "/")

        # Check whitelist mode first (if allowed_files is set, ONLY those files are allowed)
        if rules.allowed_files:
            allowed = False
            for pattern in rules.allowed_files:
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(rel_path), pattern):
                    allowed = True
                    break
            if not allowed:
                return False, f"File '{rel_path}' not in allowed files whitelist"

        # Check forbidden files (always checked, even with whitelist)
        all_forbidden = list(DEFAULT_FORBIDDEN_FILES) + rules.forbidden_files
        for pattern in all_forbidden:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(rel_path), pattern):
                return False, f"File '{rel_path}' matches forbidden pattern '{pattern}'"

        return True, ""

    def get_prompt_addendum(self, project_path: str) -> str:
        """
        Generate a rules addendum to inject into agent system prompts.

        This text is appended to the agent's task prompt so it respects
        project-specific rules during execution.
        """
        rules = self.get_or_create_rules(project_path)
        parts = []

        parts.append("PROJECT RULES (MANDATORY - violations will cause task rejection):")

        if rules.forbidden_files:
            parts.append("\nFORBIDDEN FILES (do NOT modify these):")
            for p in rules.forbidden_files:
                parts.append(f"  - {p}")

        if rules.allowed_files:
            parts.append("\nALLOWED FILES ONLY (only modify files matching these patterns):")
            for p in rules.allowed_files:
                parts.append(f"  - {p}")

        if rules.forbidden_operations:
            parts.append("\nFORBIDDEN OPERATIONS (do NOT perform these):")
            for op in rules.forbidden_operations:
                parts.append(f"  - {op.replace('_', ' ').title()}")

        if rules.coding_standards:
            parts.append(f"\nCODING STANDARDS:\n{rules.coding_standards}")

        if rules.max_file_changes_per_task > 0:
            parts.append(f"\nMAX FILE CHANGES: {rules.max_file_changes_per_task} files per task")

        if rules.require_tests:
            parts.append("\nTESTS REQUIRED: You MUST run tests before marking the task complete.")

        if rules.custom_prompt_addendum:
            parts.append(f"\nADDITIONAL RULES:\n{rules.custom_prompt_addendum}")

        return "\n".join(parts)

    def get_rules_summary(self, project_path: str) -> Dict:
        """Get a summary of rules for API/UI display."""
        rules = self.get_or_create_rules(project_path)
        return {
            "project_id": rules.project_id,
            "project_path": rules.project_path,
            "display_name": rules.display_name,
            "forbidden_files_count": len(rules.forbidden_files),
            "allowed_files_count": len(rules.allowed_files),
            "forbidden_operations_count": len(rules.forbidden_operations),
            "has_coding_standards": bool(rules.coding_standards),
            "max_file_changes_per_task": rules.max_file_changes_per_task,
            "require_tests": rules.require_tests,
            "has_custom_addendum": bool(rules.custom_prompt_addendum),
        }


# ---- Singleton ----

_engine: Optional[RulesEngine] = None
_engine_lock = threading.Lock()


def get_rules_engine() -> RulesEngine:
    """Get or create the global RulesEngine singleton."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RulesEngine()
    return _engine
