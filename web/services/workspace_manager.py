"""
Workspace Manager

Creates per-task workspaces using git worktrees to isolate changes.
Includes garbage collection for stale workspaces and orphaned branches.
"""

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_ROOT = os.path.join(os.path.expanduser("~"), ".shadowai", "workspaces")


@dataclass
class WorkspaceAllocation:
    success: bool
    repo: str
    workspace_root: str
    branch: Optional[str] = None
    reused: bool = False
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class WorkspaceManager:
    """Allocate per-task workspaces with git worktrees."""

    def __init__(self, workspace_root: str = DEFAULT_WORKSPACE_ROOT):
        self._workspace_root = workspace_root
        self._lock = threading.Lock()

    def allocate(self, repo: str, task_id: str, thread_id: str) -> WorkspaceAllocation:
        repo_root = self._resolve_repo_root(repo)
        if not repo_root or not os.path.isdir(repo_root):
            return WorkspaceAllocation(
                success=False,
                repo=repo,
                workspace_root="",
                error="Repo root not found"
            )

        safe_repo = self._slug(repo or "repo")
        safe_task = self._slug(task_id or "task")
        safe_thread = self._slug(thread_id or "thread")
        branch = f"swarm/task_{safe_task}/thread_{safe_thread}"
        workspace_root = os.path.join(
            self._workspace_root,
            safe_repo,
            f"task_{safe_task}",
            f"thread_{safe_thread}"
        )

        with self._lock:
            if os.path.isdir(workspace_root):
                return WorkspaceAllocation(
                    success=True,
                    repo=repo,
                    workspace_root=workspace_root,
                    branch=branch,
                    reused=True
                )

            os.makedirs(os.path.dirname(workspace_root), exist_ok=True)

            if not self._is_git_repo(repo_root):
                return WorkspaceAllocation(
                    success=False,
                    repo=repo,
                    workspace_root=workspace_root,
                    branch=branch,
                    error="Repo is not a git repository"
                )

            ok, error = self._create_worktree(repo_root, workspace_root, branch)
            if not ok:
                return WorkspaceAllocation(
                    success=False,
                    repo=repo,
                    workspace_root=workspace_root,
                    branch=branch,
                    error=error or "git worktree add failed"
                )

        return WorkspaceAllocation(
            success=True,
            repo=repo,
            workspace_root=workspace_root,
            branch=branch,
            reused=False
        )

    # ---- Internal helpers ----

    def _resolve_repo_root(self, repo: str) -> Optional[str]:
        repo_lower = (repo or "").lower()
        if "android" in repo_lower:
            return "C:/shadow/shadow-android"
        if "bridge" in repo_lower:
            return "C:/shadow/shadow-bridge"
        return "C:/shadow"

    def _is_git_repo(self, repo_root: str) -> bool:
        return os.path.isdir(os.path.join(repo_root, ".git"))

    def _create_worktree(self, repo_root: str, workspace_root: str, branch: str):
        base_cmd = ["git", "-C", repo_root, "worktree", "add"]
        create_cmd = base_cmd + [workspace_root, "-b", branch]
        returncode, output = self._run_command(create_cmd)
        if returncode == 0:
            return True, None

        # If branch exists, try to attach worktree without creating it.
        attach_cmd = base_cmd + [workspace_root, branch]
        returncode, output = self._run_command(attach_cmd)
        if returncode == 0:
            return True, None

        logger.error(f"Failed to create worktree: {output}")
        return False, output

    def _run_command(self, cmd):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode, output.strip()
        except Exception as exc:
            return 1, str(exc)

    # ---- Cleanup / Garbage Collection ----

    def cleanup_workspace(self, repo: str, task_id: str, thread_id: str) -> bool:
        """Remove a specific workspace and its git worktree/branch after task completion."""
        safe_repo = self._slug(repo or "repo")
        safe_task = self._slug(task_id or "task")
        safe_thread = self._slug(thread_id or "thread")
        branch = f"swarm/task_{safe_task}/thread_{safe_thread}"
        workspace_root = os.path.join(
            self._workspace_root, safe_repo,
            f"task_{safe_task}", f"thread_{safe_thread}"
        )

        repo_root = self._resolve_repo_root(repo)
        if not repo_root:
            return False

        with self._lock:
            # Remove git worktree
            if os.path.isdir(workspace_root):
                self._run_command(["git", "-C", repo_root, "worktree", "remove", workspace_root, "--force"])

            # Delete branch
            self._run_command(["git", "-C", repo_root, "branch", "-D", branch])

            # Clean up empty parent dirs
            task_dir = os.path.dirname(workspace_root)
            if os.path.isdir(task_dir) and not os.listdir(task_dir):
                shutil.rmtree(task_dir, ignore_errors=True)

        logger.info(f"Cleaned up workspace: {workspace_root} (branch: {branch})")
        return True

    def gc(self, max_age_hours: int = 24) -> Dict:
        """Garbage collect stale workspaces older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0
        errors = 0

        if not os.path.isdir(self._workspace_root):
            return {"removed": 0, "errors": 0}

        for repo_dir in os.listdir(self._workspace_root):
            repo_path = os.path.join(self._workspace_root, repo_dir)
            if not os.path.isdir(repo_path):
                continue

            for task_dir in os.listdir(repo_path):
                task_path = os.path.join(repo_path, task_dir)
                if not os.path.isdir(task_path):
                    continue

                for thread_dir in os.listdir(task_path):
                    ws_path = os.path.join(task_path, thread_dir)
                    if not os.path.isdir(ws_path):
                        continue

                    # Check age by directory mtime
                    try:
                        mtime = os.path.getmtime(ws_path)
                        if mtime < cutoff:
                            repo_root = self._resolve_repo_root(repo_dir)
                            if repo_root:
                                self._run_command([
                                    "git", "-C", repo_root,
                                    "worktree", "remove", ws_path, "--force"
                                ])
                            if os.path.isdir(ws_path):
                                shutil.rmtree(ws_path, ignore_errors=True)
                            removed += 1
                            logger.info(f"GC removed stale workspace: {ws_path}")
                    except Exception as e:
                        logger.warning(f"GC error on {ws_path}: {e}")
                        errors += 1

        # Prune worktree metadata
        for repo_root in ["C:/shadow/shadow-android", "C:/shadow/shadow-bridge"]:
            if os.path.isdir(repo_root):
                self._run_command(["git", "-C", repo_root, "worktree", "prune"])

        logger.info(f"Workspace GC complete: {removed} removed, {errors} errors")
        return {"removed": removed, "errors": errors}

    @staticmethod
    def _slug(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text or "")
        return cleaned.strip("_") or "default"
