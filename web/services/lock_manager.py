"""
Lock Manager

Provides repo-level and path-level exclusive locks for autonomous tasks.
Backed by SQLite state store for crash-proof persistence and 100+ agent scale.
"""

import os
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .state_store import get_state_store

logger = logging.getLogger(__name__)


class LockManager:
    """Manage repo and path locks with SQLite persistence."""

    def __init__(self):
        self._lock = threading.Lock()

    def acquire(
        self,
        repo: Optional[str],
        lock_paths: Optional[List[str]],
        agent_id: str,
        task_id: str
    ) -> Tuple[bool, str, List[str]]:
        """Attempt to acquire repo/path locks for a task."""
        with self._lock:
            store = get_state_store()

            repo_key = self._repo_key(repo) if repo else None
            lock_paths = lock_paths or []
            want_repo_lock = repo_key is not None and len(lock_paths) == 0

            # Repo lock: need exclusive access to entire repo
            if want_repo_lock:
                # Check for any existing locks on this repo
                existing = store.get_locks_by_repo(repo.strip().lower()) if repo else []
                for lock_info in existing:
                    if self._same_holder(lock_info, agent_id, task_id):
                        continue
                    return False, f"repo_locked_by_{lock_info.get('agent_id')}", []

                ok = store.acquire_lock(repo_key, "repo", repo.strip().lower(), None, agent_id, task_id)
                if ok:
                    return True, "", [repo_key]
                # Lock already exists from another holder
                lock_info = store.get_lock(repo_key)
                return False, f"repo_locked_by_{lock_info.get('agent_id', 'unknown')}" if lock_info else "lock_failed", []

            # Path locks: check repo-level lock first
            if repo_key:
                repo_lock = store.get_lock(repo_key)
                if repo_lock and not self._same_holder(repo_lock, agent_id, task_id):
                    return False, f"repo_locked_by_{repo_lock.get('agent_id')}", []

            # Check each path for conflicts
            lock_keys = []
            for path in lock_paths:
                key = self._path_key(repo, path)
                existing = store.get_lock(key)
                if existing and not self._same_holder(existing, agent_id, task_id):
                    return False, f"path_locked_by_{existing.get('agent_id')}", []
                lock_keys.append(key)

            # Acquire all path locks
            for key in lock_keys:
                _, repo_name, norm_path = self._parse_path_key(key)
                store.acquire_lock(
                    key, "path",
                    repo_name, norm_path,
                    agent_id, task_id,
                )

            return True, "", lock_keys

    def release(self, agent_id: str, task_id: Optional[str] = None) -> int:
        """Release all locks held by an agent (optionally by task)."""
        with self._lock:
            store = get_state_store()
            return store.release_agent_locks(agent_id, task_id)

    def release_keys(self, keys: List[str]) -> int:
        """Release specific lock keys."""
        if not keys:
            return 0
        with self._lock:
            store = get_state_store()
            return store.release_locks_by_keys(keys)

    def clear_all(self) -> int:
        """Release all locks."""
        with self._lock:
            store = get_state_store()
            return store.clear_all_locks()

    def list_locks(self) -> Dict[str, Dict]:
        with self._lock:
            store = get_state_store()
            return store.list_all_locks()

    def get_snapshot(self) -> Dict[str, object]:
        with self._lock:
            store = get_state_store()
            locks = store.list_all_locks()
            return {
                "locks": locks,
                "updated_at": datetime.utcnow().isoformat()
            }

    @staticmethod
    def _same_holder(info: Dict, agent_id: str, task_id: str) -> bool:
        return info.get("agent_id") == agent_id and info.get("task_id") == task_id

    @staticmethod
    def _repo_key(repo: Optional[str]) -> Optional[str]:
        if not repo:
            return None
        return f"repo:{repo.strip().lower()}"

    @staticmethod
    def _path_key(repo: Optional[str], path: str) -> str:
        normalized = os.path.normpath(path).replace("\\", "/").lstrip("./")
        repo_part = repo.strip().lower() if repo else ""
        return f"path:{repo_part}:{normalized}"

    @staticmethod
    def _parse_path_key(key: str) -> Tuple[str, Optional[str], Optional[str]]:
        parts = key.split(":", 2)
        if len(parts) < 3:
            return key, None, None
        return parts[0], parts[1] or None, parts[2] or None

    @staticmethod
    def _is_repo_path_key(key: str, repo: Optional[str]) -> bool:
        if not repo:
            return False
        prefix = f"path:{repo.strip().lower()}:"
        return key.startswith(prefix)


_lock_manager: Optional[LockManager] = None


def get_lock_manager() -> LockManager:
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = LockManager()
    return _lock_manager
