"""
Lock Manager

Provides repo-level and path-level exclusive locks for autonomous tasks.
Locks are persisted to disk with atomic replace to survive process restarts.
"""

import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

LOCKS_FILE = os.path.join(os.path.expanduser("~"), ".shadowai", "locks.json")


class LockManager:
    """Manage repo and path locks with file persistence."""

    def __init__(self, file_path: str = LOCKS_FILE):
        self._file_path = file_path
        self._lock = threading.Lock()
        self._locks: Dict[str, Dict] = {}
        self._updated_at: Optional[str] = None
        self._file_mtime: float = 0.0  # Track file modification time

    def acquire(
        self,
        repo: Optional[str],
        lock_paths: Optional[List[str]],
        agent_id: str,
        task_id: str
    ) -> Tuple[bool, str, List[str]]:
        """Attempt to acquire repo/path locks for a task."""
        with self._lock:
            self._load()

            repo_key = self._repo_key(repo) if repo else None
            lock_paths = lock_paths or []
            want_repo_lock = repo_key is not None and len(lock_paths) == 0

            # Repo lock conflicts: any repo or path lock for this repo held by others
            if want_repo_lock:
                for key, info in self._locks.items():
                    if self._same_holder(info, agent_id, task_id):
                        continue
                    if key == repo_key or self._is_repo_path_key(key, repo):
                        return False, f"repo_locked_by_{info.get('agent_id')}", []

                self._locks[repo_key] = self._make_lock("repo", repo, None, agent_id, task_id)
                self._save()
                return True, "", [repo_key]

            # Path lock conflicts
            if repo_key:
                repo_lock = self._locks.get(repo_key)
                if repo_lock and not self._same_holder(repo_lock, agent_id, task_id):
                    return False, f"repo_locked_by_{repo_lock.get('agent_id')}", []

            lock_keys = []
            for path in lock_paths:
                key = self._path_key(repo, path)
                existing = self._locks.get(key)
                if existing and not self._same_holder(existing, agent_id, task_id):
                    return False, f"path_locked_by_{existing.get('agent_id')}", []
                lock_keys.append(key)

            for key in lock_keys:
                _, repo_name, norm_path = self._parse_path_key(key)
                self._locks[key] = self._make_lock("path", repo_name, norm_path, agent_id, task_id)

            if lock_keys:
                self._save()
            return True, "", lock_keys

    def release(self, agent_id: str, task_id: Optional[str] = None) -> int:
        """Release all locks held by an agent (optionally by task)."""
        with self._lock:
            self._load()
            to_remove = []
            for key, info in self._locks.items():
                if info.get("agent_id") != agent_id:
                    continue
                if task_id and info.get("task_id") != task_id:
                    continue
                to_remove.append(key)

            for key in to_remove:
                self._locks.pop(key, None)

            if to_remove:
                self._save()
            return len(to_remove)

    def release_keys(self, keys: List[str]) -> int:
        """Release specific lock keys."""
        if not keys:
            return 0
        with self._lock:
            self._load()
            removed = 0
            for key in keys:
                if key in self._locks:
                    self._locks.pop(key, None)
                    removed += 1
            if removed:
                self._save()
            return removed

    def clear_all(self) -> int:
        """Release all locks."""
        with self._lock:
            self._load()
            count = len(self._locks)
            if count:
                self._locks.clear()
                self._save()
            return count

    def list_locks(self) -> Dict[str, Dict]:
        with self._lock:
            self._load()
            return dict(self._locks)

    def get_snapshot(self) -> Dict[str, object]:
        with self._lock:
            self._load()
            return {
                "locks": dict(self._locks),
                "updated_at": self._updated_at
            }

    def _load(self):
        if not os.path.exists(self._file_path):
            self._locks = {}
            self._file_mtime = 0.0
            return
        try:
            current_mtime = os.path.getmtime(self._file_path)
            if self._locks and current_mtime == self._file_mtime:
                return  # File unchanged, cache is valid
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._locks = data.get("locks", {})
                self._updated_at = data.get("updated_at")
                self._file_mtime = current_mtime
        except (OSError, json.JSONDecodeError):
            self._locks = {}
            self._updated_at = None
            self._file_mtime = 0.0

    def _save(self):
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        self._updated_at = datetime.utcnow().isoformat()
        payload = {
            "updated_at": self._updated_at,
            "locks": self._locks,
        }
        tmp_path = f"{self._file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self._file_path)

    @staticmethod
    def _make_lock(lock_type: str, repo: Optional[str], path: Optional[str],
                   agent_id: str, task_id: str) -> Dict:
        return {
            "lock_type": lock_type,
            "repo": repo,
            "path": path,
            "agent_id": agent_id,
            "task_id": task_id,
            "acquired_at": datetime.utcnow().isoformat()
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
