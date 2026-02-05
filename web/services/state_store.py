"""
Agent State Store

SQLite-backed persistent state for agent infrastructure.
Replaces JSON file persistence (locks.json, agents.json, etc.) with a single
WAL-mode SQLite database that handles 100+ concurrent agent writers.

Usage:
    store = get_state_store()
    store.upsert_lock(...)
    store.append_log(...)
    store.upsert_task(...)
"""

import os
import sqlite3
import threading
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.expanduser("~"), ".shadowai")
DB_PATH = os.path.join(DB_DIR, "agent_state.db")

# Max log lines per agent before trimming
MAX_LOG_LINES = int(os.environ.get("AGENT_MAX_LOG_LINES", "1000"))


class AgentStateStore:
    """Thread-safe SQLite state store for agent infrastructure.

    Uses WAL mode for concurrent readers + writer, busy_timeout for contention,
    and a per-thread connection pool via thread-local storage.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._create_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection (one per thread)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _create_tables(self):
        """Create all tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            -- File/repo locks (replaces locks.json)
            CREATE TABLE IF NOT EXISTS file_locks (
                lock_key TEXT PRIMARY KEY,
                lock_type TEXT NOT NULL,        -- 'repo' or 'path'
                repo TEXT,
                path TEXT,
                agent_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                acquired_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_locks_agent ON file_locks(agent_id);
            CREATE INDEX IF NOT EXISTS idx_locks_repo ON file_locks(repo);

            -- Agent output logs (ring buffer per agent, replaces in-memory agent_logs)
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                stream TEXT NOT NULL,            -- 'stdout' or 'stderr'
                line TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_logs_agent ON agent_logs(agent_id);

            -- Task state tracking (replaces in-memory agent_task_states)
            CREATE TABLE IF NOT EXISTS agent_task_state (
                agent_id TEXT PRIMARY KEY,
                task_id TEXT,
                thread_id TEXT,
                task_display TEXT,               -- human-readable task description
                status TEXT,                     -- running, done, timeout
                started_at TEXT,
                done_at TEXT,
                done_status TEXT,                -- success, fail
                done_reason TEXT,
                elapsed_ms INTEGER,
                exit_code INTEGER,
                progress_pct REAL,
                progress_msg TEXT,
                result_json TEXT,                -- JSON blob of TASK_RESULT payload
                workspace_root TEXT,
                repo TEXT,
                lock_keys_json TEXT,             -- JSON array of lock keys
                last_marker_at TEXT,
                tail_logs_json TEXT              -- JSON array of tail log lines
            );

            -- Task block events (replaces in-memory agent_task_blocks)
            CREATE TABLE IF NOT EXISTS task_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                task_id TEXT,
                thread_id TEXT,
                repo TEXT,
                lock_paths_json TEXT,
                reason TEXT,
                timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_blocks_time ON task_blocks(timestamp);

            -- Persistent task queue (replaces in-memory task_queue list)
            CREATE TABLE IF NOT EXISTS task_queue (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT,
                priority INTEGER DEFAULT 3,
                status TEXT DEFAULT 'pending',
                repo TEXT,
                file_path TEXT,
                assigned_to TEXT,
                workspace_root TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                failed_at TEXT,
                assigned_at TEXT,
                dedup_hash TEXT UNIQUE,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tq_status ON task_queue(status);
            CREATE INDEX IF NOT EXISTS idx_tq_repo ON task_queue(repo);
            CREATE INDEX IF NOT EXISTS idx_tq_assigned ON task_queue(assigned_to);
            CREATE INDEX IF NOT EXISTS idx_tq_priority ON task_queue(priority, created_at);

            -- Agent performance history (new - for smart routing in Phase 3)
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                task_id TEXT,
                task_category TEXT,
                success INTEGER NOT NULL,        -- 0 or 1
                duration_ms INTEGER,
                files_changed INTEGER,
                estimated_cost REAL,
                repo TEXT,
                completed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_perf_agent ON agent_performance(agent_id);
            CREATE INDEX IF NOT EXISTS idx_perf_category ON agent_performance(task_category);

            -- Per-project rules (forbidden files, allowed ops, coding standards)
            CREATE TABLE IF NOT EXISTS project_rules (
                project_id TEXT PRIMARY KEY,
                project_path TEXT NOT NULL,
                display_name TEXT,
                forbidden_files_json TEXT,       -- JSON array of glob patterns
                allowed_files_json TEXT,         -- JSON array of glob patterns (whitelist mode)
                forbidden_operations_json TEXT,  -- JSON array of operation names
                coding_standards TEXT,           -- injected into agent system prompts
                max_file_changes_per_task INTEGER DEFAULT 0,  -- 0 = unlimited
                require_tests INTEGER DEFAULT 0,              -- 0 or 1
                custom_prompt_addendum TEXT,     -- extra text for agent prompts
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.commit()
        logger.info(f"State store initialized at {self._db_path}")

    # ---- File Locks (replaces locks.json) ----

    def acquire_lock(
        self,
        lock_key: str,
        lock_type: str,
        repo: Optional[str],
        path: Optional[str],
        agent_id: str,
        task_id: str,
    ) -> bool:
        """Attempt to acquire a lock. Returns True if acquired."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        try:
            conn.execute(
                """INSERT INTO file_locks (lock_key, lock_type, repo, path, agent_id, task_id, acquired_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (lock_key, lock_type, repo, path, agent_id, task_id, now),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Lock key already exists - check if same holder
            row = conn.execute(
                "SELECT agent_id, task_id FROM file_locks WHERE lock_key = ?",
                (lock_key,),
            ).fetchone()
            if row and row["agent_id"] == agent_id and row["task_id"] == task_id:
                return True  # Already held by this agent+task
            return False

    def get_lock(self, lock_key: str) -> Optional[Dict]:
        """Get lock info for a key."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM file_locks WHERE lock_key = ?", (lock_key,)
        ).fetchone()
        return dict(row) if row else None

    def get_locks_by_repo(self, repo: str) -> List[Dict]:
        """Get all locks for a repo (both repo-level and path-level)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM file_locks WHERE repo = ?", (repo.strip().lower(),)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_locks_by_agent(self, agent_id: str) -> List[Dict]:
        """Get all locks held by an agent."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM file_locks WHERE agent_id = ?", (agent_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def release_lock(self, lock_key: str) -> bool:
        """Release a specific lock."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM file_locks WHERE lock_key = ?", (lock_key,))
        conn.commit()
        return cursor.rowcount > 0

    def release_agent_locks(
        self, agent_id: str, task_id: Optional[str] = None
    ) -> int:
        """Release all locks held by an agent, optionally filtered by task."""
        conn = self._get_conn()
        if task_id:
            cursor = conn.execute(
                "DELETE FROM file_locks WHERE agent_id = ? AND task_id = ?",
                (agent_id, task_id),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM file_locks WHERE agent_id = ?", (agent_id,)
            )
        conn.commit()
        return cursor.rowcount

    def release_locks_by_keys(self, keys: List[str]) -> int:
        """Release specific lock keys."""
        if not keys:
            return 0
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in keys)
        cursor = conn.execute(
            f"DELETE FROM file_locks WHERE lock_key IN ({placeholders})", keys
        )
        conn.commit()
        return cursor.rowcount

    def clear_all_locks(self) -> int:
        """Release all locks."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM file_locks")
        conn.commit()
        return cursor.rowcount

    def list_all_locks(self) -> Dict[str, Dict]:
        """List all locks as a dict keyed by lock_key."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM file_locks").fetchall()
        return {r["lock_key"]: dict(r) for r in rows}

    # ---- Agent Logs (ring buffer, replaces in-memory agent_logs dict) ----

    def append_log(self, agent_id: str, timestamp: str, stream: str, line: str):
        """Append a log line for an agent."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO agent_logs (agent_id, timestamp, stream, line) VALUES (?, ?, ?, ?)",
            (agent_id, timestamp, stream, line),
        )
        conn.commit()

    def append_logs_batch(self, entries: List[Tuple[str, str, str, str]]):
        """Append multiple log lines efficiently. Each entry: (agent_id, timestamp, stream, line)."""
        if not entries:
            return
        conn = self._get_conn()
        conn.executemany(
            "INSERT INTO agent_logs (agent_id, timestamp, stream, line) VALUES (?, ?, ?, ?)",
            entries,
        )
        conn.commit()

    def get_logs(self, agent_id: str, limit: int = 100) -> List[Dict]:
        """Get recent log lines for an agent."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT timestamp, stream, line FROM agent_logs WHERE agent_id = ? ORDER BY id DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def trim_logs(self, agent_id: str, keep: int = None):
        """Trim old log lines for an agent, keeping the most recent `keep` lines."""
        keep = keep or MAX_LOG_LINES
        conn = self._get_conn()
        conn.execute(
            """DELETE FROM agent_logs WHERE agent_id = ? AND id NOT IN (
                SELECT id FROM agent_logs WHERE agent_id = ? ORDER BY id DESC LIMIT ?
            )""",
            (agent_id, agent_id, keep),
        )
        conn.commit()

    def trim_all_logs(self, keep: int = None):
        """Trim logs for all agents."""
        keep = keep or MAX_LOG_LINES
        conn = self._get_conn()
        # Get all agent IDs that have logs
        agents = conn.execute(
            "SELECT DISTINCT agent_id FROM agent_logs"
        ).fetchall()
        for row in agents:
            self.trim_logs(row["agent_id"], keep)

    def clear_agent_logs(self, agent_id: str) -> int:
        """Delete all logs for an agent."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM agent_logs WHERE agent_id = ?", (agent_id,))
        conn.commit()
        return cursor.rowcount

    # ---- Task State (replaces in-memory agent_task_states) ----

    def upsert_task_state(self, agent_id: str, **kwargs):
        """Insert or update task state for an agent.

        Accepts any combination of task state fields as keyword args.
        """
        conn = self._get_conn()

        # Check if exists
        existing = conn.execute(
            "SELECT agent_id FROM agent_task_state WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()

        # Filter to valid columns
        valid_cols = {
            "task_id", "thread_id", "task_display", "status", "started_at",
            "done_at", "done_status", "done_reason", "elapsed_ms", "exit_code",
            "progress_pct", "progress_msg", "result_json", "workspace_root",
            "repo", "lock_keys_json", "last_marker_at", "tail_logs_json",
        }
        filtered = {k: v for k, v in kwargs.items() if k in valid_cols}
        if not filtered:
            return

        if existing:
            set_clause = ", ".join(f"{k} = ?" for k in filtered)
            values = list(filtered.values()) + [agent_id]
            conn.execute(
                f"UPDATE agent_task_state SET {set_clause} WHERE agent_id = ?",
                values,
            )
        else:
            filtered["agent_id"] = agent_id
            cols = ", ".join(filtered.keys())
            placeholders = ", ".join("?" for _ in filtered)
            conn.execute(
                f"INSERT INTO agent_task_state ({cols}) VALUES ({placeholders})",
                list(filtered.values()),
            )
        conn.commit()

    def get_task_state(self, agent_id: str) -> Optional[Dict]:
        """Get current task state for an agent."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM agent_task_state WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        return dict(row) if row else None

    def clear_task_state(self, agent_id: str):
        """Clear task state for an agent (after task completion cleanup)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM agent_task_state WHERE agent_id = ?", (agent_id,))
        conn.commit()

    def get_all_task_states(self) -> Dict[str, Dict]:
        """Get all task states indexed by agent_id."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM agent_task_state").fetchall()
        return {r["agent_id"]: dict(r) for r in rows}

    # ---- Task Blocks (replaces in-memory agent_task_blocks) ----

    def record_task_block(
        self,
        agent_id: str,
        task_id: str,
        thread_id: str,
        repo: Optional[str],
        lock_paths: List[str],
        reason: str,
    ):
        """Record a task block event."""
        import json

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO task_blocks (agent_id, task_id, thread_id, repo, lock_paths_json, reason, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (agent_id, task_id, thread_id, repo, json.dumps(lock_paths), reason,
             datetime.utcnow().isoformat()),
        )
        conn.commit()

        # Keep only last 200 block events
        conn.execute(
            """DELETE FROM task_blocks WHERE id NOT IN (
                SELECT id FROM task_blocks ORDER BY id DESC LIMIT 200
            )"""
        )
        conn.commit()

    def get_task_blocks(self, limit: int = 50) -> List[Dict]:
        """Get recent task block events."""
        import json

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM task_blocks ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in reversed(rows):
            d = dict(r)
            try:
                d["lock_paths"] = json.loads(d.pop("lock_paths_json", "[]"))
            except (json.JSONDecodeError, TypeError):
                d["lock_paths"] = []
            result.append(d)
        return result

    # ---- Agent Performance (new - for future smart routing) ----

    def record_performance(
        self,
        agent_id: str,
        task_id: str,
        task_category: str,
        success: bool,
        duration_ms: int,
        files_changed: int = 0,
        estimated_cost: float = 0.0,
        repo: str = "",
    ):
        """Record task performance for an agent."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO agent_performance
               (agent_id, task_id, task_category, success, duration_ms,
                files_changed, estimated_cost, repo, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (agent_id, task_id, task_category, 1 if success else 0,
             duration_ms, files_changed, estimated_cost, repo,
             datetime.utcnow().isoformat()),
        )
        conn.commit()

    def get_agent_performance(self, agent_id: str, limit: int = 50) -> List[Dict]:
        """Get performance history for an agent."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM agent_performance WHERE agent_id = ? ORDER BY id DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_agent_success_rate(self, agent_id: str, category: Optional[str] = None) -> float:
        """Get success rate for an agent, optionally filtered by category."""
        conn = self._get_conn()
        if category:
            row = conn.execute(
                """SELECT COUNT(*) as total, SUM(success) as wins
                   FROM agent_performance WHERE agent_id = ? AND task_category = ?""",
                (agent_id, category),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(success) as wins FROM agent_performance WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        if not row or not row["total"]:
            return 0.0
        return (row["wins"] or 0) / row["total"]

    def get_best_agents_for_category(
        self, category: str, limit: int = 10
    ) -> List[Dict]:
        """
        Rank agents by success rate for a specific task category.
        Weights recent performance higher (tasks in last 24h count 2x).
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT agent_id,
                      COUNT(*) as total,
                      SUM(success) as wins,
                      SUM(CASE WHEN completed_at >= datetime('now', '-1 day')
                          THEN success ELSE 0 END) as recent_wins,
                      SUM(CASE WHEN completed_at >= datetime('now', '-1 day')
                          THEN 1 ELSE 0 END) as recent_total,
                      AVG(duration_ms) as avg_duration_ms
               FROM agent_performance
               WHERE task_category = ?
               GROUP BY agent_id
               ORDER BY total DESC""",
            (category,),
        ).fetchall()

        results = []
        for row in rows:
            total = row["total"] or 0
            wins = row["wins"] or 0
            recent_total = row["recent_total"] or 0
            recent_wins = row["recent_wins"] or 0

            overall_rate = wins / max(total, 1)
            recent_rate = recent_wins / max(recent_total, 1) if recent_total > 0 else overall_rate
            weighted_score = overall_rate * 0.6 + recent_rate * 0.4

            results.append({
                "agent_id": row["agent_id"],
                "success_rate": round(overall_rate, 3),
                "recent_success_rate": round(recent_rate, 3),
                "total_tasks": total,
                "recent_tasks": recent_total,
                "weighted_score": round(weighted_score, 3),
                "avg_duration_ms": round(row["avg_duration_ms"] or 0),
            })

        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return results[:limit]

    # ---- Persistent Task Queue (replaces in-memory task_queue list) ----

    def enqueue_task(
        self,
        task_id: str,
        title: str,
        description: str = "",
        category: str = "",
        priority: int = 3,
        repo: str = "",
        file_path: str = "",
        dedup_hash: str = "",
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Add a task to the persistent queue. Returns False if duplicate."""
        import json

        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        try:
            conn.execute(
                """INSERT INTO task_queue
                   (id, title, description, category, priority, status, repo,
                    file_path, created_at, dedup_hash, metadata_json)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                (task_id, title, description, category, priority, repo,
                 file_path, now, dedup_hash or None,
                 json.dumps(metadata) if metadata else None),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Duplicate dedup_hash

    def enqueue_tasks_batch(self, tasks: List[Dict]) -> int:
        """Bulk-insert tasks. Returns count of successfully added tasks."""
        import json
        import hashlib

        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        added = 0
        for t in tasks:
            task_id = t.get("id", "")
            title = t.get("title", "")
            repo = t.get("repo", "")
            dedup_hash = t.get("dedup_hash") or hashlib.sha256(
                f"{repo}:{title}".encode()
            ).hexdigest()[:16]
            try:
                conn.execute(
                    """INSERT INTO task_queue
                       (id, title, description, category, priority, status, repo,
                        file_path, created_at, dedup_hash, metadata_json)
                       VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                    (task_id, title, t.get("description", ""),
                     t.get("category", ""), t.get("priority", 3),
                     repo, t.get("file_path", ""), now, dedup_hash,
                     json.dumps(t.get("metadata")) if t.get("metadata") else None),
                )
                added += 1
            except sqlite3.IntegrityError:
                continue  # Skip duplicates
        conn.commit()
        return added

    def get_pending_tasks(
        self,
        repo: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get pending tasks ordered by priority then creation time."""
        conn = self._get_conn()
        query = "SELECT * FROM task_queue WHERE status = 'pending'"
        params: list = []
        if repo:
            query += " AND repo = ?"
            params.append(repo)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY priority ASC, created_at ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [self._task_row_to_dict(r) for r in rows]

    def get_assigned_tasks(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get tasks currently assigned to agents."""
        conn = self._get_conn()
        if agent_id:
            rows = conn.execute(
                "SELECT * FROM task_queue WHERE status = 'assigned' AND assigned_to = ?",
                (agent_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM task_queue WHERE status = 'assigned'"
            ).fetchall()
        return [self._task_row_to_dict(r) for r in rows]

    def get_completed_tasks_db(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get completed tasks."""
        conn = self._get_conn()
        if repo:
            rows = conn.execute(
                "SELECT * FROM task_queue WHERE status = 'completed' AND repo = ? ORDER BY completed_at DESC LIMIT ?",
                (repo, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM task_queue WHERE status = 'completed' ORDER BY completed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._task_row_to_dict(r) for r in rows]

    def get_failed_tasks_db(self, limit: int = 100) -> List[Dict]:
        """Get failed tasks."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM task_queue WHERE status = 'failed' ORDER BY failed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._task_row_to_dict(r) for r in rows]

    def assign_task_db(self, task_id: str, agent_id: str, workspace_root: str = "") -> bool:
        """Assign a pending task to an agent. Returns False if task not pending."""
        conn = self._get_conn()
        cursor = conn.execute(
            """UPDATE task_queue SET status = 'assigned', assigned_to = ?,
               assigned_at = ?, workspace_root = ?
               WHERE id = ? AND status = 'pending'""",
            (agent_id, datetime.utcnow().isoformat(), workspace_root, task_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def complete_task_db(self, task_id: str) -> bool:
        """Mark a task as completed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE task_queue SET status = 'completed', completed_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), task_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def fail_task_db(self, task_id: str) -> bool:
        """Mark a task as failed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE task_queue SET status = 'failed', failed_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), task_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def unassign_task_db(self, task_id: str) -> bool:
        """Return an assigned task to pending (e.g., agent crashed)."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE task_queue SET status = 'pending', assigned_to = NULL, assigned_at = NULL WHERE id = ?",
            (task_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_task_db(self, task_id: str) -> bool:
        """Permanently remove a task from the queue."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM task_queue WHERE id = ?", (task_id,))
        conn.commit()
        return cursor.rowcount > 0

    def get_task_queue_stats(self) -> Dict:
        """Get task queue statistics."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'assigned' THEN 1 ELSE 0 END) as assigned,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM task_queue
        """).fetchone()
        return {
            "total": row["total"] or 0,
            "pending": row["pending"] or 0,
            "assigned": row["assigned"] or 0,
            "completed": row["completed"] or 0,
            "failed": row["failed"] or 0,
        }

    def unassign_agent_tasks(self, agent_id: str) -> int:
        """Return all tasks assigned to an agent back to pending (e.g., agent crashed)."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE task_queue SET status = 'pending', assigned_to = NULL, assigned_at = NULL WHERE assigned_to = ? AND status = 'assigned'",
            (agent_id,),
        )
        conn.commit()
        return cursor.rowcount

    @staticmethod
    def _task_row_to_dict(row) -> Dict:
        """Convert a task_queue row to a dict, parsing JSON fields."""
        import json

        d = dict(row)
        meta = d.pop("metadata_json", None)
        if meta:
            try:
                d["metadata"] = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        else:
            d["metadata"] = {}
        return d

    # ---- Project Rules ----

    def upsert_project_rules(self, project_id: str, project_path: str, **kwargs) -> bool:
        """Create or update project rules."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()

        existing = conn.execute(
            "SELECT project_id FROM project_rules WHERE project_id = ?",
            (project_id,),
        ).fetchone()

        if existing:
            sets = ["updated_at = ?"]
            vals = [now]
            for col in (
                "display_name", "forbidden_files_json", "allowed_files_json",
                "forbidden_operations_json", "coding_standards",
                "max_file_changes_per_task", "require_tests",
                "custom_prompt_addendum",
            ):
                if col in kwargs:
                    sets.append(f"{col} = ?")
                    vals.append(kwargs[col])
            vals.append(project_id)
            conn.execute(
                f"UPDATE project_rules SET {', '.join(sets)} WHERE project_id = ?",
                vals,
            )
        else:
            conn.execute(
                """INSERT INTO project_rules
                   (project_id, project_path, display_name,
                    forbidden_files_json, allowed_files_json,
                    forbidden_operations_json, coding_standards,
                    max_file_changes_per_task, require_tests,
                    custom_prompt_addendum, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id, project_path,
                    kwargs.get("display_name", ""),
                    kwargs.get("forbidden_files_json", "[]"),
                    kwargs.get("allowed_files_json", "[]"),
                    kwargs.get("forbidden_operations_json", "[]"),
                    kwargs.get("coding_standards", ""),
                    kwargs.get("max_file_changes_per_task", 0),
                    kwargs.get("require_tests", 0),
                    kwargs.get("custom_prompt_addendum", ""),
                    now, now,
                ),
            )
        conn.commit()
        return True

    def get_project_rules(self, project_id: str) -> Optional[Dict]:
        """Get rules for a specific project."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM project_rules WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if not row:
            return None
        return self._rules_row_to_dict(dict(row))

    def get_all_project_rules(self) -> List[Dict]:
        """Get all project rules."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM project_rules ORDER BY display_name").fetchall()
        return [self._rules_row_to_dict(dict(r)) for r in rows]

    def delete_project_rules(self, project_id: str) -> bool:
        """Delete rules for a project."""
        conn = self._get_conn()
        conn.execute("DELETE FROM project_rules WHERE project_id = ?", (project_id,))
        conn.commit()
        return True

    def _rules_row_to_dict(self, d: Dict) -> Dict:
        """Convert a project_rules row to a dict with parsed JSON fields."""
        import json as _json
        for field in ("forbidden_files_json", "allowed_files_json", "forbidden_operations_json"):
            raw = d.pop(field, "[]")
            key = field.replace("_json", "")
            try:
                d[key] = _json.loads(raw) if raw else []
            except (_json.JSONDecodeError, TypeError):
                d[key] = []
        return d

    # ---- Utility ----

    def close(self):
        """Close the thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    def vacuum(self):
        """Run VACUUM to reclaim space. Call during maintenance windows."""
        conn = self._get_conn()
        conn.execute("VACUUM")

    def get_stats(self) -> Dict:
        """Get store statistics."""
        conn = self._get_conn()
        locks = conn.execute("SELECT COUNT(*) as c FROM file_locks").fetchone()["c"]
        logs = conn.execute("SELECT COUNT(*) as c FROM agent_logs").fetchone()["c"]
        states = conn.execute("SELECT COUNT(*) as c FROM agent_task_state").fetchone()["c"]
        blocks = conn.execute("SELECT COUNT(*) as c FROM task_blocks").fetchone()["c"]
        perf = conn.execute("SELECT COUNT(*) as c FROM agent_performance").fetchone()["c"]
        tq = self.get_task_queue_stats()

        # DB file size
        try:
            size_mb = os.path.getsize(self._db_path) / (1024 * 1024)
        except OSError:
            size_mb = 0

        return {
            "db_path": self._db_path,
            "db_size_mb": round(size_mb, 2),
            "locks": locks,
            "log_lines": logs,
            "task_states": states,
            "task_blocks": blocks,
            "performance_records": perf,
            "task_queue": tq,
        }


# ---- Singleton ----

_state_store: Optional[AgentStateStore] = None
_store_lock = threading.Lock()


def get_state_store() -> AgentStateStore:
    """Get or create the global AgentStateStore singleton."""
    global _state_store
    if _state_store is None:
        with _store_lock:
            if _state_store is None:
                _state_store = AgentStateStore()
    return _state_store
