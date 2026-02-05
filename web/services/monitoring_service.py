"""
Monitoring Service

Collects system metrics, agent health data, and provides alerting
for the monitoring dashboard. Designed for 100+ agent scale.

Usage:
    monitor = get_monitoring_service()
    snapshot = monitor.get_snapshot()
    alerts = monitor.get_active_alerts()
"""

import os
import time
import logging
import threading
import psutil
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

from .state_store import get_state_store

logger = logging.getLogger(__name__)

# Collection interval
METRICS_INTERVAL = int(os.environ.get("MONITORING_INTERVAL", "10"))  # seconds
METRICS_HISTORY_SIZE = int(os.environ.get("MONITORING_HISTORY_SIZE", "360"))  # 1 hour at 10s

# Alert thresholds (all env-configurable)
ALERT_CPU_WARN = float(os.environ.get("ALERT_CPU_WARN", "80"))
ALERT_CPU_CRIT = float(os.environ.get("ALERT_CPU_CRIT", "95"))
ALERT_RAM_WARN = float(os.environ.get("ALERT_RAM_WARN", "80"))
ALERT_RAM_CRIT = float(os.environ.get("ALERT_RAM_CRIT", "92"))
ALERT_DISK_WARN_GB = float(os.environ.get("ALERT_DISK_WARN_GB", "5"))
ALERT_DISK_CRIT_GB = float(os.environ.get("ALERT_DISK_CRIT_GB", "1"))
ALERT_FAILURE_RATE = float(os.environ.get("ALERT_FAILURE_RATE", "0.2"))  # 20%
ALERT_STALE_TASK_MINUTES = int(os.environ.get("ALERT_STALE_TASK_MIN", "30"))

# Alert levels
LEVEL_INFO = "info"
LEVEL_WARN = "warning"
LEVEL_CRIT = "critical"


class Alert:
    """A monitoring alert."""

    __slots__ = ("id", "level", "category", "message", "timestamp", "resolved", "resolved_at")

    def __init__(self, level: str, category: str, message: str):
        self.id = f"{category}_{int(time.time() * 1000)}"
        self.level = level
        self.category = category
        self.message = message
        self.timestamp = datetime.utcnow().isoformat()
        self.resolved = False
        self.resolved_at = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }

    def resolve(self):
        self.resolved = True
        self.resolved_at = datetime.utcnow().isoformat()


class MonitoringService:
    """Collects metrics, tracks health, and generates alerts."""

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics_history: deque = deque(maxlen=METRICS_HISTORY_SIZE)
        self._alerts: List[Alert] = []
        self._active_alert_keys: Dict[str, Alert] = {}  # category -> active alert
        self._collector_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_agent_snapshot: Dict = {}
        self._start_collector()

    def _start_collector(self):
        """Start background metrics collector."""
        if self._collector_thread and self._collector_thread.is_alive():
            return
        self._stop_event.clear()
        self._collector_thread = threading.Thread(
            target=self._collect_loop,
            name="monitoring-collector",
            daemon=True,
        )
        self._collector_thread.start()
        logger.info("Monitoring service started")

    def stop(self):
        """Stop the metrics collector."""
        self._stop_event.set()

    def _collect_loop(self):
        """Background loop collecting metrics at regular intervals."""
        while not self._stop_event.wait(METRICS_INTERVAL):
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self._metrics_history.append(metrics)
                self._evaluate_alerts(metrics)
            except Exception as e:
                logger.debug(f"Metrics collection error: {e}")

    def _collect_metrics(self) -> Dict:
        """Collect a snapshot of all system and agent metrics."""
        now = datetime.utcnow().isoformat()

        # System metrics
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0)
        disk = psutil.disk_usage(os.path.expanduser("~"))

        # Per-CPU usage
        cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)

        # Network I/O (bytes since boot, we track deltas)
        net = psutil.net_io_counters()

        # Agent metrics
        agent_metrics = self._collect_agent_metrics()

        # Task queue metrics
        task_metrics = self._collect_task_metrics()

        return {
            "timestamp": now,
            "system": {
                "cpu_percent": round(cpu, 1),
                "cpu_per_core": [round(c, 1) for c in cpu_per_core],
                "cpu_count": psutil.cpu_count(),
                "ram_percent": round(mem.percent, 1),
                "ram_used_gb": round(mem.used / (1024 ** 3), 2),
                "ram_total_gb": round(mem.total / (1024 ** 3), 2),
                "ram_available_gb": round(mem.available / (1024 ** 3), 2),
                "disk_free_gb": round(disk.free / (1024 ** 3), 2),
                "disk_total_gb": round(disk.total / (1024 ** 3), 2),
                "disk_percent": round(disk.percent, 1),
                "net_bytes_sent": net.bytes_sent,
                "net_bytes_recv": net.bytes_recv,
                "thread_count": threading.active_count(),
            },
            "agents": agent_metrics,
            "tasks": task_metrics,
        }

    def _collect_agent_metrics(self) -> Dict:
        """Collect agent-level metrics."""
        try:
            from .agent_orchestrator import active_agents, agent_task_states

            total = len(active_agents)
            by_status = {}
            total_rss_mb = 0

            for agent_id, agent in active_agents.items():
                status = agent.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1

                # Memory per agent
                process = agent.get("process")
                if process:
                    try:
                        proc = psutil.Process(process.pid)
                        rss_mb = proc.memory_info().rss / (1024 * 1024)
                        total_rss_mb += rss_mb
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            working = by_status.get("working", 0) + by_status.get("busy", 0)
            idle = by_status.get("idle", 0)
            crashed = by_status.get("crashed", 0) + by_status.get("dead", 0)

            # Failure rate from state store
            failure_rate = 0.0
            try:
                store = get_state_store()
                tq_stats = store.get_task_queue_stats()
                completed = tq_stats.get("completed", 0)
                failed = tq_stats.get("failed", 0)
                total_done = completed + failed
                if total_done > 0:
                    failure_rate = failed / total_done
            except Exception:
                pass

            result = {
                "total": total,
                "working": working,
                "idle": idle,
                "crashed": crashed,
                "by_status": by_status,
                "total_rss_mb": round(total_rss_mb),
                "avg_rss_mb": round(total_rss_mb / total) if total > 0 else 0,
                "failure_rate": round(failure_rate, 3),
            }
            self._last_agent_snapshot = result
            return result

        except Exception as e:
            logger.debug(f"Agent metrics error: {e}")
            return self._last_agent_snapshot or {"total": 0, "working": 0, "idle": 0, "crashed": 0}

    def _collect_task_metrics(self) -> Dict:
        """Collect task pipeline metrics."""
        try:
            store = get_state_store()
            stats = store.get_task_queue_stats()
            return {
                "pending": stats.get("pending", 0),
                "assigned": stats.get("assigned", 0),
                "completed": stats.get("completed", 0),
                "failed": stats.get("failed", 0),
                "total": stats.get("total", 0),
            }
        except Exception:
            return {"pending": 0, "assigned": 0, "completed": 0, "failed": 0, "total": 0}

    def _evaluate_alerts(self, metrics: Dict):
        """Check metrics against thresholds and create/resolve alerts."""
        sys_metrics = metrics.get("system", {})
        agent_metrics = metrics.get("agents", {})

        # CPU alerts
        cpu = sys_metrics.get("cpu_percent", 0)
        if cpu >= ALERT_CPU_CRIT:
            self._fire_alert("cpu", LEVEL_CRIT, f"CPU at {cpu}% (critical threshold: {ALERT_CPU_CRIT}%)")
        elif cpu >= ALERT_CPU_WARN:
            self._fire_alert("cpu", LEVEL_WARN, f"CPU at {cpu}% (warning threshold: {ALERT_CPU_WARN}%)")
        else:
            self._resolve_alert("cpu")

        # RAM alerts
        ram = sys_metrics.get("ram_percent", 0)
        if ram >= ALERT_RAM_CRIT:
            self._fire_alert("ram", LEVEL_CRIT, f"RAM at {ram}% (critical threshold: {ALERT_RAM_CRIT}%)")
        elif ram >= ALERT_RAM_WARN:
            self._fire_alert("ram", LEVEL_WARN, f"RAM at {ram}% (warning threshold: {ALERT_RAM_WARN}%)")
        else:
            self._resolve_alert("ram")

        # Disk alerts
        disk_free = sys_metrics.get("disk_free_gb", 999)
        if disk_free < ALERT_DISK_CRIT_GB:
            self._fire_alert("disk", LEVEL_CRIT, f"Disk free: {disk_free:.1f}GB (critical: <{ALERT_DISK_CRIT_GB}GB)")
        elif disk_free < ALERT_DISK_WARN_GB:
            self._fire_alert("disk", LEVEL_WARN, f"Disk free: {disk_free:.1f}GB (warning: <{ALERT_DISK_WARN_GB}GB)")
        else:
            self._resolve_alert("disk")

        # Agent failure rate
        failure_rate = agent_metrics.get("failure_rate", 0)
        if failure_rate > ALERT_FAILURE_RATE:
            self._fire_alert(
                "failure_rate", LEVEL_WARN,
                f"Agent failure rate: {failure_rate * 100:.0f}% (threshold: {ALERT_FAILURE_RATE * 100:.0f}%)"
            )
        else:
            self._resolve_alert("failure_rate")

        # Stale tasks (pending too long)
        try:
            store = get_state_store()
            pending = store.get_pending_tasks(limit=500)
            stale_count = 0
            cutoff = (datetime.utcnow() - timedelta(minutes=ALERT_STALE_TASK_MINUTES)).isoformat()
            for t in pending:
                if t.get("created_at", "") < cutoff:
                    stale_count += 1
            if stale_count > 5:
                self._fire_alert(
                    "stale_tasks", LEVEL_WARN,
                    f"{stale_count} tasks pending for >{ALERT_STALE_TASK_MINUTES} minutes"
                )
            else:
                self._resolve_alert("stale_tasks")
        except Exception:
            pass

    def _fire_alert(self, category: str, level: str, message: str):
        """Create or update an alert."""
        with self._lock:
            existing = self._active_alert_keys.get(category)
            if existing and not existing.resolved:
                # Update message if level changed
                if existing.level != level:
                    existing.level = level
                    existing.message = message
                return

            alert = Alert(level, category, message)
            self._alerts.append(alert)
            self._active_alert_keys[category] = alert

            # Keep alert history bounded
            if len(self._alerts) > 500:
                self._alerts = self._alerts[-250:]

        logger.warning(f"ALERT [{level}] {category}: {message}")

    def _resolve_alert(self, category: str):
        """Resolve an active alert."""
        with self._lock:
            existing = self._active_alert_keys.get(category)
            if existing and not existing.resolved:
                existing.resolve()
                del self._active_alert_keys[category]

    # ---- Public API ----

    def get_snapshot(self) -> Dict:
        """Get the latest metrics snapshot."""
        with self._lock:
            if self._metrics_history:
                return dict(self._metrics_history[-1])
        # Collect on-demand if no history yet
        return self._collect_metrics()

    def get_metrics_history(self, minutes: int = 60) -> List[Dict]:
        """Get metrics history for the last N minutes."""
        with self._lock:
            history = list(self._metrics_history)

        if minutes <= 0:
            return history

        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
        return [m for m in history if m.get("timestamp", "") >= cutoff]

    def get_active_alerts(self) -> List[Dict]:
        """Get all unresolved alerts."""
        with self._lock:
            return [a.to_dict() for a in self._active_alert_keys.values() if not a.resolved]

    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history including resolved."""
        with self._lock:
            return [a.to_dict() for a in self._alerts[-limit:]]

    def get_agent_health_grid(self) -> List[Dict]:
        """Get agent health data formatted for dashboard grid."""
        try:
            from .agent_orchestrator import active_agents, agent_task_states

            grid = []
            for agent_id, agent in active_agents.items():
                status = agent.get("status", "unknown")
                task_state = agent_task_states.get(agent_id, {})

                # Color mapping
                if status in ("working", "busy"):
                    color = "blue"
                elif status == "idle":
                    color = "green"
                elif status in ("crashed", "dead"):
                    color = "red"
                elif status in ("paused_flood",):
                    color = "orange"
                else:
                    color = "gray"

                # Memory
                rss_mb = 0
                process = agent.get("process")
                if process:
                    try:
                        proc = psutil.Process(process.pid)
                        rss_mb = round(proc.memory_info().rss / (1024 * 1024))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                grid.append({
                    "id": agent_id,
                    "name": agent.get("name", agent_id[:8]),
                    "status": status,
                    "color": color,
                    "specialty": agent.get("specialty", ""),
                    "provider": agent.get("cli_provider", ""),
                    "current_task": task_state.get("task_display", ""),
                    "rss_mb": rss_mb,
                    "uptime": agent.get("spawned_at", ""),
                })
            return grid
        except Exception as e:
            logger.debug(f"Agent grid error: {e}")
            return []

    def get_task_pipeline(self) -> Dict:
        """Get task pipeline funnel data."""
        try:
            from .autonomous_loop import get_autonomous_loop
            loop = get_autonomous_loop()

            scanned = len(loop.task_queue) + len(loop.completed_tasks) + len(loop.failed_tasks)
            queued = sum(1 for t in loop.task_queue if t.get("status") == "pending")
            assigned = sum(1 for t in loop.task_queue if t.get("status") == "in_progress")
            completed = len(loop.completed_tasks)
            failed = len(loop.failed_tasks)

            return {
                "scanned": scanned,
                "queued": queued,
                "assigned": assigned,
                "completed": completed,
                "failed": failed,
                "success_rate": round(completed / max(completed + failed, 1) * 100, 1),
            }
        except Exception:
            return {"scanned": 0, "queued": 0, "assigned": 0, "completed": 0, "failed": 0, "success_rate": 0}

    def get_cost_summary(self) -> Dict:
        """Get cost tracking summary."""
        try:
            from .autonomous_loop import get_autonomous_loop
            loop = get_autonomous_loop()

            return {
                "total_usd": round(loop.estimated_cost_usd, 4),
                "by_provider": {k: round(v, 4) for k, v in loop._cost_by_provider.items()},
                "tasks_completed": len(loop.completed_tasks),
                "avg_cost_per_task": round(
                    loop.estimated_cost_usd / max(len(loop.completed_tasks), 1), 4
                ),
            }
        except Exception:
            return {"total_usd": 0, "by_provider": {}, "tasks_completed": 0, "avg_cost_per_task": 0}

    def get_lock_contention(self) -> Dict:
        """Get current lock state."""
        try:
            from .lock_manager import get_lock_manager
            lm = get_lock_manager()
            locks = lm.list_locks()

            store = get_state_store()
            blocks = store.get_task_blocks(limit=20)

            return {
                "active_locks": len(locks),
                "locks": {k: v for k, v in list(locks.items())[:20]},
                "recent_blocks": blocks,
            }
        except Exception:
            return {"active_locks": 0, "locks": {}, "recent_blocks": []}


# ---- Singleton ----

_monitor: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    global _monitor
    if _monitor is None:
        _monitor = MonitoringService()
    return _monitor
