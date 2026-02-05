"""
Worker Coordinator

Manages remote worker machines that run agent subprocesses. The coordinator
runs on the main ShadowBridge host and communicates with workers via HTTP
REST over Tailscale or LAN.

Architecture:
    Coordinator (this machine) -> Worker 1 (remote) -> spawns agents locally
                               -> Worker 2 (remote) -> spawns agents locally
                               -> Local worker       -> spawns agents locally

Usage:
    coord = get_worker_coordinator()
    coord.register_worker("192.168.1.50", 19400, capacity=100)
    result = coord.spawn_agent_on_best_worker(agent_config)
    coord.get_all_remote_agents()
"""

import os
import time
import json
import logging
import threading
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

from web.services.state_store import get_state_store

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = int(os.environ.get("WORKER_HEARTBEAT_INTERVAL", "10"))
HEARTBEAT_TIMEOUT = int(os.environ.get("WORKER_HEARTBEAT_TIMEOUT", "30"))
REQUEST_TIMEOUT = int(os.environ.get("WORKER_REQUEST_TIMEOUT", "15"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "20"))


@dataclass
class WorkerNode:
    """Represents a remote worker machine."""
    worker_id: str
    host: str
    port: int
    capacity: int = 100
    label: str = ""
    status: str = "unknown"          # online, offline, draining, unknown
    registered_at: str = ""
    last_heartbeat: str = ""
    agent_count: int = 0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    disk_free_gb: float = 0.0
    version: str = ""
    error: str = ""


@dataclass
class RemoteAgent:
    """Represents an agent running on a remote worker."""
    agent_id: str
    worker_id: str
    name: str = ""
    specialty: str = ""
    status: str = "unknown"
    pid: int = 0
    started_at: str = ""
    current_task: str = ""


class WorkerCoordinator:
    """Manages remote worker machines for distributed agent execution."""

    def __init__(self):
        self._lock = threading.Lock()
        self._workers: Dict[str, WorkerNode] = {}
        self._remote_agents: Dict[str, RemoteAgent] = {}
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        self._load_persisted_workers()

    def start(self):
        """Start the heartbeat monitoring loop."""
        if self._running:
            return
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="worker-heartbeat"
        )
        self._heartbeat_thread.start()
        logger.info("WorkerCoordinator started")

    def stop(self):
        """Stop the heartbeat loop."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        logger.info("WorkerCoordinator stopped")

    # ---- Worker Registration ----

    def register_worker(self, host: str, port: int, capacity: int = 100,
                        label: str = "") -> Dict:
        """Register a new remote worker machine."""
        with self._lock:
            if len(self._workers) >= MAX_WORKERS:
                return {"success": False, "error": f"Max workers ({MAX_WORKERS}) reached"}

        worker_id = f"w-{host.replace('.', '-')}:{port}"

        # Verify connectivity
        try:
            resp = requests.get(
                f"http://{host}:{port}/status",
                timeout=REQUEST_TIMEOUT
            )
            if resp.status_code != 200:
                return {"success": False, "error": f"Worker returned status {resp.status_code}"}
            data = resp.json()
        except requests.RequestException as e:
            return {"success": False, "error": f"Cannot reach worker: {e}"}

        now = datetime.utcnow().isoformat()
        worker = WorkerNode(
            worker_id=worker_id,
            host=host,
            port=port,
            capacity=capacity,
            label=label or f"{host}:{port}",
            status="online",
            registered_at=now,
            last_heartbeat=now,
            agent_count=data.get("agent_count", 0),
            cpu_percent=data.get("cpu_percent", 0),
            ram_percent=data.get("ram_percent", 0),
            disk_free_gb=data.get("disk_free_gb", 0),
            version=data.get("version", ""),
        )

        with self._lock:
            self._workers[worker_id] = worker

        self._persist_workers()
        logger.info(f"Registered worker {worker_id} (capacity={capacity})")
        return {"success": True, "worker_id": worker_id, "status": worker.status}

    def remove_worker(self, worker_id: str, drain: bool = False) -> Dict:
        """Remove a worker. If drain=True, stop its agents first."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return {"success": False, "error": "Worker not found"}

        if drain:
            worker.status = "draining"
            agents = self._get_worker_agents(worker_id)
            for agent in agents:
                try:
                    self._worker_request(worker, "POST", "/stop", json={"agent_id": agent.agent_id})
                except Exception as e:
                    logger.warning(f"Failed to stop agent {agent.agent_id} on {worker_id}: {e}")

        with self._lock:
            self._workers.pop(worker_id, None)
            # Remove associated remote agents from tracking
            to_remove = [aid for aid, a in self._remote_agents.items() if a.worker_id == worker_id]
            for aid in to_remove:
                self._remote_agents.pop(aid, None)

        self._persist_workers()
        logger.info(f"Removed worker {worker_id} (drain={drain})")
        return {"success": True}

    # ---- Agent Spawning ----

    def spawn_agent_on_best_worker(self, config: dict) -> Dict:
        """
        Spawn an agent on the worker with the most available capacity.

        Config keys: name, specialty, provider, model, project_path
        """
        worker = self._select_best_worker()
        if not worker:
            return {"success": False, "error": "No workers available"}

        return self.spawn_agent_on_worker(worker.worker_id, config)

    def spawn_agent_on_worker(self, worker_id: str, config: dict) -> Dict:
        """Spawn an agent on a specific worker."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return {"success": False, "error": "Worker not found"}
            if worker.status not in ("online",):
                return {"success": False, "error": f"Worker is {worker.status}"}
            if worker.agent_count >= worker.capacity:
                return {"success": False, "error": "Worker at capacity"}

        try:
            resp = self._worker_request(worker, "POST", "/spawn", json=config)
            data = resp.json()

            if data.get("success"):
                agent_id = data["agent_id"]
                remote_agent = RemoteAgent(
                    agent_id=agent_id,
                    worker_id=worker_id,
                    name=config.get("name", ""),
                    specialty=config.get("specialty", ""),
                    status="running",
                    pid=data.get("pid", 0),
                    started_at=datetime.utcnow().isoformat(),
                )
                with self._lock:
                    self._remote_agents[agent_id] = remote_agent
                    worker.agent_count += 1

                logger.info(f"Spawned agent {agent_id} on worker {worker_id}")
            return data

        except requests.RequestException as e:
            return {"success": False, "error": f"Worker request failed: {e}"}

    def stop_remote_agent(self, agent_id: str) -> Dict:
        """Stop an agent running on a remote worker."""
        with self._lock:
            remote_agent = self._remote_agents.get(agent_id)
            if not remote_agent:
                return {"success": False, "error": "Remote agent not found"}
            worker = self._workers.get(remote_agent.worker_id)
            if not worker:
                return {"success": False, "error": "Worker not found"}

        try:
            resp = self._worker_request(worker, "POST", "/stop", json={"agent_id": agent_id})
            data = resp.json()

            if data.get("success"):
                with self._lock:
                    self._remote_agents.pop(agent_id, None)
                    worker.agent_count = max(0, worker.agent_count - 1)

            return data
        except requests.RequestException as e:
            return {"success": False, "error": f"Worker request failed: {e}"}

    def assign_task_to_remote_agent(self, agent_id: str, task: dict) -> Dict:
        """Assign a task to a remote agent."""
        with self._lock:
            remote_agent = self._remote_agents.get(agent_id)
            if not remote_agent:
                return {"success": False, "error": "Remote agent not found"}
            worker = self._workers.get(remote_agent.worker_id)
            if not worker:
                return {"success": False, "error": "Worker not found"}

        try:
            resp = self._worker_request(
                worker, "POST", f"/agents/{agent_id}/task", json=task
            )
            return resp.json()
        except requests.RequestException as e:
            return {"success": False, "error": f"Worker request failed: {e}"}

    # ---- Status ----

    def get_all_workers(self) -> List[Dict]:
        """Get status of all registered workers."""
        with self._lock:
            workers = list(self._workers.values())
        return [asdict(w) for w in workers]

    def get_online_workers(self) -> List[Dict]:
        """Get only online workers."""
        with self._lock:
            workers = [w for w in self._workers.values() if w.status == "online"]
        return [asdict(w) for w in workers]

    def get_worker_status(self, worker_id: str) -> Optional[Dict]:
        """Get status of a specific worker."""
        with self._lock:
            worker = self._workers.get(worker_id)
        if not worker:
            return None
        return asdict(worker)

    def get_all_remote_agents(self) -> List[Dict]:
        """Get all agents across all workers."""
        with self._lock:
            agents = list(self._remote_agents.values())
        return [asdict(a) for a in agents]

    def get_capacity_summary(self) -> Dict:
        """Get aggregate capacity across all workers."""
        with self._lock:
            workers = list(self._workers.values())

        online = [w for w in workers if w.status == "online"]
        total_capacity = sum(w.capacity for w in online)
        total_agents = sum(w.agent_count for w in online)
        avg_cpu = sum(w.cpu_percent for w in online) / len(online) if online else 0
        avg_ram = sum(w.ram_percent for w in online) / len(online) if online else 0

        return {
            "total_workers": len(workers),
            "online_workers": len(online),
            "total_capacity": total_capacity,
            "total_remote_agents": total_agents,
            "available_slots": total_capacity - total_agents,
            "avg_cpu_percent": round(avg_cpu, 1),
            "avg_ram_percent": round(avg_ram, 1),
        }

    # ---- Internal ----

    def _select_best_worker(self) -> Optional[WorkerNode]:
        """Select the worker with the most headroom (capacity - current agents)."""
        with self._lock:
            candidates = [
                w for w in self._workers.values()
                if w.status == "online" and w.agent_count < w.capacity
            ]
        if not candidates:
            return None

        # Sort by available capacity (desc), then by CPU usage (asc)
        candidates.sort(key=lambda w: (
            w.capacity - w.agent_count,
            -w.cpu_percent
        ), reverse=True)
        return candidates[0]

    def _worker_request(self, worker: WorkerNode, method: str, path: str,
                        **kwargs) -> requests.Response:
        """Make an HTTP request to a worker."""
        url = f"http://{worker.host}:{worker.port}{path}"
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        resp = requests.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    def _get_worker_agents(self, worker_id: str) -> List[RemoteAgent]:
        """Get agents running on a specific worker."""
        with self._lock:
            return [a for a in self._remote_agents.values() if a.worker_id == worker_id]

    def _heartbeat_loop(self):
        """Periodically poll all workers for health status."""
        while self._running:
            try:
                self._poll_workers()
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
            time.sleep(HEARTBEAT_INTERVAL)

    def _poll_workers(self):
        """Poll all workers for current status."""
        with self._lock:
            workers = list(self._workers.values())

        for worker in workers:
            try:
                resp = requests.get(
                    f"http://{worker.host}:{worker.port}/status",
                    timeout=REQUEST_TIMEOUT
                )
                if resp.status_code == 200:
                    data = resp.json()
                    worker.status = "online"
                    worker.last_heartbeat = datetime.utcnow().isoformat()
                    worker.agent_count = data.get("agent_count", 0)
                    worker.cpu_percent = data.get("cpu_percent", 0)
                    worker.ram_percent = data.get("ram_percent", 0)
                    worker.disk_free_gb = data.get("disk_free_gb", 0)
                    worker.version = data.get("version", "")
                    worker.error = ""

                    # Sync remote agent list from worker
                    self._sync_remote_agents(worker, data.get("agents", []))
                else:
                    worker.error = f"HTTP {resp.status_code}"
                    self._mark_worker_offline(worker)

            except requests.RequestException as e:
                worker.error = str(e)
                self._mark_worker_offline(worker)

    def _mark_worker_offline(self, worker: WorkerNode):
        """Mark a worker as offline if heartbeat timeout exceeded."""
        if worker.last_heartbeat:
            try:
                last = datetime.fromisoformat(worker.last_heartbeat)
                if datetime.utcnow() - last > timedelta(seconds=HEARTBEAT_TIMEOUT):
                    if worker.status != "offline":
                        logger.warning(f"Worker {worker.worker_id} went offline")
                        worker.status = "offline"
            except ValueError:
                worker.status = "offline"
        else:
            worker.status = "offline"

    def _sync_remote_agents(self, worker: WorkerNode, agents_data: list):
        """Sync remote agent tracking from worker's reported agents."""
        reported_ids = set()
        for agent_data in agents_data:
            agent_id = agent_data.get("agent_id", "")
            if not agent_id:
                continue
            reported_ids.add(agent_id)

            with self._lock:
                if agent_id not in self._remote_agents:
                    self._remote_agents[agent_id] = RemoteAgent(
                        agent_id=agent_id,
                        worker_id=worker.worker_id,
                        name=agent_data.get("name", ""),
                        specialty=agent_data.get("specialty", ""),
                        status=agent_data.get("status", "unknown"),
                        pid=agent_data.get("pid", 0),
                        started_at=agent_data.get("started_at", ""),
                        current_task=agent_data.get("current_task", ""),
                    )
                else:
                    ra = self._remote_agents[agent_id]
                    ra.status = agent_data.get("status", ra.status)
                    ra.current_task = agent_data.get("current_task", "")

        # Remove agents no longer reported by this worker
        with self._lock:
            stale = [
                aid for aid, a in self._remote_agents.items()
                if a.worker_id == worker.worker_id and aid not in reported_ids
            ]
            for aid in stale:
                self._remote_agents.pop(aid, None)

    def _persist_workers(self):
        """Save worker registrations to state store."""
        store = get_state_store()
        with self._lock:
            workers = list(self._workers.values())
        for w in workers:
            store.upsert_worker(
                worker_id=w.worker_id,
                host=w.host,
                port=w.port,
                capacity=w.capacity,
                label=w.label,
            )

    def _load_persisted_workers(self):
        """Load worker registrations from state store on startup."""
        try:
            store = get_state_store()
            rows = store.get_all_workers()
            for row in rows:
                worker = WorkerNode(
                    worker_id=row["worker_id"],
                    host=row["host"],
                    port=row["port"],
                    capacity=row.get("capacity", 100),
                    label=row.get("label", ""),
                    status="unknown",
                    registered_at=row.get("registered_at", ""),
                )
                self._workers[worker.worker_id] = worker
            if rows:
                logger.info(f"Loaded {len(rows)} persisted workers")
        except Exception as e:
            logger.warning(f"Could not load persisted workers: {e}")


# ---- Singleton ----

_coordinator: Optional[WorkerCoordinator] = None
_coord_lock = threading.Lock()


def get_worker_coordinator() -> WorkerCoordinator:
    """Get or create the global WorkerCoordinator singleton."""
    global _coordinator
    if _coordinator is None:
        with _coord_lock:
            if _coordinator is None:
                _coordinator = WorkerCoordinator()
                _coordinator.start()
    return _coordinator
