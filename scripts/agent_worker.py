#!/usr/bin/env python3
"""
Shadow Agent Worker

Lightweight HTTP server that runs on remote machines to spawn and manage
AI agents. Communicates with the WorkerCoordinator on the main ShadowBridge.

Usage:
    python agent_worker.py --port 19400 --capacity 50
    python agent_worker.py --port 19400 --capacity 100 --coordinator 192.168.1.10:6769

The worker exposes REST endpoints for the coordinator to:
- Spawn agents (POST /spawn)
- Stop agents (POST /stop)
- Assign tasks (POST /agents/<id>/task)
- Query status (GET /status, GET /agents)
"""

import os
import sys
import json
import time
import queue
import signal
import logging
import argparse
import platform
import subprocess
import threading
import psutil
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional
from uuid import uuid4

VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent-worker")


# ---- Agent Management ----

class AgentProcess:
    """Wraps a single agent subprocess."""

    def __init__(self, agent_id: str, name: str, specialty: str,
                 provider: str, model: str, project_path: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.specialty = specialty
        self.provider = provider
        self.model = model
        self.project_path = project_path
        self.process: Optional[subprocess.Popen] = None
        self.status = "starting"
        self.pid = 0
        self.started_at = datetime.utcnow().isoformat()
        self.current_task = ""
        self.task_queue: queue.Queue = queue.Queue()
        self.output_lines: List[str] = []
        self._max_output = 500
        self._lock = threading.Lock()

    def start(self):
        """Start the agent subprocess."""
        cmd = self._build_command()
        if not cmd:
            self.status = "failed"
            raise RuntimeError(f"Cannot build command for provider '{self.provider}'")

        cwd = self.project_path if self.project_path and os.path.isdir(self.project_path) else None

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                text=True,
                bufsize=1,
            )
            self.pid = self.process.pid
            self.status = "running"

            # Start output reader thread
            t = threading.Thread(target=self._read_output, daemon=True,
                                 name=f"out-{self.agent_id[:8]}")
            t.start()

            # Start task processor thread
            t2 = threading.Thread(target=self._process_tasks, daemon=True,
                                  name=f"task-{self.agent_id[:8]}")
            t2.start()

            logger.info(f"Agent {self.agent_id} started (PID {self.pid})")

        except Exception as e:
            self.status = "failed"
            raise RuntimeError(f"Failed to start agent: {e}")

    def stop(self):
        """Stop the agent subprocess."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
        self.status = "stopped"
        logger.info(f"Agent {self.agent_id} stopped")

    def assign_task(self, task: dict):
        """Queue a task for this agent."""
        self.task_queue.put(task)

    def is_alive(self) -> bool:
        """Check if the agent process is still running."""
        return self.process is not None and self.process.poll() is None

    def to_dict(self) -> dict:
        """Serialize agent state."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "specialty": self.specialty,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "pid": self.pid,
            "started_at": self.started_at,
            "current_task": self.current_task,
            "output_lines": len(self.output_lines),
            "queued_tasks": self.task_queue.qsize(),
        }

    def _build_command(self) -> Optional[List[str]]:
        """Build the CLI command for this agent's provider."""
        provider = self.provider.lower()

        if provider in ("claude", "anthropic"):
            return [
                "claude", "--print", "--verbose",
                "--allowedTools", "Edit,Write,Bash,Read,Glob,Grep",
            ]
        elif provider in ("gemini", "google"):
            model = self.model or "gemini-2.5-pro"
            return ["gemini", "--model", model]
        elif provider in ("codex", "openai"):
            return ["codex", "--full-auto"]
        elif provider in ("ollama",):
            model = self.model or "codellama"
            return ["ollama", "run", model]
        else:
            logger.warning(f"Unknown provider: {provider}")
            return None

    def _read_output(self):
        """Read stdout/stderr from agent process."""
        try:
            for line in self.process.stdout:
                line = line.rstrip("\n")
                with self._lock:
                    self.output_lines.append(line)
                    if len(self.output_lines) > self._max_output:
                        self.output_lines = self.output_lines[-self._max_output:]

                # Detect task protocol markers
                if "<<<TASK_DONE>>" in line or "<<<TASK_RESULT>>" in line:
                    self.current_task = ""
        except Exception:
            pass

        if self.status == "running":
            self.status = "crashed"
            logger.warning(f"Agent {self.agent_id} output stream ended (crashed?)")

    def _process_tasks(self):
        """Process tasks from the queue."""
        while self.status in ("running", "starting"):
            try:
                task = self.task_queue.get(timeout=2)
            except queue.Empty:
                continue

            task_desc = task.get("title", task.get("description", "unnamed"))
            self.current_task = task_desc
            logger.info(f"Agent {self.agent_id} working on: {task_desc}")

            # Build prompt and send to agent stdin
            prompt = self._format_task_prompt(task)
            try:
                if self.process and self.process.stdin:
                    self.process.stdin.write(prompt + "\n")
                    self.process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                logger.error(f"Agent {self.agent_id} stdin write failed: {e}")
                self.status = "crashed"
                break

    def _format_task_prompt(self, task: dict) -> str:
        """Format a task as a prompt string for the agent."""
        parts = []
        if task.get("title"):
            parts.append(f"Task: {task['title']}")
        if task.get("description"):
            parts.append(f"Description: {task['description']}")
        if task.get("file_path"):
            parts.append(f"Target file: {task['file_path']}")
        if task.get("category"):
            parts.append(f"Category: {task['category']}")
        return "\n".join(parts) if parts else json.dumps(task)


class WorkerManager:
    """Manages all agent processes on this worker machine."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._agents: Dict[str, AgentProcess] = {}
        self._lock = threading.Lock()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the watchdog loop."""
        self._running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="worker-watchdog"
        )
        self._watchdog_thread.start()

    def stop(self):
        """Stop all agents and the watchdog."""
        self._running = False
        with self._lock:
            for agent in self._agents.values():
                try:
                    agent.stop()
                except Exception:
                    pass
            self._agents.clear()

    def spawn_agent(self, config: dict) -> dict:
        """Spawn a new agent process."""
        with self._lock:
            if len(self._agents) >= self.capacity:
                return {"success": False, "error": f"At capacity ({self.capacity})"}

        agent_id = f"remote-{uuid4().hex[:12]}"
        agent = AgentProcess(
            agent_id=agent_id,
            name=config.get("name", f"agent-{agent_id[:8]}"),
            specialty=config.get("specialty", "general"),
            provider=config.get("provider", "gemini"),
            model=config.get("model", ""),
            project_path=config.get("project_path", ""),
        )

        try:
            agent.start()
        except RuntimeError as e:
            return {"success": False, "error": str(e)}

        with self._lock:
            self._agents[agent_id] = agent

        return {
            "success": True,
            "agent_id": agent_id,
            "pid": agent.pid,
            "status": agent.status,
        }

    def stop_agent(self, agent_id: str) -> dict:
        """Stop a specific agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return {"success": False, "error": "Agent not found"}

        agent.stop()

        with self._lock:
            self._agents.pop(agent_id, None)

        return {"success": True}

    def assign_task(self, agent_id: str, task: dict) -> dict:
        """Assign a task to an agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return {"success": False, "error": "Agent not found"}

        agent.assign_task(task)
        return {"success": True, "queued_tasks": agent.task_queue.qsize()}

    def get_status(self) -> dict:
        """Get worker status."""
        with self._lock:
            agent_count = len(self._agents)

        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/") if platform.system() != "Windows" else psutil.disk_usage("C:\\")
        except Exception:
            cpu, mem, disk = 0, None, None

        return {
            "version": VERSION,
            "agent_count": agent_count,
            "capacity": self.capacity,
            "available_slots": self.capacity - agent_count,
            "cpu_percent": round(cpu, 1),
            "ram_percent": round(mem.percent, 1) if mem else 0,
            "ram_used_gb": round(mem.used / (1024 ** 3), 1) if mem else 0,
            "ram_total_gb": round(mem.total / (1024 ** 3), 1) if mem else 0,
            "disk_free_gb": round(disk.free / (1024 ** 3), 1) if disk else 0,
            "hostname": platform.node(),
            "platform": platform.system(),
            "agents": self.get_all_agents(),
        }

    def get_all_agents(self) -> List[dict]:
        """Get all agent states."""
        with self._lock:
            agents = list(self._agents.values())
        return [a.to_dict() for a in agents]

    def get_agent_output(self, agent_id: str, tail: int = 50) -> Optional[List[str]]:
        """Get recent output lines from an agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
        if not agent:
            return None
        with agent._lock:
            return list(agent.output_lines[-tail:])

    def _watchdog_loop(self):
        """Monitor agent health."""
        while self._running:
            try:
                self._check_agents()
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
            time.sleep(5)

    def _check_agents(self):
        """Check all agents for crashes."""
        with self._lock:
            agents = list(self._agents.items())

        for agent_id, agent in agents:
            if not agent.is_alive() and agent.status == "running":
                agent.status = "crashed"
                logger.warning(f"Agent {agent_id} crashed (PID {agent.pid})")


# ---- HTTP Handler ----

_manager: Optional[WorkerManager] = None


class WorkerHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for worker endpoints."""

    def log_message(self, format, *args):
        logger.debug(format % args)

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.rstrip("/")

        if path == "/status":
            self._send_json(_manager.get_status())

        elif path == "/agents":
            self._send_json({"agents": _manager.get_all_agents()})

        elif path.startswith("/agents/") and path.endswith("/output"):
            agent_id = path.split("/")[2]
            output = _manager.get_agent_output(agent_id)
            if output is None:
                self._send_json({"error": "Agent not found"}, 404)
            else:
                self._send_json({"agent_id": agent_id, "output": output})

        elif path == "/health":
            self._send_json({"status": "ok", "version": VERSION})

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = self.path.rstrip("/")

        if path == "/spawn":
            data = self._read_json()
            result = _manager.spawn_agent(data)
            status = 200 if result.get("success") else 400
            self._send_json(result, status)

        elif path == "/stop":
            data = self._read_json()
            agent_id = data.get("agent_id", "")
            if not agent_id:
                self._send_json({"error": "agent_id required"}, 400)
                return
            result = _manager.stop_agent(agent_id)
            self._send_json(result)

        elif path.startswith("/agents/") and path.endswith("/task"):
            agent_id = path.split("/")[2]
            task = self._read_json()
            result = _manager.assign_task(agent_id, task)
            self._send_json(result)

        elif path == "/stop-all":
            _manager.stop()
            _manager.start()
            self._send_json({"success": True, "message": "All agents stopped"})

        else:
            self._send_json({"error": "Not found"}, 404)


def main():
    global _manager

    parser = argparse.ArgumentParser(description="Shadow Agent Worker")
    parser.add_argument("--port", type=int, default=19400, help="HTTP server port (default: 19400)")
    parser.add_argument("--capacity", type=int, default=100, help="Max agents (default: 100)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--coordinator", default="", help="Coordinator host:port for auto-registration")
    args = parser.parse_args()

    _manager = WorkerManager(capacity=args.capacity)
    _manager.start()

    server = HTTPServer((args.host, args.port), WorkerHTTPHandler)
    logger.info(f"Agent Worker v{VERSION} listening on {args.host}:{args.port} (capacity={args.capacity})")

    # Auto-register with coordinator if specified
    if args.coordinator:
        try:
            import requests as req
            coord_url = f"http://{args.coordinator}/api/workers/register"
            my_ip = _get_local_ip()
            resp = req.post(coord_url, json={
                "host": my_ip, "port": args.port, "capacity": args.capacity,
            }, timeout=10)
            if resp.status_code == 200:
                logger.info(f"Registered with coordinator at {args.coordinator}")
            else:
                logger.warning(f"Coordinator registration failed: {resp.text}")
        except Exception as e:
            logger.warning(f"Could not register with coordinator: {e}")

    def shutdown(sig, frame):
        logger.info("Shutting down...")
        _manager.stop()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)


def _get_local_ip() -> str:
    """Get local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    main()
