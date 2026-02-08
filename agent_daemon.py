"""
Agent Daemon for ShadowBridge - polls for tasks and executes them locally.
Runs as a background thread, polling the local web API for pending tasks.
"""
import os
import sys
import json
import time
import threading
import subprocess
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any

log = logging.getLogger("agent_daemon")

class AgentDaemon:
    """Persistent background process that polls for tasks and executes them."""

    def __init__(self, bridge_url: str = "http://127.0.0.1:6767", poll_interval: int = 30, enabled: bool = True):
        self.bridge_url = bridge_url.rstrip('/')
        self.poll_interval = poll_interval
        self.enabled = enabled
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_task: Optional[Dict] = None
        self._current_process: Optional[subprocess.Popen] = None
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._started_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "running": self.is_running,
            "enabled": self.enabled,
            "current_task": self._current_task.get("title") if self._current_task else None,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "uptime_seconds": int(time.time() - self._started_at) if self._started_at else 0,
            "poll_interval": self.poll_interval,
        }

    def start(self):
        """Start the daemon in a background thread."""
        if self.is_running:
            log.warning("AgentDaemon already running")
            return
        if not self.enabled:
            log.info("AgentDaemon is disabled, not starting")
            return

        self._running = True
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AgentDaemon")
        self._thread.start()
        log.info(f"AgentDaemon started (poll every {self.poll_interval}s)")

    def stop(self):
        """Stop the daemon gracefully."""
        self._running = False
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        log.info("AgentDaemon stopped")

    def _run_loop(self):
        """Main daemon loop - poll, claim, execute, report."""
        while self._running:
            try:
                tasks = self._fetch_pending_tasks()
                for task in tasks[:1]:  # Process one at a time
                    self._process_task(task)
            except Exception as e:
                log.error(f"Daemon loop error: {e}")

            # Sleep in small chunks so we can stop quickly
            for _ in range(self.poll_interval):
                if not self._running:
                    return
                time.sleep(1)

    def _fetch_pending_tasks(self) -> List[Dict]:
        """Fetch pending tasks assigned to bridge execution."""
        try:
            resp = requests.get(
                f"{self.bridge_url}/api/tasks/pending",
                params={"executor": "bridge"},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                tasks = data if isinstance(data, list) else data.get("tasks", [])
                return tasks
            return []
        except requests.ConnectionError:
            return []  # Bridge not running yet, that's OK
        except Exception as e:
            log.debug(f"Failed to fetch tasks: {e}")
            return []

    def _claim_task(self, task_id: str) -> bool:
        """Claim a task for bridge execution."""
        try:
            resp = requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/claim",
                json={"executor": "bridge"},
                timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            log.error(f"Failed to claim task {task_id}: {e}")
            return False

    def _post_event(self, task_id: str, event: Dict):
        """Post an execution event for a task."""
        try:
            event["timestamp"] = datetime.now().isoformat()
            requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/events",
                json=event,
                timeout=10
            )
        except Exception as e:
            log.debug(f"Failed to post event for {task_id}: {e}")

    def _checkpoint(self, task_id: str, output: str, phase: str = "EXECUTING"):
        """Save a checkpoint for a task."""
        try:
            requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/checkpoint",
                json={"phase": phase, "partial_output": output[:50000]},
                timeout=10
            )
        except Exception as e:
            log.debug(f"Failed to checkpoint {task_id}: {e}")

    def _complete_task(self, task_id: str, output: str, success: bool = True):
        """Mark a task as complete with output."""
        try:
            requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/complete",
                json={
                    "output": output[:100000],
                    "success": success,
                    "completed_at": datetime.now().isoformat()
                },
                timeout=10
            )
        except Exception as e:
            log.error(f"Failed to complete task {task_id}: {e}")

    def _process_task(self, task: Dict):
        """Claim and execute a single task."""
        task_id = task.get("id", "")
        title = task.get("title", "Untitled")
        description = task.get("description", "")
        task_type = task.get("taskType", task.get("task_type", "GENERAL"))
        project_dir = task.get("project_dir", task.get("workingDirectory", ""))

        log.info(f"Processing task: {title} ({task_id})")

        if not self._claim_task(task_id):
            log.warning(f"Failed to claim task {task_id}")
            return

        with self._lock:
            self._current_task = task

        self._post_event(task_id, {"type": "started", "message": f"Bridge daemon claimed task: {title}"})

        try:
            # Determine which CLI tool to use
            cli_cmd = self._resolve_cli_command(task_type, description, project_dir)
            prompt = self._build_prompt(title, description)

            log.info(f"Executing: {cli_cmd[0]} for task {task_id}")
            self._post_event(task_id, {"type": "executing", "message": f"Running {cli_cmd[0]}..."})

            # Run the CLI tool
            output = self._run_cli(cli_cmd, prompt, project_dir, task_id)

            self._complete_task(task_id, output, success=True)
            self._tasks_completed += 1
            log.info(f"Task completed: {title}")

        except subprocess.TimeoutExpired:
            self._complete_task(task_id, "Task timed out after 30 minutes", success=False)
            self._tasks_failed += 1
            log.error(f"Task timed out: {title}")
        except Exception as e:
            self._complete_task(task_id, f"Error: {str(e)}", success=False)
            self._tasks_failed += 1
            log.error(f"Task failed: {title} - {e}")
        finally:
            with self._lock:
                self._current_task = None
                self._current_process = None

    def _resolve_cli_command(self, task_type: str, description: str, project_dir: str) -> List[str]:
        """Pick the right CLI tool based on task type."""
        desc_lower = description.lower()

        # Prefer Claude Code for code tasks
        if self._command_exists("claude"):
            return ["claude", "--print"]

        # Fallback to codex
        if self._command_exists("codex"):
            return ["codex", "exec"]

        # Fallback to just outputting the prompt
        return [sys.executable, "-c", "import sys; print(sys.stdin.read())"]

    def _command_exists(self, cmd: str) -> bool:
        """Check if a CLI command is available."""
        try:
            result = subprocess.run(
                ["where" if os.name == "nt" else "which", cmd],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _build_prompt(self, title: str, description: str) -> str:
        """Build a prompt for the CLI tool."""
        return f"Task: {title}\n\nDescription:\n{description}\n\nPlease complete this task. Be thorough but concise."

    def _run_cli(self, cmd: List[str], prompt: str, cwd: str, task_id: str, timeout: int = 1800) -> str:
        """Run a CLI command with the given prompt, capturing output."""
        env = os.environ.copy()

        work_dir = cwd if cwd and os.path.isdir(cwd) else os.path.expanduser("~")

        full_cmd = cmd + [prompt] if cmd[0] != sys.executable else cmd

        proc = subprocess.Popen(
            full_cmd,
            stdin=subprocess.PIPE if cmd[0] == sys.executable else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=work_dir,
            env=env,
            text=True,
            errors='replace'
        )

        with self._lock:
            self._current_process = proc

        output_lines = []
        checkpoint_interval = 60  # Checkpoint every 60 seconds
        last_checkpoint = time.time()

        try:
            if cmd[0] == sys.executable:
                proc.stdin.write(prompt)
                proc.stdin.close()

            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    output_lines.append(line)
                    # Periodic checkpoint
                    if time.time() - last_checkpoint > checkpoint_interval:
                        self._checkpoint(task_id, ''.join(output_lines))
                        last_checkpoint = time.time()

            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise

        return ''.join(output_lines)


# Singleton daemon instance
_daemon_instance: Optional[AgentDaemon] = None

def get_daemon() -> AgentDaemon:
    """Get or create the singleton daemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = AgentDaemon()
    return _daemon_instance

def start_daemon(bridge_url: str = "http://127.0.0.1:6767", poll_interval: int = 30):
    """Start the agent daemon."""
    daemon = get_daemon()
    daemon.bridge_url = bridge_url
    daemon.poll_interval = poll_interval
    daemon.start()
    return daemon
