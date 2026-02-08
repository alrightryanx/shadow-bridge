"""
Agent Daemon — persistent background loop that polls ShadowBridge API for
pending tasks and executes them via CLI tools (Claude Code, Codex, Gemini CLI).

Imported by shadow_bridge_gui.py via:
    from agent_daemon import start_daemon, get_daemon
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
    """Persistent background daemon that polls for tasks and executes them
    via CLI tools (Claude Code, Codex, Gemini CLI)."""

    def __init__(self, bridge_url: str = "http://127.0.0.1:6767",
                 poll_interval: int = 30):
        self.bridge_url = bridge_url.rstrip('/')
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_task: Optional[Dict] = None
        self._current_process: Optional[subprocess.Popen] = None
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._started_at: Optional[float] = None
        self._lock = threading.Lock()

    # ---- Properties ----

    @property
    def is_running(self) -> bool:
        return (self._running
                and self._thread is not None
                and self._thread.is_alive())

    @property
    def status(self) -> Dict[str, Any]:
        """Status dict consumed by the /api/daemon/status endpoint."""
        return {
            "running": self.is_running,
            "bridge_url": self.bridge_url,
            "current_task": (self._current_task.get("title")
                             if self._current_task else None),
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "uptime_seconds": (int(time.time() - self._started_at)
                               if self._started_at else 0),
            "poll_interval": self.poll_interval,
        }

    # ---- Lifecycle ----

    def start(self):
        """Start the polling loop in a background thread."""
        if self.is_running:
            log.warning("AgentDaemon already running")
            return
        self._running = True
        self._started_at = time.time()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="AgentDaemon"
        )
        self._thread.start()
        log.info(f"AgentDaemon started (polling {self.bridge_url} "
                 f"every {self.poll_interval}s)")

    def stop(self):
        """Gracefully stop the loop and kill any running subprocess."""
        self._running = False
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        log.info("AgentDaemon stopped")

    # ---- Main Loop ----

    def _poll_loop(self):
        """Main loop: GET /api/tasks/pending, claim one, execute, post results.
        Runs in a background thread until stop() is called."""
        # Brief initial delay so the web server can finish starting
        time.sleep(5)

        while self._running:
            try:
                tasks = self._fetch_pending_tasks()
                if tasks:
                    # Process one task at a time
                    self._execute_task(tasks[0])
            except Exception as e:
                log.error(f"Daemon poll loop error: {e}")

            # Sleep in 1-second chunks so stop() is responsive
            for _ in range(self.poll_interval):
                if not self._running:
                    return
                time.sleep(1)

    # ---- API Helpers ----

    def _fetch_pending_tasks(self) -> List[Dict]:
        """GET /api/tasks/pending?executor=bridge — fetch unclaimed tasks."""
        try:
            resp = requests.get(
                f"{self.bridge_url}/api/tasks/pending",
                params={"executor": "bridge"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("tasks", [])
            return []
        except requests.ConnectionError:
            return []  # Bridge web server not ready yet
        except Exception as e:
            log.debug(f"Failed to fetch pending tasks: {e}")
            return []

    def _claim_task(self, task_id: str) -> bool:
        """POST /api/tasks/{id}/claim — claim a pending task for execution."""
        try:
            resp = requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/claim",
                json={"executor": "bridge"},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            log.error(f"Failed to claim task {task_id}: {e}")
            return False

    def _checkpoint_task(self, task_id: str, data: Dict):
        """POST /api/tasks/{id}/checkpoint — save intermediate progress."""
        try:
            requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/checkpoint",
                json=data,
                timeout=10,
            )
        except Exception as e:
            log.debug(f"Failed to checkpoint task {task_id}: {e}")

    def _complete_task(self, task_id: str, result: Dict):
        """POST /api/tasks/{id}/complete — mark task finished with output."""
        try:
            requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/complete",
                json=result,
                timeout=10,
            )
        except Exception as e:
            log.error(f"Failed to complete task {task_id}: {e}")

    def _post_event(self, task_id: str, event_type: str, message: str):
        """POST /api/tasks/{id}/events — log an execution event."""
        try:
            requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/events",
                json={
                    "type": event_type,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                },
                timeout=10,
            )
        except Exception as e:
            log.debug(f"Failed to post event for task {task_id}: {e}")

    # ---- Task Execution ----

    def _execute_task(self, task: Dict):
        """Claim a task, run it via the appropriate CLI tool, and post results.

        Steps:
        1. Claim the task via the API
        2. Determine which CLI tool to use (claude, codex, gemini)
        3. Build the prompt from task title + description
        4. Run the subprocess with timeout
        5. Post periodic checkpoints and the final result
        """
        task_id = task.get("id", "")
        title = task.get("title", "Untitled")
        description = task.get("description", "")
        project_dir = task.get("project_dir",
                               task.get("workingDirectory", ""))
        task_input = task.get("input", {})

        # Determine preferred CLI tool from task metadata
        preferred_cli = (task_input.get("cli_tool")
                         or task.get("cli_tool")
                         or "claude")

        # Task-level timeout (default 10 minutes)
        timeout = int(task_input.get("timeout", task.get("timeout", 600)))

        log.info(f"Processing task: {title} ({task_id}) via {preferred_cli}")

        # Step 1: Claim
        if not self._claim_task(task_id):
            log.warning(f"Could not claim task {task_id} (already taken?)")
            return

        with self._lock:
            self._current_task = task

        self._post_event(task_id, "STARTED",
                         f"Bridge daemon claimed task: {title}")

        try:
            # Step 2: Resolve CLI command
            cmd = self._resolve_cli_command(preferred_cli)

            # Step 3: Build prompt
            prompt = self._build_prompt(title, description)

            log.info(f"Executing via {cmd[0]} for task {task_id}")
            self._post_event(task_id, "EXECUTING",
                             f"Running {cmd[0]}...")

            # Step 4: Run
            output = self._run_subprocess(cmd, prompt, project_dir,
                                          task_id, timeout)

            # Step 5: Complete
            self._complete_task(task_id, {
                "output": output[:100_000],
                "success": True,
                "completed_at": datetime.now().isoformat(),
            })
            self._tasks_completed += 1
            log.info(f"Task completed: {title}")

        except subprocess.TimeoutExpired:
            self._complete_task(task_id, {
                "output": f"Task timed out after {timeout} seconds",
                "success": False,
                "completed_at": datetime.now().isoformat(),
            })
            self._tasks_failed += 1
            log.error(f"Task timed out: {title}")

        except Exception as e:
            self._complete_task(task_id, {
                "output": f"Error: {str(e)}",
                "success": False,
                "completed_at": datetime.now().isoformat(),
            })
            self._tasks_failed += 1
            log.error(f"Task failed: {title} — {e}")

        finally:
            with self._lock:
                self._current_task = None
                self._current_process = None

    def _resolve_cli_command(self, preferred: str) -> List[str]:
        """Build the CLI command list for the preferred tool.

        Supported tools:
          - claude  -> claude -p "<prompt>" --output-format json
          - codex   -> codex exec "<prompt>"
          - gemini  -> gemini "<prompt>"

        Falls back through the chain if preferred tool is not installed.
        """
        preferred = preferred.lower().strip()

        tool_commands = {
            "claude": ["claude", "-p"],
            "codex": ["codex", "exec"],
            "gemini": ["gemini"],
        }

        # Try preferred tool first, then fallbacks
        order = [preferred] + [t for t in ("claude", "codex", "gemini")
                               if t != preferred]

        for tool in order:
            cmd = tool_commands.get(tool)
            if cmd and self._command_exists(cmd[0]):
                # Add output format flag for Claude
                if tool == "claude":
                    return cmd + ["--output-format", "json"]
                return list(cmd)

        # Last resort: echo the prompt via Python
        log.warning("No CLI tool found (claude, codex, gemini). "
                    "Falling back to echo.")
        return [sys.executable, "-c",
                "import sys; print(sys.stdin.read())"]

    def _command_exists(self, cmd: str) -> bool:
        """Check whether a CLI command is on the PATH."""
        try:
            locator = "where" if os.name == "nt" else "which"
            result = subprocess.run(
                [locator, cmd],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _build_prompt(self, title: str, description: str) -> str:
        """Build a prompt string from task title and description."""
        parts = [f"Task: {title}"]
        if description:
            parts.append(f"\nDescription:\n{description}")
        parts.append("\nPlease complete this task. Be thorough but concise.")
        return "\n".join(parts)

    def _run_subprocess(self, cmd: List[str], prompt: str,
                        cwd: str, task_id: str,
                        timeout: int = 600) -> str:
        """Run a CLI command with the given prompt, capturing output.

        Posts checkpoint updates every 60 seconds while the process runs.
        """
        work_dir = cwd if cwd and os.path.isdir(cwd) else os.path.expanduser("~")
        env = os.environ.copy()

        # For the fallback Python echo, we pipe to stdin.
        # For real CLI tools, we append the prompt as an argument.
        is_fallback = cmd[0] == sys.executable
        if is_fallback:
            full_cmd = cmd
        else:
            full_cmd = cmd + [prompt]

        proc = subprocess.Popen(
            full_cmd,
            stdin=subprocess.PIPE if is_fallback else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=work_dir,
            env=env,
            text=True,
            errors="replace",
        )

        with self._lock:
            self._current_process = proc

        output_lines: List[str] = []
        checkpoint_interval = 60  # seconds
        last_checkpoint = time.time()

        try:
            # Send prompt to stdin for fallback mode
            if is_fallback and proc.stdin:
                proc.stdin.write(prompt)
                proc.stdin.close()

            # Stream stdout line by line
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    output_lines.append(line)

                    # Periodic checkpoint
                    now = time.time()
                    if now - last_checkpoint > checkpoint_interval:
                        self._checkpoint_task(task_id, {
                            "phase": "EXECUTING",
                            "partial_output": ''.join(output_lines)[:50_000],
                            "lines": len(output_lines),
                        })
                        last_checkpoint = now

            proc.wait(timeout=timeout)

        except subprocess.TimeoutExpired:
            proc.kill()
            raise

        return ''.join(output_lines)


# ---- Module-level singleton ----

_daemon_instance: Optional[AgentDaemon] = None


def get_daemon() -> AgentDaemon:
    """Return the singleton AgentDaemon instance (creates one if needed)."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = AgentDaemon()
    return _daemon_instance


def start_daemon(bridge_url: str = "http://127.0.0.1:6767",
                 poll_interval: int = 30) -> AgentDaemon:
    """Create (or reconfigure) and start the singleton daemon.

    Called from shadow_bridge_gui.py:
        daemon = start_daemon(f"http://127.0.0.1:{WEB_PORT}")
    """
    global _daemon_instance
    daemon = get_daemon()
    daemon.bridge_url = bridge_url.rstrip('/')
    daemon.poll_interval = poll_interval
    daemon.start()
    return daemon
