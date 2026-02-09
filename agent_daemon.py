"""
Agent Daemon — persistent background loop that polls ShadowBridge API for
pending tasks and executes them via CLI tools (Claude Code, Codex, Gemini CLI).

Imported by shadow_bridge_gui.py via:
    from agent_daemon import start_daemon, get_daemon
"""

import os
import re
import sys
import json
import time
import threading
import subprocess
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

log = logging.getLogger("agent_daemon")

# ---- Security: Environment Variable Allowlist ----
# Only these env vars are passed to child processes.
# Keys containing sensitive patterns are always blocked.
ENV_ALLOWLIST = {
    "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR",
    "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "SHELL",
    "LANG", "LC_ALL", "LC_CTYPE", "TERM",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "APPDATA", "LOCALAPPDATA",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
    "PYTHONPATH", "NODE_PATH", "GOPATH", "CARGO_HOME", "RUSTUP_HOME",
}
ENV_BLOCKED_PATTERNS = re.compile(
    r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH|PRIVATE)",
    re.IGNORECASE,
)

# ---- Security: Path Sandboxing ----
DEFAULT_ALLOWED_ROOTS = [
    str(Path.home()),
    "C:\\shadow",
    "/c/shadow",
]

# ---- Safety: Command Blocklist ----
BLOCKED_COMMAND_PATTERNS = [
    r"rm\s+-rf\s+/\s*$",
    r"rm\s+-rf\s+/\w",        # rm -rf /etc, /usr, etc.
    r"format\s+c:",
    r"mkfs\b",
    r"dd\s+if=",
    r":\(\)\s*\{\s*:\|:\s*&\s*\}",  # fork bomb
    r"chmod\s+777\s+/\s*$",
    r"\|\s*sh\b",                     # pipe to shell
    r"\|\s*bash\b",
    r"curl\s+.*\|\s*(sh|bash)",       # curl | sh
    r"wget\s+.*\|\s*(sh|bash)",
]
_blocked_re = [re.compile(p, re.IGNORECASE) for p in BLOCKED_COMMAND_PATTERNS]


def _filter_env() -> Dict[str, str]:
    """Return a filtered copy of os.environ with only allowed variables."""
    filtered = {}
    for key, value in os.environ.items():
        if key.upper() in ENV_ALLOWLIST and not ENV_BLOCKED_PATTERNS.search(key):
            filtered[key] = value
    return filtered


def _is_path_allowed(path: str, allowed_roots: List[str] = None) -> bool:
    """Check if a path is within allowed roots (resolves symlinks)."""
    if not path:
        return False
    roots = allowed_roots or DEFAULT_ALLOWED_ROOTS
    try:
        real = os.path.realpath(path)
        for root in roots:
            root_real = os.path.realpath(root)
            if real.startswith(root_real):
                return True
    except Exception:
        pass
    return False


def _is_blocked_command(prompt: str) -> Optional[str]:
    """Check if a prompt contains blocked command patterns. Returns match or None."""
    for pattern in _blocked_re:
        match = pattern.search(prompt)
        if match:
            return match.group(0)
    return None


class AgentDaemon:
    """Persistent background daemon that polls for tasks and executes them
    via CLI tools (Claude Code, Codex, Gemini CLI)."""

    MAX_CONCURRENT_TASKS = 3  # Worker pool size for parallel task execution

    def __init__(self, bridge_url: str = "http://127.0.0.1:6767",
                 poll_interval: int = 30):
        self.bridge_url = bridge_url.rstrip('/')
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_task: Optional[Dict] = None
        self._current_process: Optional[subprocess.Popen] = None
        # Worker pool: tracks active tasks by task_id -> (thread, process)
        self._active_workers: Dict[str, Dict[str, Any]] = {}
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._started_at: Optional[float] = None
        self._lock = threading.Lock()
        self._poll_count = 0

    # ---- Properties ----

    @property
    def is_running(self) -> bool:
        return (self._running
                and self._thread is not None
                and self._thread.is_alive())

    @property
    def status(self) -> Dict[str, Any]:
        """Status dict consumed by the /api/daemon/status endpoint."""
        with self._lock:
            active_titles = [w.get("title", "?") for w in self._active_workers.values()]
        return {
            "running": self.is_running,
            "bridge_url": self.bridge_url,
            "current_task": (self._current_task.get("title")
                             if self._current_task else None),
            "active_tasks": active_titles,
            "active_task_count": len(active_titles),
            "max_concurrent": self.MAX_CONCURRENT_TASKS,
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
        """Main loop: GET /api/tasks/pending, claim up to MAX_CONCURRENT_TASKS,
        execute in parallel worker threads, post results.
        Runs in a background thread until stop() is called."""
        # Brief initial delay so the web server can finish starting
        time.sleep(5)

        while self._running:
            try:
                # Clean up finished workers
                self._cleanup_finished_workers()

                # Determine available capacity
                with self._lock:
                    available_slots = self.MAX_CONCURRENT_TASKS - len(self._active_workers)

                if available_slots > 0:
                    tasks = self._fetch_pending_tasks()
                    if tasks:
                        tasks_to_start = tasks[:available_slots]
                        for task in tasks_to_start:
                            task_id = task.get("id", "")
                            with self._lock:
                                if task_id in self._active_workers:
                                    continue  # Already running
                            self._start_worker(task)

                # Periodic autonomous work
                self._poll_count += 1
                # Every 10th poll (~5 min): routine detection + prediction-to-task
                if self._poll_count % 10 == 0:
                    self._run_routine_check()
                    self._run_prediction_to_task()
                # Every 60th poll (~30 min): health scans
                if self._poll_count % 60 == 0:
                    self._run_health_scan()
            except Exception as e:
                log.error(f"Daemon poll loop error: {e}")

            # Sleep in 1-second chunks so stop() is responsive
            for _ in range(self.poll_interval):
                if not self._running:
                    return
                time.sleep(1)

    def _start_worker(self, task: Dict):
        """Spawn a worker thread to execute a single task."""
        task_id = task.get("id", "")
        title = task.get("title", "Untitled")

        def worker():
            try:
                self._execute_task(task)
            except Exception as e:
                log.error(f"Worker error for task {task_id}: {e}")
            finally:
                with self._lock:
                    self._active_workers.pop(task_id, None)

        t = threading.Thread(target=worker, daemon=True,
                             name=f"Worker-{task_id[:8]}")
        with self._lock:
            self._active_workers[task_id] = {
                "thread": t, "title": title, "started_at": time.time()
            }
        t.start()
        log.info(f"Started worker for task: {title} ({task_id})")

    def _cleanup_finished_workers(self):
        """Remove entries for workers whose threads have finished."""
        with self._lock:
            finished = [tid for tid, w in self._active_workers.items()
                        if not w["thread"].is_alive()]
            for tid in finished:
                del self._active_workers[tid]

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

    def _heartbeat_task(self, task_id: str):
        """POST /api/tasks/{id}/heartbeat — extend lease on active task."""
        try:
            requests.post(
                f"{self.bridge_url}/api/tasks/{task_id}/heartbeat",
                timeout=10,
            )
        except Exception as e:
            log.debug(f"Failed to heartbeat task {task_id}: {e}")

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
        2. Check command safety blocklist
        3. Refresh context (git pull if applicable)
        4. Determine which CLI tool to use (claude, codex, gemini)
        5. Build the prompt from task title + description
        6. Run the subprocess with timeout
        7. Post periodic checkpoints/heartbeats and the final result
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

        # Safety: check command blocklist against prompt content
        combined_text = f"{title} {description}"
        blocked = _is_blocked_command(combined_text)
        if blocked:
            log.error(f"BLOCKED: Task {task_id} contains dangerous pattern: {blocked}")
            self._complete_task(task_id, {
                "output": f"Task blocked by safety filter: contains '{blocked}'",
                "success": False,
                "completed_at": datetime.now().isoformat(),
            })
            self._tasks_failed += 1
            return

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

            # Step 4: Validate and prepare work directory
            work_dir = self._resolve_work_dir(project_dir)

            # Step 5: Context refresh — git pull if in a repo
            self._refresh_context(work_dir)

            log.info(f"Executing via {cmd[0]} for task {task_id}")
            self._post_event(task_id, "EXECUTING",
                             f"Running {cmd[0]}...")

            # Step 6: Run
            output = self._run_subprocess(cmd, prompt, work_dir,
                                          task_id, timeout)

            # Step 7: Complete
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

    def _run_routine_check(self):
        """Execute active routines via the RoutineDetector.

        Triggered routines now auto-create tasks in TaskStore, which the
        daemon's main poll loop will pick up on the next cycle.
        """
        try:
            from web.services.routine_detector import get_routine_detector
            detector = get_routine_detector()
            triggered = detector.execute_active_routines()
            if triggered:
                log.info(f"Routine check triggered {len(triggered)} routines "
                         f"(tasks created in store)")
            # Also scan for new routine patterns every time
            detector.scan_for_routines()
        except Exception as e:
            log.debug(f"Routine check failed: {e}")

    def _run_health_scan(self):
        """Run code health analysis on known project directories.

        Pushes health findings as tasks to TaskStore. Called every 20th
        poll cycle (~10 minutes at default 30s interval).
        """
        try:
            from web.services.code_health_monitor import get_code_health_monitor
            monitor = get_code_health_monitor()

            # Scan known project directories
            project_dirs = self._discover_project_dirs()
            total_tasks = 0
            for project_id, project_dir in project_dirs.items():
                try:
                    created = monitor.push_health_tasks_to_store(
                        project_id, project_dir)
                    total_tasks += len(created)
                except Exception as e:
                    log.debug(f"Health scan failed for {project_id}: {e}")

            if total_tasks > 0:
                log.info(f"Health scan created {total_tasks} tasks "
                         f"across {len(project_dirs)} projects")
        except Exception as e:
            log.debug(f"Health scan failed: {e}")

    def _discover_project_dirs(self) -> Dict[str, str]:
        """Discover project directories from TaskStore's known projects
        and well-known paths."""
        projects = {}

        # Always include the main shadow repos
        shadow_root = os.path.realpath("C:\\shadow")
        for name in ("shadow-android", "shadow-bridge"):
            path = os.path.join(shadow_root, name)
            if os.path.isdir(path):
                projects[name] = path

        # Check TaskStore for project_dir references in recent tasks
        try:
            from web.services.task_store import get_task_store
            store = get_task_store()
            all_tasks = store.list_tasks()[:100]
            for task in all_tasks:
                inp = task.get("input", {})
                pdir = inp.get("project_dir") or inp.get("working_directory", "")
                pid = inp.get("project_id", "")
                if pdir and os.path.isdir(pdir) and pid:
                    projects[pid] = pdir
        except Exception:
            pass

        return projects

    def _run_prediction_to_task(self):
        """Convert high-confidence predictions to tasks proactively.

        Runs server-side so tasks are created even when no Android device
        is connected. Deduplicates against existing pending tasks.
        """
        try:
            from web.services.predictive_engine import get_predictive_engine
            from web.services.task_store import get_task_store
            engine = get_predictive_engine()
            store = get_task_store()

            predictions = engine.get_predictions(limit=5, min_confidence=0.8)
            if not predictions:
                return

            # Check existing tasks for dedup
            existing = store.list_tasks(status="PENDING")
            existing_pred_ids = set()
            for t in existing:
                inp = t.get("input", {})
                pid = inp.get("prediction_id")
                if pid:
                    existing_pred_ids.add(pid)

            created = 0
            for pred in predictions:
                pred_id = pred.get("id", "")
                if pred_id in existing_pred_ids:
                    continue
                if pred.get("outcome"):  # Already resolved
                    continue

                action = pred.get("predicted_action", "Unknown action")
                confidence = pred.get("confidence", 0)
                context = pred.get("context", {})

                task_data = {
                    "title": f"Predicted: {action}",
                    "description": (
                        f"High-confidence prediction ({confidence:.0%}): {action}\n"
                        f"Signal: {pred.get('signal_type', 'unknown')}\n"
                        f"Context: {json.dumps(context, default=str)[:500]}"
                    ),
                    "priority": "HIGH" if confidence >= 0.9 else "NORMAL",
                    "tags": ["predictive", f"confidence:{confidence:.2f}"],
                    "input": {
                        "source": "predictive_engine",
                        "prediction_id": pred_id,
                        "predicted_action": action,
                        "confidence": confidence,
                        "project_id": context.get("project_id", ""),
                    },
                    "created_by": "predictive_engine",
                }
                store.create_task(task_data)
                created += 1

            if created:
                log.info(f"Created {created} tasks from predictions")
        except Exception as e:
            log.debug(f"Prediction-to-task failed: {e}")

    def _resolve_work_dir(self, project_dir: str) -> str:
        """Validate project directory against allowed roots. Falls back to home."""
        if project_dir and os.path.isdir(project_dir) and _is_path_allowed(project_dir):
            return project_dir
        if project_dir:
            log.warning(f"Project dir '{project_dir}' outside allowed roots, using home")
        return os.path.expanduser("~")

    def _refresh_context(self, work_dir: str):
        """Run git pull --ff-only if work_dir is a git repo. Non-blocking."""
        git_dir = os.path.join(work_dir, ".git")
        if not os.path.isdir(git_dir):
            return
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                log.info(f"Context refreshed: git pull in {work_dir}")
            else:
                log.debug(f"git pull non-zero exit in {work_dir}: {result.stderr[:200]}")
        except Exception as e:
            log.debug(f"Context refresh failed in {work_dir}: {e}")

    def _resolve_cli_command(self, preferred: str) -> List[str]:
        """Build the CLI command list for the preferred tool.

        Supported tools:
          - claude  -> claude -p "<prompt>" --output-format json --allowedTools ...
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
                if tool == "claude":
                    return cmd + [
                        "--output-format", "json",
                        "--allowedTools",
                        "Edit,Write,Bash,Read,Glob,Grep",
                    ]
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

        Posts checkpoint updates and heartbeats every 60 seconds.
        Uses filtered environment variables for security.
        """
        env = _filter_env()

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
            cwd=cwd,
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

                    # Periodic checkpoint + heartbeat
                    now = time.time()
                    if now - last_checkpoint > checkpoint_interval:
                        self._checkpoint_task(task_id, {
                            "phase": "EXECUTING",
                            "partial_output": ''.join(output_lines)[:50_000],
                            "lines": len(output_lines),
                        })
                        self._heartbeat_task(task_id)
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
