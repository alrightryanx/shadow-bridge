"""
Agent Orchestrator Service

Manages persistent AI agent lifecycle for real-time orchestration.
Supports spawning, monitoring, task assignment, and stopping agents.
"""

import subprocess
import shutil
import threading
import queue
import os
import time
import uuid
import json
import re
import psutil
import concurrent.futures
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import defaultdict

from .lock_manager import get_lock_manager
from .subagent_loop import get_subagent_loop_controller
from .state_store import get_state_store

logger = logging.getLogger(__name__)

# Active agents (agent_id -> agent info)
active_agents: Dict[str, Dict] = {}

# Agent output logs (agent_id -> list of output lines)
agent_logs: Dict[str, List[Dict]] = defaultdict(list)

# Task queues (agent_id -> task queue)
agent_task_queues: Dict[str, queue.Queue] = {}

# Output broadcast callbacks (for WebSocket streaming)
output_callbacks: Dict[str, List[Callable]] = defaultdict(list)

# Task completion tracking
agent_task_events: Dict[str, threading.Event] = {}  # agent_id -> completion event
agent_task_states: Dict[str, Dict] = {}  # agent_id -> {task, status, started_at}

# Crash recovery tracking
agent_restart_counts: Dict[str, List[float]] = defaultdict(list)  # agent_id -> [timestamps]
agent_watchdog_thread: Optional[threading.Thread] = None
agent_watchdog_stop_event = threading.Event()

# Recovery config (all env-configurable for scaling)
MAX_RESTARTS_PER_HOUR = int(os.environ.get("AGENT_MAX_RESTARTS_PER_HOUR", "10"))
RESTART_BACKOFF_SECONDS = [1, 2, 4, 8, 16, 30, 60]
WATCHDOG_CHECK_INTERVAL = int(os.environ.get("AGENT_WATCHDOG_INTERVAL", "5"))

# Agent data file path
AGENTS_DATA_FILE = os.path.join(os.path.expanduser("~"), ".shadowai", "agents.json")
TASK_MARKER_PATTERN = re.compile(r'^<<<(TASK_STARTED|TASK_PROGRESS|TASK_RESULT|TASK_DONE)\s+(\{.*\})>>>$')
TASK_LOG_TAIL_LINES = 50
TASK_ID_LENGTH = 12
TASK_TIMEOUT_SECONDS = int(os.environ.get("AGENT_TASK_TIMEOUT", "1800"))  # 30 min default

# Resource limits for spawning
RESOURCE_CPU_LIMIT = float(os.environ.get("AGENT_CPU_LIMIT", "95"))  # percent
RESOURCE_RAM_LIMIT = float(os.environ.get("AGENT_RAM_LIMIT", "90"))  # percent
RESOURCE_DISK_MIN_GB = float(os.environ.get("AGENT_DISK_MIN_GB", "1.0"))  # GB free

# Output log limits
MAX_LOG_LINES_PER_AGENT = int(os.environ.get("AGENT_MAX_LOG_LINES", "1000"))
STATUS_FETCH_POOL_SIZE = int(os.environ.get("AGENT_STATUS_POOL_SIZE", "50"))

# WebSocket output batching config
OUTPUT_BATCH_INTERVAL = float(os.environ.get("AGENT_OUTPUT_BATCH_INTERVAL", "1.0"))
OUTPUT_BATCH_MAX_LINES = int(os.environ.get("AGENT_OUTPUT_BATCH_MAX_LINES", "20"))

# Output batch buffer (agent_id -> list of recent lines)
_output_batch: Dict[str, List[Dict]] = defaultdict(list)
_output_batch_lock = threading.Lock()
_output_batch_thread: Optional[threading.Thread] = None


def _record_task_block(entry: Dict):
    """Record a task block event to SQLite for crash-proof persistence."""
    try:
        store = get_state_store()
        store.record_task_block(
            agent_id=entry.get("agent_id", ""),
            task_id=entry.get("task_id", ""),
            thread_id=entry.get("thread_id", ""),
            repo=entry.get("repo"),
            lock_paths=entry.get("lock_paths", []),
            reason=entry.get("reason", ""),
        )
    except Exception as e:
        logger.error(f"Failed to record task block: {e}")


def get_task_blocks() -> List[Dict]:
    """Get recent task block events from SQLite."""
    try:
        store = get_state_store()
        return store.get_task_blocks(limit=50)
    except Exception as e:
        logger.error(f"Failed to get task blocks: {e}")
        return []


def _build_protocol_header(task_id: str, thread_id: str) -> str:
    """Build the strict task completion protocol header for agents."""
    started_payload = json.dumps({"task_id": task_id, "thread_id": thread_id}, separators=(",", ":"))
    progress_payload = json.dumps({"pct": 30, "msg": "..."}, separators=(",", ":"))
    result_payload = json.dumps(
        {
            "status": "success|fail|blocked",
            "summary": "...",
            "files_changed": [],
            "commands_run": [],
            "next_steps": []
        },
        separators=(",", ":")
    )
    done_payload = json.dumps({"status": "success|fail", "elapsed_ms": 12345}, separators=(",", ":"))

    return (
        "PROTOCOL REQUIRED: print these markers as standalone lines exactly.\n"
        f"<<<TASK_STARTED {started_payload}>>>\n"
        f"<<<TASK_PROGRESS {progress_payload}>>>\n"
        f"<<<TASK_RESULT {result_payload}>>>\n"
        f"<<<TASK_DONE {done_payload}>>>\n"
        "TASK_PROGRESS is optional."
    )


def _ensure_protocol_header(task_text: str, task_id: str, thread_id: str) -> str:
    """Ensure the task prompt includes the strict protocol header."""
    if "<<<TASK_STARTED" in task_text:
        return task_text
    header = _build_protocol_header(task_id, thread_id)
    return f"{header}\n\n{task_text}"


def check_system_resources() -> Dict:
    """
    Check if the system has enough resources to spawn another agent.

    Returns:
        Dict with 'can_spawn' bool, 'reason' str, and 'metrics' dict.
    """
    metrics = {}
    try:
        mem = psutil.virtual_memory()
        metrics["ram_percent"] = round(mem.percent, 1)
        metrics["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)

        cpu = psutil.cpu_percent(interval=0.5)
        metrics["cpu_percent"] = round(cpu, 1)

        disk = psutil.disk_usage(os.path.expanduser("~"))
        metrics["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)

        metrics["active_agents"] = len(active_agents)
        metrics["total_threads"] = threading.active_count()
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return {"can_spawn": True, "reason": "", "metrics": metrics}

    if metrics["cpu_percent"] > RESOURCE_CPU_LIMIT:
        return {
            "can_spawn": False,
            "reason": f"CPU at {metrics['cpu_percent']}% (limit {RESOURCE_CPU_LIMIT}%)",
            "metrics": metrics,
        }

    if metrics["ram_percent"] > RESOURCE_RAM_LIMIT:
        return {
            "can_spawn": False,
            "reason": f"RAM at {metrics['ram_percent']}% (limit {RESOURCE_RAM_LIMIT}%)",
            "metrics": metrics,
        }

    if metrics["disk_free_gb"] < RESOURCE_DISK_MIN_GB:
        return {
            "can_spawn": False,
            "reason": f"Disk free {metrics['disk_free_gb']}GB (min {RESOURCE_DISK_MIN_GB}GB)",
            "metrics": metrics,
        }

    return {"can_spawn": True, "reason": "", "metrics": metrics}


def spawn_agent(
    device_id: str,
    name: str,
    specialty: str,
    cli_provider: str,
    model: str,
    working_directory: Optional[str] = None,
    auto_accept_edits: bool = True,
    session_id: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    skip_resource_check: bool = False
) -> Dict:
    """
    Spawn a new persistent AI agent.

    Args:
        device_id: Device that spawned this agent
        name: Human-readable agent name
        specialty: Agent specialty (code_review, testing, documentation, etc.)
        cli_provider: CLI tool to use (claude, gemini, codex)
        model: Model ID
        working_directory: Working directory for agent
        auto_accept_edits: Whether to auto-accept edits
        session_id: Optional session ID link
        env: Optional environment variables for the agent process

    Returns:
        Agent info dict with id, status, etc.
    """

    # Resource check before spawning
    if not skip_resource_check:
        resource_status = check_system_resources()
        if not resource_status["can_spawn"]:
            raise RuntimeError(
                f"Cannot spawn agent: {resource_status['reason']}. "
                f"Active agents: {resource_status['metrics'].get('active_agents', '?')}"
            )

    agent_id = str(uuid.uuid4())

    # Build CLI command for persistent session
    cmd = build_agent_command(
        cli_provider=cli_provider,
        model=model,
        auto_accept_edits=auto_accept_edits,
        specialty=specialty
    )

    if not cmd:
        raise ValueError(f"Unknown CLI provider: {cli_provider}")

    # Set working directory
    cwd = working_directory or os.path.expanduser("~")

    # Prepare environment
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    logger.info(f"Spawning agent {agent_id}: {name} ({specialty}) using {model}")
    logger.debug(f"Command: {cmd}")

    # Start subprocess
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=cwd,
            env=process_env,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Create agent info
        agent_info = {
            "id": agent_id,
            "device_id": device_id,
            "name": name,
            "specialty": specialty,
            "cli_provider": cli_provider,
            "model": model,
            "status": "idle",
            "current_task": None,
            "current_task_id": None,
            "current_thread_id": None,
            "tasks_completed": 0,
            "spawned_at": datetime.utcnow().isoformat(),
            "working_directory": cwd,
            "process_id": process.pid,
            "auto_accept_edits": auto_accept_edits,
            "session_id": session_id  # Link to session for chat integration
        }

        # Store agent info
        active_agents[agent_id] = {
            **agent_info,
            "process": process
        }

        # Create task queue
        agent_task_queues[agent_id] = queue.Queue()

        # Start output monitoring: spawn stdout/stderr readers directly
        # instead of a wrapper thread that creates 2 sub-threads.
        # This reduces per-agent threads from 4 to 3.
        stdout_thread = threading.Thread(
            target=_read_agent_stream,
            args=(agent_id, process.stdout, "stdout", process),
            daemon=True,
            name=f"agent-stdout-{agent_id[:8]}"
        )
        stderr_thread = threading.Thread(
            target=_read_agent_stream,
            args=(agent_id, process.stderr, "stderr", process),
            daemon=True,
            name=f"agent-stderr-{agent_id[:8]}"
        )
        stdout_thread.start()
        stderr_thread.start()

        # Start task processing thread
        task_thread = threading.Thread(
            target=process_agent_tasks,
            args=(agent_id,),
            daemon=True,
            name=f"agent-task-{agent_id[:8]}"
        )
        task_thread.start()

        # Save agents data
        save_agents_data()

        # Quick check: did the process die immediately? (bad CLI, missing tool, etc.)
        time.sleep(0.3)
        if process.poll() is not None:
            exit_code = process.poll()
            stderr_out = ""
            try:
                stderr_out = process.stderr.read() if process.stderr else ""
            except Exception:
                pass
            # Cleanup
            active_agents.pop(agent_id, None)
            agent_task_queues.pop(agent_id, None)
            raise RuntimeError(
                f"Agent process died immediately (exit code {exit_code}). "
                f"stderr: {stderr_out[:500]}"
            )

        # Broadcast agent spawned event
        broadcast_agent_event("agent_spawned", agent_info)

        logger.info(f"Agent {agent_id} spawned successfully (PID: {process.pid})")

        return agent_info

    except Exception as e:
        logger.error(f"Failed to spawn agent: {e}")
        raise


def assign_task(
    agent_id: str,
    task: str,
    task_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    repo: Optional[str] = None,
    lock_paths: Optional[List[str]] = None,
    workspace_root: Optional[str] = None
) -> bool:
    """
    Assign a task to a running agent.

    Args:
        agent_id: Agent ID
        task: Task description
        task_id: Optional external task ID for correlation
        thread_id: Optional thread/session ID for correlation
        repo: Optional repo name for lock routing
        lock_paths: Optional list of file/dir paths to lock
        workspace_root: Optional workspace path for task isolation

    Returns:
        True if task was assigned successfully
    """

    if agent_id not in active_agents:
        logger.warning(f"Agent {agent_id} not found")
        return False

    agent = active_agents[agent_id]

    # Add task to queue
    task_queue = agent_task_queues.get(agent_id)
    if not task_queue:
        logger.error(f"No task queue for agent {agent_id}")
        return False

    assigned_task_id = task_id or str(uuid.uuid4())[:TASK_ID_LENGTH]
    assigned_thread_id = thread_id or agent_id
    task_with_protocol = _ensure_protocol_header(task, assigned_task_id, assigned_thread_id)

    lock_keys: List[str] = []
    if repo or lock_paths:
        lock_manager = get_lock_manager()
        acquired, reason, keys = lock_manager.acquire(repo, lock_paths, agent_id, assigned_task_id)
        if not acquired:
            now = datetime.utcnow().isoformat()
            agent["last_blocked_reason"] = reason
            agent["last_blocked_at"] = now
            _record_task_block({
                "agent_id": agent_id,
                "task_id": assigned_task_id,
                "thread_id": assigned_thread_id,
                "repo": repo,
                "lock_paths": lock_paths or [],
                "reason": reason,
                "timestamp": now
            })
            broadcast_agent_event("agent_task_blocked", {
                "agent_id": agent_id,
                "task_id": assigned_task_id,
                "thread_id": assigned_thread_id,
                "repo": repo,
                "lock_paths": lock_paths or [],
                "reason": reason,
                "timestamp": now
            })
            logger.info(f"Task blocked for agent {agent_id}: {reason}")
            return False
        lock_keys = keys

    task_data = {
        "task": task_with_protocol,
        "display_task": task,
        "task_id": assigned_task_id,
        "thread_id": assigned_thread_id,
        "repo": repo,
        "lock_paths": lock_paths or [],
        "lock_keys": lock_keys,
        "workspace_root": workspace_root,
        "assigned_at": datetime.utcnow().isoformat()
    }

    task_queue.put(task_data)

    # Update agent status
    agent["status"] = "busy"
    agent["current_task"] = task
    agent["current_task_id"] = assigned_task_id
    agent["current_thread_id"] = assigned_thread_id
    if workspace_root:
        agent["current_workspace"] = workspace_root

    # Broadcast status change
    broadcast_agent_event("agent_status_changed", get_agent_status(agent_id))

    logger.info(f"Assigned task to agent {agent_id}: {task}")

    return True


def get_agent_status(agent_id: str) -> Optional[Dict]:
    """
    Get real-time status of an agent.

    Args:
        agent_id: Agent ID

    Returns:
        Agent status dict with metrics, or None if not found
    """

    if agent_id not in active_agents:
        return None

    agent = active_agents[agent_id]
    process = agent.get("process")

    # Get process metrics
    cpu_percent = 0.0
    memory_mb = 0.0
    uptime_seconds = 0

    if process and process.poll() is None:
        try:
            proc = psutil.Process(process.pid)
            # Use interval=None to prevent blocking the main thread.
            # The first call may return 0.0, but subsequent calls will be accurate and non-blocking.
            cpu_percent = proc.cpu_percent(interval=None)
            memory_mb = proc.memory_info().rss / 1024 / 1024
            uptime_seconds = int(time.time() - proc.create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Get recent output lines
    output_lines = agent_logs.get(agent_id, [])[-10:]  # Last 10 lines

    # Task progress/result state (for UI)
    state = agent_task_states.get(agent_id, {})
    progress_pct = state.get("progress_pct")
    progress_msg = state.get("progress_msg")
    result = state.get("result")
    workspace_root = agent.get("current_workspace") or state.get("workspace_root")

    return {
        "id": agent["id"],
        "device_id": agent["device_id"],
        "name": agent["name"],
        "specialty": agent["specialty"],
        "cli_provider": agent["cli_provider"],
        "model": agent["model"],
        "status": agent["status"],
        "current_task": agent.get("current_task"),
        "current_task_id": agent.get("current_task_id"),
        "current_thread_id": agent.get("current_thread_id"),
        "current_workspace": workspace_root,
        "tasks_completed": agent.get("tasks_completed", 0),
        "last_blocked_reason": agent.get("last_blocked_reason"),
        "last_blocked_at": agent.get("last_blocked_at"),
        "spawned_at": agent["spawned_at"],
        "working_directory": agent["working_directory"],
        "process_id": agent["process_id"],
        "cpu_percent": round(cpu_percent, 1),
        "memory_mb": round(memory_mb, 1),
        "uptime_seconds": uptime_seconds,
        "output_lines": output_lines,
        "task_progress_pct": progress_pct,
        "task_progress_msg": progress_msg,
        "task_result": result
    }


def stop_agent(agent_id: str, graceful: bool = True) -> bool:
    """
    Stop a running agent.

    Args:
        agent_id: Agent ID
        graceful: If True, send SIGTERM and wait; if False, kill immediately

    Returns:
        True if agent was stopped successfully
    """

    if agent_id not in active_agents:
        logger.warning(f"Agent {agent_id} not found")
        return False

    agent = active_agents[agent_id]
    agent['intentional_stop'] = True  # Prevent automatic restart
    process = agent.get("process")

    if not process:
        logger.warning(f"No process found for agent {agent_id}")
        return False

    logger.info(f"Stopping agent {agent_id} (graceful={graceful})")

    try:
        if graceful:
            # Send quit command to stdin first
            try:
                process.stdin.write("quit\n")
                process.stdin.flush()
            except:
                pass

            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                logger.info(f"Agent {agent_id} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Agent {agent_id} did not stop gracefully, terminating")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Agent {agent_id} did not terminate, killing")
                    process.kill()
        else:
            # Kill immediately
            process.kill()
            process.wait(timeout=5)
            logger.info(f"Agent {agent_id} killed")

    except Exception as e:
        logger.error(f"Error stopping agent {agent_id}: {e}")
        return False

    state = agent_task_states.get(agent_id, {})
    if agent.get("current_task") and not state.get("done_at"):
        tail_logs = [entry["line"] for entry in agent_logs.get(agent_id, [])[-TASK_LOG_TAIL_LINES:]]
        exit_code = process.poll() if process else None
        _finalize_task(
            agent_id,
            "fail",
            reason="agent_stopped",
            exit_code=exit_code,
            tail_logs=tail_logs
        )

    # Update status
    agent["status"] = "offline"

    try:
        get_lock_manager().release(agent_id)
    except Exception as exc:
        logger.error(f"Failed to release locks for stopped agent {agent_id}: {exc}")

    # Broadcast event
    broadcast_agent_event("agent_stopped", {"id": agent_id})

    # Cleanup
    del active_agents[agent_id]
    if agent_id in agent_task_queues:
        del agent_task_queues[agent_id]
    if agent_id in agent_logs:
        del agent_logs[agent_id]
    if agent_id in output_callbacks:
        del output_callbacks[agent_id]

    # Save agents data
    save_agents_data()

    return True


def stop_all_agents(graceful: bool = True) -> int:
    """Stop all running agents. Returns count stopped."""
    stopped = 0
    for agent_id in list(active_agents.keys()):
        if stop_agent(agent_id, graceful=graceful):
            stopped += 1
    return stopped


def get_all_agents(device_id: Optional[str] = None) -> List[Dict]:
    """
    Get all active agents, optionally filtered by device.
    Uses parallel fetching for improved dashboard responsiveness.

    Args:
        device_id: Optional device ID to filter by

    Returns:
        List of agent status dicts
    """

    agent_ids = list(active_agents.keys())
    if not agent_ids:
        return []

    agents = []
    
    # Use ThreadPoolExecutor to fetch statuses in parallel
    # This protects against any potential blocking I/O in get_agent_status
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(agent_ids), STATUS_FETCH_POOL_SIZE)) as executor:
        # Create a mapping of future to agent_id
        future_to_id = {executor.submit(get_agent_status, aid): aid for aid in agent_ids}
        
        for future in concurrent.futures.as_completed(future_to_id):
            try:
                status = future.result()
                if status:
                    if device_id is None or status.get("device_id") == device_id:
                        agents.append(status)
            except Exception as e:
                logger.error(f"Error fetching status for agent {future_to_id[future]}: {e}")

    # Sort by name for consistent UI display
    agents.sort(key=lambda x: x.get("name", ""))
    return agents


def parse_task_marker(line: str) -> Optional[Dict]:
    """Parse task completion markers from agent output."""
    match = TASK_MARKER_PATTERN.match(line)
    if not match:
        return None

    marker_type = match.group(1)
    payload_raw = match.group(2)
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
        logger.warning(f"Invalid task marker JSON for {marker_type}: {exc}")
        return {"type": "invalid", "marker": marker_type, "raw": payload_raw}

    return {"type": marker_type, "payload": payload}


def _finalize_task(
    agent_id: str,
    status: str,
    reason: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    exit_code: Optional[int] = None,
    tail_logs: Optional[List[str]] = None
):
    """Finalize a task when TASK_DONE is observed or the subprocess exits."""
    normalized_status = status if status in ("success", "fail") else "fail"
    now = datetime.utcnow().isoformat()

    state = agent_task_states.setdefault(agent_id, {})
    if state.get("done_at"):
        return

    state["done_at"] = now
    state["done_status"] = normalized_status
    if elapsed_ms is not None:
        state["elapsed_ms"] = elapsed_ms
    if reason:
        state["done_reason"] = reason
    if exit_code is not None:
        state["exit_code"] = exit_code
    if tail_logs:
        state["tail_logs"] = tail_logs

    try:
        released = get_lock_manager().release(agent_id, task_id=state.get("task_id"))
        if released:
            state["locks_released"] = released
    except Exception as exc:
        logger.error(f"Failed to release locks for agent {agent_id}: {exc}")

    if agent_id not in agent_task_events:
        agent_task_events[agent_id] = threading.Event()
    agent_task_events[agent_id].set()

    agent = active_agents.get(agent_id)
    if agent:
        agent["status"] = "idle"
        agent["current_task"] = None
        agent["current_task_id"] = None
        agent["current_thread_id"] = None
        agent["current_workspace"] = None

    event_payload = {
        "agent_id": agent_id,
        "task": state.get("task"),
        "task_id": state.get("task_id"),
        "thread_id": state.get("thread_id"),
        "status": normalized_status,
        "completed_at": now,
        "result": state.get("result"),
        "reason": reason,
        "exit_code": exit_code,
        "tail_logs": tail_logs
    }

    if normalized_status == "success":
        if agent:
            agent["tasks_completed"] = agent.get("tasks_completed", 0) + 1
    else:
        broadcast_agent_event("agent_task_failed", event_payload)

    broadcast_agent_event("agent_task_completed", event_payload)

    # Record performance to SQLite for smart routing
    task_duration = elapsed_ms or 0
    if not task_duration and state.get("started_at"):
        try:
            started_ts = datetime.fromisoformat(state["started_at"]).timestamp()
            task_duration = int((time.time() - started_ts) * 1000)
        except (ValueError, TypeError):
            pass
    record_task_performance(
        agent_id=agent_id,
        task_id=state.get("task_id", ""),
        category=agent.get("specialty", "general") if agent else "general",
        success=(normalized_status == "success"),
        duration_ms=task_duration,
        repo=state.get("repo", ""),
    )

    if agent:
        broadcast_agent_event("agent_status_changed", get_agent_status(agent_id))


def update_task_status(agent_id: str, marker: Dict):
    """Update task status and signal completion event."""
    if marker.get("type") == "invalid":
        return

    marker_type = marker.get("type")
    payload = marker.get("payload", {})
    now = datetime.utcnow().isoformat()

    state = agent_task_states.setdefault(agent_id, {})
    state["last_marker_at"] = now

    if marker_type == "TASK_STARTED":
        state["task_id"] = payload.get("task_id") or state.get("task_id")
        state["thread_id"] = payload.get("thread_id") or state.get("thread_id")
        state["status"] = "running"
        state["started_at"] = state.get("started_at") or now
        broadcast_agent_event("agent_task_started", {
            "agent_id": agent_id,
            "task_id": state.get("task_id"),
            "thread_id": state.get("thread_id"),
            "started_at": state.get("started_at")
        })
        return

    if marker_type == "TASK_PROGRESS":
        state["progress_pct"] = payload.get("pct")
        state["progress_msg"] = payload.get("msg")
        broadcast_agent_event("agent_task_progress", {
            "agent_id": agent_id,
            "task_id": state.get("task_id"),
            "thread_id": state.get("thread_id"),
            "pct": payload.get("pct"),
            "msg": payload.get("msg"),
            "timestamp": now
        })
        return

    if marker_type == "TASK_RESULT":
        state["result"] = payload
        state["result_status"] = payload.get("status")
        broadcast_agent_event("agent_task_result", {
            "agent_id": agent_id,
            "task_id": state.get("task_id"),
            "thread_id": state.get("thread_id"),
            "result": payload,
            "timestamp": now
        })
        return

    if marker_type == "TASK_DONE":
        done_status = payload.get("status") or "fail"
        elapsed_ms = payload.get("elapsed_ms")
        _finalize_task(agent_id, done_status, elapsed_ms=elapsed_ms)


def _read_agent_stream(agent_id: str, stream, stream_type: str, process=None):
    """
    Read a single output stream (stdout or stderr) from an agent process.
    Runs as a daemon thread per stream. This replaces the old
    monitor_agent_output wrapper which spawned sub-threads.

    Args:
        agent_id: Agent ID
        stream: stdout or stderr stream
        stream_type: "stdout" or "stderr"
        process: subprocess.Popen (for exit detection on stream EOF)
    """
    try:
        for line in stream:
            if not line:
                continue

            line = line.strip()

            # Create log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "stream": stream_type,
                "line": line
            }

            # Add to logs
            agent_logs[agent_id].append(log_entry)

            # Trim logs (amortized: only trim when 50% over limit)
            log_list = agent_logs[agent_id]
            if len(log_list) > MAX_LOG_LINES_PER_AGENT * 1.5:
                agent_logs[agent_id] = log_list[-MAX_LOG_LINES_PER_AGENT:]

            # Parse task markers
            parsed_marker = parse_task_marker(line)
            if parsed_marker:
                update_task_status(agent_id, parsed_marker)

            # Parse subagent loop markers
            try:
                loop_controller = get_subagent_loop_controller()
                loop_event = loop_controller.detect_loop_trigger(line)
                if loop_event:
                    task_id = active_agents.get(agent_id, {}).get("current_task_id")
                    loop_controller.handle_loop_event(agent_id, loop_event, task_id=task_id)
            except Exception as e:
                logger.debug(f"Error handling loop marker: {e}")

            # Queue output for batched WebSocket broadcast (reduces flood at 100+ agents)
            with _output_batch_lock:
                _output_batch[agent_id].append({
                    "line": line,
                    "stream": stream_type,
                    "timestamp": log_entry["timestamp"]
                })

            # Send to session if linked
            agent = active_agents.get(agent_id)
            if agent and agent.get("session_id"):
                try:
                    import requests
                    session_id = agent["session_id"]
                    agent_name = agent.get("name", "AI Agent")
                    requests.post(
                        f"http://localhost:6767/api/sessions/{session_id}/agent-message",
                        json={
                            "agent_id": agent_id,
                            "agent_name": agent_name,
                            "content": line,
                            "type": "error" if stream_type == "stderr" else "output"
                        },
                        timeout=2
                    )
                except Exception as e:
                    logger.debug(f"Failed to post agent output to session: {e}")

    except Exception as e:
        logger.error(f"Error reading {stream_type} for agent {agent_id}: {e}")

    # On stream EOF: if this is stdout, check for unfinished tasks (process exited)
    # Only stdout does this to avoid double-finalization from both threads
    if stream_type == "stdout" and process:
        exit_code = process.poll()
        agent = active_agents.get(agent_id)
        state = agent_task_states.get(agent_id, {})
        if exit_code is not None and not state.get("done_at"):
            has_active_task = state or (agent and agent.get("current_task"))
            if has_active_task:
                if not state and agent and agent.get("current_task"):
                    agent_task_states[agent_id] = {
                        "task": agent.get("current_task"),
                        "task_id": agent.get("current_task_id"),
                        "thread_id": agent.get("current_thread_id"),
                        "status": "running",
                        "started_at": datetime.utcnow().isoformat()
                    }
                tail_logs = [entry["line"] for entry in agent_logs.get(agent_id, [])[-TASK_LOG_TAIL_LINES:]]
                _finalize_task(
                    agent_id,
                    "fail",
                    reason="process_exited_without_task_done",
                    exit_code=exit_code,
                    tail_logs=tail_logs
                )

    logger.info(f"Stream {stream_type} monitoring stopped for agent {agent_id}")


def process_agent_tasks(agent_id: str):
    """
    Process tasks from agent's task queue.

    Args:
        agent_id: Agent ID
    """

    task_queue = agent_task_queues.get(agent_id)
    if not task_queue:
        return

    agent = active_agents.get(agent_id)
    if not agent:
        return

    process = agent.get("process")
    if not process:
        return

    logger.info(f"Task processing started for agent {agent_id}")

    while True:
        try:
            # Get next task (blocking)
            task_data = task_queue.get(timeout=1)

            if not task_data:
                continue

            task = task_data["task"]
            display_task = task_data.get("display_task", task)
            task_id = task_data.get("task_id")
            thread_id = task_data.get("thread_id")
            repo = task_data.get("repo")
            lock_paths = task_data.get("lock_paths") or []
            lock_keys = task_data.get("lock_keys") or []
            workspace_root = task_data.get("workspace_root")

            logger.info(f"Agent {agent_id} executing task: {display_task}")

            # Send task to agent's stdin
            try:
                process.stdin.write(f"{task}\n")
                process.stdin.flush()

                # Update status
                agent["status"] = "busy"
                agent["current_task"] = display_task
                agent["current_task_id"] = task_id
                agent["current_thread_id"] = thread_id

                # Initialize task tracking
                if agent_id not in agent_task_events:
                    agent_task_events[agent_id] = threading.Event()

                event = agent_task_events[agent_id]
                event.clear()

                # Store task state
                agent_task_states[agent_id] = {
                    "task": display_task,
                    "task_id": task_id,
                    "thread_id": thread_id,
                    "repo": repo,
                    "lock_paths": lock_paths,
                    "lock_keys": lock_keys,
                    "workspace_root": workspace_root,
                    "status": "running",
                    "started_at": datetime.utcnow().isoformat()
                }

                # Broadcast status change
                broadcast_agent_event("agent_status_changed", get_agent_status(agent_id))

                # Wait until TASK_DONE is observed or subprocess exits
                while True:
                    if event.wait(timeout=1):
                        break
                    if process.poll() is not None:
                        tail_logs = [entry["line"] for entry in agent_logs.get(agent_id, [])[-TASK_LOG_TAIL_LINES:]]
                        _finalize_task(
                            agent_id,
                            "fail",
                            reason="process_exited_without_task_done",
                            exit_code=process.returncode,
                            tail_logs=tail_logs
                        )
                        break
                    if agent_id not in active_agents:
                        break

                state = agent_task_states.get(agent_id, {})
                if state.get("done_status") == "success":
                    logger.info(f"Agent {agent_id} completed task: {display_task}")
                else:
                    logger.warning(f"Agent {agent_id} task ended without success: {display_task}")

                # Broadcast status change
                broadcast_agent_event("agent_status_changed", get_agent_status(agent_id))

            except Exception as e:
                logger.error(f"Error sending task to agent {agent_id}: {e}")
                agent["status"] = "error"

        except queue.Empty:
            # No tasks, check if agent is still alive
            if agent_id not in active_agents:
                logger.info(f"Agent {agent_id} no longer active, stopping task processing")
                break

            if process.poll() is not None:
                logger.warning(f"Agent {agent_id} process died, stopping task processing")
                break

        except Exception as e:
            logger.error(f"Error processing tasks for agent {agent_id}: {e}")
            break

    logger.info(f"Task processing stopped for agent {agent_id}")


def build_agent_command(
    cli_provider: str,
    model: str,
    auto_accept_edits: bool,
    specialty: str
) -> Optional[str]:
    """
    Build CLI command for persistent agent session.

    Args:
        cli_provider: CLI tool (claude, gemini, codex)
        model: Model ID
        auto_accept_edits: Auto-accept edits
        specialty: Agent specialty

    Returns:
        CLI command string
    """

    # Build system prompt based on specialty
    system_prompt = get_specialty_prompt(specialty)

    # Validate CLI tool exists before building command
    cli_name = {"claude": "claude", "gemini": "gemini", "codex": "codex"}.get(cli_provider)
    if cli_name and not shutil.which(cli_name):
        logger.error(f"CLI tool '{cli_name}' not found in PATH")
        raise FileNotFoundError(
            f"CLI tool '{cli_name}' is not installed or not in PATH. "
            f"Install it first: https://docs.anthropic.com/en/docs/claude-code"
            if cli_name == "claude" else
            f"CLI tool '{cli_name}' is not installed or not in PATH."
        )

    if cli_provider == "claude":
        cmd = f'claude'
        if model:
            cmd += f' --model {model}'
        if auto_accept_edits:
            cmd += ' --dangerously-skip-permissions'
        if system_prompt:
            cmd += f' --append-system-prompt "{system_prompt}"'
        return cmd

    elif cli_provider == "gemini":
        cmd = f'gemini'
        if model:
            cmd += f' -m {model}'
        if auto_accept_edits:
            cmd += ' -y'
        # Gemini CLI doesn't support system prompts, will send as first message
        return cmd

    elif cli_provider == "codex":
        cmd = f'codex'
        if model:
            cmd += f' --model {model}'
        if auto_accept_edits:
            cmd += ' --yolo'
        cmd += ' chat'
        return cmd

    else:
        return None


def get_specialty_prompt(specialty: str) -> str:
    """
    Get system prompt for agent specialty.

    Args:
        specialty: Agent specialty

    Returns:
        System prompt text
    """

    prompts = {
        "code_review": "You are a code review specialist. Focus on finding bugs, security issues, and code quality improvements. Be thorough and constructive.",
        "testing": "You are a testing specialist. Focus on writing comprehensive tests, identifying edge cases, and improving test coverage.",
        "documentation": "You are a documentation specialist. Focus on writing clear, comprehensive documentation, API docs, and code comments.",
        "refactoring": "You are a refactoring specialist. Focus on improving code structure, reducing complexity, and applying design patterns.",
        "debugging": "You are a debugging specialist. Focus on identifying root causes, reproducing issues, and implementing fixes.",
        "performance": "You are a performance optimization specialist. Focus on profiling, identifying bottlenecks, and optimizing code.",
        "security": "You are a security specialist. Focus on identifying vulnerabilities, implementing security best practices, and hardening code.",
        "general": "You are a helpful AI assistant specialized in software development. Be proactive and thorough."
    }

    return prompts.get(specialty, prompts["general"])


def _flush_output_batches():
    """Periodically flush batched output lines to WebSocket clients.

    Instead of broadcasting every line individually (thousands/sec at 100 agents),
    this collects lines for OUTPUT_BATCH_INTERVAL seconds and sends them as a
    single 'agent_output_batch' event per agent.
    """
    while not agent_watchdog_stop_event.wait(OUTPUT_BATCH_INTERVAL):
        with _output_batch_lock:
            if not _output_batch:
                continue
            batch_snapshot = dict(_output_batch)
            _output_batch.clear()

        for agent_id, lines in batch_snapshot.items():
            if not lines:
                continue
            # Send only the last N lines to avoid flooding
            recent = lines[-OUTPUT_BATCH_MAX_LINES:]
            broadcast_agent_event("agent_output_batch", {
                "agent_id": agent_id,
                "lines": recent,
                "total_lines": len(lines),
            })


def start_output_batch_flusher():
    """Start the background thread that flushes output batches."""
    global _output_batch_thread

    if _output_batch_thread and _output_batch_thread.is_alive():
        return

    _output_batch_thread = threading.Thread(
        target=_flush_output_batches,
        daemon=True,
        name="output-batch-flusher"
    )
    _output_batch_thread.start()
    logger.info(f"Output batch flusher started (interval={OUTPUT_BATCH_INTERVAL}s)")


def start_agent_watchdog():
    """Start background watchdog thread to monitor agent health."""
    global agent_watchdog_thread

    if agent_watchdog_thread and agent_watchdog_thread.is_alive():
        return

    agent_watchdog_thread = threading.Thread(
        target=watchdog_loop,
        daemon=True
    )
    agent_watchdog_thread.start()
    logger.info("Agent watchdog started")


_watchdog_cycle_count = 0


def watchdog_loop():
    """Monitor all agents, restart crashed ones, and sync state to SQLite."""
    global _watchdog_cycle_count

    while not agent_watchdog_stop_event.wait(WATCHDOG_CHECK_INTERVAL):
        _watchdog_cycle_count += 1

        for agent_id in list(active_agents.keys()):
            agent = active_agents.get(agent_id)
            if not agent:
                continue

            process = agent.get('process')
            if not process:
                continue

            # Check if process died
            if process.poll() is not None:
                logger.warning(f"Watchdog detected crash: agent {agent_id}")
                handle_agent_crash(agent_id)
                continue

            # Check for stuck tasks (no output for TASK_TIMEOUT_SECONDS)
            state = agent_task_states.get(agent_id, {})
            if state.get("started_at") and not state.get("done_at"):
                try:
                    started = datetime.fromisoformat(state["started_at"]).timestamp()
                except (ValueError, TypeError):
                    started = time.time()
                elapsed = time.time() - started
                if elapsed > TASK_TIMEOUT_SECONDS:
                    task_id = state.get("task_id", "unknown")
                    logger.warning(
                        f"Watchdog: agent {agent_id} task {task_id} stuck for "
                        f"{elapsed:.0f}s (limit {TASK_TIMEOUT_SECONDS}s), force-completing"
                    )
                    tail_logs = [entry["line"] for entry in agent_logs.get(agent_id, [])[-TASK_LOG_TAIL_LINES:]]
                    _finalize_task(
                        agent_id,
                        "timeout",
                        reason="task_timeout",
                        exit_code=None,
                        tail_logs=tail_logs
                    )
                    agent["status"] = "idle"
                    agent["current_task"] = None
                    agent["current_task_id"] = None
                    broadcast_agent_event("agent_task_timeout", {
                        "agent_id": agent_id,
                        "task_id": task_id,
                        "elapsed": elapsed
                    })

        # Periodic state sync to SQLite (every 6 cycles = ~30 seconds)
        if _watchdog_cycle_count % 6 == 0:
            _sync_state_to_sqlite()


def handle_agent_crash(agent_id: str):
    """Handle agent crash with cleanup and restart logic."""
    agent = active_agents.get(agent_id)
    if not agent:
        return

    # Check if intentional stop (user-initiated)
    if agent.get('intentional_stop'):
        logger.info(f"Agent {agent_id} stopped intentionally, not restarting")
        return

    # Save agent info for restart
    agent_info = {
        'device_id': agent.get('device_id'),
        'name': agent.get('name'),
        'specialty': agent.get('specialty'),
        'cli_provider': agent.get('cli_provider'),
        'model': agent.get('model'),
        'working_directory': agent.get('working_directory'),
        'auto_accept_edits': agent.get('auto_accept_edits'),
    }

    # Cleanup dead agent
    if agent.get("current_task"):
        state = agent_task_states.get(agent_id, {})
        if not state.get("done_at"):
            process = agent.get("process")
            exit_code = process.poll() if process else None
            tail_logs = [entry["line"] for entry in agent_logs.get(agent_id, [])[-TASK_LOG_TAIL_LINES:]]
            _finalize_task(
                agent_id,
                "fail",
                reason="agent_crashed",
                exit_code=exit_code,
                tail_logs=tail_logs
            )

    # Stop agent (cleanup resources)
    stop_agent(agent_id, graceful=False)

    # Check restart eligibility
    if can_restart_agent(agent_id):
        restart_count = len([ts for ts in agent_restart_counts[agent_id]
                            if ts > time.time() - 3600])
        delay = get_restart_delay(restart_count)

        logger.info(f"Scheduling restart for {agent_id} in {delay}s (attempt {restart_count + 1})")
        broadcast_agent_event('agent_restart_scheduled', {
            'agent_id': agent_id,
            'delay': delay,
            'attempt': restart_count + 1
        })

        # Schedule restart
        timer = threading.Timer(delay, restart_agent, args=(agent_info,))
        timer.daemon = True
        timer.start()
    else:
        logger.error(f"Agent {agent_id} exceeded restart limit, giving up")
        broadcast_agent_event('agent_crash_fatal', {
            'agent_id': agent_id,
            'reason': 'max_restarts_exceeded'
        })


def can_restart_agent(agent_id: str) -> bool:
    """Check if agent can be restarted (rate limit)."""
    now = time.time()
    hour_ago = now - 3600
    recent_restarts = [ts for ts in agent_restart_counts[agent_id] if ts > hour_ago]
    agent_restart_counts[agent_id] = recent_restarts
    return len(recent_restarts) < MAX_RESTARTS_PER_HOUR


def get_restart_delay(restart_count: int) -> float:
    """Get exponential backoff delay."""
    if restart_count >= len(RESTART_BACKOFF_SECONDS):
        return RESTART_BACKOFF_SECONDS[-1]
    return RESTART_BACKOFF_SECONDS[restart_count]


def restart_agent(agent_info: Dict):
    """Restart crashed agent with original config."""
    try:
        new_agent = spawn_agent(**agent_info)
        agent_restart_counts[new_agent['id']].append(time.time())
        logger.info(f"Agent restarted successfully: {new_agent['id']}")
        broadcast_agent_event('agent_restarted', new_agent)
    except Exception as e:
        logger.error(f"Failed to restart agent: {e}")


def broadcast_agent_event(event_type: str, data: Dict):
    """
    Broadcast agent event to WebSocket clients.

    Args:
        event_type: Event type (agent_spawned, agent_status_changed, etc.)
        data: Event data
    """

    try:
        from ..routes.websocket import socketio
        if socketio:
            socketio.emit(event_type, data, room="all")
    except Exception as e:
        logger.debug(f"Could not broadcast agent event: {e}")


def _sync_state_to_sqlite():
    """Periodically sync in-memory task states to SQLite for crash recovery.

    Called by the watchdog every ~30 seconds. Syncs:
    - agent_task_states dict -> agent_task_state table
    - Trims old log entries from SQLite
    """
    try:
        store = get_state_store()

        # Sync task states
        for agent_id, state in list(agent_task_states.items()):
            store.upsert_task_state(
                agent_id,
                task_id=state.get("task_id"),
                thread_id=state.get("thread_id"),
                task_display=state.get("task"),
                status=state.get("status"),
                started_at=state.get("started_at"),
                done_at=state.get("done_at"),
                done_status=state.get("done_status"),
                done_reason=state.get("done_reason"),
                elapsed_ms=state.get("elapsed_ms"),
                progress_pct=state.get("progress_pct"),
                progress_msg=state.get("progress_msg"),
                result_json=json.dumps(state.get("result")) if state.get("result") else None,
                last_marker_at=state.get("last_marker_at"),
            )

        # Trim old logs periodically (every 10 sync cycles = ~5 min)
        if _watchdog_cycle_count % 60 == 0:
            store.trim_all_logs()

    except Exception as e:
        logger.debug(f"State sync to SQLite failed: {e}")


def record_task_performance(agent_id: str, task_id: str, category: str,
                            success: bool, duration_ms: int, repo: str = "",
                            estimated_cost: float = 0.0):
    """Record task performance to SQLite for smart routing."""
    try:
        store = get_state_store()
        store.record_performance(
            agent_id=agent_id,
            task_id=task_id,
            task_category=category,
            success=success,
            duration_ms=duration_ms,
            repo=repo,
            estimated_cost=estimated_cost,
        )
    except Exception as e:
        logger.debug(f"Failed to record performance: {e}")


def save_agents_data():
    """Save active agents data to file."""

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(AGENTS_DATA_FILE), exist_ok=True)

        # Build data structure (exclude process objects)
        data = {
            "agents": [
                {
                    "id": agent["id"],
                    "device_id": agent["device_id"],
                    "name": agent["name"],
                    "specialty": agent["specialty"],
                    "cli_provider": agent["cli_provider"],
                    "model": agent["model"],
                    "status": agent["status"],
                    "tasks_completed": agent.get("tasks_completed", 0),
                    "spawned_at": agent["spawned_at"],
                    "working_directory": agent["working_directory"],
                    "process_id": agent["process_id"]
                }
                for agent in active_agents.values()
            ]
        }

        # Write to file
        with open(AGENTS_DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to save agents data: {e}")


def load_agents_data():
    """Load agents data from file (for recovery after restart)."""

    try:
        if not os.path.exists(AGENTS_DATA_FILE):
            return

        with open(AGENTS_DATA_FILE, "r") as f:
            data = json.load(f)

        # Note: We can't actually restore running processes after restart
        # This is just for displaying historical data
        logger.info(f"Loaded {len(data.get('agents', []))} agents from data file")

    except Exception as e:
        logger.error(f"Failed to load agents data: {e}")


# Load agents data on module import
load_agents_data()

# Start agent watchdog and output batch flusher
start_agent_watchdog()
start_output_batch_flusher()
