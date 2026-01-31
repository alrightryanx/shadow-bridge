"""
Agent Orchestrator Service

Manages persistent AI agent lifecycle for real-time orchestration.
Supports spawning, monitoring, task assignment, and stopping agents.
"""

import subprocess
import threading
import queue
import os
import time
import uuid
import json
import re
import psutil
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import defaultdict

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

# Recovery config
MAX_RESTARTS_PER_HOUR = 3
RESTART_BACKOFF_SECONDS = [1, 2, 4, 8]
WATCHDOG_CHECK_INTERVAL = 5

# Agent data file path
AGENTS_DATA_FILE = os.path.join(os.path.expanduser("~"), ".shadowbridge", "agents.json")


def spawn_agent(
    device_id: str,
    name: str,
    specialty: str,
    cli_provider: str,
    model: str,
    working_directory: Optional[str] = None,
    auto_accept_edits: bool = True,
    session_id: Optional[str] = None
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

    Returns:
        Agent info dict with id, status, etc.
    """

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

    logger.info(f"Spawning agent {agent_id}: {name} ({specialty})")
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

        # Start output monitoring thread
        output_thread = threading.Thread(
            target=monitor_agent_output,
            args=(agent_id, process.stdout, process.stderr),
            daemon=True
        )
        output_thread.start()

        # Start task processing thread
        task_thread = threading.Thread(
            target=process_agent_tasks,
            args=(agent_id,),
            daemon=True
        )
        task_thread.start()

        # Save agents data
        save_agents_data()

        # Broadcast agent spawned event
        broadcast_agent_event("agent_spawned", agent_info)

        logger.info(f"Agent {agent_id} spawned successfully (PID: {process.pid})")

        return agent_info

    except Exception as e:
        logger.error(f"Failed to spawn agent: {e}")
        raise


def assign_task(agent_id: str, task: str) -> bool:
    """
    Assign a task to a running agent.

    Args:
        agent_id: Agent ID
        task: Task description

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

    task_data = {
        "task": task,
        "assigned_at": datetime.utcnow().isoformat()
    }

    task_queue.put(task_data)

    # Update agent status
    agent["status"] = "busy"
    agent["current_task"] = task

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
            cpu_percent = proc.cpu_percent(interval=0.1)
            memory_mb = proc.memory_info().rss / 1024 / 1024
            uptime_seconds = int(time.time() - proc.create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Get recent output lines
    output_lines = agent_logs.get(agent_id, [])[-10:]  # Last 10 lines

    return {
        "id": agent["id"],
        "device_id": agent["device_id"],
        "name": agent["name"],
        "specialty": agent["specialty"],
        "cli_provider": agent["cli_provider"],
        "model": agent["model"],
        "status": agent["status"],
        "current_task": agent.get("current_task"),
        "tasks_completed": agent.get("tasks_completed", 0),
        "spawned_at": agent["spawned_at"],
        "working_directory": agent["working_directory"],
        "process_id": agent["process_id"],
        "cpu_percent": round(cpu_percent, 1),
        "memory_mb": round(memory_mb, 1),
        "uptime_seconds": uptime_seconds,
        "output_lines": output_lines
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

    # Update status
    agent["status"] = "offline"

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


def get_all_agents(device_id: Optional[str] = None) -> List[Dict]:
    """
    Get all active agents, optionally filtered by device.

    Args:
        device_id: Optional device ID to filter by

    Returns:
        List of agent status dicts
    """

    agents = []
    for agent_id in list(active_agents.keys()):
        status = get_agent_status(agent_id)
        if status:
            if device_id is None or status.get("device_id") == device_id:
                agents.append(status)

    return agents


def parse_task_marker(line: str) -> Optional[Dict]:
    """Parse task completion markers from agent output."""
    patterns = {
        r'\[TASK:START\]\s+(\S+)': ('start', 1),
        r'\[TASK:COMPLETE\]\s+(\S+)\s+(SUCCESS|FAILED)': ('complete', 2),
        r'\[TASK:ERROR\]\s+(\S+)\s+(.+)': ('error', 2),
    }
    for pattern, (marker_type, groups) in patterns.items():
        match = re.search(pattern, line)
        if match:
            return {
                'type': marker_type,
                'task_id': match.group(1) if groups >= 1 else None,
                'status': match.group(2) if groups >= 2 else None
            }

    # Fallback: check for common completion keywords
    if re.search(r'(done|finished|completed|success)', line, re.IGNORECASE):
        return {'type': 'complete', 'task_id': None, 'status': 'SUCCESS'}
    if re.search(r'(error|exception|failed)', line, re.IGNORECASE):
        return {'type': 'error', 'task_id': None, 'status': 'FAILED'}

    return None


def update_task_status(agent_id: str, marker: Dict):
    """Update task status and signal completion event."""
    if marker['type'] == 'complete':
        # Signal task completion event
        if agent_id in agent_task_events:
            agent_task_events[agent_id].set()

        # Update agent status
        agent = active_agents.get(agent_id)
        if agent:
            agent['status'] = 'idle'
            agent['current_task'] = None
            broadcast_agent_event('agent_task_completed', {
                'agent_id': agent_id,
                'task': agent_task_states.get(agent_id, {}).get('task'),
                'completed_at': datetime.utcnow().isoformat()
            })


def monitor_agent_output(agent_id: str, stdout, stderr):
    """
    Monitor agent output streams and log/broadcast.

    Args:
        agent_id: Agent ID
        stdout: stdout stream
        stderr: stderr stream
    """

    def read_stream(stream, stream_type):
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

                # Trim logs to max 1000 lines
                if len(agent_logs[agent_id]) > 1000:
                    agent_logs[agent_id] = agent_logs[agent_id][-1000:]

                # Parse task markers
                parsed_marker = parse_task_marker(line)
                if parsed_marker:
                    update_task_status(agent_id, parsed_marker)

                # Broadcast output line
                broadcast_agent_event("agent_output_line", {
                    "agent_id": agent_id,
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

                        # Post to session endpoint
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

    # Start threads for stdout and stderr
    stdout_thread = threading.Thread(target=read_stream, args=(stdout, "stdout"), daemon=True)
    stderr_thread = threading.Thread(target=read_stream, args=(stderr, "stderr"), daemon=True)

    stdout_thread.start()
    stderr_thread.start()

    # Wait for both threads
    stdout_thread.join()
    stderr_thread.join()

    logger.info(f"Output monitoring stopped for agent {agent_id}")


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

            logger.info(f"Agent {agent_id} executing task: {task}")

            # Send task to agent's stdin
            try:
                process.stdin.write(f"{task}\n")
                process.stdin.flush()

                # Update status
                agent["status"] = "busy"
                agent["current_task"] = task

                # Initialize task tracking
                if agent_id not in agent_task_events:
                    agent_task_events[agent_id] = threading.Event()

                event = agent_task_events[agent_id]
                event.clear()

                # Store task state
                agent_task_states[agent_id] = {
                    'task': task,
                    'status': 'running',
                    'started_at': datetime.utcnow().isoformat()
                }

                # Broadcast status change
                broadcast_agent_event("agent_status_changed", get_agent_status(agent_id))

                # Wait up to 30 minutes for task completion
                completed = event.wait(timeout=1800)

                if not completed:
                    logger.warning(f"Task timeout for agent {agent_id}")
                    agent["status"] = "error"
                    broadcast_agent_event("agent_task_timeout", {
                        "agent_id": agent_id,
                        "task": task,
                        "timeout_seconds": 1800
                    })
                else:
                    # Task completed successfully
                    agent["tasks_completed"] = agent.get("tasks_completed", 0) + 1
                    logger.info(f"Agent {agent_id} completed task: {task}")

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


def watchdog_loop():
    """Monitor all agents and restart crashed ones."""
    while not agent_watchdog_stop_event.wait(WATCHDOG_CHECK_INTERVAL):
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
    if agent.get('current_task'):
        broadcast_agent_event('agent_task_failed', {
            'agent_id': agent_id,
            'task': agent['current_task'],
            'reason': 'agent_crashed'
        })

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

# Start agent watchdog
start_agent_watchdog()
