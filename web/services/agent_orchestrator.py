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

                # Broadcast status change
                broadcast_agent_event("agent_status_changed", get_agent_status(agent_id))

                # Wait for completion (simplified - real impl would parse output)
                # For now, just wait a bit and mark as complete
                time.sleep(2)

                # Task complete
                agent["tasks_completed"] = agent.get("tasks_completed", 0) + 1
                agent["status"] = "idle"
                agent["current_task"] = None

                # Broadcast completion
                broadcast_agent_event("agent_task_completed", {
                    "agent_id": agent_id,
                    "task": task,
                    "completed_at": datetime.utcnow().isoformat()
                })

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
