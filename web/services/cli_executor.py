"""
CLI Executor Service

Manages CLI tool execution with streaming output parsing for WebSocket backend.
Supports Claude Code, Gemini CLI, Codex, and other AI CLI tools.
"""

import subprocess
import threading
import queue
import os
import re
import logging
import time
from typing import Callable, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Active CLI processes (session_id -> process info)
active_processes: Dict[str, Dict] = {}

# Approval queues (approval_id -> queue for stdin response)
approval_queues: Dict[str, queue.Queue] = {}

# Map approval_id to session_id for routing
approval_to_session: Dict[str, str] = {}


def execute_cli_command(
    provider: str,
    model: Optional[str],
    query: str,
    working_dir: Optional[str],
    continue_conversation: bool,
    auto_accept_edits: bool,
    context: List[Dict],
    on_chunk: Callable[[Dict], None],
    session_id: str
):
    """
    Execute a CLI command and stream output chunks.

    Args:
        provider: CLI provider (claude, gemini, codex, etc.)
        model: Model ID to use
        query: User query
        working_dir: Working directory for command execution
        continue_conversation: Whether to continue conversation
        auto_accept_edits: Whether to auto-accept edits
        context: Previous conversation context
        on_chunk: Callback to emit stream chunks
        session_id: Session ID for tracking
    """

    # Build CLI command
    cmd = build_cli_command(
        provider=provider,
        model=model,
        query=query,
        continue_conversation=continue_conversation,
        auto_accept_edits=auto_accept_edits
    )

    if not cmd:
        on_chunk({
            "type": "error",
            "message": f"Unknown provider: {provider}"
        })
        return

    logger.info(f"Executing CLI command: {cmd}")

    # Set working directory (default to user's home if not specified)
    cwd = working_dir or os.path.expanduser("~")

    # Start subprocess - use shlex to parse command safely
    try:
        import shlex
        cmd_parts = shlex.split(cmd) if isinstance(cmd, str) else cmd
        process = subprocess.Popen(
            cmd_parts,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=cwd,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Store process info
        active_processes[session_id] = {
            "process": process,
            "provider": provider,
            "model": model,
            "query": query
        }

        # Start output monitoring threads
        stdout_thread = threading.Thread(
            target=monitor_output,
            args=(process.stdout, on_chunk, session_id, "stdout")
        )
        stderr_thread = threading.Thread(
            target=monitor_output,
            args=(process.stderr, on_chunk, session_id, "stderr")
        )

        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Start approval consumer thread
        approval_thread = threading.Thread(
            target=consume_approval_responses,
            args=(process, session_id),
            daemon=True
        )
        approval_thread.start()

        # Wait for process to complete
        def wait_for_completion():
            process.wait()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            # Send completion chunk
            on_chunk({
                "type": "complete",
                "fullResponse": "Command completed",
                "isPartial": False
            })

            # Cleanup approval queues for this session
            for approval_id in list(approval_queues.keys()):
                if approval_to_session.get(approval_id) == session_id:
                    del approval_queues[approval_id]
                    del approval_to_session[approval_id]

            # Cleanup
            if session_id in active_processes:
                del active_processes[session_id]

        completion_thread = threading.Thread(target=wait_for_completion)
        completion_thread.daemon = True
        completion_thread.start()

    except Exception as e:
        logger.error(f"Failed to execute CLI command: {e}")
        on_chunk({
            "type": "error",
            "message": f"Failed to execute command: {str(e)}"
        })


def monitor_output(
    stream,
    on_chunk: Callable[[Dict], None],
    session_id: str,
    stream_type: str
):
    """
    Monitor subprocess output stream and parse into chunks.

    Args:
        stream: stdout or stderr stream
        on_chunk: Callback to emit stream chunks
        session_id: Session ID for tracking
        stream_type: "stdout" or "stderr"
    """
    try:
        for line in stream:
            if not line:
                continue

            line = line.strip()

            # Parse line into stream chunk (now passing session_id)
            chunk = parse_output_line(line, stream_type, session_id)

            if chunk:
                on_chunk(chunk)

    except Exception as e:
        logger.error(f"Error monitoring {stream_type}: {e}")


def parse_output_line(line: str, stream_type: str, session_id: str) -> Optional[Dict]:
    """
    Parse a CLI output line into a stream chunk.

    Detects:
    - Progress indicators (Reading file, Thinking, etc.)
    - Tool usage (Edit, Write, Read, Bash, etc.)
    - Approval prompts
    - Errors (rate limits, auth failures, etc.)
    - Content (actual response text)
    """

    # Ignore empty lines
    if not line:
        return None

    # Detect approval prompts
    # Example: "Allow tool use? [y/n]"
    if re.search(r'\[y/n\]|\(y/n\)|\[yes/no\]', line, re.IGNORECASE):
        approval_id = str(uuid.uuid4())
        approval_queues[approval_id] = queue.Queue()
        approval_to_session[approval_id] = session_id  # Map approval to session

        return {
            "type": "approval_required",
            "id": approval_id,
            "prompt": line,
            "promptType": "PERMISSION",
            "options": ["Yes", "No"]
        }

    # Detect tool usage
    # Example: "Using tool: edit" or "Editing src/main.py"
    tool_patterns = {
        r'edit(?:ing)?\s+([^\s]+)': 'edit',
        r'writ(?:ing|e)\s+([^\s]+)': 'write',
        r'read(?:ing)?\s+([^\s]+)': 'read',
        r'bash:\s*(.+)': 'bash',
        r'grep:\s*(.+)': 'grep',
    }

    for pattern, tool in tool_patterns.items():
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            detail = match.group(1) if match.groups() else None
            return {
                "type": "tool_use",
                "tool": tool,
                "detail": detail
            }

    # Detect progress indicators
    progress_keywords = [
        "thinking", "reading", "analyzing", "processing",
        "searching", "generating", "writing", "loading"
    ]

    for keyword in progress_keywords:
        if keyword in line.lower():
            return {
                "type": "progress",
                "status": keyword,
                "detail": line
            }

    # Detect errors
    error_patterns = {
        r'rate limit': ("Rate limit exceeded", True),
        r'authentication failed|auth failed|invalid api key': ("Authentication failed", False),
        r'error:|exception:|failed:': (line, False),
        r'permission denied': ("Permission denied", False),
    }

    for pattern, (message, is_rate_limit) in error_patterns.items():
        if re.search(pattern, line, re.IGNORECASE):
            return {
                "type": "error",
                "message": message,
                "isRateLimit": is_rate_limit
            }

    # Detect stderr as potential errors
    if stream_type == "stderr":
        return {
            "type": "error",
            "message": line,
            "isRateLimit": False
        }

    # Default: treat as content
    return {
        "type": "content",
        "text": line
    }


def build_cli_command(
    provider: str,
    model: Optional[str],
    query: str,
    continue_conversation: bool,
    auto_accept_edits: bool
) -> Optional[str]:
    """
    Build CLI command for the specified provider.

    Args:
        provider: CLI provider name (claude, gemini, codex, etc.)
        model: Model ID
        query: User query
        continue_conversation: Whether to continue conversation
        auto_accept_edits: Whether to auto-accept edits

    Returns:
        CLI command string, or None if provider is unknown
    """

    # Escape query for shell
    escaped_query = query.replace('"', '\\"').replace('$', '\\$')

    if provider == "claude":
        cmd = f'claude'
        if model:
            cmd += f' --model {model}'
        if auto_accept_edits:
            cmd += ' --dangerously-skip-permissions'
        if continue_conversation:
            cmd += ' --continue'
        cmd += f' -p "{escaped_query}"'
        return cmd

    elif provider == "gemini":
        cmd = f'gemini'
        if model:
            cmd += f' -m {model}'
        if auto_accept_edits:
            cmd += ' -y'
        if continue_conversation:
            cmd += ' --resume'
        cmd += f' "{escaped_query}"'
        return cmd

    elif provider == "codex":
        cmd = f'codex'
        if model:
            cmd += f' --model {model}'
        if auto_accept_edits:
            cmd += ' --yolo'
        if continue_conversation:
            cmd += ' chat --continue'
        else:
            cmd += ' chat'
        cmd += f' "{escaped_query}"'
        return cmd

    elif provider == "aider":
        cmd = f'aider'
        if model:
            cmd += f' --model {model}'
        if auto_accept_edits:
            cmd += ' --yes'
        cmd += f' --message "{escaped_query}"'
        return cmd

    elif provider == "ollama":
        cmd = f'ollama run'
        if model:
            cmd += f' {model}'
        else:
            cmd += ' llama3.2'
        cmd += f' "{escaped_query}"'
        return cmd

    else:
        logger.warning(f"Unknown provider: {provider}")
        return None


def consume_approval_responses(process, session_id: str):
    """
    Monitor approval queues and write responses to process stdin.

    Args:
        process: Subprocess instance
        session_id: Session ID for this process
    """
    logger.info(f"Approval consumer started for session {session_id}")

    while process.poll() is None:  # While process alive
        try:
            # Find approval queues for this session
            for approval_id in list(approval_queues.keys()):
                if approval_to_session.get(approval_id) != session_id:
                    continue

                approval_queue = approval_queues[approval_id]

                try:
                    # Non-blocking get
                    response = approval_queue.get(timeout=0.1)

                    # Write to stdin
                    logger.info(f"Sending approval response to CLI: {response}")
                    process.stdin.write(f"{response}\n")
                    process.stdin.flush()

                    # Cleanup
                    del approval_queues[approval_id]
                    del approval_to_session[approval_id]

                except queue.Empty:
                    continue

        except Exception as e:
            logger.error(f"Error in approval consumer: {e}")
            time.sleep(0.1)

    logger.info(f"Approval consumer stopped for session {session_id}")


def send_approval_response(
    approval_id: str,
    approved: bool,
    selected_option: Optional[str]
) -> bool:
    """
    Send approval response to active CLI process.

    Args:
        approval_id: Approval ID from ApprovalRequired chunk
        approved: Whether the approval was granted
        selected_option: Selected option text (if applicable)

    Returns:
        True if response was sent successfully
    """

    # Get the approval queue
    approval_queue = approval_queues.get(approval_id)
    if not approval_queue:
        logger.warning(f"No approval queue found for ID: {approval_id}")
        return False

    # Determine response to send
    if approved:
        response = selected_option or "yes"
    else:
        response = "no"

    # Send response to queue
    try:
        approval_queue.put(response)
        return True
    except Exception as e:
        logger.error(f"Failed to send approval response: {e}")
        return False


def cleanup_session(session_id: str):
    """
    Clean up resources for a session.

    Args:
        session_id: Session ID to clean up
    """
    if session_id in active_processes:
        process_info = active_processes[session_id]
        process = process_info.get("process")

        if process and process.poll() is None:
            # Process is still running, terminate it
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Failed to terminate process: {e}")
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass  # Process already terminated

        del active_processes[session_id]

    # Clean up approval queues
    for approval_id in list(approval_queues.keys()):
        if approval_id.startswith(session_id):
            del approval_queues[approval_id]
