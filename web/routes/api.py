"""
REST API endpoints for Shadow Web Dashboard
"""

from flask import Blueprint, jsonify, request, send_file
import subprocess
import shlex
import socket
import json
import os
import time
import logging
import urllib.request
from functools import lru_cache, wraps
from threading import Lock
from collections import defaultdict

from ..services.data_service import (
    get_note_content as get_cached_note_content,
    save_note_content,
    decrypt_note_content,
    encrypt_note_content,
    get_devices,
    DYNAMIC_SALT,
    get_device,
    remove_device,
    get_projects,
    get_project,
    update_project,
    get_notes,
    get_note,
    get_sessions,
    get_session,
    upsert_session,
    delete_session,
    append_session_message,
    get_automations,
    get_automation,
    get_automation_logs,
    get_agents,
    get_agent,
    get_agent_metrics,
    add_agent,
    update_agent,
    delete_agent,
    get_teams,
    get_team,
    create_team,
    update_team,
    delete_team,
    get_team_metrics,
    get_tasks,
    get_task,
    create_task,
    update_task,
    delete_task,
    get_workflows,
    start_workflow,
    cancel_workflow,
    get_audits,
    get_audit_entry,
    get_audit_stats,
    get_audit_traces,
    export_audit_report,
    get_favorites,
    toggle_favorite,
    search_all,
    get_privacy_score,
    get_token_usage,
    get_category_breakdown,
    get_usage_stats,
    get_backend_usage,
    get_activity_timeline,
    get_status,
    # Ownership & Sharing
    share_note,
    unshare_note,
    share_project,
    unshare_project,
    get_shared_content_for_device,
    get_permission_level,
    mark_items_synced,
    # Email Verification
    request_email_verification,
    verify_email_code,
    get_verified_email,
    get_all_verified_emails,
    remove_verified_email,
    get_devices_by_email,
    set_email_config,
    # Team Membership
    invite_team_member,
    accept_team_invitation,
    get_pending_invitations,
    remove_team_member,
    get_teams_for_email,
    update_team_member_role,
    # Unified Memory (AGI-readiness)
    get_memory_stats,
    get_memory_search_results,
)
from ..services.task_service import get_task_service
from ..services.adb_service import get_adb_service

api_bp = Blueprint("api", __name__)

# Logger for API errors
logger = logging.getLogger(__name__)

# ============ Rate Limiting ============


class ApiRateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed and record it."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds

            # Clean up old requests
            self.requests[key] = [t for t in self.requests[key] if t > window_start]

            # Check limit
            if len(self.requests[key]) >= self.requests_per_minute:
                return False

            # Record request
            self.requests[key].append(now)
            return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            recent = [t for t in self.requests[key] if t > window_start]
            return max(0, self.requests_per_minute - len(recent))


# Global rate limiter instance
_api_rate_limiter = ApiRateLimiter(requests_per_minute=120)


def rate_limit(f):
    """Decorator to apply rate limiting to an endpoint."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Use client IP as rate limit key
        client_ip = request.remote_addr or "unknown"

        if not _api_rate_limiter.is_allowed(client_ip):
            return jsonify({"error": "Rate limit exceeded", "retry_after": 60}), 429

        return f(*args, **kwargs)

    return decorated_function


def api_error_handler(f):
    """Decorator to provide consistent error handling for API endpoints."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except socket.timeout:
            logger.warning(f"Timeout in {f.__name__}")
            return jsonify({"error": "Device connection timeout"}), 504
        except ConnectionRefusedError:
            logger.warning(f"Connection refused in {f.__name__}")
            return jsonify({"error": "Device not accepting connections"}), 503
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in {f.__name__}: {e}")
            return jsonify({"error": "Invalid JSON data"}), 400
        except ValueError as e:
            logger.warning(f"Value error in {f.__name__}: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception(f"Unexpected error in {f.__name__}")
            return jsonify({"error": "Internal server error"}), 500

    return decorated_function


# ============ Note Content Cache ============

# Note content cache for faster repeated access
# Key: note_id, Value: (content_dict, timestamp)
_note_content_cache = {}
_note_cache_lock = Lock()
NOTE_CACHE_TTL_SECONDS = 300  # Cache for 5 minutes


def _get_cached_note_content(note_id: str) -> dict | None:
    """Get cached note content if available and not expired."""
    with _note_cache_lock:
        if note_id in _note_content_cache:
            content, cached_time = _note_content_cache[note_id]
            if time.time() - cached_time < NOTE_CACHE_TTL_SECONDS:
                return content
            else:
                del _note_content_cache[note_id]
    return None


def _cache_note_content(note_id: str, content: dict):
    """Cache note content with current timestamp."""
    with _note_cache_lock:
        _note_content_cache[note_id] = (content, time.time())
        # Limit cache size to 100 notes
        if len(_note_content_cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(_note_content_cache.items(), key=lambda x: x[1][1])
            for old_id, _ in sorted_items[:20]:
                del _note_content_cache[old_id]


def _invalidate_note_cache(note_id: str):
    """Remove a note from cache (after update)."""
    with _note_cache_lock:
        _note_content_cache.pop(note_id, None)


DEFAULT_NOTE_CONTENT_PORT = 19285
FALLBACK_NOTE_CONTENT_PORTS = [
    DEFAULT_NOTE_CONTENT_PORT,
    DEFAULT_NOTE_CONTENT_PORT + 1,
    DEFAULT_NOTE_CONTENT_PORT + 2,
]
NOTE_SYNC_RETRY_DELAY_S = 0.4


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    """Receive an exact number of bytes or return b'' on disconnect."""
    data = b""
    # Use larger buffer for better performance with large notes
    buffer_size = min(65536, length)
    while len(data) < length:
        chunk = sock.recv(min(buffer_size, length - len(data)))
        if not chunk:
            break
        data += chunk
    return data


def _send_note_request(host: str, port: int, payload: dict, timeout_s: int) -> dict:
    """Send a length-prefixed JSON request to the device and return JSON response."""
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        # Set larger socket buffer for better performance
        sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, 262144
        )  # 256KB receive buffer

        request_data = json.dumps(payload).encode("utf-8")
        sock.sendall(len(request_data).to_bytes(4, "big") + request_data)

        length_data = _recv_exact(sock, 4)
        if len(length_data) < 4:
            raise RuntimeError("Invalid response from device")

        response_length = int.from_bytes(length_data, "big")
        # Increased limit to 50MB for large notes
        if response_length <= 0 or response_length > 50_000_000:
            raise RuntimeError("Invalid response size from device")

        response_data = _recv_exact(sock, response_length)
        if len(response_data) < response_length:
            raise RuntimeError("Incomplete response from device")

        return json.loads(response_data.decode("utf-8"))


def _try_note_request(
    device_ips: list[str], port: int, payload: dict, timeout_s: int, retries: int
) -> dict:
    """Try multiple device IPs with retries using exponential backoff."""
    from ..utils.retry import ExponentialBackoff

    last_error: Exception | None = None
    for host in device_ips:
        backoff = ExponentialBackoff(base=0.5, multiplier=1.5, max_delay=5.0)
        for attempt in range(retries):
            try:
                return _send_note_request(host, port, payload, timeout_s)
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    delay = backoff.next()
                    time.sleep(delay)
    if last_error:
        raise last_error
    raise RuntimeError("No device hosts available")


def _collect_note_device_ips(note: dict) -> list[str]:
    device_ips: list[str] = []
    note_ips = note.get("device_ips", [])
    if isinstance(note_ips, list):
        for candidate in note_ips:
            if isinstance(candidate, str) and candidate:
                device_ips.append(candidate)
    device_ip = note.get("device_ip")
    if isinstance(device_ip, str) and device_ip:
        device_ips.append(device_ip)
    device_id = note.get("device_id")
    if device_id:
        device_info = get_device(device_id)
        fallback_ip = device_info.get("ip") if device_info else None
        if isinstance(fallback_ip, str) and fallback_ip:
            device_ips.append(fallback_ip)
    deduped: list[str] = []
    for candidate in device_ips:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _build_note_ports(note_port: object) -> list[int]:
    ports: list[int] = []
    if isinstance(note_port, int) and 1 <= note_port <= 65535:
        ports.append(note_port)
    for fallback in FALLBACK_NOTE_CONTENT_PORTS:
        if fallback not in ports:
            ports.append(fallback)
    return ports


def _try_note_request_ports(
    device_ips: list[str], ports: list[int], payload: dict, timeout_s: int, retries: int
) -> dict:
    last_error: Exception | None = None
    for port in ports:
        try:
            return _try_note_request(device_ips, port, payload, timeout_s, retries)
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("No device ports available")


def _try_note_sync(device_ips: list[str], ports: list[int]) -> bool:
    try:
        response = _try_note_request_ports(
            device_ips, ports, {"action": "sync_notes"}, timeout_s=6, retries=3
        )
        return bool(response.get("success", True))
    except Exception:
        return False


# ============ Devices ============


@api_bp.route("/devices")
def api_devices():
    """List all devices with status."""
    return jsonify(get_devices())


@api_bp.route("/devices/<device_id>")
def api_device(device_id):
    """Get device details."""
    device = get_device(device_id)
    if device:
        return jsonify(device)
    return jsonify({"error": "Device not found"}), 404


@api_bp.route("/devices/<device_id>", methods=["DELETE"])
def api_remove_device(device_id):
    """Remove a specific device and all its associated data."""
    result = remove_device(device_id)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 400


# ============ Projects ============


@api_bp.route("/projects")
def api_projects():
    """List all projects."""
    try:
        device_id = request.args.get("device_id")
        return jsonify(get_projects(device_id))
    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        return jsonify([]) # Return empty list on error to prevent frontend crash


@api_bp.route("/projects", methods=["POST"])
def api_create_project():
    """Create a new project."""
    from ..services.data_service import create_project

    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400
    device_id = data.get("device_id", "web")
    result = create_project(data, device_id)
    return jsonify(result)


@api_bp.route("/projects/<project_id>")
def api_project(project_id):
    """Get project details."""
    project = get_project(project_id)
    if project:
        return jsonify(project)
    return jsonify({"error": "Project not found"}), 404


@api_bp.route("/projects/<project_id>/open", methods=["POST"])
def api_open_project(project_id):
    """Open project in editor."""
    project = get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    path = project.get("path", "")
    if not path:
        return jsonify({"error": "No path for project"}), 400

    # Validate path - must be absolute and exist
    import os

    path = os.path.abspath(path)
    if not os.path.exists(path):
        return jsonify({"error": "Path does not exist"}), 400

    # Try to open in VS Code or Claude Code
    try:
        # Try VS Code first (no shell=True for security)
        subprocess.Popen(["code", path])
        return jsonify({"success": True, "editor": "vscode", "path": path})
    except Exception:
        try:
            # Fallback to explorer (no shell=True for security)
            subprocess.Popen(["explorer", path])
            return jsonify({"success": True, "editor": "explorer", "path": path})
        except Exception as e:
            return jsonify({"error": "Failed to open editor"}), 500


@api_bp.route("/projects/sync", methods=["POST"])
def api_projects_sync():
    """
    Receive projects sync notification and broadcast to WebSocket clients.
    Called by ShadowBridge after syncing projects from Android app.
    """
    try:
        from .websocket import broadcast_projects_updated

        data = request.get_json() or {}
        device_id = data.get("device_id", "unknown")

        # Broadcast to all connected WebSocket clients
        broadcast_projects_updated(device_id)

        return jsonify({"success": True, "message": "Projects sync broadcasted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ Notes ============


@api_bp.route("/notes")
def api_notes():
    """List all note titles."""
    device_id = request.args.get("device_id")
    search = request.args.get("search")
    return jsonify(get_notes(device_id, search))


@api_bp.route("/notes", methods=["POST"])
def api_create_note():
    """Create a new note."""
    from ..services.data_service import create_note

    data = request.get_json()
    if not data or not data.get("title"):
        return jsonify({"error": "Title is required"}), 400
    device_id = data.get("device_id", "web")
    result = create_note(data, device_id)
    note = result.get("note") if isinstance(result, dict) else None
    if result.get("success") and note and device_id and device_id != "web":
        try:
            note_payload = dict(note)
            note_payload["device_id"] = device_id
            if device_id:
                note_payload["content"] = encrypt_note_content(
                    note_payload.get("content", ""), device_id
                )

            device_ips = _collect_note_device_ips({"device_id": device_id})
            if device_ips:
                ports = _build_note_ports(note_payload.get("note_content_port"))
                response = _try_note_request_ports(
                    device_ips,
                    ports,
                    {"action": "create_note", "note": note_payload},
                    timeout_s=8,
                    retries=3,
                )
                if response.get("success"):
                    mark_items_synced(device_id, "notes", [note_payload.get("id")])
                    result["pushed_to_device"] = True
                else:
                    result["pushed_to_device"] = False
                    result["device_message"] = response.get("message", "Device rejected")
            else:
                result["pushed_to_device"] = False
                result["device_message"] = "Device IP not available"
        except Exception as exc:
            result["pushed_to_device"] = False
            result["device_message"] = str(exc)
    return jsonify(result)


@api_bp.route("/notes/<note_id>")
def api_note(note_id):
    """Get note metadata."""
    note = get_note(note_id)
    if note:
        return jsonify(note)
    return jsonify({"error": "Note not found"}), 404


@api_bp.route("/notes/<note_id>/content")
def api_note_content(note_id):
    """Fetch note content from device, with caching for performance."""
    # Check in-memory cache first (for repeated requests in same session)
    cached = _get_cached_note_content(note_id)
    if cached:
        return jsonify(cached)

    # Check local file cache (notes.json with content)
    local_cached = get_cached_note_content(note_id)
    if local_cached:
        # Also populate in-memory cache
        _cache_note_content(note_id, local_cached)
        return jsonify(local_cached)

    note = get_note(note_id)
    if not note:
        return jsonify({"error": "Note not found"}), 404

    device_ip = note.get("device_ip")
    device_id = note.get("device_id")
    note_port = note.get("note_content_port")
    device_ips = []
    note_ips = note.get("device_ips", [])
    if isinstance(note_ips, list):
        for candidate in note_ips:
            if isinstance(candidate, str) and candidate:
                device_ips.append(candidate)
    if device_ip:
        device_ips.append(device_ip)
    if device_id:
        device_info = get_device(device_id)
        fallback_ip = device_info.get("ip") if device_info else None
        if fallback_ip and fallback_ip not in device_ips:
            device_ips.append(fallback_ip)
    deduped_ips = []
    for candidate in device_ips:
        if candidate not in deduped_ips:
            deduped_ips.append(candidate)
    device_ips = deduped_ips
    if not device_ips:
        return jsonify({"error": "Device IP not available"}), 400
    port = DEFAULT_NOTE_CONTENT_PORT
    if isinstance(note_port, int) and 1 <= note_port <= 65535:
        port = note_port

    # Fetch content from Android device
    try:
        request = {"action": "fetch_note", "note_id": note_id}
        response = _try_note_request(device_ips, port, request, timeout_s=15, retries=2)
        if response.get("success"):
            # Decrypt content if encrypted (SYNC_ENC:... format)
            raw_content = response.get("content", "")
            is_encrypted = response.get("encrypted", False) or raw_content.startswith(
                "SYNC_ENC:"
            )
            if is_encrypted and device_id:
                content = decrypt_note_content(raw_content, device_id)
            else:
                content = raw_content

            result = {
                "id": response.get("id"),
                "title": response.get("title"),
                "content": content,
                "updated_at": response.get("updatedAt"),
            }
            # Cache in memory for faster subsequent access
            _cache_note_content(note_id, result)
            # Also save to local file cache for offline access (save decrypted)
            save_note_content(
                note_id, result.get("title", ""), content, result.get("updated_at", 0)
            )
            return jsonify(result)
        else:
            return jsonify({"error": response.get("error", "Unknown error")}), 500

    except socket.timeout:
        return jsonify(
            {"error": f"Device connection timeout (tried: {', '.join(device_ips)})"}
        ), 504
    except ConnectionRefusedError:
        return jsonify(
            {
                "error": f"Device not accepting connections (tried: {', '.join(device_ips)})"
            }
        ), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/notes/<note_id>/content", methods=["PUT"])
def api_update_note_content(note_id):
    """Update note content on device."""
    note = get_note(note_id)
    if not note:
        return jsonify({"error": "Note not found"}), 404

    device_ip = note.get("device_ip")
    device_id = note.get("device_id")
    note_port = note.get("note_content_port")
    device_ips = []
    note_ips = note.get("device_ips", [])
    if isinstance(note_ips, list):
        for candidate in note_ips:
            if isinstance(candidate, str) and candidate:
                device_ips.append(candidate)
    if device_ip:
        device_ips.append(device_ip)
    if device_id:
        device_info = get_device(device_id)
        fallback_ip = device_info.get("ip") if device_info else None
        if fallback_ip and fallback_ip not in device_ips:
            device_ips.append(fallback_ip)
    deduped_ips = []
    for candidate in device_ips:
        if candidate not in deduped_ips:
            deduped_ips.append(candidate)
    device_ips = deduped_ips
    if not device_ips:
        return jsonify({"error": "Device IP not available"}), 400
    port = DEFAULT_NOTE_CONTENT_PORT
    if isinstance(note_port, int) and 1 <= note_port <= 65535:
        port = note_port

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    new_content = data.get("content")
    new_title = data.get("title")

    if new_content is None:
        return jsonify({"error": "Content is required"}), 400

    # Encrypt content before sending to device (if device supports encryption)
    content_to_send = new_content
    if device_id:
        content_to_send = encrypt_note_content(new_content, device_id)

    # Send update to Android device
    try:
        update_request = {
            "action": "update_note",
            "note_id": note_id,
            "content": content_to_send,
        }
        # Only include title if provided and non-empty (preserves existing title otherwise)
        if new_title:
            update_request["title"] = new_title
        response = _try_note_request(
            device_ips, port, update_request, timeout_s=12, retries=2
        )
        if response.get("success"):
            # Invalidate cache so next fetch gets fresh content
            _invalidate_note_cache(note_id)

            # Save to local cache for immediate access
            save_note_content(
                note_id,
                new_title or note.get("title", "Untitled"),
                new_content,
                int(time.time() * 1000),
            )

            # Broadcast update to all connected web clients
            try:
                from .websocket import broadcast_notes_updated

                broadcast_notes_updated(device_id or "web")
            except Exception as e:
                logger.warning(f"Failed to broadcast note update: {e}")

            return jsonify(
                {"success": True, "id": note_id, "message": "Note updated successfully"}
            )
        else:
            return jsonify(
                {"error": response.get("error", "Failed to update note")}
            ), 500

    except socket.timeout:
        return jsonify(
            {"error": f"Device connection timeout (tried: {', '.join(device_ips)})"}
        ), 504
    except ConnectionRefusedError:
        return jsonify(
            {
                "error": f"Device not accepting connections (tried: {', '.join(device_ips)})"
            }
        ), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/notes/<note_id>", methods=["DELETE"])
def api_delete_note(note_id):
    """Delete a note from the device and local cache."""
    from ..services.data_service import delete_note

    # Get note info to find device
    note = get_note(note_id)
    if not note:
        return jsonify({"error": "Note not found"}), 404

    # Gather device IPs
    device_ip = note.get("device_ip")
    device_id = note.get("device_id")
    note_port = note.get("note_content_port")
    device_ips = []
    note_ips = note.get("device_ips", [])
    if isinstance(note_ips, list):
        for candidate in note_ips:
            if isinstance(candidate, str) and candidate:
                device_ips.append(candidate)
    if device_ip:
        device_ips.append(device_ip)
    if device_id:
        device_info = get_device(device_id)
        fallback_ip = device_info.get("ip") if device_info else None
        if fallback_ip and fallback_ip not in device_ips:
            device_ips.append(fallback_ip)
    deduped_ips = []
    for candidate in device_ips:
        if candidate not in deduped_ips:
            deduped_ips.append(candidate)
    device_ips = deduped_ips

    port = DEFAULT_NOTE_CONTENT_PORT
    if isinstance(note_port, int) and 1 <= note_port <= 65535:
        port = note_port

    # Try to delete from device first
    device_deleted = False
    device_error = None
    if device_ips:
        try:
            delete_request = {"action": "delete_note", "note_id": note_id}
            response = _try_note_request(
                device_ips, port, delete_request, timeout_s=10, retries=2
            )
            if response.get("success"):
                device_deleted = True
            else:
                device_error = response.get("message", "Device delete failed")
        except socket.timeout:
            device_error = "Device connection timeout"
        except ConnectionRefusedError:
            device_error = "Device not accepting connections"
        except Exception as e:
            device_error = str(e)

    # Always delete from local cache
    local_result = delete_note(note_id)

    if device_deleted:
        return jsonify(
            {"success": True, "message": "Note deleted from device and dashboard"}
        )
    elif local_result.get("success"):
        # Local delete worked but device failed - warn user
        return jsonify(
            {
                "success": True,
                "warning": f"Deleted from dashboard. Device: {device_error or 'unreachable'}",
                "message": "Note deleted from dashboard (device may be offline)",
            }
        )
    else:
        return jsonify({"error": "Failed to delete note"}), 500


@api_bp.route("/notes/<note_id>/export", methods=["POST"])
def api_export_note(note_id):
    """Export note to file."""
    import os
    from pathlib import Path

    # First get note metadata
    note = get_note(note_id)
    if not note:
        return jsonify({"error": "Note not found"}), 404

    device_ip = note.get("device_ip")
    device_id = note.get("device_id")
    note_port = note.get("note_content_port")
    device_ips = []
    note_ips = note.get("device_ips", [])
    if isinstance(note_ips, list):
        for candidate in note_ips:
            if isinstance(candidate, str) and candidate:
                device_ips.append(candidate)
    if device_ip:
        device_ips.append(device_ip)
    if device_id:
        device_info = get_device(device_id)
        fallback_ip = device_info.get("ip") if device_info else None
        if fallback_ip and fallback_ip not in device_ips:
            device_ips.append(fallback_ip)
    deduped_ips = []
    for candidate in device_ips:
        if candidate not in deduped_ips:
            deduped_ips.append(candidate)
    device_ips = deduped_ips
    if not device_ips:
        return jsonify({"error": "Device IP not available"}), 400
    port = DEFAULT_NOTE_CONTENT_PORT
    if isinstance(note_port, int) and 1 <= note_port <= 65535:
        port = note_port

    try:
        response = _try_note_request(
            device_ips,
            port,
            {"action": "fetch_note", "note_id": note_id},
            timeout_s=12,
            retries=2,
        )
        if not response.get("success"):
            return jsonify(
                {"error": response.get("message", "Failed to fetch note")}
            ), 502

        content = response.get("content", "")
        title = response.get("title", note.get("title", "Untitled"))

        # Create export directory
        export_dir = Path.home() / "Downloads" / "Shadow_Notes"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename (only remove illegal chars for OS)
        illegal_chars = '<>:"/\\|?*' if os.name == "nt" else "/"
        safe_title = "".join(c for c in title if c not in illegal_chars).strip()[:100]
        if not safe_title:
            safe_title = "note"

        # Save file
        file_path = export_dir / f"{safe_title}.md"
        counter = 1
        while file_path.exists():
            file_path = export_dir / f"{safe_title}_{counter}.md"
            counter += 1

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(content)

        return jsonify(
            {"success": True, "path": str(file_path), "filename": file_path.name}
        )

    except socket.timeout:
        return jsonify({"error": "Device connection timeout"}), 504
    except ConnectionRefusedError:
        return jsonify({"error": "Device not connected"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/notes/sync", methods=["POST"])
def api_notes_sync():
    """
    Receive notes sync notification and broadcast to WebSocket clients.
    Called by Android app after syncing notes to ShadowBridge.
    """
    try:
        from .websocket import broadcast_notes_updated

        data = request.get_json() or {}
        device_id = data.get("device_id", "unknown")

        # Broadcast to all connected WebSocket clients
        broadcast_notes_updated(device_id)

        return jsonify({"success": True, "message": "Notes sync broadcasted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ Sessions ============


@api_bp.route("/sessions")
def api_sessions():
    """List sessions with metadata."""
    device_id = request.args.get("device_id")
    project_id = request.args.get("project_id")
    limit_raw = request.args.get("limit")
    limit = None
    if limit_raw:
        try:
            limit = int(limit_raw)
        except ValueError:
            limit = None
    return jsonify(
        get_sessions(device_id=device_id, project_id=project_id, limit=limit)
    )


@api_bp.route("/sessions", methods=["POST"])
def api_upsert_session():
    """Create or update a session."""
    data = request.get_json() or {}
    session = upsert_session(data)
    try:
        from .websocket import broadcast_sessions_updated

        broadcast_sessions_updated(session.get("device_id") or "web")
    except Exception:
        pass
    return jsonify(session)


@api_bp.route("/sessions/<session_id>")
def api_session(session_id):
    """Get full session with messages."""
    session = get_session(session_id)
    if session:
        return jsonify(session)
    return jsonify({"error": "Session not found"}), 404


@api_bp.route("/sessions/<session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    """Delete a session."""
    result = delete_session(session_id)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 404


@api_bp.route("/sessions/<session_id>/messages", methods=["POST"])
def api_session_message(session_id):
    """Append or update a session message (supports streaming)."""
    data = request.get_json() or {}
    message = data.get("message") or data.get("payload") or data
    delta = data.get("delta")
    is_final = bool(data.get("is_final", False))
    session_meta = data.get("session_meta")

    result = append_session_message(
        session_id=session_id,
        message=message if isinstance(message, dict) else {},
        delta=delta if isinstance(delta, str) else None,
        is_final=is_final,
        session_meta=session_meta if isinstance(session_meta, dict) else None,
    )

    try:
        from .websocket import broadcast_session_message, broadcast_sessions_updated

        is_update = bool(data.get("is_update")) or bool(delta)
        session = result.get("session") if isinstance(result, dict) else None
        message_payload = result.get("message") if isinstance(result, dict) else None
        if message_payload:
            broadcast_session_message(
                session_id, message_payload, is_update=is_update
            )
        device_id = (
            (session or {}).get("device_id")
            or (session or {}).get("deviceId")
            or "web"
        )
        broadcast_sessions_updated(device_id)
    except Exception:
        pass

    return jsonify(
        {
            "success": True,
            "session": result.get("session") if isinstance(result, dict) else None,
            "message": result.get("message") if isinstance(result, dict) else None,
        }
    )


@api_bp.route("/sessions/message", methods=["POST"])
def api_session_message_shortcut():
    """Append a session message with session_id in payload."""
    data = request.get_json() or {}
    session_id = data.get("session_id") or data.get("sessionId")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    return api_session_message(session_id)

@api_bp.route("/sessions/sync", methods=["POST"])
def api_sessions_sync():
    """
    Receive sessions sync notification and broadcast to WebSocket clients.
    Called by ShadowBridge after syncing sessions from Android app.
    """
    try:
        from .websocket import broadcast_sessions_updated

        data = request.get_json() or {}
        device_id = data.get("device_id", "unknown")

        broadcast_sessions_updated(device_id)

        return jsonify({"success": True, "message": "Sessions sync broadcasted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/cards/sync", methods=["POST"])
def api_cards_sync():
    """
    Receive cards sync notification and broadcast to WebSocket clients.
    Called by ShadowBridge after syncing cards from Android app.
    """
    try:
        from .websocket import broadcast_cards_updated

        data = request.get_json() or {}
        device_id = data.get("device_id", "unknown")

        broadcast_cards_updated(device_id)

        return jsonify({"success": True, "message": "Cards sync broadcasted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/collections/sync", methods=["POST"])
def api_collections_sync():
    """
    Receive collections sync notification and broadcast to WebSocket clients.
    Called by ShadowBridge after syncing collections from Android app.
    """
    try:
        from .websocket import broadcast_collections_updated

        data = request.get_json() or {}
        device_id = data.get("device_id", "unknown")

        broadcast_collections_updated(device_id)

        return jsonify({"success": True, "message": "Collections sync broadcasted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ Automations ============


@api_bp.route("/automations")
def api_automations():
    """List all automations."""
    device_id = request.args.get("device_id")
    return jsonify(get_automations(device_id))


@api_bp.route("/automations", methods=["POST"])
def api_create_automation():
    """Create a new automation."""
    from ..services.data_service import create_automation

    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400
    device_id = data.get("device_id", "web")
    result = create_automation(data, device_id)
    return jsonify(result)


@api_bp.route("/automations/<automation_id>")
def api_automation(automation_id):
    """Get automation details."""
    from ..services.data_service import get_full_automation

    auto = get_full_automation(automation_id)
    if auto:
        return jsonify(auto)
    return jsonify({"error": "Automation not found"}), 404


@api_bp.route("/automations/<automation_id>", methods=["PUT"])
def api_update_automation(automation_id):
    """Update an existing automation."""
    from ..services.data_service import update_automation

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_automation(automation_id, data)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 404


@api_bp.route("/automations/<automation_id>", methods=["DELETE"])
def api_delete_automation(automation_id):
    """Delete an automation."""
    from ..services.data_service import delete_automation

    result = delete_automation(automation_id)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 404


@api_bp.route("/automations/<automation_id>/toggle", methods=["POST"])
def api_toggle_automation(automation_id):
    """Toggle automation enabled status."""
    from ..services.data_service import toggle_automation_enabled

    result = toggle_automation_enabled(automation_id)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 404


@api_bp.route("/automations/<automation_id>/logs")
def api_automation_logs(automation_id):
    """Get automation execution logs."""
    logs = get_automation_logs(automation_id)
    return jsonify(logs)


def send_command_to_device(command_type: str, payload: dict, device_id: str = None) -> bool:
    """Send a command to Android device via Companion Relay."""
    try:
        message = {
            "type": command_type,
            "timestamp": int(time.time() * 1000),
            "deviceId": device_id,
            **payload
        }

        # Connect to ShadowBridge Companion Relay (localhost:19286)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", 19286))

            # Handshake as "web_api"
            handshake = {"type": "handshake", "deviceId": "web_api", "version": 1}
            h_data = json.dumps(handshake).encode("utf-8")
            s.sendall(len(h_data).to_bytes(4, "big") + h_data)
            time.sleep(0.1)

            # Send Command
            data = json.dumps(message).encode("utf-8")
            s.sendall(len(data).to_bytes(4, "big") + data)
        return True
    except Exception as e:
        logger.error(f"Failed to send command to relay: {e}")
        return False


@api_bp.route("/automations/<automation_id>/run", methods=["POST"])
def api_run_automation(automation_id):
    """Trigger manual automation run."""
    auto = get_automation(automation_id)
    if not auto:
        return jsonify({"error": "Automation not found"}), 404

    device_id = auto.get("device_id")
    success = send_command_to_device(
        "automation_trigger", 
        {"automationId": automation_id}, 
        device_id
    )

    if success:
        return jsonify({"success": True, "message": "Automation triggered on device"})
    else:
        return jsonify({"error": "Failed to send trigger command to device"}), 500


# ============ Agents ============


@api_bp.route("/agents")
def api_agents():
    """List all agents."""
    device_id = request.args.get("device_id")
    return jsonify(get_agents(device_id))


@api_bp.route("/agents/<agent_id>")
def api_agent(agent_id):
    """Get agent details."""
    agent = get_agent(agent_id)
    if agent:
        return jsonify(agent)
    return jsonify({"error": "Agent not found"}), 404


@api_bp.route("/agents/<agent_id>/tasks")
def api_agent_tasks(agent_id):
    """Get agent's task history."""
    # TODO: Implement task history
    return jsonify([])


@api_bp.route("/agents/metrics")
def api_agent_metrics():
    """Get aggregate agent metrics."""
    return jsonify(get_agent_metrics())


# ============ Agent Control (Kill Switches) ============

# In-memory activity log for real-time feed
_agent_activity_log = []
_activity_log_lock = Lock()
MAX_ACTIVITY_ITEMS = 100


def _add_activity(event_type: str, message: str, icon: str = None):
    """Add an activity event to the log."""
    with _activity_log_lock:
        _agent_activity_log.insert(
            0,
            {
                "type": event_type,
                "message": message,
                "icon": icon or event_type,
                "timestamp": int(time.time() * 1000),
            },
        )
        # Trim old items
        while len(_agent_activity_log) > MAX_ACTIVITY_ITEMS:
            _agent_activity_log.pop()


@api_bp.route("/agents/activity")
def api_agent_activity():
    """Get recent agent activity for real-time feed."""
    since = request.args.get("since", 0, type=int)
    with _activity_log_lock:
        events = [e for e in _agent_activity_log if e["timestamp"] > since]
    return jsonify(
        {
            "events": events[:20],  # Max 20 events per poll
            "timestamp": int(time.time() * 1000),
        }
    )


@api_bp.route("/agents/pause", methods=["POST"])
def api_pause_agents():
    """Pause all agent execution."""
    _add_activity("system", "All agents paused", "pause")
    
    send_command_to_device("agent_command", {"action": "pause_all"})
    
    return jsonify({"success": True, "message": "Pause command sent"})


@api_bp.route("/agents/resume", methods=["POST"])
def api_resume_agents():
    """Resume all agent execution."""
    _add_activity("system", "All agents resumed", "play")
    
    send_command_to_device("agent_command", {"action": "resume_all"})
    
    return jsonify({"success": True, "message": "Resume command sent"})


@api_bp.route("/agents/kill-all", methods=["POST"])
def api_kill_all_agents():
    """Emergency kill switch for all agents."""
    _add_activity("kill", "KILL ALL command executed", "cancel")
    
    send_command_to_device("agent_command", {"action": "kill_all"})
    
    return jsonify({"success": True, "message": "Kill all command sent"})


@api_bp.route("/agents/<agent_id>/kill", methods=["POST"])
def api_kill_agent(agent_id):
    """Kill a specific agent."""
    agent = get_agent(agent_id)
    agent_name = agent.get("name", agent_id) if agent else agent_id
    _add_activity("kill", f"Agent {agent_name} terminated", "cancel")
    
    send_command_to_device("agent_command", {"action": "kill", "agentId": agent_id})
    
    return jsonify(
        {"success": True, "message": f"Kill command sent for agent {agent_id}"}
    )


def notify_companion_relay(task):
    """Send task completion message to companion relay."""
    try:
        # Build message
        message = {
            "type": "task_complete",
            "timestamp": int(time.time() * 1000),
            "payload": {
                "task_id": task.id,
                "task_name": " ".join(task.command),
                "status": task.status.value,
                "exit_code": task.exit_code,
                "hostname": socket.gethostname(),
            },
        }

        # Internal relay payload
        relay_payload = {"type": "relay_message", "message": message}

        # Deliver via TCP socket to companion relay on loopback
        # ShadowBridge GUI app is listening on port 19286
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", 19286))

            # Protocol: Handshake first so we're identified
            handshake = {"type": "handshake", "deviceId": "task_service", "version": 1}
            h_data = json.dumps(handshake).encode("utf-8")
            s.sendall(len(h_data).to_bytes(4, "big") + h_data)

            # Wait a tiny bit for handshake processing
            time.sleep(0.1)

            # Now send the actual relay request
            r_data = json.dumps(relay_payload).encode("utf-8")
            s.sendall(len(r_data).to_bytes(4, "big") + r_data)

        logger.info(f"Task {task.id} complete - relay message sent to localhost:19286")
    except Exception as e:
        logger.error(f"Failed to relay task completion: {e}")


# Register callback
get_task_service().set_on_complete_callback(notify_companion_relay)


@api_bp.route("/tasks/start", methods=["POST"])
@api_error_handler
def api_start_task():
    """Start a new supervised background task."""
    data = request.get_json()
    if not data or not data.get("command"):
        return jsonify({"error": "Command is required"}), 400

    command = data.get("command")
    if isinstance(command, str):
        # Convert string command to list for safer execution
        import shlex

        command = shlex.split(command)

    cwd = data.get("cwd")
    if cwd:
        cwd = os.path.abspath(cwd)
        if not os.path.exists(cwd):
            return jsonify({"error": f"Path does not exist: {cwd}"}), 400

    task_id = get_task_service().start_task(command, cwd)
    _add_activity(
        "task", f"Started supervised task: {' '.join(command)[:50]}...", "play"
    )

    return jsonify(
        {"success": True, "task_id": task_id, "message": "Task started successfully"}
    )


@api_bp.route("/tasks/<task_id>/status")
@api_error_handler
def api_task_status(task_id):
    """Get status and metadata for a supervised task."""
    task = get_task_service().get_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    return jsonify(task.to_dict())


@api_bp.route("/tasks/<task_id>/output")
@api_error_handler
def api_task_output(task_id):
    """Get buffered output for a supervised task."""
    task = get_task_service().get_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0

    output = task.get_output(offset)
    return jsonify(
        {
            "task_id": task_id,
            "output": output,
            "count": len(output),
            "next_offset": offset + len(output),
        }
    )


@api_bp.route("/tasks/<task_id>/stop", methods=["POST"])
@api_error_handler
def api_stop_task(task_id):
    """Stop a running supervised task."""
    success = get_task_service().stop_task(task_id)
    if success:
        _add_activity("task", f"Stopped task {task_id}", "stop")
        return jsonify({"success": True, "message": "Task stop command sent"})
    return jsonify({"error": "Task not found or could not be stopped"}), 404


@api_bp.route("/tasks/<task_id>/cancel", methods=["POST"])
def api_cancel_task(task_id):
    """Alias for stop_task for backward compatibility."""
    return api_stop_task(task_id)


# ============ Git Workflow ============


@api_bp.route("/git/status")
@api_error_handler
def api_git_status():
    """Get git status and staged diff for a project."""
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Path is required"}), 400

    path = os.path.abspath(path)
    if not os.path.exists(path):
        return jsonify({"error": "Path does not exist"}), 404

    try:
        # Get status
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=path, text=True
        )
        # Get staged diff
        diff = subprocess.check_output(["git", "diff", "--staged"], cwd=path, text=True)
        # Get branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path, text=True
        ).strip()

        return jsonify(
            {
                "success": True,
                "branch": branch,
                "status": status,
                "diff": diff,
                "has_staged_changes": len(diff) > 0,
            }
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Git command failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/git/commit", methods=["POST"])
@api_error_handler
def api_git_commit():
    """Perform git commit and push."""
    data = request.get_json()
    path = data.get("path")
    message = data.get("message")
    push = data.get("push", True)

    if not path or not message:
        return jsonify({"error": "Path and message are required"}), 400

    path = os.path.abspath(path)
    if not os.path.exists(path):
        return jsonify({"error": "Path does not exist"}), 404

    try:
        # Commit
        subprocess.check_call(["git", "commit", "-m", message], cwd=path)

        result_msg = "Committed successfully"
        if push:
            # Push
            subprocess.check_call(["git", "push"], cwd=path)
            result_msg += " and pushed"

        _add_activity(
            "git", f"Git commit in {os.path.basename(path)}: {message[:50]}...", "check"
        )

        return jsonify({"success": True, "message": result_msg})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Git operation failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ Analytics ============


@api_bp.route("/analytics/usage")
def api_usage():
    """Get message usage stats."""
    return jsonify(get_usage_stats())


@api_bp.route("/analytics/backends")
def api_backends():
    """Get backend usage breakdown."""
    return jsonify(get_backend_usage())


@api_bp.route("/analytics/activity")
def api_activity():
    """Get activity timeline."""
    return jsonify(get_activity_timeline())


# ============ Status ============


@api_bp.route("/status")
def api_status():
    """Get system status."""
    return jsonify(get_status())

@api_bp.route("/tailscale/status")
def api_tailscale_status():
    """Return a quick health check for the local Tailscale daemon."""
    result = None
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=8
        )

        if result.returncode != 0:
            return (
                jsonify(
                    {
                        "error": "tailscale status failed",
                        "details": result.stderr.strip() or "Unknown error",
                    }
                ),
                502,
            )

        payload = json.loads(result.stdout)
        return jsonify(
            {
                "success": True,
                "status": "running",
                "detail": payload,
            }
        )
    except FileNotFoundError:
        return jsonify({"error": "Tailscale CLI is not installed"}), 501
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Tailscale status command timed out"}), 504
    except json.JSONDecodeError:
        return (
            jsonify(
                {
                    "error": "Could not parse Tailscale output",
                    "raw": (result.stdout if result else "").strip(),
                }
            ),
            502,
        )
    except Exception as exc:
        logger.exception("Tailscale status error")
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/autonomous-mode", methods=["POST"])
def api_autonomous_mode():
    """
    Trigger autonomous mode to analyze and improve projects.
    Reads todo.md, other docs, and uses proper reasoning/planning.
    Only works in debug builds for safety.
    """
    from .cognitive_orchestrator import get_cognitive_orchestrator

    # Check if debug mode is enabled
    import shadow_bridge_gui

    debug_mode = getattr(shadow_bridge_gui, "DEBUG_BUILD", False)

    if not debug_mode:
        return jsonify(
            {
                "success": False,
                "error": "Autonomous mode requires debug build. Run with --debug flag.",
            }
        ), 403

    try:
        orchestrator = get_cognitive_orchestrator()
    except Exception as e:
        import logging

        logging.error(f"Autonomous mode error: {e}")
        return jsonify(
            {
                "success": False,
                "error": f"Failed to initialize autonomous mode: {str(e)}",
            }
        ), 500

    return jsonify(
        {
            "success": True,
            "message": "Autonomous mode triggered. Analyzing projects, reading todo.md, and preparing autonomous improvements.",
        }
    ), 200


@api_bp.route("/qr")
def api_qr():
    """Get QR code image for device setup - full connection info format."""
    import io
    import base64

    try:
        import qrcode
        from PIL import Image

        # Import connection helpers from main GUI module
        try:
            import sys
            import os
            # Add parent directory to path for imports
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from shadow_bridge_gui import get_local_ip, get_tailscale_ip, get_hostname_local, get_username, find_ssh_port
        except ImportError:
            # Fallback implementations if import fails
            def get_local_ip(wait=False):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.settimeout(1)
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                    s.close()
                    return ip
                except Exception:
                    return socket.gethostbyname(socket.gethostname())

            def get_tailscale_ip():
                try:
                    import subprocess
                    result = subprocess.run(
                        ["tailscale", "ip", "-4"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0:
                        ip = result.stdout.strip().split("\n")[0]
                        if ip.startswith("100."):
                            return ip
                except Exception:
                    pass
                return None

            def get_hostname_local():
                return f"{socket.gethostname()}.local"

            def get_username():
                try:
                    return os.getlogin()
                except Exception:
                    import getpass
                    return getpass.getuser()

            def find_ssh_port():
                for port in [22, 2222, 2269, 22022, 8022]:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)
                    try:
                        if sock.connect_ex(("127.0.0.1", port)) == 0:
                            sock.close()
                            return port
                    except Exception:
                        pass
                    finally:
                        sock.close()
                return 22

        # Build full connection info (matches desktop GUI format)
        local_ip = get_local_ip(wait=True) if callable(get_local_ip) else get_local_ip()
        tailscale_ip = get_tailscale_ip()
        hostname_local = get_hostname_local()
        ssh_port = find_ssh_port() or 22

        # Get encryption salt
        try:
            salt_b64 = base64.b64encode(DYNAMIC_SALT).decode("ascii")
        except Exception:
            salt_b64 = None

        # Build hosts list (simplified: LAN + Tailscale)
        hosts_to_try = []
        if local_ip and not local_ip.startswith("127."):
            hosts_to_try.append({"host": local_ip, "type": "local", "stable": False, "label": "LAN"})
        if tailscale_ip:
            hosts_to_try.append({"host": tailscale_ip, "type": "tailscale", "stable": True, "label": "Tailscale"})

        # Primary host (prefer Tailscale for stability)
        primary_host = tailscale_ip or local_ip
        mode = "tailscale" if tailscale_ip else "local"

        # Full connection info (version 4 format - matches desktop)
        connection_info = {
            "type": "shadowai_connect",
            "version": 4,
            "mode": mode,
            "host": primary_host,
            "port": ssh_port,
            "username": get_username(),
            "hostname": socket.gethostname(),
            "hostname_local": hostname_local,
            "hosts": hosts_to_try,
            "local_ip": local_ip,
            "tailscale_ip": tailscale_ip,
            "encryption_salt": salt_b64,
            "timestamp": int(time.time()),
        }

        # Encode as base64 JSON (same format as desktop QR)
        info_json = json.dumps(connection_info)
        encoded = base64.urlsafe_b64encode(info_json.encode()).decode()
        qr_data = f"shadowai://connect?data={encoded}"

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)

        # Create image with dark theme colors (white on dark)
        img = qr.make_image(fill_color="white", back_color="#1a1a1a")
        img = img.resize((380, 380), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify(
            {
                "image": f"data:image/png;base64,{img_base64}",
                "data": qr_data,
                "host": primary_host,
                "port": ssh_port,
                "mode": mode,
                "hosts": hosts_to_try,
                "instruction": "Scan with Shadow app to connect",
            }
        )
    except ImportError:
        return jsonify(
            {
                "error": "QR code library not available",
                "instruction": "Install qrcode library: pip install qrcode[pil]",
            }
        )


# ============ Teams ============


@api_bp.route("/teams")
def api_teams():
    """List all teams."""
    device_id = request.args.get("device_id")
    return jsonify(get_teams(device_id))


@api_bp.route("/teams/<team_id>")
def api_team(team_id):
    """Get team details."""
    team = get_team(team_id)
    if team:
        return jsonify(team)
    return jsonify({"error": "Team not found"}), 404


@api_bp.route("/teams", methods=["POST"])
def api_create_team():
    """Create a new team."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    owner_email = data.pop("owner_email", None)
    owner_device = data.pop("owner_device", None)
    result = create_team(data, owner_email=owner_email, owner_device=owner_device)
    return jsonify(result)


@api_bp.route("/teams/<team_id>", methods=["PUT"])
def api_update_team(team_id):
    """Update team details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_team(team_id, data)
    return jsonify(result)


@api_bp.route("/teams/<team_id>", methods=["DELETE"])
def api_delete_team(team_id):
    """Delete a team."""
    result = delete_team(team_id)
    return jsonify(result)


@api_bp.route("/teams/metrics")
def api_team_metrics():
    """Get team metrics summary."""
    return jsonify(get_team_metrics())


# ============ Tasks ============


@api_bp.route("/tasks")
def api_tasks():
    """List all tasks."""
    device_id = request.args.get("device_id")
    return jsonify(get_tasks(device_id))


@api_bp.route("/tasks/<task_id>")
def api_task(task_id):
    """Get task details."""
    task = get_task(task_id)
    if task:
        return jsonify(task)
    return jsonify({"error": "Task not found"}), 404


@api_bp.route("/tasks", methods=["POST"])
def api_create_task():
    """Create a new task."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = create_task(data)
    return jsonify(result)


@api_bp.route("/tasks/<task_id>", methods=["PUT"])
def api_update_task(task_id):
    """Update task details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_task(task_id, data)
    return jsonify(result)


@api_bp.route("/tasks/<task_id>", methods=["DELETE"])
def api_delete_task(task_id):
    """Delete a task."""
    result = delete_task(task_id)
    return jsonify(result)


# ============ Project Todos ============


@api_bp.route("/projects/<project_id>/todos")
def api_project_todos(project_id):
    """List all todos for a project."""
    from ..services.data_service import get_project_todos

    return jsonify(get_project_todos(project_id))


@api_bp.route("/projects/<project_id>/todos", methods=["POST"])
def api_create_project_todo(project_id):
    """Create a new todo for a project."""
    from ..services.data_service import create_project_todo

    data = request.get_json()
    if not data or not data.get("content"):
        return jsonify({"error": "Content is required"}), 400
    result = create_project_todo(project_id, data)
    return jsonify(result)


@api_bp.route("/projects/<project_id>/todos/<todo_id>", methods=["PUT"])
def api_update_project_todo(project_id, todo_id):
    """Update a todo."""
    from ..services.data_service import update_project_todo

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_project_todo(project_id, todo_id, data)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@api_bp.route("/projects/<project_id>/todos/<todo_id>", methods=["DELETE"])
def api_delete_project_todo(project_id, todo_id):
    """Delete a todo."""
    from ..services.data_service import delete_project_todo

    result = delete_project_todo(project_id, todo_id)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@api_bp.route("/projects/<project_id>/todos/reorder", methods=["POST"])
def api_reorder_project_todos(project_id):
    """Reorder todos by providing list of todo IDs in desired order."""
    from ..services.data_service import reorder_project_todos

    data = request.get_json()
    if not data or not data.get("todo_ids"):
        return jsonify({"error": "todo_ids list is required"}), 400
    result = reorder_project_todos(project_id, data["todo_ids"])
    return jsonify(result)


# ============ CLI Launch ============


@api_bp.route("/projects/<project_id>/launch-cli", methods=["POST"])
def api_launch_cli(project_id):
    """Launch a CLI in the project directory (custom command allowed)."""
    project = get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    path = project.get("path", "")
    if not path:
        return jsonify({"error": "No path for project"}), 400

    path = os.path.normpath(os.path.abspath(path))
    if not os.path.exists(path):
        return jsonify({"error": f"Path does not exist: {path}"}), 400
    if not os.path.isdir(path):
        return jsonify({"error": "Path must be a directory"}), 400

    data = request.get_json(silent=True) or {}
    command_text = sanitize_cli_command(data.get("command") or "claude")

    if not command_text:
        command_text = "claude"

    try:
        launcher = None
        if os.name == "nt":
            try:
                subprocess.Popen(
                    ["wt", "-d", path, "cmd", "/k", command_text],
                    creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                )
                launcher = "wt"
            except FileNotFoundError:
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", f'cd /d "{path}" && {command_text}'],
                    creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                )
                launcher = "cmd"
        else:
            command_parts = shlex.split(command_text)
            subprocess.Popen(command_parts, cwd=path)
            launcher = command_parts[0] if command_parts else command_text

        return jsonify(
            {
                "success": True,
                "path": path,
                "command": command_text,
                "launcher": launcher,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to launch CLI: {str(e)}"}), 500


def sanitize_cli_command(command: str) -> str:
    """Strip newlines and extra whitespace from user-supplied CLI commands."""
    return " ".join(command.split())


# ============ Agent Management ============


@api_bp.route("/agents", methods=["POST"])
def api_add_agent():
    """Add a new agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = add_agent(data)
    return jsonify(result)


@api_bp.route("/agents/<agent_id>", methods=["PUT"])
def api_update_agent(agent_id):
    """Update agent details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_agent(agent_id, data)
    return jsonify(result)


@api_bp.route("/agents/<agent_id>", methods=["DELETE"])
def api_delete_agent(agent_id):
    """Delete an agent."""
    result = delete_agent(agent_id)
    return jsonify(result)


# ============ Agent Evolution (AGI-Readiness) ============


@api_bp.route("/agents/<agent_id>/evolve", methods=["POST"])
def api_agent_evolve(agent_id):
    """
    Manually trigger evolution analysis for an agent.
    """
    try:
        from ..services.agent_evolution import get_evolution_service
        from ..services.agent_protocol import AgentId, Task, TaskResult

        data = request.get_json() or {}

        # Mocking a successful task for evolution demo
        mock_task = Task.create(
            title=data.get("title", "Complex Code Optimization"),
            description=data.get("description", "Optimized memory usage in core loops"),
            required_capabilities=[],
        )

        mock_result = TaskResult(
            task_id=mock_task.id,
            success=True,
            output="Memory reduced by 40%",
            artifacts=[{"type": "profiling"}, {"type": "refactor"}],
            execution_time_ms=3500,
            metrics={"improvement": 0.4},
        )

        evol_service = get_evolution_service()
        new_cap = evol_service.analyze_task_completion(
            agent_id=AgentId.from_dict({"id": agent_id, "namespace": "shadowai"}),
            task=mock_task,
            result=mock_result,
        )

        if new_cap:
            return jsonify(
                {"success": True, "evolved": True, "capability": new_cap.to_dict()}
            )

        return jsonify({"success": True, "evolved": False})
    except Exception as e:
        logger.error(f"Evolution error: {e}")
        return jsonify({"error": str(e)}), 500


# ============ Workflows ============


@api_bp.route("/workflows")
def api_workflows():
    """List all workflows."""
    return jsonify(get_workflows())


@api_bp.route("/workflows", methods=["POST"])
def api_start_workflow():
    """Start a new workflow."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    workflow_type = data.get("type", "qa_pipeline")
    result = start_workflow(workflow_type, data)
    return jsonify(result)


@api_bp.route("/workflows/<workflow_id>/cancel", methods=["POST"])
def api_cancel_workflow(workflow_id):
    """Cancel a workflow."""
    result = cancel_workflow(workflow_id)
    return jsonify(result)


# ============ Audits ============


@api_bp.route("/audits")
def api_audits():
    """List audit entries."""
    period = request.args.get("period", "7D")
    device_id = request.args.get("device_id")
    return jsonify(get_audits(period, device_id))


@api_bp.route("/audits/<audit_id>")
def api_audit_entry(audit_id):
    """Get audit entry details."""
    entry = get_audit_entry(audit_id)
    if entry:
        return jsonify(entry)
    return jsonify({"error": "Audit entry not found"}), 404


@api_bp.route("/audits/stats")
def api_audit_stats():
    """Get audit statistics."""
    period = request.args.get("period", "7D")
    return jsonify(get_audit_stats(period))


@api_bp.route("/audits/<audit_id>/traces")
def api_audit_traces(audit_id):
    """Get AI decision traces for an audit entry."""
    traces = get_audit_traces(audit_id)
    return jsonify(traces)


@api_bp.route("/audits/export", methods=["POST"])
def api_export_audit():
    """Export audit report."""
    format = request.args.get("format", "json")
    period = request.args.get("period", "30D")
    result = export_audit_report(format, period)
    return jsonify(result)


# ============ Unified Memory (AGI-Readiness) ============


@api_bp.route("/memory/stats")
def api_memory_stats():
    """
    Get unified memory statistics across all data stores.

    Returns counts, storage estimates, and scope summaries for:
    - Notes, Projects (USER_DATA)
    - Agents, Automations (AI_DATA)
    - Audit entries (COMPLIANCE)

    This endpoint supports the AGI-readiness infrastructure.
    """
    return jsonify(get_memory_stats())


@api_bp.route("/memory/search")
def api_memory_search():
    """
    Unified search across memory scopes.

    Query params:
    - q: Search query (required)
    - scopes: Comma-separated list of scopes (optional, default: all)
              Valid: NOTES, PROJECTS, AGENTS, AUTOMATIONS
    - limit: Maximum results (optional, default: 20, max: 100)

    Currently uses keyword search. Will upgrade to semantic/vector search
    when ChromaDB is integrated (Phase 2).
    """
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    scopes_param = request.args.get("scopes", "")
    scopes = (
        [s.strip().upper() for s in scopes_param.split(",") if s.strip()]
        if scopes_param
        else None
    )

    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        limit = 20

    return jsonify(get_memory_search_results(query, scopes, limit))


# ========== ADB & File Transfer ==========


@api_bp.route("/adb/devices", methods=["GET"])
def api_adb_devices():
    """List connected Android devices via ADB."""
    adb = get_adb_service()
    devices = adb.get_devices()
    return jsonify({"devices": devices})


@api_bp.route("/adb/push", methods=["POST"])
def api_adb_push():
    """Push a local file to a connected Android device."""
    data = request.json
    local_path = data.get("local_path")
    remote_path = data.get("remote_path", "/sdcard/Download/")
    device_id = data.get("device_id")

    if not local_path:
        return jsonify({"error": "local_path is required"}), 400

    adb = get_adb_service()
    # If remote_path is a directory, append filename
    if remote_path.endswith("/"):
        filename = os.path.basename(local_path)
        remote_path = remote_path + filename

    result, success = adb.push_file(local_path, remote_path, device_id)
    return jsonify({"result": result, "success": success})


@api_bp.route("/adb/install", methods=["POST"])
def api_adb_install():
    """Install an APK on a connected Android device."""
    data = request.json
    apk_path = data.get("apk_path")
    device_id = data.get("device_id")

    if not apk_path:
        return jsonify({"error": "apk_path is required"}), 400

    adb = get_adb_service()
    result, success = adb.install_apk(apk_path, device_id)
    return jsonify({"result": result, "success": success})


@api_bp.route("/adb/shell", methods=["POST"])
def api_adb_shell():
    """Run a shell command on a connected Android device."""
    data = request.json
    command = data.get("command")
    device_id = data.get("device_id")

    if not command:
        return jsonify({"error": "command is required"}), 400

    adb = get_adb_service()
    result, success = adb.shell(command, device_id)
    return jsonify({"result": result, "success": success})


@api_bp.route("/memory/recent")
def api_memory_recent():
    """
    Get recent items from memory.

    Query params:
    - scopes: Comma-separated list of scopes (optional)
    - limit: Maximum results (optional, default: 10)

    Returns most recently modified/created items across scopes.
    """
    scopes_param = request.args.get("scopes", "")
    scopes = (
        [s.strip().upper() for s in scopes_param.split(",") if s.strip()]
        if scopes_param
        else None
    )

    try:
        limit = min(int(request.args.get("limit", 10)), 50)
    except ValueError:
        limit = 10

    # For now, just search with empty query to get all, then sort by recency
    # TODO: Implement proper recent items retrieval
    results = get_memory_search_results("", scopes, limit * 2)
    items = results.get("items", [])

    # Sort by timestamp descending (newest first)
    items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return jsonify(
        {
            "items": items[:limit],
            "scopes_searched": scopes or ["NOTES", "PROJECTS", "AGENTS", "AUTOMATIONS"],
        }
    )


# ============ Vector Store / Semantic Search ============

# Lazy import to handle missing dependencies gracefully
_vector_store = None


def get_vector_store():
    """Lazy load vector store module."""
    global _vector_store
    if _vector_store is None:
        try:
            from ..services import vector_store

            _vector_store = vector_store
        except ImportError:
            _vector_store = False
    return _vector_store if _vector_store else None


@api_bp.route("/vector/status")
def api_vector_status():
    """Get vector store status and statistics."""
    vs = get_vector_store()
    if not vs:
        return jsonify(
            {
                "available": False,
                "error": "Vector store module not loaded. Install: pip install chromadb sentence-transformers",
            }
        )
    return jsonify(vs.get_stats())


@api_bp.route("/vector/search")
def api_vector_search():
    """
    Perform semantic search using vector embeddings.

    Query params:
    - q: Search query (required)
    - types: Comma-separated source types (optional: note,project,automation,agent)
    - limit: Maximum results (optional, default: 10)

    Returns semantically similar content ranked by relevance score.
    """
    vs = get_vector_store()
    if not vs or not vs.is_available():
        return jsonify(
            {
                "error": "Vector store not available",
                "results": [],
                "search_type": "unavailable",
            }
        )

    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    types_param = request.args.get("types", "")
    source_types = (
        [t.strip().lower() for t in types_param.split(",") if t.strip()]
        if types_param
        else None
    )

    try:
        limit = min(int(request.args.get("limit", 10)), 50)
    except ValueError:
        limit = 10

    results = vs.semantic_search(query, source_types, limit)

    return jsonify(
        {
            "results": results,
            "query": query,
            "search_type": "semantic",
            "total_found": len(results),
        }
    )


@api_bp.route("/vector/index", methods=["POST"])
def api_vector_index():
    """
    Index a single document.

    JSON body:
    - source_type: Type (note, project, automation, agent)
    - source_id: Unique ID
    - title: Document title
    - content: Document content
    - metadata: Optional additional metadata
    """
    vs = get_vector_store()
    if not vs or not vs.is_available():
        return jsonify({"error": "Vector store not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["source_type", "source_id", "title", "content"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    success = vs.index_document(
        source_type=data["source_type"],
        source_id=data["source_id"],
        title=data["title"],
        content=data["content"],
        metadata=data.get("metadata"),
    )

    return jsonify({"success": success})


@api_bp.route("/vector/reindex", methods=["POST"])
def api_vector_reindex():
    """
    Reindex all content from data stores.

    This will clear the existing index and rebuild it from
    notes, projects, automations, and agents.
    """
    vs = get_vector_store()
    if not vs or not vs.is_available():
        return jsonify({"error": "Vector store not available"}), 503

    # Import data_service for reindexing
    from ..services import data_service

    counts = vs.reindex_all(data_service)
    return jsonify({"success": "error" not in counts, "indexed": counts})


@api_bp.route("/vector/clear", methods=["POST"])
def api_vector_clear():
    """Clear the entire vector index."""
    vs = get_vector_store()
    if not vs or not vs.is_available():
        return jsonify({"error": "Vector store not available"}), 503

    success = vs.clear_index()
    return jsonify({"success": success})


# ============ Context Window (AGI-Readiness) ============

# Lazy import for context window service
_context_service = None


def get_context_service():
    """Lazy load context window service."""
    global _context_service
    if _context_service is None:
        try:
            from ..services import context_window, data_service

            vs = get_vector_store()
            _context_service = context_window.get_context_service(
                data_service=data_service, vector_store=vs
            )
        except Exception as e:
            logger.error(f"Failed to load context service: {e}")
            _context_service = False
    return _context_service if _context_service else None


@api_bp.route("/context/track", methods=["POST"])
def api_context_track():
    """
    Track a user activity event for context building.

    JSON body:
    - event_type: Type of event (view, edit, search, create, delete)
    - resource_type: Type of resource (project, note, automation, agent)
    - resource_id: ID of the resource
    - resource_title: Human-readable title
    - metadata: Optional additional metadata
    """
    cs = get_context_service()
    if not cs:
        return jsonify({"error": "Context service not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["event_type", "resource_type", "resource_id", "resource_title"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    cs.track_activity(
        event_type=data["event_type"],
        resource_type=data["resource_type"],
        resource_id=data["resource_id"],
        resource_title=data["resource_title"],
        metadata=data.get("metadata"),
    )

    # AGI-Readiness: Predictive Context Pre-loading
    # If we can guess what the user will do next, we can pre-load it into memory
    # so the AI has zero-latency access to it.
    try:
        ps = get_preference_service()
        if ps:
            prediction = ps.predict_next_action(
                data["event_type"], data["resource_type"]
            )
            if prediction:
                pred_action, pred_type, conf = prediction
                if conf > 0.4:  # Low threshold for prototype
                    logger.info(
                        f"🧠 Predictive Engine: Expecting {pred_action}:{pred_type} (conf: {conf:.2f})"
                    )

                    # Heuristic: If we predict user will use a type, they likely want
                    # the most recently used item of that type.
                    recent = cs.get_recent_activity(resource_types=[pred_type], limit=1)
                    if recent:
                        target_id = recent[0].resource_id
                        logger.info(
                            f"🚀 Pre-loading context for {pred_type}:{target_id}"
                        )
                        # Run in background to not block response
                        import threading

                        threading.Thread(
                            target=cs.preload_context, args=(pred_type, target_id)
                        ).start()
    except Exception as e:
        logger.warning(f"Prediction failed: {e}")

    return jsonify({"success": True})


@api_bp.route("/context/window")
def api_context_window():
    """
    Build a context window for AI queries.

    Query params:
    - q: Optional query for semantic relevance
    - include_semantic: Whether to include semantic search results (default: true)
    - max_tokens: Maximum token budget (default: 4000)

    Returns a context window with:
    - active_context: Currently focused item
    - recent_items: Recently accessed items
    - related_items: Semantically related items (if query provided)
    - session_summary: Summary of current session
    - user_preferences: User preference settings
    """
    cs = get_context_service()
    if not cs:
        return jsonify({"error": "Context service not available"}), 503

    query = request.args.get("q", None)
    include_semantic = request.args.get("include_semantic", "true").lower() == "true"

    try:
        max_tokens = min(int(request.args.get("max_tokens", 4000)), 16000)
    except ValueError:
        max_tokens = 4000

    context = cs.build_context_window(
        query=query, include_semantic=include_semantic, max_tokens=max_tokens
    )

    return jsonify(context.to_dict())


@api_bp.route("/context/prompt")
def api_context_prompt():
    """
    Get context formatted for AI prompt injection.

    Query params:
    - q: Optional query for semantic relevance

    Returns a ready-to-use context string for AI prompts.
    """
    cs = get_context_service()
    if not cs:
        return jsonify({"error": "Context service not available"}), 503

    query = request.args.get("q", None)
    context = cs.build_context_window(query=query)

    return jsonify(
        {
            "context": context.to_prompt_context(),
            "token_estimate": context.token_estimate,
        }
    )


@api_bp.route("/context/recent")
def api_context_recent():
    """
    Get recent activity events.

    Query params:
    - minutes: How far back to look (default: 30)
    - event_types: Comma-separated event types filter
    - resource_types: Comma-separated resource types filter
    - limit: Maximum events (default: 10)
    """
    cs = get_context_service()
    if not cs:
        return jsonify({"error": "Context service not available"}), 503

    try:
        minutes = int(request.args.get("minutes", 30))
    except ValueError:
        minutes = 30

    try:
        limit = min(int(request.args.get("limit", 10)), 50)
    except ValueError:
        limit = 10

    event_types_param = request.args.get("event_types", "")
    event_types = (
        [t.strip() for t in event_types_param.split(",") if t.strip()]
        if event_types_param
        else None
    )

    resource_types_param = request.args.get("resource_types", "")
    resource_types = (
        [t.strip() for t in resource_types_param.split(",") if t.strip()]
        if resource_types_param
        else None
    )

    events = cs.get_recent_activity(
        minutes=minutes,
        event_types=event_types,
        resource_types=resource_types,
        limit=limit,
    )

    return jsonify({"events": [e.to_dict() for e in events], "total": len(events)})


@api_bp.route("/context/stats")
def api_context_stats():
    """Get context window service statistics."""
    cs = get_context_service()
    if not cs:
        return jsonify({"error": "Context service not available"}), 503

    return jsonify(cs.get_stats())


@api_bp.route("/context/topic", methods=["POST"])
def api_context_topic():
    """
    Record a topic discussed in this session.

    JSON body:
    - topic: Topic string to record
    """
    cs = get_context_service()
    if not cs:
        return jsonify({"error": "Context service not available"}), 503

    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Missing 'topic' field"}), 400

    cs.record_topic(data["topic"])
    return jsonify({"success": True})


@api_bp.route("/context/clear", methods=["POST"])
def api_context_clear():
    """Clear the context session state."""
    cs = get_context_service()
    if not cs:
        return jsonify({"error": "Context service not available"}), 503

    cs.clear_session()
    return jsonify({"success": True})


# ============ Metacognition (AGI-Readiness) ============


@api_bp.route("/metacognition/assess", methods=["POST"])
def api_metacognition_assess():
    """
    Assess a query for risk, ambiguity, and confidence.
    """
    try:
        from ..services.metacognition import get_metacognitive_layer

        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' field"}), 400

        meta = get_metacognitive_layer()
        assessment = meta.assess_query(query=data["query"], context=data.get("context"))

        return jsonify(assessment.to_dict())
    except ImportError:
        return jsonify({"error": "Metacognition service not available"}), 503
    except Exception as e:
        logger.error(f"Metacognition error: {e}")
        return jsonify({"error": str(e)}), 500


# ============ Multi-Model Debate (AGI-Readiness) ============


@api_bp.route("/reasoning/debate", methods=["POST"])
async def api_reasoning_debate():
    """
    Start a multi-model debate on a topic.
    """
    try:
        from ..services.debate_system import get_debate_system

        data = request.get_json()
        if not data or "topic" not in data:
            return jsonify({"error": "Missing 'topic' field"}), 400

        rounds = data.get("rounds", 2)
        debate_sys = get_debate_system()

        # Start debate asynchronously
        session = await debate_sys.conduct_debate(topic=data["topic"], rounds=rounds)

        return jsonify(session.to_dict())
    except Exception as e:
        logger.error(f"Debate system error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/reasoning/debate/<session_id>", methods=["GET"])
def api_get_debate(session_id):
    """
    Get a specific debate session.
    """
    from ..services.debate_system import get_debate_system

    session = get_debate_system().get_session(session_id)
    if session:
        return jsonify(session.to_dict())
    return jsonify({"error": "Debate session not found"}), 404


# ============ Cognitive Orchestrator (AGI-Readiness) ============


@api_bp.route("/cognitive/process", methods=["POST"])
async def api_cognitive_process():
    """
    Process a query through the full cognitive architecture.
    """
    try:
        from ..services.cognitive_orchestrator import get_cognitive_orchestrator

        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' field"}), 400

        orchestrator = get_cognitive_orchestrator()

        # Process asynchronously
        plan = await orchestrator.process_query(
            query=data["query"], client_context=data.get("context")
        )

        return jsonify(plan.to_dict())
    except Exception as e:
        logger.error(f"Cognitive Orchestrator error: {e}")
        return jsonify({"error": str(e)}), 500


# ============ Swarm Intelligence (AGI-Readiness) ============


@api_bp.route("/swarm/execute", methods=["POST"])
async def api_swarm_execute():
    """
    Execute a task using an emergent agent swarm.
    """
    try:
        from ..services.swarm_intelligence import get_swarm_orchestrator
        from ..services.agent_protocol import Task, CapabilityCategory

        data = request.get_json()
        if not data or "title" not in data:
            return jsonify({"error": "Missing 'title' field"}), 400

        task = Task.create(
            title=data["title"],
            description=data.get("description", ""),
            required_capabilities=[CapabilityCategory.RESEARCH],
        )

        agent_count = data.get("agent_count", 3)
        swarm_orch = get_swarm_orchestrator()

        result = await swarm_orch.execute_swarm_task(task, agent_count)

        return jsonify(result)
    except Exception as e:
        logger.error(f"Swarm error: {e}")
        return jsonify({"error": str(e)}), 500


# ============ Recursive Reasoning (AGI-Readiness) ============


@api_bp.route("/reasoning/recurse", methods=["POST"])
async def api_reasoning_recurse():
    """
    Refine a solution recursively until optimal.
    """
    try:
        from ..services.recursive_reasoning import get_recursive_engine

        data = request.get_json()
        if not data or "problem" not in data:
            return jsonify({"error": "Missing 'problem' field"}), 400

        problem = data["problem"]
        initial = data.get("initial_solution", f"Initial thought on {problem}")

        engine = get_recursive_engine()
        result = await engine.recurse_until_optimal(problem, initial)

        # Convert to dict manually as result is a dataclass with nested objects
        return jsonify(
            {
                "final_solution": result.final_solution,
                "total_iterations": result.total_iterations,
                "converged": result.converged,
                "final_confidence": result.final_confidence,
                "iterations": [
                    {
                        "depth": i.depth,
                        "confidence": i.confidence,
                        "improvement": i.improvement,
                        "solution_preview": i.solution[:100],
                    }
                    for i in result.iterations
                ],
            }
        )
    except Exception as e:
        logger.error(f"Recursive reasoning error: {e}")
        return jsonify({"error": str(e)}), 500


# ============ User Preference Learning (AGI-Readiness) ============

# Lazy import for preference service
_preference_service = None


def get_preference_service():
    """Lazy load preference learning service."""
    global _preference_service
    if _preference_service is None:
        try:
            from ..services import preference_learning

            _preference_service = preference_learning.get_preference_service()
        except Exception as e:
            logger.error(f"Failed to load preference service: {e}")
            _preference_service = False
    return _preference_service if _preference_service else None


@api_bp.route("/preferences")
def api_preferences():
    """Get current user preferences."""
    ps = get_preference_service()
    if not ps:
        return jsonify({"error": "Preference service not available"}), 503

    return jsonify(ps.get_preferences().to_dict())


@api_bp.route("/preferences/observe", methods=["POST"])
def api_preferences_observe():
    """
    Observe a user action to learn preferences.

    JSON body:
    - action: Type of action (view, edit, search, etc.)
    - resource_type: Type of resource (project, note, etc.)
    - metadata: Optional additional context (search_mode, category, tags)
    """
    ps = get_preference_service()
    if not ps:
        return jsonify({"error": "Preference service not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "action" not in data or "resource_type" not in data:
        return jsonify({"error": "Missing required fields: action, resource_type"}), 400

    ps.observe_action(
        action=data["action"],
        resource_type=data["resource_type"],
        metadata=data.get("metadata"),
    )

    return jsonify({"success": True})


@api_bp.route("/preferences/feedback", methods=["POST"])
def api_preferences_feedback():
    """
    Learn from explicit user feedback.

    JSON body:
    - category: Preference category (search_mode, theme, verbosity)
    - value: The preference value
    - is_positive: Whether user liked it (true/false)
    """
    ps = get_preference_service()
    if not ps:
        return jsonify({"error": "Preference service not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["category", "value", "is_positive"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    ps.learn_from_feedback(
        category=data["category"],
        value=data["value"],
        is_positive=bool(data["is_positive"]),
    )

    return jsonify({"success": True})


@api_bp.route("/preferences/ai-context")
def api_preferences_ai_context():
    """Get preferences formatted for AI context injection."""
    ps = get_preference_service()
    if not ps:
        return jsonify({"error": "Preference service not available"}), 503

    return jsonify(ps.get_context_for_ai())


@api_bp.route("/preferences/stats")
def api_preferences_stats():
    """Get preference learning statistics."""
    ps = get_preference_service()
    if not ps:
        return jsonify({"error": "Preference service not available"}), 503

    return jsonify(ps.get_stats())


@api_bp.route("/preferences/reset", methods=["POST"])
def api_preferences_reset():
    """Reset all learned preferences."""
    ps = get_preference_service()
    if not ps:
        return jsonify({"error": "Preference service not available"}), 503

    ps.reset()
    return jsonify({"success": True})


# ============ Agent Registry (AGI-Readiness Phase 3) ============

# Lazy import for agent registry
_agent_registry = None


def get_agent_registry():
    """Lazy load agent registry."""
    global _agent_registry
    if _agent_registry is None:
        try:
            from ..services import agent_registry

            _agent_registry = agent_registry.get_registry()
        except Exception as e:
            logger.error(f"Failed to load agent registry: {e}")
            _agent_registry = False
    return _agent_registry if _agent_registry else None


@api_bp.route("/registry/agents")
def api_registry_agents():
    """Get all registered agents with details."""
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    return jsonify(
        {"agents": registry.get_agent_details(), "stats": registry.get_stats()}
    )


@api_bp.route("/registry/agents/<agent_id>")
def api_registry_agent(agent_id):
    """Get a specific agent by ID."""
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    from ..services.agent_protocol import AgentId

    agent = registry.get(AgentId(id=agent_id))

    if not agent:
        return jsonify({"error": "Agent not found"}), 404

    state = registry.get_state(agent.id)
    return jsonify({**agent.to_dict(), "state": state.to_dict() if state else None})


@api_bp.route("/registry/register", methods=["POST"])
def api_registry_register():
    """
    Register a new agent.

    JSON body:
    - name: Agent name
    - agent_type: Type (SENIOR_DEVELOPER, TESTER, etc.)
    - backend: Optional backend identifier
    """
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "name" not in data or "agent_type" not in data:
        return jsonify({"error": "Missing required fields: name, agent_type"}), 400

    from ..services.agent_protocol import create_standard_agent

    try:
        agent = create_standard_agent(
            agent_type=data["agent_type"],
            name=data["name"],
            backend=data.get("backend"),
        )
        agent_id = registry.register(agent)
        return jsonify(
            {"success": True, "agent_id": agent_id.to_dict(), "agent": agent.to_dict()}
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/registry/unregister/<agent_id>", methods=["POST"])
def api_registry_unregister(agent_id):
    """Unregister an agent."""
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    from ..services.agent_protocol import AgentId

    success = registry.unregister(AgentId(id=agent_id))

    return jsonify({"success": success})


@api_bp.route("/registry/discover")
def api_registry_discover():
    """
    Discover agents by capability.

    Query params:
    - capability: Required capability (code_generation, testing, etc.)
    - min_confidence: Minimum confidence (0.0-1.0, default 0.5)
    """
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    from ..services.agent_protocol import CapabilityCategory

    capability_str = request.args.get("capability", "")
    if not capability_str:
        return jsonify({"error": "capability parameter required"}), 400

    try:
        capability = CapabilityCategory(capability_str.lower())
    except ValueError:
        valid = [c.value for c in CapabilityCategory]
        return jsonify({"error": f"Invalid capability. Valid: {valid}"}), 400

    try:
        min_confidence = float(request.args.get("min_confidence", 0.5))
    except ValueError:
        min_confidence = 0.5

    agents = registry.discover(capability, min_confidence)

    return jsonify(
        {
            "capability": capability.value,
            "agents": [a.to_dict() for a in agents],
            "total": len(agents),
        }
    )


@api_bp.route("/registry/best-for-task", methods=["POST"])
def api_registry_best_for_task():
    """
    Find the best agent for a task.

    JSON body:
    - title: Task title
    - description: Task description
    - required_capabilities: List of capability names
    """
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    from ..services.agent_protocol import Task, CapabilityCategory

    try:
        capabilities = [
            CapabilityCategory(c.lower()) for c in data.get("required_capabilities", [])
        ]
    except ValueError as e:
        return jsonify({"error": f"Invalid capability: {e}"}), 400

    if not capabilities:
        return jsonify({"error": "At least one capability required"}), 400

    task = Task.create(
        title=data.get("title", "Untitled"),
        description=data.get("description", ""),
        required_capabilities=capabilities,
    )

    best_agent = registry.find_best_for_task(task)

    if not best_agent:
        return jsonify({"found": False, "message": "No suitable agent found"})

    return jsonify(
        {"found": True, "agent": best_agent.to_dict(), "task": task.to_dict()}
    )


@api_bp.route("/registry/heartbeat/<agent_id>", methods=["POST"])
def api_registry_heartbeat(agent_id):
    """Record a heartbeat from an agent."""
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    from ..services.agent_protocol import AgentId

    registry.heartbeat(AgentId(id=agent_id))

    return jsonify({"success": True})


@api_bp.route("/registry/stats")
def api_registry_stats():
    """Get registry statistics."""
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    return jsonify(registry.get_stats())


@api_bp.route("/registry/create-team", methods=["POST"])
def api_registry_create_team():
    """Create a standard team of agents."""
    registry = get_agent_registry()
    if not registry:
        return jsonify({"error": "Agent registry not available"}), 503

    agent_ids = registry.create_standard_team()

    return jsonify(
        {
            "success": True,
            "agents_created": len(agent_ids),
            "agent_ids": [aid.to_dict() for aid in agent_ids],
        }
    )


@api_bp.route("/registry/capabilities")
def api_registry_capabilities():
    """List all available capability categories."""
    from ..services.agent_protocol import CapabilityCategory, STANDARD_AGENT_TYPES

    return jsonify(
        {
            "capabilities": [c.value for c in CapabilityCategory],
            "agent_types": {
                k: {
                    "capabilities": [c.value for c in v["capabilities"]],
                    "description": v["description"],
                }
                for k, v in STANDARD_AGENT_TYPES.items()
            },
        }
    )


# ============ Message Bus (AGI-Readiness Phase 3) ============

# Lazy import for message bus
_message_bus = None


def get_message_bus():
    """Lazy load message bus."""
    global _message_bus
    if _message_bus is None:
        try:
            from ..services import message_bus

            _message_bus = message_bus.get_message_bus()
        except Exception as e:
            logger.error(f"Failed to load message bus: {e}")
            _message_bus = False
    return _message_bus if _message_bus else None


@api_bp.route("/messages/send", methods=["POST"])
def api_messages_send():
    """
    Send a message between agents.

    JSON body:
    - from_agent_id: Sender agent ID
    - to_agent_id: Recipient agent ID
    - message_type: Type of message
    - payload: Message content
    - priority: Optional priority (1-10, default 5)
    """
    bus = get_message_bus()
    if not bus:
        return jsonify({"error": "Message bus not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["from_agent_id", "to_agent_id", "message_type", "payload"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    from ..services.agent_protocol import AgentId, AgentMessage, MessageType

    try:
        message_type = MessageType(data["message_type"])
    except ValueError:
        valid = [t.value for t in MessageType]
        return jsonify({"error": f"Invalid message_type. Valid: {valid}"}), 400

    message = AgentMessage.create(
        message_type=message_type,
        from_agent=AgentId(id=data["from_agent_id"]),
        to_agent=AgentId(id=data["to_agent_id"]),
        payload=data["payload"],
        priority=data.get("priority", 5),
    )

    success = bus.send(message)

    return jsonify({"success": success, "message_id": message.id})


@api_bp.route("/messages/broadcast", methods=["POST"])
def api_messages_broadcast():
    """
    Broadcast a message to all agents.

    JSON body:
    - from_agent_id: Sender agent ID
    - message_type: Type of message
    - payload: Message content
    """
    bus = get_message_bus()
    if not bus:
        return jsonify({"error": "Message bus not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    from ..services.agent_protocol import AgentId, AgentMessage, MessageType

    try:
        message_type = MessageType(data["message_type"])
    except ValueError:
        return jsonify({"error": "Invalid message_type"}), 400

    message = AgentMessage.create(
        message_type=message_type,
        from_agent=AgentId(id=data["from_agent_id"]),
        to_agent=None,
        payload=data["payload"],
    )

    count = bus.broadcast(message)

    return jsonify({"success": True, "message_id": message.id, "delivered_to": count})


@api_bp.route("/messages/receive/<agent_id>")
def api_messages_receive(agent_id):
    """Receive pending messages for an agent."""
    bus = get_message_bus()
    if not bus:
        return jsonify({"error": "Message bus not available"}), 503

    from ..services.agent_protocol import AgentId

    try:
        limit = int(request.args.get("limit", 10))
    except ValueError:
        limit = 10

    messages = bus.receive_all(AgentId(id=agent_id), max_messages=limit)

    return jsonify(
        {"messages": [m.to_dict() for m in messages], "count": len(messages)}
    )


@api_bp.route("/messages/history")
def api_messages_history():
    """
    Get message history.

    Query params:
    - agent_id: Optional filter by agent
    - message_type: Optional filter by type
    - limit: Max messages (default 100)
    """
    bus = get_message_bus()
    if not bus:
        return jsonify({"error": "Message bus not available"}), 503

    from ..services.agent_protocol import AgentId, MessageType

    agent_id = None
    if request.args.get("agent_id"):
        agent_id = AgentId(id=request.args.get("agent_id"))

    message_type = None
    if request.args.get("message_type"):
        try:
            message_type = MessageType(request.args.get("message_type"))
        except ValueError:
            pass

    try:
        limit = int(request.args.get("limit", 100))
    except ValueError:
        limit = 100

    messages = bus.get_message_history(agent_id, message_type, limit)

    return jsonify(
        {"messages": [m.to_dict() for m in messages], "count": len(messages)}
    )


@api_bp.route("/messages/stats")
def api_messages_stats():
    """Get message bus statistics."""
    bus = get_message_bus()
    if not bus:
        return jsonify({"error": "Message bus not available"}), 503

    return jsonify(bus.get_stats())


@api_bp.route("/messages/dead-letters")
def api_messages_dead_letters():
    """Get failed/expired messages."""
    bus = get_message_bus()
    if not bus:
        return jsonify({"error": "Message bus not available"}), 503

    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50

    messages = bus.get_dead_letters(limit)

    return jsonify(
        {"dead_letters": [m.to_dict() for m in messages], "count": len(messages)}
    )


# ============ Translation Layer (AGI-Readiness Phase 3) ============

# Lazy import for translation layer
_translation_layer = None


def get_translation_layer():
    """Lazy load translation layer."""
    global _translation_layer
    if _translation_layer is None:
        try:
            from ..services import translation_layer

            _translation_layer = translation_layer.get_translation_layer()
        except Exception as e:
            logger.error(f"Failed to load translation layer: {e}")
            _translation_layer = False
    return _translation_layer if _translation_layer else None


@api_bp.route("/translate/message", methods=["POST"])
def api_translate_message():
    """
    Translate an agent message to human-readable format.

    JSON body:
    - message: AgentMessage dict
    """
    tl = get_translation_layer()
    if not tl:
        return jsonify({"error": "Translation layer not available"}), 503

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    from ..services.agent_protocol import AgentMessage

    try:
        message = AgentMessage.from_dict(data["message"])
        translated = tl.translate(message)
        return jsonify(translated.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/translate/activity-summary", methods=["POST"])
def api_translate_activity_summary():
    """
    Get a human-readable summary of recent agent activity.

    JSON body:
    - messages: List of AgentMessage dicts
    - time_window_minutes: Optional time window (default 60)
    """
    tl = get_translation_layer()
    if not tl:
        return jsonify({"error": "Translation layer not available"}), 503

    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Missing 'messages' field"}), 400

    from ..services.agent_protocol import AgentMessage

    try:
        messages = [AgentMessage.from_dict(m) for m in data["messages"]]
        time_window = data.get("time_window_minutes", 60)
        summary = tl.summarize_activity(messages, time_window)
        return jsonify(summary.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/translate/agent-name/<agent_type>")
def api_translate_agent_name(agent_type):
    """Get a friendly name for an agent type."""
    tl = get_translation_layer()
    if not tl:
        return jsonify({"error": "Translation layer not available"}), 503

    friendly_name = tl.get_friendly_agent_name(agent_type)
    return jsonify({"agent_type": agent_type, "friendly_name": friendly_name})


@api_bp.route("/translate/recent")
def api_translate_recent():
    """Get translated recent messages from the message bus."""
    bus = get_message_bus()
    tl = get_translation_layer()

    if not bus or not tl:
        return jsonify({"error": "Services not available"}), 503

    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        limit = 20

    messages = bus.get_message_history(limit=limit)
    translated = [tl.translate(m).to_dict() for m in messages]

    return jsonify({"messages": translated, "count": len(translated)})


# ============ Favorites ============


@api_bp.route("/favorites")
def api_favorites():
    """Get all favorites."""
    return jsonify(get_favorites())


@api_bp.route("/projects/<project_id>/favorite", methods=["POST"])
def api_toggle_project_favorite(project_id):
    """Toggle project favorite status."""
    result = toggle_favorite("project", project_id)
    return jsonify(result)


@api_bp.route("/notes/<note_id>/favorite", methods=["POST"])
def api_toggle_note_favorite(note_id):
    """Toggle note favorite status."""
    result = toggle_favorite("note", note_id)
    return jsonify(result)


# ============ Project Update ============


@api_bp.route("/projects/<project_id>", methods=["PUT"])
def api_update_project(project_id):
    """Update project details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_project(project_id, data)
    return jsonify(result)


@api_bp.route("/projects/<project_id>", methods=["DELETE"])
def api_delete_project(project_id):
    """Delete a project."""
    from ..services.data_service import delete_project

    result = delete_project(project_id)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# ============ Global Search ============


@api_bp.route("/search")
def api_search():
    """Search across all content types."""
    query = request.args.get("q", "")
    types = request.args.getlist("type") or [
        "projects",
        "notes",
        "automations",
        "agents",
    ]
    if not query:
        return jsonify({"items": []})
    results = search_all(query, types)
    return jsonify(results)


# ============ Enhanced Analytics ============


@api_bp.route("/analytics/privacy")
def api_privacy_score():
    """Get privacy score and compliance data."""
    return jsonify(get_privacy_score())


@api_bp.route("/analytics/tokens")
def api_token_usage():
    """Get token usage statistics."""
    return jsonify(get_token_usage())


@api_bp.route("/analytics/categories")
def api_category_breakdown():
    """Get audit category breakdown."""
    period = request.args.get("period", "7D")
    return jsonify(get_category_breakdown(period))


# ============ Sharing & Permissions ============


@api_bp.route("/notes/<note_id>/share", methods=["POST"])
def api_share_note(note_id):
    """Share a note with another device.

    Request body:
    {
        "owner_device": "device_id_of_owner",
        "target_device": "device_id_to_share_with",
        "permission": "view" | "edit" | "full"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    owner_device = data.get("owner_device")
    target_device = data.get("target_device")
    permission = data.get("permission", "view")

    if not owner_device or not target_device:
        return jsonify({"error": "owner_device and target_device are required"}), 400

    result = share_note(note_id, owner_device, target_device, permission)
    if "error" in result:
        return jsonify(result), 403
    return jsonify(result)


@api_bp.route("/notes/<note_id>/share", methods=["DELETE"])
def api_unshare_note(note_id):
    """Remove sharing for a note.

    Request body:
    {
        "owner_device": "device_id_of_owner",
        "target_device": "device_id_to_remove"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    owner_device = data.get("owner_device")
    target_device = data.get("target_device")

    if not owner_device or not target_device:
        return jsonify({"error": "owner_device and target_device are required"}), 400

    result = unshare_note(note_id, owner_device, target_device)
    if "error" in result:
        return jsonify(result), 403
    return jsonify(result)


@api_bp.route("/projects/<project_id>/share", methods=["POST"])
def api_share_project(project_id):
    """Share a project with another device.

    Request body:
    {
        "owner_device": "device_id_of_owner",
        "target_device": "device_id_to_share_with",
        "permission": "view" | "edit" | "full"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    owner_device = data.get("owner_device")
    target_device = data.get("target_device")
    permission = data.get("permission", "view")

    if not owner_device or not target_device:
        return jsonify({"error": "owner_device and target_device are required"}), 400

    result = share_project(project_id, owner_device, target_device, permission)
    if "error" in result:
        return jsonify(result), 403
    return jsonify(result)


@api_bp.route("/projects/<project_id>/share", methods=["DELETE"])
def api_unshare_project(project_id):
    """Remove sharing for a project."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    owner_device = data.get("owner_device")
    target_device = data.get("target_device")

    if not owner_device or not target_device:
        return jsonify({"error": "owner_device and target_device are required"}), 400

    result = unshare_project(project_id, owner_device, target_device)
    if "error" in result:
        return jsonify(result), 403
    return jsonify(result)


@api_bp.route("/devices/<device_id>/shared")
def api_get_shared_with_device(device_id):
    """Get all content shared with a device.

    Returns projects and notes where this device appears in shared_with.
    """
    result = get_shared_content_for_device(device_id)
    return jsonify(result)


@api_bp.route("/notes/<note_id>/permissions")
def api_get_note_permissions(note_id):
    """Get permissions info for a note.

    Query params:
    - device_id: The device to check permissions for
    """
    device_id = request.args.get("device_id")
    if not device_id:
        return jsonify({"error": "device_id is required"}), 400

    note = get_note(note_id)
    if not note:
        return jsonify({"error": "Note not found"}), 404

    # Get the full note data to check permissions
    from ..services.data_service import _read_json_file, NOTES_FILE

    data = _read_json_file(NOTES_FILE)
    if data:
        for dev_id, device_info in data.get("devices", {}).items():
            for n in device_info.get("notes", []):
                if n.get("id") == note_id:
                    permission = get_permission_level(n, device_id)
                    return jsonify(
                        {
                            "note_id": note_id,
                            "device_id": device_id,
                            "permission": permission,
                            "can_view": permission in ("owner", "full", "edit", "view"),
                            "can_edit": permission in ("owner", "full", "edit"),
                            "can_delete": permission in ("owner", "full"),
                            "can_share": permission == "owner",
                            "owner_device": n.get("owner_device"),
                            "shared_with": n.get("shared_with", []),
                        }
                    )

    return jsonify({"error": "Note not found"}), 404


@api_bp.route("/projects/<project_id>/permissions")
def api_get_project_permissions(project_id):
    """Get permissions info for a project."""
    device_id = request.args.get("device_id")
    if not device_id:
        return jsonify({"error": "device_id is required"}), 400

    project = get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    from ..services.data_service import _read_json_file, PROJECTS_FILE

    data = _read_json_file(PROJECTS_FILE)
    if data:
        for dev_id, device_info in data.get("devices", {}).items():
            for p in device_info.get("projects", []):
                if p.get("id") == project_id or p.get("path") == project_id:
                    permission = get_permission_level(p, device_id)
                    return jsonify(
                        {
                            "project_id": project_id,
                            "device_id": device_id,
                            "permission": permission,
                            "can_view": permission in ("owner", "full", "edit", "view"),
                            "can_edit": permission in ("owner", "full", "edit"),
                            "can_delete": permission in ("owner", "full"),
                            "can_share": permission == "owner",
                            "owner_device": p.get("owner_device"),
                            "shared_with": p.get("shared_with", []),
                        }
                    )

    return jsonify({"error": "Project not found"}), 404


# ============ Email Verification ============


@api_bp.route("/email/verify/request", methods=["POST"])
def api_request_email_verification():
    """Request email verification - sends code to email.

    Request body:
    {
        "email": "user@example.com",
        "device_id": "device_fingerprint"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email = data.get("email")
    device_id = data.get("device_id")

    if not email:
        return jsonify({"error": "email is required"}), 400
    if not device_id:
        return jsonify({"error": "device_id is required"}), 400

    result = request_email_verification(email, device_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@api_bp.route("/email/verify/confirm", methods=["POST"])
def api_verify_email_code():
    """Verify the email code.

    Request body:
    {
        "email": "user@example.com",
        "code": "123456",
        "device_id": "device_fingerprint"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email = data.get("email")
    code = data.get("code")
    device_id = data.get("device_id")

    if not email or not code or not device_id:
        return jsonify({"error": "email, code, and device_id are required"}), 400

    result = verify_email_code(email, code, device_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@api_bp.route("/email/verified")
def api_get_all_verified_emails():
    """Get all verified emails."""
    return jsonify({"emails": get_all_verified_emails()})


@api_bp.route("/email/verified/<device_id>")
def api_get_verified_email(device_id):
    """Get verified email for a specific device."""
    email = get_verified_email(device_id)
    if email:
        return jsonify({"device_id": device_id, "email": email})
    return jsonify({"device_id": device_id, "email": None})


@api_bp.route("/email/verified/<device_id>", methods=["DELETE"])
def api_remove_verified_email(device_id):
    """Remove verified email for a device."""
    result = remove_verified_email(device_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@api_bp.route("/email/devices")
def api_get_devices_by_email():
    """Get all devices that have verified a specific email.

    Query params:
    - email: The email address to look up
    """
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "email is required"}), 400

    devices = get_devices_by_email(email)
    return jsonify({"email": email, "devices": devices})


@api_bp.route("/email/config", methods=["POST"])
def api_set_email_config():
    """Configure SMTP settings for sending emails.

    Request body:
    {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_user": "your_email@example.com",
        "smtp_password": "<your_app_password>",
        "from_email": "noreply@example.com"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["smtp_host", "smtp_port", "smtp_user", "smtp_password", "from_email"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400

    result = set_email_config(
        data["smtp_host"],
        data["smtp_port"],
        data["smtp_user"],
        data["smtp_password"],
        data["from_email"],
    )
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


# ============ Team Membership ============


@api_bp.route("/teams/<team_id>/invite", methods=["POST"])
def api_invite_team_member(team_id):
    """Invite a user to a team by email.

    Request body:
    {
        "inviter_email": "owner@example.com",
        "invitee_email": "newuser@example.com",
        "role": "member" | "admin"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    inviter_email = data.get("inviter_email")
    invitee_email = data.get("invitee_email")
    role = data.get("role", "member")

    if not inviter_email or not invitee_email:
        return jsonify({"error": "inviter_email and invitee_email are required"}), 400

    result = invite_team_member(team_id, inviter_email, invitee_email, role)
    if "error" in result:
        return jsonify(result), 403
    return jsonify(result)


@api_bp.route("/teams/invitations/accept", methods=["POST"])
def api_accept_team_invitation():
    """Accept a team invitation.

    Request body:
    {
        "invitation_code": "ABCD1234",
        "email": "user@example.com",
        "device_id": "optional_device_id"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    code = data.get("invitation_code")
    email = data.get("email")
    device_id = data.get("device_id")

    if not code or not email:
        return jsonify({"error": "invitation_code and email are required"}), 400

    result = accept_team_invitation(code, email, device_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@api_bp.route("/teams/invitations/pending")
def api_get_pending_invitations():
    """Get pending team invitations for an email.

    Query params:
    - email: The email to check for invitations
    """
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "email is required"}), 400

    invitations = get_pending_invitations(email)
    return jsonify({"invitations": invitations})


@api_bp.route("/teams/<team_id>/members/<member_email>", methods=["DELETE"])
def api_remove_team_member(team_id, member_email):
    """Remove a member from a team.

    Query params:
    - remover_email: Email of the person removing (must be owner/admin)
    """
    remover_email = request.args.get("remover_email")
    if not remover_email:
        return jsonify({"error": "remover_email is required"}), 400

    result = remove_team_member(team_id, remover_email, member_email)
    if "error" in result:
        return jsonify(result), 403
    return jsonify(result)


@api_bp.route("/teams/<team_id>/members/<member_email>/role", methods=["PUT"])
def api_update_member_role(team_id, member_email):
    """Update a team member's role.

    Request body:
    {
        "updater_email": "owner@example.com",
        "new_role": "admin" | "member"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    updater_email = data.get("updater_email")
    new_role = data.get("new_role")

    if not updater_email or not new_role:
        return jsonify({"error": "updater_email and new_role are required"}), 400

    result = update_team_member_role(team_id, updater_email, member_email, new_role)
    if "error" in result:
        return jsonify(result), 403
    return jsonify(result)


@api_bp.route("/teams/my-teams")
def api_get_my_teams():
    """Get all teams the user is a member of.

    Query params:
    - email: The user's email
    """
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "email is required"}), 400

    teams = get_teams_for_email(email)
    return jsonify({"teams": teams})


# ============ Human Override System (AGI-Readiness Phase 4) ============

# Lazy import for human override system
_override_system = None


def get_override_system():
    """Lazy load human override system."""
    global _override_system
    if _override_system is None:
        try:
            from ..services import human_override

            _override_system = human_override.get_override_system()
        except Exception as e:
            logger.error(f"Failed to load override system: {e}")
            _override_system = False
    return _override_system if _override_system else None


@api_bp.route("/override/status")
def api_override_status():
    """Get current override system status."""
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    return jsonify(
        {
            "stats": ov.get_stats(),
            "kill_switch": ov.get_kill_switch_status(),
            "active_pauses": [p.to_dict() for p in ov.get_active_pauses()],
        }
    )


@api_bp.route("/override/pause", methods=["POST"])
def api_override_pause():
    """
    Pause agent(s).

    JSON body:
    - scope: Scope of pause (single_agent, agent_type, team, all_agents, system)
    - target: Optional target identifier (agent ID, type, team name)
    - reason: Reason for pause
    - paused_by: Who initiated (default: "user")
    - duration_minutes: Optional auto-expire duration
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    from ..services.human_override import OverrideScope

    scope_str = data.get("scope", "all_agents")
    try:
        scope = OverrideScope(scope_str)
    except ValueError:
        valid = [s.value for s in OverrideScope]
        return jsonify({"error": f"Invalid scope. Valid: {valid}"}), 400

    token = ov.pause(
        scope=scope,
        target=data.get("target"),
        reason=data.get("reason", ""),
        paused_by=data.get("paused_by", "user"),
        duration_minutes=data.get("duration_minutes"),
    )

    # Also add to activity log
    _add_activity("pause", f"Pause issued: {scope.value}", "pause")

    return jsonify({"success": True, "pause_token": token.to_dict()})


@api_bp.route("/override/resume", methods=["POST"])
def api_override_resume():
    """
    Resume from a pause.

    JSON body:
    - token_id: The pause token ID to cancel
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    data = request.get_json()
    if not data or "token_id" not in data:
        return jsonify({"error": "token_id is required"}), 400

    success = ov.resume(data["token_id"])

    if success:
        _add_activity("play", "Agents resumed", "play")

    return jsonify({"success": success})


@api_bp.route("/override/is-paused")
def api_override_is_paused():
    """
    Check if an agent or the system is paused.

    Query params:
    - agent_id: Optional specific agent to check
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    agent_id_str = request.args.get("agent_id")
    agent_id = None

    if agent_id_str:
        from ..services.agent_protocol import AgentId

        agent_id = AgentId(id=agent_id_str)

    is_paused = ov.is_paused(agent_id)

    return jsonify({"paused": is_paused, "agent_id": agent_id_str})


@api_bp.route("/override/kill-switch", methods=["POST"])
def api_override_kill_switch():
    """
    Activate the emergency kill switch.

    JSON body:
    - reason: Reason for activating (default: "Emergency stop")
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    data = request.get_json() or {}
    reason = data.get("reason", "Emergency stop")

    success = ov.activate_kill_switch(reason)

    if success:
        _add_activity("kill", f"KILL SWITCH ACTIVATED: {reason}", "cancel")

    return jsonify(
        {
            "success": success,
            "message": "Kill switch activated"
            if success
            else "Kill switch already active",
        }
    )


@api_bp.route("/override/kill-switch/deactivate", methods=["POST"])
def api_override_deactivate_kill_switch():
    """
    Deactivate the kill switch.

    JSON body:
    - confirmation: Must be "CONFIRM_DEACTIVATE" to proceed
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    data = request.get_json()
    if not data or "confirmation" not in data:
        return jsonify(
            {
                "error": "confirmation is required",
                "hint": "Set confirmation to 'CONFIRM_DEACTIVATE' to proceed",
            }
        ), 400

    success = ov.deactivate_kill_switch(data["confirmation"])

    if success:
        _add_activity("system", "Kill switch deactivated", "play")

    return jsonify(
        {
            "success": success,
            "message": "Kill switch deactivated" if success else "Invalid confirmation",
        }
    )


@api_bp.route("/override/decision", methods=["POST"])
def api_override_request_decision():
    """
    Request human approval for a decision.

    JSON body:
    - agent_id: Agent requesting approval
    - decision_type: Type (file_access, network_access, code_execution, etc.)
    - action: Description of the action
    - context: Additional context dict
    - risk_level: Risk level (low, medium, high, critical)
    - timeout_minutes: How long before decision expires (default 30)
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["agent_id", "decision_type", "action"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    from ..services.agent_protocol import AgentId
    from ..services.human_override import DecisionType

    try:
        decision_type = DecisionType(data["decision_type"])
    except ValueError:
        valid = [d.value for d in DecisionType]
        return jsonify({"error": f"Invalid decision_type. Valid: {valid}"}), 400

    decision = ov.request_decision(
        agent_id=AgentId(id=data["agent_id"]),
        decision_type=decision_type,
        action=data["action"],
        context=data.get("context", {}),
        risk_level=data.get("risk_level", "medium"),
        timeout_minutes=data.get("timeout_minutes", 30),
    )

    _add_activity("question", f"Decision requested: {data['action'][:50]}", "help")

    return jsonify({"success": True, "decision": decision.to_dict()})


@api_bp.route("/override/decisions")
def api_override_pending_decisions():
    """
    Get pending decisions.

    Query params:
    - agent_id: Optional filter by agent
    - decision_type: Optional filter by type
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    from ..services.agent_protocol import AgentId
    from ..services.human_override import DecisionType

    agent_id = None
    if request.args.get("agent_id"):
        agent_id = AgentId(id=request.args.get("agent_id"))

    decision_type = None
    if request.args.get("decision_type"):
        try:
            decision_type = DecisionType(request.args.get("decision_type"))
        except ValueError:
            pass

    decisions = ov.get_pending_decisions(agent_id, decision_type)

    return jsonify(
        {"decisions": [d.to_dict() for d in decisions], "count": len(decisions)}
    )


@api_bp.route("/override/decisions/<decision_id>")
def api_override_decision_status(decision_id):
    """Get the status of a specific decision."""
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    decision = ov.get_decision_status(decision_id)

    if not decision:
        return jsonify({"error": "Decision not found"}), 404

    return jsonify(decision.to_dict())


@api_bp.route("/override/decisions/<decision_id>/approve", methods=["POST"])
def api_override_approve_decision(decision_id):
    """
    Approve a pending decision.

    JSON body:
    - approved_by: Who approved (default: "user")
    - reason: Optional reason for approval
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    data = request.get_json() or {}

    success = ov.approve_decision(
        decision_id=decision_id,
        approved_by=data.get("approved_by", "user"),
        reason=data.get("reason"),
    )

    if success:
        _add_activity("check", f"Decision {decision_id} approved", "check")

    return jsonify({"success": success})


@api_bp.route("/override/decisions/<decision_id>/deny", methods=["POST"])
def api_override_deny_decision(decision_id):
    """
    Deny a pending decision.

    JSON body:
    - denied_by: Who denied (default: "user")
    - reason: Optional reason for denial
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    data = request.get_json() or {}

    success = ov.deny_decision(
        decision_id=decision_id,
        denied_by=data.get("denied_by", "user"),
        reason=data.get("reason"),
    )

    if success:
        _add_activity("cancel", f"Decision {decision_id} denied", "cancel")

    return jsonify({"success": success})


@api_bp.route("/override/inspect/<agent_id>")
def api_override_inspect(agent_id):
    """Get detailed inspection of an agent."""
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    from ..services.agent_protocol import AgentId

    registry = get_agent_registry()
    bus = get_message_bus()

    inspection = ov.inspect(
        agent_id=AgentId(id=agent_id), registry=registry, message_bus=bus
    )

    return jsonify(inspection.to_dict())


@api_bp.route("/override/action", methods=["POST"])
def api_override_record_action():
    """
    Record an agent action for inspection history.

    JSON body:
    - agent_id: Agent ID
    - action: Action description
    - details: Action details dict
    """
    ov = get_override_system()
    if not ov:
        return jsonify({"error": "Override system not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["agent_id", "action", "details"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    from ..services.agent_protocol import AgentId

    ov.record_action(
        agent_id=AgentId(id=data["agent_id"]),
        action=data["action"],
        details=data["details"],
    )

    return jsonify({"success": True})


# ============ Permission Policy (AGI-Readiness Phase 4) ============

# Lazy import for permission policy
_permission_policy = None


def get_permission_policy():
    """Lazy load permission policy."""
    global _permission_policy
    if _permission_policy is None:
        try:
            from ..services import permission_policy

            _permission_policy = permission_policy.get_permission_policy()
        except Exception as e:
            logger.error(f"Failed to load permission policy: {e}")
            _permission_policy = False
    return _permission_policy if _permission_policy else None


@api_bp.route("/permissions/check", methods=["POST"])
def api_permissions_check():
    """
    Check if an agent has permission to perform an action.

    JSON body:
    - agent_id: Agent ID
    - action: Action category (read_file, write_file, execute_code, etc.)
    - resource: Resource being accessed
    - resource_type: Type of resource (file, api, database)
    - context: Optional additional context
    """
    pp = get_permission_policy()
    if not pp:
        return jsonify({"error": "Permission policy not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["agent_id", "action", "resource"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    from ..services.agent_protocol import AgentId
    from ..services.permission_policy import ActionCategory

    try:
        action = ActionCategory(data["action"])
    except ValueError:
        valid = [a.value for a in ActionCategory]
        return jsonify({"error": f"Invalid action. Valid: {valid}"}), 400

    result = pp.check_permission(
        agent_id=AgentId(id=data["agent_id"]),
        action=action,
        resource=data["resource"],
        resource_type=data.get("resource_type", "file"),
        context=data.get("context"),
    )

    return jsonify(result.to_dict())


@api_bp.route("/permissions/rules")
def api_permissions_rules():
    """List all permission rules."""
    pp = get_permission_policy()
    if not pp:
        return jsonify({"error": "Permission policy not available"}), 503

    rules = pp.list_rules()
    return jsonify({"rules": [r.to_dict() for r in rules], "count": len(rules)})


@api_bp.route("/permissions/rules/<rule_id>")
def api_permissions_rule(rule_id):
    """Get a specific permission rule."""
    pp = get_permission_policy()
    if not pp:
        return jsonify({"error": "Permission policy not available"}), 503

    rule = pp.get_rule(rule_id)
    if not rule:
        return jsonify({"error": "Rule not found"}), 404

    return jsonify(rule.to_dict())


@api_bp.route("/permissions/rules/<rule_id>", methods=["DELETE"])
def api_permissions_delete_rule(rule_id):
    """Delete a permission rule."""
    pp = get_permission_policy()
    if not pp:
        return jsonify({"error": "Permission policy not available"}), 503

    success = pp.remove_rule(rule_id)
    return jsonify({"success": success})


@api_bp.route("/permissions/trust", methods=["POST"])
def api_permissions_set_trust():
    """
    Set trust level for an agent.

    JSON body:
    - agent_id: Agent ID
    - trust_level: Trust level (0-5 or name: UNTRUSTED, LIMITED, STANDARD, ELEVATED, TRUSTED, SYSTEM)
    """
    pp = get_permission_policy()
    if not pp:
        return jsonify({"error": "Permission policy not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "agent_id" not in data or "trust_level" not in data:
        return jsonify({"error": "agent_id and trust_level are required"}), 400

    from ..services.agent_protocol import AgentId
    from ..services.permission_policy import TrustLevel

    trust_input = data["trust_level"]
    try:
        if isinstance(trust_input, int):
            trust_level = TrustLevel(trust_input)
        else:
            trust_level = TrustLevel[trust_input.upper()]
    except (ValueError, KeyError):
        valid = [f"{t.name} ({t.value})" for t in TrustLevel]
        return jsonify({"error": f"Invalid trust_level. Valid: {valid}"}), 400

    pp.set_trust_level(AgentId(id=data["agent_id"]), trust_level)

    return jsonify(
        {"success": True, "agent_id": data["agent_id"], "trust_level": trust_level.name}
    )


@api_bp.route("/permissions/trust/<agent_id>")
def api_permissions_get_trust(agent_id):
    """Get trust level for an agent."""
    pp = get_permission_policy()
    if not pp:
        return jsonify({"error": "Permission policy not available"}), 503

    from ..services.agent_protocol import AgentId

    level = pp.get_trust_level(AgentId(id=agent_id))

    return jsonify(
        {"agent_id": agent_id, "trust_level": level.name, "trust_value": level.value}
    )


@api_bp.route("/permissions/history")
def api_permissions_history():
    """Get permission check history."""
    pp = get_permission_policy()
    if not pp:
        return jsonify({"error": "Permission policy not available"}), 503

    try:
        limit = int(request.args.get("limit", 100))
    except ValueError:
        limit = 100

    history = pp.get_check_history(limit)

    return jsonify({"history": history, "count": len(history)})


@api_bp.route("/permissions/stats")
def api_permissions_stats():
    """Get permission policy statistics."""
    pp = get_permission_policy()
    if not pp:
        return jsonify({"error": "Permission policy not available"}), 503

    return jsonify(pp.get_stats())


@api_bp.route("/permissions/action-categories")
def api_permissions_action_categories():
    """List all available action categories."""
    from ..services.permission_policy import ActionCategory, TrustLevel

    return jsonify(
        {
            "action_categories": [a.value for a in ActionCategory],
            "trust_levels": [{"name": t.name, "value": t.value} for t in TrustLevel],
        }
    )


# ============ Rate Limiter (AGI-Readiness Phase 4) ============

# Lazy import for rate limiter
_rate_limiter = None


def get_rate_limiter():
    """Lazy load rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        try:
            from ..services import rate_limiter

            _rate_limiter = rate_limiter.get_rate_limiter()
        except Exception as e:
            logger.error(f"Failed to load rate limiter: {e}")
            _rate_limiter = False
    return _rate_limiter if _rate_limiter else None


@api_bp.route("/ratelimit/check", methods=["POST"])
def api_ratelimit_check():
    """
    Check if a request is within rate limits.

    JSON body:
    - agent_id: Agent ID
    - action: Action being performed
    - dry_run: If true, don't record the request (optional, default false)
    """
    rl = get_rate_limiter()
    if not rl:
        return jsonify({"error": "Rate limiter not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "agent_id" not in data or "action" not in data:
        return jsonify({"error": "agent_id and action are required"}), 400

    from ..services.agent_protocol import AgentId

    status = rl.check_rate_limit(
        agent_id=AgentId(id=data["agent_id"]),
        action=data["action"],
        record=not data.get("dry_run", False),
    )

    return jsonify(status.to_dict())


@api_bp.route("/ratelimit/configs")
def api_ratelimit_configs():
    """List all rate limit configurations."""
    rl = get_rate_limiter()
    if not rl:
        return jsonify({"error": "Rate limiter not available"}), 503

    configs = rl.list_configs()
    return jsonify({"configs": [c.to_dict() for c in configs], "count": len(configs)})


@api_bp.route("/ratelimit/configs/<config_id>")
def api_ratelimit_config(config_id):
    """Get a specific rate limit config."""
    rl = get_rate_limiter()
    if not rl:
        return jsonify({"error": "Rate limiter not available"}), 503

    config = rl.get_config(config_id)
    if not config:
        return jsonify({"error": "Config not found"}), 404

    return jsonify(config.to_dict())


@api_bp.route("/ratelimit/configs/<config_id>", methods=["DELETE"])
def api_ratelimit_delete_config(config_id):
    """Delete a rate limit config."""
    rl = get_rate_limiter()
    if not rl:
        return jsonify({"error": "Rate limiter not available"}), 503

    success = rl.remove_config(config_id)
    return jsonify({"success": success})


@api_bp.route("/ratelimit/usage/<agent_id>")
def api_ratelimit_usage(agent_id):
    """Get usage statistics for an agent."""
    rl = get_rate_limiter()
    if not rl:
        return jsonify({"error": "Rate limiter not available"}), 503

    from ..services.agent_protocol import AgentId

    usage = rl.get_usage(AgentId(id=agent_id))
    return jsonify(usage)


@api_bp.route("/ratelimit/reset/<agent_id>", methods=["POST"])
def api_ratelimit_reset(agent_id):
    """Reset rate limits for an agent."""
    rl = get_rate_limiter()
    if not rl:
        return jsonify({"error": "Rate limiter not available"}), 503

    from ..services.agent_protocol import AgentId

    rl.reset_agent_limits(AgentId(id=agent_id))
    return jsonify({"success": True})


@api_bp.route("/ratelimit/stats")
def api_ratelimit_stats():
    """Get rate limiter statistics."""
    rl = get_rate_limiter()
    if not rl:
        return jsonify({"error": "Rate limiter not available"}), 503

    return jsonify(rl.get_stats())


# ============ Safety Audit (AGI-Readiness Phase 4) ============

# Lazy import for safety audit
_safety_audit = None


def get_safety_audit():
    """Lazy load safety audit."""
    global _safety_audit
    if _safety_audit is None:
        try:
            from ..services import safety_audit

            _safety_audit = safety_audit.get_safety_audit()
        except Exception as e:
            logger.error(f"Failed to load safety audit: {e}")
            _safety_audit = False
    return _safety_audit if _safety_audit else None


@api_bp.route("/safety/events")
def api_safety_events():
    """
    Get safety events.

    Query params:
    - event_type: Optional filter by event type
    - risk_level: Optional filter by risk level
    - agent_id: Optional filter by agent
    - hours: Look back period in hours (default 24)
    - limit: Maximum events (default 100)
    """
    sa = get_safety_audit()
    if not sa:
        return jsonify({"error": "Safety audit not available"}), 503

    from ..services.safety_audit import SafetyEventType, RiskLevel

    event_types = None
    if request.args.get("event_type"):
        try:
            event_types = [SafetyEventType(request.args.get("event_type"))]
        except ValueError:
            pass

    risk_levels = None
    if request.args.get("risk_level"):
        try:
            risk_levels = [RiskLevel[request.args.get("risk_level").upper()]]
        except KeyError:
            pass

    agent_id = request.args.get("agent_id")

    try:
        hours = int(request.args.get("hours", 24))
    except ValueError:
        hours = 24

    try:
        limit = int(request.args.get("limit", 100))
    except ValueError:
        limit = 100

    since = datetime.now() - timedelta(hours=hours)

    events = sa.get_events(
        event_types=event_types,
        risk_levels=risk_levels,
        agent_id=agent_id,
        since=since,
        limit=limit,
    )

    return jsonify({"events": [e.to_dict() for e in events], "count": len(events)})


@api_bp.route("/safety/event", methods=["POST"])
def api_safety_log_event():
    """
    Log a safety event.

    JSON body:
    - event_type: Type of event
    - risk_level: Risk level (MINIMAL, LOW, MEDIUM, HIGH, CRITICAL)
    - agent_id: Optional agent ID
    - action: Optional action description
    - resource: Optional resource
    - decision: Optional decision
    - details: Optional additional details dict
    - success: Whether operation succeeded (default true)
    - error_message: Error message if failed
    """
    sa = get_safety_audit()
    if not sa:
        return jsonify({"error": "Safety audit not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "event_type" not in data:
        return jsonify({"error": "event_type is required"}), 400

    from ..services.safety_audit import SafetyEventType, RiskLevel

    try:
        event_type = SafetyEventType(data["event_type"])
    except ValueError:
        valid = [e.value for e in SafetyEventType]
        return jsonify({"error": f"Invalid event_type. Valid: {valid}"}), 400

    risk_level = RiskLevel.LOW
    if data.get("risk_level"):
        try:
            risk_level = RiskLevel[data["risk_level"].upper()]
        except KeyError:
            valid = [r.name for r in RiskLevel]
            return jsonify({"error": f"Invalid risk_level. Valid: {valid}"}), 400

    event = sa.log_event(
        event_type=event_type,
        risk_level=risk_level,
        agent_id=data.get("agent_id"),
        action=data.get("action"),
        resource=data.get("resource"),
        decision=data.get("decision"),
        details=data.get("details"),
        success=data.get("success", True),
        error_message=data.get("error_message"),
    )

    return jsonify({"success": True, "event": event.to_dict()})


@api_bp.route("/safety/risk-scores")
def api_safety_risk_scores():
    """Get all agent risk scores."""
    sa = get_safety_audit()
    if not sa:
        return jsonify({"error": "Safety audit not available"}), 503

    scores = sa.get_all_risk_scores()

    return jsonify({"scores": [s.to_dict() for s in scores], "count": len(scores)})


@api_bp.route("/safety/risk-scores/<agent_id>")
def api_safety_risk_score(agent_id):
    """Get risk score for a specific agent."""
    sa = get_safety_audit()
    if not sa:
        return jsonify({"error": "Safety audit not available"}), 503

    score = sa.get_risk_score(agent_id)

    if not score:
        return jsonify(
            {
                "agent_id": agent_id,
                "current_score": 0.0,
                "message": "No data for this agent",
            }
        )

    return jsonify(score.to_dict())


@api_bp.route("/safety/report")
def api_safety_report():
    """
    Generate a safety report.

    Query params:
    - hours: Report period in hours (default 24)
    """
    sa = get_safety_audit()
    if not sa:
        return jsonify({"error": "Safety audit not available"}), 503

    try:
        hours = int(request.args.get("hours", 24))
    except ValueError:
        hours = 24

    report = sa.generate_report(hours=hours)

    return jsonify(report.to_dict())


@api_bp.route("/safety/stats")
def api_safety_stats():
    """Get safety audit statistics."""
    sa = get_safety_audit()
    if not sa:
        return jsonify({"error": "Safety audit not available"}), 503

    return jsonify(sa.get_stats())


@api_bp.route("/safety/event-types")
def api_safety_event_types():
    """List all available event types and risk levels."""
    from ..services.safety_audit import SafetyEventType, RiskLevel

    return jsonify(
        {
            "event_types": [e.value for e in SafetyEventType],
            "risk_levels": [{"name": r.name, "value": r.value} for r in RiskLevel],
        }
    )


# ============ Data Cleanup ============


@api_bp.route("/cleanup/stale", methods=["POST"])
def api_cleanup_stale():
    """
    Remove stale projects, notes, and automations.
    Keeps data from devices active in last 30 days.
    """
    from ..services.data_service import cleanup_stale_data

    try:
        result = cleanup_stale_data()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Cleanup stale data error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/cleanup/devices", methods=["POST"])
def api_cleanup_devices():
    """
    Reset device history, keeping only the most recent device.
    """
    from ..services.data_service import reset_device_history

    try:
        result = reset_device_history()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Reset device history error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============ Image Generation Service ============


@api_bp.route("/image/status")
def api_image_status():
    """Get status of image generation services."""
    try:
        from ..services.image_service import get_image_service_status

        return jsonify(get_image_service_status())
    except Exception as e:
        logger.error(f"Image service status error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/image/generate", methods=["POST"])
@rate_limit
def api_image_generate():
    """
    Generate an image from a text prompt.

    Request body:
    {
        "prompt": "A beautiful sunset over mountains",
        "negative_prompt": "blurry, bad quality",  // optional
        "width": 1024,  // optional, default 1024
        "height": 1024,  // optional, default 1024
        "steps": 20,  // optional, default 20
        "guidance_scale": 7.5,  // optional
        "seed": 12345,  // optional, for reproducibility
        "model": "sd-xl-turbo"  // optional: sd-1.5, sd-xl, sd-xl-turbo
    }

    Returns:
    {
        "success": true,
        "image": "<base64 PNG>",
        "seed": 12345,
        "generation_time_ms": 2500,
        "width": 1024,
        "height": 1024,
        "model": "sd-xl-turbo"
    }
    """
    try:
        from ..services.image_service import get_image_generation_service

        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"success": False, "error": "prompt is required"}), 400

        service = get_image_generation_service()
        result = service.generate_image(
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt"),
            width=data.get("width", 1024),
            height=data.get("height", 1024),
            steps=data.get("steps", 20),
            guidance_scale=data.get("guidance_scale", 7.5),
            seed=data.get("seed"),
            model=data.get("model"),
            source="api",
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/image/setup", methods=["POST"])
def api_image_setup():
    """Ensure image generation dependencies and models are installed."""
    try:
        from ..services.image_service import start_image_setup

        data = request.get_json(silent=True) or {}
        model_id = data.get("model")
        return jsonify(start_image_setup(model_id))
    except Exception as e:
        logger.error(f"Image setup error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/image/setup/status")
def api_image_setup_status():
    """Get status of image generation setup."""
    try:
        from ..services.image_service import get_image_setup_status

        return jsonify(get_image_setup_status())
    except Exception as e:
        logger.error(f"Image setup status error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/image/inpaint", methods=["POST"])
@rate_limit
def api_image_inpaint():
    """
    Inpaint (regenerate) part of an image using a mask.

    Request body:
    {
        "image": "<base64 original image>",
        "mask": "<base64 mask - white = area to regenerate>",
        "prompt": "A red rose",
        "negative_prompt": "blurry",  // optional
        "steps": 20,  // optional
        "guidance_scale": 7.5  // optional
    }

    Returns:
    {
        "success": true,
        "image": "<base64 PNG>",
        "generation_time_ms": 3000
    }
    """
    try:
        from ..services.image_service import get_image_generation_service

        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Request body required"}), 400

        required = ["image", "mask", "prompt"]
        for field in required:
            if field not in data:
                return jsonify({"success": False, "error": f"{field} is required"}), 400

        service = get_image_generation_service()
        result = service.inpaint_image(
            image_base64=data["image"],
            mask_base64=data["mask"],
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt"),
            steps=data.get("steps", 20),
            guidance_scale=data.get("guidance_scale", 7.5),
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Inpainting error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/image/remove-background", methods=["POST"])
@rate_limit
def api_image_remove_background():
    """
    Remove background from an image.

    Request body:
    {
        "image": "<base64 image>"
    }

    Returns:
    {
        "success": true,
        "image": "<base64 PNG with transparent background>",
        "processing_time_ms": 1500
    }
    """
    try:
        from ..services.image_service import get_bg_removal_service

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "image is required"}), 400

        service = get_bg_removal_service()
        result = service.remove_background(data["image"])

        return jsonify(result)

    except Exception as e:
        logger.error(f"Background removal error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============ Batch Image Generation Endpoints ============


@api_bp.route("/image/batch", methods=["POST"])
@rate_limit
def api_batch_generate():
    """
    Start a batch image generation job.
    """
    try:
        from ..services.image_service import start_batch_generation

        data = request.get_json()
        print(f"[DEBUG] Batch generate request data: {data}")
        
        if not data or "prompt" not in data:
            return jsonify({"success": False, "error": "prompt is required"}), 400

        print(f"[DEBUG] Starting batch generation for prompt: {data['prompt']}")
        result = start_batch_generation(
            prompt=data["prompt"],
            count=data.get("count", 1),
            variation_strength=data.get("variation_strength", 0.3),
            negative_prompt=data.get("negative_prompt"),
            model=data.get("model", "sd-xl-turbo"),
            width=data.get("width", 1024),
            height=data.get("height", 1024),
            steps=data.get("steps"),
            guidance_scale=data.get("guidance_scale", 7.5),
            base_seed=data.get("base_seed"),
        )
        print(f"[DEBUG] Batch generation result: {result}")

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Batch generation error: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Batch generation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/image/batch/<job_id>")
def api_batch_status(job_id):
    """Get status of a batch job."""
    try:
        from ..services.image_service import get_batch_status

        status = get_batch_status(job_id)
        if status is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({"success": True, "job": status})

    except Exception as e:
        logger.error(f"Batch status error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/image/batch/<job_id>/results")
def api_batch_results(job_id):
    """
    Get results of a batch job.

    Query params:
        include_images: true/false (default true) - include base64 image data
    """
    try:
        from ..services.image_service import get_batch_results

        include_images = request.args.get("include_images", "true").lower() == "true"
        results = get_batch_results(job_id, include_images=include_images)

        if results is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({"success": True, **results})

    except Exception as e:
        logger.error(f"Batch results error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/image/batch/<job_id>", methods=["DELETE"])
def api_batch_cancel(job_id):
    """Cancel a running batch job."""
    try:
        from ..services.image_service import cancel_batch

        result = cancel_batch(job_id)
        status_code = 200 if result.get("success") else 400
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Batch cancel error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/image/batch/list")
def api_batch_list():
    """List recent batch jobs."""
    try:
        from ..services.image_service import list_batch_jobs

        limit = request.args.get("limit", 20, type=int)
        jobs = list_batch_jobs(limit=limit)

        return jsonify({"success": True, "jobs": jobs})

    except Exception as e:
        logger.error(f"Batch list error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/image/history")
def api_image_history():
    """Return recent image generations (single + batch)."""
    try:
        limit = int(request.args.get("limit", 50))
        from ..services.image_service import get_image_history

        images = get_image_history(limit=limit)
        for img in images:
            img["stream_url"] = f"/api/image/file/{img.get('id')}"
        return jsonify({"success": True, "images": images})
    except Exception as e:
        logger.error(f"Image history error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/image/file/<image_id>")
def api_image_file(image_id):
    """Serve a generated image file from local history."""
    try:
        from ..services.image_service import get_image_record

        record = get_image_record(image_id)
        if not record:
            return jsonify({"error": "Image not found"}), 404
        path = record.get("image_path")
        if not path or not os.path.exists(path):
            return jsonify({"error": "Image file missing"}), 404
        return send_file(path, mimetype="image/png", as_attachment=False)
    except Exception as e:
        logger.error(f"Image file error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/image/estimate")
def api_image_estimate():
    """
    Get time estimate for batch generation.

    Query params:
        count: Number of images (default 1)
        model: Model ID (default sd-xl-turbo)
        steps: Inference steps (optional)
    """
    try:
        from ..services.image_service import estimate_batch_time

        count = request.args.get("count", 1, type=int)
        model = request.args.get("model", "sd-xl-turbo")
        steps = request.args.get("steps", type=int)

        estimated_seconds = estimate_batch_time(count, model, steps)

        return jsonify(
            {
                "success": True,
                "count": count,
                "model": model,
                "estimated_seconds": round(estimated_seconds, 1),
                "estimated_minutes": round(estimated_seconds / 60, 2),
            }
        )

    except Exception as e:
        logger.error(f"Estimate error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
