"""
REST API endpoints for Shadow Web Dashboard
"""
from flask import Blueprint, jsonify, request
import subprocess
import socket
import json
import os
import time
from functools import lru_cache
from threading import Lock

from ..services.data_service import (
    get_note_content as get_cached_note_content,
    save_note_content,
    decrypt_note_content,
    encrypt_note_content,
    get_devices, get_device,
    get_projects, get_project, update_project,
    get_notes, get_note,
    get_automations, get_automation, get_automation_logs,
    get_agents, get_agent, get_agent_metrics, add_agent, update_agent, delete_agent,
    get_teams, get_team, create_team, update_team, delete_team, get_team_metrics,
    get_tasks, get_task, create_task, update_task, delete_task,
    get_workflows, start_workflow, cancel_workflow,
    get_audits, get_audit_entry, get_audit_stats, get_audit_traces, export_audit_report,
    get_favorites, toggle_favorite,
    search_all,
    get_privacy_score, get_token_usage, get_category_breakdown,
    get_usage_stats, get_backend_usage, get_activity_timeline,
    get_status
)

api_bp = Blueprint('api', __name__)

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
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB receive buffer

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


def _try_note_request(device_ips: list[str], port: int, payload: dict, timeout_s: int, retries: int) -> dict:
    """Try multiple device IPs with retries, return first successful response."""
    last_error: Exception | None = None
    for host in device_ips:
        for attempt in range(retries):
            try:
                return _send_note_request(host, port, payload, timeout_s)
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(0.2)
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
    device_ips: list[str],
    ports: list[int],
    payload: dict,
    timeout_s: int,
    retries: int
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
            device_ips,
            ports,
            {"action": "sync_notes"},
            timeout_s=6,
            retries=1
        )
        return bool(response.get("success", True))
    except Exception:
        return False


# ============ Devices ============

@api_bp.route('/devices')
def api_devices():
    """List all devices with status."""
    return jsonify(get_devices())


@api_bp.route('/devices/<device_id>')
def api_device(device_id):
    """Get device details."""
    device = get_device(device_id)
    if device:
        return jsonify(device)
    return jsonify({"error": "Device not found"}), 404


# ============ Projects ============

@api_bp.route('/projects')
def api_projects():
    """List all projects."""
    device_id = request.args.get('device_id')
    return jsonify(get_projects(device_id))


@api_bp.route('/projects', methods=['POST'])
def api_create_project():
    """Create a new project."""
    from ..services.data_service import create_project
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400
    device_id = data.get("device_id", "web")
    result = create_project(data, device_id)
    return jsonify(result)


@api_bp.route('/projects/<project_id>')
def api_project(project_id):
    """Get project details."""
    project = get_project(project_id)
    if project:
        return jsonify(project)
    return jsonify({"error": "Project not found"}), 404


@api_bp.route('/projects/<project_id>/open', methods=['POST'])
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


@api_bp.route('/projects/sync', methods=['POST'])
def api_projects_sync():
    """
    Receive projects sync notification and broadcast to WebSocket clients.
    Called by ShadowBridge after syncing projects from Android app.
    """
    try:
        from .websocket import broadcast_projects_updated

        data = request.get_json() or {}
        device_id = data.get('device_id', 'unknown')

        # Broadcast to all connected WebSocket clients
        broadcast_projects_updated(device_id)

        return jsonify({"success": True, "message": "Projects sync broadcasted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ Notes ============

@api_bp.route('/notes')
def api_notes():
    """List all note titles."""
    device_id = request.args.get('device_id')
    search = request.args.get('search')
    return jsonify(get_notes(device_id, search))


@api_bp.route('/notes', methods=['POST'])
def api_create_note():
    """Create a new note."""
    from ..services.data_service import create_note
    data = request.get_json()
    if not data or not data.get("title"):
        return jsonify({"error": "Title is required"}), 400
    device_id = data.get("device_id", "web")
    result = create_note(data, device_id)
    return jsonify(result)


@api_bp.route('/notes/<note_id>')
def api_note(note_id):
    """Get note metadata."""
    note = get_note(note_id)
    if note:
        return jsonify(note)
    return jsonify({"error": "Note not found"}), 404


@api_bp.route('/notes/<note_id>/content')
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
        request = {
            "action": "fetch_note",
            "note_id": note_id
        }
        response = _try_note_request(device_ips, port, request, timeout_s=15, retries=2)
        if response.get("success"):
            # Decrypt content if encrypted (SYNC_ENC:... format)
            raw_content = response.get("content", "")
            is_encrypted = response.get("encrypted", False) or raw_content.startswith("SYNC_ENC:")
            if is_encrypted and device_id:
                content = decrypt_note_content(raw_content, device_id)
            else:
                content = raw_content

            result = {
                "id": response.get("id"),
                "title": response.get("title"),
                "content": content,
                "updated_at": response.get("updatedAt")
            }
            # Cache in memory for faster subsequent access
            _cache_note_content(note_id, result)
            # Also save to local file cache for offline access (save decrypted)
            save_note_content(
                note_id,
                result.get("title", ""),
                content,
                result.get("updated_at", 0)
            )
            return jsonify(result)
        else:
            return jsonify({"error": response.get("error", "Unknown error")}), 500

    except socket.timeout:
        return jsonify({"error": f"Device connection timeout (tried: {', '.join(device_ips)})"}), 504
    except ConnectionRefusedError:
        return jsonify({"error": f"Device not accepting connections (tried: {', '.join(device_ips)})"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/notes/<note_id>/content', methods=['PUT'])
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
            "title": new_title
        }
        response = _try_note_request(device_ips, port, update_request, timeout_s=12, retries=2)
        if response.get("success"):
            # Invalidate cache so next fetch gets fresh content
            _invalidate_note_cache(note_id)
            return jsonify({
                "success": True,
                "id": note_id,
                "message": "Note updated successfully"
            })
        else:
            return jsonify({"error": response.get("error", "Failed to update note")}), 500

    except socket.timeout:
        return jsonify({"error": f"Device connection timeout (tried: {', '.join(device_ips)})"}), 504
    except ConnectionRefusedError:
        return jsonify({"error": f"Device not accepting connections (tried: {', '.join(device_ips)})"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/notes/<note_id>/export', methods=['POST'])
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
            retries=2
        )
        if not response.get('success'):
            return jsonify({"error": response.get('message', 'Failed to fetch note')}), 502

        content = response.get('content', '')
        title = response.get('title', note.get('title', 'Untitled'))

        # Create export directory
        export_dir = Path.home() / "Downloads" / "Shadow_Notes"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_title = "".join(c for c in title if c.isalnum() or c in ' -_').strip()[:50]
        if not safe_title:
            safe_title = "note"

        # Save file
        file_path = export_dir / f"{safe_title}.md"
        counter = 1
        while file_path.exists():
            file_path = export_dir / f"{safe_title}_{counter}.md"
            counter += 1

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(content)

        return jsonify({
            "success": True,
            "path": str(file_path),
            "filename": file_path.name
        })

    except socket.timeout:
        return jsonify({"error": "Device connection timeout"}), 504
    except ConnectionRefusedError:
        return jsonify({"error": "Device not connected"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/notes/sync', methods=['POST'])
def api_notes_sync():
    """
    Receive notes sync notification and broadcast to WebSocket clients.
    Called by Android app after syncing notes to ShadowBridge.
    """
    try:
        from .websocket import broadcast_notes_updated

        data = request.get_json() or {}
        device_id = data.get('device_id', 'unknown')

        # Broadcast to all connected WebSocket clients
        broadcast_notes_updated(device_id)

        return jsonify({"success": True, "message": "Notes sync broadcasted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ Automations ============

@api_bp.route('/automations')
def api_automations():
    """List all automations."""
    device_id = request.args.get('device_id')
    return jsonify(get_automations(device_id))


@api_bp.route('/automations', methods=['POST'])
def api_create_automation():
    """Create a new automation."""
    from ..services.data_service import create_automation
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400
    device_id = data.get("device_id", "web")
    result = create_automation(data, device_id)
    return jsonify(result)


@api_bp.route('/automations/<automation_id>')
def api_automation(automation_id):
    """Get automation details."""
    auto = get_automation(automation_id)
    if auto:
        return jsonify(auto)
    return jsonify({"error": "Automation not found"}), 404


@api_bp.route('/automations/<automation_id>/logs')
def api_automation_logs(automation_id):
    """Get automation execution logs."""
    logs = get_automation_logs(automation_id)
    return jsonify(logs)


@api_bp.route('/automations/<automation_id>/run', methods=['POST'])
def api_run_automation(automation_id):
    """Trigger manual automation run."""
    auto = get_automation(automation_id)
    if not auto:
        return jsonify({"error": "Automation not found"}), 404

    # TODO: Send trigger command to device
    return jsonify({"error": "Not implemented"}), 501


# ============ Agents ============

@api_bp.route('/agents')
def api_agents():
    """List all agents."""
    device_id = request.args.get('device_id')
    return jsonify(get_agents(device_id))


@api_bp.route('/agents/<agent_id>')
def api_agent(agent_id):
    """Get agent details."""
    agent = get_agent(agent_id)
    if agent:
        return jsonify(agent)
    return jsonify({"error": "Agent not found"}), 404


@api_bp.route('/agents/<agent_id>/tasks')
def api_agent_tasks(agent_id):
    """Get agent's task history."""
    # TODO: Implement task history
    return jsonify([])


@api_bp.route('/agents/metrics')
def api_agent_metrics():
    """Get aggregate agent metrics."""
    return jsonify(get_agent_metrics())


# ============ Analytics ============

@api_bp.route('/analytics/usage')
def api_usage():
    """Get message usage stats."""
    return jsonify(get_usage_stats())


@api_bp.route('/analytics/backends')
def api_backends():
    """Get backend usage breakdown."""
    return jsonify(get_backend_usage())


@api_bp.route('/analytics/activity')
def api_activity():
    """Get activity timeline."""
    return jsonify(get_activity_timeline())


# ============ Status ============

@api_bp.route('/status')
def api_status():
    """Get system status."""
    return jsonify(get_status())


@api_bp.route('/qr')
def api_qr():
    """Get QR code image for device setup."""
    import socket
    import io
    import base64

    try:
        import qrcode
        from PIL import Image

        # Get local IP
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except socket.error:
            local_ip = "127.0.0.1"

        # Create QR code data
        qr_data = f"shadowai://connect?host={local_ip}&port=19284"

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=8,
            border=2,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)

        # Create image with dark theme colors
        img = qr.make_image(fill_color="#e53935", back_color="#1a1a1a")

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            "image": f"data:image/png;base64,{img_base64}",
            "data": qr_data,
            "host": local_ip,
            "port": 19284,
            "instruction": "Scan with Shadow app to connect"
        })
    except ImportError:
        return jsonify({
            "error": "QR code library not available",
            "data": "shadowai://connect?port=19284",
            "instruction": "Install qrcode library for QR display"
        })


# ============ Teams ============

@api_bp.route('/teams')
def api_teams():
    """List all teams."""
    device_id = request.args.get('device_id')
    return jsonify(get_teams(device_id))


@api_bp.route('/teams/<team_id>')
def api_team(team_id):
    """Get team details."""
    team = get_team(team_id)
    if team:
        return jsonify(team)
    return jsonify({"error": "Team not found"}), 404


@api_bp.route('/teams', methods=['POST'])
def api_create_team():
    """Create a new team."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = create_team(data)
    return jsonify(result)


@api_bp.route('/teams/<team_id>', methods=['PUT'])
def api_update_team(team_id):
    """Update team details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_team(team_id, data)
    return jsonify(result)


@api_bp.route('/teams/<team_id>', methods=['DELETE'])
def api_delete_team(team_id):
    """Delete a team."""
    result = delete_team(team_id)
    return jsonify(result)


@api_bp.route('/teams/metrics')
def api_team_metrics():
    """Get team metrics summary."""
    return jsonify(get_team_metrics())


# ============ Tasks ============

@api_bp.route('/tasks')
def api_tasks():
    """List all tasks."""
    device_id = request.args.get('device_id')
    return jsonify(get_tasks(device_id))


@api_bp.route('/tasks/<task_id>')
def api_task(task_id):
    """Get task details."""
    task = get_task(task_id)
    if task:
        return jsonify(task)
    return jsonify({"error": "Task not found"}), 404


@api_bp.route('/tasks', methods=['POST'])
def api_create_task():
    """Create a new task."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = create_task(data)
    return jsonify(result)


@api_bp.route('/tasks/<task_id>', methods=['PUT'])
def api_update_task(task_id):
    """Update task details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_task(task_id, data)
    return jsonify(result)


@api_bp.route('/tasks/<task_id>', methods=['DELETE'])
def api_delete_task(task_id):
    """Delete a task."""
    result = delete_task(task_id)
    return jsonify(result)


# ============ Project Todos ============

@api_bp.route('/projects/<project_id>/todos')
def api_project_todos(project_id):
    """List all todos for a project."""
    from ..services.data_service import get_project_todos
    return jsonify(get_project_todos(project_id))


@api_bp.route('/projects/<project_id>/todos', methods=['POST'])
def api_create_project_todo(project_id):
    """Create a new todo for a project."""
    from ..services.data_service import create_project_todo
    data = request.get_json()
    if not data or not data.get("content"):
        return jsonify({"error": "Content is required"}), 400
    result = create_project_todo(project_id, data)
    return jsonify(result)


@api_bp.route('/projects/<project_id>/todos/<todo_id>', methods=['PUT'])
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


@api_bp.route('/projects/<project_id>/todos/<todo_id>', methods=['DELETE'])
def api_delete_project_todo(project_id, todo_id):
    """Delete a todo."""
    from ..services.data_service import delete_project_todo
    result = delete_project_todo(project_id, todo_id)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@api_bp.route('/projects/<project_id>/todos/reorder', methods=['POST'])
def api_reorder_project_todos(project_id):
    """Reorder todos by providing list of todo IDs in desired order."""
    from ..services.data_service import reorder_project_todos
    data = request.get_json()
    if not data or not data.get("todo_ids"):
        return jsonify({"error": "todo_ids list is required"}), 400
    result = reorder_project_todos(project_id, data["todo_ids"])
    return jsonify(result)


# ============ CLI Launch ============

@api_bp.route('/projects/<project_id>/launch-cli', methods=['POST'])
def api_launch_cli(project_id):
    """Launch Claude Code CLI in project directory."""
    project = get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    path = project.get("path", "")
    if not path:
        return jsonify({"error": "No path for project"}), 400

    # Validate path exists
    import os
    if not os.path.exists(path):
        return jsonify({"error": f"Path does not exist: {path}"}), 400

    try:
        # Launch Windows Terminal with Claude Code in project directory
        subprocess.Popen(
            ['wt', '-d', path, 'claude'],
            shell=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        return jsonify({
            "success": True,
            "path": path,
            "command": "claude"
        })
    except FileNotFoundError:
        # Fallback: try cmd with claude
        try:
            subprocess.Popen(
                f'start cmd /k "cd /d {path} && claude"',
                shell=True
            )
            return jsonify({
                "success": True,
                "path": path,
                "command": "claude (via cmd)"
            })
        except Exception as e:
            return jsonify({"error": f"Failed to launch CLI: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to launch CLI: {str(e)}"}), 500


# ============ Agent Management ============

@api_bp.route('/agents', methods=['POST'])
def api_add_agent():
    """Add a new agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = add_agent(data)
    return jsonify(result)


@api_bp.route('/agents/<agent_id>', methods=['PUT'])
def api_update_agent(agent_id):
    """Update agent details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_agent(agent_id, data)
    return jsonify(result)


@api_bp.route('/agents/<agent_id>', methods=['DELETE'])
def api_delete_agent(agent_id):
    """Delete an agent."""
    result = delete_agent(agent_id)
    return jsonify(result)


# ============ Workflows ============

@api_bp.route('/workflows')
def api_workflows():
    """List all workflows."""
    return jsonify(get_workflows())


@api_bp.route('/workflows', methods=['POST'])
def api_start_workflow():
    """Start a new workflow."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    workflow_type = data.get('type', 'qa_pipeline')
    result = start_workflow(workflow_type, data)
    return jsonify(result)


@api_bp.route('/workflows/<workflow_id>/cancel', methods=['POST'])
def api_cancel_workflow(workflow_id):
    """Cancel a workflow."""
    result = cancel_workflow(workflow_id)
    return jsonify(result)


# ============ Audits ============

@api_bp.route('/audits')
def api_audits():
    """List audit entries."""
    period = request.args.get('period', '7D')
    device_id = request.args.get('device_id')
    return jsonify(get_audits(period, device_id))


@api_bp.route('/audits/<audit_id>')
def api_audit_entry(audit_id):
    """Get audit entry details."""
    entry = get_audit_entry(audit_id)
    if entry:
        return jsonify(entry)
    return jsonify({"error": "Audit entry not found"}), 404


@api_bp.route('/audits/stats')
def api_audit_stats():
    """Get audit statistics."""
    period = request.args.get('period', '7D')
    return jsonify(get_audit_stats(period))


@api_bp.route('/audits/<audit_id>/traces')
def api_audit_traces(audit_id):
    """Get AI decision traces for an audit entry."""
    traces = get_audit_traces(audit_id)
    return jsonify(traces)


@api_bp.route('/audits/export', methods=['POST'])
def api_export_audit():
    """Export audit report."""
    format = request.args.get('format', 'json')
    period = request.args.get('period', '30D')
    result = export_audit_report(format, period)
    return jsonify(result)


# ============ Favorites ============

@api_bp.route('/favorites')
def api_favorites():
    """Get all favorites."""
    return jsonify(get_favorites())


@api_bp.route('/projects/<project_id>/favorite', methods=['POST'])
def api_toggle_project_favorite(project_id):
    """Toggle project favorite status."""
    result = toggle_favorite('project', project_id)
    return jsonify(result)


@api_bp.route('/notes/<note_id>/favorite', methods=['POST'])
def api_toggle_note_favorite(note_id):
    """Toggle note favorite status."""
    result = toggle_favorite('note', note_id)
    return jsonify(result)


# ============ Project Update ============

@api_bp.route('/projects/<project_id>', methods=['PUT'])
def api_update_project(project_id):
    """Update project details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = update_project(project_id, data)
    return jsonify(result)


# ============ Global Search ============

@api_bp.route('/search')
def api_search():
    """Search across all content types."""
    query = request.args.get('q', '')
    types = request.args.getlist('type') or ['projects', 'notes', 'automations', 'agents']
    if not query:
        return jsonify({"items": []})
    results = search_all(query, types)
    return jsonify(results)


# ============ Enhanced Analytics ============

@api_bp.route('/analytics/privacy')
def api_privacy_score():
    """Get privacy score and compliance data."""
    return jsonify(get_privacy_score())


@api_bp.route('/analytics/tokens')
def api_token_usage():
    """Get token usage statistics."""
    return jsonify(get_token_usage())


@api_bp.route('/analytics/categories')
def api_category_breakdown():
    """Get audit category breakdown."""
    period = request.args.get('period', '7D')
    return jsonify(get_category_breakdown(period))
