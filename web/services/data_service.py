"""
Data Service - Read ~/.shadowai/ JSON files
"""
import json
import os
import base64
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Shadow data directory
SHADOWAI_DIR = Path.home() / ".shadowai"
PROJECTS_FILE = SHADOWAI_DIR / "projects.json"
NOTES_FILE = SHADOWAI_DIR / "notes.json"
DEVICES_FILE = SHADOWAI_DIR / "devices.json"
AUTOMATIONS_FILE = SHADOWAI_DIR / "automations.json"
AGENTS_FILE = SHADOWAI_DIR / "agents.json"
ANALYTICS_FILE = SHADOWAI_DIR / "analytics.json"
SYNC_KEYS_FILE = SHADOWAI_DIR / "sync_keys.json"

# Note encryption constants (must match Android SyncEncryption.kt)
SYNC_ENC_PREFIX = "SYNC_ENC:"
FIXED_SALT = b"ShadowAI_Sync_2024"
ITERATION_COUNT = 50000
KEY_LENGTH = 32  # 256 bits
IV_LENGTH = 12
TAG_LENGTH = 16  # 128 bits in bytes


def _derive_encryption_key(password: str) -> bytes:
    """Derive AES-256 key using PBKDF2-HMAC-SHA256 (matches Android)."""
    import hashlib
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        FIXED_SALT,
        ITERATION_COUNT,
        dklen=KEY_LENGTH
    )


def _get_sync_key_for_device(device_id: str) -> Optional[str]:
    """Get the sync encryption key for a device.

    Returns user-set password if configured, otherwise derives from device ID.
    """
    # Check for user-configured sync password
    keys_data = _read_json_file(SYNC_KEYS_FILE)
    if keys_data:
        device_key = keys_data.get(device_id)
        if device_key:
            return device_key

    # Fall back to device-derived key (matches Android default)
    return f"ShadowSync_{device_id[:16]}" if device_id else None


def decrypt_note_content(content: str, device_id: str) -> str:
    """Decrypt note content if encrypted.

    Handles SYNC_ENC:[IV]:[Ciphertext] format from Android.
    Returns original content if not encrypted or decryption fails.
    """
    if not content or not content.startswith(SYNC_ENC_PREFIX):
        return content

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Parse: SYNC_ENC:[IV]:[Ciphertext]
        data = content[len(SYNC_ENC_PREFIX):]
        parts = data.split(":")
        if len(parts) != 2:
            print(f"Invalid sync encryption format")
            return content

        iv = base64.b64decode(parts[0])
        ciphertext = base64.b64decode(parts[1])

        # Get encryption key
        password = _get_sync_key_for_device(device_id)
        if not password:
            print(f"No encryption key available for device {device_id}")
            return content

        key = _derive_encryption_key(password)

        # Decrypt using AES-256-GCM
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(iv, ciphertext, None)

        return plaintext.decode('utf-8')

    except ImportError:
        print("cryptography library not installed - cannot decrypt notes")
        return content
    except Exception as e:
        print(f"Note decryption failed: {e}")
        return content


def encrypt_note_content(content: str, device_id: str) -> str:
    """Encrypt note content for sending to device.

    Returns SYNC_ENC:[IV]:[Ciphertext] format.
    Returns original content if encryption fails.
    """
    if not content:
        return content

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import secrets

        # Get encryption key
        password = _get_sync_key_for_device(device_id)
        if not password:
            return content

        key = _derive_encryption_key(password)

        # Generate random IV
        iv = secrets.token_bytes(IV_LENGTH)

        # Encrypt using AES-256-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(iv, content.encode('utf-8'), None)

        # Format: SYNC_ENC:[IV]:[Ciphertext]
        iv_b64 = base64.b64encode(iv).decode('ascii')
        ciphertext_b64 = base64.b64encode(ciphertext).decode('ascii')

        return f"{SYNC_ENC_PREFIX}{iv_b64}:{ciphertext_b64}"

    except ImportError:
        print("cryptography library not installed - cannot encrypt notes")
        return content
    except Exception as e:
        print(f"Note encryption failed: {e}")
        return content


def set_sync_key_for_device(device_id: str, password: str) -> bool:
    """Store a sync encryption password for a device."""
    keys_data = _read_json_file(SYNC_KEYS_FILE) or {}
    keys_data[device_id] = password

    try:
        SHADOWAI_DIR.mkdir(parents=True, exist_ok=True)
        with open(SYNC_KEYS_FILE, 'w', encoding='utf-8') as f:
            json.dump(keys_data, f)
        return True
    except Exception as e:
        print(f"Failed to save sync key: {e}")
        return False


def _read_json_file(filepath: Path) -> Optional[Dict]:
    """Read and parse a JSON file."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filepath}: {e}")
        return None


def _format_timestamp(ts: int) -> str:
    """Format Unix timestamp to human-readable string."""
    if not ts:
        return "Unknown"
    try:
        dt = datetime.fromtimestamp(ts / 1000)  # Convert from milliseconds
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return "Unknown"


def _time_ago(ts: int) -> str:
    """Convert Unix timestamp to relative time string."""
    if not ts:
        return "Unknown"
    try:
        dt = datetime.fromtimestamp(ts / 1000)
        now = datetime.now()
        diff = now - dt

        if diff.days > 30:
            return f"{diff.days // 30} months ago"
        elif diff.days > 0:
            return f"{diff.days} days ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600} hours ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60} minutes ago"
        else:
            return "Just now"
    except:
        return "Unknown"

def _is_private_ip(ip: str) -> bool:
    return (
        ip.startswith("10.")
        or ip.startswith("192.168.")
        or ip.startswith("172.16.")
        or ip.startswith("172.17.")
        or ip.startswith("172.18.")
        or ip.startswith("172.19.")
        or ip.startswith("172.20.")
        or ip.startswith("172.21.")
        or ip.startswith("172.22.")
        or ip.startswith("172.23.")
        or ip.startswith("172.24.")
        or ip.startswith("172.25.")
        or ip.startswith("172.26.")
        or ip.startswith("172.27.")
        or ip.startswith("172.28.")
        or ip.startswith("172.29.")
        or ip.startswith("172.30.")
        or ip.startswith("172.31.")
    )


def _collect_device_ips(device_info: Dict) -> List[str]:
    ips: List[str] = []
    primary = device_info.get("ip")
    if isinstance(primary, str) and primary:
        ips.append(primary)

    candidates = device_info.get("ip_candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                ips.append(candidate)

    deduped: List[str] = []
    for ip in ips:
        if ip not in deduped:
            deduped.append(ip)

    private_ips = [ip for ip in deduped if _is_private_ip(ip)]
    public_ips = [ip for ip in deduped if ip not in private_ips]
    return private_ips + public_ips


def _infer_note_content_port(device_id: Optional[str], explicit_port: Any) -> Optional[int]:
    if isinstance(explicit_port, int) and 1 <= explicit_port <= 65535:
        return explicit_port
    if not device_id:
        return None
    package_name = device_id.split(":")[-1]
    if package_name.endswith(".release6"):
        return 19286
    if "debug" in package_name:
        return 19287
    if package_name.endswith(".release"):
        return 19285
    return 19285


# ============ Devices ============

def get_devices() -> List[Dict]:
    """Get all known devices with their status from projects.json."""
    # Device info is stored in projects.json, not a separate devices.json
    data = _read_json_file(PROJECTS_FILE)
    if not data:
        return []

    devices_data = data.get("devices", {})
    if not devices_data:
        return []

    devices = []
    now = datetime.now().timestamp()

    for device_id, device_info in devices_data.items():
        last_seen = device_info.get("last_seen", 0)
        # Consider device online if seen within last 5 minutes
        is_online = (now - last_seen) < 300 if last_seen else False

        devices.append({
            "id": device_id,
            "name": device_info.get("name", device_id),
            "status": "online" if is_online else "offline",
            "ip": device_info.get("ip"),
            "last_seen": last_seen,
            "last_seen_formatted": _time_ago(int(last_seen * 1000)) if last_seen else "Never"
        })
    return devices


def get_device(device_id: str) -> Optional[Dict]:
    """Get a specific device by ID."""
    devices = get_devices()
    for device in devices:
        if device["id"] == device_id:
            return device
    return None


# ============ Projects ============

def get_projects(device_id: Optional[str] = None) -> List[Dict]:
    """Get all projects, optionally filtered by device."""
    data = _read_json_file(PROJECTS_FILE)
    if not data:
        return []

    # Projects are stored under "devices" key
    devices_data = data.get("devices", {})

    projects = []
    for proj_device_id, device_info in devices_data.items():
        if device_id and proj_device_id != device_id:
            continue

        for project in device_info.get("projects", []):
            projects.append({
                "id": project.get("id", project.get("path", "")),
                "name": project.get("name", os.path.basename(project.get("path", "Unknown"))),
                "path": project.get("path", ""),
                "device_id": proj_device_id,
                "device_name": device_info.get("name", proj_device_id),
                "updated_at": project.get("updated_at", 0),
                "updated_at_formatted": _format_timestamp(project.get("updated_at", 0)),
                "time_ago": _time_ago(project.get("updated_at", 0))
            })

    # Sort by updated_at descending
    projects.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return projects


def get_project(project_id: str) -> Optional[Dict]:
    """Get a specific project by ID."""
    projects = get_projects()
    for project in projects:
        if project["id"] == project_id:
            return project
    return None


# ============ Notes ============

def get_notes(device_id: Optional[str] = None, search: Optional[str] = None) -> List[Dict]:
    """Get all note titles, optionally filtered by device and search term.

    Notes are deduplicated by ID - if the same note exists on multiple device entries
    (e.g., com.shadowai.release and com.shadowai.release6 on the same phone),
    we keep the most recently updated version and use the most recently synced device's IP.

    Also filters out notes from stale device entries that share the same base fingerprint
    with a more recently synced entry (e.g., old entries without package name suffix).
    """
    data = _read_json_file(NOTES_FILE)
    if not data:
        return []

    # Notes are stored under "devices" key
    devices_data = data.get("devices", {})

    # Find the most recently synced device for each base fingerprint
    # This helps identify stale device entries (old format without package suffix)
    fingerprint_to_latest: Dict[str, tuple] = {}  # base_fingerprint -> (device_id, last_seen)
    for dev_id, dev_info in devices_data.items():
        # Extract base fingerprint (everything before the optional :package.name suffix)
        base_fingerprint = dev_id.split(':com.')[0] if ':com.' in dev_id else dev_id
        last_seen = dev_info.get("last_seen", 0)

        current = fingerprint_to_latest.get(base_fingerprint)
        if current is None or last_seen > current[1]:
            fingerprint_to_latest[base_fingerprint] = (dev_id, last_seen)

    # Build set of device IDs to skip (stale entries with same base fingerprint as a newer entry)
    stale_device_ids = set()
    for dev_id in devices_data.keys():
        base_fingerprint = dev_id.split(':com.')[0] if ':com.' in dev_id else dev_id
        latest_for_fingerprint = fingerprint_to_latest.get(base_fingerprint)
        if latest_for_fingerprint and latest_for_fingerprint[0] != dev_id:
            # This device entry is NOT the most recent for its fingerprint - mark as stale
            stale_device_ids.add(dev_id)

    # First pass: collect all notes, tracking which device entry is most recent for each note
    notes_by_id: Dict[str, Dict] = {}

    for note_device_id, device_info in devices_data.items():
        if device_id and note_device_id != device_id:
            continue

        # Skip stale device entries (older format or less recently synced duplicate)
        if note_device_id in stale_device_ids:
            continue

        device_last_seen = device_info.get("last_seen", 0)
        device_ips = _collect_device_ips(device_info)
        note_content_port = _infer_note_content_port(note_device_id, device_info.get("note_content_port"))

        for note in device_info.get("notes", []):
            note_id = note.get("id", "")
            if not note_id:
                continue

            title = note.get("title", "Untitled")
            updated_at = note.get("updatedAt", 0)

            # Apply search filter
            if search and search.lower() not in title.lower():
                continue

            # Check if we already have this note from another device
            existing = notes_by_id.get(note_id)

            if existing is None:
                # First time seeing this note
                notes_by_id[note_id] = {
                    "id": note_id,
                    "title": title,
                    "device_id": note_device_id,
                    "device_name": device_info.get("name", note_device_id),
                    "device_ip": device_ips[0] if device_ips else device_info.get("ip"),
                    "device_ips": device_ips,
                    "note_content_port": note_content_port,
                    "device_last_seen": device_last_seen,
                    "updated_at": updated_at,
                    "updated_at_formatted": _format_timestamp(updated_at),
                    "time_ago": _time_ago(updated_at)
                }
            else:
                # Duplicate note - keep the one from the most recently synced device
                # This ensures deleted notes (not in latest sync) don't reappear
                if device_last_seen > existing.get("device_last_seen", 0):
                    notes_by_id[note_id] = {
                        "id": note_id,
                        "title": title,
                        "device_id": note_device_id,
                        "device_name": device_info.get("name", note_device_id),
                        "device_ip": device_ips[0] if device_ips else device_info.get("ip"),
                        "device_ips": device_ips,
                        "note_content_port": note_content_port,
                        "device_last_seen": device_last_seen,
                        "updated_at": updated_at,
                        "updated_at_formatted": _format_timestamp(updated_at),
                        "time_ago": _time_ago(updated_at)
                    }

    # Convert to list and remove internal tracking field
    notes = []
    for note in notes_by_id.values():
        note.pop("device_last_seen", None)
        notes.append(note)

    # Sort by updated_at descending
    notes.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return notes


def get_note(note_id: str) -> Optional[Dict]:
    """Get a specific note by ID (metadata only - content requires device fetch)."""
    notes = get_notes()
    for note in notes:
        if note["id"] == note_id:
            return note
    return None


def get_note_content(note_id: str) -> Optional[Dict]:
    """Get note content from local cache if available.

    Returns full note with content if cached, None if content needs device fetch.
    """
    data = _read_json_file(NOTES_FILE)
    if not data:
        return None

    devices_data = data.get("devices", {})

    for device_id, device_info in devices_data.items():
        for note in device_info.get("notes", []):
            if note.get("id") == note_id:
                content = note.get("content")
                if content is not None:
                    return {
                        "id": note_id,
                        "title": note.get("title", "Untitled"),
                        "content": content,
                        "updated_at": note.get("updatedAt", 0)
                    }
    return None


def save_note_content(note_id: str, title: str, content: str, updated_at: int) -> bool:
    """Save note content to local cache for instant loading.

    Updates the note in notes.json with full content.
    """
    data = _read_json_file(NOTES_FILE)
    if not data:
        return False

    devices_data = data.get("devices", {})
    updated = False

    for device_id, device_info in devices_data.items():
        notes = device_info.get("notes", [])
        for i, note in enumerate(notes):
            if note.get("id") == note_id:
                notes[i]["content"] = content
                notes[i]["title"] = title
                notes[i]["updatedAt"] = updated_at
                updated = True
                break
        if updated:
            break

    if updated:
        try:
            with open(NOTES_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving note content: {e}")
            return False

    return False


# ============ Automations ============

def get_automations(device_id: Optional[str] = None) -> List[Dict]:
    """Get all automations."""
    data = _read_json_file(AUTOMATIONS_FILE)
    if not data:
        return []

    automations = []
    for auto_device_id, device_autos in data.items():
        if device_id and auto_device_id != device_id:
            continue

        for auto in device_autos.get("automations", []):
            automations.append({
                "id": auto.get("id", ""),
                "name": auto.get("name", "Unnamed"),
                "description": auto.get("description", ""),
                "enabled": auto.get("enabled", False),
                "schedule": auto.get("schedule", ""),
                "trigger": auto.get("trigger", "manual"),
                "last_run": auto.get("last_run"),
                "last_run_formatted": _time_ago(auto.get("last_run", 0)) if auto.get("last_run") else "Never",
                "device_id": auto_device_id
            })

    return automations


def get_automation(automation_id: str) -> Optional[Dict]:
    """Get a specific automation by ID."""
    automations = get_automations()
    for auto in automations:
        if auto["id"] == automation_id:
            return auto
    return None


def get_automation_logs(automation_id: str) -> List[Dict]:
    """Get execution logs for an automation."""
    # TODO: Implement log reading from device
    return []


# ============ Agents ============

def get_agents(device_id: Optional[str] = None) -> List[Dict]:
    """Get all agents with their status."""
    data = _read_json_file(AGENTS_FILE)
    if not data:
        return []

    agents = []
    for agent_device_id, device_agents in data.items():
        if device_id and agent_device_id != device_id:
            continue

        for agent in device_agents.get("agents", []):
            agents.append({
                "id": agent.get("id", ""),
                "name": agent.get("name", "Unnamed"),
                "status": agent.get("status", "offline"),
                "specialty": agent.get("specialty", ""),
                "current_task": agent.get("current_task"),
                "tasks_completed": agent.get("tasks_completed", 0),
                "device_id": agent_device_id
            })

    return agents


def get_agent(agent_id: str) -> Optional[Dict]:
    """Get a specific agent by ID."""
    agents = get_agents()
    for agent in agents:
        if agent["id"] == agent_id:
            return agent
    return None


def get_agent_metrics() -> Dict:
    """Get aggregate agent performance metrics."""
    agents = get_agents()
    return {
        "total_agents": len(agents),
        "active_agents": len([a for a in agents if a["status"] == "busy"]),
        "idle_agents": len([a for a in agents if a["status"] == "idle"]),
        "offline_agents": len([a for a in agents if a["status"] == "offline"]),
        "total_tasks_completed": sum(a.get("tasks_completed", 0) for a in agents)
    }


# ============ Analytics ============

def get_usage_stats() -> Dict:
    """Get message usage statistics."""
    data = _read_json_file(ANALYTICS_FILE)
    if not data:
        return {
            "messages_used": 0,
            "messages_limit": 100,
            "tier": "free",
            "reset_date": None
        }

    return {
        "messages_used": data.get("messages_used", 0),
        "messages_limit": data.get("messages_limit", 100),
        "tier": data.get("tier", "free"),
        "reset_date": data.get("reset_date"),
        "usage_percent": min(100, (data.get("messages_used", 0) / max(1, data.get("messages_limit", 100))) * 100)
    }


def get_backend_usage() -> List[Dict]:
    """Get backend usage breakdown."""
    data = _read_json_file(ANALYTICS_FILE)
    if not data:
        return []

    backends = data.get("backends", {})
    total = sum(backends.values()) or 1

    return [
        {
            "name": name,
            "count": count,
            "percent": (count / total) * 100
        }
        for name, count in backends.items()
    ]


def get_activity_timeline() -> List[Dict]:
    """Get activity timeline."""
    data = _read_json_file(ANALYTICS_FILE)
    if not data:
        return []

    return data.get("activity", [])


# ============ Status ============

def _check_port_in_use(port: int) -> bool:
    """Check if a port is in use (indicates service is running)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(0.5)
            s.connect(('127.0.0.1', port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def get_status() -> Dict:
    """Get overall system status."""
    import socket

    projects = get_projects()
    notes = get_notes()
    devices = get_devices()

    online_devices = [d for d in devices if d.get("status") == "online"]

    # Check if ShadowBridge is running by checking its data receiver port
    shadowbridge_running = _check_port_in_use(19284)

    # Determine SSH/device status based on online devices
    if len(online_devices) > 0:
        ssh_status = "connected"
    elif len(devices) > 0:
        ssh_status = "offline"
    else:
        ssh_status = "no_devices"

    # Get local IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    return {
        "shadowbridge_running": shadowbridge_running,
        "devices_connected": len(online_devices),
        "total_devices": len(devices),
        "total_projects": len(projects),
        "total_notes": len(notes),
        "ssh_status": ssh_status,
        "version": "1.001",
        "local_ip": local_ip,
        "data_path": str(SHADOWAI_DIR)
    }


# ============ Data Files ============

TEAMS_FILE = SHADOWAI_DIR / "teams.json"
TASKS_FILE = SHADOWAI_DIR / "tasks.json"
TODOS_FILE = SHADOWAI_DIR / "todos.json"
WORKFLOWS_FILE = SHADOWAI_DIR / "workflows.json"
AUDITS_FILE = SHADOWAI_DIR / "audits.json"
FAVORITES_FILE = SHADOWAI_DIR / "web_favorites.json"


def _write_json_file(filepath: Path, data: Dict) -> bool:
    """Write data to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error writing {filepath}: {e}")
        return False


def _generate_id() -> str:
    """Generate a unique ID."""
    import uuid
    return str(uuid.uuid4())[:8]


# ============ Teams ============

def get_teams(device_id: Optional[str] = None) -> List[Dict]:
    """Get all teams."""
    data = _read_json_file(TEAMS_FILE)
    if not data:
        return []
    teams = data.get("teams", [])
    if device_id:
        teams = [t for t in teams if t.get("device_id") == device_id]
    return teams


def get_team(team_id: str) -> Optional[Dict]:
    """Get a specific team by ID."""
    teams = get_teams()
    for team in teams:
        if team.get("id") == team_id:
            return team
    return None


def create_team(data: Dict) -> Dict:
    """Create a new team."""
    file_data = _read_json_file(TEAMS_FILE) or {"teams": []}
    team = {
        "id": _generate_id(),
        "name": data.get("name", "New Team"),
        "description": data.get("description", ""),
        "agents": [],
        "status": "ACTIVE",
        "created_at": int(datetime.now().timestamp() * 1000)
    }
    file_data["teams"].append(team)
    _write_json_file(TEAMS_FILE, file_data)
    return {"success": True, "team": team}


def update_team(team_id: str, data: Dict) -> Dict:
    """Update a team."""
    file_data = _read_json_file(TEAMS_FILE) or {"teams": []}
    for i, team in enumerate(file_data.get("teams", [])):
        if team.get("id") == team_id:
            file_data["teams"][i].update(data)
            _write_json_file(TEAMS_FILE, file_data)
            return {"success": True, "team": file_data["teams"][i]}
    return {"error": "Team not found"}


def delete_team(team_id: str) -> Dict:
    """Delete a team."""
    file_data = _read_json_file(TEAMS_FILE) or {"teams": []}
    teams = file_data.get("teams", [])
    file_data["teams"] = [t for t in teams if t.get("id") != team_id]
    _write_json_file(TEAMS_FILE, file_data)
    return {"success": True}


def get_team_metrics() -> Dict:
    """Get team metrics summary."""
    teams = get_teams()
    agents = get_agents()
    tasks = get_tasks()
    return {
        "total_teams": len(teams),
        "total_agents": len(agents),
        "active_agents": len([a for a in agents if a.get("status") == "BUSY"]),
        "pending_tasks": len([t for t in tasks if t.get("status") == "PENDING"]),
        "completed_tasks": len([t for t in tasks if t.get("status") == "COMPLETED"])
    }


# ============ Tasks ============

def get_tasks(device_id: Optional[str] = None) -> List[Dict]:
    """Get all tasks."""
    data = _read_json_file(TASKS_FILE)
    if not data:
        return []
    tasks = data.get("tasks", [])
    if device_id:
        tasks = [t for t in tasks if t.get("device_id") == device_id]
    return tasks


def get_task(task_id: str) -> Optional[Dict]:
    """Get a specific task by ID."""
    tasks = get_tasks()
    for task in tasks:
        if task.get("id") == task_id:
            return task
    return None


def create_task(data: Dict) -> Dict:
    """Create a new task."""
    file_data = _read_json_file(TASKS_FILE) or {"tasks": []}
    task = {
        "id": _generate_id(),
        "title": data.get("title", "New Task"),
        "description": data.get("description", ""),
        "priority": data.get("priority", "MEDIUM"),
        "status": "PENDING",
        "assigned_agent_id": data.get("assigned_agent_id"),
        "created_at": int(datetime.now().timestamp() * 1000)
    }
    file_data["tasks"].append(task)
    _write_json_file(TASKS_FILE, file_data)
    return {"success": True, "task": task}


def update_task(task_id: str, data: Dict) -> Dict:
    """Update a task."""
    file_data = _read_json_file(TASKS_FILE) or {"tasks": []}
    for i, task in enumerate(file_data.get("tasks", [])):
        if task.get("id") == task_id:
            file_data["tasks"][i].update(data)
            _write_json_file(TASKS_FILE, file_data)
            return {"success": True, "task": file_data["tasks"][i]}
    return {"error": "Task not found"}


def delete_task(task_id: str) -> Dict:
    """Delete a task."""
    file_data = _read_json_file(TASKS_FILE) or {"tasks": []}
    tasks = file_data.get("tasks", [])
    file_data["tasks"] = [t for t in tasks if t.get("id") != task_id]
    _write_json_file(TASKS_FILE, file_data)
    return {"success": True}


# ============ Agent Management ============

def add_agent(data: Dict) -> Dict:
    """Add a new agent."""
    file_data = _read_json_file(AGENTS_FILE) or {}
    device_id = data.get("device_id", "default")
    if device_id not in file_data:
        file_data[device_id] = {"agents": []}

    agent = {
        "id": _generate_id(),
        "name": data.get("name", "New Agent"),
        "type": data.get("type", "GENERAL_PURPOSE"),
        "status": "IDLE",
        "is_available": True,
        "specializations": data.get("specializations", []),
        "tasks_completed": 0,
        "workload": 0,
        "created_at": int(datetime.now().timestamp() * 1000)
    }
    file_data[device_id]["agents"].append(agent)
    _write_json_file(AGENTS_FILE, file_data)
    return {"success": True, "agent": agent}


def update_agent(agent_id: str, data: Dict) -> Dict:
    """Update an agent."""
    file_data = _read_json_file(AGENTS_FILE) or {}
    for device_id, device_agents in file_data.items():
        agents = device_agents.get("agents", [])
        for i, agent in enumerate(agents):
            if agent.get("id") == agent_id:
                file_data[device_id]["agents"][i].update(data)
                _write_json_file(AGENTS_FILE, file_data)
                return {"success": True, "agent": file_data[device_id]["agents"][i]}
    return {"error": "Agent not found"}


def delete_agent(agent_id: str) -> Dict:
    """Delete an agent."""
    file_data = _read_json_file(AGENTS_FILE) or {}
    for device_id, device_agents in file_data.items():
        agents = device_agents.get("agents", [])
        file_data[device_id]["agents"] = [a for a in agents if a.get("id") != agent_id]
    _write_json_file(AGENTS_FILE, file_data)
    return {"success": True}


# ============ Workflows ============

def get_workflows() -> List[Dict]:
    """Get all workflows."""
    data = _read_json_file(WORKFLOWS_FILE)
    if not data:
        return []
    return data.get("workflows", [])


def start_workflow(workflow_type: str, options: Dict) -> Dict:
    """Start a new workflow."""
    file_data = _read_json_file(WORKFLOWS_FILE) or {"workflows": []}
    workflow = {
        "id": _generate_id(),
        "type": workflow_type,
        "name": f"{workflow_type.replace('_', ' ').title()} Workflow",
        "status": "PENDING",
        "current_stage": "Starting",
        "progress": 0,
        "started_at": int(datetime.now().timestamp() * 1000)
    }
    file_data["workflows"].append(workflow)
    _write_json_file(WORKFLOWS_FILE, file_data)
    return {"success": True, "workflow": workflow}


def cancel_workflow(workflow_id: str) -> Dict:
    """Cancel a workflow."""
    file_data = _read_json_file(WORKFLOWS_FILE) or {"workflows": []}
    for i, workflow in enumerate(file_data.get("workflows", [])):
        if workflow.get("id") == workflow_id:
            file_data["workflows"][i]["status"] = "CANCELLED"
            _write_json_file(WORKFLOWS_FILE, file_data)
            return {"success": True}
    return {"error": "Workflow not found"}


# ============ Audits ============

def _get_period_timestamp(period: str) -> int:
    """Get timestamp for period start."""
    now = datetime.now()
    periods = {
        "1D": 1,
        "7D": 7,
        "30D": 30,
        "90D": 90,
        "ANNUAL": 365,
        "ALL_TIME": 3650
    }
    days = periods.get(period, 7)
    from datetime import timedelta
    start = now - timedelta(days=days)
    return int(start.timestamp() * 1000)


def get_audits(period: str = "7D", device_id: Optional[str] = None) -> List[Dict]:
    """Get audit entries for a period."""
    data = _read_json_file(AUDITS_FILE)
    if not data:
        return []

    cutoff = _get_period_timestamp(period)
    entries = data.get("entries", [])

    # Filter by period
    entries = [e for e in entries if e.get("timestamp", 0) >= cutoff]

    # Filter by device
    if device_id:
        entries = [e for e in entries if e.get("device_id") == device_id]

    # Add relative time
    for entry in entries:
        entry["relative_time"] = _time_ago(entry.get("timestamp", 0))

    # Sort by timestamp descending
    entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return entries


def get_audit_entry(audit_id: str) -> Optional[Dict]:
    """Get a specific audit entry."""
    data = _read_json_file(AUDITS_FILE)
    if not data:
        return None
    for entry in data.get("entries", []):
        if entry.get("id") == audit_id:
            return entry
    return None


def get_audit_stats(period: str = "7D") -> Dict:
    """Get audit statistics for a period."""
    entries = get_audits(period)
    data = _read_json_file(AUDITS_FILE) or {}

    ai_decisions = len([e for e in entries if e.get("category") == "AI_DECISION"])
    data_accesses = len([e for e in entries if e.get("category") == "DATA_ACCESS"])
    security_events = len([e for e in entries if e.get("severity") in ["SECURITY", "CRITICAL"]])

    # Calculate privacy score (inverse of security events ratio)
    privacy_score = max(0, 100 - (security_events * 5)) if entries else 100

    # Determine risk level
    if security_events > 10:
        risk_level = "HIGH"
    elif security_events > 5:
        risk_level = "MEDIUM"
    elif security_events > 0:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"

    return {
        "total_events": len(entries),
        "ai_decisions": ai_decisions,
        "data_accesses": data_accesses,
        "security_events": security_events,
        "privacy_score": privacy_score,
        "risk_level": risk_level
    }


def get_audit_traces(audit_id: str) -> List[Dict]:
    """Get AI decision traces for an audit entry."""
    data = _read_json_file(AUDITS_FILE)
    if not data:
        return []
    return data.get("traces", {}).get(audit_id, [])


def export_audit_report(format: str, period: str) -> Dict:
    """Export audit report."""
    entries = get_audits(period)
    stats = get_audit_stats(period)

    export_dir = Path.home() / "Downloads" / "Shadow_Audits"
    export_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "json":
        filepath = export_dir / f"audit_report_{timestamp}.json"
        report = {"stats": stats, "entries": entries}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        return {"success": True, "path": str(filepath)}

    elif format == "csv":
        filepath = export_dir / f"audit_report_{timestamp}.csv"
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Timestamp", "Category", "Severity", "Summary", "Action"])
            for entry in entries:
                writer.writerow([
                    entry.get("id", ""),
                    _format_timestamp(entry.get("timestamp", 0)),
                    entry.get("category", ""),
                    entry.get("severity", ""),
                    entry.get("summary", ""),
                    entry.get("action", "")
                ])
        return {"success": True, "path": str(filepath)}

    else:  # PDF not implemented yet
        return {"error": "PDF export not implemented"}


# ============ Favorites ============

def get_favorites() -> Dict:
    """Get all favorites."""
    data = _read_json_file(FAVORITES_FILE)
    if not data:
        return {"projects": [], "notes": []}
    return data


def toggle_favorite(item_type: str, item_id: str) -> Dict:
    """Toggle favorite status for an item."""
    data = _read_json_file(FAVORITES_FILE) or {"projects": [], "notes": []}

    key = f"{item_type}s"  # projects or notes
    if key not in data:
        data[key] = []

    if item_id in data[key]:
        data[key].remove(item_id)
        is_favorite = False
    else:
        data[key].append(item_id)
        is_favorite = True

    _write_json_file(FAVORITES_FILE, data)
    return {"success": True, "is_favorite": is_favorite}


# ============ Project Update ============

def update_project(project_id: str, data: Dict) -> Dict:
    """Update a project."""
    file_data = _read_json_file(PROJECTS_FILE)
    if not file_data:
        return {"error": "Projects file not found"}

    for device_id, device_info in file_data.get("devices", {}).items():
        for i, project in enumerate(device_info.get("projects", [])):
            if project.get("id") == project_id or project.get("path") == project_id:
                device_info["projects"][i].update(data)
                _write_json_file(PROJECTS_FILE, file_data)
                return {"success": True, "project": device_info["projects"][i]}

    return {"error": "Project not found"}


# ============ Global Search ============

def search_all(query: str, types: List[str]) -> Dict:
    """Search across all content types."""
    query_lower = query.lower()
    results = []

    if "projects" in types:
        for project in get_projects():
            name = project.get("name", "")
            path = project.get("path", "")
            if query_lower in name.lower() or query_lower in path.lower():
                results.append({
                    "type": "project",
                    "id": project.get("id"),
                    "title": name,
                    "preview": path
                })

    if "notes" in types:
        for note in get_notes():
            title = note.get("title", "")
            if query_lower in title.lower():
                results.append({
                    "type": "note",
                    "id": note.get("id"),
                    "title": title,
                    "preview": note.get("time_ago", "")
                })

    if "automations" in types:
        for auto in get_automations():
            name = auto.get("name", "")
            desc = auto.get("description", "")
            if query_lower in name.lower() or query_lower in desc.lower():
                results.append({
                    "type": "automation",
                    "id": auto.get("id"),
                    "title": name,
                    "preview": desc[:50] if desc else ""
                })

    if "agents" in types:
        for agent in get_agents():
            name = agent.get("name", "")
            specialty = agent.get("specialty", "")
            if query_lower in name.lower() or query_lower in specialty.lower():
                results.append({
                    "type": "agent",
                    "id": agent.get("id"),
                    "title": name,
                    "preview": specialty
                })

    return {"items": results[:20]}  # Limit to 20 results


# ============ Enhanced Analytics ============

def get_privacy_score() -> Dict:
    """Get privacy score and compliance data."""
    stats = get_audit_stats("30D")
    return {
        "privacy_score": stats.get("privacy_score", 100),
        "risk_level": stats.get("risk_level", "MINIMAL"),
        "pii_access_count": 0,  # TODO: Track from audit traces
        "encrypted_access_count": 0,
        "compliance_status": "COMPLIANT"
    }


def get_token_usage() -> Dict:
    """Get token usage statistics."""
    data = _read_json_file(ANALYTICS_FILE)
    if not data:
        return {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "avg_per_query": 0
        }
    return {
        "total_tokens": data.get("total_tokens", 0),
        "input_tokens": data.get("input_tokens", 0),
        "output_tokens": data.get("output_tokens", 0),
        "avg_per_query": data.get("avg_tokens_per_query", 0)
    }


def get_category_breakdown(period: str = "7D") -> Dict:
    """Get audit category breakdown."""
    entries = get_audits(period)
    categories = {}

    for entry in entries:
        cat = entry.get("category", "UNKNOWN")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "categories": [
            {"category": cat, "count": count}
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)
        ]
    }


# ============ Create Functions (Web-originated) ============

def create_project(data: Dict, device_id: str = "web") -> Dict:
    """Create a new project from web dashboard.

    Projects are stored under the specified device_id with pending_sync flag
    for bi-directional sync back to Android.
    """
    file_data = _read_json_file(PROJECTS_FILE) or {"devices": {}}

    timestamp = int(datetime.now().timestamp() * 1000)
    project = {
        "id": _generate_id(),
        "name": data.get("name", "New Project"),
        "path": data.get("path", ""),
        "description": data.get("description", ""),
        "labels": data.get("labels", []),
        "group": data.get("group"),
        "color": data.get("color"),
        "github_repo_url": data.get("github_repo_url"),
        "github_branch": data.get("github_branch"),
        "created_at": timestamp,
        "updated_at": timestamp,
        "source": "web",
        "pending_sync": True,
        "synced_at": None
    }

    # Ensure device entry exists
    if "devices" not in file_data:
        file_data["devices"] = {}
    if device_id not in file_data["devices"]:
        file_data["devices"][device_id] = {"projects": []}
    if "projects" not in file_data["devices"][device_id]:
        file_data["devices"][device_id]["projects"] = []

    file_data["devices"][device_id]["projects"].append(project)
    _write_json_file(PROJECTS_FILE, file_data)
    return {"success": True, "project": project}


def create_note(data: Dict, device_id: str = "web") -> Dict:
    """Create a new note from web dashboard.

    Notes are stored under the specified device_id with pending_sync flag
    for bi-directional sync back to Android.
    """
    file_data = _read_json_file(NOTES_FILE) or {"devices": {}}

    timestamp = int(datetime.now().timestamp() * 1000)
    note = {
        "id": _generate_id(),
        "title": data.get("title", "New Note"),
        "content": data.get("content", ""),
        "category": data.get("category", "General"),
        "tags": data.get("tags", []),
        "priority": data.get("priority", "NORMAL"),
        "note_type": data.get("note_type", "TEXT"),
        "is_pinned": data.get("is_pinned", False),
        "is_favorite": data.get("is_favorite", False),
        "is_archived": False,
        "ai_protection_level": data.get("ai_protection_level", "STANDARD"),
        "ai_can_read": True,
        "ai_can_edit": False,
        "ai_can_summarize": True,
        "ai_can_extract_tasks": True,
        "project_id": data.get("project_id"),
        "folder_path": data.get("folder_path"),
        "createdAt": timestamp,
        "updatedAt": timestamp,
        "source": "web",
        "pending_sync": True,
        "synced_at": None
    }

    # Ensure device entry exists
    if "devices" not in file_data:
        file_data["devices"] = {}
    if device_id not in file_data["devices"]:
        file_data["devices"][device_id] = {"notes": []}
    if "notes" not in file_data["devices"][device_id]:
        file_data["devices"][device_id]["notes"] = []

    file_data["devices"][device_id]["notes"].append(note)
    _write_json_file(NOTES_FILE, file_data)
    return {"success": True, "note": note}


def create_automation(data: Dict, device_id: str = "web") -> Dict:
    """Create a new automation from web dashboard.

    Automations are stored under the specified device_id with pending_sync flag
    for bi-directional sync back to Android.
    """
    file_data = _read_json_file(AUTOMATIONS_FILE) or {}

    timestamp = int(datetime.now().timestamp() * 1000)
    automation = {
        "id": _generate_id(),
        "name": data.get("name", "New Automation"),
        "description": data.get("description", ""),
        "enabled": data.get("enabled", True),
        "trigger_type": data.get("trigger_type", "MANUAL"),
        "schedule_expression": data.get("schedule_expression"),
        "voice_trigger_phrase": data.get("voice_trigger_phrase"),
        "ai_command": data.get("ai_command", ""),
        "ai_permission_level": data.get("ai_permission_level", "FULL"),
        "use_note_as_context": data.get("use_note_as_context", True),
        "max_iterations": data.get("max_iterations", 50),
        "timeout_minutes": data.get("timeout_minutes", 30),
        "project_id": data.get("project_id"),
        "note_id": data.get("note_id"),
        "output_destination": data.get("output_destination", "SHOW_DIALOG"),
        "output_note_title": data.get("output_note_title"),
        "icon_name": data.get("icon_name", "ic_automation"),
        "color": data.get("color", "#FF9800"),
        "created_at": timestamp,
        "last_run_at": None,
        "last_run_status": "NEVER_RUN",
        "last_run_result": None,
        "run_count": 0,
        "source": "web",
        "pending_sync": True,
        "synced_at": None
    }

    # Ensure device entry exists
    if device_id not in file_data:
        file_data[device_id] = {"automations": []}
    if "automations" not in file_data[device_id]:
        file_data[device_id]["automations"] = []

    file_data[device_id]["automations"].append(automation)
    _write_json_file(AUTOMATIONS_FILE, file_data)
    return {"success": True, "automation": automation}


def get_pending_sync_items(device_id: str) -> Dict:
    """Get all items pending sync for a specific device."""
    pending = {
        "projects": [],
        "notes": [],
        "automations": []
    }

    # Check projects
    projects_data = _read_json_file(PROJECTS_FILE) or {"devices": {}}
    device_projects = projects_data.get("devices", {}).get(device_id, {}).get("projects", [])
    pending["projects"] = [p for p in device_projects if p.get("pending_sync")]

    # Check notes
    notes_data = _read_json_file(NOTES_FILE) or {"devices": {}}
    device_notes = notes_data.get("devices", {}).get(device_id, {}).get("notes", [])
    pending["notes"] = [n for n in device_notes if n.get("pending_sync")]

    # Check automations
    auto_data = _read_json_file(AUTOMATIONS_FILE) or {}
    device_autos = auto_data.get(device_id, {}).get("automations", [])
    pending["automations"] = [a for a in device_autos if a.get("pending_sync")]

    return pending


def mark_items_synced(device_id: str, item_type: str, item_ids: List[str]) -> bool:
    """Mark items as synced after successful push to Android."""
    timestamp = int(datetime.now().timestamp() * 1000)

    if item_type == "projects":
        file_data = _read_json_file(PROJECTS_FILE) or {"devices": {}}
        projects = file_data.get("devices", {}).get(device_id, {}).get("projects", [])
        for p in projects:
            if p.get("id") in item_ids:
                p["pending_sync"] = False
                p["synced_at"] = timestamp
        _write_json_file(PROJECTS_FILE, file_data)

    elif item_type == "notes":
        file_data = _read_json_file(NOTES_FILE) or {"devices": {}}
        notes = file_data.get("devices", {}).get(device_id, {}).get("notes", [])
        for n in notes:
            if n.get("id") in item_ids:
                n["pending_sync"] = False
                n["synced_at"] = timestamp
        _write_json_file(NOTES_FILE, file_data)

    elif item_type == "automations":
        file_data = _read_json_file(AUTOMATIONS_FILE) or {}
        autos = file_data.get(device_id, {}).get("automations", [])
        for a in autos:
            if a.get("id") in item_ids:
                a["pending_sync"] = False
                a["synced_at"] = timestamp
        _write_json_file(AUTOMATIONS_FILE, file_data)

    return True


# ============ Project Todos ============

def get_project_todos(project_id: str) -> List[Dict]:
    """Get all todos for a specific project."""
    file_data = _read_json_file(TODOS_FILE)
    if not file_data:
        return []

    project_todos = file_data.get("projects", {}).get(project_id, {}).get("todos", [])

    # Add formatted timestamps
    for todo in project_todos:
        todo["time_ago"] = _time_ago(todo.get("createdAt", 0))
        todo["updated_ago"] = _time_ago(todo.get("updatedAt", 0))

    # Sort by sortOrder (maintains user's order), then by createdAt
    project_todos.sort(key=lambda x: (x.get("sortOrder", 0), x.get("createdAt", 0)))
    return project_todos


def create_project_todo(project_id: str, data: Dict) -> Dict:
    """Create a new todo for a project.

    Todo schema aligned with Android TodoItem:
    - id, projectId, content, priority
    - userVerificationStatus, aiExecutionStatus
    - createdAt, updatedAt, completedAt, sortOrder
    """
    file_data = _read_json_file(TODOS_FILE) or {"version": 1, "projects": {}}

    timestamp = int(datetime.now().timestamp() * 1000)
    todo = {
        "id": _generate_id(),
        "projectId": project_id,
        "content": data.get("content", ""),
        "enhancedContent": data.get("enhancedContent"),
        "priority": data.get("priority", "MEDIUM"),
        "userVerificationStatus": "PENDING_VERIFICATION",
        "aiExecutionStatus": "NOT_STARTED",
        "linkedSessionId": None,
        "linkedMessageId": None,
        "createdAt": timestamp,
        "updatedAt": timestamp,
        "completedAt": None,
        "sentToChatAt": None,
        "sortOrder": timestamp,
        "source": "web",
        "pending_sync": True
    }

    # Ensure project entry exists
    if "projects" not in file_data:
        file_data["projects"] = {}
    if project_id not in file_data["projects"]:
        file_data["projects"][project_id] = {"todos": []}
    if "todos" not in file_data["projects"][project_id]:
        file_data["projects"][project_id]["todos"] = []

    file_data["projects"][project_id]["todos"].append(todo)
    file_data["updated"] = datetime.now().timestamp()
    _write_json_file(TODOS_FILE, file_data)

    return {"success": True, "todo": todo}


def update_project_todo(project_id: str, todo_id: str, data: Dict) -> Dict:
    """Update a todo's content, priority, or status."""
    file_data = _read_json_file(TODOS_FILE)
    if not file_data:
        return {"error": "Todos file not found"}

    todos = file_data.get("projects", {}).get(project_id, {}).get("todos", [])

    for i, todo in enumerate(todos):
        if todo.get("id") == todo_id:
            timestamp = int(datetime.now().timestamp() * 1000)

            # Update allowed fields
            if "content" in data:
                todo["content"] = data["content"]
            if "priority" in data:
                todo["priority"] = data["priority"]
            if "userVerificationStatus" in data:
                todo["userVerificationStatus"] = data["userVerificationStatus"]
                # Set completedAt if marking as complete
                if data["userVerificationStatus"] == "VERIFIED_COMPLETE":
                    todo["completedAt"] = timestamp
                elif data["userVerificationStatus"] == "PENDING_VERIFICATION":
                    todo["completedAt"] = None
            if "aiExecutionStatus" in data:
                todo["aiExecutionStatus"] = data["aiExecutionStatus"]
            if "sortOrder" in data:
                todo["sortOrder"] = data["sortOrder"]

            todo["updatedAt"] = timestamp
            todo["pending_sync"] = True

            file_data["projects"][project_id]["todos"][i] = todo
            file_data["updated"] = datetime.now().timestamp()
            _write_json_file(TODOS_FILE, file_data)

            return {"success": True, "todo": todo}

    return {"error": "Todo not found"}


def delete_project_todo(project_id: str, todo_id: str) -> Dict:
    """Delete a todo from a project."""
    file_data = _read_json_file(TODOS_FILE)
    if not file_data:
        return {"error": "Todos file not found"}

    todos = file_data.get("projects", {}).get(project_id, {}).get("todos", [])
    original_len = len(todos)

    file_data["projects"][project_id]["todos"] = [
        t for t in todos if t.get("id") != todo_id
    ]

    if len(file_data["projects"][project_id]["todos"]) < original_len:
        file_data["updated"] = datetime.now().timestamp()
        _write_json_file(TODOS_FILE, file_data)
        return {"success": True}

    return {"error": "Todo not found"}


def reorder_project_todos(project_id: str, todo_ids: List[str]) -> Dict:
    """Reorder todos by updating their sortOrder based on position in list."""
    file_data = _read_json_file(TODOS_FILE)
    if not file_data:
        return {"error": "Todos file not found"}

    todos = file_data.get("projects", {}).get(project_id, {}).get("todos", [])
    timestamp = int(datetime.now().timestamp() * 1000)

    # Create a map of todo_id to new sortOrder
    order_map = {tid: idx for idx, tid in enumerate(todo_ids)}

    for todo in todos:
        if todo.get("id") in order_map:
            todo["sortOrder"] = order_map[todo["id"]]
            todo["updatedAt"] = timestamp
            todo["pending_sync"] = True

    file_data["updated"] = datetime.now().timestamp()
    _write_json_file(TODOS_FILE, file_data)

    return {"success": True}
