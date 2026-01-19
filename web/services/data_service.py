"""
Data Service - Read ~/.shadowai/ JSON files
"""

import json
import os
import base64
import hashlib
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import notifier
from .notifier_service import get_notifier

# SECURITY: Thread lock for file I/O operations to prevent race conditions
_file_lock = threading.Lock()

# Shadow data directory
SHADOWAI_DIR = Path.home() / ".shadowai"
CARDS_FILE = SHADOWAI_DIR / "cards.json"
COLLECTIONS_FILE = SHADOWAI_DIR / "collections.json"
PROJECTS_FILE = SHADOWAI_DIR / "projects.json"
NOTES_FILE = SHADOWAI_DIR / "notes.json"
SESSIONS_FILE = SHADOWAI_DIR / "sessions.json"
AUTOMATIONS_FILE = SHADOWAI_DIR / "automations.json"
AUTOMATION_LOGS_FILE = SHADOWAI_DIR / "automation_logs.json"
AGENTS_FILE = SHADOWAI_DIR / "agents.json"
SYNC_KEYS_FILE = SHADOWAI_DIR / "sync_keys.json"
ANALYTICS_FILE = SHADOWAI_DIR / "analytics.json"
EMAILS_FILE = SHADOWAI_DIR / "verified_emails.json"

# Note encryption constants (must match Android SyncEncryption.kt)
SYNC_ENC_PREFIX = "SYNC_ENC:"
SALT_FILE = SHADOWAI_DIR / "salt.bin"


def _get_or_create_salt() -> bytes:
    """Get the persistent random salt or create it if it doesn't exist."""
    try:
        SHADOWAI_DIR.mkdir(parents=True, exist_ok=True)
        if SALT_FILE.exists():
            with open(SALT_FILE, "rb") as f:
                salt = f.read()
                if len(salt) == 16:
                    return salt

        # Create new random salt
        import secrets

        salt = secrets.token_bytes(16)
        with open(SALT_FILE, "wb") as f:
            f.write(salt)
        return salt
    except Exception as e:
        print(f"Error managing salt: {e}")
        return b"ShadowAI_Sync_2024"  # Safe fallback


DYNAMIC_SALT = _get_or_create_salt()
ITERATION_COUNT = 50000
KEY_LENGTH = 32  # 256 bits
IV_LENGTH = 12
TAG_LENGTH = 16  # 128 bits in bytes


def _derive_encryption_key(password: str) -> bytes:
    """Derive AES-256 key using PBKDF2-HMAC-SHA256 (matches Android)."""
    import hashlib

    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        DYNAMIC_SALT,
        ITERATION_COUNT,
        dklen=KEY_LENGTH,
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
        data = content[len(SYNC_ENC_PREFIX) :]
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

        return plaintext.decode("utf-8")

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
        ciphertext = aesgcm.encrypt(iv, content.encode("utf-8"), None)

        # Format: SYNC_ENC:[IV]:[Ciphertext]
        iv_b64 = base64.b64encode(iv).decode("ascii")
        ciphertext_b64 = base64.b64encode(ciphertext).decode("ascii")

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
        with open(SYNC_KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(keys_data, f)
        return True
    except Exception as e:
        print(f"Failed to save sync key: {e}")
        return False


def _read_json_file(filepath: Path) -> Optional[Dict]:
    """Read and parse a JSON file with thread safety."""
    if not filepath.exists():
        return None
    try:
        with _file_lock:
            with open(filepath, "r", encoding="utf-8") as f:
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
    except (ValueError, OSError, OverflowError):
        return "Unknown"


def _coerce_timestamp_ms(raw_ts: Any) -> int:
    if isinstance(raw_ts, (int, float)):
        ts = int(raw_ts)
        if ts < 10_000_000_000:
            return ts * 1000
        return ts
    return 0


def _normalize_automation_log(entry: Dict) -> Optional[Dict]:
    if not isinstance(entry, dict):
        return None
    raw_ts = (
        entry.get("timestamp")
        or entry.get("startedAt")
        or entry.get("completedAt")
        or entry.get("created_at")
    )
    ts_ms = _coerce_timestamp_ms(raw_ts)
    status = str(entry.get("status", "")).upper()
    success = entry.get("success")
    if not isinstance(success, bool):
        success = status in {"SUCCESS", "PARTIAL_SUCCESS", "COMPLETED", "DONE"}
    message = entry.get("message") or entry.get("summary") or entry.get("details") or ""
    if not message and status:
        message = status.replace("_", " ").title()
    return {
        "timestamp": _format_timestamp(ts_ms),
        "timestamp_raw": ts_ms,
        "success": bool(success),
        "status": status,
        "message": message,
        "details": entry.get("details"),
        "duration_ms": entry.get("durationMs") or entry.get("duration_ms"),
        "automation_id": entry.get("automationId") or entry.get("automation_id"),
    }


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
    except (ValueError, OSError, OverflowError):
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


def _infer_note_content_port(
    device_id: Optional[str], explicit_port: Any
) -> Optional[int]:
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

        devices.append(
            {
                "id": device_id,
                "name": device_info.get("name", device_id),
                "status": "online" if is_online else "offline",
                "ip": device_info.get("ip"),
                "last_seen": last_seen,
                "last_seen_formatted": _time_ago(int(last_seen * 1000))
                if last_seen
                else "Never",
            }
        )
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
    """Get all projects, optionally filtered by device.

    Projects are deduplicated by ID - if the same project exists on multiple device entries
    (e.g., com.shadowai.release and com.shadowai.release.debug on the same phone),
    we keep the most recently updated version.
    """
    data = _read_json_file(PROJECTS_FILE)
    if not data:
        return []

    # Projects are stored under "devices" key
    devices_data = data.get("devices", {})
    if not isinstance(devices_data, dict):
        devices_data = {}

    # Track best project per ID (deduplicate across device entries)
    projects_by_id = {}

    for proj_device_id, device_info in devices_data.items():
        if device_id and proj_device_id != device_id:
            continue

        synced_at = device_info.get("synced_at", 0)

        for project in device_info.get("projects", []):
            if not isinstance(project, dict):
                continue
            project_id = project.get("id", project.get("path", ""))
            updated_at = project.get("updated_at", 0)

            # Check if we already have this project from another device entry
            if project_id in projects_by_id:
                existing = projects_by_id[project_id]
                # Keep the more recently updated or synced version
                if updated_at <= existing.get("updated_at", 0):
                    continue

            projects_by_id[project_id] = {
                "id": project_id,
                "name": project.get(
                    "name", os.path.basename(project.get("path", "Unknown"))
                ),
                "path": project.get("path", ""),
                "device_id": proj_device_id,
                "device_name": device_info.get("name", proj_device_id),
                "updated_at": updated_at,
                "updated_at_formatted": _format_timestamp(updated_at),
                "time_ago": _time_ago(updated_at),
            }

    # Convert to list and sort by updated_at descending
    projects = list(projects_by_id.values())
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


def get_notes(
    device_id: Optional[str] = None, search: Optional[str] = None
) -> List[Dict]:
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
    fingerprint_to_latest: Dict[
        str, tuple
    ] = {}  # base_fingerprint -> (device_id, last_seen)
    for dev_id, dev_info in devices_data.items():
        # Extract base fingerprint (everything before the optional :package.name suffix)
        base_fingerprint = dev_id.split(":com.")[0] if ":com." in dev_id else dev_id
        last_seen = dev_info.get("last_seen", 0)

        current = fingerprint_to_latest.get(base_fingerprint)
        if current is None or last_seen > current[1]:
            fingerprint_to_latest[base_fingerprint] = (dev_id, last_seen)

    # Build set of device IDs to skip (stale entries with same base fingerprint as a newer entry)
    stale_device_ids = set()
    for dev_id in devices_data.keys():
        base_fingerprint = dev_id.split(":com.")[0] if ":com." in dev_id else dev_id
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
        note_content_port = _infer_note_content_port(
            note_device_id, device_info.get("note_content_port")
        )

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
                    "time_ago": _time_ago(updated_at),
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
                        "device_ip": device_ips[0]
                        if device_ips
                        else device_info.get("ip"),
                        "device_ips": device_ips,
                        "note_content_port": note_content_port,
                        "device_last_seen": device_last_seen,
                        "updated_at": updated_at,
                        "updated_at_formatted": _format_timestamp(updated_at),
                        "time_ago": _time_ago(updated_at),
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
                        "updated_at": note.get("updatedAt", 0),
                    }
    return None


# ============ Sessions ============


def _load_sessions_file() -> Dict[str, Any]:
    """Load sessions data, normalizing legacy shapes."""
    data = _read_json_file(SESSIONS_FILE)
    if not data:
        return {"sessions": {}}

    sessions = data.get("sessions")
    if isinstance(sessions, list):
        sessions_map = {}
        for session in sessions:
            if not isinstance(session, dict):
                continue
            session_id = session.get("id") or session.get("sessionId") or _generate_id()
            session["id"] = session_id
            sessions_map[session_id] = session
        data = {"sessions": sessions_map}
    elif not isinstance(sessions, dict):
        data["sessions"] = {}

    return data


def _session_last_activity(session: Dict[str, Any]) -> int:
    """Find last activity timestamp for sorting."""
    last_activity = session.get("lastActivityAt") or session.get("last_activity_at")
    if isinstance(last_activity, int) and last_activity > 0:
        return last_activity

    messages = session.get("messages", [])
    if isinstance(messages, list) and messages:
        last_message = messages[-1]
        if isinstance(last_message, dict):
            ts = last_message.get("timestamp")
            if isinstance(ts, int) and ts > 0:
                return ts

    for key in ("timestamp", "updatedAt", "updated_at", "createdAt", "created_at"):
        ts = session.get(key)
        if isinstance(ts, int) and ts > 0:
            return ts

    return 0


def _session_project_id(session: Dict[str, Any]) -> Optional[str]:
    project_id = session.get("projectId") or session.get("project_id")
    return project_id if isinstance(project_id, str) else None


def _session_device_id(session: Dict[str, Any]) -> Optional[str]:
    device_id = session.get("device_id") or session.get("deviceId")
    return device_id if isinstance(device_id, str) else None


def _session_title(session: Dict[str, Any]) -> str:
    title = session.get("title")
    if isinstance(title, str) and title.strip():
        return title

    messages = session.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            if role == "user":
                content = str(msg.get("content", "")).strip()
                if content:
                    return (content[:47] + "...") if len(content) > 50 else content

    return "New Session"


def _normalize_backend_value(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _extract_backend_meta(payload: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    if not isinstance(payload, dict):
        return {}
    return {
        "backend_type": _normalize_backend_value(
            payload.get("backend_type") or payload.get("backendType")
        ),
        "provider": _normalize_backend_value(payload.get("provider")),
        "model": _normalize_backend_value(payload.get("model")),
    }


def get_sessions(
    device_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Get session metadata list, optionally filtered."""
    data = _load_sessions_file()
    sessions = []

    for session in data.get("sessions", {}).values():
        if not isinstance(session, dict):
            continue

        if device_id and _session_device_id(session) != device_id:
            continue
        if project_id and _session_project_id(session) != project_id:
            continue

        last_activity = _session_last_activity(session)
        messages = session.get("messages", [])
        last_message = None
        if isinstance(messages, list) and messages:
            last_message = messages[-1] if isinstance(messages[-1], dict) else None

        sessions.append(
            {
                "id": session.get("id") or session.get("sessionId"),
                "projectId": _session_project_id(session),
                "title": _session_title(session),
                "device_id": _session_device_id(session),
                "device_name": session.get("device_name") or session.get("deviceName"),
                "backend_type": session.get("backend_type")
                or session.get("backendType"),
                "provider": session.get("provider"),
                "model": session.get("model"),
                "backend_last_changed_at": session.get("backend_last_changed_at"),
                "backend_change_count": session.get("backend_change_count", 0),
                "backend_switch_count": session.get("backend_switch_count", 0),
                "model_switch_count": session.get("model_switch_count", 0),
                "lastActivityAt": last_activity,
                "time_ago": _time_ago(last_activity),
                "message_count": len(messages) if isinstance(messages, list) else 0,
                "last_message": (last_message or {}).get("content", "")
                if last_message
                else "",
                "is_running": (datetime.now().timestamp() * 1000 - last_activity)
                <= 600000
                if last_activity
                else False,
            }
        )

    sessions.sort(key=lambda s: s.get("lastActivityAt", 0), reverse=True)
    if isinstance(limit, int) and limit > 0:
        sessions = sessions[:limit]
    return sessions


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get full session with messages."""
    data = _load_sessions_file()
    session = data.get("sessions", {}).get(session_id)
    if isinstance(session, dict):
        session["id"] = session.get("id") or session_id
        session["title"] = _session_title(session)
        return session

    # Backward compatibility: search list if needed
    for item in data.get("sessions", {}).values():
        if isinstance(item, dict) and item.get("id") == session_id:
            item["title"] = _session_title(item)
            return item
    return None


def upsert_session(session: Dict[str, Any]) -> Dict[str, Any]:
    """Insert or update a session record."""
    data = _load_sessions_file()
    sessions = data.get("sessions", {})

    session_id = session.get("id") or session.get("sessionId") or _generate_id()
    existing = sessions.get(session_id, {})

    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(session)
    merged["id"] = session_id

    if "pending_sync" not in session:
        prefers_backend = any(
            key in session
            for key in (
                "preferred_backend_key",
                "preferred_backend_type",
                "preferred_provider",
                "preferred_model",
                "preferred_model_key",
                "backend_change_pending",
            )
        )
        if session.get("backend_change_pending") or prefers_backend:
            merged["pending_sync"] = True

    if "messages" not in session and "messages" in existing:
        merged["messages"] = existing.get("messages", [])

    if "createdAt" not in merged and "created_at" not in merged:
        merged["createdAt"] = int(datetime.now().timestamp() * 1000)

    last_activity = _session_last_activity(merged)
    if last_activity:
        merged["lastActivityAt"] = last_activity
        merged["timestamp"] = last_activity

    merged["title"] = _session_title(merged)

    sessions[session_id] = merged
    data["sessions"] = sessions
    _write_json_file(SESSIONS_FILE, data)
    return merged


def delete_session(session_id: str) -> Dict[str, Any]:
    """Delete a session from the local cache."""
    data = _load_sessions_file()
    sessions = data.get("sessions", {})

    if session_id in sessions:
        del sessions[session_id]
        data["sessions"] = sessions
        _write_json_file(SESSIONS_FILE, data)
        return {"success": True, "message": "Session deleted"}
    
    # Check backward compatibility list-style sessions
    if isinstance(sessions, list):
        original_len = len(sessions)
        sessions = [s for s in sessions if (s.get("id") != session_id and s.get("sessionId") != session_id)]
        if len(sessions) < original_len:
            data["sessions"] = sessions
            _write_json_file(SESSIONS_FILE, data)
            return {"success": True, "message": "Session deleted"}

    return {"success": False, "error": "Session not found"}


def save_sessions_from_device(
    device_id: str, device_name: str, sessions: List[Dict[str, Any]]
) -> bool:
    """Merge sessions from a device into the local store."""
    data = _load_sessions_file()
    sessions_map = data.get("sessions", {})

    for session in sessions:
        if not isinstance(session, dict):
            continue

        session_id = session.get("id") or session.get("sessionId") or _generate_id()
        session["id"] = session_id
        session.setdefault("device_id", device_id)
        session.setdefault("device_name", device_name)
        # Device-originated sessions shouldn't remain pending for sync back.
        session.pop("pending_sync", None)

        incoming_last = _session_last_activity(session)
        existing = sessions_map.get(session_id)

        if isinstance(existing, dict):
            existing_last = _session_last_activity(existing)
            if incoming_last < existing_last:
                # Keep newer session but still merge metadata
                existing.update({k: v for k, v in session.items() if k != "messages"})
                sessions_map[session_id] = existing
                continue

        sessions_map[session_id] = session

    data["sessions"] = sessions_map
    return _write_json_file(SESSIONS_FILE, data)


def append_session_message(
    session_id: str,
    message: Dict[str, Any],
    delta: Optional[str] = None,
    is_final: bool = False,
    session_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append or update a session message (supports streaming updates)."""
    data = _load_sessions_file()
    sessions = data.get("sessions", {})

    if not isinstance(message, dict):
        message = {}

    now_ms = int(datetime.now().timestamp() * 1000)

    session = sessions.get(session_id)
    if not isinstance(session, dict):
        session = {
            "id": session_id,
            "projectId": message.get("projectId") or message.get("project_id"),
            "createdAt": now_ms,
            "messages": [],
        }

    prev_backend = _extract_backend_meta(session)

    if isinstance(session_meta, dict):
        session.update({k: v for k, v in session_meta.items() if v is not None})

    if isinstance(session_meta, dict):
        for key in (
            "device_id",
            "deviceId",
            "device_name",
            "deviceName",
            "backend_type",
            "backendType",
            "provider",
            "model",
        ):
            if message.get(key) is None:
                val = session_meta.get(key)
                if val is not None:
                    message[key] = val

    for key in (
        "device_id",
        "deviceId",
        "device_name",
        "deviceName",
    ):
        val = message.get(key)
        if val is not None:
            session[key] = val

    incoming_backend = _extract_backend_meta(session_meta)
    message_backend = _extract_backend_meta(message)
    for key, val in message_backend.items():
        if val is not None:
            incoming_backend[key] = val

    timestamp = message.get("timestamp")
    if not isinstance(timestamp, int) or timestamp <= 0:
        timestamp = now_ms

    if incoming_backend:
        changes = {}
        updated_backend = dict(prev_backend)
        for key, val in incoming_backend.items():
            if val is None:
                continue
            updated_backend[key] = val
            if val != prev_backend.get(key):
                changes[key] = {"from": prev_backend.get(key), "to": val}

        if changes:
            if updated_backend.get("backend_type"):
                session["backend_type"] = updated_backend["backend_type"]
            if updated_backend.get("provider"):
                session["provider"] = updated_backend["provider"]
            if updated_backend.get("model"):
                session["model"] = updated_backend["model"]

            history = session.get("backend_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "timestamp": timestamp,
                    "from": prev_backend,
                    "to": updated_backend,
                    "changes": changes,
                }
            )
            if len(history) > 50:
                history = history[-50:]
            session["backend_history"] = history
            session["backend_last_changed_at"] = history[-1]["timestamp"]
            session["backend_change_count"] = int(
                session.get("backend_change_count", 0)
            ) + 1
            if any(key in changes for key in ("backend_type", "provider")):
                session["backend_switch_count"] = int(
                    session.get("backend_switch_count", 0)
                ) + 1
            if "model" in changes:
                session["model_switch_count"] = int(
                    session.get("model_switch_count", 0)
                ) + 1

    if session.get("backend_change_pending"):
        pref_type = _normalize_backend_value(session.get("preferred_backend_type"))
        pref_provider = _normalize_backend_value(session.get("preferred_provider"))
        pref_model = _normalize_backend_value(
            session.get("preferred_model") or session.get("preferred_model_key")
        )
        if pref_type or pref_provider or pref_model:
            current_backend = _extract_backend_meta(session)
            matches = True
            if pref_type and current_backend.get("backend_type"):
                matches = pref_type.lower() == str(current_backend.get("backend_type")).lower()
            if matches and pref_provider and current_backend.get("provider"):
                matches = pref_provider.lower() == str(current_backend.get("provider")).lower()
            if matches and pref_model and current_backend.get("model"):
                matches = pref_model.lower() == str(current_backend.get("model")).lower()
            if matches:
                session["backend_change_pending"] = False
                session["backend_change_completed_at"] = timestamp

    messages = session.get("messages")
    if not isinstance(messages, list):
        messages = []

    message_id = message.get("id") or message.get("messageId") or _generate_id()
    role = message.get("role", "assistant")

    content = message.get("content")
    if not isinstance(content, str):
        content = ""

    if isinstance(delta, str) and delta:
        content = content or ""

    existing_index = None
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("id") == message_id:
            existing_index = idx
            break

    if existing_index is None:
        new_message = dict(message)
        new_message["id"] = message_id
        new_message["role"] = role
        new_message["timestamp"] = timestamp
        new_message["content"] = delta if isinstance(delta, str) and delta else content
        new_message["is_final"] = bool(is_final)
        messages.append(new_message)
        message = new_message
    else:
        existing = messages[existing_index]
        if isinstance(existing, dict):
            if isinstance(delta, str) and delta:
                existing["content"] = (existing.get("content", "") or "") + delta
            elif content:
                existing["content"] = content
            existing["timestamp"] = timestamp
            existing["role"] = role
            if is_final:
                existing["is_final"] = True
            existing.update({k: v for k, v in message.items() if k not in ("content",)})
            message = existing
            messages[existing_index] = existing

    session["messages"] = messages
    session["lastActivityAt"] = timestamp
    session["timestamp"] = timestamp
    session["title"] = _session_title(session)
    sessions[session_id] = session
    data["sessions"] = sessions
    if is_final and content:
        # Trigger background quality sentinel check
        notifier = get_notifier()
        provider = session.get("provider", "shadow")
        
        # Determine content type for sentinel
        content_type = "prompt" if role == "user" else "response"
        
        # Non-blocking check (since we are in data service)
        def check_quality():
            notifier.notify_quality_issue(content, content_type, session_id, provider)
            
        import threading
        threading.Thread(target=check_quality, daemon=True).start()

    _write_json_file(SESSIONS_FILE, data)
    return {"session": session, "message": message}


def delete_note(note_id: str) -> Dict:
    """Delete a note from the local cache.

    Removes the note from notes.json. Note: This is a local deletion only.
    The note will reappear if the device syncs it again unless the device
    also deletes it.

    Returns: {"success": True} or {"success": False, "error": "..."}
    """
    data = _read_json_file(NOTES_FILE)
    if not data:
        return {"success": False, "error": "Notes file not found"}

    devices_data = data.get("devices", {})
    deleted = False

    for device_id, device_info in devices_data.items():
        notes = device_info.get("notes", [])
        original_len = len(notes)
        device_info["notes"] = [n for n in notes if n.get("id") != note_id]
        if len(device_info["notes"]) < original_len:
            deleted = True

    if deleted:
        _write_json_file(NOTES_FILE, data)
        return {"success": True}
    else:
        return {"success": False, "error": "Note not found"}


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
            with open(NOTES_FILE, "w", encoding="utf-8") as f:
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
            automations.append(
                {
                    "id": auto.get("id", ""),
                    "name": auto.get("name", "Unnamed"),
                    "description": auto.get("description", ""),
                    "enabled": auto.get("enabled", False),
                    "schedule": auto.get("schedule", ""),
                    "trigger": auto.get("trigger", "manual"),
                    "last_run": auto.get("last_run"),
                    "last_run_formatted": _time_ago(auto.get("last_run", 0))
                    if auto.get("last_run")
                    else "Never",
                    "device_id": auto_device_id,
                }
            )

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
    data = _read_json_file(AUTOMATION_LOGS_FILE)
    if not data:
        return []

    raw_logs: List[Dict] = []
    if isinstance(data, list):
        raw_logs = data
    elif isinstance(data, dict):
        for key in ("logs", "automation_logs"):
            entries = data.get(key)
            if isinstance(entries, list):
                raw_logs = entries
                break
        if not raw_logs:
            for key in ("by_automation", "automations"):
                entries = data.get(key)
                if isinstance(entries, dict):
                    automation_entries = entries.get(automation_id)
                    if isinstance(automation_entries, list):
                        raw_logs = automation_entries
                        break
    if not raw_logs:
        return []

    normalized = []
    for entry in raw_logs:
        normalized_entry = _normalize_automation_log(entry)
        if not normalized_entry:
            continue
        if normalized_entry.get("automation_id") and (
            normalized_entry["automation_id"] != automation_id
        ):
            continue
        normalized.append(normalized_entry)

    normalized.sort(key=lambda item: item.get("timestamp_raw", 0), reverse=True)
    return normalized


def update_automation(automation_id: str, data: Dict, device_id: str = "web") -> Dict:
    """Update an existing automation."""
    file_data = _read_json_file(AUTOMATIONS_FILE) or {}

    # Find the automation across all devices
    found = False
    for dev_id, device_data in file_data.items():
        automations = device_data.get("automations", [])
        for i, auto in enumerate(automations):
            if auto.get("id") == automation_id:
                # Update fields (preserve id and created_at)
                updated = {
                    "id": automation_id,
                    "name": data.get("name", auto.get("name", "Unnamed")),
                    "description": data.get("description", auto.get("description", "")),
                    "enabled": data.get("enabled", auto.get("enabled", True)),
                    "trigger_type": data.get(
                        "trigger_type", auto.get("trigger_type", "MANUAL")
                    ),
                    "schedule_expression": data.get(
                        "schedule_expression", auto.get("schedule_expression")
                    ),
                    "schedule_timezone": data.get(
                        "schedule_timezone", auto.get("schedule_timezone")
                    ),
                    "voice_trigger_phrase": data.get(
                        "voice_trigger_phrase", auto.get("voice_trigger_phrase")
                    ),
                    "ai_command": data.get("ai_command", auto.get("ai_command", "")),
                    "ai_permission_level": data.get(
                        "ai_permission_level", auto.get("ai_permission_level", "FULL")
                    ),
                    "use_note_as_context": data.get(
                        "use_note_as_context", auto.get("use_note_as_context", True)
                    ),
                    "max_iterations": data.get(
                        "max_iterations", auto.get("max_iterations", 50)
                    ),
                    "timeout_minutes": data.get(
                        "timeout_minutes", auto.get("timeout_minutes", 30)
                    ),
                    "allow_autonomous_edits": data.get(
                        "allow_autonomous_edits",
                        auto.get("allow_autonomous_edits", False),
                    ),
                    "execution_mode": data.get(
                        "execution_mode", auto.get("execution_mode", "SINGLE_SHOT")
                    ),
                    "team_type": data.get("team_type", auto.get("team_type")),
                    "require_approval_for": data.get(
                        "require_approval_for", auto.get("require_approval_for", [])
                    ),
                    "automation_type": data.get(
                        "automation_type", auto.get("automation_type", "TEXT")
                    ),
                    "image_provider": data.get(
                        "image_provider", auto.get("image_provider")
                    ),
                    "image_quality": data.get(
                        "image_quality", auto.get("image_quality")
                    ),
                    "image_width": data.get(
                        "image_width", auto.get("image_width", 1024)
                    ),
                    "image_height": data.get(
                        "image_height", auto.get("image_height", 1024)
                    ),
                    "image_style": data.get("image_style", auto.get("image_style")),
                    "n8n_workflow_id": data.get(
                        "n8n_workflow_id", auto.get("n8n_workflow_id")
                    ),
                    "n8n_workflow_name": data.get(
                        "n8n_workflow_name", auto.get("n8n_workflow_name")
                    ),
                    "project_id": data.get("project_id", auto.get("project_id")),
                    "note_id": data.get("note_id", auto.get("note_id")),
                    "output_destination": data.get(
                        "output_destination",
                        auto.get("output_destination", "SHOW_DIALOG"),
                    ),
                    "output_note_title": data.get(
                        "output_note_title", auto.get("output_note_title")
                    ),
                    "output_file_path": data.get(
                        "output_file_path", auto.get("output_file_path")
                    ),
                    "file_operations": data.get(
                        "file_operations", auto.get("file_operations", [])
                    ),
                    "icon_name": data.get(
                        "icon_name", auto.get("icon_name", "ic_automation")
                    ),
                    "color": data.get("color", auto.get("color", "#FF9800")),
                    "created_at": auto.get("created_at"),
                    "last_run_at": auto.get("last_run_at"),
                    "last_run_status": auto.get("last_run_status", "NEVER_RUN"),
                    "last_run_result": auto.get("last_run_result"),
                    "run_count": auto.get("run_count", 0),
                    "source": auto.get("source", "web"),
                    "pending_sync": True,
                    "synced_at": None,
                    "updated_at": int(datetime.now().timestamp() * 1000),
                }
                file_data[dev_id]["automations"][i] = updated
                found = True
                break
        if found:
            break

    if not found:
        return {"success": False, "error": "Automation not found"}

    _write_json_file(AUTOMATIONS_FILE, file_data)
    return {"success": True, "automation": updated}


def delete_automation(automation_id: str) -> Dict:
    """Delete an automation by ID."""
    file_data = _read_json_file(AUTOMATIONS_FILE) or {}

    found = False
    deleted_name = None
    for dev_id, device_data in file_data.items():
        automations = device_data.get("automations", [])
        for i, auto in enumerate(automations):
            if auto.get("id") == automation_id:
                deleted_name = auto.get("name", "Unnamed")
                del file_data[dev_id]["automations"][i]
                found = True
                break
        if found:
            break

    if not found:
        return {"success": False, "error": "Automation not found"}

    _write_json_file(AUTOMATIONS_FILE, file_data)
    return {"success": True, "message": f"Deleted automation: {deleted_name}"}


def toggle_automation_enabled(automation_id: str) -> Dict:
    """Toggle the enabled status of an automation."""
    file_data = _read_json_file(AUTOMATIONS_FILE) or {}

    found = False
    new_status = None
    for dev_id, device_data in file_data.items():
        automations = device_data.get("automations", [])
        for auto in automations:
            if auto.get("id") == automation_id:
                auto["enabled"] = not auto.get("enabled", False)
                auto["pending_sync"] = True
                auto["synced_at"] = None
                new_status = auto["enabled"]
                found = True
                break
        if found:
            break

    if not found:
        return {"success": False, "error": "Automation not found"}

    _write_json_file(AUTOMATIONS_FILE, file_data)
    return {"success": True, "enabled": new_status}


def get_full_automation(automation_id: str) -> Optional[Dict]:
    """Get full automation details including all fields."""
    file_data = _read_json_file(AUTOMATIONS_FILE) or {}

    for dev_id, device_data in file_data.items():
        automations = device_data.get("automations", [])
        for auto in automations:
            if auto.get("id") == automation_id:
                auto["device_id"] = dev_id
                auto["last_run_formatted"] = (
                    _time_ago(auto.get("last_run_at", 0))
                    if auto.get("last_run_at")
                    else "Never"
                )
                return auto
    return None


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
            agents.append(
                {
                    "id": agent.get("id", ""),
                    "name": agent.get("name", "Unnamed"),
                    "status": agent.get("status", "offline"),
                    "specialty": agent.get("specialty", ""),
                    "current_task": agent.get("current_task"),
                    "tasks_completed": agent.get("tasks_completed", 0),
                    "device_id": agent_device_id,
                }
            )

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
        "total_tasks_completed": sum(a.get("tasks_completed", 0) for a in agents),
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
            "reset_date": None,
        }

    return {
        "messages_used": data.get("messages_used", 0),
        "messages_limit": data.get("messages_limit", 100),
        "tier": data.get("tier", "free"),
        "reset_date": data.get("reset_date"),
        "usage_percent": min(
            100,
            (data.get("messages_used", 0) / max(1, data.get("messages_limit", 100)))
            * 100,
        ),
    }


def get_backend_usage() -> List[Dict]:
    """Get backend usage breakdown."""
    data = _read_json_file(ANALYTICS_FILE)
    if not data:
        return []

    backends = data.get("backends", {})
    total = sum(backends.values()) or 1

    return [
        {"name": name, "count": count, "percent": (count / total) * 100}
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
            s.connect(("127.0.0.1", port))
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
    except (socket.gaierror, socket.herror, OSError):
        local_ip = "127.0.0.1"

    # Check for debug mode and thread health
    debug_mode = False
    thread_health = {}
    try:
        # Import at function level to avoid circular import
        import shadow_bridge_gui

        debug_mode = getattr(shadow_bridge_gui, "DEBUG_BUILD", False)

        # Check thread health if app instance exists
        app = getattr(shadow_bridge_gui, "_app_instance", None)
        if app:
            # DataReceiver health
            if hasattr(app, "data_receiver") and app.data_receiver:
                thread_health["data_receiver"] = {
                    "alive": app.data_receiver.is_alive(),
                    "bind_failed": getattr(app.data_receiver, "bind_failed", False),
                    "bind_error": getattr(app.data_receiver, "bind_error", None),
                    "port": 19284
                }

            # CompanionRelay health
            if hasattr(app, "companion_relay") and app.companion_relay:
                thread_health["companion_relay"] = {
                    "alive": app.companion_relay.is_alive(),
                    "bind_failed": getattr(app.companion_relay, "bind_failed", False),
                    "bind_error": getattr(app.companion_relay, "bind_error", None),
                    "port": 19286
                }

            # DiscoveryServer health
            if hasattr(app, "discovery_server") and app.discovery_server:
                thread_health["discovery_server"] = {
                    "alive": app.discovery_server.is_alive(),
                    "bind_failed": getattr(app.discovery_server, "bind_failed", False),
                    "bind_error": getattr(app.discovery_server, "bind_error", None),
                    "port": 19283
                }
    except (ImportError, AttributeError):
        pass

    result = {
        "shadowbridge_running": shadowbridge_running,
        "devices_connected": len(online_devices),
        "total_devices": len(devices),
        "total_projects": len(projects),
        "total_notes": len(notes),
        "ssh_status": ssh_status,
        "version": "1.062",
        "local_ip": local_ip,
        "data_path": str(SHADOWAI_DIR),
        "debug_mode": debug_mode,
    }

    # Add thread health if available
    if thread_health:
        result["threads"] = thread_health
        # Calculate overall health
        all_healthy = all(
            t.get("alive") and not t.get("bind_failed")
            for t in thread_health.values()
        )
        result["all_threads_healthy"] = all_healthy

    return result


# ============ Data Files ============

TEAMS_FILE = SHADOWAI_DIR / "teams.json"
TASKS_FILE = SHADOWAI_DIR / "tasks.json"
TODOS_FILE = SHADOWAI_DIR / "todos.json"
WORKFLOWS_FILE = SHADOWAI_DIR / "workflows.json"
AUDITS_FILE = SHADOWAI_DIR / "audits.json"
FAVORITES_FILE = SHADOWAI_DIR / "web_favorites.json"


def _write_json_file(filepath: Path, data: Dict) -> bool:
    """Write data to a JSON file atomically with thread safety.

    Uses atomic write (temp file + replace) to prevent corruption from
    concurrent writes or crashes mid-write.
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock:
            # Write to temp file first, then atomically replace
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix=filepath.stem + "_", dir=filepath.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                # Atomic replace (works on Windows and POSIX)
                os.replace(tmp_path, filepath)
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
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


def create_team(
    data: Dict, owner_email: Optional[str] = None, owner_device: Optional[str] = None
) -> Dict:
    """Create a new team with email-based membership.

    Args:
        data: Team data (name, description)
        owner_email: Email of the team owner (optional but recommended)
        owner_device: Device ID of the team owner

    Returns:
        {"success": True, "team": {...}} or {"error": "..."}
    """
    file_data = _read_json_file(TEAMS_FILE) or {"teams": [], "invitations": []}

    team_id = _generate_id()
    timestamp = int(datetime.now().timestamp() * 1000)

    team = {
        "id": team_id,
        "name": data.get("name", "New Team"),
        "description": data.get("description", ""),
        "agents": [],
        "status": "ACTIVE",
        "created_at": timestamp,
        # Email-based membership
        "owner_email": owner_email,
        "owner_device": owner_device,
        "members": [
            {
                "email": owner_email,
                "device_id": owner_device,
                "role": "owner",  # owner, admin, member
                "joined_at": timestamp,
            }
        ]
        if owner_email or owner_device
        else [],
    }

    if "teams" not in file_data:
        file_data["teams"] = []
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
        "completed_tasks": len([t for t in tasks if t.get("status") == "COMPLETED"]),
    }


# ============ Team Membership ============

TEAM_INVITATION_EXPIRY_SECONDS = 7 * 24 * 60 * 60  # 7 days


def invite_team_member(
    team_id: str, inviter_email: str, invitee_email: str, role: str = "member"
) -> Dict:
    """Invite a user to a team by email.

    Args:
        team_id: Team to invite to
        inviter_email: Email of the person inviting (must be owner/admin)
        invitee_email: Email to invite
        role: Role to assign (member, admin)

    Returns:
        {"success": True, "invitation_code": "..."} or {"error": "..."}
    """
    if role not in ("member", "admin"):
        return {"error": "Invalid role. Use 'member' or 'admin'."}

    invitee_email = invitee_email.lower().strip()
    inviter_email = inviter_email.lower().strip()

    file_data = _read_json_file(TEAMS_FILE) or {"teams": [], "invitations": []}

    # Find team and check permissions
    team = None
    for t in file_data.get("teams", []):
        if t.get("id") == team_id:
            team = t
            break

    if not team:
        return {"error": "Team not found"}

    # Check if inviter has permission
    inviter_role = None
    for member in team.get("members", []):
        if member.get("email") == inviter_email:
            inviter_role = member.get("role")
            break

    if inviter_role not in ("owner", "admin"):
        return {"error": "Only owners and admins can invite members"}

    # Check if already a member
    for member in team.get("members", []):
        if member.get("email") == invitee_email:
            return {"error": "User is already a team member"}

    # Generate invitation code
    invitation_code = _generate_id()[:8].upper()
    expires_at = datetime.now().timestamp() + TEAM_INVITATION_EXPIRY_SECONDS

    if "invitations" not in file_data:
        file_data["invitations"] = []

    # Remove any existing invitation for this email/team
    file_data["invitations"] = [
        inv
        for inv in file_data["invitations"]
        if not (inv.get("team_id") == team_id and inv.get("email") == invitee_email)
    ]

    # Add new invitation
    file_data["invitations"].append(
        {
            "code": invitation_code,
            "team_id": team_id,
            "team_name": team.get("name"),
            "email": invitee_email,
            "role": role,
            "inviter_email": inviter_email,
            "created_at": datetime.now().timestamp(),
            "expires_at": expires_at,
        }
    )

    _write_json_file(TEAMS_FILE, file_data)

    # Try to send invitation email
    _send_team_invitation_email(
        invitee_email, team.get("name"), inviter_email, invitation_code
    )

    return {
        "success": True,
        "invitation_code": invitation_code,
        "message": f"Invitation sent to {invitee_email}",
    }


def _send_team_invitation_email(
    to_email: str, team_name: str, inviter_email: str, code: str
) -> bool:
    """Send team invitation email."""
    config = _get_email_config()
    if not config:
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = config.get("from_email", config.get("smtp_user"))
        msg["To"] = to_email
        msg["Subject"] = f"ShadowAI - Team Invitation: {team_name}"

        body = f"""
You've been invited to join a team on ShadowAI!

Team: {team_name}
Invited by: {inviter_email}

Your invitation code: {code}

To accept, enter this code in the ShadowAI app or web dashboard.
This invitation expires in 7 days.

If you didn't expect this invitation, you can ignore this email.

- ShadowAI Team
"""
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(config["smtp_host"], config["smtp_port"])
        server.starttls()
        server.login(config["smtp_user"], config["smtp_password"])
        server.send_message(msg)
        server.quit()

        return True
    except Exception as e:
        print(f"Failed to send team invitation email: {e}")
        return False


def accept_team_invitation(
    invitation_code: str, accepter_email: str, accepter_device: Optional[str] = None
) -> Dict:
    """Accept a team invitation.

    Args:
        invitation_code: The invitation code
        accepter_email: Email of the user accepting
        accepter_device: Device ID of the accepter (optional)

    Returns:
        {"success": True, "team": {...}} or {"error": "..."}
    """
    accepter_email = accepter_email.lower().strip()
    invitation_code = invitation_code.upper().strip()

    file_data = _read_json_file(TEAMS_FILE) or {"teams": [], "invitations": []}

    # Find invitation
    invitation = None
    for inv in file_data.get("invitations", []):
        if inv.get("code") == invitation_code:
            invitation = inv
            break

    if not invitation:
        return {"error": "Invalid invitation code"}

    # Check expiry
    if datetime.now().timestamp() > invitation.get("expires_at", 0):
        return {"error": "Invitation has expired"}

    # Check email matches (or any verified email for the device)
    if invitation.get("email") != accepter_email:
        return {"error": "This invitation was sent to a different email address"}

    # Find team
    team = None
    team_index = None
    for i, t in enumerate(file_data.get("teams", [])):
        if t.get("id") == invitation.get("team_id"):
            team = t
            team_index = i
            break

    if not team:
        return {"error": "Team no longer exists"}

    # Add member to team
    if "members" not in team:
        team["members"] = []

    team["members"].append(
        {
            "email": accepter_email,
            "device_id": accepter_device,
            "role": invitation.get("role", "member"),
            "joined_at": int(datetime.now().timestamp() * 1000),
        }
    )

    file_data["teams"][team_index] = team

    # Remove the invitation
    file_data["invitations"] = [
        inv for inv in file_data["invitations"] if inv.get("code") != invitation_code
    ]

    _write_json_file(TEAMS_FILE, file_data)

    return {"success": True, "team": team}


def get_pending_invitations(email: str) -> List[Dict]:
    """Get pending team invitations for an email."""
    email = email.lower().strip()
    file_data = _read_json_file(TEAMS_FILE) or {"teams": [], "invitations": []}

    now = datetime.now().timestamp()
    invitations = []

    for inv in file_data.get("invitations", []):
        if inv.get("email") == email and inv.get("expires_at", 0) > now:
            invitations.append(
                {
                    "code": inv.get("code"),
                    "team_id": inv.get("team_id"),
                    "team_name": inv.get("team_name"),
                    "role": inv.get("role"),
                    "inviter_email": inv.get("inviter_email"),
                    "expires_at": inv.get("expires_at"),
                }
            )

    return invitations


def remove_team_member(team_id: str, remover_email: str, member_email: str) -> Dict:
    """Remove a member from a team.

    Args:
        team_id: Team ID
        remover_email: Email of person removing (must be owner/admin)
        member_email: Email of member to remove

    Returns:
        {"success": True} or {"error": "..."}
    """
    remover_email = remover_email.lower().strip()
    member_email = member_email.lower().strip()

    file_data = _read_json_file(TEAMS_FILE) or {"teams": [], "invitations": []}

    # Find team
    team = None
    team_index = None
    for i, t in enumerate(file_data.get("teams", [])):
        if t.get("id") == team_id:
            team = t
            team_index = i
            break

    if not team:
        return {"error": "Team not found"}

    # Check permissions
    remover_role = None
    member_role = None
    for member in team.get("members", []):
        if member.get("email") == remover_email:
            remover_role = member.get("role")
        if member.get("email") == member_email:
            member_role = member.get("role")

    if remover_role not in ("owner", "admin"):
        return {"error": "Only owners and admins can remove members"}

    if member_role == "owner":
        return {"error": "Cannot remove the team owner"}

    if member_role == "admin" and remover_role != "owner":
        return {"error": "Only owners can remove admins"}

    # Remove member
    team["members"] = [
        m for m in team.get("members", []) if m.get("email") != member_email
    ]
    file_data["teams"][team_index] = team
    _write_json_file(TEAMS_FILE, file_data)

    return {"success": True}


def get_teams_for_email(email: str) -> List[Dict]:
    """Get all teams a user is a member of."""
    email = email.lower().strip()
    teams = get_teams()

    user_teams = []
    for team in teams:
        for member in team.get("members", []):
            if member.get("email") == email:
                user_teams.append({**team, "my_role": member.get("role")})
                break

    return user_teams


def update_team_member_role(
    team_id: str, updater_email: str, member_email: str, new_role: str
) -> Dict:
    """Update a team member's role.

    Args:
        team_id: Team ID
        updater_email: Email of person updating (must be owner)
        member_email: Email of member to update
        new_role: New role (admin, member)

    Returns:
        {"success": True} or {"error": "..."}
    """
    if new_role not in ("admin", "member"):
        return {"error": "Invalid role. Use 'admin' or 'member'."}

    updater_email = updater_email.lower().strip()
    member_email = member_email.lower().strip()

    file_data = _read_json_file(TEAMS_FILE) or {"teams": [], "invitations": []}

    # Find team
    team = None
    team_index = None
    for i, t in enumerate(file_data.get("teams", [])):
        if t.get("id") == team_id:
            team = t
            team_index = i
            break

    if not team:
        return {"error": "Team not found"}

    # Check permissions - only owner can change roles
    updater_role = None
    for member in team.get("members", []):
        if member.get("email") == updater_email:
            updater_role = member.get("role")
            break

    if updater_role != "owner":
        return {"error": "Only the team owner can change roles"}

    # Update member role
    found = False
    for member in team.get("members", []):
        if member.get("email") == member_email:
            if member.get("role") == "owner":
                return {"error": "Cannot change the owner's role"}
            member["role"] = new_role
            found = True
            break

    if not found:
        return {"error": "Member not found in team"}

    file_data["teams"][team_index] = team
    _write_json_file(TEAMS_FILE, file_data)

    return {"success": True}


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
        "created_at": int(datetime.now().timestamp() * 1000),
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
        "created_at": int(datetime.now().timestamp() * 1000),
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
    
    workflow_id = _generate_id()
    workflow = {
        "id": workflow_id,
        "type": workflow_type,
        "name": f"{workflow_type.replace('_', ' ').title()} Workflow",
        "status": "RUNNING",
        "current_stage": "Initializing Agents",
        "progress": 5,
        "started_at": int(datetime.now().timestamp() * 1000),
    }
    
    # If it's an agent-based workflow, trigger the background process
    if workflow_type == "AGENT_STAFF":
        task = options.get("task", "AGI Project")
        team = options.get("team", "dev")
        # Trigger background shadow_agent execution
        import subprocess
        try:
            # We run this detached to not block the API
            cmd = f"python -m shadow_agent --team {team} --task \"{task}\""
            subprocess.Popen(cmd, shell=True)
            workflow["current_stage"] = f"Team '{team}' working on task"
        except Exception as e:
            logger.error(f"Failed to start shadow_agent: {e}")
            workflow["status"] = "FAILED"
            workflow["error"] = str(e)

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
    periods = {"1D": 1, "7D": 7, "30D": 30, "90D": 90, "ANNUAL": 365, "ALL_TIME": 3650}
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
    security_events = len(
        [e for e in entries if e.get("severity") in ["SECURITY", "CRITICAL"]]
    )

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
        "risk_level": risk_level,
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
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return {"success": True, "path": str(filepath)}

    elif format == "csv":
        filepath = export_dir / f"audit_report_{timestamp}.csv"
        import csv

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["ID", "Timestamp", "Category", "Severity", "Summary", "Action"]
            )
            for entry in entries:
                writer.writerow(
                    [
                        entry.get("id", ""),
                        _format_timestamp(entry.get("timestamp", 0)),
                        entry.get("category", ""),
                        entry.get("severity", ""),
                        entry.get("summary", ""),
                        entry.get("action", ""),
                    ]
                )
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


def delete_project(project_id: str) -> Dict:
    """Delete a project."""
    file_data = _read_json_file(PROJECTS_FILE)
    if not file_data:
        return {"error": "Projects file not found"}

    for device_id, device_info in file_data.get("devices", {}).items():
        projects = device_info.get("projects", [])
        for i, project in enumerate(projects):
            if project.get("id") == project_id or project.get("path") == project_id:
                deleted_project = projects.pop(i)
                _write_json_file(PROJECTS_FILE, file_data)
                return {"success": True, "deleted": deleted_project}

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
                results.append(
                    {
                        "type": "project",
                        "id": project.get("id"),
                        "title": name,
                        "preview": path,
                    }
                )

    if "notes" in types:
        for note in get_notes():
            title = note.get("title", "")
            if query_lower in title.lower():
                results.append(
                    {
                        "type": "note",
                        "id": note.get("id"),
                        "title": title,
                        "preview": note.get("time_ago", ""),
                    }
                )

    if "automations" in types:
        for auto in get_automations():
            name = auto.get("name", "")
            desc = auto.get("description", "")
            if query_lower in name.lower() or query_lower in desc.lower():
                results.append(
                    {
                        "type": "automation",
                        "id": auto.get("id"),
                        "title": name,
                        "preview": desc[:50] if desc else "",
                    }
                )

    if "agents" in types:
        for agent in get_agents():
            name = agent.get("name", "")
            specialty = agent.get("specialty", "")
            if query_lower in name.lower() or query_lower in specialty.lower():
                results.append(
                    {
                        "type": "agent",
                        "id": agent.get("id"),
                        "title": name,
                        "preview": specialty,
                    }
                )

    return {"items": results[:20]}  # Limit to 20 results


# ============ Cards ============


def get_cards(device_id: Optional[str] = None) -> List[Dict]:
    """Get all cards, optionally filtered by device.

    Cards are deduplicated by ID across devices.
    """
    data = _read_json_file(CARDS_FILE)
    if not data:
        return []

    devices_data = data.get("devices", {})
    cards_by_id = {}

    for card_device_id, device_info in devices_data.items():
        if device_id and card_device_id != device_id:
            continue

        synced_at = device_info.get("synced_at", 0)

        for card in device_info.get("cards", []):
            card_id = card.get("id")
            if not card_id:
                continue

            updated_at = card.get("updated_at", 0)

            if card_id in cards_by_id:
                existing = cards_by_id[card_id]
                if updated_at <= existing.get("updated_at", 0):
                    continue

            cards_by_id[card_id] = {
                "id": card_id,
                "title": card.get("title", ""),
                "description": card.get("description"),
                "type": card.get("type", "OTHER"),
                "url": card.get("url"),
                "item_id": card.get("item_id"),
                "item_type": card.get("item_type"),
                "thumbnail_url": card.get("thumbnail_url"),
                "collection_id": card.get("collection_id"),
                "is_favorite": card.get("is_favorite", False),
                "device_id": card_device_id,
                "device_name": device_info.get("name", card_device_id),
                "updated_at": updated_at,
                "updated_at_formatted": _format_timestamp(updated_at),
                "time_ago": _time_ago(updated_at),
            }

    cards = list(cards_by_id.values())
    cards.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return cards


def get_card(card_id: str) -> Optional[Dict]:
    """Get a specific card by ID."""
    cards = get_cards()
    for card in cards:
        if card["id"] == card_id:
            return card
    return None


# ============ Collections ============


def get_collections(device_id: Optional[str] = None) -> List[Dict]:
    """Get all collections, optionally filtered by device.

    Collections are deduplicated by ID across devices.
    """
    data = _read_json_file(COLLECTIONS_FILE)
    if not data:
        return []

    devices_data = data.get("devices", {})
    collections_by_id = {}

    for coll_device_id, device_info in devices_data.items():
        if device_id and coll_device_id != device_id:
            continue

        synced_at = device_info.get("synced_at", 0)

        for collection in device_info.get("collections", []):
            collection_id = collection.get("id")
            if not collection_id:
                continue

            updated_at = collection.get("updated_at", 0)

            if collection_id in collections_by_id:
                existing = collections_by_id[collection_id]
                if updated_at <= existing.get("updated_at", 0):
                    continue

            collections_by_id[collection_id] = {
                "id": collection_id,
                "name": collection.get("name", ""),
                "description": collection.get("description"),
                "color": collection.get("color", "#6200EA"),
                "cover_card_id": collection.get("cover_card_id"),
                "is_website_collection": collection.get("is_website_collection", False),
                "card_count": collection.get("card_count", 0),
                "device_id": coll_device_id,
                "device_name": device_info.get("name", coll_device_id),
                "updated_at": updated_at,
                "updated_at_formatted": _format_timestamp(updated_at),
                "time_ago": _time_ago(updated_at),
            }

    collections = list(collections_by_id.values())
    collections.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return collections


def get_collection(collection_id: str) -> Optional[Dict]:
    """Get a specific collection by ID."""
    collections = get_collections()
    for collection in collections:
        if collection["id"] == collection_id:
            return collection
    return None


def get_collection_cards(
    collection_id: str, device_id: Optional[str] = None
) -> List[Dict]:
    """Get all cards in a collection."""
    all_cards = get_cards(device_id)
    return [card for card in all_cards if card.get("collection_id") == collection_id]


# ============ Enhanced Analytics ============


def get_privacy_score() -> Dict:
    """Get privacy score and compliance data."""
    stats = get_audit_stats("30D")
    return {
        "privacy_score": stats.get("privacy_score", 100),
        "risk_level": stats.get("risk_level", "MINIMAL"),
        "pii_access_count": 0,  # TODO: Track from audit traces
        "encrypted_access_count": 0,
        "compliance_status": "COMPLIANT",
    }


def get_token_usage() -> Dict:
    """Get token usage statistics."""
    data = _read_json_file(ANALYTICS_FILE)
    if not data:
        return {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "avg_per_query": 0,
        }
    return {
        "total_tokens": data.get("total_tokens", 0),
        "input_tokens": data.get("input_tokens", 0),
        "output_tokens": data.get("output_tokens", 0),
        "avg_per_query": data.get("avg_tokens_per_query", 0),
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
            for cat, count in sorted(
                categories.items(), key=lambda x: x[1], reverse=True
            )
        ]
    }


# ============ Create Functions (Web-originated) ============


def create_project(data: Dict, device_id: str = "web") -> Dict:
    """Create a new project from web dashboard.

    Projects are stored under the specified device_id with pending_sync flag
    for bi-directional sync back to Android.

    Ownership: The creating device becomes the owner. shared_with list controls
    which other devices can access and at what permission level.
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
        "synced_at": None,
        # Ownership tracking
        "owner_device": device_id,
        "shared_with": [],  # List of {"device_id": str, "permission": "view"|"edit"|"full"}
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

    Ownership: The creating device becomes the owner. shared_with list controls
    which other devices can access and at what permission level.
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
        "synced_at": None,
        # Ownership tracking
        "owner_device": device_id,
        "shared_with": [],  # List of {"device_id": str, "permission": "view"|"edit"|"full"}
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
        "synced_at": None,
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
        "automations": [],
        "sessions": [],
        "cards": [],
        "collections": [],
    }

    # Check projects
    projects_data = _read_json_file(PROJECTS_FILE)
    if projects_data:
        device_projects = (
            projects_data.get("devices", {}).get(device_id, {}).get("projects", [])
        )
        pending["projects"] = [p for p in device_projects if p.get("pending_sync")]

    # Check notes
    notes_data = _read_json_file(NOTES_FILE)
    if notes_data:
        device_notes = notes_data.get("devices", {}).get(device_id, {}).get("notes", [])
        pending["notes"] = [n for n in device_notes if n.get("pending_sync")]

    # Check automations
    automations_data = _read_json_file(AUTOMATIONS_FILE)
    if automations_data:
        device_autos = (
            automations_data.get("devices", {})
            .get(device_id, {})
            .get("automations", [])
        )
        pending["automations"] = [a for a in device_autos if a.get("pending_sync")]

    # Check sessions
    sessions_data = _load_sessions_file()
    if sessions_data:
        sessions_map = sessions_data.get("sessions", {})
        pending["sessions"] = [
            s
            for s in sessions_map.values()
            if isinstance(s, dict)
            and s.get("device_id") == device_id
            and s.get("pending_sync")
        ]

    # Check cards
    cards_data = _read_json_file(CARDS_FILE)
    if cards_data:
        device_cards = cards_data.get("devices", {}).get(device_id, {}).get("cards", [])
        pending["cards"] = [c for c in device_cards if c.get("pending_sync")]

    # Check collections
    collections_data = _read_json_file(COLLECTIONS_FILE)
    if collections_data:
        device_collections = (
            collections_data.get("devices", {})
            .get(device_id, {})
            .get("collections", [])
        )
        pending["collections"] = [
            c for c in device_collections if c.get("pending_sync")
        ]

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

    elif item_type == "sessions":
        file_data = _load_sessions_file()
        sessions = file_data.get("sessions", {})
        if isinstance(sessions, dict):
            for session_id, session in sessions.items():
                if session_id in item_ids and isinstance(session, dict):
                    session["pending_sync"] = False
                    session["synced_at"] = timestamp
            file_data["sessions"] = sessions
            _write_json_file(SESSIONS_FILE, file_data)

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
        "pending_sync": True,
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


# ============ Ownership & Sharing ============


def get_permission_level(item: Dict, requesting_device: str) -> Optional[str]:
    """Get the permission level for a device on a content item.

    Permission levels:
    - 'owner': Full control including deletion and sharing management
    - 'full': Everything except removing owner or changing ownership
    - 'edit': Can view and modify, but cannot delete
    - 'view': Read-only access
    - None: No access

    Args:
        item: The project/note/automation dict with owner_device and shared_with
        requesting_device: The device_id requesting access

    Returns:
        Permission level string or None if no access
    """
    owner_device = item.get("owner_device")

    # Owner has full control
    if owner_device == requesting_device:
        return "owner"

    # Check shared_with list
    shared_with = item.get("shared_with", [])
    for share in shared_with:
        if share.get("device_id") == requesting_device:
            return share.get("permission", "view")

    # Legacy items without owner_device - treat original device as owner
    # This handles backward compatibility with pre-ownership data
    if owner_device is None:
        # If the item is under this device's storage, they're the implicit owner
        return None  # No access by default for non-owners on legacy items

    return None


def can_view(item: Dict, requesting_device: str) -> bool:
    """Check if device can view this item."""
    permission = get_permission_level(item, requesting_device)
    return permission in ("owner", "full", "edit", "view")


def can_edit(item: Dict, requesting_device: str) -> bool:
    """Check if device can edit this item."""
    permission = get_permission_level(item, requesting_device)
    return permission in ("owner", "full", "edit")


def can_delete(item: Dict, requesting_device: str) -> bool:
    """Check if device can delete this item.

    Only owner and 'full' permission can delete.
    """
    permission = get_permission_level(item, requesting_device)
    return permission in ("owner", "full")


def can_share(item: Dict, requesting_device: str) -> bool:
    """Check if device can share this item with others.

    Only owner can manage sharing.
    """
    permission = get_permission_level(item, requesting_device)
    return permission == "owner"


def share_note(
    note_id: str, owner_device: str, target_device: str, permission: str
) -> Dict:
    """Share a note with another device.

    Args:
        note_id: The note to share
        owner_device: Device requesting to share (must be owner)
        target_device: Device to share with
        permission: Permission level ('view', 'edit', 'full')

    Returns:
        {"success": True} or {"error": "..."}
    """
    if permission not in ("view", "edit", "full"):
        return {"error": f"Invalid permission level: {permission}"}

    data = _read_json_file(NOTES_FILE)
    if not data:
        return {"error": "Notes file not found"}

    devices_data = data.get("devices", {})

    for device_id, device_info in devices_data.items():
        notes = device_info.get("notes", [])
        for i, note in enumerate(notes):
            if note.get("id") == note_id:
                # Check ownership
                if not can_share(note, owner_device):
                    return {"error": "Only the owner can share this note"}

                # Initialize shared_with if needed
                if "shared_with" not in note:
                    note["shared_with"] = []

                # Update or add share entry
                shared_with = note.get("shared_with", [])
                found = False
                for share in shared_with:
                    if share.get("device_id") == target_device:
                        share["permission"] = permission
                        found = True
                        break

                if not found:
                    shared_with.append(
                        {"device_id": target_device, "permission": permission}
                    )

                note["shared_with"] = shared_with
                note["updatedAt"] = int(datetime.now().timestamp() * 1000)
                devices_data[device_id]["notes"][i] = note
                _write_json_file(NOTES_FILE, data)
                return {"success": True, "shared_with": shared_with}

    return {"error": "Note not found"}


def unshare_note(note_id: str, owner_device: str, target_device: str) -> Dict:
    """Remove sharing for a note.

    Args:
        note_id: The note to unshare
        owner_device: Device requesting to unshare (must be owner)
        target_device: Device to remove access from

    Returns:
        {"success": True} or {"error": "..."}
    """
    data = _read_json_file(NOTES_FILE)
    if not data:
        return {"error": "Notes file not found"}

    devices_data = data.get("devices", {})

    for device_id, device_info in devices_data.items():
        notes = device_info.get("notes", [])
        for i, note in enumerate(notes):
            if note.get("id") == note_id:
                # Check ownership
                if not can_share(note, owner_device):
                    return {"error": "Only the owner can manage sharing"}

                # Remove from shared_with
                shared_with = note.get("shared_with", [])
                note["shared_with"] = [
                    s for s in shared_with if s.get("device_id") != target_device
                ]
                note["updatedAt"] = int(datetime.now().timestamp() * 1000)
                devices_data[device_id]["notes"][i] = note
                _write_json_file(NOTES_FILE, data)
                return {"success": True}

    return {"error": "Note not found"}


def share_project(
    project_id: str, owner_device: str, target_device: str, permission: str
) -> Dict:
    """Share a project with another device.

    Args:
        project_id: The project to share
        owner_device: Device requesting to share (must be owner)
        target_device: Device to share with
        permission: Permission level ('view', 'edit', 'full')

    Returns:
        {"success": True} or {"error": "..."}
    """
    if permission not in ("view", "edit", "full"):
        return {"error": f"Invalid permission level: {permission}"}

    data = _read_json_file(PROJECTS_FILE)
    if not data:
        return {"error": "Projects file not found"}

    devices_data = data.get("devices", {})

    for device_id, device_info in devices_data.items():
        projects = device_info.get("projects", [])
        for i, project in enumerate(projects):
            if project.get("id") == project_id or project.get("path") == project_id:
                # Check ownership
                if not can_share(project, owner_device):
                    return {"error": "Only the owner can share this project"}

                # Initialize shared_with if needed
                if "shared_with" not in project:
                    project["shared_with"] = []

                # Update or add share entry
                shared_with = project.get("shared_with", [])
                found = False
                for share in shared_with:
                    if share.get("device_id") == target_device:
                        share["permission"] = permission
                        found = True
                        break

                if not found:
                    shared_with.append(
                        {"device_id": target_device, "permission": permission}
                    )

                project["shared_with"] = shared_with
                project["updated_at"] = int(datetime.now().timestamp() * 1000)
                devices_data[device_id]["projects"][i] = project
                _write_json_file(PROJECTS_FILE, data)
                return {"success": True, "shared_with": shared_with}

    return {"error": "Project not found"}


def unshare_project(project_id: str, owner_device: str, target_device: str) -> Dict:
    """Remove sharing for a project."""
    data = _read_json_file(PROJECTS_FILE)
    if not data:
        return {"error": "Projects file not found"}

    devices_data = data.get("devices", {})

    for device_id, device_info in devices_data.items():
        projects = device_info.get("projects", [])
        for i, project in enumerate(projects):
            if project.get("id") == project_id or project.get("path") == project_id:
                # Check ownership
                if not can_share(project, owner_device):
                    return {"error": "Only the owner can manage sharing"}

                # Remove from shared_with
                shared_with = project.get("shared_with", [])
                project["shared_with"] = [
                    s for s in shared_with if s.get("device_id") != target_device
                ]
                project["updated_at"] = int(datetime.now().timestamp() * 1000)
                devices_data[device_id]["projects"][i] = project
                _write_json_file(PROJECTS_FILE, data)
                return {"success": True}

    return {"error": "Project not found"}


def get_shared_content_for_device(device_id: str) -> Dict:
    """Get all content shared with a specific device.

    Returns projects and notes that have been shared with this device
    (where device_id appears in shared_with but is not the owner).
    """
    shared = {"projects": [], "notes": []}

    # Check projects
    projects_data = _read_json_file(PROJECTS_FILE) or {"devices": {}}
    for owner_device_id, device_info in projects_data.get("devices", {}).items():
        for project in device_info.get("projects", []):
            shared_with = project.get("shared_with", [])
            for share in shared_with:
                if share.get("device_id") == device_id:
                    shared["projects"].append(
                        {
                            **project,
                            "owner_device_name": device_info.get(
                                "name", owner_device_id
                            ),
                            "my_permission": share.get("permission", "view"),
                        }
                    )
                    break

    # Check notes
    notes_data = _read_json_file(NOTES_FILE) or {"devices": {}}
    for owner_device_id, device_info in notes_data.get("devices", {}).items():
        for note in device_info.get("notes", []):
            shared_with = note.get("shared_with", [])
            for share in shared_with:
                if share.get("device_id") == device_id:
                    shared["notes"].append(
                        {
                            **note,
                            "owner_device_name": device_info.get(
                                "name", owner_device_id
                            ),
                            "my_permission": share.get("permission", "view"),
                        }
                    )
                    break

    return shared


def delete_note_with_permission(note_id: str, requesting_device: str) -> Dict:
    """Delete a note with permission check.

    Only owner or devices with 'full' permission can delete.

    Args:
        note_id: The note to delete
        requesting_device: The device attempting to delete

    Returns:
        {"success": True} or {"error": "..."}
    """
    data = _read_json_file(NOTES_FILE)
    if not data:
        return {"error": "Notes file not found"}

    devices_data = data.get("devices", {})

    for device_id, device_info in devices_data.items():
        notes = device_info.get("notes", [])
        for note in notes:
            if note.get("id") == note_id:
                # Check delete permission
                if not can_delete(note, requesting_device):
                    permission = get_permission_level(note, requesting_device)
                    if permission:
                        return {
                            "error": f"Your permission level ({permission}) does not allow deletion"
                        }
                    else:
                        return {"error": "You don't have access to this note"}

                # Permission granted - delete the note
                device_info["notes"] = [n for n in notes if n.get("id") != note_id]
                _write_json_file(NOTES_FILE, data)
                return {"success": True}

    return {"error": "Note not found"}


def delete_project_with_permission(project_id: str, requesting_device: str) -> Dict:
    """Delete a project with permission check.

    Only owner or devices with 'full' permission can delete.
    """
    data = _read_json_file(PROJECTS_FILE)
    if not data:
        return {"error": "Projects file not found"}

    devices_data = data.get("devices", {})

    for device_id, device_info in devices_data.items():
        projects = device_info.get("projects", [])
        for project in projects:
            if project.get("id") == project_id or project.get("path") == project_id:
                # Check delete permission
                if not can_delete(project, requesting_device):
                    permission = get_permission_level(project, requesting_device)
                    if permission:
                        return {
                            "error": f"Your permission level ({permission}) does not allow deletion"
                        }
                    else:
                        return {"error": "You don't have access to this project"}

                # Permission granted - delete the project
                device_info["projects"] = [
                    p
                    for p in projects
                    if p.get("id") != project_id and p.get("path") != project_id
                ]
                _write_json_file(PROJECTS_FILE, data)
                return {"success": True}

    return {"error": "Project not found"}


# ============ Email Verification ============

import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Verification code settings
VERIFICATION_CODE_LENGTH = 6
VERIFICATION_CODE_EXPIRY_SECONDS = 600  # 10 minutes

# Email configuration file
EMAIL_CONFIG_FILE = SHADOWAI_DIR / "email_config.json"


def _generate_verification_code() -> str:
    """Generate a random 6-digit verification code."""
    return "".join(random.choices(string.digits, k=VERIFICATION_CODE_LENGTH))


def _get_email_config() -> Optional[Dict]:
    """Get SMTP email configuration."""
    return _read_json_file(EMAIL_CONFIG_FILE)


def set_email_config(
    smtp_host: str, smtp_port: int, smtp_user: str, smtp_password: str, from_email: str
) -> Dict:
    """Configure SMTP settings for sending verification emails."""
    config = {
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "smtp_user": smtp_user,
        "smtp_password": smtp_password,
        "from_email": from_email,
    }
    if _write_json_file(EMAIL_CONFIG_FILE, config):
        return {"success": True}
    return {"error": "Failed to save email configuration"}


def request_email_verification(email: str, device_id: str) -> Dict:
    """Request email verification - generates code and sends email.

    Args:
        email: Email address to verify
        device_id: Device requesting verification

    Returns:
        {"success": True, "message": "..."} or {"error": "..."}
    """
    if not email or "@" not in email:
        return {"error": "Invalid email address"}

    email = email.lower().strip()

    # Check if email is already verified for this device
    emails_data = _read_json_file(EMAILS_FILE) or {
        "verified_emails": {},
        "pending_verifications": {},
    }

    verified = emails_data.get("verified_emails", {})
    if device_id in verified and verified[device_id].get("email") == email:
        return {"error": "Email already verified for this device"}

    # Generate verification code
    code = _generate_verification_code()
    expires_at = datetime.now().timestamp() + VERIFICATION_CODE_EXPIRY_SECONDS

    # Store pending verification
    if "pending_verifications" not in emails_data:
        emails_data["pending_verifications"] = {}

    emails_data["pending_verifications"][email] = {
        "code": code,
        "device_id": device_id,
        "expires_at": expires_at,
        "created_at": datetime.now().timestamp(),
    }

    _write_json_file(EMAILS_FILE, emails_data)

    # Try to send email
    email_sent = _send_verification_email(email, code)

    if email_sent:
        return {
            "success": True,
            "message": f"Verification code sent to {email}",
            "expires_in_seconds": VERIFICATION_CODE_EXPIRY_SECONDS,
        }
    else:
        # Email not configured - return code for manual testing
        # In production, you'd want to fail here
        return {
            "success": True,
            "message": "Email service not configured. Code generated for testing.",
            "code": code,  # Only for development/testing
            "expires_in_seconds": VERIFICATION_CODE_EXPIRY_SECONDS,
        }


def _send_verification_email(to_email: str, code: str) -> bool:
    """Send verification email via SMTP.

    Returns True if email was sent, False if email is not configured.
    """
    config = _get_email_config()
    if not config:
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = config.get("from_email", config.get("smtp_user"))
        msg["To"] = to_email
        msg["Subject"] = "ShadowAI - Email Verification Code"

        body = f"""
Your ShadowAI verification code is:

    {code}

This code expires in 10 minutes.

If you didn't request this, you can ignore this email.

- ShadowAI Team
"""
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(config["smtp_host"], config["smtp_port"])
        server.starttls()
        server.login(config["smtp_user"], config["smtp_password"])
        server.send_message(msg)
        server.quit()

        return True
    except Exception as e:
        print(f"Failed to send verification email: {e}")
        return False


def verify_email_code(email: str, code: str, device_id: str) -> Dict:
    """Verify the email code and mark email as verified.

    Args:
        email: Email address being verified
        code: Verification code entered by user
        device_id: Device requesting verification

    Returns:
        {"success": True} or {"error": "..."}
    """
    email = email.lower().strip()

    emails_data = _read_json_file(EMAILS_FILE)
    if not emails_data:
        return {"error": "No pending verifications"}

    pending = emails_data.get("pending_verifications", {}).get(email)
    if not pending:
        return {"error": "No pending verification for this email"}

    # Check expiry
    if datetime.now().timestamp() > pending.get("expires_at", 0):
        # Clean up expired verification
        del emails_data["pending_verifications"][email]
        _write_json_file(EMAILS_FILE, emails_data)
        return {"error": "Verification code expired. Please request a new one."}

    # Check code
    if pending.get("code") != code:
        return {"error": "Invalid verification code"}

    # Check device
    if pending.get("device_id") != device_id:
        return {"error": "Verification was requested from a different device"}

    # Get device name
    device = get_device(device_id)
    device_name = device.get("name", device_id) if device else device_id

    # Mark as verified
    if "verified_emails" not in emails_data:
        emails_data["verified_emails"] = {}

    emails_data["verified_emails"][device_id] = {
        "email": email,
        "verified_at": datetime.now().timestamp(),
        "device_name": device_name,
    }

    # Clean up pending verification
    del emails_data["pending_verifications"][email]

    _write_json_file(EMAILS_FILE, emails_data)

    return {"success": True, "email": email}


def get_verified_email(device_id: str) -> Optional[str]:
    """Get the verified email for a device."""
    emails_data = _read_json_file(EMAILS_FILE)
    if not emails_data:
        return None

    verified = emails_data.get("verified_emails", {}).get(device_id)
    if verified:
        return verified.get("email")
    return None


def get_all_verified_emails() -> List[Dict]:
    """Get all verified emails with device info."""
    emails_data = _read_json_file(EMAILS_FILE)
    if not emails_data:
        return []

    verified = emails_data.get("verified_emails", {})
    result = []

    for device_id, info in verified.items():
        result.append(
            {
                "device_id": device_id,
                "email": info.get("email"),
                "verified_at": info.get("verified_at"),
                "verified_at_formatted": _format_timestamp(
                    int(info.get("verified_at", 0) * 1000)
                ),
                "device_name": info.get("device_name", device_id),
            }
        )

    return result


def remove_verified_email(device_id: str) -> Dict:
    """Remove verified email for a device."""
    emails_data = _read_json_file(EMAILS_FILE)
    if not emails_data:
        return {"error": "No verified emails"}

    verified = emails_data.get("verified_emails", {})
    if device_id not in verified:
        return {"error": "No verified email for this device"}

    del verified[device_id]
    emails_data["verified_emails"] = verified
    _write_json_file(EMAILS_FILE, emails_data)

    return {"success": True}


def get_devices_by_email(email: str) -> List[Dict]:
    """Get all devices that have verified a specific email.

    Useful for finding all devices belonging to a user.
    """
    email = email.lower().strip()
    emails_data = _read_json_file(EMAILS_FILE)
    if not emails_data:
        return []

    verified = emails_data.get("verified_emails", {})
    devices = []

    for device_id, info in verified.items():
        if info.get("email") == email:
            device = get_device(device_id)
            devices.append(
                {
                    "device_id": device_id,
                    "device_name": info.get("device_name", device_id),
                    "verified_at": info.get("verified_at"),
                    "is_online": device.get("status") == "online" if device else False,
                }
            )

    return devices


# ============ Unified Memory Stats ============


def get_memory_stats() -> Dict:
    """
    Get unified memory statistics across all data stores.

    This aggregates stats from:
    - Notes
    - Projects
    - Automations
    - Agents
    - Audit entries

    Used by the AGI-readiness infrastructure for context awareness.
    """
    stats = {
        "total_items": 0,
        "by_scope": {},
        "by_type": {},
        "storage_estimate_kb": 0,
        "oldest_timestamp": None,
        "newest_timestamp": None,
    }

    # Notes stats
    notes = get_notes()
    notes_count = len(notes) if notes else 0
    stats["by_scope"]["NOTES"] = notes_count
    stats["by_type"]["NOTE"] = notes_count
    stats["total_items"] += notes_count

    # Calculate notes storage estimate (rough)
    if NOTES_FILE.exists():
        stats["storage_estimate_kb"] += NOTES_FILE.stat().st_size // 1024

    # Projects stats
    projects = get_projects()
    projects_count = len(projects) if projects else 0
    stats["by_scope"]["PROJECTS"] = projects_count
    stats["by_type"]["PROJECT"] = projects_count
    stats["total_items"] += projects_count

    if PROJECTS_FILE.exists():
        stats["storage_estimate_kb"] += PROJECTS_FILE.stat().st_size // 1024

    # Automations stats
    automations = get_automations()
    automations_count = len(automations) if automations else 0
    stats["by_scope"]["AUTOMATIONS"] = automations_count
    stats["by_type"]["AUTOMATION"] = automations_count
    stats["total_items"] += automations_count

    if AUTOMATIONS_FILE.exists():
        stats["storage_estimate_kb"] += AUTOMATIONS_FILE.stat().st_size // 1024

    # Agents stats
    agents = get_agents()
    agents_count = len(agents) if agents else 0
    stats["by_scope"]["AGENTS"] = agents_count
    stats["by_type"]["AGENT"] = agents_count
    stats["total_items"] += agents_count

    if AGENTS_FILE.exists():
        stats["storage_estimate_kb"] += AGENTS_FILE.stat().st_size // 1024

    # Audit stats (from audit_stats function)
    audit_stats_data = get_audit_stats("30D")
    audit_count = audit_stats_data.get("total_events", 0) if audit_stats_data else 0
    stats["by_scope"]["AUDIT"] = audit_count
    stats["by_type"]["AUDIT_ENTRY"] = audit_count
    stats["total_items"] += audit_count

    # Get timestamps from recent notes
    timestamps = []
    for note in notes or []:
        ts = note.get("timestamp")
        if ts:
            timestamps.append(ts)

    for project in projects or []:
        ts = project.get("last_modified") or project.get("created_at")
        if ts:
            timestamps.append(ts)

    if timestamps:
        # timestamps could be strings or ints - handle both
        numeric_ts = []
        for ts in timestamps:
            if isinstance(ts, (int, float)):
                numeric_ts.append(ts)
            elif isinstance(ts, str):
                try:
                    # Try parsing ISO format
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    numeric_ts.append(dt.timestamp() * 1000)
                except:
                    pass

        if numeric_ts:
            stats["oldest_timestamp"] = int(min(numeric_ts))
            stats["newest_timestamp"] = int(max(numeric_ts))

    # Add scope summaries
    stats["scope_summary"] = {
        "USER_DATA": stats["by_scope"].get("NOTES", 0)
        + stats["by_scope"].get("PROJECTS", 0),
        "AI_DATA": stats["by_scope"].get("AGENTS", 0)
        + stats["by_scope"].get("AUTOMATIONS", 0),
        "COMPLIANCE": stats["by_scope"].get("AUDIT", 0),
    }

    return stats


def get_memory_search_results(
    query: str, scopes: List[str] = None, limit: int = 20
) -> Dict:
    """
    Unified search across memory scopes.

    This is a keyword-based search until vector store is implemented.
    Matches the UnifiedMemoryManager interface on Android.

    Args:
        query: Search query string
        scopes: List of scopes to search (NOTES, PROJECTS, AGENTS, AUTOMATIONS)
        limit: Maximum results to return

    Returns:
        Dict with items list and metadata
    """
    if scopes is None:
        scopes = ["NOTES", "PROJECTS", "AGENTS", "AUTOMATIONS"]

    # Map scopes to search_all types
    type_map = {
        "NOTES": "notes",
        "PROJECTS": "projects",
        "AGENTS": "agents",
        "AUTOMATIONS": "automations",
    }

    types = [type_map[s] for s in scopes if s in type_map]

    # Use existing search_all function
    results = search_all(query, types)
    items = results.get("items", [])[:limit]

    # Transform to unified memory format
    memory_items = []
    for item in items:
        memory_items.append(
            {
                "id": item.get("id"),
                "type": item.get("type", "").upper(),
                "scope": item.get("type", "").upper() + "S",  # "note" -> "NOTES"
                "title": item.get("title", ""),
                "content": item.get("preview", ""),
                "timestamp": datetime.now().timestamp()
                * 1000,  # TODO: Get actual timestamp
                "relevance_score": 1.0,  # TODO: Implement ranking when vector store is added
            }
        )

    return {
        "items": memory_items,
        "total_found": len(memory_items),
        "query": query,
        "scopes_searched": scopes,
        "search_type": "keyword",  # Will become "semantic" when vector store is added
    }


# ============ Data Cleanup ============


def cleanup_stale_data(days_threshold: int = 30) -> Dict:
    """
    Remove stale projects, notes, and automations.
    Keeps data from devices active in last N days.

    Args:
        days_threshold: Number of days - devices not seen within this period are stale

    Returns:
        Dict with success status and counts of removed items
    """
    threshold_ts = datetime.now().timestamp() - (days_threshold * 24 * 60 * 60)

    # Read projects data to find stale devices
    projects_data = _read_json_file(PROJECTS_FILE) or {"devices": {}}
    devices_data = projects_data.get("devices", {})

    # Find stale device IDs (not seen in last N days)
    stale_devices = set()
    most_recent_device = None
    most_recent_ts = 0

    for device_id, device_info in devices_data.items():
        last_seen = device_info.get("last_seen", 0)
        if last_seen > most_recent_ts:
            most_recent_ts = last_seen
            most_recent_device = device_id
        if last_seen < threshold_ts:
            stale_devices.add(device_id)

    # Never remove the most recent device
    if most_recent_device and most_recent_device in stale_devices:
        stale_devices.remove(most_recent_device)

    if not stale_devices:
        return {
            "success": True,
            "removed_projects": 0,
            "removed_notes": 0,
            "removed_automations": 0,
            "message": "No stale data found",
        }

    removed_projects = 0
    removed_notes = 0
    removed_automations = 0

    # Clean projects
    for device_id in stale_devices:
        if device_id in projects_data.get("devices", {}):
            device_projects = (
                projects_data["devices"].get(device_id, {}).get("projects", [])
            )
            removed_projects += len(device_projects)
            del projects_data["devices"][device_id]

    _write_json_file(PROJECTS_FILE, projects_data)

    # Clean notes
    notes_data = _read_json_file(NOTES_FILE) or {"devices": {}}
    for device_id in stale_devices:
        if device_id in notes_data.get("devices", {}):
            device_notes = notes_data["devices"].get(device_id, {}).get("notes", [])
            removed_notes += len(device_notes)
            del notes_data["devices"][device_id]

    _write_json_file(NOTES_FILE, notes_data)

    # Clean automations
    auto_data = _read_json_file(AUTOMATIONS_FILE) or {}
    for device_id in stale_devices:
        if device_id in auto_data:
            device_autos = auto_data.get(device_id, {}).get("automations", [])
            removed_automations += len(device_autos)
            del auto_data[device_id]

    _write_json_file(AUTOMATIONS_FILE, auto_data)

    return {
        "success": True,
        "removed_projects": removed_projects,
        "removed_notes": removed_notes,
        "removed_automations": removed_automations,
        "stale_devices_cleaned": len(stale_devices),
    }


def reset_device_history() -> Dict:
    """
    Reset device history, keeping only the most recently active device.

    Returns:
        Dict with success status and info about kept/removed devices
    """
    # Read projects data to find devices
    projects_data = _read_json_file(PROJECTS_FILE) or {"devices": {}}
    devices_data = projects_data.get("devices", {})

    if not devices_data:
        return {
            "success": True,
            "kept_device": None,
            "removed_count": 0,
            "message": "No devices to clean",
        }

    # Find the most recent device
    most_recent_device = None
    most_recent_ts = 0
    most_recent_name = None

    for device_id, device_info in devices_data.items():
        last_seen = device_info.get("last_seen", 0)
        if last_seen > most_recent_ts:
            most_recent_ts = last_seen
            most_recent_device = device_id
            most_recent_name = device_info.get("name", device_id)

    if not most_recent_device:
        # Just pick the first one if no timestamps
        most_recent_device = list(devices_data.keys())[0]
        most_recent_name = devices_data[most_recent_device].get(
            "name", most_recent_device
        )

    # Get list of devices to remove (all except most recent)
    devices_to_remove = [d for d in devices_data.keys() if d != most_recent_device]
    removed_count = len(devices_to_remove)

    # Clean projects data
    new_devices_data = {most_recent_device: devices_data[most_recent_device]}
    projects_data["devices"] = new_devices_data
    _write_json_file(PROJECTS_FILE, projects_data)

    # Clean notes
    notes_data = _read_json_file(NOTES_FILE) or {"devices": {}}
    if most_recent_device in notes_data.get("devices", {}):
        notes_data["devices"] = {
            most_recent_device: notes_data["devices"][most_recent_device]
        }
    else:
        notes_data["devices"] = {}
    _write_json_file(NOTES_FILE, notes_data)

    # Clean automations
    auto_data = _read_json_file(AUTOMATIONS_FILE) or {}
    if most_recent_device in auto_data:
        auto_data = {most_recent_device: auto_data[most_recent_device]}
    else:
        auto_data = {}
    _write_json_file(AUTOMATIONS_FILE, auto_data)

    return {
        "success": True,
        "kept_device": most_recent_name,
        "removed_count": removed_count,
    }


def remove_device(device_id: str) -> Dict:
    """
    Remove a specific device and all its associated data.

    Args:
        device_id: The ID of the device to remove

    Returns:
        Dict with success status and details
    """
    # Read projects data
    projects_data = _read_json_file(PROJECTS_FILE) or {"devices": {}}
    devices_data = projects_data.get("devices", {})

    if device_id not in devices_data:
        return {
            "success": False,
            "error": "Device not found",
        }

    # Get device name before removing
    device_name = devices_data[device_id].get("name", device_id)

    # Check if this is the only device
    if len(devices_data) == 1:
        return {
            "success": False,
            "error": "Cannot remove the only device. At least one device must remain.",
        }

    # Remove from projects data
    del devices_data[device_id]
    projects_data["devices"] = devices_data
    _write_json_file(PROJECTS_FILE, projects_data)

    # Remove from notes
    notes_data = _read_json_file(NOTES_FILE) or {"devices": {}}
    if device_id in notes_data.get("devices", {}):
        del notes_data["devices"][device_id]
        _write_json_file(NOTES_FILE, notes_data)

    # Remove from automations
    auto_data = _read_json_file(AUTOMATIONS_FILE) or {}
    if device_id in auto_data:
        del auto_data[device_id]
        _write_json_file(AUTOMATIONS_FILE, auto_data)

    return {
        "success": True,
        "removed_device": device_name,
        "device_id": device_id,
    }
