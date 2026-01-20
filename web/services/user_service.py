"""
User Service - Multi-user management for ShadowAI collaboration

Provides user registration, authentication, device linking, and session management.
Uses JSON file storage to match existing shadow-bridge patterns.
"""

import json
import threading
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from ..models.user import User, UserDevice, Session

# File paths
SHADOWAI_DIR = Path.home() / ".shadowai"
USERS_FILE = SHADOWAI_DIR / "users.json"
SESSIONS_FILE = SHADOWAI_DIR / "user_sessions.json"

# Thread lock for file I/O
_file_lock = threading.Lock()

# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


def _ensure_dir():
    """Ensure the shadow data directory exists."""
    SHADOWAI_DIR.mkdir(parents=True, exist_ok=True)


def _read_json_file(path: Path) -> Dict[str, Any]:
    """Read JSON file with thread safety."""
    with _file_lock:
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {path}: {e}")
    return {}


def _write_json_file(path: Path, data: Dict[str, Any]):
    """Write JSON file with thread safety."""
    with _file_lock:
        try:
            _ensure_dir()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error writing {path}: {e}")
            raise


# =============================================================================
# User Management
# =============================================================================

def get_all_users() -> List[User]:
    """Get all registered users."""
    data = _read_json_file(USERS_FILE)
    users_data = data.get("users", [])
    return [User.from_dict(u) for u in users_data]


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID."""
    users = get_all_users()
    for user in users:
        if user.id == user_id:
            return user
    return None


def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email address."""
    email = email.lower().strip()
    users = get_all_users()
    for user in users:
        if user.email == email:
            return user
    return None


def get_user_by_device(device_id: str) -> Optional[User]:
    """Get user by linked device ID."""
    users = get_all_users()
    for user in users:
        if user.get_device(device_id):
            return user
    return None


def create_user(email: str, password: str, display_name: Optional[str] = None) -> User:
    """
    Create a new user account.

    Raises:
        ValueError: If email is invalid or already registered
    """
    email = email.lower().strip()

    # Validate email
    if not EMAIL_REGEX.match(email):
        raise ValueError("Invalid email address format")

    # Check for existing user
    if get_user_by_email(email):
        raise ValueError("Email already registered")

    # Validate password
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    # Create user
    user = User.create(email, password, display_name)

    # Save to file
    data = _read_json_file(USERS_FILE)
    users_data = data.get("users", [])
    users_data.append(user.to_dict(include_password=True))
    data["users"] = users_data
    _write_json_file(USERS_FILE, data)

    return user


def update_user(user: User):
    """Update user in storage."""
    data = _read_json_file(USERS_FILE)
    users_data = data.get("users", [])

    # Replace user data
    users_data = [u for u in users_data if u.get("id") != user.id]
    users_data.append(user.to_dict(include_password=True))

    data["users"] = users_data
    _write_json_file(USERS_FILE, data)


def delete_user(user_id: str) -> bool:
    """Delete a user account and all associated data."""
    data = _read_json_file(USERS_FILE)
    users_data = data.get("users", [])
    original_count = len(users_data)

    users_data = [u for u in users_data if u.get("id") != user_id]
    data["users"] = users_data
    _write_json_file(USERS_FILE, data)

    # Also delete user's sessions
    invalidate_all_sessions(user_id)

    return len(users_data) < original_count


# =============================================================================
# Authentication
# =============================================================================

def authenticate(email: str, password: str) -> Optional[User]:
    """
    Authenticate user with email and password.

    Returns User if successful, None otherwise.
    """
    user = get_user_by_email(email)
    if user and user.is_active and user.check_password(password):
        # Update last login
        user.last_login = datetime.utcnow().isoformat()
        update_user(user)
        return user
    return None


def create_session(user_id: str, device_id: Optional[str] = None,
                   ip_address: Optional[str] = None,
                   user_agent: Optional[str] = None,
                   expires_hours: int = 24 * 7) -> Session:
    """Create a new session for a user."""
    session = Session.create(user_id, device_id, expires_hours)
    session.ip_address = ip_address
    session.user_agent = user_agent

    # Save session
    data = _read_json_file(SESSIONS_FILE)
    sessions_data = data.get("sessions", [])
    sessions_data.append(session.to_dict())
    data["sessions"] = sessions_data
    _write_json_file(SESSIONS_FILE, data)

    return session


def get_session(session_id: str) -> Optional[Session]:
    """Get session by ID."""
    data = _read_json_file(SESSIONS_FILE)
    sessions_data = data.get("sessions", [])
    for s in sessions_data:
        if s.get("id") == session_id:
            session = Session.from_dict(s)
            if not session.is_expired():
                return session
    return None


def validate_session(session_id: str) -> Optional[User]:
    """
    Validate session and return associated user.

    Returns User if session is valid, None otherwise.
    """
    session = get_session(session_id)
    if session:
        return get_user_by_id(session.user_id)
    return None


def invalidate_session(session_id: str) -> bool:
    """Invalidate a specific session."""
    data = _read_json_file(SESSIONS_FILE)
    sessions_data = data.get("sessions", [])
    original_count = len(sessions_data)

    sessions_data = [s for s in sessions_data if s.get("id") != session_id]
    data["sessions"] = sessions_data
    _write_json_file(SESSIONS_FILE, data)

    return len(sessions_data) < original_count


def invalidate_all_sessions(user_id: str) -> int:
    """Invalidate all sessions for a user."""
    data = _read_json_file(SESSIONS_FILE)
    sessions_data = data.get("sessions", [])
    original_count = len(sessions_data)

    sessions_data = [s for s in sessions_data if s.get("user_id") != user_id]
    data["sessions"] = sessions_data
    _write_json_file(SESSIONS_FILE, data)

    return original_count - len(sessions_data)


def cleanup_expired_sessions() -> int:
    """Remove all expired sessions. Returns count removed."""
    data = _read_json_file(SESSIONS_FILE)
    sessions_data = data.get("sessions", [])
    original_count = len(sessions_data)

    valid_sessions = []
    for s in sessions_data:
        session = Session.from_dict(s)
        if not session.is_expired():
            valid_sessions.append(s)

    data["sessions"] = valid_sessions
    _write_json_file(SESSIONS_FILE, data)

    return original_count - len(valid_sessions)


# =============================================================================
# Device Management
# =============================================================================

def link_device(user_id: str, device_id: str, device_name: str,
                platform: str = "android", push_token: Optional[str] = None) -> bool:
    """
    Link a device to a user account.

    If device is already linked to another user, it will be unlinked first.
    """
    # Check if device is linked to another user
    existing_user = get_user_by_device(device_id)
    if existing_user and existing_user.id != user_id:
        existing_user.remove_device(device_id)
        update_user(existing_user)

    # Link to new user
    user = get_user_by_id(user_id)
    if not user:
        return False

    device = UserDevice(
        device_id=device_id,
        device_name=device_name,
        platform=platform,
        push_token=push_token,
    )
    user.add_device(device)
    update_user(user)

    return True


def unlink_device(user_id: str, device_id: str) -> bool:
    """Unlink a device from a user account."""
    user = get_user_by_id(user_id)
    if not user:
        return False

    user.remove_device(device_id)
    update_user(user)
    return True


def update_device_push_token(device_id: str, push_token: str) -> bool:
    """Update push notification token for a device."""
    user = get_user_by_device(device_id)
    if not user:
        return False

    device = user.get_device(device_id)
    if device:
        device.push_token = push_token
        update_user(user)
        return True

    return False


def update_device_last_seen(device_id: str) -> bool:
    """Update device's last seen timestamp."""
    user = get_user_by_device(device_id)
    if user:
        user.update_device_last_seen(device_id)
        update_user(user)
        return True
    return False


# =============================================================================
# Password Management
# =============================================================================

def change_password(user_id: str, old_password: str, new_password: str) -> bool:
    """
    Change user's password.

    Requires old password for verification.
    """
    user = get_user_by_id(user_id)
    if not user:
        return False

    if not user.check_password(old_password):
        return False

    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters")

    user.update_password(new_password)
    update_user(user)

    # Invalidate all sessions for security
    invalidate_all_sessions(user_id)

    return True


def reset_password(email: str, new_password: str) -> bool:
    """
    Reset password for a user (admin/recovery operation).

    In production, this should be gated by email verification.
    """
    user = get_user_by_email(email)
    if not user:
        return False

    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters")

    user.update_password(new_password)
    update_user(user)

    # Invalidate all sessions
    invalidate_all_sessions(user.id)

    return True


# =============================================================================
# User Settings
# =============================================================================

def update_user_settings(user_id: str, settings: Dict[str, Any]) -> bool:
    """Update user settings."""
    user = get_user_by_id(user_id)
    if not user:
        return False

    user.settings.update(settings)
    user.updated_at = datetime.utcnow().isoformat()
    update_user(user)
    return True


def update_user_profile(user_id: str, display_name: Optional[str] = None,
                        avatar_url: Optional[str] = None,
                        presence_visible: Optional[bool] = None,
                        allow_invites: Optional[bool] = None) -> bool:
    """Update user profile fields."""
    user = get_user_by_id(user_id)
    if not user:
        return False

    if display_name is not None:
        user.display_name = display_name
    if avatar_url is not None:
        user.avatar_url = avatar_url
    if presence_visible is not None:
        user.presence_visible = presence_visible
    if allow_invites is not None:
        user.allow_invites = allow_invites

    user.updated_at = datetime.utcnow().isoformat()
    update_user(user)
    return True


# =============================================================================
# Statistics
# =============================================================================

def get_user_stats() -> Dict[str, Any]:
    """Get user statistics."""
    users = get_all_users()
    active_users = [u for u in users if u.is_active]

    total_devices = sum(len(u.devices) for u in users)

    # Count sessions
    data = _read_json_file(SESSIONS_FILE)
    sessions_data = data.get("sessions", [])
    active_sessions = [s for s in sessions_data if not Session.from_dict(s).is_expired()]

    return {
        "total_users": len(users),
        "active_users": len(active_users),
        "total_devices": total_devices,
        "active_sessions": len(active_sessions),
    }
