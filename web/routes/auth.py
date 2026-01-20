"""
Authentication Routes - Login, Register, Session management

Provides REST API endpoints for multi-user authentication in ShadowAI.
"""

from flask import Blueprint, request, jsonify, g
from functools import wraps
from typing import Optional

from ..services import user_service

auth_bp = Blueprint("auth", __name__)


# =============================================================================
# Authentication Decorator
# =============================================================================

def get_current_user():
    """Get the current authenticated user from the request."""
    # Check for session token in header or cookie
    session_id = request.headers.get("X-Session-Token") or \
                 request.cookies.get("session_token")

    if session_id:
        return user_service.validate_session(session_id)
    return None


def login_required(f):
    """Decorator to require authentication for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return decorated_function


def login_optional(f):
    """Decorator that loads user if authenticated but doesn't require it."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.current_user = get_current_user()
        return f(*args, **kwargs)
    return decorated_function


# =============================================================================
# Registration & Login
# =============================================================================

@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register a new user account.

    Request body:
        email: User's email address
        password: Password (min 8 characters)
        display_name: Optional display name

    Returns:
        User object and session token on success
    """
    data = request.get_json() or {}

    email = data.get("email", "").strip()
    password = data.get("password", "")
    display_name = data.get("display_name")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        user = user_service.create_user(email, password, display_name)

        # Create session
        session = user_service.create_session(
            user.id,
            device_id=data.get("device_id"),
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
        )

        response = jsonify({
            "success": True,
            "user": user.to_dict(),
            "session_token": session.id,
        })

        # Set session cookie
        response.set_cookie(
            "session_token",
            session.id,
            httponly=True,
            secure=request.is_secure,
            samesite="Lax",
            max_age=7 * 24 * 60 * 60,  # 7 days
        )

        return response

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Authenticate user and create session.

    Request body:
        email: User's email address
        password: User's password
        device_id: Optional device ID to link
        device_name: Optional device name

    Returns:
        User object and session token on success
    """
    data = request.get_json() or {}

    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = user_service.authenticate(email, password)
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    # Create session
    session = user_service.create_session(
        user.id,
        device_id=data.get("device_id"),
        ip_address=request.remote_addr,
        user_agent=request.headers.get("User-Agent"),
    )

    # Link device if provided
    device_id = data.get("device_id")
    device_name = data.get("device_name")
    if device_id and device_name:
        user_service.link_device(
            user.id,
            device_id,
            device_name,
            platform=data.get("platform", "android"),
        )

    response = jsonify({
        "success": True,
        "user": user.to_dict(),
        "session_token": session.id,
    })

    # Set session cookie
    response.set_cookie(
        "session_token",
        session.id,
        httponly=True,
        secure=request.is_secure,
        samesite="Lax",
        max_age=7 * 24 * 60 * 60,
    )

    return response


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    """
    Log out current session.

    Invalidates the current session token.
    """
    session_id = request.headers.get("X-Session-Token") or \
                 request.cookies.get("session_token")

    if session_id:
        user_service.invalidate_session(session_id)

    response = jsonify({"success": True})
    response.delete_cookie("session_token")
    return response


@auth_bp.route("/logout-all", methods=["POST"])
@login_required
def logout_all():
    """
    Log out all sessions for the current user.

    Useful for security or when changing password.
    """
    count = user_service.invalidate_all_sessions(g.current_user.id)

    response = jsonify({
        "success": True,
        "sessions_invalidated": count,
    })
    response.delete_cookie("session_token")
    return response


# =============================================================================
# Current User
# =============================================================================

@auth_bp.route("/me", methods=["GET"])
@login_required
def get_me():
    """Get current authenticated user's profile."""
    return jsonify({
        "user": g.current_user.to_dict(),
    })


@auth_bp.route("/me", methods=["PATCH"])
@login_required
def update_me():
    """
    Update current user's profile.

    Request body (all optional):
        display_name: New display name
        avatar_url: Avatar image URL
        presence_visible: Whether online status is visible
        allow_invites: Whether to allow collaboration invites
    """
    data = request.get_json() or {}

    user_service.update_user_profile(
        g.current_user.id,
        display_name=data.get("display_name"),
        avatar_url=data.get("avatar_url"),
        presence_visible=data.get("presence_visible"),
        allow_invites=data.get("allow_invites"),
    )

    # Reload user
    user = user_service.get_user_by_id(g.current_user.id)
    return jsonify({
        "success": True,
        "user": user.to_dict(),
    })


@auth_bp.route("/me/settings", methods=["GET"])
@login_required
def get_my_settings():
    """Get current user's settings."""
    return jsonify({
        "settings": g.current_user.settings,
    })


@auth_bp.route("/me/settings", methods=["PATCH"])
@login_required
def update_my_settings():
    """Update current user's settings (partial update)."""
    data = request.get_json() or {}

    user_service.update_user_settings(g.current_user.id, data)

    user = user_service.get_user_by_id(g.current_user.id)
    return jsonify({
        "success": True,
        "settings": user.settings,
    })


# =============================================================================
# Password Management
# =============================================================================

@auth_bp.route("/change-password", methods=["POST"])
@login_required
def change_password():
    """
    Change current user's password.

    Request body:
        old_password: Current password
        new_password: New password (min 8 characters)
    """
    data = request.get_json() or {}

    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    if not old_password or not new_password:
        return jsonify({"error": "Both old and new passwords are required"}), 400

    try:
        success = user_service.change_password(
            g.current_user.id,
            old_password,
            new_password,
        )

        if success:
            # Create new session after password change
            session = user_service.create_session(
                g.current_user.id,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
            )

            response = jsonify({
                "success": True,
                "session_token": session.id,
                "message": "Password changed. All other sessions invalidated.",
            })
            response.set_cookie(
                "session_token",
                session.id,
                httponly=True,
                secure=request.is_secure,
                samesite="Lax",
                max_age=7 * 24 * 60 * 60,
            )
            return response
        else:
            return jsonify({"error": "Current password is incorrect"}), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# =============================================================================
# Device Management
# =============================================================================

@auth_bp.route("/devices", methods=["GET"])
@login_required
def get_devices():
    """Get all devices linked to current user."""
    return jsonify({
        "devices": [d.to_dict() for d in g.current_user.devices],
    })


@auth_bp.route("/devices", methods=["POST"])
@login_required
def link_device():
    """
    Link a new device to current user.

    Request body:
        device_id: Unique device identifier
        device_name: Human-readable device name
        platform: Device platform (android, ios, web, desktop)
        push_token: Optional push notification token
    """
    data = request.get_json() or {}

    device_id = data.get("device_id")
    device_name = data.get("device_name")

    if not device_id or not device_name:
        return jsonify({"error": "device_id and device_name are required"}), 400

    success = user_service.link_device(
        g.current_user.id,
        device_id,
        device_name,
        platform=data.get("platform", "android"),
        push_token=data.get("push_token"),
    )

    if success:
        user = user_service.get_user_by_id(g.current_user.id)
        return jsonify({
            "success": True,
            "devices": [d.to_dict() for d in user.devices],
        })
    else:
        return jsonify({"error": "Failed to link device"}), 500


@auth_bp.route("/devices/<device_id>", methods=["DELETE"])
@login_required
def unlink_device(device_id):
    """Unlink a device from current user."""
    success = user_service.unlink_device(g.current_user.id, device_id)

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Device not found"}), 404


@auth_bp.route("/devices/<device_id>/push-token", methods=["PUT"])
@login_required
def update_push_token(device_id):
    """Update push notification token for a device."""
    data = request.get_json() or {}
    push_token = data.get("push_token")

    if not push_token:
        return jsonify({"error": "push_token is required"}), 400

    # Verify device belongs to user
    if not g.current_user.get_device(device_id):
        return jsonify({"error": "Device not found"}), 404

    success = user_service.update_device_push_token(device_id, push_token)

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to update push token"}), 500


# =============================================================================
# Validation / Status
# =============================================================================

@auth_bp.route("/validate", methods=["POST"])
def validate_session():
    """
    Validate a session token.

    Request body or header:
        session_token: Token to validate

    Returns:
        User info if valid, error if invalid
    """
    data = request.get_json() or {}
    session_id = data.get("session_token") or \
                 request.headers.get("X-Session-Token") or \
                 request.cookies.get("session_token")

    if not session_id:
        return jsonify({"valid": False, "error": "No session token provided"}), 400

    user = user_service.validate_session(session_id)

    if user:
        return jsonify({
            "valid": True,
            "user": user.to_public_dict(),
        })
    else:
        return jsonify({"valid": False}), 401


@auth_bp.route("/check-email", methods=["POST"])
def check_email():
    """
    Check if an email is already registered.

    Request body:
        email: Email to check

    Returns:
        exists: True if email is registered
    """
    data = request.get_json() or {}
    email = data.get("email", "").strip()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    user = user_service.get_user_by_email(email)
    return jsonify({
        "exists": user is not None,
    })


# =============================================================================
# Admin / Stats (Optional)
# =============================================================================

@auth_bp.route("/stats", methods=["GET"])
def get_auth_stats():
    """Get user/auth statistics (for dashboard)."""
    stats = user_service.get_user_stats()
    return jsonify(stats)
