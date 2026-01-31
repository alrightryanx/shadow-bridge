"""
WebSocket handlers for real-time updates and presence indicators

Supports:
- Device/client connection tracking
- User presence (online/offline/idle)
- Typing indicators
- Real-time collaboration events
"""

from flask import request
from datetime import datetime
from typing import Dict, Any, Optional

# Client tracking - maps session ID to client info
connected_clients: Dict[str, Dict[str, Any]] = {}

# User presence tracking - maps user_id to presence info
user_presence: Dict[str, Dict[str, Any]] = {}

# Import socketio - may be None in frozen builds
from ..app import socketio

# Presence status constants
PRESENCE_ONLINE = "online"
PRESENCE_IDLE = "idle"
PRESENCE_OFFLINE = "offline"
PRESENCE_BUSY = "busy"


def _get_user_from_session(session_token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Validate session and get user info."""
    if not session_token:
        return None
    try:
        from ..services import user_service
        user = user_service.validate_session(session_token)
        if user:
            return user.to_public_dict()
    except Exception:
        pass
    return None


# Only register handlers if socketio is available
if socketio is not None:
    from flask_socketio import emit, join_room, leave_room

    @socketio.on("connect")
    def handle_connect():
        """Handle client connection."""
        sid = request.sid if hasattr(request, "sid") else "unknown"
        connected_clients[sid] = {
            "sid": sid,
            "connected_at": datetime.utcnow().isoformat(),
            "user_id": None,
            "device_id": None,
        }
        emit("status", {"connected": True, "clients": len(connected_clients)})

    @socketio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection."""
        sid = request.sid if hasattr(request, "sid") else "unknown"
        client = connected_clients.pop(sid, None)

        # Update user presence if this was an authenticated user
        if client and client.get("user_id"):
            user_id = client["user_id"]
            # Check if user has other active connections
            has_other_connections = any(
                c.get("user_id") == user_id for c in connected_clients.values()
            )
            if not has_other_connections:
                # User is now offline
                _update_user_presence(user_id, PRESENCE_OFFLINE)
                broadcast_presence_change(user_id, PRESENCE_OFFLINE)

    @socketio.on("authenticate")
    def handle_authenticate(data):
        """Authenticate WebSocket connection with session token."""
        sid = request.sid if hasattr(request, "sid") else "unknown"
        session_token = data.get("session_token")
        device_id = data.get("device_id")

        user = _get_user_from_session(session_token)

        if user:
            # Update client info
            if sid in connected_clients:
                connected_clients[sid]["user_id"] = user["id"]
                connected_clients[sid]["device_id"] = device_id
                connected_clients[sid]["user"] = user

            # Join user's personal room for targeted messages
            join_room(f"user:{user['id']}")

            # Update presence
            _update_user_presence(user["id"], PRESENCE_ONLINE)
            broadcast_presence_change(user["id"], PRESENCE_ONLINE, user)

            emit("authenticated", {
                "success": True,
                "user": user,
            })
        else:
            emit("authenticated", {
                "success": False,
                "error": "Invalid session",
            })

    @socketio.on("subscribe")
    def handle_subscribe(data):
        """Subscribe to specific update channels."""
        channel = data.get("channel", "all")
        join_room(channel)
        emit("subscribed", {"channel": channel})

    @socketio.on("unsubscribe")
    def handle_unsubscribe(data):
        """Unsubscribe from update channels."""
        channel = data.get("channel", "all")
        leave_room(channel)

    # =========================================================================
    # Presence Events
    # =========================================================================

    @socketio.on("presence_update")
    def handle_presence_update(data):
        """Handle user presence status update."""
        sid = request.sid if hasattr(request, "sid") else "unknown"
        client = connected_clients.get(sid)

        if not client or not client.get("user_id"):
            emit("error", {"message": "Not authenticated"})
            return

        user_id = client["user_id"]
        status = data.get("status", PRESENCE_ONLINE)

        # Validate status
        if status not in [PRESENCE_ONLINE, PRESENCE_IDLE, PRESENCE_BUSY]:
            emit("error", {"message": "Invalid status"})
            return

        _update_user_presence(user_id, status)
        broadcast_presence_change(user_id, status, client.get("user"))

        emit("presence_updated", {"status": status})

    @socketio.on("typing_start")
    def handle_typing_start(data):
        """Handle typing indicator start."""
        sid = request.sid if hasattr(request, "sid") else "unknown"
        client = connected_clients.get(sid)

        if not client or not client.get("user_id"):
            return

        session_id = data.get("session_id")
        if session_id:
            broadcast_typing_indicator(
                session_id,
                client["user_id"],
                client.get("user"),
                is_typing=True,
            )

    @socketio.on("typing_stop")
    def handle_typing_stop(data):
        """Handle typing indicator stop."""
        sid = request.sid if hasattr(request, "sid") else "unknown"
        client = connected_clients.get(sid)

        if not client or not client.get("user_id"):
            return

        session_id = data.get("session_id")
        if session_id:
            broadcast_typing_indicator(
                session_id,
                client["user_id"],
                client.get("user"),
                is_typing=False,
            )

    @socketio.on("join_session")
    def handle_join_session(data):
        """Join a shared session room for collaboration."""
        sid = request.sid if hasattr(request, "sid") else "unknown"
        client = connected_clients.get(sid)

        if not client or not client.get("user_id"):
            emit("error", {"message": "Not authenticated"})
            return

        session_id = data.get("session_id")
        if session_id:
            room = f"session:{session_id}"
            join_room(room)
            broadcast_session_join(session_id, client["user_id"], client.get("user"))
            emit("joined_session", {"session_id": session_id})

    @socketio.on("leave_session")
    def handle_leave_session(data):
        """Leave a shared session room."""
        sid = request.sid if hasattr(request, "sid") else "unknown"
        client = connected_clients.get(sid)

        if not client or not client.get("user_id"):
            return

        session_id = data.get("session_id")
        if session_id:
            room = f"session:{session_id}"
            broadcast_session_leave(session_id, client["user_id"], client.get("user"))
            leave_room(room)
            emit("left_session", {"session_id": session_id})

    @socketio.on("get_presence")
    def handle_get_presence(data):
        """Get current presence of specific users."""
        user_ids = data.get("user_ids", [])
        presence_list = []

        for user_id in user_ids:
            if user_id in user_presence:
                presence_list.append({
                    "user_id": user_id,
                    **user_presence[user_id],
                })
            else:
                presence_list.append({
                    "user_id": user_id,
                    "status": PRESENCE_OFFLINE,
                })

        emit("presence_list", {"users": presence_list})

    # =========================================================================
    # CLI Execution Events (for WebSocket Backend)
    # =========================================================================

    @socketio.on("cli_execute")
    def handle_cli_execute(data):
        """Execute CLI command and stream results."""
        sid = request.sid if hasattr(request, "sid") else "unknown"

        # Validate authentication (device must be connected)
        device_id = request.headers.get("X-Device-ID")
        if not device_id:
            # Try to get from connected_clients
            client = connected_clients.get(sid)
            device_id = client.get("device_id") if client else None

        if not device_id:
            emit("error", {"message": "Authentication required"})
            return

        # Extract parameters
        provider = data.get("provider", "claude")
        model = data.get("model")
        query = data.get("query")
        working_dir = data.get("workingDirectory")
        continue_conversation = data.get("continueConversation", False)
        auto_accept_edits = data.get("autoAcceptEdits", True)
        channel_id = data.get("channelId")
        context = data.get("context", [])

        if not query:
            emit("error", {"message": "Query is required"})
            return

        # Import CLI executor (lazy import to avoid circular dependencies)
        try:
            from ..services.cli_executor import execute_cli_command
        except ImportError:
            emit("error", {"message": "CLI executor not available"})
            return

        # Callback to stream chunks back to client
        def on_chunk(chunk_data):
            emit("stream_chunk", chunk_data)

        # Execute CLI command asynchronously
        try:
            execute_cli_command(
                provider=provider,
                model=model,
                query=query,
                working_dir=working_dir,
                continue_conversation=continue_conversation,
                auto_accept_edits=auto_accept_edits,
                context=context,
                on_chunk=on_chunk,
                session_id=sid
            )
        except Exception as e:
            emit("error", {"message": f"Failed to execute command: {str(e)}"})

    @socketio.on("approval_response")
    def handle_approval_response(data):
        """Handle approval response from Android client."""
        approval_id = data.get("approvalId")
        approved = data.get("approved", False)
        selected_option = data.get("selectedOption")

        if not approval_id:
            emit("error", {"message": "Approval ID is required"})
            return

        # Import CLI executor
        try:
            from ..services.cli_executor import send_approval_response
        except ImportError:
            emit("error", {"message": "CLI executor not available"})
            return

        # Send approval to active CLI process
        success = send_approval_response(approval_id, approved, selected_option)

        if not success:
            emit("error", {"message": "Failed to send approval response"})


def _update_user_presence(user_id: str, status: str):
    """Update user presence in tracking dict."""
    user_presence[user_id] = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat(),
    }


# Functions to broadcast updates (called from data service or main app)
# These gracefully handle socketio being None


def broadcast_device_connected(device_id, device_name):
    """Broadcast device connection event."""
    if socketio is not None:
        socketio.emit(
            "device_connected",
            {"device_id": device_id, "device_name": device_name},
            room="all",
        )


def broadcast_device_disconnected(device_id):
    """Broadcast device disconnection event."""
    if socketio is not None:
        socketio.emit("device_disconnected", {"device_id": device_id}, room="all")


def broadcast_projects_updated(device_id):
    """Broadcast projects sync event."""
    if socketio is not None:
        socketio.emit("projects_updated", {"device_id": device_id}, room="all")


def broadcast_notes_updated(device_id):
    """Broadcast notes sync event."""
    if socketio is not None:
        socketio.emit("notes_updated", {"device_id": device_id}, room="all")


def broadcast_celebrate(message: str = "Connection Successful!"):
    """
    Broadcast celebration event to all web clients.
    Shows confetti and plays sound in the web dashboard.
    Called when Android app successfully connects via SSH test.
    """
    if socketio is not None:
        socketio.emit("celebrate", {"message": message}, room="all")


def broadcast_sessions_updated(device_id):
    """Broadcast sessions sync event."""
    if socketio is not None:
        socketio.emit("sessions_updated", {"device_id": device_id}, room="all")


def broadcast_session_message(session_id, message, is_update=False):
    """Broadcast a session message (supports streaming updates)."""
    if socketio is not None:
        socketio.emit(
            "session_message",
            {
                "session_id": session_id,
                "message": message,
                "is_update": bool(is_update),
            },
            room="all",
        )


def broadcast_automation_status(automation_id, status, result=None):
    """Broadcast automation status change."""
    if socketio is not None:
        socketio.emit(
            "automation_status",
            {"automation_id": automation_id, "status": status, "result": result},
            room="all",
        )

def broadcast_agent_status(agent_id, status, task=None):
    """Broadcast agent status change."""
    if socketio is not None:
        socketio.emit(
            "agent_status",
            {"device_id": agent_id, "status": status, "task": task},
            room="all",
        )


def broadcast_activity(event_type, resource_type, resource_id, resource_title, metadata=None):
    """Broadcast a user activity event."""
    if socketio is not None:
        socketio.emit(
            "activity_event",
            {
                "event_type": event_type,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "resource_title": resource_title,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
            room="all",
        )


def broadcast_cards_updated(device_id):
    """Broadcast cards sync event."""
    if socketio is not None:
        socketio.emit("cards_updated", {"device_id": device_id}, room="all")


def broadcast_collections_updated(device_id):
    """Broadcast collections sync event."""
    if socketio is not None:
        socketio.emit("collections_updated", {"device_id": device_id}, room="all")


# =============================================================================
# Presence Broadcast Functions
# =============================================================================

def broadcast_presence_change(user_id: str, status: str, user_info: dict = None):
    """Broadcast user presence change to all connected clients."""
    if socketio is not None:
        socketio.emit(
            "presence_change",
            {
                "user_id": user_id,
                "status": status,
                "user": user_info,
                "timestamp": datetime.utcnow().isoformat(),
            },
            room="all",
        )


def broadcast_typing_indicator(session_id: str, user_id: str, user_info: dict = None, is_typing: bool = True):
    """Broadcast typing indicator for a session."""
    if socketio is not None:
        socketio.emit(
            "typing_indicator",
            {
                "session_id": session_id,
                "user_id": user_id,
                "user": user_info,
                "is_typing": is_typing,
                "timestamp": datetime.utcnow().isoformat(),
            },
            room=f"session:{session_id}",
        )


def broadcast_session_join(session_id: str, user_id: str, user_info: dict = None):
    """Broadcast when a user joins a shared session."""
    if socketio is not None:
        socketio.emit(
            "session_user_joined",
            {
                "session_id": session_id,
                "user_id": user_id,
                "user": user_info,
                "timestamp": datetime.utcnow().isoformat(),
            },
            room=f"session:{session_id}",
        )


def broadcast_session_leave(session_id: str, user_id: str, user_info: dict = None):
    """Broadcast when a user leaves a shared session."""
    if socketio is not None:
        socketio.emit(
            "session_user_left",
            {
                "session_id": session_id,
                "user_id": user_id,
                "user": user_info,
                "timestamp": datetime.utcnow().isoformat(),
            },
            room=f"session:{session_id}",
        )


def broadcast_to_user(user_id: str, event: str, data: dict):
    """Broadcast an event to a specific user (all their connected devices)."""
    if socketio is not None:
        socketio.emit(event, data, room=f"user:{user_id}")


def get_online_users() -> list:
    """Get list of currently online users."""
    online = []
    for user_id, presence in user_presence.items():
        if presence.get("status") != PRESENCE_OFFLINE:
            online.append({
                "user_id": user_id,
                **presence,
            })
    return online


def get_session_participants(session_id: str) -> list:
    """Get list of users currently in a session."""
    participants = []
    for client in connected_clients.values():
        if client.get("user_id") and f"session:{session_id}" in getattr(client, "rooms", []):
            participants.append({
                "user_id": client["user_id"],
                "user": client.get("user"),
            })
    return participants
