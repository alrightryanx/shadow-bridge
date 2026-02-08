"""WebSocket events and broadcast functions for real-time updates."""

import logging
import time
from flask import Blueprint
from flask_socketio import emit

log = logging.getLogger(__name__)

ws_bp = Blueprint('websocket', __name__)

# Flag to track if SocketIO handlers have been registered
_handlers_registered = False


def _get_socketio():
    """Get the SocketIO instance. Import lazily to avoid circular imports."""
    from web.app import socketio
    return socketio


def register_socketio_handlers(sio=None):
    """Register SocketIO event handlers. Called after app creation."""
    global _handlers_registered
    if _handlers_registered:
        return
    _handlers_registered = True

    if sio is None:
        sio = _get_socketio()

    @sio.on('connect')
    def handle_connect():
        log.debug("WebSocket client connected")
        emit('connected', {'status': 'ok', 'timestamp': time.time()})

    @sio.on('disconnect')
    def handle_disconnect():
        log.debug("WebSocket client disconnected")

    @sio.on('subscribe_tasks')
    def handle_subscribe_tasks(data=None):
        """Client subscribes to task updates. Optional filter in data."""
        log.debug(f"Client subscribed to task updates: {data}")
        emit('subscribed', {'channel': 'tasks', 'timestamp': time.time()})

    @sio.on('heartbeat')
    def handle_heartbeat(data=None):
        """Keep-alive heartbeat."""
        emit('heartbeat_ack', {'timestamp': time.time()})

    log.debug("SocketIO event handlers registered")


# ---- Broadcast functions (imported by shadow_bridge_gui.py) ----

def broadcast_agents_updated(device_id=None):
    """Broadcast that agents have been updated."""
    try:
        sio = _get_socketio()
        sio.emit('agents_updated', {
            'device_id': device_id,
            'timestamp': time.time(),
        })
        log.debug(f"Broadcast agents_updated for device {device_id}")
    except Exception as e:
        log.debug(f"agents_updated broadcast failed: {e}")


def broadcast_projects_updated(device_id=None):
    """Broadcast that projects have been updated."""
    try:
        sio = _get_socketio()
        sio.emit('projects_updated', {
            'device_id': device_id,
            'timestamp': time.time(),
        })
        log.debug(f"Broadcast projects_updated for device {device_id}")
    except Exception as e:
        log.debug(f"projects_updated broadcast failed: {e}")


def broadcast_notes_updated(device_id=None):
    """Broadcast that notes have been updated."""
    try:
        sio = _get_socketio()
        sio.emit('notes_updated', {
            'device_id': device_id,
            'timestamp': time.time(),
        })
        log.debug(f"Broadcast notes_updated for device {device_id}")
    except Exception as e:
        log.debug(f"notes_updated broadcast failed: {e}")


def broadcast_celebrate(message=""):
    """Broadcast a celebration event to web dashboard."""
    try:
        sio = _get_socketio()
        sio.emit('celebrate', {
            'message': message,
            'timestamp': time.time(),
        })
        log.debug(f"Broadcast celebrate: {message}")
    except Exception as e:
        log.debug(f"celebrate broadcast failed: {e}")


def broadcast_task_update(task_id, action, task_data):
    """Broadcast a task status change."""
    try:
        sio = _get_socketio()
        sio.emit('task_update', {
            'task_id': task_id,
            'action': action,
            'task': task_data,
            'timestamp': time.time(),
        })
    except Exception as e:
        log.debug(f"task_update broadcast failed: {e}")


def broadcast_agent_event(task_id, event):
    """Broadcast a real-time agent execution event."""
    try:
        sio = _get_socketio()
        sio.emit('agent_event', {
            'task_id': task_id,
            'event': event,
            'timestamp': time.time(),
        })
    except Exception as e:
        log.debug(f"agent_event broadcast failed: {e}")
