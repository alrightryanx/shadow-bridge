"""WebSocket events and broadcast functions for real-time updates."""

import logging
import threading
import time
from flask import Blueprint, request as flask_request
from flask_socketio import emit

log = logging.getLogger(__name__)

ws_bp = Blueprint('websocket', __name__)

# Flag to track if SocketIO handlers have been registered
_handlers_registered = False

# ---- Client Tracking ----

_connected_clients = {}   # sid -> {connected_at, last_heartbeat, heartbeat_count, subscriptions}
_clients_lock = threading.Lock()


def _track_client(sid):
    """Record a new client connection."""
    with _clients_lock:
        _connected_clients[sid] = {
            'connected_at': time.time(),
            'last_heartbeat': time.time(),
            'heartbeat_count': 0,
            'subscriptions': [],
        }


def _untrack_client(sid):
    """Remove a client and return its connection duration."""
    with _clients_lock:
        info = _connected_clients.pop(sid, None)
    if info:
        return time.time() - info['connected_at']
    return 0.0


def _record_heartbeat(sid):
    """Update last_heartbeat timestamp and increment count."""
    with _clients_lock:
        if sid in _connected_clients:
            _connected_clients[sid]['last_heartbeat'] = time.time()
            _connected_clients[sid]['heartbeat_count'] += 1


def _record_subscription(sid, channel):
    """Record a channel subscription for a client."""
    with _clients_lock:
        if sid in _connected_clients:
            subs = _connected_clients[sid]['subscriptions']
            if channel not in subs:
                subs.append(channel)


def get_stale_clients(timeout_seconds=90):
    """Return list of SIDs that haven't sent a heartbeat within timeout."""
    cutoff = time.time() - timeout_seconds
    stale = []
    with _clients_lock:
        for sid, info in _connected_clients.items():
            if info['last_heartbeat'] < cutoff:
                stale.append(sid)
    return stale


def get_connected_clients_summary():
    """Return a list of client info dicts for the API/dashboard."""
    now = time.time()
    result = []
    with _clients_lock:
        for sid, info in _connected_clients.items():
            result.append({
                'sid': sid,
                'connected_at': info['connected_at'],
                'duration_seconds': round(now - info['connected_at'], 1),
                'last_heartbeat': info['last_heartbeat'],
                'seconds_since_heartbeat': round(now - info['last_heartbeat'], 1),
                'heartbeat_count': info['heartbeat_count'],
                'subscriptions': list(info['subscriptions']),
            })
    return result


def get_connected_client_count():
    """Return the number of currently tracked clients."""
    with _clients_lock:
        return len(_connected_clients)


# ---- SocketIO Handler Registration ----

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
        sid = flask_request.sid
        log.info(f"WS CONNECT: sid={sid}")
        _track_client(sid)
        emit('connected', {'status': 'ok', 'timestamp': time.time()})

    @sio.on('disconnect')
    def handle_disconnect():
        sid = flask_request.sid
        dur = _untrack_client(sid)
        log.info(f"WS DISCONNECT: sid={sid}, duration={dur:.1f}s")

    @sio.on('subscribe_tasks')
    def handle_subscribe_tasks(data=None):
        """Client subscribes to task updates. Optional filter in data."""
        sid = flask_request.sid
        log.info(f"WS SUBSCRIBE: sid={sid}, data={data}")
        _record_subscription(sid, 'tasks')
        emit('subscribed', {'channel': 'tasks', 'timestamp': time.time()})

    @sio.on('heartbeat')
    def handle_heartbeat(data=None):
        """Keep-alive heartbeat."""
        sid = flask_request.sid
        log.debug(f"WS HEARTBEAT: sid={sid}")
        _record_heartbeat(sid)
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
