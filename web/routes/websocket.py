"""
WebSocket handlers for real-time updates
"""
from flask import request

# Client tracking
connected_clients = set()

# Import socketio - may be None in frozen builds
from ..app import socketio

# Only register handlers if socketio is available
if socketio is not None:
    from flask_socketio import emit, join_room, leave_room

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        connected_clients.add(request.sid if hasattr(request, 'sid') else 'unknown')
        emit('status', {'connected': True, 'clients': len(connected_clients)})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        connected_clients.discard(request.sid if hasattr(request, 'sid') else 'unknown')

    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Subscribe to specific update channels."""
        channel = data.get('channel', 'all')
        join_room(channel)
        emit('subscribed', {'channel': channel})

    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Unsubscribe from update channels."""
        channel = data.get('channel', 'all')
        leave_room(channel)


# Functions to broadcast updates (called from data service or main app)
# These gracefully handle socketio being None

def broadcast_device_connected(device_id, device_name):
    """Broadcast device connection event."""
    if socketio is not None:
        socketio.emit('device_connected', {
            'device_id': device_id,
            'device_name': device_name
        }, room='all')


def broadcast_device_disconnected(device_id):
    """Broadcast device disconnection event."""
    if socketio is not None:
        socketio.emit('device_disconnected', {
            'device_id': device_id
        }, room='all')


def broadcast_projects_updated(device_id):
    """Broadcast projects sync event."""
    if socketio is not None:
        socketio.emit('projects_updated', {
            'device_id': device_id
        }, room='all')


def broadcast_notes_updated(device_id):
    """Broadcast notes sync event."""
    if socketio is not None:
        socketio.emit('notes_updated', {
            'device_id': device_id
        }, room='all')


def broadcast_automation_status(automation_id, status, result=None):
    """Broadcast automation status change."""
    if socketio is not None:
        socketio.emit('automation_status', {
            'automation_id': automation_id,
            'status': status,
            'result': result
        }, room='all')


def broadcast_agent_status(agent_id, status, task=None):
    """Broadcast agent status change."""
    if socketio is not None:
        socketio.emit('agent_status', {
            'agent_id': agent_id,
            'status': status,
            'task': task
        }, room='all')
