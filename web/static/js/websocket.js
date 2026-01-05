/**
 * Shadow Web Dashboard - WebSocket Client
 */

class ShadowWebSocket {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.handlers = {};
    }

    connect() {
        // Use socket.io client
        if (typeof io === 'undefined') {
            console.warn('Socket.IO not loaded, real-time updates disabled');
            return;
        }

        this.socket = io({
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: this.maxReconnectAttempts,
            reconnectionDelay: this.reconnectDelay
        });

        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.updateStatusIndicator(true);
            this.subscribe('all');
        });

        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.connected = false;
            this.updateStatusIndicator(false);
        });

        this.socket.on('connect_error', (error) => {
            console.warn('WebSocket connection error:', error);
            this.reconnectAttempts++;
        });

        // Event handlers
        this.socket.on('device_connected', (data) => {
            this.trigger('device_connected', data);
            this.showNotification(`Device connected: ${data.device_name}`);
            this.refreshCurrentPage();
        });

        this.socket.on('device_disconnected', (data) => {
            this.trigger('device_disconnected', data);
            this.showNotification(`Device disconnected: ${data.device_id}`);
            this.refreshCurrentPage();
        });

        this.socket.on('projects_updated', (data) => {
            this.trigger('projects_updated', data);
            if (typeof loadProjects === 'function') loadProjects();
            if (typeof loadDashboard === 'function') loadDashboard();
        });

        this.socket.on('notes_updated', (data) => {
            this.trigger('notes_updated', data);
            if (typeof loadNotes === 'function') loadNotes();
            if (typeof loadDashboard === 'function') loadDashboard();
        });

        this.socket.on('sessions_updated', (data) => {
            this.trigger('sessions_updated', data);
            if (typeof loadRecentSessions === 'function') loadRecentSessions(getDeviceIdParam());
            if (typeof loadSessionsPanel === 'function') loadSessionsPanel(getDeviceIdParam(), true);
        });

        this.socket.on('session_message', (data) => {
            this.trigger('session_message', data);
            if (typeof handleSessionMessage === 'function') handleSessionMessage(data);
        });

        this.socket.on('automation_status', (data) => {
            this.trigger('automation_status', data);
            if (typeof loadAutomations === 'function') loadAutomations();
        });

        this.socket.on('agent_status', (data) => {
            this.trigger('agent_status', data);
            if (typeof loadAgents === 'function') loadAgents();
        });
    }

    subscribe(channel) {
        if (this.socket && this.connected) {
            this.socket.emit('subscribe', { channel: channel });
        }
    }

    unsubscribe(channel) {
        if (this.socket && this.connected) {
            this.socket.emit('unsubscribe', { channel: channel });
        }
    }

    on(event, handler) {
        if (!this.handlers[event]) {
            this.handlers[event] = [];
        }
        this.handlers[event].push(handler);
    }

    trigger(event, data) {
        if (this.handlers[event]) {
            this.handlers[event].forEach(handler => handler(data));
        }
    }

    updateStatusIndicator(connected) {
        // Update both status dot and text for websocket connection state
        const indicator = document.getElementById('status-indicator');
        if (!indicator) return;

        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');

        if (connected) {
            // Show connecting state briefly, then let updateStatus() determine final state
            dot.className = 'status-dot connecting';
            text.textContent = 'Connecting...';

            // Trigger full status refresh after short delay to get actual device state
            setTimeout(() => {
                if (typeof updateStatus === 'function') {
                    updateStatus();
                }
            }, 500);
        } else {
            // WebSocket disconnected - show as offline
            dot.className = 'status-dot offline';
            text.textContent = 'Disconnected';

            // Schedule a status refresh to check if API is still reachable via HTTP polling
            setTimeout(() => {
                if (typeof updateStatus === 'function') {
                    updateStatus();
                }
            }, 2000);
        }
    }

    showNotification(message, type = 'info') {
        // Use toast notification system
        if (typeof showToast === 'function') {
            showToast(message, type);
        } else {
            console.log('Notification:', message);
        }
    }

    refreshCurrentPage() {
        // Refresh data based on current page
        if (typeof loadDashboard === 'function') loadDashboard();
        if (typeof loadProjects === 'function') loadProjects();
        if (typeof loadNotes === 'function') loadNotes();
        if (typeof loadAutomations === 'function') loadAutomations();
        if (typeof loadAgents === 'function') loadAgents();
        if (typeof loadStatus === 'function') loadStatus();
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
        }
    }
}

// Global WebSocket instance
const shadowWS = new ShadowWebSocket();

// Auto-connect when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Delay connection slightly to let socket.io load
    setTimeout(() => {
        shadowWS.connect();
    }, 500);
});
