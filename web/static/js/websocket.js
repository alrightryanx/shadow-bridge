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
        this.handlerWarnings = new Set();
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
            if (typeof markSessionsSynced === 'function') markSessionsSynced();
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

        this.socket.on('activity_event', (data) => {
            this.trigger('activity_event', data);
            if (typeof handleActivityEvent === 'function') handleActivityEvent(data);
        });

        this.socket.on('cards_updated', (data) => {
            this.trigger('cards_updated', data);
            if (typeof refreshData === 'function') {
                refreshData();
            } else if (typeof loadDashboard === 'function') {
                loadDashboard();
            }
        });

        this.socket.on('collections_updated', (data) => {
            this.trigger('collections_updated', data);
            if (typeof refreshData === 'function') {
                refreshData();
            } else if (typeof loadDashboard === 'function') {
                loadDashboard();
            }
        });

        // Celebration event - connection success from Android app
        this.socket.on('celebrate', (data) => {
            this.trigger('celebrate', data);
            this.showCelebration(data.message || 'Connection Successful!');
        });
    }

    showCelebration(message) {
        // Play sound
        this.playSuccessSound();

        // Show confetti
        this.showConfetti();

        // Show toast notification
        this.showNotification(message, 'success');
    }

    playSuccessSound() {
        try {
            // Create a pleasant success sound using Web Audio API
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioCtx.destination);

            oscillator.frequency.setValueAtTime(523.25, audioCtx.currentTime); // C5
            oscillator.frequency.setValueAtTime(659.25, audioCtx.currentTime + 0.1); // E5
            oscillator.frequency.setValueAtTime(783.99, audioCtx.currentTime + 0.2); // G5

            gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.4);

            oscillator.start(audioCtx.currentTime);
            oscillator.stop(audioCtx.currentTime + 0.4);
        } catch (e) {
            console.warn('Could not play success sound:', e);
        }
    }

    showConfetti() {
        // Create confetti container
        const container = document.createElement('div');
        container.id = 'confetti-container';
        container.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:9999;overflow:hidden;';
        document.body.appendChild(container);

        const colors = ['#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50', '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722'];

        // Create confetti pieces
        for (let i = 0; i < 150; i++) {
            const confetti = document.createElement('div');
            const color = colors[Math.floor(Math.random() * colors.length)];
            const size = Math.random() * 10 + 5;
            const left = Math.random() * 100;
            const delay = Math.random() * 3;
            const duration = Math.random() * 3 + 2;

            confetti.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                background: ${color};
                left: ${left}%;
                top: -20px;
                animation: confetti-fall ${duration}s ease-out ${delay}s forwards;
                transform: rotate(${Math.random() * 360}deg);
            `;
            container.appendChild(confetti);
        }

        // Add animation keyframes if not exists
        if (!document.getElementById('confetti-styles')) {
            const style = document.createElement('style');
            style.id = 'confetti-styles';
            style.textContent = `
                @keyframes confetti-fall {
                    0% { transform: translateY(0) rotate(0deg); opacity: 1; }
                    100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }

        // Remove confetti after animation
        setTimeout(() => {
            container.remove();
        }, 6000);
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
        const handlers = this.handlers[event];
        if (!handlers || handlers.length === 0) {
            if (!this.handlerWarnings.has(event)) {
                this.handlerWarnings.add(event);
                console.warn('No handlers registered for WebSocket event:', event);
                this.reportIssue('missing_handler', {
                    event,
                    detail: 'No handlers registered for event'
                });
            }
            return;
        }

        handlers.forEach(handler => {
            try {
                handler(data);
            } catch (error) {
                console.error('WebSocket handler failed:', event, error);
                this.reportIssue('handler_error', {
                    event,
                    message: error.message,
                    stack: error.stack
                });
            }
        });
    }

    reportIssue(kind, metadata) {
        if (typeof api === 'undefined' || typeof api.trackActivity !== 'function') {
            return;
        }
        const payload = metadata || {};
        const eventId = payload.event || kind;
        api.trackActivity(
            `ws_${kind}`,
            'websocket',
            eventId,
            'WebSocket diagnostic',
            payload
        ).catch(() => {});
    }

    updateStatusIndicator(connected) {
        // Update both status dot and text for websocket connection state
        const dot = document.getElementById('device-status-dot');
        const text = document.getElementById('device-switcher-status');
        if (!dot || !text) return;

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
        if (typeof initDeviceSelector === 'function') initDeviceSelector();
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
