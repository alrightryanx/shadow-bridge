/**
 * Agent Orchestration WebSocket Client
 *
 * Real-time monitoring and control of persistent AI agents.
 * Connects to /ws/agents endpoint for live updates.
 */

class AgentOrchestrator {
    constructor() {
        this.socket = null;
        this.agents = new Map();
        this.callbacks = {
            'agent_spawned': [],
            'agent_stopped': [],
            'agent_status_changed': [],
            'agent_output_line': [],
            'agent_task_completed': [],
            'agent_task_blocked': [],
            'agent_task_started': [],
            'agent_task_progress': [],
            'agent_task_result': []
        };

        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        if (this.socket && this.connected) {
            console.log('Already connected to agent orchestrator');
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/socket.io/`;

        console.log('Connecting to agent orchestrator:', wsUrl);

        // Use Socket.IO client (should be loaded in base template)
        if (typeof io === 'undefined') {
            console.error('Socket.IO client not loaded!');
            return;
        }

        this.socket = io();

        // Connection events
        this.socket.on('connect', () => {
            console.log('Connected to agent orchestrator');
            this.connected = true;
            this.reconnectAttempts = 0;
            updateConnectionStatus('connected');

            // Request current agents list
            this.getAgents();
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from agent orchestrator');
            this.connected = false;
            updateConnectionStatus('disconnected');
            this.attemptReconnect();
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            updateConnectionStatus('error');
            this.attemptReconnect();
        });

        // Agent events
        this.socket.on('agent_spawned', (data) => {
            console.log('Agent spawned:', data);
            this.agents.set(data.id, data);
            this.trigger('agent_spawned', data);
            this.updateAgentUI();
        });

        this.socket.on('agent_stopped', (data) => {
            console.log('Agent stopped:', data);
            this.agents.delete(data.id);
            this.trigger('agent_stopped', data);
            this.updateAgentUI();
        });

        this.socket.on('agent_status_changed', (data) => {
            console.log('Agent status changed:', data);
            const existing = this.agents.get(data.id);
            if (existing) {
                this.agents.set(data.id, {...existing, ...data});
            }
            this.trigger('agent_status_changed', data);
            this.updateAgentCard(data);
        });

        this.socket.on('agent_output_line', (data) => {
            console.log('Agent output:', data.line);
            this.trigger('agent_output_line', data);
            this.appendOutputLine(data.agent_id, data.line, data.stream);
        });

        this.socket.on('agent_task_completed', (data) => {
            console.log('Agent task completed:', data);
            this.trigger('agent_task_completed', data);
        });

        this.socket.on('agent_task_started', (data) => {
            const existing = this.agents.get(data.agent_id);
            if (existing) {
                this.agents.set(data.agent_id, {
                    ...existing,
                    current_task_id: data.task_id || existing.current_task_id,
                    current_thread_id: data.thread_id || existing.current_thread_id,
                    task_progress_pct: 0,
                    task_progress_msg: 'Started'
                });
                this.updateAgentCard({ id: data.agent_id, task_progress_pct: 0, task_progress_msg: 'Started' });
            }
            this.trigger('agent_task_started', data);
        });

        this.socket.on('agent_task_progress', (data) => {
            const existing = this.agents.get(data.agent_id);
            if (existing) {
                this.agents.set(data.agent_id, {
                    ...existing,
                    task_progress_pct: data.pct,
                    task_progress_msg: data.msg || existing.task_progress_msg
                });
                this.updateAgentCard({ id: data.agent_id, task_progress_pct: data.pct, task_progress_msg: data.msg });
            }
            this.trigger('agent_task_progress', data);
        });

        this.socket.on('agent_task_result', (data) => {
            const existing = this.agents.get(data.agent_id);
            if (existing) {
                this.agents.set(data.agent_id, {
                    ...existing,
                    task_result: data.result
                });
                this.updateAgentCard({ id: data.agent_id, task_result: data.result });
            }
            this.trigger('agent_task_result', data);
        });

        this.socket.on('agent_task_blocked', (data) => {
            console.warn('Agent task blocked:', data);
            const existing = this.agents.get(data.agent_id);
            if (existing) {
                this.agents.set(data.agent_id, {
                    ...existing,
                    last_blocked_reason: data.reason,
                    last_blocked_at: data.timestamp
                });
                this.updateAgentCard({ id: data.agent_id, last_blocked_reason: data.reason, last_blocked_at: data.timestamp });
            }
            this.trigger('agent_task_blocked', data);
            this.showToast(`Task blocked: ${data.reason}`, 'warning');
            this.fetchLocks();
        });

        this.socket.on('agent_locks_released', (data) => {
            console.log('Locks released:', data);
            this.fetchLocks();
            if (data.agent_id) {
                this.showToast(`Locks released for ${data.agent_id}`, 'success');
            } else if (data.scope === 'all') {
                this.showToast('All locks released', 'success');
            }
        });

        // Response events
        this.socket.on('agents_list', (data) => {
            console.log('Received agents list:', data);
            this.agents.clear();
            (data.agents || []).forEach(agent => {
                this.agents.set(agent.id, agent);
            });
            this.updateAgentUI();
        });

        this.socket.on('agent_status', (data) => {
            console.log('Received agent status:', data);
            const existing = this.agents.get(data.id);
            if (existing) {
                this.agents.set(data.id, {...existing, ...data});
            }
            this.updateAgentCard(data);
        });

        // Android device agent sync events
        this.socket.on('agents_updated', (data) => {
            console.log('Agents updated from device:', data.device_id);
            // Refresh the full agent list to include device-synced agents
            this.getAgents(data.device_id);
            // Also fetch from REST API for merged view (SQLite + JSON + orchestrator)
            this.fetchMergedAgents();
        });

        this.socket.on('error', (data) => {
            console.error('Agent orchestrator error:', data.message);
            this.showToast(data.message, 'error');
        });
    }

    /**
     * Attempt to reconnect after disconnect
     */
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnect attempts reached');
            this.showToast('Lost connection to agent orchestrator', 'error');
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => {
            if (!this.connected) {
                this.connect();
            }
        }, delay);
    }

    /**
     * Spawn a new agent
     */
    spawnAgent(config) {
        if (!this.socket || !this.connected) {
            this.showToast('Not connected to server', 'error');
            return;
        }

        console.log('Spawning agent:', config);

        this.socket.emit('spawn_agent', {
            name: config.name,
            specialty: config.specialty,
            provider: config.provider || 'claude',
            model: config.model || 'claude-sonnet-4-20250514',
            working_directory: config.workingDirectory,
            auto_accept_edits: config.autoAcceptEdits !== false
        });

        this.showToast(`Spawning ${config.name}...`, 'info');
    }

    /**
     * Stop an agent
     */
    stopAgent(agentId, graceful = true) {
        if (!this.socket || !this.connected) {
            this.showToast('Not connected to server', 'error');
            return;
        }

        console.log('Stopping agent:', agentId);

        this.socket.emit('stop_agent', {
            agent_id: agentId,
            graceful: graceful
        });

        this.showToast('Stopping agent...', 'info');
    }

    /**
     * Assign a task to an agent
     */
    assignTask(agentId, task) {
        if (!this.socket || !this.connected) {
            this.showToast('Not connected to server', 'error');
            return;
        }

        console.log('Assigning task to agent:', agentId, task);

        this.socket.emit('assign_task', {
            agent_id: agentId,
            task: task
        });

        this.showToast('Task assigned', 'success');
    }

    /**
     * Get all agents
     */
    getAgents(deviceId = null) {
        if (!this.socket || !this.connected) {
            console.warn('Cannot get agents: not connected');
            return;
        }

        this.socket.emit('get_agents', {
            device_id: deviceId
        });
    }

    /**
     * Get status of a specific agent
     */
    getAgentStatus(agentId) {
        if (!this.socket || !this.connected) {
            console.warn('Cannot get agent status: not connected');
            return;
        }

        this.socket.emit('get_agent_status', {
            agent_id: agentId
        });
    }

    /**
     * Fetch merged agent list from REST API (includes all sources: SQLite, JSON, orchestrator).
     * Complements WebSocket-based updates for a complete picture.
     */
    async fetchMergedAgents() {
        try {
            const response = await fetch('/api/agents');
            if (!response.ok) return;
            const agents = await response.json();
            if (Array.isArray(agents)) {
                // Merge with existing orchestrator agents (live agents take precedence)
                const liveIds = new Set(Array.from(this.agents.keys()));
                agents.forEach(agent => {
                    if (!liveIds.has(agent.id)) {
                        this.agents.set(agent.id, agent);
                    }
                });
                this.updateAgentUI();
                this.updateOverviewStats();
            }
        } catch (e) {
            console.debug('fetchMergedAgents failed:', e.message);
        }
    }

    /**
     * Update overview stats cards with current agent data.
     */
    updateOverviewStats() {
        const agents = Array.from(this.agents.values());
        const total = agents.length;
        const activeStatuses = ['busy', 'working', 'active'];
        const active = agents.filter(a => activeStatuses.includes(a.status)).length;
        const tasksCompleted = agents.reduce((sum, a) => sum + (a.tasks_completed || 0), 0);

        const totalEl = document.getElementById('overview-total-agents');
        const activeEl = document.getElementById('overview-active-agents');
        const completedEl = document.getElementById('overview-completed-tasks');

        if (totalEl) totalEl.textContent = total;
        if (activeEl) activeEl.textContent = active;
        if (completedEl) completedEl.textContent = tasksCompleted;
    }

    /**
     * Fetch and render lock status
     */
    async fetchLocks() {
        if (typeof api === 'undefined' || typeof api.getLocks !== 'function') return;
        const snapshot = await api.getLocks();
        this.renderLocks(snapshot || {});
    }

    renderLocks(snapshot) {
        const container = document.getElementById('lock-list');
        if (!container) return;

        const locks = snapshot.locks || [];
        const blocked = snapshot.blocked || [];

        const lockItems = locks.length
            ? locks.map(lock => {
                const title = `${lock.lock_type.toUpperCase()} ${lock.repo || 'unknown'}`;
                const path = lock.path ? ` | ${lock.path}` : '';
                return `
                    <div class="lock-item">
                        <div class="lock-title">${this.escapeHtml(title)}${this.escapeHtml(path)}</div>
                        <div class="lock-meta">Agent: ${this.escapeHtml(lock.agent_id || 'unknown')} • Task: ${this.escapeHtml(lock.task_id || 'n/a')}</div>
                    </div>
                `;
            }).join('')
            : '<div class="empty-state-small">No active locks</div>';

        const blockedItems = blocked.length
            ? blocked.slice(-5).reverse().map(entry => `
                <div class="lock-item blocked">
                    <div class="lock-title">Blocked: ${this.escapeHtml(entry.reason || 'unknown')}</div>
                    <div class="lock-meta">Agent: ${this.escapeHtml(entry.agent_id || 'unknown')} • Task: ${this.escapeHtml(entry.task_id || 'n/a')}</div>
                </div>
            `).join('')
            : '<div class="empty-state-small">No recent blocks</div>';

        container.innerHTML = `
            <div class="lock-group">
                <div class="lock-group-title">Active Locks</div>
                ${lockItems}
            </div>
            <div class="lock-group">
                <div class="lock-group-title">Recent Blocks</div>
                ${blockedItems}
            </div>
        `;
    }

    /**
     * Register event callback
     */
    on(event, callback) {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [];
        }
        this.callbacks[event].push(callback);
    }

    /**
     * Trigger event callbacks
     */
    trigger(event, data) {
        const callbacks = this.callbacks[event] || [];
        callbacks.forEach(callback => {
            try {
                callback(data);
            } catch (e) {
                console.error(`Error in ${event} callback:`, e);
            }
        });
    }

    /**
     * Update agent UI (override this in your app)
     */
    updateAgentUI() {
        console.log('Updating agent UI with', this.agents.size, 'agents');

        // Update agent grid
        const container = document.getElementById('live-agents-grid');
        if (!container) return;

        if (this.agents.size === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <span class="material-symbols-outlined" style="font-size: 48px; color: var(--text-dim);">smart_toy</span>
                    <p>No persistent agents running</p>
                    <button class="btn btn-primary" onclick="showSpawnAgentModal()">
                        <span class="material-symbols-outlined">add</span> Spawn Agent
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = Array.from(this.agents.values()).map(agent => this.renderAgentCard(agent)).join('');
        this.updateOverviewStats();
    }

    /**
     * Render agent card HTML
     */
    renderAgentCard(agent) {
        const statusClass = agent.status || 'idle';
        const statusColors = {
            'idle': 'var(--text-dim)',
            'busy': 'var(--warning)',
            'working': 'var(--warning)',
            'active': 'var(--success, #4caf50)',
            'error': 'var(--error)',
            'offline': 'var(--text-dim)'
        };

        const uptime = this.formatUptime(agent.uptime_seconds);

        return `
            <div class="agent-card live" id="agent-${agent.id}" data-agent-id="${agent.id}">
                <div class="agent-card-header">
                    <div class="agent-card-title">
                        <span class="material-symbols-outlined agent-icon">smart_toy</span>
                        <div>
                            <h3>${this.escapeHtml(agent.name)}</h3>
                            <span class="agent-specialty">${this.escapeHtml(agent.specialty)}</span>
                        </div>
                    </div>
                    <span class="status-badge" style="background: ${statusColors[statusClass]}">
                        ${statusClass.toUpperCase()}
                    </span>
                </div>

                <div class="agent-card-metrics">
                    <div class="metric">
                        <span class="material-symbols-outlined">memory</span>
                        <span>${agent.cpu_percent || 0}% CPU</span>
                    </div>
                    <div class="metric">
                        <span class="material-symbols-outlined">storage</span>
                        <span>${agent.memory_mb || 0} MB</span>
                    </div>
                    <div class="metric">
                        <span class="material-symbols-outlined">schedule</span>
                        <span>${uptime}</span>
                    </div>
                    <div class="metric">
                        <span class="material-symbols-outlined">task_alt</span>
                        <span>${agent.tasks_completed || 0} tasks</span>
                    </div>
                </div>

                ${agent.current_task ? `
                    <div class="agent-current-task">
                        <strong>Current Task:</strong>
                        <div class="task-text">${this.escapeHtml(agent.current_task)}</div>
                    </div>
                ` : ''}

                ${agent.last_blocked_reason ? `
                    <div class="agent-blocked">
                        <strong>Last Blocked:</strong>
                        <div class="blocked-text">${this.escapeHtml(agent.last_blocked_reason)}</div>
                    </div>
                ` : ''}

                <div class="agent-output" id="agent-output-${agent.id}">
                    ${(agent.output_lines || []).map(line =>
                        `<div class="output-line">${this.escapeHtml(line.line)}</div>`
                    ).join('')}
                </div>

                <div class="agent-card-actions">
                    <button class="btn btn-small" onclick="showAssignTaskModal('${agent.id}')">
                        <span class="material-symbols-outlined">assignment</span> Assign Task
                    </button>
                    <button class="btn btn-small btn-secondary" onclick="releaseAgentLocks('${agent.id}')">
                        <span class="material-symbols-outlined">lock_open</span> Release Locks
                    </button>
                    <button class="btn btn-small btn-secondary" onclick="orchestrator.getAgentStatus('${agent.id}')">
                        <span class="material-symbols-outlined">refresh</span>
                    </button>
                    <button class="btn btn-small btn-danger" onclick="confirmStopAgent('${agent.id}')">
                        <span class="material-symbols-outlined">stop_circle</span> Stop
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Update a single agent card
     */
    updateAgentCard(agent) {
        const card = document.getElementById(`agent-${agent.id}`);
        if (!card) {
            // Agent card doesn't exist, refresh full UI
            this.updateAgentUI();
            return;
        }

        // Update status badge
        const statusBadge = card.querySelector('.status-badge');
        if (statusBadge) {
            const statusClass = agent.status || 'idle';
            const statusColors = {
                'idle': 'var(--text-dim)',
                'busy': 'var(--warning)',
                'error': 'var(--error)',
                'offline': 'var(--text-dim)'
            };
            statusBadge.textContent = statusClass.toUpperCase();
            statusBadge.style.background = statusColors[statusClass];
        }

        // Update metrics
        if (agent.cpu_percent !== undefined) {
            const cpuMetric = card.querySelector('.metric:nth-child(1) span:last-child');
            if (cpuMetric) cpuMetric.textContent = `${agent.cpu_percent}% CPU`;
        }

        if (agent.memory_mb !== undefined) {
            const memMetric = card.querySelector('.metric:nth-child(2) span:last-child');
            if (memMetric) memMetric.textContent = `${agent.memory_mb} MB`;
        }

        if (agent.uptime_seconds !== undefined) {
            const uptimeMetric = card.querySelector('.metric:nth-child(3) span:last-child');
            if (uptimeMetric) uptimeMetric.textContent = this.formatUptime(agent.uptime_seconds);
        }

        if (agent.tasks_completed !== undefined) {
            const tasksMetric = card.querySelector('.metric:nth-child(4) span:last-child');
            if (tasksMetric) tasksMetric.textContent = `${agent.tasks_completed} tasks`;
        }

        // Update current task
        const taskSection = card.querySelector('.agent-current-task');
        if (agent.current_task) {
            if (taskSection) {
                taskSection.querySelector('.task-text').textContent = agent.current_task;
            } else {
                // Insert task section
                const metricsSection = card.querySelector('.agent-card-metrics');
                const newTaskSection = document.createElement('div');
                newTaskSection.className = 'agent-current-task';
                newTaskSection.innerHTML = `
                    <strong>Current Task:</strong>
                    <div class="task-text">${this.escapeHtml(agent.current_task)}</div>
                `;
                metricsSection.after(newTaskSection);
            }
        } else if (taskSection) {
            taskSection.remove();
        }

        // Update blocked info
        const blockedSection = card.querySelector('.agent-blocked');
        if (agent.last_blocked_reason) {
            if (blockedSection) {
                const text = blockedSection.querySelector('.blocked-text');
                if (text) text.textContent = agent.last_blocked_reason;
            } else {
                const insertAfter = card.querySelector('.agent-current-task') || card.querySelector('.agent-card-metrics');
                const newBlockedSection = document.createElement('div');
                newBlockedSection.className = 'agent-blocked';
                newBlockedSection.innerHTML = `
                    <strong>Last Blocked:</strong>
                    <div class="blocked-text">${this.escapeHtml(agent.last_blocked_reason)}</div>
                `;
                insertAfter.after(newBlockedSection);
            }
        } else if (blockedSection) {
            blockedSection.remove();
        }
    }

    /**
     * Append output line to agent card
     */
    appendOutputLine(agentId, line, stream = 'stdout') {
        const outputContainer = document.getElementById(`agent-output-${agentId}`);
        if (!outputContainer) return;

        const lineEl = document.createElement('div');
        lineEl.className = `output-line ${stream}`;
        lineEl.textContent = line;

        outputContainer.appendChild(lineEl);

        // Auto-scroll to bottom
        outputContainer.scrollTop = outputContainer.scrollHeight;

        // Limit to 100 lines
        while (outputContainer.children.length > 100) {
            outputContainer.removeChild(outputContainer.firstChild);
        }
    }

    /**
     * Format uptime seconds to human-readable string
     */
    formatUptime(seconds) {
        if (!seconds) return '0s';

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;

        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        // Use global showToast if available, otherwise log
        if (typeof window.showToast === 'function') {
            window.showToast(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Global instance
const orchestrator = new AgentOrchestrator();

// Auto-connect on page load
document.addEventListener('DOMContentLoaded', () => {
    orchestrator.connect();
    orchestrator.fetchLocks();
    // Fetch merged agents (REST) on initial load for complete picture
    orchestrator.fetchMergedAgents();

    // Refresh agent status every 5 seconds
    setInterval(() => {
        if (orchestrator.connected) {
            orchestrator.getAgents();
        }
        // Also refresh merged agents periodically for device-synced agents
        orchestrator.fetchMergedAgents();
    }, 5000);

    // Refresh lock status every 10 seconds
    setInterval(() => {
        orchestrator.fetchLocks();
    }, 10000);
});

// Global functions for UI interactions
function confirmSpawnAgent() {
    const nameInput = document.getElementById('inline-agent-name');
    const projectSelect = document.getElementById('inline-agent-project');
    const modelInput = document.getElementById('inline-agent-model');
    const nameError = document.getElementById('name-error');
    const projectError = document.getElementById('project-error');
    const statusDiv = document.getElementById('spawn-status');
    const spawnBtn = document.getElementById('spawn-agent-btn');
    const btnText = document.getElementById('spawn-btn-text');

    const name = nameInput.value.trim();
    const specialty = document.getElementById('inline-agent-specialty').value;
    const provider = document.getElementById('inline-agent-provider').value;
    const model = modelInput.value.trim();
    const selectedProject = projectSelect.options[projectSelect.selectedIndex];

    // Clear previous errors
    nameError.classList.remove('visible');
    projectError.classList.remove('visible');
    statusDiv.classList.remove('visible', 'success', 'error', 'info');

    // Validate name
    if (!name) {
        nameError.textContent = 'Agent name is required';
        nameError.classList.add('visible');
        nameInput.focus();
        return;
    }

    if (name.length < 3) {
        nameError.textContent = 'Agent name must be at least 3 characters';
        nameError.classList.add('visible');
        nameInput.focus();
        return;
    }

    // Validate project
    if (!projectSelect.value) {
        projectError.textContent = 'Please select a project';
        projectError.classList.add('visible');
        projectSelect.focus();
        return;
    }

    // Get working directory from selected project
    const workingDir = selectedProject.dataset.workingDir;

    if (!workingDir) {
        projectError.textContent = 'Selected project has no working directory';
        projectError.classList.add('visible');
        return;
    }

    // Show loading state
    spawnBtn.disabled = true;
    spawnBtn.classList.add('loading');
    btnText.textContent = 'Spawning...';
    statusDiv.textContent = `Spawning ${name}...`;
    statusDiv.className = 'form-status visible info';

    try {
        orchestrator.spawnAgent({
            name: name,
            specialty: specialty,
            provider: provider,
            model: model || 'claude-sonnet-4-20250514',
            workingDirectory: workingDir,
            autoAcceptEdits: true  // Always true for agents
        });

        // Show success message
        setTimeout(() => {
            statusDiv.textContent = `✓ ${name} spawned successfully!`;
            statusDiv.className = 'form-status visible success';

            // Clear form after 2 seconds
            setTimeout(() => {
                clearSpawnForm();
                statusDiv.classList.remove('visible');
            }, 2000);
        }, 500);

    } catch (error) {
        console.error('Failed to spawn agent:', error);
        statusDiv.textContent = `✗ Failed to spawn agent: ${error.message || 'Unknown error'}`;
        statusDiv.className = 'form-status visible error';
        spawnBtn.disabled = false;
        spawnBtn.classList.remove('loading');
        btnText.textContent = 'Spawn Agent';
    }
}

function clearSpawnForm() {
    document.getElementById('inline-agent-name').value = '';
    document.getElementById('inline-agent-specialty').value = 'general';
    document.getElementById('inline-agent-provider').value = 'claude';
    document.getElementById('inline-agent-model').value = 'claude-sonnet-4-20250514';
    document.getElementById('inline-agent-project').value = '';

    const spawnBtn = document.getElementById('spawn-agent-btn');
    const btnText = document.getElementById('spawn-btn-text');
    spawnBtn.disabled = false;
    spawnBtn.classList.remove('loading');
    btnText.textContent = 'Spawn Agent';
}

function updateConnectionStatus(status) {
    const statusDot = document.getElementById('ws-status-dot');
    const statusText = document.getElementById('ws-status-text');

    if (!statusDot || !statusText) return;

    statusDot.className = 'status-dot';
    statusText.className = 'status-text';

    switch (status) {
        case 'connected':
            statusDot.classList.add('connected');
            statusText.classList.add('connected');
            statusText.textContent = 'Connected';
            break;
        case 'disconnected':
            statusDot.classList.add('disconnected');
            statusText.classList.add('disconnected');
            statusText.textContent = 'Disconnected';
            break;
        case 'error':
            statusDot.classList.add('disconnected');
            statusText.classList.add('disconnected');
            statusText.textContent = 'Connection Error';
            break;
        default:
            statusText.textContent = 'Connecting...';
    }
}

function showAssignTaskModal(agentId) {
    const modal = document.getElementById('assign-task-modal');
    if (!modal) {
        console.error('Assign task modal not found');
        return;
    }

    // Store agent ID in modal
    modal.dataset.agentId = agentId;

    modal.classList.remove('hidden');
}

function closeAssignTaskModal() {
    const modal = document.getElementById('assign-task-modal');
    if (modal) {
        modal.classList.add('hidden');
        delete modal.dataset.agentId;
    }
}

function confirmAssignTask() {
    const modal = document.getElementById('assign-task-modal');
    const agentId = modal?.dataset.agentId;
    const task = document.getElementById('assign-task-input').value.trim();

    if (!agentId || !task) {
        orchestrator.showToast('Task description is required', 'warning');
        return;
    }

    orchestrator.assignTask(agentId, task);
    closeAssignTaskModal();

    // Clear input
    document.getElementById('assign-task-input').value = '';
}

function confirmStopAgent(agentId) {
    if (!confirm('Stop this agent? Any running tasks will be interrupted.')) {
        return;
    }

    orchestrator.stopAgent(agentId, true);
}

async function releaseAgentLocks(agentId) {
    if (!confirm('Release all locks held by this agent? This will not stop running processes.')) {
        return;
    }

    if (typeof api === 'undefined' || typeof api.releaseLocks !== 'function') {
        orchestrator.showToast('API not available', 'error');
        return;
    }

    const result = await api.releaseLocks({ agent_id: agentId });
    if (result && result.success) {
        orchestrator.showToast(`Released ${result.released || 0} locks`, 'success');
        orchestrator.fetchLocks();
    } else {
        orchestrator.showToast(result.error || 'Failed to release locks', 'error');
    }
}
