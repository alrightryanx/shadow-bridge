/**
 * ShadowAI Web Dashboard - API Client
 * With offline caching and retry logic
 */

const api = {
    baseUrl: '/api',
    cache: new Map(),
    cacheExpiry: 60000, // 1 minute cache
    retryAttempts: 3,
    retryDelay: 1000,
    isOnline: true,

    // Offline cache storage
    getOfflineCache(key) {
        try {
            const cached = localStorage.getItem('shadow_cache_' + key);
            if (cached) {
                const data = JSON.parse(cached);
                // Check expiry (24 hours for offline cache)
                if (Date.now() - data.timestamp < 86400000) {
                    return data.value;
                }
            }
        } catch (e) {}
        return null;
    },

    setOfflineCache(key, value) {
        try {
            localStorage.setItem('shadow_cache_' + key, JSON.stringify({
                timestamp: Date.now(),
                value: value
            }));
        } catch (e) {}
    },

    async fetch(endpoint, options = {}, useCache = false) {
        const cacheKey = endpoint + JSON.stringify(options.body || '');

        // Check memory cache first
        if (useCache && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheExpiry) {
                return cached.data;
            }
        }

        let lastError = null;
        for (let attempt = 0; attempt < this.retryAttempts; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 10000);

                const response = await fetch(this.baseUrl + endpoint, {
                    ...options,
                    signal: controller.signal,
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    }
                });

                clearTimeout(timeoutId);
                const data = await response.json();

                // Update online status
                this.isOnline = true;

                // Cache successful GET responses
                if (!options.method || options.method === 'GET') {
                    this.cache.set(cacheKey, { data, timestamp: Date.now() });
                    this.setOfflineCache(cacheKey, data);
                }

                return data;
            } catch (error) {
                lastError = error;
                console.warn(`API attempt ${attempt + 1} failed:`, error.message);

                if (attempt < this.retryAttempts - 1) {
                    await new Promise(r => setTimeout(r, this.retryDelay * (attempt + 1)));
                }
            }
        }

        // All retries failed - check offline cache
        console.error('API Error after retries:', lastError);
        this.isOnline = false;

        const offlineData = this.getOfflineCache(cacheKey);
        if (offlineData) {
            console.log('Using offline cache for:', endpoint);
            return { ...offlineData, _offline: true };
        }

        return { error: lastError?.message || 'Connection failed', _offline: true };
    },

    // Devices
    async getDevices() {
        return this.fetch('/devices');
    },

    async getDevice(id) {
        return this.fetch(`/devices/${id}`);
    },

    // Projects
    async getProjects(deviceId = null) {
        const params = deviceId ? `?device_id=${deviceId}` : '';
        return this.fetch(`/projects${params}`);
    },

    async getProject(id) {
        return this.fetch(`/projects/${id}`);
    },

    async openProject(id) {
        return this.fetch(`/projects/${id}/open`, { method: 'POST' });
    },

    async createProject(data) {
        return this.fetch('/projects', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    // Notes
    async getNotes(deviceId = null, search = null) {
        const params = new URLSearchParams();
        if (deviceId) params.append('device_id', deviceId);
        if (search) params.append('search', search);
        const queryString = params.toString();
        return this.fetch(`/notes${queryString ? '?' + queryString : ''}`);
    },

    async getNote(id) {
        return this.fetch(`/notes/${id}`);
    },

    async getNoteContent(id) {
        return this.fetch(`/notes/${id}/content`);
    },

    async updateNoteContent(id, content, title = null) {
        return this.fetch(`/notes/${id}/content`, {
            method: 'PUT',
            body: JSON.stringify({ content, title })
        });
    },

    async exportNote(id) {
        return this.fetch(`/notes/${id}/export`, { method: 'POST' });
    },

    async deleteNote(id) {
        return this.fetch(`/notes/${id}`, { method: 'DELETE' });
    },

    async createNote(data) {
        return this.fetch('/notes', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    // Automations
    async getAutomations(deviceId = null) {
        const params = deviceId ? `?device_id=${deviceId}` : '';
        return this.fetch(`/automations${params}`);
    },

    async getAutomation(id) {
        return this.fetch(`/automations/${id}`);
    },

    async getAutomationLogs(id) {
        return this.fetch(`/automations/${id}/logs`);
    },

    async runAutomation(id) {
        return this.fetch(`/automations/${id}/run`, { method: 'POST' });
    },

    async createAutomation(data) {
        return this.fetch('/automations', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    // Agents
    async getAgents(deviceId = null) {
        const params = deviceId ? `?device_id=${deviceId}` : '';
        return this.fetch(`/agents${params}`);
    },

    async getAgent(id) {
        return this.fetch(`/agents/${id}`);
    },

    async getAgentTasks(id) {
        return this.fetch(`/agents/${id}/tasks`);
    },

    async getAgentMetrics() {
        return this.fetch('/agents/metrics');
    },

    // Analytics
    async getUsageStats() {
        return this.fetch('/analytics/usage');
    },

    async getBackendUsage() {
        return this.fetch('/analytics/backends');
    },

    async getActivityTimeline() {
        return this.fetch('/analytics/activity');
    },

    // Status
    async getStatus() {
        return this.fetch('/status');
    },

    async getQRCode() {
        return this.fetch('/qr');
    },

    // ============ Teams ============
    async getTeams(deviceId = null) {
        const params = deviceId ? `?device_id=${deviceId}` : '';
        return this.fetch(`/teams${params}`, {}, true);
    },

    async getTeam(id) {
        return this.fetch(`/teams/${id}`);
    },

    async createTeam(data) {
        return this.fetch('/teams', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    async updateTeam(id, data) {
        return this.fetch(`/teams/${id}`, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    async deleteTeam(id) {
        return this.fetch(`/teams/${id}`, { method: 'DELETE' });
    },

    async getTeamMetrics() {
        return this.fetch('/teams/metrics', {}, true);
    },

    // ============ Tasks ============
    async getTasks(deviceId = null) {
        const params = deviceId ? `?device_id=${deviceId}` : '';
        return this.fetch(`/tasks${params}`, {}, true);
    },

    async getTask(id) {
        return this.fetch(`/tasks/${id}`);
    },

    async createTask(data) {
        return this.fetch('/tasks', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    async updateTask(id, data) {
        return this.fetch(`/tasks/${id}`, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    async deleteTask(id) {
        return this.fetch(`/tasks/${id}`, { method: 'DELETE' });
    },

    // ============ Agent Management ============
    async addAgent(data) {
        return this.fetch('/agents', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    async updateAgent(id, data) {
        return this.fetch(`/agents/${id}`, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    async deleteAgent(id) {
        return this.fetch(`/agents/${id}`, { method: 'DELETE' });
    },

    // ============ Workflows ============
    async getWorkflows() {
        return this.fetch('/workflows', {}, true);
    },

    async startWorkflow(type, options = {}) {
        return this.fetch('/workflows', {
            method: 'POST',
            body: JSON.stringify({ type, ...options })
        });
    },

    async cancelWorkflow(id) {
        return this.fetch(`/workflows/${id}/cancel`, { method: 'POST' });
    },

    // ============ Audits ============
    async getAudits(period = '7D', deviceId = null) {
        const params = new URLSearchParams({ period });
        if (deviceId) params.append('device_id', deviceId);
        return this.fetch(`/audits?${params}`, {}, true);
    },

    async getAuditEntry(id) {
        return this.fetch(`/audits/${id}`);
    },

    async getAuditStats(period = '7D') {
        return this.fetch(`/audits/stats?period=${period}`, {}, true);
    },

    async getAuditTraces(id) {
        return this.fetch(`/audits/${id}/traces`);
    },

    async exportAuditReport(format = 'json', period = '30D') {
        return this.fetch(`/audits/export?format=${format}&period=${period}`, { method: 'POST' });
    },

    // ============ Favorites ============
    async getFavorites() {
        return this.fetch('/favorites', {}, true);
    },

    async toggleProjectFavorite(id) {
        return this.fetch(`/projects/${id}/favorite`, { method: 'POST' });
    },

    async toggleNoteFavorite(id) {
        return this.fetch(`/notes/${id}/favorite`, { method: 'POST' });
    },

    // ============ Global Search ============
    async search(query, types = ['projects', 'notes', 'automations', 'agents']) {
        const params = new URLSearchParams({ q: query });
        types.forEach(t => params.append('type', t));
        return this.fetch(`/search?${params}`, {}, true);
    },

    // ============ Enhanced Analytics ============
    async getPrivacyScore() {
        return this.fetch('/analytics/privacy', {}, true);
    },

    async getTokenUsage() {
        return this.fetch('/analytics/tokens', {}, true);
    },

    async getCategoryBreakdown(period = '7D') {
        return this.fetch(`/analytics/categories?period=${period}`, {}, true);
    },

    // ============ Project Editing ============
    async updateProject(id, data) {
        return this.fetch(`/projects/${id}`, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    // ============ Project Todos ============
    async getProjectTodos(projectId) {
        return this.fetch(`/projects/${projectId}/todos`);
    },

    async createProjectTodo(projectId, data) {
        return this.fetch(`/projects/${projectId}/todos`, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    async updateProjectTodo(projectId, todoId, data) {
        return this.fetch(`/projects/${projectId}/todos/${todoId}`, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    async deleteProjectTodo(projectId, todoId) {
        return this.fetch(`/projects/${projectId}/todos/${todoId}`, {
            method: 'DELETE'
        });
    },

    async reorderProjectTodos(projectId, todoIds) {
        return this.fetch(`/projects/${projectId}/todos/reorder`, {
            method: 'POST',
            body: JSON.stringify({ todo_ids: todoIds })
        });
    },

    // ============ CLI Launch ============
    async launchCLI(projectId) {
        return this.fetch(`/projects/${projectId}/launch-cli`, {
            method: 'POST'
        });
    }
};
