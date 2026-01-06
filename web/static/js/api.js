/**
 * ShadowAI Web Dashboard - API Client
 * With offline caching and retry logic
 */

function reportConnectionStatus() {
    if (typeof window !== 'undefined' && typeof window.updateConnectionStatus === 'function') {
        window.updateConnectionStatus();
    }
}

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
                reportConnectionStatus();

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
        reportConnectionStatus();

        const offlineData = this.getOfflineCache(cacheKey);
        if (offlineData) {
            console.log('Using offline cache for:', endpoint);
            if (Array.isArray(offlineData)) {
                offlineData._offline = true;
                return offlineData;
            }
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

    // Sessions
    async getSessions(deviceId = null, projectId = null, limit = null) {
        const params = new URLSearchParams();
        if (deviceId) params.append('device_id', deviceId);
        if (projectId) params.append('project_id', projectId);
        if (limit) params.append('limit', limit);
        const query = params.toString();
        return this.fetch(`/sessions${query ? '?' + query : ''}`);
    },

    async getSession(id) {
        return this.fetch(`/sessions/${id}`);
    },

    async upsertSession(data) {
        return this.fetch('/sessions', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    async appendSessionMessage(sessionId, payload) {
        return this.fetch(`/sessions/${sessionId}/messages`, {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    },

    async deleteSession(sessionId) {
        return this.fetch(`/sessions/${sessionId}`, {
            method: 'DELETE'
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

    // ============ Semantic Search (Vector Store) ============
    async semanticSearch(query, types = null, limit = 10) {
        const params = new URLSearchParams({ q: query, limit: limit });
        if (types && types.length > 0) {
            params.set('types', types.join(','));
        }
        return this.fetch(`/vector/search?${params}`, {}, false);
    },

    async getVectorStatus() {
        return this.fetch('/vector/status', {}, false);
    },

    async reindexVectorStore() {
        return this.fetch('/vector/reindex', { method: 'POST' });
    },

    async clearVectorStore() {
        return this.fetch('/vector/clear', { method: 'POST' });
    },

    // ============ Unified Memory ============
    async getMemoryStats() {
        return this.fetch('/memory/stats', {}, true);
    },

    async memorySearch(query, scopes = null, limit = 20) {
        const params = new URLSearchParams({ q: query, limit: limit });
        if (scopes && scopes.length > 0) {
            params.set('scopes', scopes.join(','));
        }
        return this.fetch(`/memory/search?${params}`, {}, false);
    },

    async getRecentMemory(scopes = null, limit = 10) {
        const params = new URLSearchParams({ limit: limit });
        if (scopes && scopes.length > 0) {
            params.set('scopes', scopes.join(','));
        }
        return this.fetch(`/memory/recent?${params}`, {}, false);
    },

    // ============ Hybrid Search (combines keyword + semantic) ============
    async hybridSearch(query, types = null, limit = 10) {
        // Try semantic search first, fallback to keyword
        const semanticResults = await this.semanticSearch(query, types, limit);

        if (semanticResults.error || !semanticResults.results || semanticResults.results.length === 0) {
            // Fallback to keyword search
            const keywordTypes = types || ['projects', 'notes', 'automations', 'agents'];
            const keywordResults = await this.search(query, keywordTypes);
            return {
                results: (keywordResults.items || []).map(item => ({
                    ...item,
                    score: 1.0,
                    source_type: item.type,
                    source_id: item.id
                })),
                search_type: 'keyword',
                query: query
            };
        }

        return semanticResults;
    },

    // ============ Context Window (AGI-Readiness) ============
    async trackActivity(eventType, resourceType, resourceId, resourceTitle, metadata = null) {
        return this.fetch('/context/track', {
            method: 'POST',
            body: JSON.stringify({
                event_type: eventType,
                resource_type: resourceType,
                resource_id: resourceId,
                resource_title: resourceTitle,
                metadata: metadata
            })
        });
    },

    async getContextWindow(query = null, includeSemantic = true, maxTokens = 4000) {
        const params = new URLSearchParams({ include_semantic: includeSemantic, max_tokens: maxTokens });
        if (query) params.set('q', query);
        return this.fetch(`/context/window?${params}`, {}, false);
    },

    async getContextPrompt(query = null) {
        const params = query ? `?q=${encodeURIComponent(query)}` : '';
        return this.fetch(`/context/prompt${params}`, {}, false);
    },

    async getRecentActivity(minutes = 30, limit = 10) {
        return this.fetch(`/context/recent?minutes=${minutes}&limit=${limit}`, {}, false);
    },

    async getContextStats() {
        return this.fetch('/context/stats', {}, false);
    },

    async recordTopic(topic) {
        return this.fetch('/context/topic', {
            method: 'POST',
            body: JSON.stringify({ topic })
        });
    },

    async clearContextSession() {
        return this.fetch('/context/clear', { method: 'POST' });
    },

    // ============ User Preference Learning (AGI-Readiness) ============
    async getPreferences() {
        return this.fetch('/preferences', {}, false);
    },

    async observePreference(action, resourceType, metadata = null) {
        return this.fetch('/preferences/observe', {
            method: 'POST',
            body: JSON.stringify({
                action: action,
                resource_type: resourceType,
                metadata: metadata
            })
        });
    },

    async submitPreferenceFeedback(category, value, isPositive) {
        return this.fetch('/preferences/feedback', {
            method: 'POST',
            body: JSON.stringify({
                category: category,
                value: value,
                is_positive: isPositive
            })
        });
    },

    async getPreferencesForAI() {
        return this.fetch('/preferences/ai-context', {}, false);
    },

    async getPreferenceStats() {
        return this.fetch('/preferences/stats', {}, false);
    },

    async resetPreferences() {
        return this.fetch('/preferences/reset', { method: 'POST' });
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
    },

    // ============ Video Generation ============
    async generateVideo(data) {
        return this.fetch('/video/generate', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    async getVideoModels() {
        return this.fetch('/video/models');
    },

    async getGenerationStatus(generationId) {
        return this.fetch(`/video/status/${generationId}`);
    },

    async getGenerationResult(generationId) {
        return this.fetch(`/video/result/${generationId}`);
    },

    async cancelGeneration(generationId) {
        return this.fetch(`/video/cancel/${generationId}`, { method: 'DELETE' });
    },

    async getVideoHistory(deviceId) {
        const params = deviceId ? `?device_id=${deviceId}` : '';
        return this.fetch(`/video/history${params}`);
    }
};
