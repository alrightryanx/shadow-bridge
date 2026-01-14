/**
 * Shadow Web Dashboard - Main App Logic
 */

// Current state
let currentNoteId = null;
let currentNoteContent = null;
let currentNoteTitle = null;
let isEditMode = false;
let selectedDevice = '__ALL__';  // '__ALL__' means show all devices
let lastStatusSnapshot = null;
const deviceNameMap = new Map();
const CONNECTION_BACKEND_OPTIONS = [
    { value: 'shadowbridge', label: 'ShadowBridge Local' },
    { value: 'ssh_gemini_cli', label: 'Gemini CLI (SSH)' },
    { value: 'ssh_claude_code', label: 'Claude Code (SSH)' },
    { value: 'openai_api', label: 'OpenAI API' },
    { value: 'anthropic_api', label: 'Anthropic API' }
];
const CONNECTION_MODEL_OPTIONS = {
    shadowbridge: [
        { value: 'shadowbridge-core', label: 'ShadowBridge Core' }
    ],
    ssh_gemini_cli: [
        { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' },
        { value: 'gemini-1.5-mini', label: 'Gemini 1.5 Mini' }
    ],
    ssh_claude_code: [
        { value: 'claude-3.5-pro', label: 'Claude 3.5 Pro' },
        { value: 'claude-3.1', label: 'Claude 3.1' }
    ],
    openai_api: [
        { value: 'gpt-4o', label: 'gpt-4o' },
        { value: 'gpt-4o-mini', label: 'gpt-4o-mini' },
        { value: 'gpt-3.5-turbo', label: 'gpt-3.5-turbo' }
    ],
    anthropic_api: [
        { value: 'claude-3.6', label: 'Claude 3.6' },
        { value: 'claude-instant', label: 'Claude Instant' }
    ],
    default: [
        { value: 'default-model', label: 'Default model' }
    ]
};
const CONNECTION_STATE_KEY = 'shadow-connection-selection';
function getModelOptionsForBackend(backend) {
    return CONNECTION_MODEL_OPTIONS[backend] || CONNECTION_MODEL_OPTIONS.default || [];
}
function loadConnectionSelection() {
    const defaultBackend = CONNECTION_BACKEND_OPTIONS[0]?.value || '';
    const defaultModel = getModelOptionsForBackend(defaultBackend)[0]?.value || '';

    if (!defaultBackend) {
        return { backend: '', model: '' };
    }

    if (typeof localStorage === 'undefined') {
        return { backend: defaultBackend, model: defaultModel };
    }

    try {
        const stored = localStorage.getItem(CONNECTION_STATE_KEY);
        if (stored) {
            const parsed = JSON.parse(stored);
            const backend = CONNECTION_BACKEND_OPTIONS.some(o => o.value === parsed.backend)
                ? parsed.backend
                : defaultBackend;
            const models = getModelOptionsForBackend(backend);
            const modelValue = models.some(m => m.value === parsed.model)
                ? parsed.model
                : models[0]?.value || '';
            return {
                backend,
                model: modelValue || models[0]?.value || defaultModel
            };
        }
    } catch (error) {
        console.warn('Failed to restore connection selection', error);
    }

    return { backend: defaultBackend, model: defaultModel };
}
let connectionSelection = loadConnectionSelection();
if (typeof window !== 'undefined') {
    window.deviceNameMap = deviceNameMap;
}

// ============ Theme Management ============

function initTheme() {
    // Check for saved theme preference or default to dark
    const savedTheme = localStorage.getItem('shadow-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
}

async function fetchWithRetry(url, options = {}, { attempts = 3, delay = 1200 } = {}) {
    let lastError = null;
    for (let attempt = 1; attempt <= attempts; attempt++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                lastError = new Error(`HTTP ${response.status}: ${response.statusText}`);
                throw lastError;
            }
            return response;
        } catch (error) {
            lastError = error;
            if (attempt < attempts) {
                await new Promise(resolve => setTimeout(resolve, delay * attempt));
            }
        }
    }
    throw lastError;
}

if (typeof window !== 'undefined') {
    window.fetchWithRetry = fetchWithRetry;
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('shadow-theme', newTheme);

    // Show feedback
    showToast(`Switched to ${newTheme} mode`, 'info', 2000);
}

// Initialize theme immediately to prevent flash
initTheme();

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initDeviceSelector();
    initConnectionControls();
    updateStatus();

    // Refresh status periodically
    setInterval(updateStatus, 30000);

    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true });
    }

    updateConnectionStatus();
});

// ============ Toast Notification System ============

function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    // Icon based on type
    const icons = {
        success: '&#10003;',  // checkmark
        error: '&#10007;',    // X
        warning: '&#9888;',   // warning triangle
        info: '&#8505;'       // info
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
    `;

    container.appendChild(toast);

    // Trigger animation
    requestAnimationFrame(() => toast.classList.add('show'));

    // Auto dismiss
    if (duration > 0) {
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
}

let connectionOfflineToastShown = false;

// ============ Device Selector ============

async function initDeviceSelector() {
    const menu = document.getElementById('device-switcher-menu');
    const button = document.getElementById('device-switcher-button');
    const title = document.getElementById('device-switcher-title');
    if (!menu || !button || !title) return;

    const devices = await api.getDevices();
    deviceNameMap.clear();
    devices.forEach(device => {
        if (device?.id) {
            deviceNameMap.set(device.id, device.name || device.displayName || device.id);
        }
    });

    const savedDevice = localStorage.getItem('selectedDevice');
    if (savedDevice && (savedDevice === '__ALL__' || devices.some(d => d.id === savedDevice))) {
        selectedDevice = savedDevice;
    } else {
        selectedDevice = '__ALL__';
        localStorage.setItem('selectedDevice', selectedDevice);
    }

    renderDeviceMenu(devices);
    updateDeviceSwitcherTitle();

    if (!button.dataset.bound) {
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            toggleDeviceMenu(true);
        });

        document.addEventListener('click', (event) => {
            if (!menu.contains(event.target) && !button.contains(event.target)) {
                toggleDeviceMenu(false);
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                toggleDeviceMenu(false);
            }
        });

        button.dataset.bound = 'true';
    }
}

function renderDeviceMenu(devices) {
    const menu = document.getElementById('device-switcher-menu');
    if (!menu) return;

    const onlineCount = devices.filter(d => d.status === 'online').length;
    const allStatus = onlineCount > 0 ? 'online' : (devices.length > 0 ? 'offline' : 'idle');
    const allStatusLabel = devices.length > 0 ? `${onlineCount} online` : 'No devices';

    const items = [
        { id: '__ALL__', name: 'All Devices', status: allStatus, state: allStatusLabel },
        ...devices.map(device => ({
            id: device.id,
            name: device.name || device.id,
            status: device.status === 'online' ? 'online' : 'offline',
            state: device.status === 'online' ? 'Online' : 'Offline'
        }))
    ];

    menu.innerHTML = items.map(item => `
        <button type="button" class="device-switcher-item ${item.id === selectedDevice ? 'active' : ''}" data-device-id="${escapeHtml(item.id)}">
            <span class="status-dot ${item.status}"></span>
            <span class="device-name">${escapeHtml(item.name)}</span>
            <span class="device-state">${escapeHtml(item.state)}</span>
        </button>
    `).join('');

    menu.querySelectorAll('.device-switcher-item').forEach(entry => {
        entry.addEventListener('click', () => {
            const deviceId = entry.getAttribute('data-device-id');
            if (!deviceId) return;
            if (deviceId === selectedDevice) {
                toggleDeviceMenu(false);
                return;
            }
            selectedDevice = deviceId;
            localStorage.setItem('selectedDevice', selectedDevice);
            updateDeviceSwitcherTitle();
            toggleDeviceMenu(false);
            refreshData();
        });
    });
}

function updateDeviceSwitcherTitle() {
    const title = document.getElementById('device-switcher-title');
    const button = document.getElementById('device-switcher-button');
    if (!title) return;

    const label = selectedDevice === '__ALL__'
        ? 'All Devices'
        : (deviceNameMap.get(selectedDevice) || selectedDevice);
    title.textContent = label;
    if (button) {
        button.title = label;
    }
}

function toggleDeviceMenu(shouldOpen) {
    const menu = document.getElementById('device-switcher-menu');
    const button = document.getElementById('device-switcher-button');
    if (!menu || !button) return;

    const isOpen = !menu.classList.contains('hidden');
    const nextState = shouldOpen === undefined ? !isOpen : shouldOpen;

    menu.classList.toggle('hidden', !nextState);
    button.setAttribute('aria-expanded', String(nextState));
}

function initConnectionControls() {
    const backendSelect = document.getElementById('backend-selector');
    const modelSelect = document.getElementById('model-selector');
    if (!backendSelect || !modelSelect) return;

    backendSelect.innerHTML = CONNECTION_BACKEND_OPTIONS.map(option => `
        <option value="${option.value}">${option.label}</option>
    `).join('');

    if (!CONNECTION_BACKEND_OPTIONS.some(option => option.value === connectionSelection.backend)) {
        connectionSelection.backend = CONNECTION_BACKEND_OPTIONS[0]?.value || '';
    }

    backendSelect.value = connectionSelection.backend;
    backendSelect.addEventListener('change', function() {
        const value = this.value;
        connectionSelection.backend = value;
        updateModelOptions(value);
    });

    modelSelect.addEventListener('change', function() {
        connectionSelection.model = this.value;
        persistConnectionSelection();
        updateConnectionSelectionHint();
    });

    updateModelOptions(connectionSelection.backend, connectionSelection.model);
    updateConnectionSelectionHint();
}

function updateModelOptions(backend, preferredModel = null) {
    const modelSelect = document.getElementById('model-selector');
    if (!modelSelect) return;

    const options = getModelOptionsForBackend(backend);
    if (options.length === 0) {
        modelSelect.innerHTML = '<option value="">Unavailable</option>';
        connectionSelection.model = '';
        persistConnectionSelection();
        updateConnectionSelectionHint();
        return;
    }

    modelSelect.innerHTML = options.map(option => `
        <option value="${option.value}">${option.label}</option>
    `).join('');

    const desiredValue = preferredModel && options.some(option => option.value === preferredModel)
        ? preferredModel
        : options[0].value;
    modelSelect.value = desiredValue;
    connectionSelection.model = desiredValue;
    persistConnectionSelection();
    updateConnectionSelectionHint();
}

function persistConnectionSelection() {
    if (typeof localStorage === 'undefined') return;
    try {
        localStorage.setItem(CONNECTION_STATE_KEY, JSON.stringify(connectionSelection));
    } catch (error) {
        console.warn('Could not persist connection selection', error);
    }
}

function updateConnectionSelectionHint() {
    const hint = document.getElementById('connection-selection-hint');
    if (!hint) return;

    const activeSession = resolveActiveSessionMeta();
    if (activeSession) {
        const backendLabel = getBackendLabelForBanner(activeSession);
        const modelLabel = getModelLabelForBanner(activeSession);
        if (backendLabel && modelLabel) {
            hint.textContent = `Active: ${backendLabel} Â· ${modelLabel}`;
            return;
        }
        if (backendLabel) {
            hint.textContent = `Active: ${backendLabel}`;
            return;
        }
    }

    const backendLabel = getConnectionBackendLabel(connectionSelection.backend);
    const modelLabel = getConnectionModelLabel(connectionSelection.backend, connectionSelection.model);

    if (backendLabel && modelLabel) {
        hint.textContent = `${backendLabel} · ${modelLabel}`;
    } else if (backendLabel) {
        hint.textContent = backendLabel;
    } else {
        hint.textContent = 'Switch backend/model anytime';
    }
}

function getConnectionBackendLabel(value) {
    const option = CONNECTION_BACKEND_OPTIONS.find(item => item.value === value);
    return option ? option.label : '';
}

function getConnectionModelLabel(backend, value) {
    const option = getModelOptionsForBackend(backend).find(item => item.value === value);
    return option ? option.label : '';
}

function resolveActiveSessionMeta() {
    if (typeof currentSessionDetail !== 'undefined' && currentSessionDetail) {
        return currentSessionDetail;
    }
    if (
        typeof selectedSessionId !== 'undefined' &&
        selectedSessionId &&
        typeof dashboardSessions !== 'undefined' &&
        Array.isArray(dashboardSessions)
    ) {
        return dashboardSessions.find(s => s.id === selectedSessionId) || null;
    }
    return null;
}

function getBackendLabelForBanner(session) {
    if (session && typeof window.formatBackendLabel === 'function') {
        return window.formatBackendLabel(session);
    }
    const rawType = String(session?.backend_type || session?.backendType || '').toLowerCase();
    const provider = String(session?.provider || '').trim();
    if (rawType.includes('ssh')) return provider ? `SSH - ${provider}` : 'SSH';
    if (rawType.includes('api')) return provider || 'API';
    if (rawType.includes('local')) return 'On-device LLM';
    if (provider) return provider;
    if (rawType) return rawType.toUpperCase();
    return '';
}

function getModelLabelForBanner(session) {
    if (session && typeof window.formatModelLabel === 'function') {
        return window.formatModelLabel(session);
    }
    return String(session?.model || '').trim();
}

function updateConnectionBanner(status = null) {
    const dot = document.getElementById('connection-health-dot');
    const text = document.getElementById('connection-health-text');
    if (!dot || !text) return;

    const statusSnapshot = status || lastStatusSnapshot;
    const session = resolveActiveSessionMeta();
    const backendLabel = session ? getBackendLabelForBanner(session) : '';
    const modelLabel = session ? getModelLabelForBanner(session) : '';
    const activeLabel = backendLabel
        ? `${backendLabel}${modelLabel ? ` Â· ${modelLabel}` : ''}`
        : 'No active session';

    let dotClass = 'status-dot idle';
    let healthLabel = 'Unknown health';

    if (!api.isOnline) {
        dotClass = 'status-dot offline';
        healthLabel = 'Bridge offline';
    } else if (session) {
        const rawType = String(session.backend_type || session.backendType || '').toLowerCase();
        if (rawType.includes('ssh')) {
            const sshStatus = statusSnapshot?.ssh_status || 'unknown';
            if (sshStatus === 'connected') {
                dotClass = 'status-dot online';
                healthLabel = 'SSH connected';
            } else if (sshStatus === 'offline') {
                dotClass = 'status-dot offline';
                healthLabel = 'SSH offline';
            } else if (sshStatus === 'no_devices') {
                dotClass = 'status-dot idle';
                healthLabel = 'No devices';
            } else {
                dotClass = 'status-dot idle';
                healthLabel = 'SSH unknown';
            }
        } else if (rawType.includes('api')) {
            dotClass = api.isOnline ? 'status-dot online' : 'status-dot offline';
            healthLabel = api.isOnline ? 'API reachable' : 'API offline';
        } else if (rawType.includes('local')) {
            dotClass = 'status-dot idle';
            healthLabel = 'Local device';
        } else {
            dotClass = api.isOnline ? 'status-dot online' : 'status-dot offline';
            healthLabel = api.isOnline ? 'Bridge online' : 'Bridge offline';
        }
    } else {
        dotClass = api.isOnline ? 'status-dot idle' : 'status-dot offline';
        healthLabel = api.isOnline ? 'Bridge online' : 'Bridge offline';
    }

    dot.className = dotClass;
    text.textContent = `${activeLabel} Â· ${healthLabel}`;
}

// Get device ID for API calls (null for all devices)
function getDeviceIdParam() {
    return selectedDevice === '__ALL__' ? null : selectedDevice;
}

// ============ Refresh Data ============

async function refreshData() {
    const btn = document.getElementById('refresh-btn');
    if (btn) {
        btn.classList.add('spinning');
        btn.disabled = true;
    }

    try {
        // Refresh device selector
        await initDeviceSelector();

        // Trigger page-specific refresh functions
        if (typeof loadDashboard === 'function') await loadDashboard();
        if (typeof loadProjects === 'function') await loadProjects();
        if (typeof loadNotes === 'function') await loadNotes();
        if (typeof loadAutomations === 'function') await loadAutomations();
        if (typeof loadAgents === 'function') await loadAgents();
        if (typeof loadAnalytics === 'function') await loadAnalytics();
        if (typeof loadSettings === 'function') await loadSettings();

        await updateStatus();
        showToast('Data refreshed', 'success');
    } catch (error) {
        if (api.isOnline) {
            showToast('Failed to refresh: ' + error.message, 'error');
        }
    } finally {
        if (btn) {
            btn.classList.remove('spinning');
            btn.disabled = false;
        }
    }
}

// ============ Status Indicator ============

async function updateStatus() {
    const status = await api.getStatus();
    lastStatusSnapshot = status;
    const dot = document.getElementById('device-status-dot');
    const text = document.getElementById('device-switcher-status');
    if (status && status.error && !status._offline) {
        showToast(`Status error: ${status.error}`, 'warning', 5000);
        if (typeof api.trackActivity === 'function') {
            api.trackActivity(
                'status_error',
                'bridge',
                'status',
                'Status endpoint error',
                { error: status.error }
            ).catch(() => {});
        }
    }
    if (dot && text) {
        // Check both API reachability AND device connections for accurate status
        const isApiReachable = api.isOnline;
        const hasConnectedDevices = status.devices_connected > 0;

        if (isApiReachable && hasConnectedDevices) {
            // Full system online
            dot.className = 'status-dot online';
            text.textContent = status.devices_connected + ' online';
        } else if (isApiReachable && !hasConnectedDevices) {
            // Bridge reachable but no devices connected
            dot.className = 'status-dot idle';
            text.textContent = 'No devices connected';
        } else if (!isApiReachable) {
            // Bridge unreachable
            dot.className = 'status-dot offline';
            text.textContent = 'Offline (cached)';
        }
    }

    // Update sidebar counts
    await updateSidebarCounts();
    updateConnectionStatus();
    updateConnectionBanner(status);
}

if (typeof window !== 'undefined') {
    window.updateConnectionBanner = updateConnectionBanner;
    window.updateConnectionSelectionHint = updateConnectionSelectionHint;
}

async function updateSidebarCounts() {
    const deviceId = getDeviceIdParam();

    const projects = await api.getProjects(deviceId);
    const projectsCount = document.getElementById('sidebar-projects-count');
    if (projectsCount) {
        projectsCount.textContent = projects.length || '';
    }

    const notes = await api.getNotes(deviceId);
    const notesCount = document.getElementById('sidebar-notes-count');
    if (notesCount) {
        notesCount.textContent = notes.length || '';
    }
}

// ============ Project Actions ============

async function openProject(id) {
    const result = await api.openProject(id);
    if (result.error) {
        if (!result._offline) {
            showToast('Error opening project: ' + result.error, 'error');
        }
    } else {
        showToast('Opening project...', 'success');
        // Track activity for context window (AGI-Readiness)
        api.trackActivity('open', 'project', id, result.name || 'Project').catch(() => {});
    }
}

// ============ Note Actions ============

async function openNote(id) {
    currentNoteId = id;
    isEditMode = false;
    const modal = document.getElementById('note-modal');
    const title = document.getElementById('modal-title');
    const content = document.getElementById('modal-content');

    // Show loading state
    modal.classList.remove('hidden');
    title.textContent = 'Loading...';
    content.innerHTML = '<div class="loading">Fetching note content from device...</div>';
    updateModalButtons();

    // Fetch content
    const result = await api.getNoteContent(id);

    if (result.error) {
        title.textContent = 'Error';
        content.innerHTML = '<div class="error-message">' + escapeHtml(result.error) + '</div>';
        if (!result._offline) {
            showToast('Failed to load note: ' + result.error, 'error');
        }
        currentNoteContent = null;
        currentNoteTitle = null;
    } else {
        currentNoteTitle = result.title || 'Untitled';
        currentNoteContent = result.content || '';
        title.textContent = currentNoteTitle;
        renderNoteContent();

        // Track activity for context window (AGI-Readiness)
        api.trackActivity('view', 'note', id, currentNoteTitle).catch(() => {});
    }
}

function renderNoteContent() {
    const content = document.getElementById('modal-content');
    if (!content) return;
    content.classList.remove('markdown-body');

    if (isEditMode) {
        // Edit mode - show textarea
        content.innerHTML = `
            <textarea id="note-editor" class="note-editor">${escapeHtml(currentNoteContent)}</textarea>
        `;
        const editor = document.getElementById('note-editor');
        if (editor) {
            editor.focus();
            // Auto-resize
            editor.style.height = 'auto';
            editor.style.height = Math.max(300, editor.scrollHeight) + 'px';
            editor.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.max(300, this.scrollHeight) + 'px';
            });
        }
    } else {
        // View mode - render markdown
        const noteContent = currentNoteContent || '(Empty note)';
        if (typeof marked !== 'undefined') {
            content.classList.add('markdown-body');
            content.innerHTML = marked.parse(noteContent, { breaks: true });
        } else {
            content.innerHTML = '<pre>' + escapeHtml(noteContent) + '</pre>';
        }
    }
    updateModalButtons();
}

function updateModalButtons() {
    const editBtn = document.getElementById('edit-note-btn');
    const saveBtn = document.getElementById('save-note-btn');
    const exportBtn = document.getElementById('export-note-btn');
    const toolbar = document.getElementById('markdown-toolbar');

    if (isEditMode) {
        if (editBtn) editBtn.classList.add('hidden');
        if (saveBtn) saveBtn.classList.remove('hidden');
        if (exportBtn) exportBtn.classList.add('hidden');
        if (toolbar) toolbar.classList.remove('hidden');
    } else {
        if (editBtn) editBtn.classList.remove('hidden');
        if (saveBtn) saveBtn.classList.add('hidden');
        if (exportBtn) exportBtn.classList.remove('hidden');
        if (toolbar) toolbar.classList.add('hidden');
    }
}

function editNote() {
    if (!currentNoteId) return;
    isEditMode = true;
    renderNoteContent();
}

function cancelEdit() {
    isEditMode = false;
    renderNoteContent();
}

async function saveNote() {
    if (!currentNoteId) {
        showToast('No note selected', 'error');
        return;
    }

    const editor = document.getElementById('note-editor');
    if (!editor) {
        showToast('Editor not found', 'error');
        return;
    }

    const newContent = editor.value;
    const saveBtn = document.getElementById('save-note-btn');

    // Show saving state
    if (saveBtn) {
        saveBtn.disabled = true;
        saveBtn.textContent = 'Saving...';
    }

    const result = await api.updateNoteContent(currentNoteId, newContent, currentNoteTitle);

    if (result.error) {
        if (!result._offline) {
            showToast('Failed to save: ' + result.error, 'error');
        }
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.textContent = 'Save';
        }
    } else {
        currentNoteContent = newContent;
        isEditMode = false;

        // Update the notes page cache if it exists
        if (typeof loadedNoteContents !== 'undefined' && currentNoteId) {
            loadedNoteContents[currentNoteId] = newContent;
        }

        renderNoteContent();
        showToast('Note saved successfully', 'success');
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.textContent = 'Save';
        }

        // Track activity for context window (AGI-Readiness)
        api.trackActivity('edit', 'note', currentNoteId, currentNoteTitle).catch(() => {});
    }
}

function closeModal() {
    const modal = document.getElementById('note-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
    currentNoteId = null;
    currentNoteContent = null;
    currentNoteTitle = null;
    isEditMode = false;
}

async function exportNote() {
    if (!currentNoteId) return;

    const result = await api.exportNote(currentNoteId);
    if (result.error) {
        if (!result._offline) {
            showToast('Export failed: ' + result.error, 'error');
        }
    } else {
        showToast('Note exported to: ' + result.path, 'success');
    }
}

async function copyNoteContent() {
    if (!currentNoteContent) {
        showToast('No content to copy', 'warning');
        return;
    }

    try {
        await navigator.clipboard.writeText(currentNoteContent);
        showToast('Note content copied to clipboard', 'success');
    } catch (err) {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = currentNoteContent;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            showToast('Note content copied to clipboard', 'success');
        } catch (e) {
            showToast('Failed to copy: ' + e.message, 'error');
        }
        document.body.removeChild(textarea);
    }
}

// Escape HTML for safe rendering
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Close modal on escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// Close modal on backdrop click
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('modal')) {
        closeModal();
    }
});

// ============ Global Search ============

let searchTimeout = null;
let searchResultsVisible = false;
let searchMode = 'hybrid'; // 'keyword', 'semantic', or 'hybrid'

async function handleSearchKeyup(event) {
    const query = event.target.value.trim();

    // Clear previous timeout
    if (searchTimeout) clearTimeout(searchTimeout);

    // Hide results if empty
    if (!query) {
        hideSearchResults();
        return;
    }

    // Debounce search
    searchTimeout = setTimeout(async () => {
        await performSearch(query);
    }, 300);

    // Enter key navigates to first result
    if (event.key === 'Enter') {
        const firstResult = document.querySelector('.search-result-item');
        if (firstResult) firstResult.click();
    }
}

function setSearchMode(mode) {
    searchMode = mode;
    // Update toggle buttons
    document.querySelectorAll('.search-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    // Re-run search if there's a query
    const query = document.getElementById('global-search')?.value?.trim();
    if (query) performSearch(query);

    // Learn preference for search mode (AGI-Readiness)
    api.observePreference('search', 'search_mode', { search_mode: mode }).catch(() => {});
}

async function performSearch(query) {
    const container = document.getElementById('search-results');

    // Show loading state
    container.innerHTML = '<div class="search-loading"><span class="spinner"></span> Searching...</div>';
    container.classList.remove('hidden');

    let results;
    let isSemanticResult = false;

    try {
        if (searchMode === 'semantic') {
            results = await api.semanticSearch(query, null, 10);
            isSemanticResult = true;
        } else if (searchMode === 'hybrid') {
            results = await api.hybridSearch(query, null, 10);
            isSemanticResult = results.search_type === 'semantic';
        } else {
            results = await api.search(query);
        }
    } catch (e) {
        console.error('Search error:', e);
        results = { error: e.message };
    }

    // Handle semantic results format
    const items = results.results || results.items || [];

    if (results.error || items.length === 0) {
        container.innerHTML = `
            <div class="search-mode-header">
                <div class="search-mode-toggle">
                    <button class="search-mode-btn ${searchMode === 'keyword' ? 'active' : ''}" data-mode="keyword" onclick="setSearchMode('keyword')" title="Keyword search">ABC</button>
                    <button class="search-mode-btn ${searchMode === 'hybrid' ? 'active' : ''}" data-mode="hybrid" onclick="setSearchMode('hybrid')" title="Hybrid search">&#x2726;</button>
                    <button class="search-mode-btn ${searchMode === 'semantic' ? 'active' : ''}" data-mode="semantic" onclick="setSearchMode('semantic')" title="Semantic AI search">&#x1F9E0;</button>
                </div>
            </div>
            <div class="search-no-results">No results found${results.error ? ' - ' + results.error : ''}</div>`;
        return;
    }

    // Build search mode header
    const modeHeader = `
        <div class="search-mode-header">
            <div class="search-mode-toggle">
                <button class="search-mode-btn ${searchMode === 'keyword' ? 'active' : ''}" data-mode="keyword" onclick="setSearchMode('keyword')" title="Keyword search">ABC</button>
                <button class="search-mode-btn ${searchMode === 'hybrid' ? 'active' : ''}" data-mode="hybrid" onclick="setSearchMode('hybrid')" title="Hybrid search">&#x2726;</button>
                <button class="search-mode-btn ${searchMode === 'semantic' ? 'active' : ''}" data-mode="semantic" onclick="setSearchMode('semantic')" title="Semantic AI search">&#x1F9E0;</button>
            </div>
            ${isSemanticResult ? '<span class="search-type-badge">AI</span>' : '<span class="search-type-badge keyword">Keyword</span>'}
        </div>`;

    container.innerHTML = modeHeader + items.slice(0, 10).map(item => {
        // Handle both keyword and semantic result formats
        const type = item.type || item.source_type || 'unknown';
        const id = item.id || item.source_id || '';
        const title = item.title || item.name || 'Untitled';
        const preview = item.preview || (item.content ? item.content.substring(0, 80) : '');
        const score = item.score;

        return `
        <div class="search-result-item" onclick="navigateToResult('${type}', '${id}')">
            <span class="search-result-icon">${getSearchIcon(type)}</span>
            <div class="search-result-content">
                <div class="search-result-title">${escapeHtml(title)}</div>
                <div class="search-result-meta">${type} ${preview ? '- ' + escapeHtml(preview.substring(0, 50)) : ''}</div>
            </div>
            ${score !== undefined ? `<span class="search-score" title="Relevance score">${Math.round(score * 100)}%</span>` : ''}
        </div>`;
    }).join('');

    searchResultsVisible = true;

    // Track search as a topic for context window (AGI-Readiness)
    api.recordTopic(query).catch(() => {});
}

function getSearchIcon(type) {
    const icons = {
        'project': '&#128193;',
        'note': '&#128221;',
        'automation': '&#9881;',
        'agent': '&#129302;',
        'team': '&#128101;'
    };
    return icons[type] || '&#128196;';
}

function navigateToResult(type, id) {
    hideSearchResults();
    document.getElementById('global-search').value = '';

    switch (type) {
        case 'project':
            window.location.href = '/projects';
            break;
        case 'note':
            window.location.href = '/notes';
            setTimeout(() => openNote(id), 500);
            break;
        case 'automation':
            window.location.href = '/automations';
            break;
        case 'agent':
            window.location.href = '/agents';
            break;
        case 'team':
            window.location.href = '/teams';
            break;
    }
}

function showSearchResults() {
    const container = document.getElementById('search-results');
    const query = document.getElementById('global-search').value.trim();
    if (query && container.innerHTML) {
        container.classList.remove('hidden');
        searchResultsVisible = true;
    }
}

function hideSearchResults() {
    document.getElementById('search-results').classList.add('hidden');
    searchResultsVisible = false;
}

function hideSearchResultsDelayed() {
    setTimeout(hideSearchResults, 200);
}

// ============ Markdown Toolbar ============

function insertMarkdown(before, after) {
    const editor = document.getElementById('note-editor');
    if (!editor) return;

    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    const text = editor.value;
    const selectedText = text.substring(start, end);

    const newText = text.substring(0, start) + before + selectedText + after + text.substring(end);
    editor.value = newText;

    // Set cursor position
    const newCursorPos = start + before.length + selectedText.length;
    editor.setSelectionRange(newCursorPos, newCursorPos);
    editor.focus();
}

// ============ Offline Status Indicator ============

function updateConnectionStatus() {
    const dot = document.getElementById('device-status-dot');
    const text = document.getElementById('device-switcher-status');
    const button = document.getElementById('device-switcher-button');
    if (!dot || !text) return;

    if (!api.isOnline) {
        dot.className = 'status-dot offline';
        text.textContent = 'Offline (cached)';
        if (button) {
            button.title = 'Using cached data - connection unavailable';
        }

        if (!connectionOfflineToastShown) {
            showToast('ShadowBridge is unreachable. Working with cached data.', 'warning', 6000);
            connectionOfflineToastShown = true;
        }
    } else {
        if (connectionOfflineToastShown) {
            showToast('ShadowBridge connection restored', 'success', 3000);
            connectionOfflineToastShown = false;
        }
    }
}

// ============ Mobile Menu Toggle ============

function toggleMobileMenu() {
    const sidebar = document.querySelector('.sidebar');
    const isOpen = sidebar.classList.contains('mobile-open');

    if (isOpen) {
        sidebar.classList.remove('mobile-open');
        document.getElementById('mobile-menu-toggle').innerHTML = '<span class="material-symbols-outlined">menu</span>';
    } else {
        sidebar.classList.add('mobile-open');
        document.getElementById('mobile-menu-toggle').innerHTML = '<span class="material-symbols-outlined">close</span>';
    }
}

// Close mobile menu when clicking outside
document.addEventListener('click', function(event) {
    const sidebar = document.querySelector('.sidebar');
    const menuToggle = document.getElementById('mobile-menu-toggle');

    if (sidebar && menuToggle && sidebar.classList.contains('mobile-open')) {
        if (!sidebar.contains(event.target) && !menuToggle.contains(event.target)) {
            sidebar.classList.remove('mobile-open');
            menuToggle.innerHTML = '<span class="material-symbols-outlined">menu</span>';
        }
    }
});

// Close mobile menu on navigation
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', function() {
        const sidebar = document.querySelector('.sidebar');
        const menuToggle = document.getElementById('mobile-menu-toggle');
        if (sidebar && sidebar.classList.contains('mobile-open')) {
            sidebar.classList.remove('mobile-open');
            if (menuToggle) {
                menuToggle.innerHTML = '<span class="material-symbols-outlined">menu</span>';
            }
        }
    });
});

