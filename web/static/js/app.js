/**
 * Shadow Web Dashboard - Main App Logic
 */

// Current state
let currentNoteId = null;
let currentNoteContent = null;
let currentNoteTitle = null;
let isEditMode = false;
let selectedDevice = '__ALL__';  // '__ALL__' means show all devices

// ============ Theme Management ============

function initTheme() {
    // Check for saved theme preference or default to dark
    const savedTheme = localStorage.getItem('shadow-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
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
    updateStatus();

    // Refresh status periodically
    setInterval(updateStatus, 30000);

    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true });
    }
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

// ============ Device Selector ============

async function initDeviceSelector() {
    const selector = document.getElementById('device-selector');
    if (!selector) return;

    const devices = await api.getDevices();
    selector.innerHTML = '<option value="__ALL__">All Devices</option>';

    devices.forEach(device => {
        const option = document.createElement('option');
        option.value = device.id;
        // Show shortened device name with status
        const shortName = device.name.length > 30 ? device.name.substring(0, 27) + '...' : device.name;
        const statusIcon = device.status === 'online' ? ' ●' : ' ○';
        option.textContent = shortName + statusIcon;
        option.title = device.name;  // Full name on hover
        selector.appendChild(option);
    });

    // Restore previously selected device
    const savedDevice = localStorage.getItem('selectedDevice');
    if (savedDevice) {
        selector.value = savedDevice;
        selectedDevice = savedDevice;
    }

    selector.addEventListener('change', function() {
        selectedDevice = this.value;
        localStorage.setItem('selectedDevice', selectedDevice);
        refreshData();
    });
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
        showToast('Failed to refresh: ' + error.message, 'error');
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
    const indicator = document.getElementById('status-indicator');
    if (indicator) {
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');

        if (status.devices_connected > 0) {
            dot.className = 'status-dot online';
            text.textContent = status.devices_connected + ' online';
        } else {
            dot.className = 'status-dot offline';
            text.textContent = 'Offline';
        }
    }

    // Update sidebar counts
    await updateSidebarCounts();
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
        showToast('Error opening project: ' + result.error, 'error');
    } else {
        showToast('Opening project...', 'success');
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
        showToast('Failed to load note: ' + result.error, 'error');
        currentNoteContent = null;
        currentNoteTitle = null;
    } else {
        currentNoteTitle = result.title || 'Untitled';
        currentNoteContent = result.content || '';
        title.textContent = currentNoteTitle;
        renderNoteContent();
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
        showToast('Failed to save: ' + result.error, 'error');
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
        showToast('Export failed: ' + result.error, 'error');
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
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;

    const dot = indicator.querySelector('.status-dot');
    const text = indicator.querySelector('.status-text');

    if (!api.isOnline) {
        dot.className = 'status-dot offline';
        text.textContent = 'Offline (cached)';
        indicator.title = 'Using cached data - connection unavailable';
    }
}
