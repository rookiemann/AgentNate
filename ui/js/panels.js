/**
 * Multi-panel chat system for AgentNate UI.
 * Each panel is an independent chat environment with its own
 * conversation, persona, model, routing preset, and SSE stream.
 */

import { state, API, createPanelState, getActivePanel, getPanelDomId } from './state.js';

// Cached persona list for panel dropdowns
let _cachedPersonas = [];

// ─── Panel Lifecycle ───────────────────────────────────────────────

/**
 * Initialize the panel system. Called once on DOMContentLoaded.
 * Creates the first default panel.
 */
export function initPanelSystem() {
    const firstPanel = _createPanelInternal('Chat 1');
    state.activePanelId = firstPanel.panelId;
    _renderAllTabs();
    _updateTabBarVisibility();
}

/**
 * Create a new chat panel and switch to it.
 */
export function createNewPanel(name) {
    const panel = _createPanelInternal(name || `Chat ${state.panelCounter + 1}`);
    switchToPanel(panel.panelId);
    _renderAllTabs();
    _updateTabBarVisibility();
    // Focus the new panel's input
    const input = document.getElementById(getPanelDomId(panel.panelId, 'input'));
    if (input) setTimeout(() => input.focus(), 50);
    return panel;
}

/**
 * Create a new panel in the background without switching focus.
 */
export function createPanelInBackground(name) {
    const panel = _createPanelInternal(name || `Chat ${state.panelCounter + 1}`);
    _renderAllTabs();
    _updateTabBarVisibility();
    return panel;
}

/**
 * Create a minimal display-only worker panel (no welcome, no input, no controls).
 * Used for sub-agent worker tabs that only display streaming tool output.
 */
export function createWorkerPanel(name) {
    state.panelCounter++;
    const panelId = 'p' + state.panelCounter;
    const panel = createPanelState(panelId, name || 'Worker');
    panel.isWorkerPanel = true;
    panel.agentMode = true;
    panel.autonomous = true;
    state.panels[panelId] = panel;

    const container = document.getElementById('panel-container');
    if (container) {
        const div = document.createElement('div');
        const d = (suffix) => getPanelDomId(panelId, suffix);
        div.innerHTML = `
        <div class="chat-panel worker-panel" id="panel-${panelId}" data-panel-id="${panelId}">
            <div class="chat-header worker-header">
                <span class="agent-indicator">
                    <span class="agent-icon">⚙</span>
                    <span class="agent-label">${_escapeHtml(name || 'Worker')}</span>
                </span>
            </div>
            <div id="${d('messages')}" class="chat-messages"></div>
        </div>`;
        const panelEl = div.firstElementChild;
        container.appendChild(panelEl);
        // Hide if not active
        if (state.activePanelId && state.activePanelId !== panelId) {
            panelEl.style.display = 'none';
        }
    }

    _renderAllTabs();
    _updateTabBarVisibility();
    return panel;
}

/**
 * Force-close a panel even if it's the last one (for worker cleanup).
 */
export function forceClosePanel(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    // Abort if generating
    if (panel.isGenerating && panel.agentAbortId) {
        fetch(`${API}/tools/agent/abort`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ abort_id: panel.agentAbortId }),
        }).catch(() => {});
    }

    // Remove panel DOM
    const panelDom = document.getElementById(`panel-${panelId}`);
    if (panelDom) panelDom.remove();

    // Remove from state
    delete state.panels[panelId];

    // Switch to neighbor if this was the active panel
    if (state.activePanelId === panelId) {
        const remaining = Object.keys(state.panels);
        if (remaining.length > 0) {
            switchToPanel(remaining[remaining.length - 1]);
        }
    }

    _renderAllTabs();
    _updateTabBarVisibility();
}

/**
 * Switch to a panel by ID. Hides all others, shows the target.
 */
export function switchToPanel(panelId) {
    if (!state.panels[panelId]) return;
    const prev = state.activePanelId;
    state.activePanelId = panelId;

    // Toggle panel DOM visibility
    const container = document.getElementById('panel-container');
    if (container) {
        for (const child of container.children) {
            child.style.display = child.dataset.panelId === panelId ? '' : 'none';
        }
    }

    // Update tab active states
    document.querySelectorAll('.panel-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.panelId === panelId);
    });

    // Sync legacy state fields for backward compat with modules that still read them
    _syncLegacyState(panelId);
}

/**
 * Close a panel. Aborts any active generation. Switches to a neighbor.
 */
export function closePanel(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    // Don't close the last panel
    const panelIds = Object.keys(state.panels);
    if (panelIds.length <= 1) return;

    // Abort if generating
    if (panel.isGenerating && panel.agentAbortId) {
        fetch(`${API}/tools/agent/abort`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ abort_id: panel.agentAbortId }),
        }).catch(() => {});
    }

    // Close any orphaned worker tabs owned by this head panel
    if (panel.workerPanelMap) {
        for (const [agId, wpId] of Object.entries(panel.workerPanelMap)) {
            if (wpId !== panelId && state.panels[wpId]) {
                const wpDom = document.getElementById(`panel-${wpId}`);
                if (wpDom) wpDom.remove();
                delete state.panels[wpId];
            }
        }
    }

    // Remove panel DOM
    const panelDom = document.getElementById(`panel-${panelId}`);
    if (panelDom) panelDom.remove();

    // Remove from state
    delete state.panels[panelId];

    // Switch to neighbor if this was the active panel
    if (state.activePanelId === panelId) {
        const remaining = Object.keys(state.panels);
        switchToPanel(remaining[remaining.length - 1]);
    }

    _renderAllTabs();
    _updateTabBarVisibility();
}

/**
 * Rename a panel's display name.
 */
export function renamePanel(panelId, newName) {
    const panel = state.panels[panelId];
    if (!panel) return;
    panel.name = newName;
    const tabName = document.querySelector(`.panel-tab[data-panel-id="${panelId}"] .panel-tab-name`);
    if (tabName) tabName.textContent = newName;
}

// ─── DOM Generation ────────────────────────────────────────────────

/**
 * Internal: create panel state + DOM, register in state.panels.
 */
function _createPanelInternal(name) {
    state.panelCounter++;
    const panelId = 'p' + state.panelCounter;
    const panel = createPanelState(panelId, name);
    state.panels[panelId] = panel;

    // Generate and insert DOM
    const container = document.getElementById('panel-container');
    if (container) {
        const div = document.createElement('div');
        div.innerHTML = _renderPanelHTML(panel);
        const panelEl = div.firstElementChild;
        container.appendChild(panelEl);

        // Hide if not active
        if (state.activePanelId && state.activePanelId !== panelId) {
            panelEl.style.display = 'none';
        }

        // Wire up panel input keydown
        const input = document.getElementById(getPanelDomId(panelId, 'input'));
        if (input) {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    window.panelSendMessage(panelId);
                }
            });
            input.addEventListener('input', () => _autoResizePanelInput(panelId));
        }
    }

    // Populate dropdowns
    populatePanelModelDropdown(panelId);
    populatePanelPersonaDropdown(panelId);
    populatePanelRoutingDropdown(panelId);

    return panel;
}

/**
 * Generate the full HTML for a chat panel. All IDs are scoped to panelId.
 */
function _renderPanelHTML(panel) {
    const id = panel.panelId;
    const d = (suffix) => getPanelDomId(id, suffix);

    return `
    <div class="chat-panel" id="panel-${id}" data-panel-id="${id}">
        <div class="chat-header">
            <!-- Regular Chat: System Prompt Selector -->
            <div class="system-prompt-selector" id="${d('regular-prompt-selector')}">
                <button class="prompt-selector-btn" onclick="openPromptModal()">
                    <span class="prompt-icon">🤖</span>
                    <span class="prompt-name" id="${d('prompt-name')}">Default Assistant</span>
                    <span class="dropdown-arrow">▼</span>
                </button>
            </div>
            <!-- Agent Mode Controls -->
            <div class="agent-controls" id="${d('agent-controls')}" style="display: none;">
                <div class="agent-simple-view" id="${d('agent-simple-view')}">
                    <span class="agent-indicator">
                        <span class="agent-icon">🤖</span>
                        <span class="agent-label">Agent Mode</span>
                    </span>
                    <button class="btn-icon btn-tune" onclick="togglePanelAgentAdvanced('${id}')" title="Advanced agent settings">⚙️</button>
                </div>
                <div class="agent-advanced-view hidden" id="${d('agent-advanced-view')}">
                    <div class="persona-selector">
                        <label class="persona-label">Persona:</label>
                        <select id="${d('persona-select')}" onchange="onPanelPersonaChange('${id}')"></select>
                    </div>
                    <button class="btn-text btn-manual-override" onclick="toggleManualOverride('${id}')" id="${d('manual-override-btn')}">Manual Override &#9654;</button>
                    <div class="manual-override-section hidden" id="${d('manual-override')}">
                        <div class="panel-config-row">
                            <label class="panel-config-label">Model:</label>
                            <select id="${d('model-select')}" onchange="onPanelModelChange('${id}')">
                                <option value="">Auto (sidebar)</option>
                            </select>
                        </div>
                        <div class="panel-config-row">
                            <label class="panel-config-label">Routing:</label>
                            <select id="${d('routing-select')}" onchange="onPanelRoutingChange('${id}')">
                                <option value="">Default (global)</option>
                                <option value="__none__">None</option>
                            </select>
                        </div>
                    </div>
                    <button class="btn-icon btn-instructions" onclick="togglePanelInstructions('${id}')" title="Add custom instructions">
                        <span id="${d('instructions-icon')}">📝</span>
                    </button>
                    <button class="btn-icon btn-collapse" onclick="togglePanelAgentAdvanced('${id}')" title="Hide">✕</button>
                </div>
                <!-- Auto toggle — always visible when agent mode is on -->
                <div class="agent-auto-toggle" id="${d('autonomous-container')}">
                    <label class="toggle-switch" title="Let the agent chain tools automatically">
                        <input type="checkbox" id="${d('autonomous-toggle')}" onchange="togglePanelAutonomous('${id}')" checked>
                        <span class="toggle-slider"></span>
                    </label>
                    <span class="toggle-label">Auto</span>
                </div>
            </div>
            <div class="chat-header-actions">
                <div class="agent-mode-toggle">
                    <label class="toggle-switch" title="Enable Agent Mode">
                        <input type="checkbox" id="${d('agent-toggle')}" onchange="togglePanelAgentMode('${id}')">
                        <span class="toggle-slider"></span>
                    </label>
                    <span class="toggle-label">Agent</span>
                </div>
                <button class="btn-icon" onclick="openConversationHistory()" title="Saved Conversations">📂</button>
                <button class="btn-icon" onclick="savePanelConversation('${id}')" title="Save Conversation">💾</button>
                <button class="btn-icon" onclick="clearPanelChat('${id}')" title="Clear Chat">🗑️</button>
            </div>
        </div>
        <!-- Additional Instructions Panel -->
        <div id="${d('instructions-panel')}" class="additional-instructions-panel hidden">
            <div class="instructions-header">
                <span>Additional Instructions</span>
                <span class="instructions-hint">(tone, style, focus)</span>
            </div>
            <textarea id="${d('instructions')}" placeholder="e.g., Be concise. Focus on Python."></textarea>
        </div>
        <!-- Messages -->
        <div id="${d('messages')}" class="chat-messages">
            <div class="welcome-message">
                <h2>AgentNate</h2>
                <p class="welcome-subtitle">Your local AI platform</p>
                <div class="welcome-steps">
                    <div class="welcome-step">
                        <span class="welcome-step-num">1</span>
                        <span>Load a model from the sidebar</span>
                    </div>
                    <div class="welcome-step">
                        <span class="welcome-step-num">2</span>
                        <span>Start chatting</span>
                    </div>
                </div>
            </div>
        </div>
        <!-- Input Area -->
        <div class="chat-input-area">
            <div id="${d('image-preview')}" class="image-preview-area hidden"></div>
            <div id="${d('pdf-preview')}" class="pdf-preview-area hidden"></div>
            <div class="chat-input-row">
                <button id="${d('pdf-btn')}" class="btn-icon pdf-upload-btn" onclick="triggerPanelPdfUpload('${id}')" title="Attach PDF">📄</button>
                <input type="file" id="${d('pdf-input')}" accept=".pdf,application/pdf" multiple style="display: none" onchange="handlePanelPdfSelect(event, '${id}')">
                <textarea id="${d('input')}" class="message-input" placeholder="Load a model to start chatting..." rows="1" disabled></textarea>
                <button id="${d('send-btn')}" class="send-btn" onclick="panelSendMessage('${id}')" disabled><span>Send</span></button>
            </div>
        </div>
        <!-- Agent stop button (hidden by default) -->
        <button id="${d('agent-stop-btn')}" class="agent-stop-btn" style="display: none;" onclick="stopPanelAgent('${id}')">⏹ Stop Agent</button>
    </div>`;
}

// ─── Tab Bar ───────────────────────────────────────────────────────

function _renderAllTabs() {
    const tabsContainer = document.getElementById('panel-tabs');
    if (!tabsContainer) return;

    const panelIds = Object.keys(state.panels);
    tabsContainer.innerHTML = panelIds.map(pid => {
        const p = state.panels[pid];
        const active = pid === state.activePanelId ? ' active' : '';
        const statusIcon = p.isGenerating ? '<span class="panel-tab-spinner"></span>' : '';
        const closeBtn = panelIds.length > 1
            ? `<button class="panel-tab-close" onclick="event.stopPropagation(); closePanel('${pid}')" title="Close">×</button>`
            : '';
        return `<div class="panel-tab${active}" data-panel-id="${pid}" onclick="switchToPanel('${pid}')">
            <span class="panel-tab-name">${_escapeHtml(p.name)}</span>
            ${statusIcon}
            ${closeBtn}
        </div>`;
    }).join('');
}

function _updateTabBarVisibility() {
    const tabBar = document.getElementById('chat-panel-tabs');
    if (!tabBar) return;
    const count = Object.keys(state.panels).length;
    tabBar.style.display = count > 1 ? '' : 'none';
}

/**
 * Update a single panel's tab status indicator (spinner/done/error).
 */
export function setPanelTabStatus(panelId, status) {
    const tab = document.querySelector(`.panel-tab[data-panel-id="${panelId}"]`);
    if (!tab) return;

    // Remove old status classes
    tab.classList.remove('tab-generating', 'tab-done', 'tab-error');

    const existing = tab.querySelector('.panel-tab-spinner');

    if (status === 'generating') {
        tab.classList.add('tab-generating');
        if (!existing) {
            const spinner = document.createElement('span');
            spinner.className = 'panel-tab-spinner';
            tab.querySelector('.panel-tab-name').after(spinner);
        }
    } else {
        if (existing) existing.remove();
        if (status === 'done') {
            tab.classList.add('tab-done');
            setTimeout(() => tab.classList.remove('tab-done'), 2000);
        } else if (status === 'error') {
            tab.classList.add('tab-error');
            setTimeout(() => tab.classList.remove('tab-error'), 3000);
        }
    }
}

export function updatePanelTabLabel(panelId, label) {
    const tab = document.querySelector(`.panel-tab[data-panel-id="${panelId}"]`);
    if (!tab) return;
    const nameEl = tab.querySelector('.panel-tab-name');
    if (nameEl) nameEl.textContent = label;
}

// ─── Dropdown Population ───────────────────────────────────────────

/**
 * Populate a panel's model dropdown from loaded models.
 * Called on panel creation and when models change.
 */
export function populatePanelModelDropdown(panelId) {
    const select = document.getElementById(getPanelDomId(panelId, 'model-select'));
    if (!select) return;

    const panel = state.panels[panelId];
    const currentVal = panel?.instanceId || '';

    select.innerHTML = '<option value="">Auto (sidebar)</option>';
    for (const [id, model] of Object.entries(state.models)) {
        if (!id.includes('-')) continue; // Skip non-UUID entries (provider model lists)
        if (model.status === 'error') continue;
        const name = model.name || model.model || id;
        select.innerHTML += `<option value="${id}">${name}</option>`;
    }

    select.value = currentVal;
}

/**
 * Populate a panel's persona dropdown.
 */
export function populatePanelPersonaDropdown(panelId) {
    const select = document.getElementById(getPanelDomId(panelId, 'persona-select'));
    if (!select) return;

    const panel = state.panels[panelId];
    if (_cachedPersonas.length === 0) {
        // Fetch and cache — will re-populate when data arrives
        _fetchAndCachePersonas().then(() => populatePanelPersonaDropdown(panelId));
        return;
    }

    select.innerHTML = '<option value="auto" title="Automatic persona selection based on your request">Auto (Recommended)</option>';
    for (const persona of _cachedPersonas) {
        const toolHint = persona.tools?.includes('all') ? 'All Tools' :
            persona.tools?.length > 0 ? `${persona.tools.length} tool groups` : 'Chat only';
        select.innerHTML += `<option value="${persona.id}" title="${_escapeHtml(persona.description || '')}">${persona.name} (${toolHint})</option>`;
    }
    select.value = panel?.selectedPersonaId || 'auto';
}

/**
 * Populate a panel's routing preset dropdown.
 */
export async function populatePanelRoutingDropdown(panelId) {
    const select = document.getElementById(getPanelDomId(panelId, 'routing-select'));
    if (!select) return;

    const panel = state.panels[panelId];
    const currentVal = panel?.routingPresetId || '';

    try {
        const resp = await fetch(`${API}/routing/presets`);
        const data = await resp.json();
        const presets = data.presets || [];

        select.innerHTML = '<option value="">Default (global)</option><option value="__none__">None</option>';
        for (const preset of presets) {
            select.innerHTML += `<option value="${preset.id}">${_escapeHtml(preset.name)}</option>`;
        }
        select.value = currentVal;
    } catch (e) {
        // Keep existing options on error
    }
}

/**
 * Refresh all panel model dropdowns (called when models load/unload).
 */
export function refreshAllPanelModelDropdowns() {
    for (const panelId of Object.keys(state.panels)) {
        populatePanelModelDropdown(panelId);
    }
}

/**
 * Refresh all panel routing dropdowns (called when presets change).
 */
export function refreshAllPanelRoutingDropdowns() {
    for (const panelId of Object.keys(state.panels)) {
        populatePanelRoutingDropdown(panelId);
    }
}

// ─── Panel Event Handlers ──────────────────────────────────────────

export function onPanelModelChange(panelId) {
    const select = document.getElementById(getPanelDomId(panelId, 'model-select'));
    const panel = state.panels[panelId];
    if (!select || !panel) return;
    panel.instanceId = select.value || null;
    _updatePanelInputState(panelId);
}

export function onPanelPersonaChange(panelId) {
    const select = document.getElementById(getPanelDomId(panelId, 'persona-select'));
    const panel = state.panels[panelId];
    if (!select || !panel) return;
    panel.selectedPersonaId = select.value;

    // Update badge
    const badge = document.getElementById(getPanelDomId(panelId, 'persona-badge'));
    if (badge) {
        if (select.value === 'auto') {
            badge.textContent = 'Auto';
        } else {
            const persona = _cachedPersonas.find(p => p.id === select.value);
            badge.textContent = persona ? persona.name : select.value;
        }
    }
}

export function onPanelRoutingChange(panelId) {
    const select = document.getElementById(getPanelDomId(panelId, 'routing-select'));
    const panel = state.panels[panelId];
    if (!select || !panel) return;
    panel.routingPresetId = select.value === '__none__' ? '__none__' : (select.value || null);

    // Update routing badge
    const badge = document.getElementById(getPanelDomId(panelId, 'routing-badge'));
    if (badge) {
        if (panel.routingPresetId && panel.routingPresetId !== '__none__') {
            const option = select.options[select.selectedIndex];
            badge.textContent = '🔀 ' + (option?.textContent || 'Routed');
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    }
}

export function togglePanelAgentAdvanced(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;
    panel.agentAdvancedOpen = !panel.agentAdvancedOpen;

    const simple = document.getElementById(getPanelDomId(panelId, 'agent-simple-view'));
    const advanced = document.getElementById(getPanelDomId(panelId, 'agent-advanced-view'));
    if (simple) simple.style.display = panel.agentAdvancedOpen ? 'none' : '';
    if (advanced) advanced.classList.toggle('hidden', !panel.agentAdvancedOpen);
}

export function togglePanelInstructions(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;
    panel.showAdditionalInstructions = !panel.showAdditionalInstructions;
    const el = document.getElementById(getPanelDomId(panelId, 'instructions-panel'));
    if (el) el.classList.toggle('hidden', !panel.showAdditionalInstructions);
}

export function togglePanelAutonomous(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;
    const toggle = document.getElementById(getPanelDomId(panelId, 'autonomous-toggle'));
    panel.autonomous = toggle?.checked || false;
}

export function toggleManualOverride(panelId) {
    const section = document.getElementById(getPanelDomId(panelId, 'manual-override'));
    const btn = document.getElementById(getPanelDomId(panelId, 'manual-override-btn'));
    if (!section) return;
    const isHidden = section.classList.toggle('hidden');
    if (btn) btn.innerHTML = isHidden ? 'Manual Override &#9654;' : 'Manual Override &#9660;';
}

// ─── Panel UI State ────────────────────────────────────────────────

/**
 * Update a panel's input enabled/disabled state based on model availability.
 */
export function _updatePanelInputState(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    const modelId = panel.instanceId || state.currentModel;
    const hasModel = !!modelId;
    const disabled = !hasModel || panel.isGenerating;

    const input = document.getElementById(getPanelDomId(panelId, 'input'));
    const sendBtn = document.getElementById(getPanelDomId(panelId, 'send-btn'));

    if (input) {
        input.disabled = disabled;
        input.placeholder = hasModel ? 'Type a message...' : 'Load a model to start chatting...';
    }
    if (sendBtn) sendBtn.disabled = disabled;
}

/**
 * Update all panels' input states (called when global model changes).
 */
export function updateAllPanelInputStates() {
    for (const panelId of Object.keys(state.panels)) {
        _updatePanelInputState(panelId);
    }
}

// ─── Legacy State Sync ─────────────────────────────────────────────

/**
 * Mirror the active panel's state to legacy global fields.
 * This keeps backward compat with modules that still read state.messages, state.agentMode, etc.
 */
function _syncLegacyState(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;
    state.messages = panel.messages;
    state.agentMode = panel.agentMode;
    state.autonomous = panel.autonomous;
    state.selectedPersonaId = panel.selectedPersonaId;
    state.isGenerating = panel.isGenerating;
    state.agentAbortId = panel.agentAbortId;
    state.conversationId = panel.conversationId;
    state.pendingImages = panel.pendingImages;
    state.loadedPdfFiles = panel.loadedPdfFiles;
    state.agentPlan = panel.agentPlan;
    state.agentThinking = panel.agentThinking;
}

// ─── Helpers ───────────────────────────────────────────────────────

function _autoResizePanelInput(panelId) {
    const input = document.getElementById(getPanelDomId(panelId, 'input'));
    if (!input) return;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
}

async function _fetchAndCachePersonas() {
    try {
        const resp = await fetch(`${API}/tools/personas`);
        const data = await resp.json();
        if (data.personas && data.personas.length > 0) {
            _cachedPersonas = data.personas.sort((a, b) => a.name.localeCompare(b.name));
        }
    } catch (e) {
        console.error('Failed to fetch personas for panels:', e);
    }
}

function _escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Pre-cache personas so panel dropdowns populate instantly.
 */
export async function initPersonaCache() {
    await _fetchAndCachePersonas();
}

/**
 * Get the cached personas list (for other modules).
 */
export function getCachedPersonas() {
    return _cachedPersonas;
}
