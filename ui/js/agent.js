/**
 * Agent mode, personas, tool calling, SSE streaming for AgentNate UI.
 * All functions are panel-aware: they take a panelId parameter.
 * Legacy global wrappers dispatch to the active panel.
 */

import { state, API, getActivePanel, getPanelDomId } from './state.js';
import { log, escapeHtml, linkifyText, updateDebugStatus } from './utils.js';

// Lazy chat module import (circular dependency)
let _chatModule = null;
async function getChatModule() {
    if (!_chatModule) _chatModule = await import('./chat.js');
    return _chatModule;
}

// Lazy panels module import (avoid hard circular coupling)
let _panelsModule = null;
async function getPanelsModule() {
    if (!_panelsModule) _panelsModule = await import('./panels.js');
    return _panelsModule;
}

// Worker event SSE streams per head panel (replaces HTTP polling for real-time delivery).
const _workerEventSources = {};

// Per-worker-panel token batching cache (avoid DOM updates on every token).
const _workerTokenCache = {};

function _stopWorkerSSE(panelId) {
    const es = _workerEventSources[panelId];
    if (es) {
        es.close();
        delete _workerEventSources[panelId];
    }
}

function _ensureWorkerSSE(panelId) {
    if (_workerEventSources[panelId]) return;
    const panel = state.panels[panelId];
    if (!panel || !panel.conversationId) return;

    const url = `${API}/tools/agent/workers/${encodeURIComponent(panel.conversationId)}/stream`;
    const es = new EventSource(url);
    _workerEventSources[panelId] = es;

    es.onmessage = async (e) => {
        if (!e.data || e.data.startsWith(':')) return;
        let data;
        try {
            data = JSON.parse(e.data);
        } catch { return; }

        switch (data.type) {
            case 'sub_agent_update':
                if (data.agents) {
                    await _updatePanelSubAgentPanel(panelId, data.agents);
                }
                break;
            case 'worker_events':
                if (data.events && data.events.length > 0) {
                    await _applyWorkerEvents(panelId, data.events);
                }
                break;
            case 'workers_done':
                _stopWorkerSSE(panelId);
                break;
        }
    };

    es.onerror = () => {
        // EventSource auto-reconnects, but if panel is gone, close
        const p = state.panels[panelId];
        if (!p) _stopWorkerSSE(panelId);
    };
}

// ==================== Routing Badge ====================

export async function refreshPanelRoutingBadge(panelId) {
    const badge = document.getElementById(getPanelDomId(panelId, 'routing-badge'));
    if (!badge) return;
    try {
        const resp = await fetch(`${API}/routing/status`);
        if (!resp.ok) throw new Error(resp.statusText);
        const data = await resp.json();
        if (data.routing_enabled && data.active_preset_name) {
            badge.textContent = `\u2194 ${data.active_preset_name}`;
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    } catch {
        badge.classList.add('hidden');
    }
}

// Legacy wrapper
export async function refreshRoutingBadge() {
    if (state.activePanelId) return refreshPanelRoutingBadge(state.activePanelId);
}

// ==================== Tool Display Helpers ====================

const TOOL_ICONS = {
    // Web & Search
    web_search: '\u{1F50D}', fetch_url: '\u{1F310}', scrape_page: '\u{1F310}',
    // Browser
    browser_navigate: '\u{1F310}', browser_screenshot: '\u{1F4F8}', browser_click: '\u{1F5B1}',
    browser_type: '\u{2328}', browser_close: '\u{1F310}',
    // Code
    execute_code: '\u{1F40D}', run_python: '\u{1F40D}', execute_shell: '\u{1F4BB}',
    // Files
    read_file: '\u{1F4C1}', write_file: '\u{1F4C1}', list_files: '\u{1F4C2}',
    delete_file: '\u{1F5D1}', move_file: '\u{1F4C1}', copy_file: '\u{1F4C1}',
    // n8n / Automation
    spawn_n8n: '\u{26A1}', stop_n8n: '\u{26A1}', deploy_workflow: '\u{26A1}',
    execute_workflow: '\u{26A1}', list_workflows: '\u{26A1}',
    // ComfyUI
    comfyui_status: '\u{1F3A8}', comfyui_install: '\u{1F3A8}', comfyui_start_api: '\u{1F3A8}',
    comfyui_stop_api: '\u{1F3A8}', comfyui_list_instances: '\u{1F3A8}', comfyui_add_instance: '\u{1F3A8}',
    comfyui_start_instance: '\u{1F3A8}', comfyui_stop_instance: '\u{1F3A8}', comfyui_list_models: '\u{1F3A8}',
    // Models
    load_model: '\u{1F9E0}', unload_model: '\u{1F9E0}', list_models: '\u{1F9E0}',
    model_info: '\u{1F9E0}', quick_setup: '\u{1F9E0}',
    // Communication
    send_email: '\u{1F4E8}', send_notification: '\u{1F514}',
    // Data
    analyze_data: '\u{1F4CA}', parse_json: '\u{1F4CA}', parse_csv: '\u{1F4CA}',
    // Vision
    describe_image: '\u{1F441}', ocr_image: '\u{1F441}',
    // System
    system_info: '\u{1F5A5}', get_datetime: '\u{1F552}', calculate: '\u{1F522}',
};

const TOOL_FRIENDLY_NAMES = {
    web_search: 'Web Search', fetch_url: 'Fetch URL', scrape_page: 'Scrape Page',
    browser_navigate: 'Navigate', browser_screenshot: 'Screenshot', browser_click: 'Click',
    browser_type: 'Type Text', browser_close: 'Close Browser',
    execute_code: 'Run Code', run_python: 'Run Python', execute_shell: 'Shell Command',
    read_file: 'Read File', write_file: 'Write File', list_files: 'List Files',
    delete_file: 'Delete File', move_file: 'Move File', copy_file: 'Copy File',
    spawn_n8n: 'Start n8n', stop_n8n: 'Stop n8n', deploy_workflow: 'Deploy Workflow',
    execute_workflow: 'Run Workflow', list_workflows: 'List Workflows',
    comfyui_status: 'ComfyUI Status', comfyui_install: 'Install ComfyUI',
    comfyui_start_api: 'Start ComfyUI API', comfyui_stop_api: 'Stop ComfyUI API',
    comfyui_list_instances: 'List Instances', comfyui_add_instance: 'Add Instance',
    comfyui_start_instance: 'Start Instance', comfyui_stop_instance: 'Stop Instance',
    comfyui_list_models: 'List Models',
    load_model: 'Load Model', unload_model: 'Unload Model', list_models: 'List Models',
    model_info: 'Model Info', quick_setup: 'Quick Setup',
    send_email: 'Send Email', send_notification: 'Notify',
    analyze_data: 'Analyze Data', parse_json: 'Parse JSON', parse_csv: 'Parse CSV',
    describe_image: 'Describe Image', ocr_image: 'OCR Image',
    system_info: 'System Info', get_datetime: 'Date/Time', calculate: 'Calculate',
};

function getToolIcon(toolName) {
    const baseName = toolName.replace(/^Tool #\d+:\s*/, '');
    return TOOL_ICONS[baseName] || '\u{1F527}';
}

function getToolFriendlyName(toolName) {
    const baseName = toolName.replace(/^Tool #\d+:\s*/, '');
    const numPrefix = toolName.match(/^(Tool #\d+):\s*/);
    const friendly = TOOL_FRIENDLY_NAMES[baseName] || baseName.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    return numPrefix ? `${numPrefix[1]}: ${friendly}` : friendly;
}

function summarizeArgs(toolName, args) {
    if (!args || Object.keys(args).length === 0) return '';
    const baseName = toolName.replace(/^Tool #\d+:\s*/, '');

    if (baseName === 'web_search' && args.query) return `"${args.query}"`;
    if (baseName === 'fetch_url' && args.url) return truncate(args.url, 60);
    if ((baseName === 'read_file' || baseName === 'write_file') && args.path) return truncate(args.path, 60);
    if (baseName === 'execute_code' && args.language) return args.language;
    if (baseName === 'load_model' && args.model_name) return truncate(args.model_name, 50);
    if (baseName === 'deploy_workflow' && args.name) return args.name;
    if (baseName === 'browser_navigate' && args.url) return truncate(args.url, 60);

    const entries = Object.entries(args);
    if (entries.length === 1) {
        const [key, val] = entries[0];
        if (typeof val === 'string') return `${key}: ${truncate(val, 50)}`;
    }
    return `${entries.length} params`;
}

function truncate(str, max) {
    if (!str) return '';
    return str.length > max ? str.slice(0, max - 3) + '...' : str;
}

// ==================== Welcome Message ====================

export function showPanelAgentWelcome(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;
    if (container.querySelector('.agent-welcome')) return;

    const modelId = panel.instanceId || state.currentModel;
    const hasModel = !!modelId;

    const el = document.createElement('div');
    el.className = 'agent-welcome';
    el.innerHTML = `
        <div class="agent-welcome-header">
            <span class="agent-welcome-icon">\u{1F916}</span>
            <span class="agent-welcome-title">Agent Mode Active</span>
        </div>
        <p class="agent-welcome-desc">I can use tools to help you accomplish tasks. Here are some things I can do:</p>
        <div class="agent-welcome-capabilities">
            <div class="agent-capability"><span>\u{1F50D}</span> Search the web &amp; fetch pages</div>
            <div class="agent-capability"><span>\u{1F40D}</span> Write &amp; run code</div>
            <div class="agent-capability"><span>\u{1F4C1}</span> Read, write &amp; manage files</div>
            <div class="agent-capability"><span>\u{26A1}</span> Build &amp; run n8n automations</div>
            <div class="agent-capability"><span>\u{1F3A8}</span> Generate images with ComfyUI</div>
            <div class="agent-capability"><span>\u{1F9E0}</span> Load &amp; manage AI models</div>
        </div>
        <div class="agent-quick-actions">
            <button onclick="fillPanelAgentPrompt('${panelId}', 'What can you do? List your capabilities.')" ${!hasModel ? 'disabled' : ''}>\u{2753} What can you do?</button>
            <button onclick="fillPanelAgentPrompt('${panelId}', 'Search the web for the latest AI news and summarize the top 3 stories.')" ${!hasModel ? 'disabled' : ''}>\u{1F50D} Web search</button>
            <button onclick="fillPanelAgentPrompt('${panelId}', 'Check system status - what models are loaded, is n8n running, ComfyUI status?')" ${!hasModel ? 'disabled' : ''}>\u{1F5A5} System check</button>
            <button onclick="fillPanelAgentPrompt('${panelId}', 'Help me set up a workflow: load a model and start n8n so I can build automations.')" ${!hasModel ? 'disabled' : ''}>\u{1F680} Quick setup</button>
        </div>
        ${!hasModel ? '<p class="agent-welcome-tip" style="color: var(--warning);">Load a model first using the <strong>+</strong> button in the sidebar or a <strong>Quick Load</strong> preset.</p>' : ''}
        <p class="agent-welcome-tip">Tip: <strong>Auto</strong> is on — the agent will chain tools together automatically. Use ⚙️ for advanced settings.</p>
    `;

    container.appendChild(el);
    _scrollPanelToBottom(panelId);
}

export function fillPanelAgentPrompt(panelId, text) {
    const panel = state.panels[panelId];
    if (!panel) return;
    const modelId = panel.instanceId || state.currentModel;
    if (!modelId) {
        log('Load a model first to use agent features', 'warning');
        return;
    }
    const input = document.getElementById(getPanelDomId(panelId, 'input'));
    if (input) {
        input.value = text;
        input.focus();
        input.dispatchEvent(new Event('input'));
    }
}

// Legacy wrappers
export function showAgentWelcome() {
    if (state.activePanelId) showPanelAgentWelcome(state.activePanelId);
}
export function fillAgentPrompt(text) {
    if (state.activePanelId) fillPanelAgentPrompt(state.activePanelId, text);
}

// ==================== Stop Agent ====================

function _showPanelStopButton(panelId) {
    const btn = document.getElementById(getPanelDomId(panelId, 'agent-stop-btn'));
    if (btn) btn.style.display = 'block';
}

function _hidePanelStopButton(panelId) {
    const btn = document.getElementById(getPanelDomId(panelId, 'agent-stop-btn'));
    if (btn) btn.style.display = 'none';
}

export async function stopPanelAgent(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    if (panel.agentAbortId) {
        try {
            await fetch(`${API}/tools/agent/abort`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ abort_id: panel.agentAbortId })
            });
            console.log('[AgentMode] Abort signal sent:', panel.agentAbortId);
        } catch (e) {
            console.error('[AgentMode] Failed to send abort:', e);
        }
    }
    panel.agentAbortId = null;
    _hidePanelStopButton(panelId);
    _stopWorkerSSE(panelId);

    // Close any worker tabs owned by this head panel
    if (panel.workerPanelMap) {
        getPanelsModule().then(panels => {
            for (const [agId, wpId] of Object.entries(panel.workerPanelMap)) {
                if (state.panels[wpId]) panels.forceClosePanel(wpId);
            }
            panel.workerPanelMap = {};
            panel.workerStatusMap = {};
            panel.workerCompletionNotified = {};
        });
    }
}

// Legacy wrapper
export async function stopAgent() {
    if (state.activePanelId) return stopPanelAgent(state.activePanelId);
}

// ==================== Core Agent Functions ====================

export function togglePanelAgentMode(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    const toggle = document.getElementById(getPanelDomId(panelId, 'agent-toggle'));
    if (!toggle) return;

    if (panel.isGenerating) {
        toggle.checked = panel.agentMode;
        return;
    }

    panel.agentMode = toggle.checked;

    const regularPromptSelector = document.getElementById(getPanelDomId(panelId, 'regular-prompt-selector'));
    const agentControls = document.getElementById(getPanelDomId(panelId, 'agent-controls'));
    const simpleView = document.getElementById(getPanelDomId(panelId, 'agent-simple-view'));
    const advancedView = document.getElementById(getPanelDomId(panelId, 'agent-advanced-view'));
    const toggleContainer = toggle.closest('.agent-mode-toggle');
    const header = toggle.closest('.chat-header');

    if (panel.agentMode) {
        if (toggleContainer) toggleContainer.classList.add('active');
        if (header) header.classList.add('agent-mode');
        if (regularPromptSelector) regularPromptSelector.style.display = 'none';
        if (agentControls) agentControls.style.display = 'flex';
        if (simpleView) simpleView.classList.remove('hidden');
        if (advancedView) advancedView.classList.add('hidden');
        panel.agentAdvancedOpen = false;
        // Default Auto ON when entering agent mode
        panel.autonomous = true;
        const autoToggle = document.getElementById(getPanelDomId(panelId, 'autonomous-toggle'));
        if (autoToggle) autoToggle.checked = true;
        log('Agent mode enabled - tool calling active (Auto ON)', 'info');
        showPanelAgentWelcome(panelId);
    } else {
        if (toggleContainer) toggleContainer.classList.remove('active');
        if (header) header.classList.remove('agent-mode');
        if (regularPromptSelector) regularPromptSelector.style.display = 'block';
        if (agentControls) agentControls.style.display = 'none';
        const instructionsPanel = document.getElementById(getPanelDomId(panelId, 'instructions-panel'));
        if (instructionsPanel) instructionsPanel.classList.add('hidden');
        if (simpleView) simpleView.classList.remove('hidden');
        if (advancedView) advancedView.classList.add('hidden');
        panel.showAdditionalInstructions = false;
        panel.agentAdvancedOpen = false;
        panel.autonomous = true;  // Keep default for next time agent mode is enabled
        const autoToggle = document.getElementById(getPanelDomId(panelId, 'autonomous-toggle'));
        if (autoToggle) autoToggle.checked = true;

        // Remove agent-specific UI elements
        const container = document.getElementById(getPanelDomId(panelId, 'messages'));
        if (container) {
            const welcome = container.querySelector('.agent-welcome');
            if (welcome) welcome.remove();
            // Remove plan cards and thinking cards (orphaned agent UI)
            container.querySelectorAll('.agent-plan-card, .agent-thinking-card').forEach(el => el.remove());
        }
        // Remove stop agent button if present
        const stopBtn = document.getElementById(getPanelDomId(panelId, 'stop-agent'));
        if (stopBtn) stopBtn.remove();

        // Clear agent plan state
        panel.agentPlan = null;
        panel.agentThinking = null;

        log('Agent mode disabled', 'info');
    }

    // Sync legacy state if active panel
    if (panelId === state.activePanelId) {
        state.agentMode = panel.agentMode;
        state.autonomous = panel.autonomous;
    }

    getChatModule().then(chat => chat.updatePanelUI(panelId));
}

// Legacy wrapper
export function toggleAgentMode() {
    if (state.activePanelId) togglePanelAgentMode(state.activePanelId);
}

export function toggleAutonomous() {
    if (state.activePanelId) {
        const panel = state.panels[state.activePanelId];
        if (!panel) return;
        const toggle = document.getElementById(getPanelDomId(state.activePanelId, 'autonomous-toggle'));
        if (toggle) {
            panel.autonomous = toggle.checked;
            state.autonomous = panel.autonomous;
        }
    }
}

export function onPersonaChange() {
    if (state.activePanelId) {
        import('./panels.js').then(m => m.onPanelPersonaChange(state.activePanelId));
    }
}

export function toggleAgentAdvanced() {
    if (state.activePanelId) {
        import('./panels.js').then(m => m.togglePanelAgentAdvanced(state.activePanelId));
    }
}

export function toggleAdditionalInstructions() {
    if (state.activePanelId) {
        import('./panels.js').then(m => m.togglePanelInstructions(state.activePanelId));
    }
}

export function updateInstructionsButtonState() {
    // No-op — per-panel instructions don't need global state tracking
}

export function getAdditionalInstructions() {
    if (!state.activePanelId) return '';
    const textarea = document.getElementById(getPanelDomId(state.activePanelId, 'instructions'));
    return textarea ? textarea.value.trim() : '';
}

// ==================== Load Personas (legacy) ====================

export async function loadPersonas() {
    // Delegate to panel persona cache
    const { initPersonaCache } = await import('./panels.js');
    await initPersonaCache();
}

// ==================== Send Agent Message ====================

export async function sendPanelAgentMessage(panelId, content) {
    const panel = state.panels[panelId];
    if (!panel) return;

    console.log('[AgentMode] Sending message via SSE for panel:', panelId);

    // Init scroll tracking for this panel (once).
    _initPanelScrollTracking(panelId);
    // Force scroll to bottom on new user message.
    _forceScrollPanelToBottom(panelId);

    // Head tab keeps its stable name (e.g. "Chat 1"). No renaming to message text.

    const chat = await getChatModule();
    const { getEffectiveParams } = await import('./model-settings.js');

    chat.addPanelMessage(panelId, 'assistant', '');

    let currentContent = '';

    // Generate abort ID and show stop button
    const abortId = 'agent_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    panel.agentAbortId = abortId;
    if (panelId === state.activePanelId) state.agentAbortId = abortId;
    _showPanelStopButton(panelId);

    // Update tab status
    import('./panels.js').then(m => m.setPanelTabStatus(panelId, 'generating'));

    try {
        const modelId = panel.instanceId || state.currentModel;
        const params = getEffectiveParams(modelId);
        const instructionsEl = document.getElementById(getPanelDomId(panelId, 'instructions'));
        const additionalInstructions = instructionsEl ? instructionsEl.value.trim() : '';

        const body = {
            message: content,
            instance_id: modelId,
            conversation_id: panel.conversationId,
            persona_id: panel.selectedPersonaId || 'auto',
            additional_instructions: additionalInstructions || null,
            max_tool_calls: 25,
            autonomous: panel.autonomous,
            abort_id: abortId,
            params: params,
        };
        // Per-panel routing preset
        if (panel.routingPresetId && panel.routingPresetId !== '__none__') {
            body.routing_preset_id = panel.routingPresetId;
        }

        console.log('[AgentMode] Panel:', panelId, 'persona:', panel.selectedPersonaId,
            additionalInstructions ? '+ custom instructions' : '',
            panel.routingPresetId ? `routing: ${panel.routingPresetId}` : '');

        const resp = await fetch(`${API}/tools/agent/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!resp.ok) {
            const errText = await resp.text().catch(() => '');
            throw new Error(`Agent stream failed (${resp.status}): ${errText.slice(0, 300)}`);
        }
        if (!resp.body) {
            throw new Error('Agent stream returned no response body');
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let cachedTextEl = null;
        let cachedMsgIdx = -1;
        let sseBuffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (!state.panels[panelId]) break; // Panel closed mid-stream

            sseBuffer += decoder.decode(value, { stream: true });
            const events = sseBuffer.split(/\r?\n\r?\n/);
            sseBuffer = events.pop() || '';

            for (const eventChunk of events) {
                const lines = eventChunk.split('\n');
                for (const line of lines) {
                    if (!line.startsWith('data:')) continue;
                    try {
                        const data = JSON.parse(line.slice(5).trimStart());

                        switch (data.type) {
                            case 'conv_id':
                                panel.conversationId = data.conversation_id;
                                if (panelId === state.activePanelId) {
                                    state.conversationId = data.conversation_id;
                                }
                                _ensureWorkerSSE(panelId);
                                if (data.instance_id) {
                                    panel.resolvedHeadInstanceId = data.instance_id;
                                }
                                // If this panel is using auto persona, reflect resolved persona in badge.
                                if (panel.selectedPersonaId === 'auto' && data.resolved_persona_id) {
                                    const badge = document.getElementById(getPanelDomId(panelId, 'persona-badge'));
                                    if (badge) {
                                        const resolved = data.resolved_persona_id
                                            .replace(/_/g, ' ')
                                            .replace(/\b\w/g, c => c.toUpperCase());
                                        badge.textContent = `Auto: ${resolved}`;
                                    }
                                }
                                if (data.head_model_forced) {
                                    _addPanelAgentStatusMessage(panelId, 'Head agent is pinned to OpenRouter for orchestration.');
                                }
                                console.log('[AgentMode] Conversation ID:', data.conversation_id);
                                break;

                            case 'token':
                                currentContent += data.text;
                                const lastIdx = panel.messages.length - 1;
                                if (lastIdx >= 0) {
                                    panel.messages[lastIdx].content = currentContent;
                                    if (cachedMsgIdx !== lastIdx) {
                                        const container = document.getElementById(getPanelDomId(panelId, 'messages'));
                                        if (container) {
                                            const msgEl = container.querySelector(`.message[data-index="${lastIdx}"]`);
                                            cachedTextEl = msgEl ? msgEl.querySelector('.message-text') : null;
                                        }
                                        cachedMsgIdx = lastIdx;
                                    }
                                    if (cachedTextEl) {
                                        cachedTextEl.innerHTML = linkifyText(currentContent);
                                    }
                                }
                                _scrollPanelToBottom(panelId);
                                break;

                            case 'plan':
                                // Stash plan — it will be rendered in the worker tab
                                // when delegation creates it. If no delegation (direct
                                // execution), render on head tab.
                                panel._pendingPlan = data.plan;
                                break;

                            case 'thinking':
                                _addPanelThinkingMessage(panelId, data.content);
                                break;

                            case 'tool_executing':
                                _updatePanelToolProgress(panelId, data.tool, data.elapsed);
                                break;

                            case 'tool_call':
                                _removePanelToolProgress(panelId);
                                addPanelToolCallMessage(panelId, data.tool, data.arguments, data.result, data.tool_number);
                                currentContent = '';
                                chat.addPanelMessage(panelId, 'assistant', '');
                                cachedTextEl = null;
                                cachedMsgIdx = -1;
                                break;

                            case 'continuing':
                                console.log(`[AgentMode] Autonomous: continuing after ${data.tool_calls_so_far} tool calls`);
                                const contIdx = panel.messages.length - 1;
                                if (contIdx >= 0) {
                                    panel.messages[contIdx].content = '\u{1F504} *Continuing to next step...*';
                                    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
                                    if (container) {
                                        const contMsgEl = container.querySelector(`.message[data-index="${contIdx}"]`);
                                        if (contMsgEl) {
                                            const contTextEl = contMsgEl.querySelector('.message-text');
                                            if (contTextEl) {
                                                contTextEl.innerHTML = '<em>\u{1F504} Continuing to next step...</em>';
                                            }
                                        }
                                    }
                                }
                                currentContent = '';
                                cachedTextEl = null;
                                cachedMsgIdx = -1;
                                break;

                            case 'followup_start':
                                currentContent = '';
                                cachedTextEl = null;
                                cachedMsgIdx = -1;
                                break;

                            case 'error':
                                _removePanelToolProgress(panelId);
                                chat.showPanelError(panelId, data.error);
                                break;

                            case 'aborted':
                                _removePanelToolProgress(panelId);
                                console.log('[AgentMode] Agent aborted by user');
                                _addPanelAgentStatusMessage(panelId, 'Agent stopped by user');
                                break;

                            case 'sub_agent_update':
                                if (data.agents) {
                                    await _updatePanelSubAgentPanel(panelId, data.agents);
                                }
                                if (data.events) {
                                    await _applyWorkerEvents(panelId, data.events);
                                }
                                _ensureWorkerSSE(panelId);
                                break;

                            case 'head_heartbeat':
                                _addPanelAgentStatusMessage(
                                    panelId,
                                    `Supervisor check: ${data.running_workers || 0} workers running` +
                                    ((data.nudged || 0) > 0 ? `, nudged ${data.nudged}` : '')
                                );
                                break;

                            case 'ask_user':
                                _showPanelAskUserCard(panelId, data.question, data.options, data.tool_number);
                                break;

                            case 'done':
                                _removePanelToolProgress(panelId);
                                if (data.delegated && panel.workerPanelMap && Object.keys(panel.workerPanelMap).length > 0) {
                                    // Delegation mode: plan goes to worker tab, head stays clean
                                    _ensureWorkerSSE(panelId);
                                } else {
                                    // Direct execution: render stashed plan on head tab
                                    if (panel._pendingPlan) {
                                        _addPanelPlanMessage(panelId, panel._pendingPlan);
                                        delete panel._pendingPlan;
                                    }
                                    _removePanelSubAgentPanel(panelId);
                                }
                                break;
                        }
                    } catch (e) {
                        console.debug('[AgentMode] SSE parse skip:', line.slice(6, 80));
                    }
                }
            }
        }

        // Flush any remaining complete event in buffer.
        if (sseBuffer.trim().startsWith('data: ')) {
            try {
                const data = JSON.parse(sseBuffer.trim().slice(6));
                if (data.type === 'done') {
                    _removePanelToolProgress(panelId);
                    if (!(panel.workerPanelMap && Object.keys(panel.workerPanelMap).length > 0)) {
                        _removePanelSubAgentPanel(panelId);
                    }
                }
            } catch (_) {
                // Ignore trailing partial frame.
            }
        }

    } catch (e) {
        console.error('[AgentMode] Error:', e);
        const chat2 = await getChatModule();
        chat2.showPanelError(panelId, 'Agent communication failed: ' + e.message);
    }

    if (state.panels[panelId]) {
        _pruneTrailingEmptyAssistant(panelId);
        state.panels[panelId].agentAbortId = null;
    }
    _hidePanelStopButton(panelId);

    const chat3 = await getChatModule();
    chat3.finishPanelGeneration(panelId);
}

function _pruneTrailingEmptyAssistant(panelId) {
    const panel = state.panels[panelId];
    if (!panel || panel.messages.length === 0) return;
    const lastIdx = panel.messages.length - 1;
    const last = panel.messages[lastIdx];
    if (!last || last.role !== 'assistant') return;
    const content = (last.content || '').trim();
    if (content.length > 0) return;

    // Don't prune if it contains an error indicator — the error IS the message
    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;
    const msgEl = container.querySelector(`.message[data-index="${lastIdx}"]`);
    if (msgEl && msgEl.querySelector('.error-indicator')) return;

    panel.messages.pop();
    if (msgEl) msgEl.remove();
}

// Legacy wrapper
export async function sendAgentModeMessage(content) {
    if (state.activePanelId) return sendPanelAgentMessage(state.activePanelId, content);
}

// ==================== Panel-Scoped Scroll ====================

// Track whether user has manually scrolled up (per panel).
const _userScrolledUp = {};

function _initPanelScrollTracking(panelId) {
    if (_userScrolledUp[panelId] !== undefined) return;
    _userScrolledUp[panelId] = false;
    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;
    container.addEventListener('scroll', () => {
        const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
        // If user scrolled more than 80px from bottom, pause auto-scroll.
        _userScrolledUp[panelId] = distFromBottom > 80;
    });
}

function _scrollPanelToBottom(panelId) {
    // Don't auto-scroll if user has scrolled up to read history.
    if (_userScrolledUp[panelId]) return;
    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (container) container.scrollTop = container.scrollHeight;
}

// Force scroll (used when user sends a new message — always scroll).
function _forceScrollPanelToBottom(panelId) {
    _userScrolledUp[panelId] = false;
    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (container) container.scrollTop = container.scrollHeight;
}

// Legacy wrapper
export function scrollChatToBottom() {
    if (state.activePanelId) _forceScrollPanelToBottom(state.activePanelId);
}

// ==================== Tool Progress Indicator ====================

function _updatePanelToolProgress(panelId, toolName, elapsed) {
    const chatMessages = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!chatMessages) return;

    const indicatorId = `panel-${panelId}-tool-progress`;
    let indicator = document.getElementById(indicatorId);
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = indicatorId;
        indicator.className = 'tool-progress-indicator';
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    const friendlyName = toolName.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    indicator.innerHTML = `
        <div class="tool-progress-spinner"></div>
        <span>Executing <strong>${friendlyName}</strong>... ${Math.floor(elapsed)}s</span>
    `;
}

function _removePanelToolProgress(panelId) {
    const indicator = document.getElementById(`panel-${panelId}-tool-progress`);
    if (indicator) indicator.remove();
}

// ==================== Tool Call Display ====================

export function addPanelToolCallMessage(panelId, toolName, args, result, toolNumber) {
    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;

    const msgEl = document.createElement('div');
    const isSuccess = result && (result.success === true || result.status === 'success' || !result.error);
    msgEl.className = `message tool-call-card ${isSuccess ? 'success' : 'error'}`;

    const icon = getToolIcon(toolName);
    const displayName = toolNumber ? `#${toolNumber} ${getToolFriendlyName(toolName)}` : getToolFriendlyName(toolName);
    const argsSummary = summarizeArgs(toolName, args);

    let resultStr = '';
    if (result) {
        try {
            resultStr = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
        } catch (e) {
            resultStr = String(result);
        }
    }

    const statusIcon = isSuccess ? '\u{2705}' : '\u{274C}';

    msgEl.innerHTML = `
        <div class="tool-call-header">
            <span class="tool-icon">${icon}</span>
            <span class="tool-name">${escapeHtml(displayName)}</span>
            ${argsSummary ? `<span class="tool-args-summary">${escapeHtml(argsSummary)}</span>` : ''}
            <span class="tool-status ${isSuccess ? 'success' : 'error'}">${statusIcon}</span>
        </div>
        <details class="tool-call-result">
            <summary>View Details</summary>
            ${args && Object.keys(args).length > 0 ? `<div class="tool-detail-section"><strong>Arguments:</strong><pre>${escapeHtml(JSON.stringify(args, null, 2))}</pre></div>` : ''}
            ${resultStr ? `<div class="tool-detail-section"><strong>Result:</strong><pre>${escapeHtml(resultStr)}</pre></div>` : ''}
        </details>
    `;

    // Detect ComfyUI media results and render inline previews
    if (isSuccess && result?.images?.length > 0 &&
        ['comfyui_get_result', 'comfyui_generate_image'].includes(toolName)) {
        const mediaHtml = result.images.map(item => {
            const proxyUrl = `${API}/comfyui/images/${encodeURIComponent(item.filename)}` +
                (item.subfolder ? `?subfolder=${encodeURIComponent(item.subfolder)}` : '');
            const mt = item.media_type || 'image';

            if (mt === 'video') {
                return `<div class="tool-result-media tool-result-video">
                    <video src="${proxyUrl}" controls preload="metadata" loop></video>
                </div>`;
            }
            if (mt === 'audio') {
                return `<div class="tool-result-media tool-result-audio">
                    <audio src="${proxyUrl}" controls preload="metadata"></audio>
                    <span class="audio-filename">${item.filename}</span>
                </div>`;
            }
            return `<div class="tool-result-media tool-result-image" onclick="openImageModal('${proxyUrl}')">
                <img src="${proxyUrl}" loading="lazy" alt="Generated image" />
            </div>`;
        }).join('');
        const genLink = result.generation_id
            ? `<div class="tool-result-gen-link"><a href="#" onclick="switchTab('comfyui'); switchComfyUISubtab('gallery'); return false;">View in Gallery</a></div>`
            : '';
        const detailsEl = msgEl.querySelector('.tool-call-result');
        if (detailsEl) {
            detailsEl.insertAdjacentHTML('beforebegin',
                `<div class="tool-result-images">${mediaHtml}${genLink}</div>`);
        }
    }

    container.appendChild(msgEl);
    _scrollPanelToBottom(panelId);
    log(`Tool: ${displayName}${argsSummary ? ' - ' + argsSummary : ''}`, 'info');
}

// Legacy wrapper
export function addToolCallMessage(toolName, args, result, toolNumber) {
    if (state.activePanelId) addPanelToolCallMessage(state.activePanelId, toolName, args, result, toolNumber);
}

// ==================== Agent Status Message ====================

function _addPanelAgentStatusMessage(panelId, text) {
    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'agent-status-message';
    el.innerHTML = `<span>\u{1F6D1}</span> <span>${escapeHtml(text)}</span>`;
    container.appendChild(el);
    _scrollPanelToBottom(panelId);
}

// ==================== Plan & Thinking Display ====================

function _addPanelPlanMessage(panelId, plan) {
    if (!plan || !plan.steps || plan.steps.length === 0) return;

    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;

    const panel = state.panels[panelId];
    const el = document.createElement('div');
    el.className = 'agent-plan-card';

    const complexityClass = plan.complexity || 'simple';

    let stepsHtml = plan.steps.map((step, i) => {
        const cat = step.tool_category && step.tool_category !== 'none'
            ? `<span class="plan-category-badge">${escapeHtml(step.tool_category)}</span>`
            : '';
        return `<li><span class="plan-step-num">${i + 1}</span> ${escapeHtml(step.step)} ${cat}</li>`;
    }).join('');

    el.innerHTML = `
        <div class="plan-header">
            <span class="plan-icon">\u{1F4CB}</span>
            <span class="plan-title">Plan</span>
            <span class="plan-complexity ${complexityClass}">${complexityClass}</span>
        </div>
        <p class="plan-summary">${escapeHtml(plan.summary || '')}</p>
        <ol class="plan-steps">${stepsHtml}</ol>
    `;

    container.appendChild(el);
    if (panel) panel.agentPlan = plan;
    _scrollPanelToBottom(panelId);
}

function _addPanelThinkingMessage(panelId, content) {
    if (!content) return;

    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;

    const panel = state.panels[panelId];
    const el = document.createElement('div');
    el.className = 'agent-thinking-card';

    el.innerHTML = `
        <details class="thinking-details">
            <summary>
                <span class="thinking-icon">\u{1F4AD}</span>
                <span>Thinking...</span>
            </summary>
            <div class="thinking-content">${escapeHtml(content)}</div>
        </details>
    `;

    container.appendChild(el);
    if (panel) panel.agentThinking = content;
    _scrollPanelToBottom(panelId);
}

// ==================== Sub-Agent Progress Panel ====================

function _getStatusIcon(status) {
    switch (status) {
        case 'running': return '<span class="sub-agent-spinner"></span>';
        case 'completed': return '<span class="sub-agent-check">&#10003;</span>';
        case 'failed': return '<span class="sub-agent-fail">&#10007;</span>';
        case 'timeout': return '<span class="sub-agent-fail">&#9202;</span>';
        case 'aborted': return '<span class="sub-agent-fail">&#9632;</span>';
        default: return '<span>?</span>';
    }
}

function _isTerminalWorkerStatus(status) {
    return status === 'completed' || status === 'failed' || status === 'timeout' || status === 'aborted';
}

async function _appendWorkerToken(workerPanelId, text) {
    const panel = state.panels[workerPanelId];
    if (!panel || !text) return;
    const chat = await getChatModule();

    if (panel._workerLiveMsgIdx === undefined) {
        chat.addPanelMessage(workerPanelId, 'assistant', '');
        panel._workerLiveMsgIdx = panel.messages.length - 1;
    }

    const idx = panel._workerLiveMsgIdx;
    if (idx < 0 || idx >= panel.messages.length) {
        panel._workerLiveMsgIdx = undefined;
        return;
    }
    panel.messages[idx].content = (panel.messages[idx].content || '') + text;

    let cache = _workerTokenCache[workerPanelId];
    if (!cache) {
        cache = { pending: false, idx: idx, textEl: null };
        _workerTokenCache[workerPanelId] = cache;
    }
    cache.idx = idx;
    if (cache.pending) return;
    cache.pending = true;

    requestAnimationFrame(() => {
        cache.pending = false;
        const activePanel = state.panels[workerPanelId];
        if (!activePanel) return;
        const msgIdx = cache.idx;
        if (msgIdx < 0 || msgIdx >= activePanel.messages.length) return;
        const content = activePanel.messages[msgIdx].content || '';
        // Suppress rendering if content looks like a raw tool call JSON —
        // the tool_call event will discard this message and render a tool card instead.
        const trimmed = content.replace(/^[\s`]*(?:json\s*)?/, '');
        if (trimmed.startsWith('{"tool"') || trimmed.startsWith("{'tool'")) {
            // Hide the element but keep tracking — discard will remove it
            if (!cache.textEl) {
                const container = document.getElementById(getPanelDomId(workerPanelId, 'messages'));
                if (!container) return;
                cache.textEl = container.querySelector(`.message[data-index="${msgIdx}"] .message-text`);
            }
            if (cache.textEl) {
                cache.textEl.closest('.message').style.display = 'none';
            }
            return;
        }
        if (!cache.textEl) {
            const container = document.getElementById(getPanelDomId(workerPanelId, 'messages'));
            if (!container) return;
            cache.textEl = container.querySelector(`.message[data-index="${msgIdx}"] .message-text`);
        }
        if (cache.textEl) {
            // Un-hide if it was previously hidden
            cache.textEl.closest('.message').style.display = '';
            cache.textEl.innerHTML = linkifyText(content);
        }
        _scrollPanelToBottom(workerPanelId);
    });
}

function _finalizeWorkerLiveMessage(workerPanelId) {
    const panel = state.panels[workerPanelId];
    if (!panel) return;
    panel._workerLiveMsgIdx = undefined;
    delete _workerTokenCache[workerPanelId];
}

function _discardWorkerLiveMessage(workerPanelId) {
    const panel = state.panels[workerPanelId];
    if (!panel) return;
    const idx = panel._workerLiveMsgIdx;
    if (idx !== undefined && idx >= 0 && idx < panel.messages.length) {
        // Remove the live message from data and DOM
        panel.messages.splice(idx, 1);
        const container = document.getElementById(getPanelDomId(workerPanelId, 'messages'));
        if (container) {
            const msgEl = container.querySelector(`.message[data-index="${idx}"]`);
            if (msgEl) msgEl.remove();
            // Reindex remaining messages
            container.querySelectorAll('.message').forEach((el, i) => el.dataset.index = i);
        }
    }
    panel._workerLiveMsgIdx = undefined;
    delete _workerTokenCache[workerPanelId];
}

// ---------------------------------------------------------------------------
// Tool-level race container (expandable panels inside worker tab)
// ---------------------------------------------------------------------------
const _raceTokenBuffers = {};

function _truncBadge(text, max = 30) {
    if (!text) return 'failed';
    return text.length > max ? text.slice(0, max) + '...' : text;
}

function _renderRaceContainer(workerPanelId, raceId, toolName, numCandidates) {
    const container = document.getElementById(getPanelDomId(workerPanelId, 'messages'));
    if (!container) return;

    const friendlyTool = escapeHtml(toolName.replace(/_/g, ' '));
    const el = document.createElement('div');
    // Start EXPANDED — show all candidates racing
    el.className = 'race-container expanded';
    el.id = `race-${raceId}`;
    el.innerHTML = `
        <div class="race-header" onclick="this.parentElement.classList.toggle('expanded')">
            <span class="race-icon">\u26A1</span>
            <span class="race-title">Racing ${numCandidates} approaches for ${friendlyTool}</span>
            <span class="race-status" id="race-${raceId}-status">Running...</span>
        </div>
        <div class="race-details">
            ${Array.from({length: numCandidates}, (_, i) => {
                const cid = i + 1;
                const label = `Approach ${cid}`;
                return `<div class="race-row" data-candidate="${cid}">
                    <span class="race-row-label">${label}</span>
                    <span class="race-badge pending">\u2022\u2022\u2022</span>
                    <span class="race-row-info" id="race-${raceId}-c${cid}-info"></span>
                </div>`;
            }).join('')}
        </div>`;
    container.appendChild(el);
    _scrollPanelToBottom(workerPanelId);
}

function _appendRaceCandidateToken(raceId, candidateId, text) {
    // Just accumulate — no visible streaming. Keeps UI clean.
    const key = `${raceId}-c${candidateId}`;
    _raceTokenBuffers[key] = (_raceTokenBuffers[key] || '') + text;
}

function _updateRaceCandidateStatus(raceId, candidateId, status) {
    const el = document.querySelector(`#race-${raceId} .race-row[data-candidate="${candidateId}"] .race-badge`);
    if (!el) return;
    el.className = 'race-badge ' + status;
    el.textContent = status === 'inferring' ? '\u2022\u2022\u2022' : status === 'executing' ? '\u2699' : status === 'canceled' ? '\u2014' : '\u2022\u2022\u2022';
}

function _updateRaceCandidateResult(raceId, candidateId, isValid, reason, elapsed) {
    const badge = document.querySelector(`#race-${raceId} .race-row[data-candidate="${candidateId}"] .race-badge`);
    if (badge) {
        badge.className = 'race-badge ' + (isValid ? 'valid' : 'invalid');
        badge.textContent = isValid ? '\u2713' : '\u2717';
    }
    const info = document.getElementById(`race-${raceId}-c${candidateId}-info`);
    if (info) {
        const elapsedStr = elapsed ? ` (${elapsed}s)` : '';
        info.textContent = isValid ? `valid${elapsedStr}` : `${_truncBadge(reason, 40)}${elapsedStr}`;
        if (reason) info.title = reason;
    }
}

function _markRaceWinner(raceId, winnerCandidateId) {
    const row = document.querySelector(`#race-${raceId} .race-row[data-candidate="${winnerCandidateId}"]`);
    if (row) row.classList.add('winner');
}

function _cleanupRaceBuffers(raceId) {
    setTimeout(() => {
        for (const key of Object.keys(_raceTokenBuffers)) {
            if (key.startsWith(raceId)) delete _raceTokenBuffers[key];
        }
    }, 200);
}

function _finalizeRace(raceId, winnerCandidateId, validCount, total) {
    const statusEl = document.getElementById(`race-${raceId}-status`);
    const container = document.getElementById(`race-${raceId}`);
    if (statusEl) {
        statusEl.textContent = `\u2713 Approach ${winnerCandidateId} won`;
        statusEl.className = 'race-status done';
    }
    if (container) {
        container.classList.add('expanded');
    }
    _cleanupRaceBuffers(raceId);
}

function _markRaceFailed(raceId, failureNotes) {
    const statusEl = document.getElementById(`race-${raceId}-status`);
    if (statusEl) {
        statusEl.textContent = 'all failed';
        statusEl.className = 'race-status failed';
    }
    const container = document.getElementById(`race-${raceId}`);
    if (container) container.classList.add('race-all-failed');
    _cleanupRaceBuffers(raceId);
}

async function _applyWorkerEvents(headPanelId, events) {
    const headPanel = state.panels[headPanelId];
    if (!headPanel || !events || events.length === 0) return;
    const workerMap = headPanel.workerPanelMap || {};
    const chat = await getChatModule();
    const tokenBuffers = {};
    // Events are now delivered via SSE — low-latency streaming

    async function flushToken(workerPanelId) {
        const text = tokenBuffers[workerPanelId];
        if (!text) return;
        delete tokenBuffers[workerPanelId];
        await _appendWorkerToken(workerPanelId, text);
    }

    for (const evt of events) {
        // Tool-level race events (emitted from within a single worker's agent loop)
        if (evt.event === 'race_started' && evt.race_id && evt.tool_name) {
            const workerPanelId = workerMap[evt.agent_id];
            if (workerPanelId) {
                // Discard raw JSON tokens — the race container replaces them
                delete tokenBuffers[workerPanelId];
                _discardWorkerLiveMessage(workerPanelId);
                _renderRaceContainer(workerPanelId, evt.race_id, evt.tool_name || 'tool', evt.num_candidates || 3);
            }
            continue;
        }
        if (evt.event === 'race_candidate_status' && evt.race_id) {
            _updateRaceCandidateStatus(evt.race_id, evt.candidate_id, evt.status);
            continue;
        }
        if (evt.event === 'race_candidate_token' && evt.race_id) {
            _appendRaceCandidateToken(evt.race_id, evt.candidate_id, evt.text || '');
            continue;
        }
        if (evt.event === 'race_candidate_evaluated' && evt.race_id) {
            _updateRaceCandidateResult(evt.race_id, evt.candidate_id, evt.is_valid, evt.reason, evt.elapsed);
            continue;
        }
        if (evt.event === 'race_winner_selected' && evt.race_id) {
            _markRaceWinner(evt.race_id, evt.winner, evt.is_original);
            continue;
        }
        if (evt.event === 'race_completed' && evt.race_id && !evt.winner_agent_id) {
            _finalizeRace(evt.race_id, evt.winner, evt.valid_count, evt.total);
            continue;
        }
        if (evt.event === 'race_failed' && evt.race_id) {
            _markRaceFailed(evt.race_id, evt.failure_notes || '');
            continue;
        }

        const workerPanelId = workerMap[evt.agent_id];
        if (!workerPanelId || !state.panels[workerPanelId]) {
            // Skip events for unmapped agents
            continue;
        }

        switch (evt.event) {
            case 'token':
                tokenBuffers[workerPanelId] = (tokenBuffers[workerPanelId] || '') + (evt.text || '');
                break;
            case 'tool_call': {
                // Tool call event → render as card in worker panel
                // Discard pending tokens (raw JSON) — the tool card replaces them
                delete tokenBuffers[workerPanelId];
                _discardWorkerLiveMessage(workerPanelId);
                addPanelToolCallMessage(workerPanelId, evt.tool || 'tool', evt.arguments || {}, evt.result || {}, evt.number);
                break;
            }
            case 'plan':
                if (evt.plan) _addPanelPlanMessage(workerPanelId, evt.plan);
                break;
            case 'thinking':
                if (evt.content) _addPanelThinkingMessage(workerPanelId, evt.content);
                break;
            case 'supervisor_guidance':
                await flushToken(workerPanelId);
                _finalizeWorkerLiveMessage(workerPanelId);
                _addPanelAgentStatusMessage(workerPanelId, 'Supervisor guidance applied');
                break;
            case 'worker_model_switched':
                await flushToken(workerPanelId);
                _finalizeWorkerLiveMessage(workerPanelId);
                _addPanelAgentStatusMessage(workerPanelId, `Model switched to ${evt.to_instance_id || '?'}`);
                break;
            case 'done':
                // agent_loop "done" event — just finalize live message, no completion handling
                // (the actual response comes in the separate "agent_done" event)
                await flushToken(workerPanelId);
                _finalizeWorkerLiveMessage(workerPanelId);
                break;
            case 'agent_done':
                await flushToken(workerPanelId);
                _finalizeWorkerLiveMessage(workerPanelId);
                // agent_done has the actual response — trigger completion
                if (evt.agent_id && headPanel) {
                    await _handleWorkerCompletion(headPanelId, evt.agent_id, workerPanelId, evt);
                }
                break;
            default:
                break;
        }
    }

    for (const panelId of Object.keys(tokenBuffers)) {
        await flushToken(panelId);
    }
}

function _renderWorkerModelSelect(panelId, agentId, selectedInstanceId, loadedModels, status) {
    if (!loadedModels || loadedModels.length === 0) return '';
    const disabled = status !== 'running' ? 'disabled' : '';
    const options = loadedModels.map(m => {
        const id = m.instance_id || '';
        const label = `${m.model || id} [${m.provider || 'unknown'}] gpu=${m.gpu ?? 'n/a'}`;
        const selected = id === selectedInstanceId ? 'selected' : '';
        return `<option value="${escapeHtml(id)}" ${selected}>${escapeHtml(label)}</option>`;
    }).join('');
    return `<select class="worker-model-select" ${disabled}
        onchange="_setWorkerModelForPanel('${panelId}','${agentId}', this.value)">
        ${options}
    </select>`;
}

async function _setWorkerModelForPanel(panelId, agentId, instanceId) {
    const panel = state.panels[panelId];
    if (!panel || !panel.conversationId || !agentId || !instanceId) return;
    try {
        const resp = await fetch(`${API}/tools/agent/workers/${encodeURIComponent(panel.conversationId)}/switch-model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_id: agentId, instance_id: instanceId }),
        });
        if (!resp.ok) {
            const txt = await resp.text().catch(() => '');
            throw new Error(`HTTP ${resp.status}${txt ? `: ${txt.slice(0, 200)}` : ''}`);
        }
        // Status update will arrive via worker SSE stream
    } catch (e) {
        console.error('[AgentMode] Failed switching worker model:', e);
        const chat = await getChatModule();
        chat.showPanelError(panelId, `Failed to switch worker model: ${e.message}`);
    }
}
window._setWorkerModelForPanel = _setWorkerModelForPanel;

// Worker events render directly into head panel — no separate tabs to manage.

async function _syncWorkerTabs(panelId, agents) {
    const headPanel = state.panels[panelId];
    if (!headPanel) return;

    if (!headPanel.workerPanelMap) headPanel.workerPanelMap = {};
    if (!headPanel.workerStatusMap) headPanel.workerStatusMap = {};
    if (!headPanel.workerCompletionNotified) headPanel.workerCompletionNotified = {};

    const panels = await getPanelsModule();
    const chat = await getChatModule();

    for (const a of agents) {
        if (!a || !a.agent_id) continue;

        let workerPanelId = headPanel.workerPanelMap[a.agent_id];
        let workerPanel = workerPanelId ? state.panels[workerPanelId] : null;

        // Create ONE worker tab for this agent. Force-close ALL old worker tabs first.
        if (!workerPanel) {
            // Force-close ALL existing worker tabs (regardless of status)
            for (const [oldAgentId, oldPanelId] of Object.entries({...headPanel.workerPanelMap})) {
                if (state.panels[oldPanelId]) {
                    panels.forceClosePanel(oldPanelId);
                }
                delete headPanel.workerPanelMap[oldAgentId];
                delete headPanel.workerStatusMap[oldAgentId];
                delete headPanel.workerCompletionNotified[oldAgentId];
            }

            // Create minimal worker panel (no welcome page, no input, no controls)
            workerPanel = panels.createWorkerPanel('Worker');
            workerPanelId = workerPanel.panelId;
            headPanel.workerPanelMap[a.agent_id] = workerPanelId;

            workerPanel.parentPanelId = panelId;
            workerPanel.workerAgentId = a.agent_id;
            workerPanel.selectedPersonaId = a.persona_id || 'auto';
            workerPanel.instanceId = a.instance_id || headPanel.instanceId || state.currentModel;
            workerPanel.conversationId = a.conversation_id || null;
            panels.setPanelTabStatus(workerPanelId, 'generating');

            // Render pending plan from head into the worker tab
            if (headPanel._pendingPlan) {
                _addPanelPlanMessage(workerPanelId, headPanel._pendingPlan);
                delete headPanel._pendingPlan;
            }

            // Auto-switch to worker tab so user sees activity
            panels.switchToPanel(workerPanelId);
        } else {
            workerPanel.selectedPersonaId = a.persona_id || workerPanel.selectedPersonaId;
            if (a.instance_id) workerPanel.instanceId = a.instance_id;
            if (a.conversation_id) workerPanel.conversationId = a.conversation_id;
        }

        const prev = headPanel.workerStatusMap[a.agent_id];
        if (prev !== a.status) {
            headPanel.workerStatusMap[a.agent_id] = a.status;

            if (_isTerminalWorkerStatus(a.status) && !headPanel.workerCompletionNotified[a.agent_id]) {
                _handleWorkerCompletion(panelId, a.agent_id, workerPanelId, a);
            }
        }

        if (a.status === 'running') {
            panels.setPanelTabStatus(workerPanelId, 'generating');
        } else if (a.status === 'completed') {
            panels.setPanelTabStatus(workerPanelId, 'done');
        } else if (_isTerminalWorkerStatus(a.status)) {
            panels.setPanelTabStatus(workerPanelId, 'error');
        }
    }
}

/**
 * Centralized worker completion handler. Called from both _syncWorkerTabs
 * (sub_agent_update) and _applyWorkerEvents (agent_done). Uses
 * workerCompletionNotified to prevent double-firing.
 */
async function _handleWorkerCompletion(headPanelId, agentId, workerPanelId, agentData) {
    const headPanel = state.panels[headPanelId];
    if (!headPanel) return;
    if (!headPanel.workerCompletionNotified) headPanel.workerCompletionNotified = {};
    if (headPanel.workerCompletionNotified[agentId]) return; // Already handled
    headPanel.workerCompletionNotified[agentId] = true;

    const panels = await getPanelsModule();
    const chat = await getChatModule();

    const elapsed = agentData.elapsed_seconds !== undefined
        ? `${agentData.elapsed_seconds.toFixed(1)}s`
        : '';
    const statusLabel = agentData.status === 'completed' ? 'Done' : (agentData.status || 'Done');

    // Post full response to head tab (prefer full response over truncated preview)
    const fullResp = agentData.response || agentData.response_full || agentData.response_preview || '';
    if (fullResp) {
        chat.addPanelMessage(headPanelId, 'assistant', fullResp);
    } else {
        chat.addPanelMessage(headPanelId, 'assistant', `${statusLabel} (${elapsed})`);
    }

    // Switch to head tab
    panels.switchToPanel(headPanelId);

    // Close worker tab after a brief delay so user can see transition
    panels.setPanelTabStatus(workerPanelId, 'done');

    // Only stop SSE when ALL workers are done (not just this one)
    const hasActiveWorkers = Object.entries(headPanel.workerPanelMap || {}).some(([aid, _wpid]) => {
        if (aid === agentId) return false; // skip the one we just completed
        const st = headPanel.workerStatusMap?.[aid];
        return st && !_isTerminalWorkerStatus(st);
    });
    if (!hasActiveWorkers) {
        _stopWorkerSSE(headPanelId);
    }

    const wpId = workerPanelId;
    const agId = agentId;
    setTimeout(() => {
        if (state.panels[wpId]) {
            panels.forceClosePanel(wpId);
        }
        if (headPanel.workerPanelMap) {
            delete headPanel.workerPanelMap[agId];
        }
    }, 2000);
}

async function _updatePanelSubAgentPanel(panelId, agents, loadedModels = null) {
    if (!agents || agents.length === 0) return;
    // Sync worker tabs (creates worker tab, handles completion/switch-back).
    // Head tab stays clean — no Workers panel rendered here.
    await _syncWorkerTabs(panelId, agents);
}

function _removePanelSubAgentPanel(panelId) {
    const panel = document.getElementById(`panel-${panelId}-sub-agent-panel`);
    if (panel) {
        setTimeout(() => {
            panel.classList.add('fade-out');
            setTimeout(() => panel.remove(), 500);
        }, 2000);
    }
}

// ==================== Ask User Card ====================

function _showPanelAskUserCard(panelId, question, options, toolNumber) {
    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;

    const card = document.createElement('div');
    card.className = 'ask-user-card';
    card.dataset.panelId = panelId;
    card.dataset.askActive = 'true';
    card.dataset.abortId = state.panels[panelId]?.agentAbortId || '';

    let optionsHtml = '';
    if (options && options.length > 0) {
        const btns = options.map((opt, i) =>
            `<button class="ask-user-option" data-option-index="${i}">${escapeHtml(opt)}</button>`
        ).join('');
        optionsHtml = `<div class="ask-user-options">${btns}</div>`;
    }

    card.innerHTML = `
        <div class="ask-user-header">
            <span class="ask-user-icon">&#10067;</span>
            <span class="ask-user-label">#${toolNumber || '?'} Agent needs your input</span>
        </div>
        <div class="ask-user-question">${escapeHtml(question)}</div>
        ${optionsHtml}
        <div class="ask-user-input-row">
            <input type="text" class="ask-user-input" placeholder="Type your response...">
            <button class="ask-user-submit">Send</button>
        </div>
    `;

    // Wire up option buttons
    if (options && options.length > 0) {
        card.querySelector('.ask-user-options').addEventListener('click', (e) => {
            const btn = e.target.closest('.ask-user-option');
            if (btn) {
                const idx = parseInt(btn.dataset.optionIndex);
                _respondToAgentPanel(panelId, card, options[idx]);
            }
        });
    }

    // Wire up text input
    const input = card.querySelector('.ask-user-input');
    const submitBtn = card.querySelector('.ask-user-submit');
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') _respondToAgentPanel(panelId, card, input.value);
    });
    submitBtn.addEventListener('click', () => {
        _respondToAgentPanel(panelId, card, input.value);
    });

    container.appendChild(card);
    _scrollPanelToBottom(panelId);
    if (input) input.focus();
}

async function _respondToAgentPanel(panelId, card, response) {
    if (!response || !response.trim()) return;
    const panel = state.panels[panelId];
    if (!panel) return;

    // Show what user answered
    card.innerHTML = `
        <div class="ask-user-header answered">
            <span class="ask-user-icon">&#10003;</span>
            <span class="ask-user-label">Answered</span>
        </div>
        <div class="ask-user-response">${escapeHtml(response)}</div>
    `;
    card.classList.add('answered');
    delete card.dataset.askActive;

    // Use the session-specific abort_id captured when the ask card was shown.
    const abortId = card.dataset.abortId || panel.agentAbortId;
    if (!abortId) {
        console.warn('[AgentMode] Missing abort_id for ask_user response');
        return;
    }

    // Send response to backend
    try {
        const resp = await fetch(`${API}/tools/agent/respond`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                abort_id: abortId,
                response: response.trim(),
            }),
        });
        if (!resp.ok) {
            const txt = await resp.text().catch(() => '');
            throw new Error(`HTTP ${resp.status}${txt ? `: ${txt.slice(0, 200)}` : ''}`);
        }
    } catch (e) {
        console.error('[AgentMode] Failed to send response:', e);
        const err = document.createElement('div');
        err.className = 'error-indicator';
        err.textContent = `Failed to deliver response: ${e.message}`;
        card.appendChild(err);
    }
}

// Global handler for backward compat — finds active ask-user card
window._respondToAgent = async function(response) {
    if (!response || !response.trim()) return;

    // Find the active ask-user card across all panels
    const card = document.querySelector('.ask-user-card[data-ask-active="true"]');
    if (!card) return;

    const panelId = card.dataset.panelId || state.activePanelId;
    await _respondToAgentPanel(panelId, card, response);
};
