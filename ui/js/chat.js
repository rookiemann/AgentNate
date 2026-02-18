/**
 * Chat core for AgentNate UI - WebSocket, messages, input handling.
 * All functions are panel-aware: they take a panelId parameter.
 * Legacy global wrappers (sendMessage, clearChat, etc.) dispatch to the active panel.
 */

import { state, API, getActivePanel, getPanelDomId } from './state.js';
import { log, linkifyText, escapeHtml, updateDebugStatus, estimateTokens, getConversationTokens, getCurrentContextLength, updateConnectionStatus } from './utils.js';
import { getEffectiveParams } from './model-settings.js';
import { renderImagePreviews } from './images.js';
import { retrievePdfContext, clearPdfSession } from './pdf.js';
import { getActiveSystemPrompt } from './prompts.js';
import { updateWelcomeMessage } from './onboarding.js';

// ====================== WebSocket =======================

export function connectWebSocket() {
    const apiUrl = new URL(API);
    const wsProto = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${apiUrl.host}/api/chat/stream`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        log('WebSocket connected', 'success');
        state.connectionFailures = 0;
        if (!state.shutdownReason) {
            updateConnectionStatus(true);
        }
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWSMessage(data);
        } catch (e) {
            console.error('WebSocket JSON parse error:', e, event.data);
        }
    };

    state.ws.onclose = () => {
        if (state.shutdownReason) {
            console.log('WebSocket closed (shutdown)');
            return;
        }

        state.connectionFailures++;
        console.log('WebSocket disconnected, attempt', state.connectionFailures);

        if (state.connectionFailures >= 3) {
            updateConnectionStatus(false, 'Lost connection to server');
        }

        const delay = Math.min(2000 * Math.pow(1.5, state.connectionFailures - 1), 30000);
        setTimeout(connectWebSocket, delay);
    };

    state.ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        state.connectionFailures++;
    };
}

export function handleWSMessage(data) {
    // Route message to the correct panel via panel_id field
    const panelId = data.panel_id || state.activePanelId;
    const panel = state.panels[panelId];
    if (!panel) return;
    const requestId = data.request_id;

    // Ignore events for requests explicitly cancelled by user.
    if (requestId && _isIgnoredPanelRequest(panelId, requestId)) {
        if (data.type === 'done' || data.type === 'cancelled' || data.type === 'error') {
            _clearIgnoredPanelRequest(panelId, requestId);
        }
        return;
    }

    // Ignore stale events from older request IDs on this panel.
    if (requestId && panel.currentRequestId && requestId !== panel.currentRequestId) {
        return;
    }

    switch (data.type) {
        case 'token':
            appendPanelToken(panelId, data.content);
            break;
        case 'done':
            finishPanelGeneration(panelId);
            break;
        case 'error':
            if (data.error && data.error.includes('cancelled')) {
                finishPanelGeneration(panelId);
            } else {
                showPanelError(panelId, data.error);
                finishPanelGeneration(panelId);
            }
            break;
        case 'cancelled':
            finishPanelGeneration(panelId);
            break;
        case 'pong':
            break;
    }
}

// ====================== Panel-Scoped Chat ======================

/**
 * Send a message from a specific panel.
 */
export async function panelSendMessage(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    const input = document.getElementById(getPanelDomId(panelId, 'input'));
    const content = input ? input.value.trim() : '';

    const modelId = panel.instanceId || state.currentModel;

    if ((!content && panel.pendingImages.length === 0) || !modelId || panel.isGenerating) {
        return;
    }

    const images = panel.pendingImages.map(img => img.dataUri);

    addPanelMessage(panelId, 'user', content || '(image)', images.length > 0 ? images : null);
    if (input) {
        input.value = '';
        autoResizeInput(input);
    }

    panel.pendingImages = [];
    // Render image previews for this panel
    const previewArea = document.getElementById(getPanelDomId(panelId, 'image-preview'));
    if (previewArea) previewArea.innerHTML = '';

    panel.isGenerating = true;
    _syncLegacyGenerating(panelId);
    updatePanelUI(panelId);

    // Agent mode: delegate to agent.js
    if (panel.agentMode) {
        const { sendPanelAgentMessage } = await import('./agent.js');
        await sendPanelAgentMessage(panelId, content);
        return;
    }

    // Regular chat via WebSocket
    addPanelMessage(panelId, 'assistant', '');

    const conversationMessages = panel.messages.map(m => {
        const msg = { role: m.role, content: m.content };
        if (m.images && m.images.length > 0) {
            msg.images = m.images;
        }
        return msg;
    });

    let pdfContext = null;
    if (panel.loadedPdfFiles.length > 0) {
        pdfContext = await retrievePdfContext(content);
    }

    let systemPrompt = getActiveSystemPrompt() || '';
    if (pdfContext) {
        const pdfInstructions = `The following document excerpts are relevant to the user's question. Use them to provide accurate answers, citing page numbers when available:\n\n${pdfContext}\n\n---\n\n`;
        systemPrompt = pdfInstructions + systemPrompt;
    }

    const messages = systemPrompt
        ? [{ role: 'system', content: systemPrompt }, ...conversationMessages]
        : conversationMessages;

    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        const requestId = `chat-${panelId}-${Date.now()}`;
        panel.currentRequestId = requestId;

        const payload = {
            action: 'chat',
            panel_id: panelId,
            instance_id: modelId,
            messages: messages,
            request_id: requestId,
            params: getEffectiveParams(modelId)
        };
        state.ws.send(JSON.stringify(payload));
    } else {
        showPanelError(panelId, 'WebSocket not connected');
        finishPanelGeneration(panelId);
    }
}

/**
 * Add a message to a specific panel's message list and DOM.
 */
export function addPanelMessage(panelId, role, content, images = null) {
    const panel = state.panels[panelId];
    if (!panel) return;

    panel.messages.push({ role, content, images });

    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;

    const welcome = container.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;
    msgEl.dataset.index = panel.messages.length - 1;
    msgEl.dataset.panelId = panelId;

    let html = '';
    if (images && images.length > 0) {
        html += '<div class="message-images">';
        images.forEach(img => {
            html += `<img src="${img}" class="message-image" onclick="openImageModal(this.src)" alt="Attached image">`;
        });
        html += '</div>';
    }
    html += `<div class="message-text">${linkifyText(content)}</div>`;
    msgEl.innerHTML = html;

    container.appendChild(msgEl);
    container.scrollTop = container.scrollHeight;
}

// Per-panel token cache for streaming performance
const _panelTokenCache = {};
// Per-panel request IDs to ignore after user pressed Stop.
const _ignoredPanelRequestIds = {};

function _getIgnoredRequestSet(panelId) {
    if (!_ignoredPanelRequestIds[panelId]) {
        _ignoredPanelRequestIds[panelId] = new Set();
    }
    return _ignoredPanelRequestIds[panelId];
}

function _ignorePanelRequest(panelId, requestId) {
    if (!panelId || !requestId) return;
    _getIgnoredRequestSet(panelId).add(requestId);
}

function _isIgnoredPanelRequest(panelId, requestId) {
    if (!panelId || !requestId) return false;
    const set = _ignoredPanelRequestIds[panelId];
    return !!(set && set.has(requestId));
}

function _clearIgnoredPanelRequest(panelId, requestId) {
    if (!panelId || !requestId) return;
    const set = _ignoredPanelRequestIds[panelId];
    if (!set) return;
    set.delete(requestId);
    if (set.size === 0) delete _ignoredPanelRequestIds[panelId];
}

/**
 * Append a streaming token to the last message in a panel.
 */
export function appendPanelToken(panelId, token) {
    const panel = state.panels[panelId];
    if (!panel) return;

    const lastIndex = panel.messages.length - 1;
    if (lastIndex < 0) return;

    panel.messages[lastIndex].content += token;

    // Use per-panel cache for DOM element lookup
    let cache = _panelTokenCache[panelId];
    if (!cache || cache.idx !== lastIndex) {
        const container = document.getElementById(getPanelDomId(panelId, 'messages'));
        if (!container) return;
        const msgEl = container.querySelector(`.message[data-index="${lastIndex}"]`);
        cache = {
            el: msgEl ? (msgEl.querySelector('.message-text') || msgEl) : null,
            idx: lastIndex,
        };
        _panelTokenCache[panelId] = cache;
    }

    if (cache.el) {
        cache.el.innerHTML = linkifyText(panel.messages[lastIndex].content);
        const container = document.getElementById(getPanelDomId(panelId, 'messages'));
        if (container) container.scrollTop = container.scrollHeight;
    }
}

export function resetPanelTokenCache(panelId) {
    delete _panelTokenCache[panelId];
}

export function scrollPanelToBottom(panelId) {
    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (container) container.scrollTop = container.scrollHeight;
}

export function finishPanelGeneration(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    panel.isGenerating = false;
    panel.currentRequestId = null;
    resetPanelTokenCache(panelId);
    _syncLegacyGenerating(panelId);
    updatePanelUI(panelId);

    // Update tab status
    import('./panels.js').then(m => m.setPanelTabStatus(panelId, 'done'));
}

export function cancelPanelInference(panelId) {
    const panel = state.panels[panelId];
    if (!panel || !panel.isGenerating || !panel.currentRequestId) return;
    const requestId = panel.currentRequestId;
    _ignorePanelRequest(panelId, requestId);

    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            action: 'cancel',
            panel_id: panelId,
            request_id: requestId
        }));
    }

    finishPanelGeneration(panelId);
}

export function showPanelError(panelId, error) {
    const panel = state.panels[panelId];
    if (!panel) return;

    const lastIndex = panel.messages.length - 1;
    if (lastIndex < 0) return;

    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (!container) return;

    const msgEl = container.querySelector(`.message[data-index="${lastIndex}"]`);
    if (msgEl) {
        msgEl.classList.add('error');
        const errDiv = document.createElement('div');
        errDiv.className = 'error-indicator';
        errDiv.textContent = `Error: ${error}`;
        msgEl.appendChild(errDiv);
    }

    // Update tab status
    import('./panels.js').then(m => m.setPanelTabStatus(panelId, 'error'));
}

/**
 * Update a specific panel's input/button state.
 */
export function updatePanelUI(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    const input = document.getElementById(getPanelDomId(panelId, 'input'));
    const sendBtn = document.getElementById(getPanelDomId(panelId, 'send-btn'));
    const agentToggle = document.getElementById(getPanelDomId(panelId, 'agent-toggle'));

    const modelId = panel.instanceId || state.currentModel;
    const canSend = modelId && !panel.isGenerating;

    if (input) {
        input.disabled = !canSend;
        if (!modelId) {
            input.placeholder = panel.agentMode
                ? 'Load a model to use agent features...'
                : 'Load a model to start chatting...';
        } else if (panel.isGenerating) {
            input.placeholder = 'Waiting for response...';
        } else {
            input.placeholder = panel.agentMode
                ? 'Ask the agent to do something...'
                : 'Type a message...';
        }
    }

    if (sendBtn) {
        sendBtn.disabled = !modelId;
        if (panel.isGenerating) {
            if (panel.agentMode) {
                sendBtn.disabled = true;
                sendBtn.innerHTML = '<span>Send</span>';
                sendBtn.classList.remove('btn-stop');
            } else {
                sendBtn.innerHTML = '<span>Stop</span>';
                sendBtn.classList.add('btn-stop');
                sendBtn.onclick = () => cancelPanelInference(panelId);
            }
        } else {
            sendBtn.innerHTML = '<span>Send</span>';
            sendBtn.classList.remove('btn-stop');
            sendBtn.onclick = () => panelSendMessage(panelId);
        }
    }

    if (agentToggle) {
        agentToggle.disabled = panel.isGenerating;
    }

    updateDebugStatus();
}

export async function clearPanelChat(panelId) {
    const panel = state.panels[panelId];
    if (!panel) return;

    panel.messages = [];
    panel.conversationId = null;
    resetPanelTokenCache(panelId);

    if (panel.loadedPdfFiles.length > 0) {
        await clearPdfSession();
    }

    const container = document.getElementById(getPanelDomId(panelId, 'messages'));
    if (container) {
        container.innerHTML = '';
        const welcome = document.createElement('div');
        welcome.className = 'welcome-message';
        welcome.innerHTML = '<h2>AgentNate</h2><p class="welcome-subtitle">Your local AI platform</p>';
        container.appendChild(welcome);
    }

    // Sync legacy state
    if (panelId === state.activePanelId) {
        state.messages = panel.messages;
        state.conversationId = null;
    }

    log('Chat cleared', 'info');
}

// ====================== Legacy Wrappers ======================
// These dispatch to the active panel for backward compat with
// modules that still call sendMessage(), clearChat(), etc.

export async function sendMessage() {
    if (state.activePanelId) {
        return panelSendMessage(state.activePanelId);
    }
}

export function addMessage(role, content, images = null) {
    if (state.activePanelId) {
        addPanelMessage(state.activePanelId, role, content, images);
    }
}

export function appendToken(token) {
    if (state.activePanelId) {
        appendPanelToken(state.activePanelId, token);
    }
}

export function resetTokenCache() {
    if (state.activePanelId) {
        resetPanelTokenCache(state.activePanelId);
    }
}

export function scrollChatToBottom() {
    if (state.activePanelId) {
        scrollPanelToBottom(state.activePanelId);
    }
}

export function finishGeneration() {
    if (state.activePanelId) {
        finishPanelGeneration(state.activePanelId);
    }
}

export function cancelInference() {
    if (state.activePanelId) {
        cancelPanelInference(state.activePanelId);
    }
}

export function showError(error) {
    if (state.activePanelId) {
        showPanelError(state.activePanelId, error);
    }
}

export function updateChatUI() {
    if (state.activePanelId) {
        updatePanelUI(state.activePanelId);
    }
}

export async function clearChat() {
    if (state.activePanelId) {
        return clearPanelChat(state.activePanelId);
    }
}

export function handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

export function autoResizeInput(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

// ====================== Internal ======================

/**
 * Sync the legacy state.isGenerating flag when the active panel changes generation state.
 */
function _syncLegacyGenerating(panelId) {
    if (panelId === state.activePanelId) {
        const panel = state.panels[panelId];
        if (panel) {
            state.isGenerating = panel.isGenerating;
            state.currentRequestId = panel.currentRequestId;
        }
    }
}
