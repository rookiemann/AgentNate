/**
 * Conversation save/load/rename/delete for AgentNate UI.
 * Panel-aware: save/load operate on the active panel.
 */

import { state, API, getActivePanel, getPanelDomId } from './state.js';
import { log, escapeHtml, formatDate } from './utils.js';

// Lazy import to avoid circular deps
let _chatModule = null;
async function getChatModule() {
    if (!_chatModule) _chatModule = await import('./chat.js');
    return _chatModule;
}

export async function openConversationHistory() {
    const modal = document.getElementById('conversation-modal');
    modal.classList.remove('hidden');
    await loadConversations();
}

export function closeConversationHistory() {
    document.getElementById('conversation-modal').classList.add('hidden');
}

export async function loadConversations() {
    const list = document.getElementById('conversation-list');
    list.innerHTML = '<div class="loading">Loading conversations...</div>';

    try {
        const resp = await fetch(`${API}/tools/conversations`);
        const data = await resp.json();

        state.savedConversations = data.conversations || [];

        if (state.savedConversations.length === 0) {
            list.innerHTML = '<div class="empty-state">No saved conversations yet</div>';
            return;
        }

        list.innerHTML = state.savedConversations.map(conv => `
            <div class="conversation-item" onclick="loadConversation('${conv.id}')">
                <div class="conversation-info">
                    <div class="conversation-name">${escapeHtml(conv.name || 'Untitled')}</div>
                    <div class="conversation-meta">
                        <span class="conv-type-badge ${conv.conv_type || 'agent'}">${(conv.conv_type || 'agent') === 'agent' ? 'Agent' : 'Chat'}</span>
                        <span>${conv.message_count} messages</span>
                        <span>${formatDate(conv.updated_at)}</span>
                        ${conv.persona_id && conv.persona_id !== 'none' ? `<span class="persona-badge">${conv.persona_id}</span>` : ''}
                    </div>
                </div>
                <div class="conversation-actions">
                    <button class="btn-icon" onclick="event.stopPropagation(); renameConversation('${conv.id}')" title="Rename">&#9998;</button>
                    <button class="btn-icon" onclick="event.stopPropagation(); deleteConversation('${conv.id}')" title="Delete">&#128465;</button>
                </div>
            </div>
        `).join('');

    } catch (e) {
        console.error('Failed to load conversations:', e);
        list.innerHTML = '<div class="empty-state">Failed to load conversations</div>';
    }
}

/**
 * Load a conversation into the active panel.
 */
export async function loadConversation(convId) {
    try {
        const resp = await fetch(`${API}/tools/conversations/${convId}`);
        const data = await resp.json();

        if (data.conversation) {
            const panelId = state.activePanelId;
            const panel = state.panels[panelId];
            if (!panel) return;

            // Clear panel state
            panel.messages = [];
            panel.conversationId = convId;

            const conv = data.conversation;

            // Clear DOM
            const container = document.getElementById(getPanelDomId(panelId, 'messages'));
            if (container) container.innerHTML = '';

            // Set agent mode based on conversation type
            const isAgent = (conv.conv_type || 'agent') === 'agent';
            panel.agentMode = isAgent;

            // Sync legacy state
            if (panelId === state.activePanelId) {
                state.messages = panel.messages;
                state.conversationId = convId;
                state.agentMode = isAgent;
            }

            const chat = await getChatModule();
            conv.messages.forEach(m => {
                chat.addPanelMessage(panelId, m.role, m.content);
            });

            closeConversationHistory();
            log(`Loaded ${isAgent ? 'agent' : 'chat'} conversation: ${conv.name || convId}`, 'info');
        }

    } catch (e) {
        console.error('Failed to load conversation:', e);
        const chat = await getChatModule();
        chat.showPanelError(state.activePanelId, 'Failed to load conversation');
    }
}

export async function renameConversation(convId) {
    const newName = prompt('Enter new name:');
    if (!newName) return;

    try {
        await fetch(`${API}/tools/conversations/${convId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName })
        });
        loadConversations();
    } catch (e) {
        console.error('Failed to rename:', e);
    }
}

export async function deleteConversation(convId) {
    if (!confirm('Delete this conversation?')) return;

    try {
        await fetch(`${API}/tools/conversations/${convId}`, {
            method: 'DELETE'
        });

        // Clear from any panel that had this conversation loaded
        for (const panel of Object.values(state.panels)) {
            if (panel.conversationId === convId) {
                panel.conversationId = null;
            }
        }
        if (state.conversationId === convId) {
            state.conversationId = null;
        }

        loadConversations();
    } catch (e) {
        console.error('Failed to delete:', e);
    }
}

export async function clearUntitledConversations() {
    const untitledCount = (state.savedConversations || []).filter(c => (c.name || 'Untitled') === 'Untitled').length;
    if (untitledCount === 0) {
        log('No untitled conversations to delete', 'info');
        return;
    }

    if (!confirm(`Delete ${untitledCount} untitled conversations?`)) return;

    try {
        const resp = await fetch(`${API}/tools/conversations/batch-delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filter: 'untitled' })
        });
        const data = await resp.json();
        log(`Deleted ${data.deleted} untitled conversations`, 'success');
        loadConversations();
    } catch (e) {
        console.error('Failed to clear untitled:', e);
        log('Failed to clear untitled conversations', 'error');
    }
}

/**
 * Open save dialog for a specific panel (called from panel header save button).
 */
export function savePanelConversation(panelId) {
    const panel = state.panels[panelId];
    if (!panel || panel.messages.length === 0) {
        log('No messages to save', 'warning');
        return;
    }

    // Store which panel we're saving
    state._savingPanelId = panelId;

    document.getElementById('save-conversation-modal').classList.remove('hidden');

    const firstUserMsg = panel.messages.find(m => m.role === 'user');
    const suggestedName = firstUserMsg
        ? firstUserMsg.content.substring(0, 50) + (firstUserMsg.content.length > 50 ? '...' : '')
        : 'Conversation ' + new Date().toLocaleString();

    document.getElementById('save-conversation-name').value = suggestedName;
}

/**
 * Legacy wrapper — saves active panel.
 */
export function saveCurrentConversation() {
    if (!state.activePanelId) return;
    savePanelConversation(state.activePanelId);
}

export function closeSaveConversation() {
    document.getElementById('save-conversation-modal').classList.add('hidden');
    delete state._savingPanelId;
}

export async function confirmSaveConversation() {
    const name = document.getElementById('save-conversation-name').value.trim() || 'Untitled';

    // Determine which panel we're saving
    const panelId = state._savingPanelId || state.activePanelId;
    const panel = state.panels[panelId];
    if (!panel) return;

    try {
        if (panel.conversationId) {
            // Agent chat — mark existing conversation as saved
            await fetch(`${API}/tools/conversations/${panel.conversationId}/mark-saved`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
        } else {
            // Regular chat — send all messages to create a saved conversation
            const messages = panel.messages.map(m => ({
                role: m.role,
                content: m.content
            }));
            const modelId = panel.instanceId || state.currentModel;
            await fetch(`${API}/tools/conversations/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages,
                    name,
                    persona_id: panel.agentMode ? panel.selectedPersonaId : 'none',
                    model_id: modelId,
                    conv_type: panel.agentMode ? 'agent' : 'chat'
                })
            });
        }

        log(`Conversation saved: ${name}`, 'success');
        closeSaveConversation();

    } catch (e) {
        console.error('Failed to save conversation:', e);
        const chat = await getChatModule();
        chat.showPanelError(panelId, 'Failed to save conversation');
    }
}
