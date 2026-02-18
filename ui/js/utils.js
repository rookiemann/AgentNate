/**
 * Utility functions for AgentNate UI
 */

import { state, API, logs, MAX_LOGS } from './state.js';

// ==================== Logging System ====================

export function log(message, level = 'info') {
    const time = new Date().toLocaleTimeString();
    const entry = { time, message, level };
    logs.push(entry);
    if (logs.length > MAX_LOGS) logs.shift();

    appendLogToUI(entry);

    const consoleFn = level === 'error' ? console.error : level === 'warning' ? console.warn : console.log;
    consoleFn(`[${time}] ${message}`);
}

/**
 * Send a debug event to the backend debug log file.
 * Fire-and-forget — never blocks UI.
 */
export function debugLog(action, detail = '') {
    const msg = `${action} ${detail}`;
    console.debug(`[DEBUG] ${msg}`);
    try {
        fetch(`${API}/debug/log`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, detail }),
        }).catch(() => {}); // silent fail
    } catch (_) {}
}

export function appendLogToUI(entry) {
    const output = document.getElementById('log-output');
    if (!output) return;

    const div = document.createElement('div');
    div.className = `log-entry ${entry.level}`;
    div.innerHTML = `<span class="log-time">${entry.time}</span>${escapeHtml(entry.message)}`;
    output.appendChild(div);
    output.scrollTop = output.scrollHeight;
}

export function clearLogs() {
    logs.length = 0;
    const output = document.getElementById('log-output');
    if (output) output.innerHTML = '';
    log('Logs cleared', 'info');
}

export function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

export function linkifyText(text) {
    const escaped = escapeHtml(text);
    const urlPattern = /(https?:\/\/[^\s<>"{}|\\^`\[\]]+)/g;
    return escaped.replace(urlPattern, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
}

/**
 * Render simple markdown to HTML. Handles headings, bold, italic, links,
 * unordered/ordered lists, inline code, and line breaks. Safe: escapes HTML first.
 */
export function renderSimpleMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Headings: ### heading → <h4>, ## → <h3>, # → <h2>
    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');

    // Bold: **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Italic: *text* or _text_ (but not inside words like file_name)
    html = html.replace(/(?<!\w)\*([^*]+?)\*(?!\w)/g, '<em>$1</em>');

    // Inline code: `code`
    html = html.replace(/`([^`]+?)`/g, '<code>$1</code>');

    // Links: [text](url)
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // Bare URLs
    html = html.replace(/(^|[^"=])(https?:\/\/[^\s<>"{}|\\^`\[\]]+)/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');

    // Unordered list items: * item or - item (at start of line)
    html = html.replace(/^[*\-] (.+)$/gm, '<li>$1</li>');

    // Ordered list items: 1. item
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> in <ul>
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Paragraphs: double newline → paragraph break
    html = html.replace(/\n{2,}/g, '</p><p>');

    // Single newlines → <br> (but not inside tags)
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph
    html = '<p>' + html + '</p>';

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');

    return html;
}

// ==================== Toast Notifications ====================

export function showToast(message, type = 'info', duration = 4000) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        info: '&#8505;',
        success: '&#10003;',
        warning: '&#9888;',
        error: '&#10007;',
        loading: '<span class="toast-spinner"></span>'
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));

    if (type !== 'loading' && duration > 0) {
        setTimeout(() => removeToast(toast), duration);
    }

    return toast;
}

export function removeToast(toast) {
    if (!toast) return;
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
}

export function updateToast(toast, message, type = 'success') {
    if (!toast) return;
    const icons = {
        info: '&#8505;',
        success: '&#10003;',
        warning: '&#9888;',
        error: '&#10007;',
    };
    toast.className = `toast toast-${type} show`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;
    setTimeout(() => removeToast(toast), 3000);
}

// ==================== Connection Status ====================

export function updateConnectionStatus(connected, reason = null) {
    const wasConnected = state.serverConnected;
    state.serverConnected = connected;

    const banner = document.getElementById('connection-banner');
    if (!banner) return;

    if (connected) {
        state.connectionFailures = 0;
        state.lastError = null;
        banner.classList.add('hidden');

        if (!wasConnected) {
            log('Server connection restored', 'success');
        }
    } else {
        banner.classList.remove('hidden');

        const messageEl = banner.querySelector('.connection-message');
        const reasonEl = banner.querySelector('.connection-reason');

        if (state.shutdownReason) {
            messageEl.textContent = 'Server Shut Down';
            reasonEl.textContent = state.shutdownReason;
        } else if (reason) {
            messageEl.textContent = 'Server Disconnected';
            reasonEl.textContent = reason;
        } else {
            messageEl.textContent = 'Server Disconnected';
            reasonEl.textContent = 'Connection lost. The server may have crashed or been stopped.';
        }

        if (wasConnected) {
            log('Server connection lost: ' + (reason || 'Unknown'), 'error');
        }
    }
}

export async function apiFetch(url, options = {}) {
    try {
        const resp = await fetch(url, options);

        if (!state.serverConnected && !state.shutdownReason) {
            updateConnectionStatus(true);
        }
        state.connectionFailures = 0;

        return resp;
    } catch (e) {
        state.connectionFailures++;
        state.lastError = e.message;

        if (state.connectionFailures >= 2 && !state.shutdownReason) {
            updateConnectionStatus(false, e.message);
        }

        throw e;
    }
}

// ==================== Token Estimation ====================

export function estimateTokens(text) {
    if (!text) return 0;
    const words = text.split(/\s+/).filter(w => w.length > 0).length;
    return Math.ceil(words * 1.3);
}

export function getConversationTokens() {
    let total = 0;

    if (state.activePromptContent) {
        total += estimateTokens(state.activePromptContent);
    }

    for (const msg of state.messages) {
        total += estimateTokens(msg.content);
        total += 4;
    }

    return total;
}

export function getCurrentContextLength() {
    if (!state.currentModel) return 0;
    const modelInfo = state.models[state.currentModel];
    return modelInfo?.context_length || 4096;
}

export function updateContextDisplay() {
    const contextEl = document.getElementById('debug-context');
    if (!contextEl) return;

    const used = getConversationTokens();
    const max = getCurrentContextLength();

    const formatTokens = (n) => n >= 1000 ? `${(n/1000).toFixed(1)}K` : n;

    contextEl.textContent = `${formatTokens(used)}/${formatTokens(max)}`;

    const usage = max > 0 ? used / max : 0;
    if (usage > 0.9) {
        contextEl.style.color = 'var(--error, #ff4444)';
    } else if (usage > 0.75) {
        contextEl.style.color = 'var(--warning, #ffaa00)';
    } else {
        contextEl.style.color = '';
    }
}

// ==================== Debug Status ====================

export function updateDebugStatus() {
    const debugBar = document.getElementById('debug-status');
    if (!debugBar || debugBar.classList.contains('hidden')) return;

    document.getElementById('debug-model').textContent = state.currentModel ? state.currentModel.substring(0, 12) + '...' : 'none';
    document.getElementById('debug-vision').textContent = state.currentModelHasVision ? 'YES' : 'no';
    document.getElementById('debug-prompt').textContent = state.activePromptId || 'none';
    document.getElementById('debug-generating').textContent = state.isGenerating ? 'YES' : 'no';
    document.getElementById('debug-ws').textContent = state.ws ? ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][state.ws.readyState] : 'null';
    document.getElementById('debug-images').textContent = state.pendingImages.length;

    updateContextDisplay();
}

// ==================== Date Formatting ====================

export function formatDate(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now - date;

    if (diff < 86400000) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diff < 604800000) {
        return date.toLocaleDateString([], { weekday: 'short' });
    } else {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
}
