/**
 * Executions Tab - Central dashboard for all workflow execution data.
 *
 * Data sources:
 * 1. History DB (completed executions from stopped workers)
 * 2. Live worker n8n APIs (running/recent executions)
 * 3. Worker queue state (pending queued runs)
 */

import { API, executionsState } from './state.js';

const FAST_INTERVAL = 5_000;   // 5s when workers are active
const IDLE_INTERVAL = 30_000;  // 30s when nothing running

// ==================== Initialization ====================

export function initExecutions() {
    if (!executionsState.initialized) {
        executionsState.initialized = true;
        populateWorkflowFilter();
    }
    refreshExecutions();
    startAutoRefresh();
}

// ==================== Auto-Refresh ====================

export function startAutoRefresh() {
    stopAutoRefresh();
    // Start at fast interval; refreshExecutions will adjust
    executionsState.autoRefreshInterval = setInterval(refreshExecutions, FAST_INTERVAL);
    executionsState._currentInterval = FAST_INTERVAL;
}

export function stopAutoRefresh() {
    if (executionsState.autoRefreshInterval) {
        clearInterval(executionsState.autoRefreshInterval);
        executionsState.autoRefreshInterval = null;
    }
}

function adjustRefreshRate() {
    const s = executionsState.stats;
    const hasActivity = (s.running > 0) || (s.queued !== null && s.queued > 0) || (s.queued === null);
    const desiredInterval = hasActivity ? FAST_INTERVAL : IDLE_INTERVAL;

    if (executionsState._currentInterval !== desiredInterval) {
        executionsState._currentInterval = desiredInterval;
        stopAutoRefresh();
        executionsState.autoRefreshInterval = setInterval(refreshExecutions, desiredInterval);
    }
}

// ==================== Data Fetching ====================

export async function refreshExecutions() {
    const params = new URLSearchParams();
    const f = executionsState.filters;

    if (f.workflowId) params.set('workflow_id', f.workflowId);
    if (f.status && f.status !== 'queued') params.set('status', f.status);
    if (f.since) params.set('since', f.since);
    params.set('limit', '200');

    try {
        const resp = await fetch(`${API}/n8n/executions?${params}`);
        const data = await resp.json();

        executionsState.executions = data.executions || [];
        executionsState.queued = data.queued || [];
        executionsState.stats = data.stats || {};

        renderExecutionStats();
        renderExecutionList();
        adjustRefreshRate();
    } catch (e) {
        console.error('Failed to fetch executions:', e);
    }
}

async function populateWorkflowFilter() {
    try {
        const resp = await fetch(`${API}/n8n/history/workflows`);
        const data = await resp.json();
        executionsState.workflows = data.workflows || [];

        const select = document.getElementById('exec-filter-workflow');
        if (!select) return;

        // Keep "All Workflows" option, clear the rest
        while (select.options.length > 1) select.remove(1);

        for (const wf of executionsState.workflows) {
            const opt = document.createElement('option');
            opt.value = wf.id;
            opt.textContent = wf.name || wf.id;
            select.appendChild(opt);
        }
    } catch (e) {
        console.error('Failed to load workflow filter:', e);
    }
}

// ==================== Filters ====================

export function applyExecutionFilters() {
    const wfSelect = document.getElementById('exec-filter-workflow');
    const statusSelect = document.getElementById('exec-filter-status');
    const timeSelect = document.getElementById('exec-filter-time');

    executionsState.filters.workflowId = wfSelect?.value || null;
    executionsState.filters.status = statusSelect?.value || null;

    // Convert time range to ISO since timestamp
    const timeVal = timeSelect?.value || '';
    if (timeVal) {
        const now = Date.now();
        const offsets = { '1h': 3600000, '24h': 86400000, '7d': 604800000 };
        const ms = offsets[timeVal] || 0;
        executionsState.filters.since = ms ? new Date(now - ms).toISOString() : null;
    } else {
        executionsState.filters.since = null;
    }

    refreshExecutions();
    populateWorkflowFilter();
}

// ==================== Rendering: Stats ====================

function renderExecutionStats() {
    const el = document.getElementById('executions-stats');
    if (!el) return;

    const s = executionsState.stats;
    const total = (s.total || 0) + (s.running || 0);
    const success = s.success || 0;
    const error = s.error || 0;
    const running = s.running || 0;
    const queued = s.queued;
    const queuedText = queued === null ? '\u221e' : (queued || 0);

    el.innerHTML = `
        <span class="exec-stat">Total: <strong>${total}</strong></span>
        <span class="exec-stat stat-success">Success: <strong>${success}</strong></span>
        <span class="exec-stat stat-error">Errors: <strong>${error}</strong></span>
        <span class="exec-stat stat-running">Running: <strong>${running}</strong></span>
        <span class="exec-stat stat-queued">Queued: <strong>${queuedText}</strong></span>
    `;
}

// ==================== Rendering: List ====================

function renderExecutionList() {
    const el = document.getElementById('executions-list');
    if (!el) return;

    const execs = executionsState.executions;
    const queued = executionsState.queued;
    const statusFilter = executionsState.filters.status;

    // If filtering by "queued", show only queued rows
    if (statusFilter === 'queued') {
        if (queued.length === 0) {
            el.innerHTML = '<div class="empty-state">No queued executions</div>';
            return;
        }
        el.innerHTML = queued.map(renderQueuedRow).join('');
        return;
    }

    if (execs.length === 0 && queued.length === 0) {
        el.innerHTML = '<div class="empty-state">No executions yet</div>';
        return;
    }

    let html = '';

    // Column headers
    html += `<div class="execution-row execution-header">
        <span class="exec-status-icon"></span>
        <span class="exec-workflow">Workflow</span>
        <span class="exec-port">Port</span>
        <span class="exec-started">Started</span>
        <span class="exec-duration">Duration</span>
        <span class="exec-id">ID</span>
    </div>`;

    // Queued summary rows at top
    if (!statusFilter && queued.length > 0) {
        html += queued.map(renderQueuedRow).join('');
    }

    // Execution rows
    html += execs.map(renderExecutionRow).join('');

    el.innerHTML = html;
}

function renderQueuedRow(q) {
    const remaining = q.remaining === null ? '\u221e' : q.remaining;
    const mode = q.parallel ? 'PAR' : 'SEQ';
    const state = q.paused ? 'PAUSED' : (q.processing ? 'PROCESSING' : 'IDLE');
    const stateClass = q.paused ? 'paused' : (q.processing ? 'running' : '');

    return `
        <div class="execution-row queued">
            <span class="exec-status-icon queued-icon">&#9719;</span>
            <span class="exec-workflow">${esc(q.workflow_name || q.workflow_id)}</span>
            <span class="exec-port">:${q.worker_port}</span>
            <span class="exec-started">Queued</span>
            <span class="exec-duration">${remaining} remaining (${mode})</span>
            <span class="exec-id ${stateClass}">${state}</span>
        </div>
    `;
}

function renderExecutionRow(ex) {
    const status = normalizeStatus(ex.status);
    const icon = statusIcon(status);
    const workflow = ex.workflow_name || ex.workflow_id || '?';
    const port = ex.worker_port || '?';
    const started = ex.started_at ? timeAgo(ex.started_at) : '-';
    const duration = formatDuration(ex.execution_time_ms);
    const execId = ex.id || '?';
    const expanded = executionsState.expandedIds.has(execId);

    let html = `
        <div class="execution-row ${status}" onclick="toggleExecutionDetail('${esc(execId)}')">
            <span class="exec-status-icon">${icon}</span>
            <span class="exec-workflow">${esc(workflow)}</span>
            <span class="exec-port">:${port}</span>
            <span class="exec-started" title="${esc(ex.started_at || '')}">${started}</span>
            <span class="exec-duration">${duration}</span>
            <span class="exec-id">${esc(String(execId))}</span>
        </div>
    `;

    if (expanded) {
        html += renderExecutionDetail(ex);
    }

    return html;
}

function renderExecutionDetail(ex) {
    const parts = [];

    if (ex.started_at) parts.push(`<div><strong>Started:</strong> ${esc(ex.started_at)}</div>`);
    if (ex.finished_at) parts.push(`<div><strong>Finished:</strong> ${esc(ex.finished_at)}</div>`);
    if (ex.execution_time_ms != null) parts.push(`<div><strong>Duration:</strong> ${ex.execution_time_ms}ms</div>`);
    if (ex.mode) parts.push(`<div><strong>Mode:</strong> ${esc(ex.mode)}</div>`);
    if (ex.error_message) parts.push(`<div class="exec-error-msg"><strong>Error:</strong> ${esc(ex.error_message)}</div>`);

    return `<div class="execution-detail">${parts.join('')}</div>`;
}

// ==================== Actions ====================

export function toggleExecutionDetail(id) {
    if (executionsState.expandedIds.has(id)) {
        executionsState.expandedIds.delete(id);
    } else {
        executionsState.expandedIds.add(id);
    }
    renderExecutionList();
}

export async function clearExecutionHistory() {
    if (!confirm('Clear all execution history? This cannot be undone.')) return;

    try {
        await fetch(`${API}/n8n/history`, { method: 'DELETE' });
        executionsState.expandedIds.clear();
        await refreshExecutions();
        await populateWorkflowFilter();
    } catch (e) {
        console.error('Failed to clear history:', e);
    }
}

// ==================== Helpers ====================

function normalizeStatus(s) {
    if (!s) return 'unknown';
    if (s === 'finished' || s === 'success') return 'success';
    if (s === 'failed' || s === 'crashed' || s === 'error') return 'error';
    if (s === 'running' || s === 'waiting' || s === 'new') return 'running';
    return s;
}

function statusIcon(status) {
    switch (status) {
        case 'success': return '<span class="status-dot success"></span>';
        case 'error':   return '<span class="status-dot error"></span>';
        case 'running': return '<span class="status-dot running"></span>';
        default:        return '<span class="status-dot unknown"></span>';
    }
}

function timeAgo(isoStr) {
    if (!isoStr) return '-';
    try {
        // n8n stores timestamps in UTC without timezone suffix
        // e.g. "2026-02-14 17:45:56.275" — normalize to proper ISO with Z
        let str = isoStr;
        if (str.includes(' ') && !str.includes('T')) {
            str = str.replace(' ', 'T');
        }
        if (!/[Z+]/.test(str.slice(10))) {
            str += 'Z';
        }
        const d = new Date(str);
        if (isNaN(d.getTime())) return isoStr;
        const diff = Date.now() - d.getTime();
        if (diff < 0) return 'just now';
        if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return `${Math.floor(diff / 86400000)}d ago`;
    } catch {
        return isoStr;
    }
}

function formatDuration(ms) {
    if (ms == null) return '-';
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
}

function esc(s) {
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}
