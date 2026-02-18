/**
 * n8n queue, workers, tabs, polling for AgentNate UI
 */

import { state, API, queueState, marketplaceState } from './state.js';
import { log, escapeHtml, showToast, updateToast, debugLog } from './utils.js';

// Lazy imports
async function getTabs() {
    return await import('./tabs.js');
}

const N8N_HOST = (typeof window !== 'undefined' && window.location && window.location.hostname)
    ? window.location.hostname
    : 'localhost';

export async function refreshQueueStatus() {
    try {
        const mainResp = await fetch(`${API}/n8n/main/status`);
        const mainData = await mainResp.json();
        queueState.mainRunning = mainData.running;
        queueState.mainPort = mainData.port || 5678;

        const workersResp = await fetch(`${API}/n8n/workers`);
        const workersData = await workersResp.json();
        queueState.workers = workersData.workers || [];

        const workerSummary = queueState.workers.map(w => `${w.port}:${w.active?'active':'inactive'}:${w.mode}`).join(', ');
        debugLog('refreshQueueStatus', `main=${queueState.mainRunning} workers=[${workerSummary}]`);

        renderQueueSidebar();
        updateWorkersCount();

        if (marketplaceState.currentSubtab === 'workers') {
            renderWorkersFullList();
        }
    } catch (e) {
        console.error('Failed to refresh queue status:', e);
    }
}

function relativeTime(isoString) {
    if (!isoString) return '';
    const diff = (Date.now() - new Date(isoString).getTime()) / 1000;
    if (diff < 5) return 'just now';
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

function statusIcon(status) {
    if (status === 'success') return '<span class="exec-status success" title="Last: Success">&#10003;</span>';
    if (status === 'error') return '<span class="exec-status error" title="Last: Failed">&#10007;</span>';
    if (status === 'running') return '<span class="exec-status running" title="Executing...">&#9679;</span>';
    return '';
}

function _formatCounter(w) {
    const completed = w.execution_count || 0;
    const total = w.queued_total;
    if (total === null || total === undefined) {
        return `${completed}/\u221E`;
    } else if (total > 0) {
        return `${completed}/${total}`;
    } else {
        // queued_total is 0 — no runs queued yet
        return `${completed}`;
    }
}

function _getDropdownVal(w) {
    // For trigger-based, use mode + loop_target
    if (w.trigger_count > 0) {
        const target = w.mode === 'once' ? 1 : (w.loop_target || 0);
        if (w.mode === 'loop' || w.mode === 'standby') {
            if (target === 10) return '10';
            if (target === 100) return '100';
            if (target === 0 || target === null) return '0';
            return String(target);
        }
        return '1';
    }
    // For triggerless, use queued_total as hint for dropdown
    const qt = w.queued_total;
    if (qt === null) return '0';  // infinite
    if (qt >= 100) return '100';
    if (qt >= 10) return '10';
    return '1';
}

export function renderQueueSidebar() {
    const statusDot = document.getElementById('main-status-dot');
    const statusText = document.getElementById('main-status-text');
    const openBtn = document.getElementById('main-open-btn');

    if (statusDot) {
        statusDot.className = 'status-dot' + (queueState.mainRunning ? ' running' : '');
    }
    if (statusText) {
        statusText.textContent = queueState.mainRunning ? 'Running' : 'Stopped';
    }
    if (openBtn) {
        openBtn.disabled = false;
        openBtn.textContent = queueState.mainRunning ? 'Open' : 'Start';
        openBtn.title = queueState.mainRunning ? 'Open n8n dashboard' : 'Start Main Admin server';
    }
    // Workers are rendered in the Workflows tab (renderWorkersFullList), not in sidebar
}

export async function openMainAdminTab() {
    debugLog('openMainAdminTab', `mainRunning=${queueState.mainRunning}`);
    if (!queueState.mainRunning) {
        log('Starting Main Admin (this may take 1-2 minutes on first run)...', 'info');
        try {
            const resp = await fetch(`${API}/n8n/main/start`, { method: 'POST' });
            const data = await resp.json();
            if (data.success && data.instance) {
                queueState.mainRunning = true;
                queueState.mainPort = data.instance.port;
                renderQueueSidebar();
                log('Main Admin process started on port ' + data.instance.port + ', waiting for n8n to initialize...', 'success');
            } else {
                log('Failed to start Main Admin: ' + (data.error || 'Unknown error'), 'error');
                return;
            }
        } catch (e) {
            log('Failed to start Main Admin: ' + e.message, 'error');
            return;
        }
    }

    openN8nTab(queueState.mainPort);
}

export async function toggleWorker(port) {
    // Used for trigger-based workflows only (activate/deactivate)
    const worker = queueState.workers.find(w => w.port === port);
    if (!worker) {
        debugLog('toggleWorker', `port=${port} ERROR: worker not found in queueState`);
        return;
    }

    debugLog('toggleWorker', `port=${port} active=${worker.active} trigger_count=${worker.trigger_count} -> will ${worker.active ? 'deactivate' : 'activate'}`);

    if (worker.active) {
        await deactivateWorker(port);
    } else {
        await activateWorker(port);
    }
}

export async function enqueueRuns(port) {
    const worker = queueState.workers.find(w => w.port === port);
    const dropdown = document.querySelector(`.worker-card[data-port="${port}"] .run-count-select`)
        || document.querySelector(`.run-count-select[data-port="${port}"]`);
    const rawVal = dropdown ? parseInt(dropdown.value, 10) : 1;
    const count = rawVal === 0 ? null : rawVal;  // 0 means infinite
    const parallel = worker?.queued_parallel || false;

    debugLog('enqueueRuns', `port=${port} count=${count} parallel=${parallel}`);

    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/enqueue`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({count, parallel})
        });
        const data = await resp.json();
        if (data.success) {
            log(`Enqueued ${count || '\u221E'} runs on :${port} (total: ${data.queued_total || '\u221E'})`, 'success');
        } else {
            log(`Failed to enqueue: ${data.detail || data.error}`, 'error');
        }
        await refreshQueueStatus();
    } catch (e) {
        log(`Error enqueueing runs: ${e.message}`, 'error');
    }
}

export async function pauseWorker(port) {
    debugLog('pauseWorker', `port=${port}`);
    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/pause`, {method: 'POST'});
        const data = await resp.json();
        if (data.success) {
            log(`Worker :${port} paused`, 'success');
        } else {
            log(`Failed to pause: ${data.detail || data.error}`, 'error');
        }
        await refreshQueueStatus();
    } catch (e) {
        log(`Error pausing worker: ${e.message}`, 'error');
    }
}

export async function toggleParallel(port) {
    const worker = queueState.workers.find(w => w.port === port);
    if (!worker) return;

    // Toggle locally — sent with next enqueue call
    worker.queued_parallel = !worker.queued_parallel;
    debugLog('toggleParallel', `port=${port} parallel=${worker.queued_parallel}`);
    renderQueueSidebar();
    if (marketplaceState.currentSubtab === 'workers') {
        renderWorkersFullList();
    }
}

export async function setRunCount(port, value) {
    const count = parseInt(value, 10);
    const mode = count === 1 ? 'once' : 'loop';
    const loopTarget = count === 0 ? null : (count === 1 ? null : count);
    debugLog('setRunCount', `port=${port} value=${value} -> mode=${mode} loop_target=${loopTarget}`);

    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/mode`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, loop_target: loopTarget })
        });
        const data = await resp.json();
        if (data.success) {
            await refreshQueueStatus();
        } else {
            log(`Failed to set run count: ${data.detail || data.error}`, 'error');
            await refreshQueueStatus();
        }
    } catch (e) {
        log(`Error setting run count: ${e.message}`, 'error');
        await refreshQueueStatus();
    }
}

export async function clearRunCounter(port) {
    debugLog('clearRunCounter', `port=${port}`);
    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/reset-counter`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            await refreshQueueStatus();
        } else {
            log(`Failed to reset counter: ${data.detail || data.error}`, 'error');
        }
    } catch (e) {
        log(`Error resetting counter: ${e.message}`, 'error');
    }
}

export async function closeWorker(port) {
    debugLog('closeWorker', `port=${port}`);
    try {
        log(`Removing worker on port ${port}...`, 'info');
        const resp = await fetch(`${API}/n8n/workers/${port}`, { method: 'DELETE' });
        const data = await resp.json();

        if (data.success) {
            log(`Worker on port ${port} removed`, 'success');
            closeN8nTab(port);
            await refreshQueueStatus();
        } else {
            log(`Failed to remove worker: ${data.error}`, 'error');
        }
    } catch (e) {
        log(`Error removing worker: ${e.message}`, 'error');
    }
}

export function openWorkerTab(port) {
    const worker = queueState.workers.find(w => w.port === port);
    debugLog('openWorkerTab', `port=${port} workflow_id=${worker?.workflow_id || 'none'}`);
    openN8nTab(port, worker ? worker.workflow_id : null);
}

export async function refreshN8nInstances() {
    await refreshQueueStatus();
}

export function renderN8nInstances(instances) {
    renderQueueSidebar();
}

export async function stopAllN8n() {
    debugLog('stopAllN8n', 'starting');
    const btn = event?.target;
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Stopping...';
    }

    try {
        const resp = await fetch(`${API}/n8n/list`);
        const data = await resp.json();

        log(`Stopping ${data.instances.length} n8n instances...`, 'info');

        for (const instance of data.instances) {
            try {
                await fetch(`${API}/n8n/${instance.port}`, { method: 'DELETE' });
                closeN8nTab(instance.port);
            } catch (e) {
                console.error(`Failed to stop n8n on port ${instance.port}:`, e);
            }
        }

        await refreshN8nInstances();
        log('All n8n instances stopped', 'success');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Stop All';
        }
    }
}

export async function spawnN8n() {
    debugLog('spawnN8n', 'starting');
    if (state.spawningN8n) {
        log('n8n spawn already in progress...', 'warning');
        return;
    }

    state.spawningN8n = true;
    const btn = document.querySelector('[onclick="spawnN8n()"]');
    if (btn) btn.classList.add('loading');

    log('Spawning new n8n instance...', 'info');

    try {
        const resp = await fetch(`${API}/n8n/spawn`, { method: 'POST' });
        const result = await resp.json();

        if (result.success) {
            log(`n8n spawned on port ${result.instance.port} (PID: ${result.instance.pid})`, 'success');
            await refreshN8nInstances();
            openN8nTab(result.instance.port);
        } else {
            log('Failed to spawn n8n: ' + result.error, 'error');
            alert('Failed to spawn n8n: ' + result.error);
        }
    } catch (e) {
        log('Error spawning n8n: ' + e.message, 'error');
        alert('Error: ' + e.message);
    } finally {
        state.spawningN8n = false;
        if (btn) btn.classList.remove('loading');
    }
}

export async function stopN8n(port, btn = null) {
    debugLog('stopN8n', `port=${port}`);
    if (!btn && typeof event !== 'undefined' && event?.target) {
        btn = event.target;
    }
    if (btn) {
        btn.disabled = true;
        btn.textContent = '...';
    }

    try {
        log(`Stopping n8n on port ${port}...`, 'info');
        const resp = await fetch(`${API}/n8n/${port}`, { method: 'DELETE' });
        const data = await resp.json();

        if (data.success) {
            log(`n8n on port ${port} stopped`, 'success');
            closeN8nTab(port);
        } else {
            log(`Failed to stop n8n: ${data.error || 'Unknown error'}`, 'error');
        }

        await refreshN8nInstances();
    } catch (e) {
        console.error('Failed to stop n8n:', e);
        log(`Error stopping n8n: ${e.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = '×';
        }
    }
}

export async function openN8nTab(port, workflowId = null) {
    debugLog('openN8nTab', `port=${port} workflowId=${workflowId}`);
    const tabId = `n8n-${port}`;

    if (document.getElementById(`tab-${tabId}`)) {
        const tabs = await getTabs();
        tabs.switchTab(tabId);
        // Update iframe URL if a specific workflow was requested
        if (workflowId) {
            const panel = document.getElementById(`tab-${tabId}`);
            const iframe = panel?.querySelector('iframe.n8n-frame');
            if (iframe) {
                const targetUrl = `http://${N8N_HOST}:${port}/workflow/${workflowId}`;
                if (!iframe.src.includes(`/workflow/${workflowId}`)) {
                    iframe.src = targetUrl;
                }
            }
        }
        return;
    }

    const tabsContainer = document.getElementById('tabs');
    const tabBtn = document.createElement('button');
    tabBtn.className = 'tab';
    tabBtn.dataset.tab = tabId;
    tabBtn.innerHTML = `n8n :${port} <span class="close-tab" onclick="event.stopPropagation(); closeN8nTab(${port})">&times;</span>`;
    tabBtn.onclick = () => {
        import('./tabs.js').then(tabs => tabs.switchTab(tabId));
    };
    tabsContainer.appendChild(tabBtn);

    const contentContainer = document.getElementById('tab-content');
    const panel = document.createElement('div');
    panel.id = `tab-${tabId}`;
    panel.className = 'tab-panel';
    panel.innerHTML = `
        <div class="n8n-loading" id="n8n-loading-${port}">
            <div class="loading-spinner"></div>
            <div class="loading-text">Starting n8n on port ${port}...</div>
        </div>
    `;
    contentContainer.appendChild(panel);

    const tabs = await getTabs();
    tabs.switchTab(tabId);
    state.n8nTabs.push(port);

    await waitForN8nReady(port, 60, workflowId);
}

export async function waitForN8nReady(port, maxAttempts = 60, workflowId = null) {
    const loadingEl = document.getElementById(`n8n-loading-${port}`);
    const panel = document.getElementById(`tab-n8n-${port}`);

    log(`Waiting for n8n on port ${port} to be ready (may take up to 2 minutes)...`, 'info');

    for (let i = 0; i < maxAttempts; i++) {
        try {
            const resp = await fetch(`${API}/n8n/${port}/ready`);
            const data = await resp.json();

            if (data.ready) {
                log(`n8n on port ${port} is ready!`, 'success');
                await refreshQueueStatus();
                let n8nUrl = `http://${N8N_HOST}:${port}/`;
                if (workflowId) {
                    n8nUrl = `http://${N8N_HOST}:${port}/workflow/${workflowId}`;
                }
                if (panel) {
                    panel.innerHTML = `<iframe class="n8n-frame" src="${n8nUrl}"></iframe>`;
                }
                return true;
            }

            if (loadingEl) {
                const loadingText = loadingEl.querySelector('.loading-text');
                if (loadingText) {
                    const elapsed = (i + 1) * 2;
                    loadingText.textContent = `Starting n8n on port ${port}... (${elapsed}s elapsed, this can take 1-2 minutes on first run)`;
                }
            }

            if (i % 5 === 0) {
                log(`n8n on port ${port} not ready yet (${i + 1}/${maxAttempts})`, 'info');
            }
        } catch (e) {
            if (i % 10 === 0) {
                log(`Waiting for n8n startup...`, 'info');
            }
        }

        await new Promise(r => setTimeout(r, 2000));
    }

    if (panel) {
        panel.innerHTML = `
            <div class="n8n-error">
                <div class="error-text">n8n on port ${port} is taking longer than expected</div>
                <p style="color: var(--text-muted); margin: 10px 0;">The process may still be starting. You can try again or open it directly.</p>
                <div style="display: flex; gap: 8px; justify-content: center;">
                    <button class="btn" onclick="retryN8nTab(${port})">Retry</button>
                    <button class="btn-secondary" onclick="window.open('http://${N8N_HOST}:${port}/', '_blank')">Open in Browser</button>
                </div>
            </div>
        `;
    }
    return false;
}

export function retryN8nTab(port) {
    const tabId = `n8n-${port}`;
    const panel = document.getElementById(`tab-${tabId}`);
    if (panel) {
        panel.innerHTML = `
            <div class="n8n-loading" id="n8n-loading-${port}">
                <div class="loading-spinner"></div>
                <div class="loading-text">Retrying n8n on port ${port}...</div>
            </div>
        `;
    }
    waitForN8nReady(port);
}

export function closeN8nTab(port) {
    const tabId = `n8n-${port}`;

    const tabBtn = document.querySelector(`.tab[data-tab="${tabId}"]`);
    if (tabBtn) tabBtn.remove();

    const panel = document.getElementById(`tab-${tabId}`);
    if (panel) panel.remove();

    if (document.querySelector('.tab.active')?.dataset.tab === tabId) {
        import('./tabs.js').then(tabs => tabs.switchTab('chat'));
    }

    state.n8nTabs = state.n8nTabs.filter(p => p !== port);
}

export function renderWorkersFullList() {
    const container = document.getElementById('workers-full-list');
    const stopAllBtn = document.getElementById('stop-all-workers-btn');
    if (!container) return;

    const workers = queueState.workers;

    if (stopAllBtn) {
        stopAllBtn.disabled = workers.length === 0;
    }

    if (workers.length === 0) {
        container.innerHTML = '<div class="empty-state">No workers running</div>';
        return;
    }

    // Check if we can do a surgical update instead of full rebuild
    const existingItems = container.querySelectorAll('.worker-full-item');
    const existingPorts = new Set();
    existingItems.forEach(item => {
        const port = item.dataset.port;
        if (port) existingPorts.add(parseInt(port));
    });

    const newPorts = new Set(workers.map(w => w.port));
    const canPatch = existingPorts.size === newPorts.size
        && [...existingPorts].every(p => newPorts.has(p));

    if (canPatch) {
        // Surgical update: only change dynamic fields, preserve DOM elements
        for (const w of workers) {
            const item = container.querySelector(`.worker-full-item[data-port="${w.port}"]`);
            if (!item) { canPatch && _fullRebuild(container, workers); return; }

            const isActive = w.active || w.processing;
            const isTriggerBased = w.trigger_count > 0;

            // Update status dot
            const dot = item.querySelector('.worker-status-dot');
            if (dot) dot.className = `worker-status-dot ${w.is_running ? (isActive ? 'running' : 'paused') : 'stopped'}`;

            // Update relative time
            const metaSpans = item.querySelectorAll('.worker-full-meta span:not(.worker-port):not(.exec-status)');
            const lastExec = w.last_execution ? relativeTime(w.last_execution) : '';
            if (metaSpans.length > 0) metaSpans[0].textContent = lastExec;

            // Update status icon
            const statusEl = item.querySelector('.exec-status');
            const newStatus = w.last_status;
            if (statusEl) {
                statusEl.className = `exec-status ${newStatus || ''}`;
                if (newStatus === 'success') { statusEl.innerHTML = '&#10003;'; statusEl.title = 'Last: Success'; }
                else if (newStatus === 'error') { statusEl.innerHTML = '&#10007;'; statusEl.title = 'Last: Failed'; }
                else if (newStatus === 'running') { statusEl.innerHTML = '&#9679;'; statusEl.title = 'Executing...'; }
            }

            // Update counter
            const counter = item.querySelector('.run-counter');
            if (counter) counter.textContent = _formatCounter(w);

            // Update pause button active state (triggerless only)
            if (!isTriggerBased) {
                const pauseBtn = item.querySelector('.transport-btn.pause');
                if (pauseBtn) {
                    if (w.paused || !w.processing) pauseBtn.classList.remove('active');
                    else pauseBtn.classList.add('active');
                }
                const seqParBtn = item.querySelector('.transport-btn.seq-par');
                if (seqParBtn) {
                    seqParBtn.innerHTML = w.queued_parallel ? '\u21F6' : '\u27F6';
                    seqParBtn.title = w.queued_parallel ? 'Parallel' : 'Sequential';
                }
            } else {
                const toggleBtn = item.querySelector('.transport-btn.toggle');
                if (toggleBtn) {
                    if (isActive) { toggleBtn.classList.add('active'); toggleBtn.innerHTML = '&#9646;&#9646;'; }
                    else { toggleBtn.classList.remove('active'); toggleBtn.innerHTML = '&#9654;'; }
                    toggleBtn.title = isActive ? 'Deactivate' : 'Activate';
                }
            }
        }
        return;
    }

    // Full rebuild (workers added/removed)
    _fullRebuild(container, workers);
}

function _fullRebuild(container, workers) {
    container.innerHTML = workers.map(w => {
        const isTriggerBased = w.trigger_count > 0;
        const isActive = w.active || w.processing;
        const counterText = _formatCounter(w);
        const lastExec = w.last_execution ? relativeTime(w.last_execution) : '';
        const dropdownVal = _getDropdownVal(w);

        const controlsHtml = isTriggerBased ? `
                <button class="transport-btn toggle ${isActive ? 'active' : ''}" onclick="toggleWorker(${w.port})" title="${isActive ? 'Deactivate' : 'Activate'}">
                    ${isActive ? '&#9646;&#9646;' : '&#9654;'}
                </button>
                <select class="run-count-select" onchange="setRunCount(${w.port}, this.value)" title="Run count">
                    <option value="1" ${dropdownVal === '1' ? 'selected' : ''}>1</option>
                    <option value="10" ${dropdownVal === '10' ? 'selected' : ''}>10</option>
                    <option value="100" ${dropdownVal === '100' ? 'selected' : ''}>100</option>
                    <option value="0" ${dropdownVal === '0' ? 'selected' : ''}>\u221E</option>
                </select>` : `
                <button class="transport-btn play" onclick="enqueueRuns(${w.port})" title="Enqueue runs">&#9654;</button>
                <button class="transport-btn pause ${w.paused || !w.processing ? '' : 'active'}" onclick="pauseWorker(${w.port})" title="Pause queue">&#9646;&#9646;</button>
                <select class="run-count-select" data-port="${w.port}" title="Batch size">
                    <option value="1" ${dropdownVal === '1' ? 'selected' : ''}>1</option>
                    <option value="10" ${dropdownVal === '10' ? 'selected' : ''}>10</option>
                    <option value="100" ${dropdownVal === '100' ? 'selected' : ''}>100</option>
                    <option value="0" ${dropdownVal === '0' ? 'selected' : ''}>\u221E</option>
                </select>
                <button class="transport-btn seq-par" onclick="toggleParallel(${w.port})" title="${w.queued_parallel ? 'Parallel' : 'Sequential'}">
                    ${w.queued_parallel ? '\u21F6' : '\u27F6'}
                </button>`;

        return `
        <div class="worker-full-item" data-port="${w.port}">
            <div class="worker-full-info" onclick="openWorkerTab(${w.port})">
                <div class="worker-full-name">
                    <span class="worker-status-dot ${w.is_running ? (isActive ? 'running' : 'paused') : 'stopped'}"></span>
                    ${escapeHtml(w.workflow_name || 'Workflow')}
                </div>
                <div class="worker-full-meta">
                    <span class="worker-port">Port ${w.port}</span>
                    ${lastExec ? `<span>${lastExec}</span>` : ''}
                    ${statusIcon(w.last_status)}
                </div>
            </div>
            <div class="worker-full-actions">
                ${controlsHtml}
                <span class="run-counter" title="Executions">${counterText}</span>
                <button class="transport-btn clr" onclick="clearRunCounter(${w.port})" title="Clear queue & reset">CLR</button>
                <button class="transport-btn close" onclick="closeWorker(${w.port})" title="Remove worker">&times;</button>
                <button class="btn-secondary btn-small" onclick="openWorkerTab(${w.port})">Open</button>
            </div>
        </div>`;
    }).join('');
}

export async function stopAllWorkers() {
    if (!confirm('Stop all running workers?')) return;

    log('Stopping all workers...', 'info');

    for (const worker of queueState.workers) {
        try {
            await fetch(`${API}/n8n/workers/${worker.port}`, { method: 'DELETE' });
            closeN8nTab(worker.port);
        } catch (e) {
            console.error(`Failed to stop worker ${worker.port}:`, e);
        }
    }

    await refreshQueueStatus();
    renderWorkersFullList();
    log('All workers stopped', 'success');
}

export function updateWorkersCount() {
    const badge = document.getElementById('workers-count');
    if (badge) {
        badge.textContent = queueState.workers.length;
    }
}

export async function executeWorker(port) {
    debugLog('executeWorker', `port=${port} (manual/triggerless)`);
    const toast = showToast(`Executing workflow on :${port}...`, 'info', 0);
    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/execute`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            if (data.status === 'running') {
                updateToast(toast, `Workflow on :${port} still running (exec #${data.executionId})`, 'info');
                log(`Worker :${port} execution #${data.executionId} still running after 30s`, 'info');
            } else {
                updateToast(toast, `Workflow on :${port} completed successfully`, 'success');
                log(`Worker :${port} executed successfully (exec #${data.executionId})`, 'success');
            }
        } else {
            const errorMsg = data.error || 'Unknown error';
            updateToast(toast, `Execution failed: ${errorMsg}`, 'error');
            log(`Execute failed on :${port}: ${errorMsg}`, 'error');
        }
        await refreshQueueStatus();
    } catch (e) {
        updateToast(toast, `Error executing: ${e.message}`, 'error');
        log(`Error executing worker: ${e.message}`, 'error');
    }
}

export async function activateWorker(port) {
    debugLog('activateWorker', `port=${port}`);
    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/activate`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            log(`Worker :${port} activated`, 'success');
        } else {
            showToast(`Cannot activate: ${data.error || 'Unknown'}`, 'error', 6000);
            log(`Cannot activate: ${data.error || data.detail || 'Unknown error'}`, 'error');
        }
        await refreshQueueStatus();
    } catch (e) {
        showToast(`Error activating: ${e.message}`, 'error');
        log(`Error activating worker: ${e.message}`, 'error');
    }
}

export async function deactivateWorker(port) {
    debugLog('deactivateWorker', `port=${port}`);
    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/deactivate`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            log(`Worker :${port} paused`, 'success');
            await refreshQueueStatus();
        } else {
            log(`Failed to pause worker: ${data.detail || data.error}`, 'error');
        }
    } catch (e) {
        log(`Error pausing worker: ${e.message}`, 'error');
    }
}

export async function changeWorkerMode(port, mode) {
    let loopTarget = null;
    if (mode === 'loop') {
        const input = prompt('Loop target (leave empty for infinite):', '');
        if (input === null) {
            // User cancelled - revert dropdown
            await refreshQueueStatus();
            return;
        }
        loopTarget = input ? parseInt(input, 10) : null;
        if (loopTarget !== null && (isNaN(loopTarget) || loopTarget < 1)) {
            log('Invalid loop target - must be a positive number', 'error');
            await refreshQueueStatus();
            return;
        }
    }

    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/mode`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, loop_target: loopTarget })
        });
        const data = await resp.json();
        if (data.success) {
            log(`Worker :${port} mode changed to ${mode}${loopTarget ? ' (target: ' + loopTarget + ')' : ''}`, 'success');
            await refreshQueueStatus();
        } else {
            log(`Failed to change mode: ${data.detail || data.error}`, 'error');
            await refreshQueueStatus();
        }
    } catch (e) {
        log(`Error changing mode: ${e.message}`, 'error');
        await refreshQueueStatus();
    }
}

export async function setLoopTarget(port) {
    const worker = queueState.workers.find(w => w.port === port);
    const current = worker?.loop_target || '';
    const input = prompt('Set loop target (leave empty for infinite):', current);
    if (input === null) return;

    const loopTarget = input ? parseInt(input, 10) : null;
    if (loopTarget !== null && (isNaN(loopTarget) || loopTarget < 1)) {
        log('Invalid loop target - must be a positive number', 'error');
        return;
    }

    try {
        const resp = await fetch(`${API}/n8n/workers/${port}/mode`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: worker?.mode || 'loop', loop_target: loopTarget })
        });
        const data = await resp.json();
        if (data.success) {
            log(`Loop target set to ${loopTarget || 'infinite'}`, 'success');
            await refreshQueueStatus();
        } else {
            log(`Failed to set loop target: ${data.detail || data.error}`, 'error');
        }
    } catch (e) {
        log(`Error setting loop target: ${e.message}`, 'error');
    }
}
