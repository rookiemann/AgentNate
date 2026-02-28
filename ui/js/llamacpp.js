/**
 * llama.cpp Module UI
 *
 * Handles the llama.cpp tab: status display and one-click wheel install.
 * Much simpler than vLLM — just a single pip install into AgentNate's own Python.
 */

import { API } from './state.js';
import { log, apiFetch } from './utils.js';

let llamacppInitialized = false;
let llamacppPollingInterval = null;
let installPollingInterval = null;

// ======================== Init & Polling ========================

export async function initLlamaCpp() {
    if (llamacppInitialized) {
        await refreshLlamaCppStatus();
        return;
    }
    llamacppInitialized = true;
    await refreshLlamaCppStatus();
    startLlamaCppPolling();
}

export function startLlamaCppPolling() {
    stopLlamaCppPolling();
    llamacppPollingInterval = setInterval(refreshLlamaCppStatus, 10000);
}

export function stopLlamaCppPolling() {
    if (llamacppPollingInterval) {
        clearInterval(llamacppPollingInterval);
        llamacppPollingInterval = null;
    }
    stopInstallPolling();
}

function startInstallPolling() {
    stopInstallPolling();
    installPollingInterval = setInterval(pollInstallProgress, 1000);
}

function stopInstallPolling() {
    if (installPollingInterval) {
        clearInterval(installPollingInterval);
        installPollingInterval = null;
    }
}

// ======================== Status ========================

export async function refreshLlamaCppStatus() {
    try {
        const resp = await apiFetch(`${API}/llamacpp/status`);
        const data = await resp.json();
        renderStatus(data);
    } catch (e) {
        console.error('Failed to refresh llama.cpp status:', e);
    }
}

function renderStatus(data) {
    const { installed, status } = data;

    // Status card
    const statusEl = document.getElementById('llamacpp-status-installed');
    if (statusEl) {
        if (status === 'downloading' || status === 'installing') {
            statusEl.textContent = status === 'downloading' ? 'Downloading...' : 'Installing...';
            statusEl.closest('.tts-status-card')?.classList.remove('ok', 'off');
        } else {
            statusEl.textContent = installed ? 'Installed' : 'Not Installed';
            statusEl.closest('.tts-status-card')?.classList.toggle('ok', installed);
            statusEl.closest('.tts-status-card')?.classList.toggle('off', !installed);
        }
    }

    // Install button
    const btn = document.getElementById('llamacpp-install-btn');
    if (btn) {
        if (installed) {
            btn.textContent = 'Installed';
            btn.disabled = true;
        } else if (status === 'downloading' || status === 'installing') {
            btn.textContent = status === 'downloading' ? 'Downloading...' : 'Installing...';
            btn.disabled = true;
        } else {
            btn.textContent = 'Install llama.cpp';
            btn.disabled = false;
        }
    }

    // Usage section
    const usageSection = document.getElementById('llamacpp-usage-section');
    if (usageSection) {
        usageSection.style.display = installed ? '' : 'none';
    }
}

// ======================== Install ========================

export async function installLlamaCpp() {
    const btn = document.getElementById('llamacpp-install-btn');
    if (btn) {
        btn.textContent = 'Downloading...';
        btn.disabled = true;
    }

    // Show progress bar
    const progressWrap = document.getElementById('llamacpp-install-progress');
    if (progressWrap) progressWrap.classList.remove('hidden');

    log('Installing llama-cpp-python CUDA wheel...', 'info');

    startInstallPolling();

    try {
        const resp = await apiFetch(`${API}/llamacpp/install`, { method: 'POST' });
        const data = await resp.json();

        stopInstallPolling();

        if (data.success) {
            log('llama-cpp-python installed successfully', 'success');
            const fill = document.getElementById('llamacpp-progress-fill');
            const text = document.getElementById('llamacpp-progress-text');
            if (fill) fill.style.width = '100%';
            if (text) text.textContent = 'Installation complete';
        } else {
            log('llama.cpp install failed: ' + (data.error || 'Unknown error'), 'error');
            const text = document.getElementById('llamacpp-progress-text');
            if (text) text.textContent = 'Error: ' + (data.error || 'Unknown');
        }
    } catch (e) {
        stopInstallPolling();
        log('llama.cpp install error: ' + e.message, 'error');
    }

    await refreshLlamaCppStatus();
}

async function pollInstallProgress() {
    try {
        const resp = await apiFetch(`${API}/llamacpp/install-progress`);
        const data = await resp.json();

        const fill = document.getElementById('llamacpp-progress-fill');
        const text = document.getElementById('llamacpp-progress-text');

        if (data.phase === 'downloading' && data.total_bytes > 0) {
            const pct = Math.round((data.downloaded_bytes / data.total_bytes) * 100);
            const downloadedMB = (data.downloaded_bytes / 1024 / 1024).toFixed(1);
            const totalMB = (data.total_bytes / 1024 / 1024).toFixed(1);
            if (fill) fill.style.width = pct + '%';
            if (text) text.textContent = `Downloading: ${downloadedMB} / ${totalMB} MB (${pct}%)`;
        } else if (data.phase === 'installing') {
            if (fill) fill.style.width = '90%';
            if (text) text.textContent = 'Running pip install...';
        } else if (data.phase === 'error') {
            if (text) text.textContent = 'Error: ' + data.error;
            stopInstallPolling();
        } else if (data.phase === 'done') {
            if (fill) fill.style.width = '100%';
            if (text) text.textContent = 'Done';
            stopInstallPolling();
        }
    } catch (e) {
        // Ignore polling errors
    }
}
