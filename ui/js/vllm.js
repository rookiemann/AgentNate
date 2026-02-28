/**
 * vLLM Module UI
 *
 * Handles the vLLM tab: status display, download from GitHub releases,
 * and bootstrap (install.bat). No server management — that stays in
 * the vLLM provider (loaded via Chat sidebar).
 */

import { API } from './state.js';
import { log, apiFetch } from './utils.js';

// Local state (simple module — no need for global state)
let vllmInitialized = false;
let vllmPollingInterval = null;
let downloadPollingInterval = null;

// ======================== Init & Polling ========================

export async function initVLLM() {
    if (vllmInitialized) {
        await refreshVLLMStatus();
        return;
    }
    vllmInitialized = true;
    await refreshVLLMStatus();
    startVLLMPolling();
}

export function startVLLMPolling() {
    stopVLLMPolling();
    vllmPollingInterval = setInterval(refreshVLLMStatus, 10000);
}

export function stopVLLMPolling() {
    if (vllmPollingInterval) {
        clearInterval(vllmPollingInterval);
        vllmPollingInterval = null;
    }
    stopDownloadPolling();
}

function startDownloadPolling() {
    stopDownloadPolling();
    downloadPollingInterval = setInterval(pollDownloadProgress, 1000);
}

function stopDownloadPolling() {
    if (downloadPollingInterval) {
        clearInterval(downloadPollingInterval);
        downloadPollingInterval = null;
    }
}

// ======================== Status ========================

export async function refreshVLLMStatus() {
    try {
        const resp = await apiFetch(`${API}/vllm/status`);
        const data = await resp.json();
        renderStatus(data);
    } catch (e) {
        console.error('Failed to refresh vLLM status:', e);
    }
}

function renderStatus(data) {
    const { module_downloaded, installed, status } = data;

    // Module status card
    const dlEl = document.getElementById('vllm-status-downloaded');
    if (dlEl) {
        dlEl.textContent = module_downloaded ? 'Downloaded' : 'Not Downloaded';
        dlEl.closest('.tts-status-card')?.classList.toggle('ok', module_downloaded);
        dlEl.closest('.tts-status-card')?.classList.toggle('off', !module_downloaded);
    }

    // Installed status card
    const instEl = document.getElementById('vllm-status-installed');
    if (instEl) {
        if (status === 'bootstrapping') {
            instEl.textContent = 'Installing...';
            instEl.closest('.tts-status-card')?.classList.remove('ok', 'off');
        } else {
            instEl.textContent = installed ? 'Installed' : 'Not Installed';
            instEl.closest('.tts-status-card')?.classList.toggle('ok', installed);
            instEl.closest('.tts-status-card')?.classList.toggle('off', !installed);
        }
    }

    // Download button state
    const dlBtn = document.getElementById('vllm-download-btn');
    if (dlBtn) {
        if (module_downloaded) {
            dlBtn.textContent = 'Downloaded';
            dlBtn.disabled = true;
        } else if (status === 'downloading') {
            dlBtn.textContent = 'Downloading...';
            dlBtn.disabled = true;
        } else {
            dlBtn.textContent = 'Download Module';
            dlBtn.disabled = false;
        }
    }

    // Bootstrap button state
    const bsBtn = document.getElementById('vllm-bootstrap-btn');
    if (bsBtn) {
        if (installed) {
            bsBtn.textContent = 'Installed';
            bsBtn.disabled = true;
        } else if (status === 'bootstrapping') {
            bsBtn.textContent = 'Installing...';
            bsBtn.disabled = true;
        } else {
            bsBtn.textContent = 'Install';
            bsBtn.disabled = !module_downloaded;
        }
    }

    // Usage section visibility
    const usageSection = document.getElementById('vllm-usage-section');
    if (usageSection) {
        usageSection.style.display = installed ? '' : 'none';
    }
}

// ======================== Download ========================

export async function downloadVLLMModule() {
    const btn = document.getElementById('vllm-download-btn');
    if (btn) {
        btn.textContent = 'Downloading...';
        btn.disabled = true;
    }

    // Show progress bar
    const progressWrap = document.getElementById('vllm-download-progress');
    if (progressWrap) progressWrap.classList.remove('hidden');

    log('Downloading vLLM module from GitHub...', 'info');

    // Start polling download progress
    startDownloadPolling();

    try {
        const resp = await apiFetch(`${API}/vllm/module/download`, { method: 'POST' });
        const data = await resp.json();

        stopDownloadPolling();

        if (data.success) {
            log('vLLM module downloaded successfully', 'success');
            // Update progress to 100%
            const fill = document.getElementById('vllm-progress-fill');
            const text = document.getElementById('vllm-progress-text');
            if (fill) fill.style.width = '100%';
            if (text) text.textContent = 'Extraction complete';
        } else {
            log('vLLM download failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        stopDownloadPolling();
        log('vLLM download error: ' + e.message, 'error');
    }

    await refreshVLLMStatus();
}

async function pollDownloadProgress() {
    try {
        const resp = await apiFetch(`${API}/vllm/module/download-progress`);
        const data = await resp.json();

        const fill = document.getElementById('vllm-progress-fill');
        const text = document.getElementById('vllm-progress-text');

        if (data.phase === 'downloading' && data.total_bytes > 0) {
            const pct = Math.round((data.downloaded_bytes / data.total_bytes) * 100);
            const downloadedMB = (data.downloaded_bytes / 1024 / 1024).toFixed(1);
            const totalMB = (data.total_bytes / 1024 / 1024).toFixed(1);
            if (fill) fill.style.width = pct + '%';
            if (text) text.textContent = `Downloading: ${downloadedMB} / ${totalMB} MB (${pct}%)`;
        } else if (data.phase === 'extracting') {
            if (fill) fill.style.width = '95%';
            if (text) text.textContent = 'Extracting...';
        } else if (data.phase === 'fetching_release') {
            if (fill) fill.style.width = '2%';
            if (text) text.textContent = 'Fetching release info...';
        } else if (data.phase === 'error') {
            if (text) text.textContent = 'Error: ' + data.error;
            stopDownloadPolling();
        } else if (data.phase === 'done') {
            if (fill) fill.style.width = '100%';
            if (text) text.textContent = 'Done';
            stopDownloadPolling();
        }
    } catch (e) {
        // Ignore polling errors
    }
}

// ======================== Bootstrap ========================

export async function bootstrapVLLMModule() {
    const btn = document.getElementById('vllm-bootstrap-btn');
    if (btn) {
        btn.textContent = 'Installing...';
        btn.disabled = true;
    }

    log('Installing vLLM (Python 3.10 + PyTorch + vLLM wheel)... This may take several minutes.', 'info');

    try {
        const resp = await apiFetch(`${API}/vllm/module/bootstrap`, { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            log('vLLM installed successfully!', 'success');
        } else {
            log('vLLM install failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        log('vLLM install error: ' + e.message, 'error');
    }

    await refreshVLLMStatus();
}
