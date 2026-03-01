/**
 * vLLM Module UI
 *
 * Handles the vLLM tab: status display and single-step install
 * (download + bootstrap) with real-time progress. No server management —
 * that stays in the vLLM provider (loaded via Chat sidebar).
 */

import { API } from './state.js';
import { log, apiFetch } from './utils.js';

// Local state
let vllmInitialized = false;
let vllmPollingInterval = null;
let installPollingInterval = null;

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
    const { installed, status } = data;

    // Status card
    const instEl = document.getElementById('vllm-status-installed');
    if (instEl) {
        if (status === 'installing') {
            instEl.textContent = 'Installing...';
            instEl.closest('.tts-status-card')?.classList.remove('ok', 'off');
        } else {
            instEl.textContent = installed ? 'Installed' : 'Not Installed';
            instEl.closest('.tts-status-card')?.classList.toggle('ok', installed);
            instEl.closest('.tts-status-card')?.classList.toggle('off', !installed);
        }
    }

    // Install button state
    const btn = document.getElementById('vllm-install-btn');
    if (btn) {
        if (installed) {
            btn.textContent = 'Installed';
            btn.disabled = true;
        } else if (status === 'installing') {
            btn.textContent = 'Installing...';
            btn.disabled = true;
        } else {
            btn.textContent = 'Install vLLM';
            btn.disabled = false;
        }
    }

    // Usage section visibility
    const usageSection = document.getElementById('vllm-usage-section');
    if (usageSection) {
        usageSection.style.display = installed ? '' : 'none';
    }
}

// ======================== Install ========================

export async function installVLLMModule() {
    const btn = document.getElementById('vllm-install-btn');
    if (btn) {
        btn.textContent = 'Installing...';
        btn.disabled = true;
    }

    // Show progress bar
    const progressWrap = document.getElementById('vllm-install-progress');
    if (progressWrap) progressWrap.classList.remove('hidden');

    log('Installing vLLM (download + Python 3.10 + PyTorch + vLLM wheel)... This may take 10-20 minutes.', 'info');

    // Start polling install progress
    startInstallPolling();

    try {
        const resp = await apiFetch(`${API}/vllm/install`, { method: 'POST' });
        const data = await resp.json();

        if (!data.success) {
            log('vLLM install failed to start: ' + (data.error || 'Unknown error'), 'error');
            stopInstallPolling();
            await refreshVLLMStatus();
        }
        // Otherwise polling handles the rest — progress updates come via pollInstallProgress
    } catch (e) {
        stopInstallPolling();
        log('vLLM install error: ' + e.message, 'error');
        await refreshVLLMStatus();
    }
}

async function pollInstallProgress() {
    try {
        const resp = await apiFetch(`${API}/vllm/install-progress`);
        const data = await resp.json();

        const fill = document.getElementById('vllm-progress-fill');
        const text = document.getElementById('vllm-progress-text');

        if (data.phase === 'fetching_release') {
            if (fill) fill.style.width = '2%';
            if (text) text.textContent = 'Checking latest release...';
        } else if (data.phase === 'downloading') {
            if (data.total_bytes > 0) {
                const pct = Math.round((data.downloaded_bytes / data.total_bytes) * 100);
                const downloadedMB = (data.downloaded_bytes / 1024 / 1024).toFixed(1);
                const totalMB = (data.total_bytes / 1024 / 1024).toFixed(1);
                // Download is roughly 0-30% of total progress
                const barPct = Math.round(pct * 0.3);
                if (fill) fill.style.width = barPct + '%';
                if (text) text.textContent = `Downloading: ${downloadedMB} / ${totalMB} MB (${pct}%)`;
            } else {
                if (fill) fill.style.width = '5%';
                if (text) text.textContent = 'Downloading...';
            }
        } else if (data.phase === 'extracting') {
            if (fill) fill.style.width = '32%';
            if (text) text.textContent = 'Extracting files...';
        } else if (data.phase === 'installing_python') {
            if (fill) fill.style.width = '38%';
            if (text) text.textContent = data.detail || 'Installing Python 3.10...';
        } else if (data.phase === 'installing_pytorch') {
            if (fill) fill.style.width = '50%';
            if (text) text.textContent = 'Installing PyTorch (~2.5 GB)... this takes several minutes';
        } else if (data.phase === 'installing_vllm') {
            if (fill) fill.style.width = '80%';
            if (text) text.textContent = 'Installing vLLM wheel...';
        } else if (data.phase === 'installing_deps') {
            if (fill) fill.style.width = '90%';
            if (text) text.textContent = 'Installing additional dependencies...';
        } else if (data.phase === 'verifying') {
            if (fill) fill.style.width = '92%';
            if (text) text.textContent = 'Verifying installation...';
        } else if (data.phase === 'done') {
            if (fill) fill.style.width = '100%';
            if (text) text.textContent = 'Installation complete!';
            stopInstallPolling();
            log('vLLM installed successfully!', 'success');
            await refreshVLLMStatus();
        } else if (data.phase === 'error') {
            if (text) text.textContent = 'Error: ' + data.error;
            stopInstallPolling();
            log('vLLM install failed: ' + data.error, 'error');
            await refreshVLLMStatus();
        }
    } catch (e) {
        // Ignore polling errors
    }
}
