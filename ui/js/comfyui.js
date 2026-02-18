/**
 * ComfyUI Module - Frontend
 *
 * Manages the ComfyUI tab: installation, instances, models, custom nodes.
 * All instance management is inline within the tab (no sidebar rows).
 */

import { state, API, comfyuiState } from './state.js';
import { log, showToast, updateToast, debugLog, apiFetch } from './utils.js';

let pollingInterval = null;

// ======================== Initialization ========================

export async function initComfyUI() {
    if (comfyuiState.initialized) {
        startComfyUIPolling();
        return;
    }
    comfyuiState.initialized = true;
    await refreshComfyUIStatus();
    startComfyUIPolling();
}

export function startComfyUIPolling() {
    if (pollingInterval) return;
    pollingInterval = setInterval(refreshComfyUIStatus, 5000);
}

export function stopComfyUIPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// ======================== Status ========================

export async function refreshComfyUIStatus() {
    try {
        const resp = await fetch(`${API}/comfyui/status`);
        const data = await resp.json();

        comfyuiState.moduleDownloaded = data.module_downloaded || false;
        comfyuiState.bootstrapped = data.bootstrapped || false;
        comfyuiState.comfyuiInstalled = data.comfyui_installed || false;
        comfyuiState.apiRunning = data.api_running || false;
        comfyuiState.instances = data.instances || [];
        comfyuiState.gpus = data.gpus || [];

        renderOverview();
        renderInstances();
    } catch (e) {
        // Silent fail during polling
    }
}

// ======================== Subtab Switching ========================

export function switchComfyUISubtab(subtab) {
    comfyuiState.currentSubtab = subtab;

    // Update subtab buttons
    document.querySelectorAll('#tab-comfyui .comfyui-subtab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.subtab === subtab);
    });

    // Update subtab content
    document.querySelectorAll('#tab-comfyui .comfyui-subtab-content').forEach(panel => {
        const isActive = panel.id === `comfyui-subtab-${subtab}`;
        panel.classList.toggle('active', isActive);
        panel.classList.toggle('hidden', !isActive);
    });

    // Lazy load data for subtabs
    if (subtab === 'models' && comfyuiState.models.registry.length === 0 && comfyuiState.apiRunning) {
        loadModelsRegistry();
    }
    if (subtab === 'nodes' && comfyuiState.nodes.registry.length === 0 && comfyuiState.apiRunning) {
        loadNodesRegistry();
    }
    if (subtab === 'gallery') {
        // Auto-scan for new images then load gallery
        fetch(`${API}/comfyui/media/scan`, { method: 'POST' })
            .then(() => import('./gallery.js').then(m => m.initGallery()))
            .catch(() => import('./gallery.js').then(m => m.initGallery()));
    }
}

// ======================== Full Install (one-click) ========================

export async function fullInstall() {
    const toast = showToast('Step 1/4: Downloading module from GitHub...', 'loading', 0);
    debugLog('comfyui.fullInstall', 'starting');

    try {
        // Step 1: Download module
        if (!comfyuiState.moduleDownloaded) {
            const resp = await fetch(`${API}/comfyui/module/download`, { method: 'POST' });
            const data = await resp.json();
            if (!data.success) {
                updateToast(toast, 'Download failed: ' + (data.error || 'Unknown'), 'error');
                return;
            }
            await refreshComfyUIStatus();
        }

        // Step 2: Bootstrap
        updateToast(toast, 'Step 2/4: Bootstrapping Python, Git, FFmpeg (this takes a few minutes)...', 'loading');
        if (!comfyuiState.bootstrapped) {
            const resp = await fetch(`${API}/comfyui/module/bootstrap`, { method: 'POST' });
            const data = await resp.json();
            if (!data.success) {
                updateToast(toast, 'Bootstrap failed: ' + (data.error || 'Unknown'), 'error');
                return;
            }
            await refreshComfyUIStatus();
        }

        // Step 3: Start API server
        updateToast(toast, 'Step 3/4: Starting management API server...', 'loading');
        if (!comfyuiState.apiRunning) {
            const resp = await fetch(`${API}/comfyui/api/start`, { method: 'POST' });
            const data = await resp.json();
            if (!data.success) {
                updateToast(toast, 'API server failed: ' + (data.error || 'Unknown'), 'error');
                return;
            }
            await refreshComfyUIStatus();
        }

        // Step 4: Install ComfyUI (async job with polling)
        // Re-check API is actually running (don't trust stale state)
        await refreshComfyUIStatus();
        if (!comfyuiState.apiRunning) {
            updateToast(toast, 'API server not running after start. Try again.', 'error');
            return;
        }
        if (!comfyuiState.comfyuiInstalled) {
            updateToast(toast, 'Step 4/4: Installing ComfyUI + PyTorch (this may take several minutes)...', 'loading');
            const resp = await fetch(`${API}/comfyui/install`, { method: 'POST' });
            const data = await resp.json();

            if (data.job_id) {
                pollJob(data.job_id, (progress) => {
                    const msg = progress.message || `${progress.current}/${progress.total}`;
                    updateToast(toast, `Step 4/4: Installing ComfyUI: ${msg}`, 'loading');
                }, (result) => {
                    if (result.status === 'completed') {
                        updateToast(toast, 'ComfyUI fully installed and ready!', 'success');
                    } else {
                        updateToast(toast, 'Install failed: ' + (result.error || 'Unknown'), 'error');
                    }
                    refreshComfyUIStatus();
                });
                return; // pollJob handles completion
            } else if (!data.success && data.error) {
                updateToast(toast, 'Install failed: ' + data.error, 'error');
                return;
            }
        }

        updateToast(toast, 'ComfyUI fully installed and ready!', 'success');
        await refreshComfyUIStatus();
    } catch (e) {
        updateToast(toast, 'Setup error: ' + e.message, 'error');
    }
}

// ======================== Module Lifecycle ========================

export async function downloadModule() {
    const toast = showToast('Downloading ComfyUI module from GitHub...', 'loading', 0);
    debugLog('comfyui.downloadModule', 'starting');

    try {
        const resp = await fetch(`${API}/comfyui/module/download`, { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            updateToast(toast, 'Module downloaded', 'success');
            await refreshComfyUIStatus();
            // Auto-chain: continue to bootstrap
            if (!comfyuiState.bootstrapped) await bootstrapModule();
        } else {
            updateToast(toast, 'Download failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        updateToast(toast, 'Download error: ' + e.message, 'error');
    }
}

export async function bootstrapModule() {
    const toast = showToast('Bootstrapping: downloading Python, Git, FFmpeg (this takes a few minutes)...', 'loading', 0);
    debugLog('comfyui.bootstrap', 'starting');

    try {
        const resp = await fetch(`${API}/comfyui/module/bootstrap`, { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            updateToast(toast, 'Bootstrap completed', 'success');
            await refreshComfyUIStatus();
            // Auto-chain: start API server then install
            if (!comfyuiState.apiRunning) await startAPIServer();
        } else {
            updateToast(toast, 'Bootstrap failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        updateToast(toast, 'Bootstrap error: ' + e.message, 'error');
    }
}

export async function installComfyUI() {
    // Live check — don't trust stale polling state
    await refreshComfyUIStatus();
    if (!comfyuiState.apiRunning) {
        showToast('API server is not running. Start it first.', 'error');
        return;
    }

    const toast = showToast('Installing ComfyUI + PyTorch (this may take several minutes)...', 'loading', 0);
    debugLog('comfyui.install', 'starting');

    try {
        const resp = await fetch(`${API}/comfyui/install`, { method: 'POST' });
        const data = await resp.json();

        if (data.job_id) {
            pollJob(data.job_id, (progress) => {
                const msg = progress.message || `${progress.current}/${progress.total}`;
                updateToast(toast, `Installing ComfyUI: ${msg}`, 'loading');
            }, (result) => {
                if (result.status === 'completed') {
                    updateToast(toast, 'ComfyUI installed successfully', 'success');
                } else {
                    updateToast(toast, 'Install failed: ' + (result.error || 'Unknown'), 'error');
                }
                refreshComfyUIStatus();
            });
        } else if (data.success) {
            updateToast(toast, 'ComfyUI installed', 'success');
            await refreshComfyUIStatus();
        } else {
            updateToast(toast, 'Install failed: ' + (data.error || 'Unknown'), 'error');
        }
    } catch (e) {
        updateToast(toast, 'Install error: ' + e.message, 'error');
    }
}

export async function updateComfyUI() {
    if (!comfyuiState.apiRunning) {
        showToast('Start the API server first', 'error');
        return;
    }

    const toast = showToast('Updating ComfyUI...', 'loading', 0);

    try {
        const resp = await fetch(`${API}/comfyui/update`, { method: 'POST' });
        const data = await resp.json();

        if (data.job_id) {
            pollJob(data.job_id, null, (result) => {
                if (result.status === 'completed') {
                    updateToast(toast, 'ComfyUI updated', 'success');
                } else {
                    updateToast(toast, 'Update failed: ' + (result.error || 'Unknown'), 'error');
                }
            });
        } else if (data.success) {
            updateToast(toast, 'ComfyUI updated', 'success');
        } else {
            updateToast(toast, 'Update failed: ' + (data.error || 'Unknown'), 'error');
        }
    } catch (e) {
        updateToast(toast, 'Update error: ' + e.message, 'error');
    }
}

// ======================== API Server ========================

export async function startAPIServer() {
    const toast = showToast('Starting ComfyUI API server...', 'loading', 0);
    debugLog('comfyui.startAPI', 'starting');

    try {
        const resp = await fetch(`${API}/comfyui/api/start`, { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            updateToast(toast, data.message || 'API server started', 'success');
            await refreshComfyUIStatus();
            // Auto-chain: install ComfyUI if not yet installed
            if (!comfyuiState.comfyuiInstalled) await installComfyUI();
        } else {
            updateToast(toast, 'Failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

export async function stopAPIServer() {
    const toast = showToast('Stopping ComfyUI API server...', 'loading', 0);

    try {
        const resp = await fetch(`${API}/comfyui/api/stop`, { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            updateToast(toast, 'API server stopped', 'success');
            comfyuiState.apiRunning = false;
            comfyuiState.instances = [];
            renderOverview();
            renderInstances();
        } else {
            updateToast(toast, 'Failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

// ======================== Instances ========================

export async function addInstance() {
    if (!comfyuiState.apiRunning) {
        showToast('Start the API server first', 'error');
        return;
    }

    const gpu = document.getElementById('comfyui-add-gpu')?.value || '0';
    const port = parseInt(document.getElementById('comfyui-add-port')?.value || '8188');
    const vram = document.getElementById('comfyui-add-vram')?.value || 'normal';

    const gpuInfo = comfyuiState.gpus.find(g => String(g.index) === String(gpu));
    const gpuLabel = gpuInfo ? `GPU ${gpuInfo.index}: ${gpuInfo.name}` : `GPU ${gpu}`;

    try {
        const resp = await fetch(`${API}/comfyui/instances`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gpu_device: gpu === 'cpu' ? 'cpu' : gpu,
                gpu_label: gpu === 'cpu' ? 'CPU' : gpuLabel,
                port: port,
                vram_mode: vram,
            }),
        });
        const data = await resp.json();

        if (data.error) {
            showToast('Failed: ' + data.error, 'error');
        } else {
            showToast(`Instance added on port ${port}`, 'success');
            await refreshComfyUIStatus();
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

export async function removeInstance(instanceId) {
    if (!confirm(`Remove instance ${instanceId}?`)) return;

    try {
        const resp = await fetch(`${API}/comfyui/instances/${instanceId}`, { method: 'DELETE' });
        const data = await resp.json();

        if (data.error) {
            showToast('Failed: ' + data.error, 'error');
        } else {
            showToast('Instance removed', 'success');
            await refreshComfyUIStatus();
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

export async function startInstance(instanceId) {
    const toast = showToast(`Starting instance ${instanceId}...`, 'loading', 0);

    try {
        const resp = await fetch(`${API}/comfyui/instances/${instanceId}/start`, { method: 'POST' });
        const data = await resp.json();

        if (data.error) {
            updateToast(toast, 'Failed: ' + data.error, 'error');
        } else {
            updateToast(toast, `Instance ${instanceId} started`, 'success');
            await refreshComfyUIStatus();
        }
    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

export async function stopInstance(instanceId) {
    try {
        const resp = await fetch(`${API}/comfyui/instances/${instanceId}/stop`, { method: 'POST' });
        const data = await resp.json();

        if (data.error) {
            showToast('Failed: ' + data.error, 'error');
        } else {
            showToast(`Instance ${instanceId} stopped`, 'success');
            await refreshComfyUIStatus();
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

export async function startAllInstances() {
    const toast = showToast('Starting all instances...', 'loading', 0);

    try {
        const resp = await fetch(`${API}/comfyui/instances/start-all`, { method: 'POST' });
        const data = await resp.json();
        updateToast(toast, data.error ? 'Failed: ' + data.error : 'All instances started', data.error ? 'error' : 'success');
        await refreshComfyUIStatus();
    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

export async function stopAllInstances() {
    try {
        const resp = await fetch(`${API}/comfyui/instances/stop-all`, { method: 'POST' });
        showToast('All instances stopped', 'success');
        await refreshComfyUIStatus();
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

export function openComfyUIAdmin(port) {
    window.open(`http://127.0.0.1:${port}`, '_blank');
}

// ======================== Models ========================

async function loadModelsRegistry() {
    if (!comfyuiState.apiRunning) return;

    try {
        const [regResp, catResp] = await Promise.all([
            fetch(`${API}/comfyui/models/registry`),
            fetch(`${API}/comfyui/models/categories`),
        ]);

        const regData = await regResp.json();
        const catData = await catResp.json();

        comfyuiState.models.registry = Array.isArray(regData) ? regData : regData.models || [];
        comfyuiState.models.categories = Array.isArray(catData) ? catData : catData.categories || [];

        renderModels();
    } catch (e) {
        console.error('Failed to load models registry:', e);
    }
}

export async function downloadModel(modelId) {
    const toast = showToast('Starting model download...', 'loading', 0);

    try {
        const resp = await fetch(`${API}/comfyui/models/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_ids: [modelId] }),
        });
        const data = await resp.json();

        if (data.job_id) {
            pollJob(data.job_id, (progress) => {
                const msg = progress.message || `${progress.current}/${progress.total}`;
                updateToast(toast, `Downloading: ${msg}`, 'loading');
            }, (result) => {
                if (result.status === 'completed') {
                    updateToast(toast, 'Model downloaded', 'success');
                    loadModelsRegistry();
                } else {
                    updateToast(toast, 'Download failed: ' + (result.error || 'Unknown'), 'error');
                }
            });
        } else {
            updateToast(toast, data.error ? 'Failed: ' + data.error : 'Download started', data.error ? 'error' : 'success');
        }
    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

export async function filterModelsByCategory() {
    const category = document.getElementById('comfyui-model-category')?.value || '';

    try {
        const url = category
            ? `${API}/comfyui/models/registry?category=${encodeURIComponent(category)}`
            : `${API}/comfyui/models/registry`;
        const resp = await fetch(url);
        const data = await resp.json();
        comfyuiState.models.registry = Array.isArray(data) ? data : data.models || [];
        renderModels();
    } catch (e) {
        console.error('Failed to filter models:', e);
    }
}

export async function searchModels() {
    const query = document.getElementById('comfyui-model-search')?.value?.trim();
    if (!query) return;

    try {
        const resp = await fetch(`${API}/comfyui/models/search?q=${encodeURIComponent(query)}`);
        const data = await resp.json();
        comfyuiState.models.registry = Array.isArray(data) ? data : data.models || [];
        renderModels();
    } catch (e) {
        console.error('Failed to search models:', e);
    }
}

// ======================== Custom Nodes ========================

async function loadNodesRegistry() {
    if (!comfyuiState.apiRunning) return;

    try {
        const resp = await fetch(`${API}/comfyui/nodes/registry`);
        const data = await resp.json();
        comfyuiState.nodes.registry = Array.isArray(data) ? data : data.nodes || [];
        renderNodes();
    } catch (e) {
        console.error('Failed to load nodes registry:', e);
    }
}

export async function installNode(nodeId) {
    const toast = showToast('Installing custom node...', 'loading', 0);

    try {
        const resp = await fetch(`${API}/comfyui/nodes/install`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ node_ids: [nodeId] }),
        });
        const data = await resp.json();

        if (data.job_id) {
            pollJob(data.job_id, null, (result) => {
                if (result.status === 'completed') {
                    updateToast(toast, 'Custom node installed', 'success');
                    loadNodesRegistry();
                } else {
                    updateToast(toast, 'Install failed: ' + (result.error || 'Unknown'), 'error');
                }
            });
        } else {
            updateToast(toast, data.error ? 'Failed: ' + data.error : 'Node installed', data.error ? 'error' : 'success');
        }
    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

export async function removeNode(nodeName) {
    if (!confirm(`Remove custom node "${nodeName}"?`)) return;

    try {
        const resp = await fetch(`${API}/comfyui/nodes/${encodeURIComponent(nodeName)}`, { method: 'DELETE' });
        const data = await resp.json();
        showToast(data.error ? 'Failed: ' + data.error : 'Node removed', data.error ? 'error' : 'success');
        await loadNodesRegistry();
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

export async function updateAllNodes() {
    const toast = showToast('Updating all custom nodes...', 'loading', 0);

    try {
        const resp = await fetch(`${API}/comfyui/nodes/update-all`, { method: 'POST' });
        const data = await resp.json();

        if (data.job_id) {
            pollJob(data.job_id, null, (result) => {
                updateToast(toast, result.status === 'completed' ? 'All nodes updated' : 'Update failed', result.status === 'completed' ? 'success' : 'error');
                loadNodesRegistry();
            });
        } else {
            updateToast(toast, 'Update started', 'success');
        }
    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

// ======================== External ComfyUI ========================

export async function addExternalDir() {
    const input = document.getElementById('comfyui-external-path');
    const path = input?.value?.trim();
    if (!path) return;

    try {
        const resp = await fetch(`${API}/comfyui/external`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        const data = await resp.json();

        if (data.error) {
            showToast('Failed: ' + data.error, 'error');
        } else {
            showToast('External directory added', 'success');
            input.value = '';
            await loadExternalDirs();
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

export async function removeExternalDir(path) {
    try {
        const resp = await fetch(`${API}/comfyui/external`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        showToast('External directory removed', 'success');
        await loadExternalDirs();
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function loadExternalDirs() {
    if (!comfyuiState.apiRunning) return;

    try {
        const resp = await fetch(`${API}/comfyui/external`);
        const data = await resp.json();
        comfyuiState.externalDirs = Array.isArray(data) ? data : data.dirs || [];
        renderExternalDirs();
    } catch (e) {
        console.error('Failed to load external dirs:', e);
    }
}

// ======================== Job Polling ========================

function pollJob(jobId, onProgress, onComplete) {
    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`${API}/comfyui/jobs/${jobId}`);
            const job = await resp.json();

            if (onProgress && job.progress) {
                onProgress(job.progress);
            }

            if (job.status === 'completed' || job.status === 'failed') {
                clearInterval(interval);
                comfyuiState.activeJob = null;
                if (onComplete) onComplete(job);
            }
        } catch (e) {
            clearInterval(interval);
            comfyuiState.activeJob = null;
            if (onComplete) onComplete({ status: 'failed', error: e.message });
        }
    }, 2000);

    comfyuiState.activeJob = { jobId, interval };
}

// ======================== Rendering ========================

function renderOverview() {
    const container = document.getElementById('comfyui-overview');
    if (!container) return;

    const s = comfyuiState;
    const runningCount = s.instances.filter(i => i.running || i.status === 'running').length;

    // Status cards
    const statusGrid = container.querySelector('.comfyui-status-grid');
    if (statusGrid) {
        statusGrid.innerHTML = `
            <div class="comfyui-status-card">
                <span class="status-dot ${s.moduleDownloaded ? 'running' : 'stopped'}"></span>
                <span>Module</span>
                <span class="comfyui-status-label">${s.moduleDownloaded ? 'Downloaded' : 'Not Downloaded'}</span>
            </div>
            <div class="comfyui-status-card">
                <span class="status-dot ${s.bootstrapped ? 'running' : 'stopped'}"></span>
                <span>Bootstrap</span>
                <span class="comfyui-status-label">${s.bootstrapped ? 'Ready' : 'Not Done'}</span>
            </div>
            <div class="comfyui-status-card">
                <span class="status-dot ${s.comfyuiInstalled ? 'running' : 'stopped'}"></span>
                <span>ComfyUI</span>
                <span class="comfyui-status-label">${s.comfyuiInstalled ? 'Installed' : 'Not Installed'}</span>
            </div>
            <div class="comfyui-status-card">
                <span class="status-dot ${s.apiRunning ? 'running' : 'stopped'}"></span>
                <span>API Server</span>
                <span class="comfyui-status-label">${s.apiRunning ? 'Running' : 'Stopped'}</span>
            </div>
            <div class="comfyui-status-card">
                <span class="status-dot ${runningCount > 0 ? 'running' : 'stopped'}"></span>
                <span>Instances</span>
                <span class="comfyui-status-label">${runningCount} running</span>
            </div>
        `;
    }

    // Full install button
    const fullBtn = document.getElementById('comfyui-btn-full-install');
    if (fullBtn) {
        const allDone = s.comfyuiInstalled && s.apiRunning;
        fullBtn.disabled = allDone;
        fullBtn.textContent = allDone ? 'Fully Installed' : 'Set Up Everything';
    }

    // Install steps - enable/disable based on state
    const step1Btn = document.getElementById('comfyui-btn-download');
    const step2Btn = document.getElementById('comfyui-btn-bootstrap');
    const step3Btn = document.getElementById('comfyui-btn-install');

    if (step1Btn) {
        step1Btn.disabled = s.moduleDownloaded;
        step1Btn.textContent = s.moduleDownloaded ? 'Downloaded' : 'Download Module';
    }
    if (step2Btn) {
        step2Btn.disabled = !s.moduleDownloaded || s.bootstrapped;
        step2Btn.textContent = s.bootstrapped ? 'Bootstrapped' : 'Bootstrap';
    }
    if (step3Btn) {
        step3Btn.disabled = !s.apiRunning || s.comfyuiInstalled;
        step3Btn.textContent = s.comfyuiInstalled ? 'Installed' : 'Install ComfyUI';
    }

    // Quick actions
    const startApiBtn = document.getElementById('comfyui-btn-start-api');
    const stopApiBtn = document.getElementById('comfyui-btn-stop-api');
    const updateBtn = document.getElementById('comfyui-btn-update');

    if (startApiBtn) startApiBtn.disabled = !s.bootstrapped || s.apiRunning;
    if (stopApiBtn) stopApiBtn.disabled = !s.apiRunning;
    if (updateBtn) updateBtn.disabled = !s.apiRunning;

    // External ComfyUI section — only show when API is running
    const extSection = document.getElementById('comfyui-external-section');
    if (extSection) {
        extSection.classList.toggle('hidden', !s.apiRunning);
    }
}

function renderInstances() {
    const container = document.getElementById('comfyui-instances-list');
    if (!container) return;

    const instances = comfyuiState.instances;

    if (!instances || instances.length === 0) {
        container.innerHTML = '<div class="empty-state">No instances configured. Add one above.</div>';
        return;
    }

    container.innerHTML = instances.map(inst => {
        const id = inst.instance_id || inst.id || 'unknown';
        const isRunning = inst.running || inst.status === 'running';
        const gpu = inst.gpu_label || inst.gpu_device || 'Unknown GPU';
        const port = inst.port || '?';
        const vram = inst.vram_mode || 'normal';

        return `
            <div class="comfyui-instance-row ${isRunning ? 'running' : 'stopped'}">
                <div class="comfyui-instance-info">
                    <span class="status-dot ${isRunning ? 'running' : 'stopped'}"></span>
                    <span class="comfyui-instance-id">${id}</span>
                    <span class="comfyui-instance-detail">${gpu}</span>
                    <span class="comfyui-instance-detail">Port: ${port}</span>
                    <span class="comfyui-instance-detail">VRAM: ${vram}</span>
                    <span class="comfyui-instance-status">${isRunning ? 'Running' : 'Stopped'}</span>
                </div>
                <div class="comfyui-instance-actions">
                    ${isRunning
                        ? `<button class="btn-small" onclick="stopInstance('${id}')">Stop</button>
                           <button class="btn-small btn-accent" onclick="openComfyUIAdmin(${port})">Open Admin</button>`
                        : `<button class="btn-small btn-accent" onclick="startInstance('${id}')">Start</button>`
                    }
                    <button class="btn-small btn-danger" onclick="removeInstance('${id}')">&times;</button>
                </div>
            </div>
        `;
    }).join('');
}

function renderModels() {
    const container = document.getElementById('comfyui-models-list');
    if (!container) return;

    const models = comfyuiState.models.registry;

    // Populate category dropdown
    const catSelect = document.getElementById('comfyui-model-category');
    if (catSelect && catSelect.options.length <= 1) {
        comfyuiState.models.categories.forEach(cat => {
            const opt = document.createElement('option');
            opt.value = cat;
            opt.textContent = cat;
            catSelect.appendChild(opt);
        });
    }

    if (!models || models.length === 0) {
        container.innerHTML = '<div class="empty-state">No models found. Start the API server and switch to this tab.</div>';
        return;
    }

    container.innerHTML = models.map(m => {
        const name = m.name || m.filename || 'Unknown';
        const category = m.category || m.type || '';
        const size = m.size ? formatSize(m.size) : '';
        const filename = m.filename || '';
        const downloaded = m.downloaded || m.installed || false;
        const modelId = m.id || m.name || '';

        return `
            <div class="comfyui-model-row">
                <div class="comfyui-model-info">
                    <span class="comfyui-model-name">${escapeHtml(name)}</span>
                    <span class="comfyui-model-detail">${escapeHtml(category)}${size ? ' | ' + size : ''}</span>
                    ${filename ? `<span class="comfyui-model-filename">${escapeHtml(filename)}</span>` : ''}
                </div>
                <div class="comfyui-model-actions">
                    ${downloaded
                        ? '<span class="comfyui-badge success">Downloaded</span>'
                        : `<button class="btn-small btn-accent" onclick="downloadModel('${escapeHtml(modelId)}')">Download</button>`
                    }
                </div>
            </div>
        `;
    }).join('');
}

function renderNodes() {
    const container = document.getElementById('comfyui-nodes-list');
    if (!container) return;

    const nodes = comfyuiState.nodes.registry;

    if (!nodes || nodes.length === 0) {
        container.innerHTML = '<div class="empty-state">No custom nodes found. Start the API server and switch to this tab.</div>';
        return;
    }

    container.innerHTML = nodes.map(n => {
        const name = n.name || 'Unknown';
        const desc = n.description || '';
        const installed = n.installed || false;
        const nodeId = n.id || n.name || '';

        return `
            <div class="comfyui-node-row">
                <div class="comfyui-node-info">
                    <span class="comfyui-node-name">${escapeHtml(name)}</span>
                    ${desc ? `<span class="comfyui-node-desc">${escapeHtml(desc)}</span>` : ''}
                </div>
                <div class="comfyui-node-actions">
                    ${installed
                        ? `<span class="comfyui-badge success">Installed</span>
                           <button class="btn-small" onclick="removeNode('${escapeHtml(name)}')">Remove</button>`
                        : `<button class="btn-small btn-accent" onclick="installNode('${escapeHtml(nodeId)}')">Install</button>`
                    }
                </div>
            </div>
        `;
    }).join('');
}

function renderExternalDirs() {
    const container = document.getElementById('comfyui-external-list');
    if (!container) return;

    const dirs = comfyuiState.externalDirs;

    if (!dirs || dirs.length === 0) {
        container.innerHTML = '<div class="empty-state-small">No external directories saved</div>';
        return;
    }

    container.innerHTML = dirs.map(d => {
        const path = typeof d === 'string' ? d : d.path || d;
        return `
            <div class="comfyui-external-row">
                <span>${escapeHtml(path)}</span>
                <button class="btn-small btn-danger" onclick="removeExternalDir('${escapeHtml(path)}')">&times;</button>
            </div>
        `;
    }).join('');
}

// ======================== Helpers ========================

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function formatSize(bytes) {
    if (!bytes) return '';
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return gb.toFixed(1) + ' GB';
    const mb = bytes / (1024 * 1024);
    return mb.toFixed(0) + ' MB';
}
