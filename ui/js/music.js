/**
 * Music Module — Frontend controller
 *
 * Manages the Music tab with 5 subtabs: Overview, Models, Workers, Generate, Library.
 * Follows the same pattern as tts.js.
 */

import { musicState, API } from './state.js';
import { log, showToast, updateToast } from './utils.js';

let pollingInterval = null;

// ======================== Init / Polling ========================

export function initMusic() {
    if (!musicState.initialized) {
        musicState.initialized = true;
    }
    refreshMusicStatus();
    startMusicPolling();
}

export function startMusicPolling() {
    if (pollingInterval) return;
    pollingInterval = setInterval(refreshMusicStatus, 5000);
}

export function stopMusicPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

export async function refreshMusicStatus() {
    try {
        const resp = await fetch(`${API}/music/status`);
        if (!resp.ok) return;
        const data = await resp.json();

        musicState.moduleDownloaded = data.module_downloaded || false;
        musicState.bootstrapped = data.bootstrapped || false;
        musicState.installed = data.installed || false;
        musicState.apiRunning = data.api_running || false;
        musicState.workers = data.workers || [];
        musicState.models = data.models || [];
        musicState.devices = data.devices || [];

        // Update server badge on Workers tab
        const badge = document.getElementById('music-server-badge');
        if (badge) {
            if (musicState.apiRunning) {
                badge.textContent = 'Gateway Running';
                badge.className = 'tts-setup-badge ready';
            } else {
                badge.textContent = 'Gateway Stopped';
                badge.className = 'tts-setup-badge pending';
            }
        }

        renderOverview();

        if (musicState.currentSubtab === 'workers') renderWorkers();
    } catch (e) {
        // Server not reachable
    }
}

// ======================== Subtab Switching ========================

export function switchMusicSubtab(subtab) {
    musicState.currentSubtab = subtab;

    document.querySelectorAll('#tab-music .tts-subtab').forEach(b => {
        b.classList.toggle('active', b.dataset.subtab === subtab);
    });
    document.querySelectorAll('#tab-music .tts-subtab-content').forEach(p => {
        const id = p.id.replace('music-subtab-', '');
        if (id === subtab) {
            p.classList.remove('hidden');
            p.classList.add('active');
        } else {
            p.classList.add('hidden');
            p.classList.remove('active');
        }
    });

    if (subtab === 'models') refreshModelsSubtab();
    if (subtab === 'workers') renderWorkers();
    if (subtab === 'generate') initGenerateSubtab();
    if (subtab === 'library') refreshMusicLibrary();
}

// ======================== Overview Subtab ========================

function renderOverview() {
    // Update status cards
    const cards = [
        { id: 'music-status-downloaded', ok: musicState.moduleDownloaded, text: musicState.moduleDownloaded ? 'Downloaded' : 'Not Downloaded' },
        { id: 'music-status-bootstrapped', ok: musicState.bootstrapped, text: musicState.bootstrapped ? 'Ready' : 'Not Done' },
        { id: 'music-status-installed', ok: musicState.installed, text: musicState.installed ? 'Installed' : 'Not Installed' },
        { id: 'music-status-server', ok: musicState.apiRunning, text: musicState.apiRunning ? `Running (:${9150})` : 'Stopped' },
    ];

    for (const c of cards) {
        const el = document.getElementById(c.id);
        if (!el) continue;
        el.textContent = c.text;
        const card = el.closest('.tts-status-card');
        if (card) {
            card.classList.toggle('ok', c.ok);
            card.classList.toggle('off', !c.ok);
        }
    }

    // Update button states
    const btnDownload = document.getElementById('music-download-btn');
    const btnBootstrap = document.getElementById('music-bootstrap-btn');
    const btnStart = document.getElementById('music-start-btn');
    const btnStop = document.getElementById('music-stop-btn');
    const btnFull = document.getElementById('music-full-install-btn');

    if (btnDownload) {
        btnDownload.disabled = musicState.moduleDownloaded;
        btnDownload.textContent = musicState.moduleDownloaded ? 'Downloaded' : 'Download';
    }
    if (btnBootstrap) {
        btnBootstrap.disabled = !musicState.moduleDownloaded || musicState.installed;
        btnBootstrap.textContent = musicState.installed ? 'Installed' : 'Bootstrap';
    }
    const btnUpdate = document.getElementById('music-update-btn');

    if (btnStart) btnStart.disabled = !musicState.installed || musicState.apiRunning;
    if (btnStop) btnStop.disabled = !musicState.apiRunning;
    if (btnUpdate) btnUpdate.disabled = !musicState.moduleDownloaded;
    if (btnFull) {
        btnFull.disabled = musicState.apiRunning;
        if (musicState.apiRunning) {
            btnFull.textContent = 'Music Server Running';
        } else if (musicState.installed) {
            btnFull.textContent = 'Start Server';
        } else {
            btnFull.textContent = 'Set Up Everything';
        }
    }
}

// ======================== Installation ========================

export async function musicFullInstall() {
    const toast = showToast('Starting Music setup...', 'loading', 0);

    try {
        if (!musicState.moduleDownloaded) {
            updateToast(toast, 'Step 1/3: Cloning Music server from GitHub...', 'loading');
            const dlResp = await fetch(`${API}/music/module/download`, { method: 'POST' });
            const dl = await dlResp.json();
            if (!dl.success) { updateToast(toast, `Download failed: ${dl.error}`, 'error'); return; }
            await refreshMusicStatus();
        }

        if (!musicState.installed) {
            updateToast(toast, 'Step 2/3: Bootstrapping (Python, FFmpeg, requirements)... This may take 10-20 minutes.', 'loading');
            const bsResp = await fetch(`${API}/music/module/bootstrap`, { method: 'POST' });
            const bs = await bsResp.json();
            if (!bs.success) { updateToast(toast, `Bootstrap failed: ${bs.error}`, 'error'); return; }
            await refreshMusicStatus();
        }

        if (!musicState.apiRunning) {
            updateToast(toast, 'Step 3/3: Starting Music API server...', 'loading');
            const startResp = await fetch(`${API}/music/server/start`, { method: 'POST' });
            const start = await startResp.json();
            if (!start.success) { updateToast(toast, `Server start failed: ${start.error}`, 'error'); return; }
            await refreshMusicStatus();
        }

        updateToast(toast, 'Music server is ready!', 'success');
        log('Music setup complete', 'success');
    } catch (e) {
        updateToast(toast, `Setup failed: ${e.message}`, 'error');
        log(`Music setup error: ${e.message}`, 'error');
    }
}

export async function downloadMusicModule() {
    const toast = showToast('Cloning Music server from GitHub...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/music/module/download`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'Music module downloaded!', 'success');
            await refreshMusicStatus();
        } else {
            updateToast(toast, `Download failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Download error: ${e.message}`, 'error');
    }
}

export async function bootstrapMusicModule() {
    const toast = showToast('Bootstrapping Music module (this may take 10-20 minutes)...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/music/module/bootstrap`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'Bootstrap complete!', 'success');
            await refreshMusicStatus();
        } else {
            updateToast(toast, `Bootstrap failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Bootstrap error: ${e.message}`, 'error');
    }
}

export async function startMusicServer() {
    const toast = showToast('Starting Music server...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/music/server/start`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'Music server started!', 'success');
            await refreshMusicStatus();
        } else {
            updateToast(toast, `Start failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Start error: ${e.message}`, 'error');
    }
}

export async function stopMusicServer() {
    const toast = showToast('Stopping Music server...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/music/server/stop`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'Music server stopped', 'success');
            await refreshMusicStatus();
        } else {
            updateToast(toast, `Stop failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Stop error: ${e.message}`, 'error');
    }
}

export async function updateMusicModule() {
    const toast = showToast('Updating Music module...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/music/module/update`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, `Updated: ${data.message}`, 'success');
        } else {
            updateToast(toast, `Update failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Update error: ${e.message}`, 'error');
    }
}

// ======================== Models Subtab ========================

export async function refreshModelsSubtab() {
    const grid = document.getElementById('music-models-grid');
    if (!grid) return;

    if (!musicState.apiRunning) {
        grid.innerHTML = '<div class="empty-state">Music server not running. Start it from the Overview tab.</div>';
        return;
    }

    // Fetch install status from running server
    try {
        const resp = await fetch(`${API}/music/install/status`);
        if (!resp.ok) {
            grid.innerHTML = '<div class="empty-state">Could not fetch model status. Server may be restarting...</div>';
            return;
        }
        const data = await resp.json();
        const models = data.models || data;

        // Check for active install jobs
        const jobChecks = {};
        for (const [id, info] of Object.entries(models)) {
            if (info.active_job_id) {
                jobChecks[id] = info.active_job_id;
            }
        }

        // Fetch active job progress in parallel
        const jobProgress = {};
        const jobEntries = Object.entries(jobChecks);
        if (jobEntries.length > 0) {
            const results = await Promise.allSettled(
                jobEntries.map(([id, jobId]) =>
                    fetch(`${API}/music/install/jobs/${jobId}`).then(r => r.json()).then(j => ({ id, job: j }))
                )
            );
            for (const r of results) {
                if (r.status === 'fulfilled') {
                    jobProgress[r.value.id] = r.value.job;
                }
            }
        }

        grid.innerHTML = Object.entries(models).map(([id, info]) => {
            const displayName = info.display_name || id;
            const envOk = info.env_installed;
            const weightsOk = info.weights_installed;
            const isComplete = info.status === 'complete';
            const isWeightsOnly = info.status === 'weights_only';
            const activeJob = jobProgress[id];

            // Status badges for env and weights
            const envBadge = envOk
                ? '<span class="tts-setup-badge ready" style="font-size:10px">Env Ready</span>'
                : '<span class="tts-setup-badge needs-download" style="font-size:10px">Needs Env</span>';
            const weightsBadge = weightsOk
                ? '<span class="tts-setup-badge ready" style="font-size:10px">Weights OK</span>'
                : '<span class="tts-setup-badge needs-download" style="font-size:10px">Needs Weights</span>';

            // Overall status badge
            let mainBadge;
            if (isComplete) {
                mainBadge = '<span class="tts-setup-badge ready">Ready</span>';
            } else if (isWeightsOnly) {
                mainBadge = '<span class="tts-setup-badge pending">Needs Env</span>';
            } else {
                mainBadge = '<span class="tts-setup-badge needs-download">Not Installed</span>';
            }

            // Progress bar for active jobs
            let progressHtml = '';
            if (activeJob && activeJob.status === 'running') {
                const pct = Math.round(activeJob.progress_pct || 0);
                const step = activeJob.current_step || 'Installing...';
                const stepInfo = activeJob.step_index && activeJob.step_total
                    ? ` (${activeJob.step_index}/${activeJob.step_total})`
                    : '';
                progressHtml = `
                    <div style="margin-top:8px">
                        <div style="font-size:11px;color:var(--accent);margin-bottom:4px">${step}${stepInfo}</div>
                        <div style="background:var(--bg-tertiary);border-radius:4px;height:6px;overflow:hidden">
                            <div style="background:var(--accent);height:100%;width:${pct}%;transition:width 0.3s"></div>
                        </div>
                        <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">${pct}%</div>
                    </div>
                `;
            } else if (activeJob && activeJob.status === 'failed') {
                progressHtml = `<div style="margin-top:6px;font-size:11px;color:var(--error)">${activeJob.error || 'Install failed'}</div>`;
            }

            // Actions
            let actions = '';
            if (activeJob && activeJob.status === 'running') {
                // Job in progress — no buttons
            } else if (isComplete) {
                actions = `<button class="btn-danger btn-small" onclick="uninstallMusicModel('${id}')">Uninstall</button>`;
            } else if (isWeightsOnly) {
                actions = `<button class="btn-accent btn-small" onclick="installMusicModel('${id}')">Install Env</button>`;
            } else {
                actions = `<button class="btn-accent btn-small" onclick="installMusicModel('${id}')">Install</button>`;
            }

            return `
                <div class="tts-model-card ${isComplete ? 'loaded' : 'unloaded'}" style="padding:12px">
                    <div class="tts-model-card-header">
                        <span class="tts-model-name">${displayName}</span>
                        ${mainBadge}
                    </div>
                    <div style="display:flex;gap:6px;margin-top:6px">
                        ${envBadge} ${weightsBadge}
                    </div>
                    ${progressHtml}
                    <div class="tts-model-card-actions" style="margin-top:8px">
                        ${actions}
                    </div>
                </div>
            `;
        }).join('');

        // Auto-refresh if any jobs are active
        if (Object.keys(jobProgress).some(id => jobProgress[id]?.status === 'running')) {
            if (!musicState._modelsRefreshTimer) {
                musicState._modelsRefreshTimer = setTimeout(() => {
                    musicState._modelsRefreshTimer = null;
                    if (musicState.currentSubtab === 'models') refreshModelsSubtab();
                }, 3000);
            }
        }
    } catch (e) {
        grid.innerHTML = `<div class="empty-state">Failed to load install status: ${e.message}</div>`;
    }
}

export async function installMusicModel(modelId) {
    const toast = showToast(`Installing ${modelId}... This may take a while.`, 'loading', 0);
    try {
        const resp = await fetch(`${API}/music/install/${modelId}`, { method: 'POST' });
        const data = await resp.json();
        if (data.job_id) {
            updateToast(toast, `Install started (job: ${data.job_id.substring(0, 8)})`, 'loading');
            // Poll job status
            await pollInstallJob(data.job_id, toast, modelId);
        } else if (data.success === false) {
            updateToast(toast, `Install failed: ${data.error || data.detail}`, 'error');
        } else {
            updateToast(toast, `${modelId} installed!`, 'success');
            await refreshModelsSubtab();
        }
    } catch (e) {
        updateToast(toast, `Install error: ${e.message}`, 'error');
    }
}

async function pollInstallJob(jobId, toast, modelId) {
    const maxPolls = 360; // 30 minutes at 5s intervals
    for (let i = 0; i < maxPolls; i++) {
        await new Promise(r => setTimeout(r, 5000));
        try {
            const resp = await fetch(`${API}/music/install/jobs/${jobId}`);
            const job = await resp.json();
            const pct = job.progress?.percent || 0;
            const msg = job.progress?.message || job.status;

            if (job.status === 'completed') {
                updateToast(toast, `${modelId} installed successfully!`, 'success');
                await refreshModelsSubtab();
                return;
            } else if (job.status === 'failed') {
                updateToast(toast, `Install failed: ${job.error || 'Unknown error'}`, 'error');
                return;
            } else if (job.status === 'cancelled') {
                updateToast(toast, `Install cancelled`, 'error');
                return;
            }

            updateToast(toast, `Installing ${modelId}: ${msg} (${pct}%)`, 'loading');
        } catch (e) {
            // Keep polling on transient errors
        }
    }
    updateToast(toast, `Install timed out after 30 minutes`, 'error');
}

export async function uninstallMusicModel(modelId) {
    if (!confirm(`Uninstall ${modelId}? This will remove its environment and weights.`)) return;

    const toast = showToast(`Uninstalling ${modelId}...`, 'loading', 0);
    try {
        const resp = await fetch(`${API}/music/install/${modelId}`, { method: 'DELETE' });
        const data = await resp.json();
        updateToast(toast, `${modelId} uninstalled`, 'success');
        await refreshModelsSubtab();
    } catch (e) {
        updateToast(toast, `Uninstall error: ${e.message}`, 'error');
    }
}

// ======================== Workers Subtab ========================

function populateSpawnDropdown() {
    const modelSelect = document.getElementById('music-spawn-model');
    if (!modelSelect) return;
    let options = '';
    if (musicState.models.length > 0) {
        options = musicState.models.map(m => {
            const id = m.id || m.name || m;
            const name = m.name || m.id || m;
            return `<option value="${id}">${name}</option>`;
        }).join('');
    }
    if (options) {
        modelSelect.innerHTML = '<option value="">Select model...</option>' + options;
    }
}

function renderWorkers() {
    const list = document.getElementById('music-workers-list');
    if (!list) return;

    // Populate spawn dropdown
    populateSpawnDropdown();

    if (!musicState.apiRunning) {
        list.innerHTML = '<div class="empty-state">Music server not running. Start it from the Overview tab.</div>';
        return;
    }

    if (musicState.workers.length === 0) {
        list.innerHTML = '<div class="empty-state">No workers running. Spawn one above or generate music to auto-spawn.</div>';
        return;
    }

    list.innerHTML = musicState.workers.map(w => `
        <div class="tts-worker-row">
            <div class="tts-worker-info">
                <span class="tts-worker-model">${w.model || 'unknown'}</span>
                <span class="tts-worker-detail">Port ${w.port || '?'} | ${w.device || 'cpu'}</span>
            </div>
            <div class="tts-worker-status">
                <span class="status-dot ${w.status === 'ready' ? 'online' : w.status === 'busy' ? 'loading' : 'offline'}"></span>
                <span>${w.status || 'unknown'}</span>
            </div>
            <div class="tts-worker-vram">
                ${w.vram_used_mb ? `${Math.round(w.vram_used_mb)}/${Math.round(w.vram_total_mb || 0)} MB` : '-'}
            </div>
            <div class="tts-worker-actions">
                <button class="btn-danger btn-small" onclick="killMusicWorker('${w.worker_id}')">Kill</button>
            </div>
        </div>
    `).join('');
}

export async function spawnMusicWorker() {
    const model = document.getElementById('music-spawn-model')?.value;
    const device = document.getElementById('music-spawn-device')?.value || 'cuda:1';

    if (!model) {
        showToast('Select a model first', 'error');
        return;
    }

    const toast = showToast(`Spawning ${model} worker on ${device}...`, 'loading', 0);
    try {
        // Auto-start gateway if not running
        if (!musicState.apiRunning) {
            updateToast(toast, 'Starting Music gateway server...', 'loading');
            const startResp = await fetch(`${API}/music/server/start`, { method: 'POST' });
            const startData = await startResp.json();
            if (!startData.success) {
                updateToast(toast, `Gateway start failed: ${startData.error}`, 'error');
                return;
            }
            await refreshMusicStatus();
            updateToast(toast, `Spawning ${model} worker on ${device}...`, 'loading');
        }

        const resp = await fetch(`${API}/music/workers/spawn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, device }),
        });
        const data = await resp.json();
        if (data.worker_id || data.status === 'spawned') {
            updateToast(toast, `Worker spawned: ${model} on ${device}`, 'success');
            await refreshMusicStatus();
        } else {
            updateToast(toast, `Spawn failed: ${data.detail || JSON.stringify(data)}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Spawn error: ${e.message}`, 'error');
    }
}

export async function killMusicWorker(workerId) {
    const toast = showToast('Killing worker...', 'loading', 0);
    try {
        await fetch(`${API}/music/workers/${workerId}`, { method: 'DELETE' });
        updateToast(toast, 'Worker killed', 'success');
        await refreshMusicStatus();
    } catch (e) {
        updateToast(toast, `Kill error: ${e.message}`, 'error');
    }
}

// ======================== Generate Subtab ========================

async function initGenerateSubtab() {
    const modelSelect = document.getElementById('music-gen-model');
    if (!modelSelect) return;

    // Populate model dropdown
    if (musicState.apiRunning && musicState.models.length > 0) {
        const currentVal = modelSelect.value;
        modelSelect.innerHTML = '<option value="">Select model...</option>' +
            musicState.models.map(m => {
                const id = m.id || m.name || m;
                const name = m.name || m.id || m;
                return `<option value="${id}">${name}</option>`;
            }).join('');
        if (currentVal) modelSelect.value = currentVal;
    }

    // Populate device dropdown
    const deviceSelect = document.getElementById('music-gen-device');
    if (deviceSelect && musicState.devices.length > 0) {
        const currentDevice = deviceSelect.value;
        deviceSelect.innerHTML = '<option value="">Auto</option>' +
            musicState.devices.map(d => {
                const id = d.id || d;
                const name = d.name || d;
                return `<option value="${id}">${name}</option>`;
            }).join('');
        if (currentDevice) deviceSelect.value = currentDevice;
    }

    // If model already selected, trigger param load
    if (modelSelect.value) {
        await onMusicModelSelect();
    }
}

export async function onMusicModelSelect() {
    const model = document.getElementById('music-gen-model')?.value;
    if (!model) return;

    // Clear dynamic params
    const paramsContainer = document.getElementById('music-gen-model-params');
    if (paramsContainer) paramsContainer.innerHTML = '<div style="color:var(--text-secondary);font-size:12px;">Loading parameters...</div>';

    // Fetch model params from API
    try {
        const resp = await fetch(`${API}/music/models/${model}/params`);
        const data = await resp.json();
        const params = data.params || data.controls || data;
        musicState.modelParams[model] = Array.isArray(params) ? params : [];
        renderModelParams(model);
    } catch (e) {
        if (paramsContainer) paramsContainer.innerHTML = '';
    }

    // Fetch presets
    try {
        const resp = await fetch(`${API}/music/models/${model}/presets`);
        const data = await resp.json();
        const presets = data.presets || [];
        musicState.modelPresets[model] = presets;
        populatePresetDropdown(model);
    } catch (e) {
        // No presets available
    }

    // Show/hide lyrics and reference audio based on model capabilities
    updateModelCapabilities(model);
}

function renderModelParams(model) {
    const container = document.getElementById('music-gen-model-params');
    if (!container) return;

    const params = musicState.modelParams[model] || [];
    if (params.length === 0) {
        container.innerHTML = '';
        return;
    }

    let html = '<div class="tts-gen-row tts-gen-options" style="flex-wrap:wrap">';

    for (const p of params) {
        if (p.type === 'heading') {
            html += `</div><div class="tts-gen-params-header" style="margin-top:8px"><span class="tts-gen-params-label">${p.label || ''}</span></div><div class="tts-gen-row tts-gen-options" style="flex-wrap:wrap">`;
            continue;
        }

        // Skip params already in the main form (prompt, lyrics, duration, seed, etc.)
        const skipIds = ['prompt', 'lyrics', 'tags', 'duration', 'seed', 'output_format', 'reference_audio', 'src_audio'];
        if (skipIds.includes(p.id)) continue;

        const fieldId = `music-gen-p-${p.id}`;

        html += '<div class="tts-gen-field">';
        html += `<label>${p.label || p.id}</label>`;

        if (p.type === 'combo') {
            html += `<select id="${fieldId}" class="input-field">`;
            const options = p.options || [];
            for (const opt of options) {
                const val = typeof opt === 'string' ? opt : opt.value || opt.v || opt;
                const label = typeof opt === 'string' ? opt : opt.label || opt.l || val;
                const sel = String(val) === String(p.default) ? ' selected' : '';
                html += `<option value="${val}"${sel}>${label}</option>`;
            }
            html += '</select>';
        } else if (p.type === 'check') {
            const checked = p.default ? ' checked' : '';
            html += `<input type="checkbox" id="${fieldId}"${checked}>`;
        } else if (p.type === 'text') {
            const rows = p.rows || 2;
            html += `<textarea id="${fieldId}" class="input-field" rows="${rows}">${p.default || ''}</textarea>`;
        } else if (p.type === 'spin' || p.type === 'entry') {
            const min = p.from_ !== undefined ? p.from_ : (p.min !== undefined ? p.min : '');
            const max = p.to !== undefined ? p.to : (p.max !== undefined ? p.max : '');
            const step = p.increment || p.step || 1;
            const val = p.default !== undefined ? p.default : '';
            html += `<input type="number" id="${fieldId}" class="input-field" value="${val}" min="${min}" max="${max}" step="${step}">`;
        } else if (p.type === 'file') {
            html += `<input type="file" id="${fieldId}" class="input-field" accept="audio/*">`;
        } else {
            // Fallback: text input
            html += `<input type="text" id="${fieldId}" class="input-field" value="${p.default || ''}">`;
        }

        html += '</div>';
    }

    html += '</div>';
    container.innerHTML = html;
}

function populatePresetDropdown(model) {
    const presetSelect = document.getElementById('music-gen-preset');
    if (!presetSelect) return;

    const presets = musicState.modelPresets[model] || [];
    presetSelect.innerHTML = '<option value="">Default</option>' +
        presets.map((p, i) => `<option value="${i}">${p.name}</option>`).join('');
}

export function onMusicPresetSelect() {
    const model = document.getElementById('music-gen-model')?.value;
    const presetIdx = document.getElementById('music-gen-preset')?.value;
    if (!model || presetIdx === '' || presetIdx === undefined) return;

    const presets = musicState.modelPresets[model] || [];
    const preset = presets[parseInt(presetIdx)];
    if (!preset || !preset.params) return;

    // Apply preset values to form fields
    for (const [key, val] of Object.entries(preset.params)) {
        // Check main form fields first
        const mainFields = {
            'prompt': 'music-gen-prompt',
            'caption': 'music-gen-prompt',
            'lyrics': 'music-gen-lyrics',
            'duration': 'music-gen-duration',
            'seed': 'music-gen-seed',
        };

        const mainId = mainFields[key];
        if (mainId) {
            const el = document.getElementById(mainId);
            if (el) el.value = val;
            continue;
        }

        // Check dynamic param fields
        const el = document.getElementById(`music-gen-p-${key}`);
        if (el) {
            if (el.type === 'checkbox') {
                el.checked = val === true || val === 'true' || val === 1;
            } else {
                el.value = val;
            }
        }
    }

    showToast(`Applied preset: ${preset.name}`, 'success');
}

/** Show/hide lyrics and ref audio sections based on model. */
function updateModelCapabilities(model) {
    const lyricsModels = ['ace_step_v15', 'ace_step_v1', 'heartmula', 'diffrhythm', 'yue'];
    const refAudioModels = ['ace_step_v15', 'ace_step_v1', 'musicgen'];

    const lyricsWrap = document.getElementById('music-gen-lyrics-wrap');
    const refAudioWrap = document.getElementById('music-gen-ref-audio-wrap');

    if (lyricsWrap) {
        if (lyricsModels.includes(model)) {
            lyricsWrap.classList.remove('hidden');
        } else {
            lyricsWrap.classList.add('hidden');
        }
    }

    if (refAudioWrap) {
        if (refAudioModels.includes(model)) {
            refAudioWrap.classList.remove('hidden');
        } else {
            refAudioWrap.classList.add('hidden');
        }
    }
}

export async function generateMusic() {
    const model = document.getElementById('music-gen-model')?.value;
    if (!model) { showToast('Select a model first', 'error'); return; }

    const prompt = document.getElementById('music-gen-prompt')?.value?.trim();
    if (!prompt) { showToast('Enter a prompt/description', 'error'); return; }

    const format = document.getElementById('music-gen-format')?.value || 'wav';
    const duration = parseFloat(document.getElementById('music-gen-duration')?.value) || 30;
    const seed = parseInt(document.getElementById('music-gen-seed')?.value);
    const denoise = parseFloat(document.getElementById('music-gen-denoise')?.value);
    const eq = document.getElementById('music-gen-eq')?.value || 'balanced';
    const lufs = parseFloat(document.getElementById('music-gen-lufs')?.value);
    const stereo = parseFloat(document.getElementById('music-gen-stereo')?.value);
    const compression = parseFloat(document.getElementById('music-gen-compression')?.value);
    const peak = parseFloat(document.getElementById('music-gen-peak')?.value);
    const device = document.getElementById('music-gen-device')?.value || undefined;
    const skipPost = document.getElementById('music-gen-skip-post')?.checked || false;

    const body = {
        prompt,
        duration,
        output_format: format,
        seed: isNaN(seed) ? -1 : seed,
        denoise_strength: isNaN(denoise) ? 0.2 : denoise,
        eq_preset: eq,
        target_lufs: isNaN(lufs) ? -14 : lufs,
        stereo_width: isNaN(stereo) ? 1.2 : stereo,
        compression_ratio: isNaN(compression) ? 2.0 : compression,
        clipping: isNaN(peak) ? 0.95 : peak,
        skip_post_process: skipPost,
    };
    if (device) body.device = device;

    // Lyrics
    const lyrics = document.getElementById('music-gen-lyrics')?.value?.trim();
    if (lyrics) body.lyrics = lyrics;

    // Reference audio
    const refInput = document.getElementById('music-gen-ref-audio');
    if (refInput && refInput.files.length > 0) {
        try {
            const file = refInput.files[0];
            const arrayBuf = await file.arrayBuffer();
            body.reference_audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuf)));
        } catch (e) {
            showToast('Failed to read reference audio', 'error');
            return;
        }
    }

    // Collect dynamic model params
    const modelParams = {};
    const params = musicState.modelParams[model] || [];
    for (const p of params) {
        if (p.type === 'heading') continue;
        const skipIds = ['prompt', 'lyrics', 'tags', 'duration', 'seed', 'output_format', 'reference_audio', 'src_audio'];
        if (skipIds.includes(p.id)) continue;

        const el = document.getElementById(`music-gen-p-${p.id}`);
        if (!el) continue;

        if (p.type === 'check') {
            modelParams[p.id] = el.checked;
        } else if (p.type === 'combo' || p.type === 'text') {
            if (el.value) modelParams[p.id] = el.value;
        } else if (p.type === 'file') {
            if (el.files.length > 0) {
                try {
                    const file = el.files[0];
                    const buf = await file.arrayBuffer();
                    modelParams[p.id] = btoa(String.fromCharCode(...new Uint8Array(buf)));
                } catch (e) { /* skip */ }
            }
        } else {
            const val = el.value;
            if (val !== '' && val !== undefined) {
                modelParams[p.id] = parseFloat(val);
                if (isNaN(modelParams[p.id])) modelParams[p.id] = val;
            }
        }
    }

    // Merge model_params into body (music server may accept them flat or nested)
    Object.assign(body, modelParams);

    // UI state
    const resultDiv = document.getElementById('music-gen-result');
    const infoDiv = document.getElementById('music-gen-result-info');
    const audioEl = document.getElementById('music-gen-audio');
    const btn = document.getElementById('music-gen-btn');

    if (resultDiv) resultDiv.classList.add('hidden');
    if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }

    musicState.activeGeneration = model;

    try {
        // Auto-start gateway if not running
        if (!musicState.apiRunning) {
            if (btn) btn.textContent = 'Starting server...';
            const startResp = await fetch(`${API}/music/server/start`, { method: 'POST' });
            const startData = await startResp.json();
            if (!startData.success) {
                showToast(`Server start failed: ${startData.error}`, 'error');
                return;
            }
            await refreshMusicStatus();
            if (btn) btn.textContent = 'Generating...';
        }

        const resp = await fetch(`${API}/music/generate/${model}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await resp.json();

        if (data.status === 'completed' && data.audio_base64) {
            const audioBytes = atob(data.audio_base64);
            const audioArray = new Uint8Array(audioBytes.length);
            for (let i = 0; i < audioBytes.length; i++) {
                audioArray[i] = audioBytes.charCodeAt(i);
            }

            const mimeType = { wav: 'audio/wav', mp3: 'audio/mpeg', ogg: 'audio/ogg', flac: 'audio/flac', m4a: 'audio/mp4' }[format] || 'audio/wav';
            const blob = new Blob([audioArray], { type: mimeType });
            const url = URL.createObjectURL(blob);

            if (infoDiv) {
                infoDiv.innerHTML = `Generated! Duration: ${data.duration_sec?.toFixed(1) || '?'}s | ${data.sample_rate || '?'} Hz | Inference: ${data.inference_time_sec?.toFixed(1) || '?'}s`;
            }
            if (audioEl) {
                audioEl.src = url;
                audioEl.type = mimeType;
            }
            if (resultDiv) resultDiv.classList.remove('hidden');

            log(`Music generated: ${model}, ${data.duration_sec?.toFixed(1)}s`, 'success');
        } else if (data.status === 'completed' && data.entry_id) {
            // Completed but no inline audio — play from library
            if (infoDiv) infoDiv.innerHTML = `Generated! Entry: ${data.entry_id}`;
            if (audioEl) audioEl.src = `${API}/music/outputs/${data.entry_id}/audio`;
            if (resultDiv) resultDiv.classList.remove('hidden');
        } else {
            const errMsg = data.detail || data.error || data.message || JSON.stringify(data);
            showToast(`Generation failed: ${errMsg}`, 'error');
            log(`Music generation failed: ${errMsg}`, 'error');
        }
    } catch (e) {
        showToast(`Generation error: ${e.message}`, 'error');
        log(`Music generation error: ${e.message}`, 'error');
    } finally {
        musicState.activeGeneration = null;
        if (btn) { btn.disabled = false; btn.textContent = 'Generate Music'; }
    }
}

// ======================== Library Subtab ========================

export async function refreshMusicLibrary() {
    const grid = document.getElementById('music-library-grid');
    if (!grid) return;

    if (!musicState.apiRunning) {
        grid.innerHTML = '<div class="empty-state">Music server not running.</div>';
        return;
    }

    try {
        const resp = await fetch(`${API}/music/outputs`);
        const data = await resp.json();
        musicState.library.items = data.outputs || data.items || [];

        // Populate model filter
        const modelFilter = document.getElementById('music-library-model-filter');
        if (modelFilter && modelFilter.options.length <= 1) {
            const models = [...new Set(musicState.library.items.map(i => i.model))];
            modelFilter.innerHTML = '<option value="">All Models</option>' +
                models.map(m => `<option value="${m}">${m}</option>`).join('');
        }

        renderLibrary();
    } catch (e) {
        grid.innerHTML = `<div class="empty-state">Failed to load library: ${e.message}</div>`;
    }
}

export function filterMusicLibrary() {
    const model = document.getElementById('music-library-model-filter')?.value || '';
    const query = document.getElementById('music-library-search')?.value?.toLowerCase() || '';

    musicState.library.filters.model = model;
    musicState.library.filters.query = query;

    renderLibrary();
}

function renderLibrary() {
    const grid = document.getElementById('music-library-grid');
    if (!grid) return;

    let items = [...musicState.library.items];

    const { model, query } = musicState.library.filters;
    if (model) items = items.filter(i => i.model === model);
    if (query) items = items.filter(i =>
        (i.prompt || '').toLowerCase().includes(query) ||
        (i.model || '').toLowerCase().includes(query)
    );

    if (items.length === 0) {
        grid.innerHTML = '<div class="empty-state">No music outputs yet. Generate some music first!</div>';
        return;
    }

    grid.innerHTML = items.map(item => `
        <div class="tts-library-card" data-id="${item.id || item.entry_id}">
            <div class="tts-library-card-header">
                <span class="tts-library-model">${item.model}</span>
                <span class="tts-library-duration">${item.duration_sec ? item.duration_sec.toFixed(1) + 's' : '-'}</span>
            </div>
            <div class="tts-library-text">${(item.prompt || '').substring(0, 120)}${(item.prompt || '').length > 120 ? '...' : ''}</div>
            <div class="tts-library-player">
                <audio controls preload="none" class="tts-audio-player">
                    <source src="${API}/music/outputs/${item.id || item.entry_id}/audio" type="audio/${item.format === 'mp3' ? 'mpeg' : item.format || 'wav'}">
                </audio>
            </div>
            <div class="tts-library-meta">
                <span>${(item.format || 'wav').toUpperCase()} | ${item.sample_rate || '?'} Hz</span>
                <span>${item.created_at ? new Date(item.created_at).toLocaleDateString() : ''}</span>
            </div>
            <div class="tts-library-actions">
                <a href="${API}/music/outputs/${item.id || item.entry_id}/audio" download class="btn-secondary btn-small">Download</a>
                <button class="btn-danger btn-small" onclick="deleteMusicLibraryItem('${item.id || item.entry_id}')">Delete</button>
            </div>
        </div>
    `).join('');
}

export async function deleteMusicLibraryItem(entryId) {
    if (!confirm('Delete this audio file?')) return;

    try {
        const resp = await fetch(`${API}/music/outputs/${entryId}`, { method: 'DELETE' });
        const data = await resp.json();
        showToast('Deleted', 'success');
        musicState.library.items = musicState.library.items.filter(i => (i.id || i.entry_id) !== entryId);
        renderLibrary();
    } catch (e) {
        showToast(`Delete error: ${e.message}`, 'error');
    }
}
