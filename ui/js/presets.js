/**
 * Model load presets for AgentNate UI
 */

import { state, API } from './state.js';
import { log, escapeHtml } from './utils.js';

// Lazy imports to avoid circular deps
async function getModelsModule() {
    return await import('./models.js');
}

export async function initModelPresets() {
    await loadPresetsFromServer();
    renderPresetsDropdown();
    log('Model presets initialized', 'info');
}

async function loadPresetsFromServer() {
    try {
        const resp = await fetch(`${API}/presets/list`);
        if (resp.ok) {
            state.modelPresets = await resp.json();
            console.log('[Presets] Loaded', state.modelPresets.length, 'presets from server');
        } else {
            console.error('[Presets] Failed to load presets:', resp.statusText);
            state.modelPresets = [];
        }
    } catch (e) {
        console.error('[Presets] Failed to load presets:', e);
        state.modelPresets = [];
    }
}

async function savePresetToServer(preset) {
    try {
        const resp = await fetch(`${API}/presets/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preset })
        });
        const result = await resp.json();
        if (result.success) {
            console.log('[Presets] Saved preset to server:', preset.name);
        } else {
            console.error('[Presets] Failed to save preset:', result.error);
        }
        return result.success;
    } catch (e) {
        console.error('[Presets] Failed to save preset:', e);
        return false;
    }
}

export async function createPreset(name) {
    const provider = document.getElementById('load-provider').value;
    const modelId = document.getElementById('load-model').value;
    const modelSelect = document.getElementById('load-model');
    const modelName = modelSelect.options[modelSelect.selectedIndex]?.text || modelId;
    const gpuSelect = document.getElementById('load-gpu');
    const gpuIndex = gpuSelect && gpuSelect.value !== '' ? parseInt(gpuSelect.value) : 0;
    const contextSlider = document.getElementById('context-length');
    const contextLength = contextSlider ? parseInt(contextSlider.value) : 4096;
    const gpuLayersInput = document.getElementById('gpu-layers');
    const gpuLayers = gpuLayersInput ? parseInt(gpuLayersInput.value) : -1;

    const preset = {
        id: 'preset-' + Date.now(),
        name: name,
        provider: provider,
        modelId: modelId,
        modelName: modelName,
        contextLength: contextLength,
        gpuLayers: gpuLayers,
        gpuIndex: gpuIndex,
        createdAt: Date.now()
    };

    const saved = await savePresetToServer(preset);
    if (saved) {
        state.modelPresets.push(preset);
        renderPresetsDropdown();
        log(`Preset "${name}" saved`, 'success');
    } else {
        log(`Failed to save preset "${name}"`, 'error');
    }
    return preset;
}

export async function deletePreset(presetId) {
    const index = state.modelPresets.findIndex(p => p.id === presetId);
    if (index !== -1) {
        const name = state.modelPresets[index].name;

        try {
            const resp = await fetch(`${API}/presets/${presetId}`, { method: 'DELETE' });
            const result = await resp.json();
            if (result.success) {
                state.modelPresets.splice(index, 1);
                renderPresetsDropdown();
                renderPresetsList();
                log(`Preset "${name}" deleted`, 'info');
            } else {
                log(`Failed to delete preset: ${result.error}`, 'error');
            }
        } catch (e) {
            console.error('[Presets] Delete error:', e);
            log('Failed to delete preset', 'error');
        }
    }
}

export async function loadFromPreset(presetId) {
    if (!presetId) return;

    const preset = state.modelPresets.find(p => p.id === presetId);
    if (!preset) {
        log('Preset not found', 'error');
        return;
    }

    if (state.loadInProgress) {
        log('Load already in progress', 'warning');
        return;
    }

    log(`Loading from preset: ${preset.name}`, 'info');
    console.log('[Presets] Loading preset:', preset);

    state.loadInProgress = true;

    const options = {
        n_ctx: preset.contextLength,
        n_gpu_layers: preset.gpuLayers,
        gpu_index: preset.gpuIndex
    };

    try {
        const resp = await fetch(`${API}/models/load-async`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: preset.provider,
                model_id: preset.modelId,
                options
            })
        });

        const result = await resp.json();
        state.loadInProgress = false;

        if (!result.success) {
            log('Failed to load preset: ' + result.error, 'error');
            alert('Failed to load preset: ' + result.error);
            document.getElementById('preset-dropdown').value = '';
            return;
        }

        const pendingId = result.pending_id;
        state.pendingLoads[pendingId] = {
            model: preset.modelName,
            provider: preset.provider,
            modelId: preset.modelId,
            startTime: Date.now()
        };

        const models = await getModelsModule();
        const currentModels = await models.getLoadedModels();
        models.renderLoadedModels(currentModels);

        log(`Preset "${preset.name}" loading started`, 'info');

        const pollInterval = setInterval(async () => {
            if (!state.pendingLoads[pendingId]) {
                clearInterval(pollInterval);
                const m = await getModelsModule();
                m.renderLoadedModels(await m.getLoadedModels());
                return;
            }

            try {
                const pendingResp = await fetch(`${API}/models/pending`);
                const pendingData = await pendingResp.json();
                const pendingItem = pendingData.pending?.find(p => p.id === pendingId);

                if (pendingItem && pendingItem.status === 'error') {
                    clearInterval(pollInterval);
                    const loadInfo = state.pendingLoads[pendingId];
                    delete state.pendingLoads[pendingId];

                    state.errorLoads[pendingId] = {
                        model: loadInfo.model,
                        provider: loadInfo.provider,
                        error: 'Preset load failed (check logs for details)',
                        timestamp: Date.now()
                    };

                    log(`Preset "${preset.name}" load failed`, 'error');
                    const m = await getModelsModule();
                    m.renderLoadedModels(await m.getLoadedModels());
                    return;
                }

                const m = await getModelsModule();
                const loaded = await m.getLoadedModels();

                const nowLoaded = loaded.find(mdl =>
                    mdl.model === preset.modelId ||
                    mdl.name === preset.modelName ||
                    (mdl.model && mdl.model.includes(preset.modelName))
                );

                if (nowLoaded) {
                    clearInterval(pollInterval);
                    delete state.pendingLoads[pendingId];
                    state.currentModel = nowLoaded.id;
                    log(`Preset "${preset.name}" loaded successfully`, 'success');

                    m.renderLoadedModels(loaded);
                    const chat = await import('./chat.js');
                    chat.updateChatUI();
                } else if (!pendingItem) {
                    clearInterval(pollInterval);
                    const loadInfo = state.pendingLoads[pendingId];
                    delete state.pendingLoads[pendingId];

                    state.errorLoads[pendingId] = {
                        model: loadInfo.model,
                        provider: loadInfo.provider,
                        error: 'Load failed - model process terminated (OOM or other error)',
                        timestamp: Date.now()
                    };

                    log(`Preset "${preset.name}" load failed (process terminated)`, 'error');
                    m.renderLoadedModels(loaded);
                }
            } catch (e) {
                console.error('Preset poll error:', e);
            }
        }, 2000);

        setTimeout(() => {
            if (state.pendingLoads[pendingId]) {
                clearInterval(pollInterval);
                const loadInfo = state.pendingLoads[pendingId];
                delete state.pendingLoads[pendingId];

                state.errorLoads[pendingId] = {
                    model: loadInfo.model,
                    provider: loadInfo.provider,
                    error: 'Preset load timed out (5 minutes)',
                    timestamp: Date.now()
                };

                log('Preset load timed out', 'error');
                getModelsModule().then(m => m.refreshLoadedModels());
            }
        }, 300000);

    } catch (e) {
        state.loadInProgress = false;
        log('Preset load error: ' + e.message, 'error');
        console.error('[Presets] Load error:', e);
    }

    document.getElementById('preset-dropdown').value = '';
}

export function renderPresetsDropdown() {
    const dropdown = document.getElementById('preset-dropdown');
    if (!dropdown) return;

    dropdown.innerHTML = '<option value="">Select a preset...</option>';

    state.modelPresets.forEach(preset => {
        const option = document.createElement('option');
        option.value = preset.id;
        option.textContent = preset.name;
        option.title = `${preset.modelName} (ctx: ${preset.contextLength}, layers: ${preset.gpuLayers})`;
        dropdown.appendChild(option);
    });
}

export function renderPresetsList() {
    const listEl = document.getElementById('presets-list');
    if (!listEl) return;

    if (state.modelPresets.length === 0) {
        listEl.innerHTML = '<div class="empty-state">No presets saved yet</div>';
        return;
    }

    listEl.innerHTML = state.modelPresets.map(preset => `
        <div class="preset-item" data-id="${preset.id}">
            <div class="preset-item-info">
                <div class="preset-item-name">${escapeHtml(preset.name)}</div>
                <div class="preset-item-details">
                    <span title="Model">${escapeHtml(preset.modelName)}</span>
                    <span title="Context Length">ctx: ${preset.contextLength.toLocaleString()}</span>
                    <span title="GPU Layers">layers: ${preset.gpuLayers}</span>
                </div>
            </div>
            <div class="preset-item-actions">
                <button class="btn-icon" onclick="loadFromPreset('${preset.id}')" title="Load this preset">&#9654;</button>
                <button class="btn-icon btn-danger" onclick="confirmDeletePreset('${preset.id}')" title="Delete preset">&#128465;</button>
            </div>
        </div>
    `).join('');
}

export async function confirmDeletePreset(presetId) {
    const preset = state.modelPresets.find(p => p.id === presetId);
    if (!preset) return;

    if (confirm(`Delete preset "${preset.name}"?`)) {
        await deletePreset(presetId);
    }
}

export function openPresetsModal() {
    renderPresetsList();
    document.getElementById('presets-modal').classList.remove('hidden');
}

export function closePresetsModal() {
    document.getElementById('presets-modal').classList.add('hidden');
}

export function updateContextLengthDisplay() {
    const slider = document.getElementById('context-length');
    const display = document.getElementById('context-length-value');
    if (slider && display) {
        const value = parseInt(slider.value);
        display.textContent = value.toLocaleString();
    }
}

export function togglePresetNameInput() {
    const checkbox = document.getElementById('save-as-preset');
    const nameInput = document.getElementById('preset-name');
    if (checkbox && nameInput) {
        nameInput.classList.toggle('hidden', !checkbox.checked);
        if (checkbox.checked) {
            nameInput.focus();
        }
    }
}
