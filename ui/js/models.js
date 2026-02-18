/**
 * Model load/unload/select/modal functions for AgentNate UI
 */

import { state, API } from './state.js';
import { log, escapeHtml, updateConnectionStatus, updateDebugStatus } from './utils.js';
import { updateVisionUI } from './images.js';

// Lazy imports to avoid circular deps
async function getModelSettings() {
    return await import('./model-settings.js');
}

async function getPresets() {
    return await import('./presets.js');
}

export async function refreshLoadedModels() {
    try {
        const resp = await fetch(`${API}/models/loaded`);
        const models = await resp.json();
        console.log('refreshLoadedModels: got', models.length, 'models from server');
        console.log('refreshLoadedModels: pending loads:', Object.keys(state.pendingLoads).length);
        renderLoadedModels(models);

        // Auto-select first ready model if none selected (e.g. after page reload)
        if (!state.currentModel && models.length > 0) {
            const readyModel = models.find(m => m.status !== 'error');
            if (readyModel) {
                selectModel(readyModel.id);
                console.log('Auto-selected model on load:', readyModel.id);
            }
        }
    } catch (e) {
        console.error('Failed to refresh models:', e);
        log('Failed to refresh models: ' + e.message, 'error');
    }
}

export function renderLoadedModels(models) {
    const container = document.getElementById('loaded-models');
    if (!container) return;

    if (!Array.isArray(models)) {
        console.warn('renderLoadedModels: models is not an array', models);
        models = [];
    }

    // Sync state.models cache with the authoritative API response
    // Only clean up UUID-keyed entries (loaded instances), preserve provider-keyed entries (model lists)
    const activeIds = new Set(models.map(m => m.id));
    for (const id of Object.keys(state.models)) {
        if (!activeIds.has(id) && id.includes('-')) {
            delete state.models[id];
        }
    }
    models.forEach(m => {
        state.models[m.id] = m;
    });

    if (state.currentModel && state.models[state.currentModel]) {
        state.currentModelHasVision = state.models[state.currentModel].has_vision || false;
        updateVisionUI();
    }

    const now = Date.now();
    const maxAge = 5 * 60 * 1000;
    const pendingEntries = Object.entries(state.pendingLoads).filter(([id, info]) => {
        if (now - info.startTime > maxAge) {
            console.warn('Removing stale pending load:', id);
            delete state.pendingLoads[id];
            return false;
        }
        return true;
    });

    const errorMaxAge = 10 * 60 * 1000;
    const errorEntries = Object.entries(state.errorLoads).filter(([id, info]) => {
        if (now - info.timestamp > errorMaxAge) {
            console.warn('Auto-removing old error load:', id);
            delete state.errorLoads[id];
            return false;
        }
        return true;
    });

    const totalCount = models.length + pendingEntries.length + errorEntries.length;

    if (totalCount === 0) {
        container.innerHTML = '<div class="empty-state">No models loaded</div>';
        return;
    }

    let html = errorEntries.map(([id, info]) => `
        <div class="model-item error" title="${escapeHtml(info.error)}">
            <div class="model-info">
                <div class="model-name error-name">${escapeHtml(info.model)}</div>
                <div class="model-provider error-text">ERROR: ${escapeHtml(info.error)}</div>
            </div>
            <button class="btn-icon btn-dismiss" onclick="event.stopPropagation(); dismissErrorLoad('${escapeHtml(id)}')" title="Dismiss">&times;</button>
        </div>
    `).join('');

    html += pendingEntries.map(([id, info]) => `
        <div class="model-item pending" title="Loading ${escapeHtml(info.model)}...">
            <div class="model-info">
                <div class="model-name">${escapeHtml(info.model)}</div>
                <div class="model-provider">${escapeHtml(info.provider)} <span class="loading-dots">...</span></div>
            </div>
            <button class="btn-icon btn-cancel" onclick="event.stopPropagation(); cancelModelLoad('${escapeHtml(id)}')" title="Cancel Load">&times;</button>
        </div>
    `).join('');

    html += models.map(m => {
        const isActive = m.id === state.currentModel;
        const isUnloading = state.unloadingModels[m.id];
        const isError = m.status === 'error';
        const hasVision = m.has_vision || false;
        const modelTitle = escapeHtml(m.model || m.name || 'Unknown');

        const modelSettings = state.modelSettings[m.id];
        const displayName = modelSettings?.displayName || m.name || 'Unknown';
        const modelName = escapeHtml(displayName);

        const modelProvider = escapeHtml(m.provider || 'Unknown');
        const modelId = escapeHtml(m.id);
        const visionBadge = hasVision ? '<span class="vision-badge" title="Vision model">&#128247;</span>' : '';

        if (isError) {
            const errorMsg = m.metadata?.error || 'Model in error state';
            return `
                <div class="model-item error" title="${escapeHtml(errorMsg)}">
                    <div class="model-info">
                        <div class="model-name error-name">${modelName}</div>
                        <div class="model-provider error-text">ERROR: ${escapeHtml(errorMsg)}</div>
                    </div>
                    <button class="btn-icon btn-unload" onclick="event.stopPropagation(); unloadModel('${modelId}')" title="Remove">&times;</button>
                </div>
            `;
        }

        return `
            <div class="model-item ${isActive ? 'active' : ''} ${isUnloading ? 'unloading' : ''}"
                 onclick="selectModel('${modelId}')"
                 title="${modelTitle}">
                <div class="model-info">
                    <div class="model-name">${modelName}${visionBadge}</div>
                    <div class="model-provider">${modelProvider}${isUnloading ? ' <span class="loading-dots">...</span>' : ''}</div>
                </div>
                <button class="btn-icon btn-settings" onclick="event.stopPropagation(); openModelSettings('${modelId}')" title="Settings">&#9881;</button>
                <button class="btn-icon btn-unload" ${isUnloading ? 'disabled' : ''} onclick="event.stopPropagation(); unloadModel('${modelId}')" title="Unload">&times;</button>
            </div>
        `;
    }).join('');

    container.innerHTML = html;

    // Refresh panel model dropdowns when model list changes
    import('./panels.js').then(m => {
        m.refreshAllPanelModelDropdowns();
        m.updateAllPanelInputStates();
    });
}

export async function cancelModelLoad(instanceId) {
    log(`Cancelling model load: ${instanceId}`, 'info');
    delete state.pendingLoads[instanceId];
    renderLoadedModels(await getLoadedModels());

    try {
        const resp = await fetch(`${API}/models/cancel/${instanceId}`, { method: 'POST' });
        const result = await resp.json();
        if (result.success) {
            log('Model load cancelled', 'success');
        } else {
            log('Cancel sent (worker may already be stopped)', 'warning');
        }
    } catch (e) {
        console.error('Cancel API error:', e);
    }
}

export async function dismissErrorLoad(errorId) {
    delete state.errorLoads[errorId];
    log('Error dismissed', 'info');
    renderLoadedModels(await getLoadedModels());
}

export async function getLoadedModels() {
    try {
        const resp = await fetch(`${API}/models/loaded`);
        return await resp.json();
    } catch (e) {
        return [];
    }
}

export async function getPendingLoads() {
    try {
        const resp = await fetch(`${API}/models/pending`);
        const data = await resp.json();
        return data.pending || [];
    } catch (e) {
        return [];
    }
}

export function selectModel(instanceId) {
    console.log('selectModel:', instanceId);
    if (!instanceId) {
        console.error('selectModel called with empty instanceId');
        return;
    }
    state.currentModel = instanceId;

    const modelInfo = state.models[instanceId];
    state.currentModelHasVision = modelInfo?.has_vision || false;
    console.log('Model has vision:', state.currentModelHasVision);

    updateVisionUI();

    getLoadedModels().then(models => {
        renderLoadedModels(models);
        import('./chat.js').then(chat => chat.updateChatUI());
    });
}

export async function unloadModel(instanceId) {
    console.log('=== unloadModel ===');
    console.log('Instance ID:', instanceId);
    console.log('Current model:', state.currentModel);

    if (!instanceId) {
        console.error('unloadModel called with empty instanceId');
        return;
    }

    if (state.unloadingModels[instanceId]) {
        console.log('Already unloading this model, ignoring duplicate call');
        return;
    }
    state.unloadingModels[instanceId] = true;

    getLoadedModels().then(models => renderLoadedModels(models));
    log(`Unloading model...`, 'info');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        const resp = await fetch(`${API}/models/${instanceId}`, {
            method: 'DELETE',
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        const result = await resp.json();
        console.log('Unload result:', result);

        if (!result.success) {
            log('Failed to unload: ' + (result.error || 'Unknown error'), 'error');
            delete state.unloadingModels[instanceId];
            await refreshLoadedModels();
            return;
        }

        log(`Model unloaded successfully`, 'success');
        delete state.unloadingModels[instanceId];

        if (state.currentModel === instanceId) {
            state.currentModel = null;
            state.currentModelHasVision = false;
            updateVisionUI();
        }

        await refreshLoadedModels();
        import('./chat.js').then(chat => chat.updateChatUI());
    } catch (e) {
        console.error('Failed to unload model:', e);
        if (e.name === 'AbortError') {
            log('Unload taking longer than expected, checking status...', 'warning');
            pollUnloadStatus(instanceId);
        } else {
            log('Error unloading model: ' + e.message, 'error');
            delete state.unloadingModels[instanceId];
            await refreshLoadedModels();
        }
    }
}

export async function pollUnloadStatus(instanceId) {
    const maxAttempts = 12;
    let attempts = 0;

    const checkStatus = async () => {
        attempts++;
        try {
            const resp = await fetch(`${API}/models/loaded`);
            const loaded = await resp.json();
            const stillLoaded = loaded.some(m => m.id === instanceId);

            if (!stillLoaded) {
                log('Model unloaded successfully', 'success');
                delete state.unloadingModels[instanceId];
                if (state.currentModel === instanceId) {
                    state.currentModel = null;
                    state.currentModelHasVision = false;
                    updateVisionUI();
                }
                await refreshLoadedModels();
                import('./chat.js').then(chat => chat.updateChatUI());
                return;
            }

            if (attempts >= maxAttempts) {
                log('Unload timed out - please try again', 'error');
                delete state.unloadingModels[instanceId];
                await refreshLoadedModels();
                return;
            }

            setTimeout(checkStatus, 5000);
        } catch (e) {
            console.error('Error polling unload status:', e);
            delete state.unloadingModels[instanceId];
            await refreshLoadedModels();
        }
    };

    setTimeout(checkStatus, 5000);
}

export async function showLoadModal() {
    document.getElementById('load-modal').classList.remove('hidden');

    const contextSlider = document.getElementById('context-length');
    if (contextSlider) {
        contextSlider.value = 4096;
        const { updateContextLengthDisplay } = await import('./presets.js');
        updateContextLengthDisplay();
    }

    const gpuLayersInput = document.getElementById('gpu-layers');
    if (gpuLayersInput) {
        gpuLayersInput.value = -1;
    }

    const presetSaveOption = document.getElementById('preset-save-option');
    if (presetSaveOption) {
        presetSaveOption.classList.remove('hidden');
    }

    const saveAsPresetCheckbox = document.getElementById('save-as-preset');
    if (saveAsPresetCheckbox) {
        saveAsPresetCheckbox.checked = false;
    }

    const presetNameInput = document.getElementById('preset-name');
    if (presetNameInput) {
        presetNameInput.classList.add('hidden');
        presetNameInput.value = '';
    }

    await loadAvailableModels();
}

export function hideLoadModal() {
    document.getElementById('load-modal').classList.add('hidden');
}

export async function loadAvailableModels() {
    try {
        console.log('loadAvailableModels: fetching health only...');

        const healthResp = await fetch(`${API}/models/health/all`);
        const health = await healthResp.json();

        state.models = {};
        state.gpus = health.llama_cpp?.gpus || [];

        const gpuSelect = document.getElementById('load-gpu');
        let gpuOptions = '';
        if (state.gpus && state.gpus.length > 0) {
            state.gpus.forEach((g, idx) => {
                const freeMB = g.memory_free_mb || g.memory_free || 0;
                const freeGB = Math.round(freeMB / 1024);
                const selected = idx === 0 ? 'selected' : '';
                gpuOptions += `<option value="${g.index}" ${selected}>GPU ${g.index}: ${g.name} (${freeGB}GB free)</option>`;
            });
        } else {
            gpuOptions = '<option value="">No GPUs detected</option>';
        }
        gpuSelect.innerHTML = gpuOptions;

        await onProviderChange();

    } catch (e) {
        console.error('Failed to load models:', e);
    }
}

export async function onProviderChange() {
    const provider = document.getElementById('load-provider').value;
    const modelSelect = document.getElementById('load-model');

    console.log('onProviderChange called, provider:', provider);

    const gpuSelectGroup = document.getElementById('load-gpu')?.closest('.form-group');
    const contextGroup = document.getElementById('context-length')?.closest('.form-group');
    const gpuLayersGroup = document.getElementById('gpu-layers')?.closest('.form-group');
    const ggufSection = document.getElementById('gguf-browse-section');

    // Show GGUF browse for llama_cpp provider
    if (ggufSection) {
        ggufSection.classList.toggle('hidden', provider !== 'llama_cpp');
    }

    if (provider === 'openrouter') {
        if (gpuSelectGroup) gpuSelectGroup.style.display = 'none';
        if (contextGroup) contextGroup.style.display = 'none';
        if (gpuLayersGroup) gpuLayersGroup.style.display = 'none';
    } else if (provider === 'ollama') {
        if (gpuSelectGroup) gpuSelectGroup.style.display = 'none';
        if (contextGroup) contextGroup.style.display = '';
        if (gpuLayersGroup) gpuLayersGroup.style.display = 'none';
    } else if (provider === 'lm_studio') {
        if (gpuSelectGroup) gpuSelectGroup.style.display = '';
        if (contextGroup) contextGroup.style.display = '';
        if (gpuLayersGroup) gpuLayersGroup.style.display = 'none';
    } else if (provider === 'vllm') {
        if (gpuSelectGroup) gpuSelectGroup.style.display = '';
        if (contextGroup) contextGroup.style.display = '';
        if (gpuLayersGroup) gpuLayersGroup.style.display = 'none';
    } else {
        if (gpuSelectGroup) gpuSelectGroup.style.display = '';
        if (contextGroup) contextGroup.style.display = '';
        if (gpuLayersGroup) gpuLayersGroup.style.display = '';
    }

    if (!state.models[provider]) {
        modelSelect.innerHTML = '<option value="">Loading models...</option>';
        modelSelect.disabled = true;

        try {
            console.log(`Fetching models for ${provider}...`);
            const resp = await fetch(`${API}/models/list/${provider}`);
            const data = await resp.json();

            state.models[provider] = data.models || [];
            console.log(`Got ${state.models[provider].length} models for ${provider}`);

        } catch (e) {
            console.error(`Failed to fetch models for ${provider}:`, e);
            state.models[provider] = [];
        }

        modelSelect.disabled = false;
    }

    const models = state.models[provider] || [];
    modelSelect.innerHTML = '<option value="">Select a model...</option>' +
        models.map(m => `<option value="${m.id}">${m.name}</option>`).join('');

    onModelSelectChange();
}

export async function onModelSelectChange() {
    const provider = document.getElementById('load-provider').value;
    const modelId = document.getElementById('load-model').value;
    const contextSlider = document.getElementById('context-length');
    const contextDisplay = document.getElementById('context-length-value');
    const hint = contextSlider?.closest('.form-group')?.querySelector('.form-hint');

    if (!modelId || !contextSlider) return;

    const models = state.models[provider] || [];
    const selectedModel = models.find(m => m.id === modelId);

    if (!selectedModel) return;

    if (hint) {
        hint.innerHTML = '<span style="color: var(--text-muted);">Fetching context length...</span>';
    }

    let maxContext = selectedModel.context_length;

    try {
        const resp = await fetch(`${API}/models/context-length/${provider}/${encodeURIComponent(modelId)}`);
        const data = await resp.json();

        if (data.success && data.context_length) {
            maxContext = data.context_length;
            const suffix = data.estimated ? ' (estimated)' : '';
            console.log(`Model ${selectedModel.name}: context = ${maxContext}${suffix}`);
            selectedModel.context_length = maxContext;
        }
    } catch (e) {
        console.warn('Context length fetch failed, using estimate:', e);
    }

    if (!maxContext) {
        if (provider === 'openrouter') {
            maxContext = 200000;
        } else {
            maxContext = 131072;
        }
    }

    contextSlider.max = maxContext;

    let currentValue = parseInt(contextSlider.value);

    if (currentValue > maxContext) {
        currentValue = maxContext;
    }

    if (currentValue === 4096) {
        if (maxContext <= 8192) {
            currentValue = maxContext;
        } else if (maxContext <= 32768) {
            currentValue = maxContext;
        } else {
            currentValue = Math.min(32768, maxContext);
        }
    }

    contextSlider.value = currentValue;

    if (contextDisplay) {
        contextDisplay.textContent = currentValue.toLocaleString();
    }

    if (hint) {
        hint.textContent = `Max: ${maxContext.toLocaleString()} tokens`;
    }
}

export async function loadSelectedModel() {
    if (state.loadInProgress) {
        console.log('Load already in progress, ignoring click');
        return;
    }

    const provider = document.getElementById('load-provider').value;
    const modelId = document.getElementById('load-model').value;

    if (!modelId) {
        alert('Please select a model');
        return;
    }

    state.loadInProgress = true;
    const loadBtn = document.getElementById('load-model-btn');
    if (loadBtn) {
        loadBtn.classList.add('loading');
        loadBtn.disabled = true;
    }

    const options = {};

    const gpuSelect = document.getElementById('load-gpu');
    if (gpuSelect && gpuSelect.value !== '') {
        options.gpu_index = parseInt(gpuSelect.value);
    }

    const contextSlider = document.getElementById('context-length');
    const contextLength = contextSlider ? parseInt(contextSlider.value) : 4096;

    const gpuLayersInput = document.getElementById('gpu-layers');
    const gpuLayers = gpuLayersInput ? parseInt(gpuLayersInput.value) : -1;

    if (provider === 'llama_cpp' || provider === 'lm_studio' || provider === 'ollama' || provider === 'vllm') {
        options.n_ctx = contextLength;
    }

    if (provider === 'llama_cpp') {
        options.n_gpu_layers = gpuLayers;
    }

    const modelSelect = document.getElementById('load-model');
    const modelName = modelSelect.options[modelSelect.selectedIndex]?.text || modelId;

    log(`Loading model: ${modelName}...`, 'info');
    console.log('=== loadSelectedModel ===');
    console.log('Provider:', provider);
    console.log('Model ID:', modelId);
    console.log('Options:', options);

    const saveAsPresetCheckbox = document.getElementById('save-as-preset');
    if (saveAsPresetCheckbox && saveAsPresetCheckbox.checked) {
        const presetNameInput = document.getElementById('preset-name');
        const presetName = presetNameInput ? presetNameInput.value.trim() : '';
        const { createPreset } = await getPresets();
        if (presetName) {
            await createPreset(presetName);
        } else {
            await createPreset(modelName);
        }
    }

    hideLoadModal();

    try {
        const resp = await fetch(`${API}/models/load-async`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, model_id: modelId, options })
        });

        const result = await resp.json();
        console.log('Async load started:', result);

        state.loadInProgress = false;
        resetLoadButton();

        if (!result.success) {
            log('Failed to start model load: ' + result.error, 'error');
            alert('Failed to start model load: ' + result.error);
            return;
        }

        const pendingId = result.pending_id;
        state.pendingLoads[pendingId] = {
            model: modelName,
            provider: provider,
            modelId: modelId,
            startTime: Date.now()
        };

        const currentModels = await getLoadedModels();
        renderLoadedModels(currentModels);

        log(`Model load started: ${modelName} (${pendingId})`, 'info');

        const alreadyLoaded = currentModels.find(m => {
            return m.model === modelId || m.name === modelName || (m.model && m.model.includes(modelName));
        });

        if (alreadyLoaded) {
            delete state.pendingLoads[pendingId];
            state.currentModel = alreadyLoaded.id;
            log(`Model loaded: ${modelName} (${alreadyLoaded.id})`, 'success');
            state.models[alreadyLoaded.id] = alreadyLoaded;
            renderLoadedModels(currentModels);
            import('./chat.js').then(chat => chat.updateChatUI());
            hideLoadModal();
            return;
        }

        const pollInterval = setInterval(async () => {
            if (!state.pendingLoads[pendingId]) {
                clearInterval(pollInterval);
                renderLoadedModels(await getLoadedModels());
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
                        error: 'Model load failed (check logs for details)',
                        timestamp: Date.now()
                    };
                    log(`Model load failed: ${modelName}`, 'error');
                    renderLoadedModels(await getLoadedModels());
                    return;
                }

                const loaded = await getLoadedModels();
                const nowLoaded = loaded.find(m =>
                    m.model === modelId || m.name === modelName || (m.model && m.model.includes(modelName))
                );

                if (nowLoaded) {
                    clearInterval(pollInterval);
                    delete state.pendingLoads[pendingId];
                    state.currentModel = nowLoaded.id;
                    log(`Model loaded: ${modelName} (${nowLoaded.id})`, 'success');

                    const instanceId = nowLoaded.id;
                    const contextLen = nowLoaded.context_length || contextLength;
                    if (contextLen && !state.modelSettings[instanceId]?.params?.max_tokens) {
                        // Use 4096 as default, capped by context length
                        // Don't set max_tokens = contextLen — that's the window, not generation limit
                        const autoMaxTokens = Math.min(4096, contextLen);
                        console.log(`[ModelSettings] Auto-initializing max_tokens to ${autoMaxTokens} (context: ${contextLen}) for ${instanceId}`);
                        if (!state.modelSettings[instanceId]) {
                            state.modelSettings[instanceId] = { params: {} };
                        }
                        if (!state.modelSettings[instanceId].params) {
                            state.modelSettings[instanceId].params = {};
                        }
                        state.modelSettings[instanceId].params.max_tokens = autoMaxTokens;
                        getModelSettings().then(ms => ms.saveModelSettingsToStorage());
                    }

                    state.models[instanceId] = nowLoaded;

                    if (provider === 'lm_studio') {
                        log('Warming up LM Studio model...', 'info');
                        await sendWarmupMessage(nowLoaded.id);
                    }

                    renderLoadedModels(loaded);
                    import('./chat.js').then(chat => chat.updateChatUI());
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
                    log(`Model load failed: ${modelName} (process terminated)`, 'error');
                    renderLoadedModels(loaded);
                }
            } catch (e) {
                console.error('Poll error:', e);
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
                    error: 'Model load timed out (5 minutes)',
                    timestamp: Date.now()
                };
                log('Model load timed out', 'error');
                refreshLoadedModels();
            }
        }, 300000);

    } catch (e) {
        console.error('Load error:', e);
        state.loadInProgress = false;
        resetLoadButton();
        log('Error starting model load: ' + e.message, 'error');
        alert('Error: ' + e.message);
        await refreshLoadedModels();
    }
}

export function resetLoadButton() {
    const loadBtn = document.getElementById('load-model-btn');
    if (loadBtn) {
        loadBtn.classList.remove('loading');
        loadBtn.disabled = false;
    }
}

export async function sendWarmupMessage(instanceId) {
    return new Promise((resolve) => {
        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
            resolve();
            return;
        }

        const requestId = `warmup-${Date.now()}`;

        const onMessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.request_id === requestId && (data.type === 'done' || data.type === 'error')) {
                state.ws.removeEventListener('message', onMessage);
                resolve();
            }
        };

        state.ws.addEventListener('message', onMessage);

        state.ws.send(JSON.stringify({
            action: 'chat',
            instance_id: instanceId,
            messages: [{ role: 'user', content: 'Hi' }],
            request_id: requestId,
            params: { max_tokens: 5 }
        }));

        setTimeout(() => {
            state.ws.removeEventListener('message', onMessage);
            resolve();
        }, 30000);
    });
}
