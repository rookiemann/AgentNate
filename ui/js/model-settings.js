/**
 * Model inference settings panel for AgentNate UI
 */

import { state, API, MODEL_SETTINGS_KEY } from './state.js';
import { log } from './utils.js';

export function initModelSettings() {
    loadModelSettingsFromStorage();
    setupParamSliderListeners();
    log('Model settings initialized', 'info');
}

export function loadModelSettingsFromStorage() {
    try {
        const stored = localStorage.getItem(MODEL_SETTINGS_KEY);
        state.modelSettings = stored ? JSON.parse(stored) : {};
        console.log('[ModelSettings] Loaded settings for', Object.keys(state.modelSettings).length, 'models');
    } catch (e) {
        console.error('[ModelSettings] Failed to load settings:', e);
        state.modelSettings = {};
    }
}

export function saveModelSettingsToStorage() {
    try {
        localStorage.setItem(MODEL_SETTINGS_KEY, JSON.stringify(state.modelSettings));
        console.log('[ModelSettings] Saved settings for', Object.keys(state.modelSettings).length, 'models');
    } catch (e) {
        console.error('[ModelSettings] Failed to save settings:', e);
    }
}

export function getGlobalDefaultParams() {
    return {
        max_tokens: 2048,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 40,
        repeat_penalty: 1.1,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        mirostat: 0,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        typical_p: 1.0,
        tfs_z: 1.0,
    };
}

export function getEffectiveParams(instanceId) {
    const defaults = getGlobalDefaultParams();
    const modelOverrides = state.modelSettings[instanceId]?.params || {};
    return { ...defaults, ...modelOverrides };
}

export async function openModelSettings(instanceId) {
    state.settingsPanelModelId = instanceId;
    let model = state.models[instanceId];
    if (!model) {
        // Model data missing from cache - re-fetch loaded models
        try {
            const resp = await fetch(`${API}/models/loaded`);
            const models = await resp.json();
            models.forEach(m => { state.models[m.id] = m; });
            model = state.models[instanceId];
        } catch (e) {
            console.error('Failed to refresh models for settings:', e);
        }
    }
    if (!model) {
        log('Model not found: ' + instanceId, 'error');
        return;
    }

    const settings = state.modelSettings[instanceId] || {};
    const params = getEffectiveParams(instanceId);

    document.getElementById('model-display-name').value = settings.displayName || model.name || '';
    document.getElementById('model-original-name').textContent = model.model || model.name || '';

    const isCloudProvider = model.provider === 'openrouter';
    const maxTokensSlider = document.getElementById('param-max-tokens');
    const maxTokensLimitEl = document.getElementById('max-tokens-limit');
    const maxTokensHint = maxTokensSlider?.parentElement?.querySelector('.form-hint');

    const hasSavedMaxTokens = settings.params?.max_tokens !== undefined;

    if (isCloudProvider) {
        const cloudMaxTokens = 200000;
        if (maxTokensSlider) maxTokensSlider.max = cloudMaxTokens;
        if (maxTokensLimitEl) maxTokensLimitEl.textContent = cloudMaxTokens.toLocaleString();
        if (maxTokensHint) maxTokensHint.innerHTML = 'Cloud model - set based on model\'s max output';
        if (!hasSavedMaxTokens) {
            params.max_tokens = 4096;
            if (!state.modelSettings[instanceId]) state.modelSettings[instanceId] = { params: {} };
            if (!state.modelSettings[instanceId].params) state.modelSettings[instanceId].params = {};
            state.modelSettings[instanceId].params.max_tokens = 4096;
            saveModelSettingsToStorage();
        }
    } else {
        const contextLength = model.context_length || 4096;
        if (maxTokensSlider) maxTokensSlider.max = contextLength;
        if (maxTokensLimitEl) maxTokensLimitEl.textContent = contextLength.toLocaleString();
        if (maxTokensHint) maxTokensHint.innerHTML = 'Limited by model context: <span id="max-tokens-limit">' + contextLength.toLocaleString() + '</span>';
        if (!hasSavedMaxTokens) {
            // Use 4096 as default generation limit, capped by context length
            const autoMaxTokens = Math.min(4096, contextLength);
            params.max_tokens = autoMaxTokens;
            console.log(`[ModelSettings] Auto-setting max_tokens to ${autoMaxTokens} (context: ${contextLength})`);
            if (!state.modelSettings[instanceId]) state.modelSettings[instanceId] = { params: {} };
            if (!state.modelSettings[instanceId].params) state.modelSettings[instanceId].params = {};
            state.modelSettings[instanceId].params.max_tokens = autoMaxTokens;
            saveModelSettingsToStorage();
        } else if (params.max_tokens > contextLength) {
            params.max_tokens = contextLength;
        }
    }

    populateParamSliders(params);

    const panel = document.getElementById('model-settings-panel');
    if (panel) {
        panel.classList.remove('hidden');
        requestAnimationFrame(() => panel.classList.add('visible'));
    }
    state.settingsPanelOpen = true;

    log('Opened settings for: ' + (model.name || instanceId), 'info');
}

export function closeModelSettingsPanel() {
    const panel = document.getElementById('model-settings-panel');
    if (panel) {
        panel.classList.remove('visible');
        setTimeout(() => panel.classList.add('hidden'), 300);
    }
    state.settingsPanelOpen = false;
    state.settingsPanelModelId = null;
}

export function populateParamSliders(params) {
    setSliderValue('param-max-tokens', params.max_tokens, 'max-tokens-value');
    setSliderValue('param-temperature', params.temperature, 'temp-value');
    setSliderValue('param-top-p', params.top_p, 'top-p-value');
    setSliderValue('param-top-k', params.top_k, 'top-k-value');
    setSliderValue('param-repeat-penalty', params.repeat_penalty, 'repeat-penalty-value');
    setSliderValue('param-presence-penalty', params.presence_penalty, 'presence-penalty-value');
    setSliderValue('param-frequency-penalty', params.frequency_penalty, 'frequency-penalty-value');

    const mirostatSelect = document.getElementById('param-mirostat');
    if (mirostatSelect) mirostatSelect.value = params.mirostat;

    setSliderValue('param-mirostat-tau', params.mirostat_tau, 'mirostat-tau-value');
    setSliderValue('param-mirostat-eta', params.mirostat_eta, 'mirostat-eta-value');
    setSliderValue('param-typical-p', params.typical_p, 'typical-p-value');
    setSliderValue('param-tfs-z', params.tfs_z, 'tfs-z-value');

    updateMirostatVisibility();
}

export function setSliderValue(sliderId, value, displayId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    if (slider) slider.value = value;
    if (display) display.textContent = value;
}

export function saveModelSettings() {
    const instanceId = state.settingsPanelModelId;
    if (!instanceId) return;

    const displayName = document.getElementById('model-display-name').value.trim();

    const settings = {
        displayName: displayName || null,
        params: {
            max_tokens: parseInt(document.getElementById('param-max-tokens').value),
            temperature: parseFloat(document.getElementById('param-temperature').value),
            top_p: parseFloat(document.getElementById('param-top-p').value),
            top_k: parseInt(document.getElementById('param-top-k').value),
            repeat_penalty: parseFloat(document.getElementById('param-repeat-penalty').value),
            presence_penalty: parseFloat(document.getElementById('param-presence-penalty').value),
            frequency_penalty: parseFloat(document.getElementById('param-frequency-penalty').value),
            mirostat: parseInt(document.getElementById('param-mirostat').value),
            mirostat_tau: parseFloat(document.getElementById('param-mirostat-tau').value),
            mirostat_eta: parseFloat(document.getElementById('param-mirostat-eta').value),
            typical_p: parseFloat(document.getElementById('param-typical-p').value),
            tfs_z: parseFloat(document.getElementById('param-tfs-z').value),
        }
    };

    state.modelSettings[instanceId] = settings;
    saveModelSettingsToStorage();

    // Lazy import to avoid circular dep
    import('./models.js').then(({ getLoadedModels, renderLoadedModels }) => {
        getLoadedModels().then(models => renderLoadedModels(models));
    });

    log('Settings saved for model', 'success');
}

export function resetToDefaults() {
    populateParamSliders(getGlobalDefaultParams());
}

export function toggleAdvancedParams() {
    const section = document.getElementById('advanced-params');
    const parent = section?.parentElement;
    if (section) section.classList.toggle('collapsed');
    if (parent) parent.classList.toggle('expanded');
}

export function updateMirostatVisibility() {
    const mirostatSelect = document.getElementById('param-mirostat');
    const mode = mirostatSelect ? parseInt(mirostatSelect.value) : 0;
    const mirostatParams = document.querySelectorAll('.mirostat-params');
    mirostatParams.forEach(el => {
        el.classList.toggle('hidden', mode === 0);
    });
}

export function setupParamSliderListeners() {
    const sliders = [
        ['param-max-tokens', 'max-tokens-value'],
        ['param-temperature', 'temp-value'],
        ['param-top-p', 'top-p-value'],
        ['param-top-k', 'top-k-value'],
        ['param-repeat-penalty', 'repeat-penalty-value'],
        ['param-presence-penalty', 'presence-penalty-value'],
        ['param-frequency-penalty', 'frequency-penalty-value'],
        ['param-mirostat-tau', 'mirostat-tau-value'],
        ['param-mirostat-eta', 'mirostat-eta-value'],
        ['param-typical-p', 'typical-p-value'],
        ['param-tfs-z', 'tfs-z-value'],
    ];

    sliders.forEach(([sliderId, displayId]) => {
        const slider = document.getElementById(sliderId);
        if (slider) {
            slider.addEventListener('input', () => {
                const display = document.getElementById(displayId);
                if (display) display.textContent = slider.value;
            });
        }
    });

    const mirostatSelect = document.getElementById('param-mirostat');
    if (mirostatSelect) {
        mirostatSelect.addEventListener('change', updateMirostatVisibility);
    }
}
