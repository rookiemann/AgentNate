/**
 * Settings modal for AgentNate UI
 */

import { state, API, currentSettings, setCurrentSettings } from './state.js';
import { log, apiFetch } from './utils.js';
import { gpuState, stopGpuAutoRefresh, fetchGpuStats } from './gpu.js';

function setValue(id, value) {
    const el = document.getElementById(id);
    if (el && value !== undefined && value !== null) {
        el.value = value;
    }
}

function setChecked(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.checked = !!value;
    }
}

function getValue(id) {
    const el = document.getElementById(id);
    return el ? el.value : null;
}

function getChecked(id) {
    const el = document.getElementById(id);
    return el ? el.checked : false;
}

function getOptionalInt(id) {
    const raw = getValue(id);
    if (raw === null || raw === undefined || String(raw).trim() === '') {
        return null;
    }
    const parsed = parseInt(raw, 10);
    return Number.isNaN(parsed) ? null : parsed;
}

let _settingsShowId = 0;   // Monotonic counter — prevents stale async from reopening modal

export async function showSettings() {
    const myId = ++_settingsShowId;
    try {
        const resp = await fetch(`${API}/settings`);
        if (myId !== _settingsShowId) return;   // hideSettings() was called — abort
        const data = await resp.json();
        if (data.success) {
            setCurrentSettings(data.settings);
            populateSettingsForm(data.settings);
        }

        const pathResp = await fetch(`${API}/settings/path/info`);
        if (myId !== _settingsShowId) return;   // stale — abort
        const pathData = await pathResp.json();
        if (pathData.success) {
            document.getElementById('setting-path-display').value = pathData.settings_path;
        }
    } catch (e) {
        console.error('Failed to load settings:', e);
    }

    if (myId !== _settingsShowId) return;   // final guard before showing
    document.getElementById('settings-modal').classList.remove('hidden');

    document.querySelectorAll('.settings-tab').forEach(tab => {
        tab.onclick = () => switchSettingsTab(tab.dataset.tab);
    });

    // Show scroll hint if content overflows
    requestAnimationFrame(() => {
        const panel = document.querySelector('.settings-panel.active');
        if (panel && panel.scrollHeight > panel.clientHeight + 20) {
            panel.classList.add('has-overflow');
            panel.addEventListener('scroll', function onScroll() {
                if (panel.scrollTop > 30) {
                    panel.classList.remove('has-overflow');
                    panel.removeEventListener('scroll', onScroll);
                }
            });
        }
    });
}

export function hideSettings() {
    _settingsShowId++;   // Cancel any in-flight showSettings() async
    document.getElementById('settings-modal').classList.add('hidden');
}

export function switchSettingsTab(tabName) {
    document.querySelectorAll('.settings-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    document.querySelectorAll('.settings-panel').forEach(panel => {
        panel.classList.toggle('active', panel.dataset.panel === tabName);
    });
}

function populateSettingsForm(settings) {
    // Providers - llama.cpp
    setValue('setting-llama-models-dir', settings.providers?.llama_cpp?.models_directory);
    setValue('setting-llama-n-ctx', settings.providers?.llama_cpp?.default_n_ctx);
    setValue('setting-llama-gpu-layers', settings.providers?.llama_cpp?.default_n_gpu_layers);
    setChecked('setting-llama-mmap', settings.providers?.llama_cpp?.use_mmap);
    setChecked('setting-llama-flash-attn', settings.providers?.llama_cpp?.flash_attn);
    setChecked('setting-llama-mlock', settings.providers?.llama_cpp?.use_mlock);

    // Providers - LM Studio
    setValue('setting-lmstudio-url', settings.providers?.lm_studio?.base_url);
    const lmStudioDefaultGpuEl = document.getElementById('setting-lmstudio-default-gpu');
    if (lmStudioDefaultGpuEl) {
        const v = settings.providers?.lm_studio?.default_gpu_index;
        lmStudioDefaultGpuEl.value = (v === null || v === undefined) ? '' : String(v);
    }
    setChecked('setting-lmstudio-enabled', settings.providers?.lm_studio?.enabled);

    // Providers - OpenRouter
    setValue('setting-openrouter-key', settings.providers?.openrouter?.api_key);
    setValue('setting-openrouter-model', settings.providers?.openrouter?.default_model);
    setChecked('setting-openrouter-enabled', settings.providers?.openrouter?.enabled);

    // Services - Search Engines
    const search = settings.services?.search || {};
    setValue('setting-search-default', search.default_engine || 'duckduckgo');
    setChecked('setting-google-enabled', search.google?.enabled);
    setChecked('setting-serper-enabled', search.serper?.enabled);
    renderSearchKeys('google', search.google?.keys || []);
    renderSearchKeys('serper', search.serper?.keys || []);

    // Providers - Ollama
    setValue('setting-ollama-url', settings.providers?.ollama?.base_url);
    setValue('setting-ollama-keepalive', settings.providers?.ollama?.keep_alive);
    setChecked('setting-ollama-enabled', settings.providers?.ollama?.enabled);

    // Providers - vLLM
    setValue('setting-vllm-models-dir', settings.providers?.vllm?.models_directory);
    setValue('setting-vllm-gpu-util', settings.providers?.vllm?.default_gpu_memory_utilization);
    setValue('setting-vllm-timeout', settings.providers?.vllm?.load_timeout);
    setChecked('setting-vllm-enforce-eager', settings.providers?.vllm?.enforce_eager);
    setChecked('setting-vllm-enabled', settings.providers?.vllm?.enabled);

    // UI
    setValue('setting-ui-theme', settings.ui?.theme);
    setChecked('setting-ui-autoscroll', settings.ui?.auto_scroll);
    setChecked('setting-ui-timestamps', settings.ui?.show_timestamps);
    setChecked('setting-ui-highlighting', settings.ui?.code_highlighting);

    // Chat
    setChecked('setting-chat-save', settings.chat?.save_history);
    setValue('setting-chat-max-history', settings.chat?.max_history_messages);

    // GPU
    setValue('setting-gpu-refresh', localStorage.getItem('gpuRefreshInterval') || 2);
    setValue('setting-gpu-warn-threshold', localStorage.getItem('gpuWarnThreshold') || 80);
    setValue('setting-gpu-crit-threshold', localStorage.getItem('gpuCritThreshold') || 95);
    setValue('setting-gpu-temp-warn', localStorage.getItem('gpuTempWarn') || 80);

    // n8n
    setValue('setting-n8n-port', localStorage.getItem('n8nDefaultPort') || 5678);
    setChecked('setting-n8n-autostart', localStorage.getItem('n8nAutoStart') === 'true');

    // Startup - auto-load preset dropdown
    const autoLoadSelect = document.getElementById('setting-auto-load-preset');
    if (autoLoadSelect) {
        autoLoadSelect.innerHTML = '<option value="">None (disabled)</option>';
        (state.modelPresets || []).forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name;
            autoLoadSelect.appendChild(opt);
        });
        setValue('setting-auto-load-preset', settings.ui?.auto_load_preset || '');
    }
    setChecked('setting-startup-remember-size', localStorage.getItem('startupRememberSize') === 'true');
}

export async function checkOpenRouterBalance() {
    const infoEl = document.getElementById('openrouter-balance-info');
    infoEl.classList.remove('hidden', 'success', 'error');
    infoEl.innerHTML = 'Checking...';

    try {
        const response = await fetch(`${API}/models/providers/openrouter/credits`);
        const data = await response.json();

        console.log('OpenRouter credits response:', data);

        if (data.success) {
            const credits = data.credits;
            let html = '';

            if (credits.label) {
                html += `<div class="balance-row"><span class="balance-label">Key</span><span>${credits.label}</span></div>`;
            }

            const usage = credits.usage_total || 0;
            html += `<div class="balance-row"><span class="balance-label">Total Used</span><span class="balance-value">$${usage.toFixed(4)}</span></div>`;

            if (credits.usage_monthly !== undefined && credits.usage_monthly !== null) {
                html += `<div class="balance-row"><span class="balance-label">This Month</span><span>$${credits.usage_monthly.toFixed(4)}</span></div>`;
            }

            if (credits.limit !== null && credits.limit !== undefined) {
                html += `<div class="balance-row"><span class="balance-label">Credit Limit</span><span>$${credits.limit.toFixed(2)}</span></div>`;
                const remaining = credits.limit_remaining || 0;
                html += `<div class="balance-row"><span class="balance-label">Remaining</span><span class="balance-value">$${remaining.toFixed(4)}</span></div>`;
            }

            if (credits.is_free_tier) {
                html += '<div class="balance-row"><small>(Free tier)</small></div>';
            }

            infoEl.innerHTML = html;
            infoEl.classList.add('success');
        } else {
            infoEl.innerHTML = data.error || 'Failed to fetch balance';
            infoEl.classList.add('error');
        }
    } catch (e) {
        infoEl.innerHTML = 'Error: ' + e.message;
        infoEl.classList.add('error');
    }
}

async function reloadProvider(providerName) {
    try {
        const response = await fetch(`${API}/models/providers/reload/${providerName}`, {
            method: 'POST'
        });
        const data = await response.json();
        if (data.success) {
            log(`Provider ${providerName} reloaded`, 'success');
            const { loadAvailableModels } = await import('./models.js');
            await loadAvailableModels();
        } else {
            log(`Failed to reload ${providerName}: ${data.error}`, 'error');
        }
        return data.success;
    } catch (e) {
        log(`Error reloading ${providerName}: ${e.message}`, 'error');
        return false;
    }
}

export async function saveSettings() {
    const btn = document.querySelector('#settings-modal .btn-primary');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const updates = [];

        // Providers - llama.cpp
        updates.push({ key: 'providers.llama_cpp.models_directory', value: getValue('setting-llama-models-dir') });
        updates.push({ key: 'providers.llama_cpp.default_n_ctx', value: parseInt(getValue('setting-llama-n-ctx')) || 4096 });
        updates.push({ key: 'providers.llama_cpp.default_n_gpu_layers', value: parseInt(getValue('setting-llama-gpu-layers')) || 99 });
        updates.push({ key: 'providers.llama_cpp.use_mmap', value: getChecked('setting-llama-mmap') });
        updates.push({ key: 'providers.llama_cpp.flash_attn', value: getChecked('setting-llama-flash-attn') });
        updates.push({ key: 'providers.llama_cpp.use_mlock', value: getChecked('setting-llama-mlock') });

        // Providers - LM Studio
        updates.push({ key: 'providers.lm_studio.base_url', value: getValue('setting-lmstudio-url') });
        updates.push({ key: 'providers.lm_studio.default_gpu_index', value: getOptionalInt('setting-lmstudio-default-gpu') });
        updates.push({ key: 'providers.lm_studio.enabled', value: getChecked('setting-lmstudio-enabled') });

        // Providers - OpenRouter
        updates.push({ key: 'providers.openrouter.api_key', value: getValue('setting-openrouter-key') });
        updates.push({ key: 'providers.openrouter.default_model', value: getValue('setting-openrouter-model') });
        updates.push({ key: 'providers.openrouter.enabled', value: getChecked('setting-openrouter-enabled') });

        // Services - Search Engines
        updates.push({ key: 'services.search.default_engine', value: getValue('setting-search-default') });
        updates.push({ key: 'services.search.google.enabled', value: getChecked('setting-google-enabled') });
        updates.push({ key: 'services.search.google.keys', value: collectSearchKeys('google') });
        updates.push({ key: 'services.search.serper.enabled', value: getChecked('setting-serper-enabled') });
        updates.push({ key: 'services.search.serper.keys', value: collectSearchKeys('serper') });

        // Providers - Ollama
        updates.push({ key: 'providers.ollama.base_url', value: getValue('setting-ollama-url') });
        updates.push({ key: 'providers.ollama.keep_alive', value: getValue('setting-ollama-keepalive') });
        updates.push({ key: 'providers.ollama.enabled', value: getChecked('setting-ollama-enabled') });

        // Providers - vLLM
        updates.push({ key: 'providers.vllm.models_directory', value: getValue('setting-vllm-models-dir') });
        updates.push({ key: 'providers.vllm.default_gpu_memory_utilization', value: parseFloat(getValue('setting-vllm-gpu-util')) || 0.9 });
        updates.push({ key: 'providers.vllm.load_timeout', value: parseInt(getValue('setting-vllm-timeout')) || 600 });
        updates.push({ key: 'providers.vllm.enforce_eager', value: getChecked('setting-vllm-enforce-eager') });
        updates.push({ key: 'providers.vllm.enabled', value: getChecked('setting-vllm-enabled') });

        // UI
        updates.push({ key: 'ui.theme', value: getValue('setting-ui-theme') });
        updates.push({ key: 'ui.auto_scroll', value: getChecked('setting-ui-autoscroll') });
        updates.push({ key: 'ui.show_timestamps', value: getChecked('setting-ui-timestamps') });
        updates.push({ key: 'ui.code_highlighting', value: getChecked('setting-ui-highlighting') });

        // Chat
        updates.push({ key: 'chat.save_history', value: getChecked('setting-chat-save') });
        updates.push({ key: 'chat.max_history_messages', value: parseInt(getValue('setting-chat-max-history')) || 100 });

        // Startup - auto-load preset
        updates.push({ key: 'ui.auto_load_preset', value: getValue('setting-auto-load-preset') || null });

        const openrouterChanged = updates.some(u => u.key.startsWith('providers.openrouter.'));
        const ollamaChanged = updates.some(u => u.key.startsWith('providers.ollama.'));
        const lmStudioChanged = updates.some(u => u.key.startsWith('providers.lm_studio.'));
        const vllmChanged = updates.some(u => u.key.startsWith('providers.vllm.'));

        for (const update of updates) {
            if (update.value !== null && update.value !== undefined) {
                await fetch(`${API}/settings/update`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(update)
                });
            }
        }

        if (openrouterChanged) {
            await reloadProvider('openrouter');
        }
        if (ollamaChanged) {
            await reloadProvider('ollama');
        }
        if (lmStudioChanged) {
            await reloadProvider('lm_studio');
        }
        if (vllmChanged) {
            await reloadProvider('vllm');
        }

        // Save GPU settings to localStorage
        localStorage.setItem('gpuRefreshInterval', getValue('setting-gpu-refresh'));
        localStorage.setItem('gpuWarnThreshold', getValue('setting-gpu-warn-threshold'));
        localStorage.setItem('gpuCritThreshold', getValue('setting-gpu-crit-threshold'));
        localStorage.setItem('gpuTempWarn', getValue('setting-gpu-temp-warn'));

        // Save n8n settings to localStorage
        localStorage.setItem('n8nDefaultPort', getValue('setting-n8n-port'));
        localStorage.setItem('n8nAutoStart', getChecked('setting-n8n-autostart'));

        // Save startup settings to localStorage
        localStorage.setItem('startupRememberSize', getChecked('setting-startup-remember-size'));

        // Apply GPU settings immediately
        applyGpuSettings();

        hideSettings();
    } catch (e) {
        console.error('Failed to save settings:', e);
        alert('Failed to save settings: ' + e.message);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

export async function resetSettings() {
    if (!confirm('Reset all settings to defaults? This cannot be undone.')) {
        return;
    }

    try {
        await fetch(`${API}/settings/reset`, { method: 'POST' });

        localStorage.removeItem('gpuRefreshInterval');
        localStorage.removeItem('gpuWarnThreshold');
        localStorage.removeItem('gpuCritThreshold');
        localStorage.removeItem('gpuTempWarn');
        localStorage.removeItem('n8nDefaultPort');
        localStorage.removeItem('n8nAutoStart');
        localStorage.removeItem('startupRememberSize');

        await showSettings();
    } catch (e) {
        console.error('Failed to reset settings:', e);
        alert('Failed to reset settings: ' + e.message);
    }
}

export function applyGpuSettings() {
    const newInterval = parseInt(localStorage.getItem('gpuRefreshInterval')) || 2;
    if (gpuState.refreshInterval) {
        stopGpuAutoRefresh();
        gpuState.refreshInterval = setInterval(() => {
            const gpuTab = document.getElementById('tab-gpu');
            if (gpuTab && gpuTab.classList.contains('active')) {
                fetchGpuStats();
            }
        }, newInterval * 1000);
    }
}

// ==================== Search Key Management ====================

function renderSearchKeys(engine, keys) {
    const container = document.getElementById(`${engine}-keys-container`);
    if (!container) return;
    container.innerHTML = '';

    if (keys.length === 0) {
        container.innerHTML = '<div class="empty-state-small">No keys added yet</div>';
        return;
    }

    keys.forEach((key, idx) => {
        const row = document.createElement('div');
        row.className = 'search-key-row';
        row.dataset.engine = engine;
        row.dataset.index = idx;

        let fieldsHtml = `
            <input type="text" class="search-key-label" placeholder="Label" value="${esc(key.label || '')}" />
            <input type="password" class="search-key-value" placeholder="API Key" value="${esc(key.api_key || '')}" />
        `;

        if (engine === 'google') {
            fieldsHtml += `<input type="text" class="search-key-cx" placeholder="Search Engine ID (cx)" value="${esc(key.cx || '')}" />`;
        }

        row.innerHTML = `
            ${fieldsHtml}
            <button class="btn-secondary btn-small" onclick="validateSearchKey('${engine}', ${idx})">Test</button>
            <button class="btn-danger btn-small" onclick="removeSearchKey('${engine}', ${idx})">&#215;</button>
            <span class="search-key-status" id="search-key-status-${engine}-${idx}"></span>
        `;
        container.appendChild(row);
    });
}

export function addSearchKey(engine) {
    const container = document.getElementById(`${engine}-keys-container`);
    if (!container) return;

    // Clear empty state
    const empty = container.querySelector('.empty-state-small');
    if (empty) empty.remove();

    const idx = container.querySelectorAll('.search-key-row').length;
    const row = document.createElement('div');
    row.className = 'search-key-row';
    row.dataset.engine = engine;
    row.dataset.index = idx;

    let fieldsHtml = `
        <input type="text" class="search-key-label" placeholder="Label (e.g. Account 1)" value="" />
        <input type="password" class="search-key-value" placeholder="API Key" value="" />
    `;

    if (engine === 'google') {
        fieldsHtml += `<input type="text" class="search-key-cx" placeholder="Search Engine ID (cx)" value="" />`;
    }

    row.innerHTML = `
        ${fieldsHtml}
        <button class="btn-secondary btn-small" onclick="validateSearchKey('${engine}', ${idx})">Test</button>
        <button class="btn-danger btn-small" onclick="removeSearchKey('${engine}', ${idx})">&#215;</button>
        <span class="search-key-status" id="search-key-status-${engine}-${idx}"></span>
    `;
    container.appendChild(row);
}

export function removeSearchKey(engine, idx) {
    const container = document.getElementById(`${engine}-keys-container`);
    if (!container) return;
    const rows = container.querySelectorAll('.search-key-row');
    if (rows[idx]) rows[idx].remove();

    // Re-index remaining rows
    container.querySelectorAll('.search-key-row').forEach((row, i) => {
        row.dataset.index = i;
        const testBtn = row.querySelector('button');
        if (testBtn) testBtn.setAttribute('onclick', `validateSearchKey('${engine}', ${i})`);
        const removeBtn = row.querySelectorAll('button')[1];
        if (removeBtn) removeBtn.setAttribute('onclick', `removeSearchKey('${engine}', ${i})`);
        const status = row.querySelector('.search-key-status');
        if (status) status.id = `search-key-status-${engine}-${i}`;
    });

    if (container.children.length === 0) {
        container.innerHTML = '<div class="empty-state-small">No keys added yet</div>';
    }
}

export async function validateSearchKey(engine, idx) {
    const container = document.getElementById(`${engine}-keys-container`);
    if (!container) return;
    const row = container.querySelectorAll('.search-key-row')[idx];
    if (!row) return;

    const apiKey = row.querySelector('.search-key-value')?.value || '';
    const cx = row.querySelector('.search-key-cx')?.value || '';
    const statusEl = document.getElementById(`search-key-status-${engine}-${idx}`);

    if (!apiKey) {
        if (statusEl) statusEl.innerHTML = '<span style="color:var(--error)">No key entered</span>';
        return;
    }

    if (statusEl) statusEl.innerHTML = '<span class="spinner-small"></span> Testing...';

    try {
        const resp = await fetch(`${API}/settings/validate-search-key`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ engine, api_key: apiKey, cx }),
        });
        const data = await resp.json();

        if (data.valid) {
            if (statusEl) statusEl.innerHTML = '<span style="color:var(--success)">&#10003; Valid</span>';
        } else {
            if (statusEl) statusEl.innerHTML = `<span style="color:var(--error)">&#10007; ${esc(data.error || 'Invalid')}</span>`;
        }
    } catch (e) {
        if (statusEl) statusEl.innerHTML = `<span style="color:var(--error)">Error: ${esc(e.message)}</span>`;
    }
}

function collectSearchKeys(engine) {
    const container = document.getElementById(`${engine}-keys-container`);
    if (!container) return [];

    const keys = [];
    container.querySelectorAll('.search-key-row').forEach(row => {
        const apiKey = row.querySelector('.search-key-value')?.value || '';
        if (!apiKey) return; // Skip empty keys

        const entry = {
            api_key: apiKey,
            label: row.querySelector('.search-key-label')?.value || '',
        };

        if (engine === 'google') {
            entry.cx = row.querySelector('.search-key-cx')?.value || '';
        }

        keys.push(entry);
    });

    return keys;
}

export function toggleSearchSection(engine) {
    // Visual feedback only - actual enable/disable is on save
}

function esc(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
