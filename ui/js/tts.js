/**
 * TTS Module — Frontend controller
 *
 * Manages the TTS tab with 5 subtabs: Overview, Workers, Generate, Jobs, Library.
 * Follows the same pattern as comfyui.js.
 */

import { ttsState, API } from './state.js';
import { log, showToast, updateToast } from './utils.js';

let pollingInterval = null;

// ======================== Hardcoded Voice Lists ========================
// Always available regardless of API/worker state or file downloads.

const KOKORO_VOICES = [
    // American English — Female
    'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore',
    'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    // American English — Male
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael',
    'am_onyx', 'am_puck', 'am_santa',
    // British English — Female
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    // British English — Male
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
    // Japanese
    'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo',
    // Mandarin Chinese
    'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
    'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang',
    // Spanish
    'ef_dora', 'em_alex', 'em_santa',
    // French
    'ff_siwis',
    // Hindi
    'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi',
    // Italian
    'if_sara', 'im_nicola',
    // Brazilian Portuguese
    'pf_dora', 'pm_alex', 'pm_santa',
];

const XTTS_VOICES = [
    'Aaron Dreschner', 'Abrahan Mack', 'Adde Michal', 'Alexandra Hisakawa',
    'Alison Dietlinde', 'Alma María', 'Ana Florence', 'Andrew Chipper',
    'Annmarie Nele', 'Asya Anara', 'Badr Odhiambo', 'Baldur Sanjin',
    'Barbora MacLean', 'Brenda Stern', 'Camilla Holmström', 'Chandra MacFarland',
    'Claribel Dervla', 'Craig Gutsy', 'Daisy Studious', 'Damien Black',
    'Damjan Chapman', 'Dionisio Schuyler', 'Eugenio Mataracı', 'Ferran Simen',
    'Filip Traverse', 'Gilberto Mathias', 'Gitta Nikolina', 'Gracie Wise',
    'Henriette Usha', 'Ige Behringer', 'Ilkin Urbano', 'Kazuhiko Atallah',
    'Kumar Dahl', 'Lidiya Szekeres', 'Lilya Stainthorpe', 'Ludvig Milivoj',
    'Luis Moray', 'Maja Ruoho', 'Marcos Rudaski', 'Narelle Moon',
    'Nova Hogarth', 'Rosemary Okafor', 'Royston Min', 'Sofia Hellen',
    'Suad Qasim', 'Szofi Granger', 'Tammie Ema', 'Tammy Grit',
    'Tanja Adelina', 'Torcull Diarmuid', 'Uta Obando', 'Viktor Eka',
    'Viktor Menelaos', 'Vjollca Johnnie', 'Wulf Carlevaro', 'Xavier Hayasaka',
    'Zacharie Aimilios', 'Zofija Kendrick',
];

// Bark v2 speaker presets — generated programmatically
const BARK_VOICES = (() => {
    const langs = ['de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'tr', 'zh'];
    const voices = ['v2/announcer'];
    for (const lang of langs) {
        for (let i = 0; i <= 9; i++) voices.push(`v2/${lang}_speaker_${i}`);
    }
    return voices;
})();

const QWEN_VOICES = ['Chelsie', 'Ethan'];

// ======================== Per-Model Parameter Config ========================

const LANGUAGES = [
    { v: 'en', l: 'English' }, { v: 'es', l: 'Spanish' }, { v: 'fr', l: 'French' },
    { v: 'de', l: 'German' }, { v: 'it', l: 'Italian' }, { v: 'pt', l: 'Portuguese' },
    { v: 'pl', l: 'Polish' }, { v: 'tr', l: 'Turkish' }, { v: 'ru', l: 'Russian' },
    { v: 'nl', l: 'Dutch' }, { v: 'cs', l: 'Czech' }, { v: 'ar', l: 'Arabic' },
    { v: 'zh', l: 'Chinese' }, { v: 'ja', l: 'Japanese' }, { v: 'ko', l: 'Korean' },
    { v: 'hu', l: 'Hungarian' },
];

const TTS_MODEL_CONFIG = {
    kokoro: {
        label: 'Kokoro 82M',
        desc: 'Lightweight, fast TTS with 54 built-in voices. Best for quick, natural speech.',
        cloning: false,
        builtinVoices: true,
        voices: KOKORO_VOICES,
        refTextRequired: false,
        params: [
            { id: 'speed', label: 'Speed', type: 'number', default: 1.0, min: 0.5, max: 2.0, step: 0.1 },
        ],
    },
    xtts: {
        label: 'XTTS v2',
        desc: 'Multilingual TTS with 58 built-in voices and voice cloning via reference audio.',
        cloning: true,
        builtinVoices: true,
        voices: XTTS_VOICES,
        refTextRequired: false,
        params: [
            { id: 'speed', label: 'Speed', type: 'number', default: 1.0, min: 0.5, max: 2.0, step: 0.1 },
            { id: 'temperature', label: 'Temperature', type: 'number', default: 0.65, min: 0.0, max: 3.0, step: 0.05 },
            { id: 'repetition_penalty', label: 'Rep. Penalty', type: 'number', default: 2.0, min: 0.5, max: 5.0, step: 0.1 },
            { id: 'language', label: 'Language', type: 'select', default: 'en',
              options: LANGUAGES.map(l => ({ v: l.v, l: l.l })) },
            { id: 'mode', label: 'Mode', type: 'select', default: 'built-in',
              options: [{ v: 'built-in', l: 'Built-in voice' }, { v: 'cloned', l: 'Cloned (reference)' }] },
        ],
    },
    bark: {
        label: 'Bark',
        desc: 'Expressive multilingual model. Two-stage generation: semantic tokens + waveform.',
        cloning: false,
        builtinVoices: true,
        voices: BARK_VOICES,
        refTextRequired: false,
        params: [
            { id: 'temperature', label: 'Temperature', type: 'number', default: 0.7, min: 0.0, max: 3.0, step: 0.05,
              hint: 'Semantic token randomness' },
            { id: 'waveform_temperature', label: 'Waveform Temp', type: 'number', default: 0.7, min: 0.0, max: 3.0, step: 0.05,
              hint: 'Acoustic waveform randomness' },
        ],
    },
    fish: {
        label: 'Fish Speech 1.5',
        desc: 'Voice cloning model with nucleus sampling. Reference audio optional.',
        cloning: true,
        builtinVoices: false,
        refTextRequired: false,
        params: [
            { id: 'temperature', label: 'Temperature', type: 'number', default: 0.8, min: 0.0, max: 3.0, step: 0.05 },
            { id: 'repetition_penalty', label: 'Rep. Penalty', type: 'number', default: 1.1, min: 0.5, max: 5.0, step: 0.1 },
            { id: 'top_p', label: 'Top P', type: 'number', default: 0.8, min: 0.0, max: 1.0, step: 0.05,
              hint: 'Nucleus sampling threshold' },
            { id: 'chunk_length', label: 'Chunk Length', type: 'number', default: 200, min: 1, max: 500, step: 1,
              hint: 'Text chunk size (chars)' },
        ],
    },
    chatterbox: {
        label: 'Chatterbox',
        desc: 'Expressive voice cloning with emotion control. Reference audio required.',
        cloning: true,
        builtinVoices: false,
        refTextRequired: false,
        params: [
            { id: 'temperature', label: 'Temperature', type: 'number', default: 0.8, min: 0.0, max: 3.0, step: 0.05 },
            { id: 'repetition_penalty', label: 'Rep. Penalty', type: 'number', default: 1.2, min: 0.5, max: 5.0, step: 0.1 },
            { id: 'exaggeration', label: 'Exaggeration', type: 'number', default: 0.5, min: 0.0, max: 1.0, step: 0.05,
              hint: 'Emotional intensity (0=flat, 1=max)' },
            { id: 'cfg_weight', label: 'CFG Weight', type: 'number', default: 0.5, min: 0.0, max: 1.0, step: 0.05,
              hint: 'Classifier-free guidance strength' },
        ],
    },
    f5: {
        label: 'F5-TTS',
        desc: 'High-quality voice cloning. Requires reference audio AND transcript.',
        cloning: true,
        builtinVoices: false,
        refTextRequired: true,
        params: [
            { id: 'speed', label: 'Speed', type: 'number', default: 1.0, min: 0.5, max: 2.0, step: 0.1 },
            { id: 'seed', label: 'Seed', type: 'number', default: '', min: 0, max: 999999, step: 1,
              hint: 'Random seed (blank=random)' },
        ],
    },
    dia: {
        label: 'Dia 1.6B',
        desc: 'Dialogue model supporting [S1]/[S2] speaker tags. 44.1kHz output.',
        cloning: true,
        builtinVoices: false,
        refTextRequired: false,
        params: [
            { id: 'temperature', label: 'Temperature', type: 'number', default: 1.8, min: 0.0, max: 3.0, step: 0.05 },
            { id: 'cfg_scale', label: 'CFG Scale', type: 'number', default: 3.0, min: 0.0, max: 10.0, step: 0.5,
              hint: 'Classifier-free guidance strength' },
            { id: 'top_p', label: 'Top P', type: 'number', default: 0.90, min: 0.0, max: 1.0, step: 0.05 },
            { id: 'top_k', label: 'Top K', type: 'number', default: 50, min: 1, max: 200, step: 1 },
            { id: 'max_new_tokens', label: 'Max Tokens', type: 'number', default: 3072, min: 1, max: 8192, step: 1,
              hint: 'Max tokens per chunk' },
        ],
    },
    qwen: {
        label: 'Qwen Omni 7B',
        desc: 'Large language model with TTS. Named speaker voices (e.g. Chelsie).',
        cloning: false,
        builtinVoices: true,
        voices: QWEN_VOICES,
        refTextRequired: false,
        params: [
            { id: 'temperature', label: 'Temperature', type: 'number', default: 0.9, min: 0.0, max: 3.0, step: 0.05 },
            { id: 'top_p', label: 'Top P', type: 'number', default: 0.8, min: 0.0, max: 1.0, step: 0.05 },
            { id: 'top_k', label: 'Top K', type: 'number', default: 40, min: 1, max: 200, step: 1 },
        ],
    },
    vibevoice: {
        label: 'VibeVoice',
        desc: 'Voice cloning with classifier-free guidance. Reference audio required.',
        cloning: true,
        builtinVoices: false,
        refTextRequired: false,
        params: [
            { id: 'cfg_scale', label: 'CFG Scale', type: 'number', default: 1.3, min: 0.0, max: 10.0, step: 0.5,
              hint: 'Classifier-free guidance strength' },
        ],
    },
    higgs: {
        label: 'Higgs Audio 3B',
        desc: 'Voice cloning model. Requires reference audio AND transcript.',
        cloning: true,
        builtinVoices: false,
        refTextRequired: true,
        params: [
            { id: 'temperature', label: 'Temperature', type: 'number', default: 0.3, min: 0.0, max: 3.0, step: 0.05 },
            { id: 'top_p', label: 'Top P', type: 'number', default: 0.95, min: 0.0, max: 1.0, step: 0.05 },
            { id: 'top_k', label: 'Top K', type: 'number', default: 50, min: 1, max: 200, step: 1 },
        ],
    },
};

/**
 * Render model-specific parameter fields into #tts-gen-model-params.
 * Also shows/hides voice cloning section based on model config.
 */
function renderModelParams(modelId) {
    const container = document.getElementById('tts-gen-model-params');
    const cloningSection = document.getElementById('tts-gen-cloning-section');
    if (!container) return;

    const cfg = TTS_MODEL_CONFIG[modelId];
    if (!cfg) {
        container.innerHTML = '';
        if (cloningSection) cloningSection.classList.add('hidden');
        return;
    }

    // Build params HTML
    let html = `<div class="tts-gen-params-header">
        <span class="tts-gen-params-label">${cfg.label} Parameters</span>
        <span class="tts-gen-params-desc">${cfg.desc}</span>
    </div>`;

    if (cfg.params.length > 0) {
        html += '<div class="tts-gen-row tts-gen-options">';
        for (const p of cfg.params) {
            html += '<div class="tts-gen-field">';
            const hintSpan = p.hint ? ` <span class="tts-param-hint" title="${p.hint}">?</span>` : '';
            html += `<label>${p.label}${hintSpan}</label>`;

            if (p.type === 'select') {
                html += `<select id="tts-gen-p-${p.id}" class="input-field">`;
                for (const opt of p.options) {
                    const sel = opt.v === p.default ? ' selected' : '';
                    html += `<option value="${opt.v}"${sel}>${opt.l}</option>`;
                }
                html += '</select>';
            } else {
                html += `<input type="number" id="tts-gen-p-${p.id}" class="input-field" value="${p.default}" min="${p.min}" max="${p.max}" step="${p.step}">`;
            }
            html += '</div>';
        }
        html += '</div>';
    } else {
        html += '<div class="tts-gen-row"><div class="tts-gen-field tts-gen-full" style="color:var(--text-secondary);font-size:12px;">No model-specific parameters — uses defaults.</div></div>';
    }

    container.innerHTML = html;

    // Wire up XTTS mode toggle to show/hide cloning section
    const modeSelect = document.getElementById('tts-gen-p-mode');
    if (modeSelect) {
        modeSelect.addEventListener('change', () => updateCloningVisibility(modelId));
    }

    updateCloningVisibility(modelId);
}

/**
 * Show/hide voice dropdown and cloning section based on model type and mode.
 *
 * Three categories:
 *   1. Built-in only (kokoro, bark, qwen) → Voice visible, cloning hidden
 *   2. Cloning only (fish, chatterbox, f5, dia, vibevoice, higgs) → Voice hidden, cloning visible
 *   3. Both (xtts) → Mode toggle decides: built-in → Voice visible + cloning hidden,
 *                                          cloned  → Voice hidden  + cloning visible
 */
function updateCloningVisibility(modelId) {
    const cfg = TTS_MODEL_CONFIG[modelId];
    const cloningSection = document.getElementById('tts-gen-cloning-section');
    const voiceWrap = document.getElementById('tts-gen-voice-wrap');

    if (!cfg) {
        if (cloningSection) cloningSection.classList.add('hidden');
        if (voiceWrap) voiceWrap.style.display = '';
        return;
    }

    // Read actual mode value from DOM (survives browser form restoration)
    const modeSelect = document.getElementById('tts-gen-p-mode');
    const hasModeToggle = cfg.cloning && cfg.builtinVoices; // only XTTS currently
    const isClonedMode = hasModeToggle && modeSelect && modeSelect.value !== 'built-in';

    // Voice dropdown visibility
    if (voiceWrap) {
        if (hasModeToggle) {
            // XTTS: show voice only in built-in mode
            voiceWrap.style.display = isClonedMode ? 'none' : '';
        } else {
            // All others: show if model has built-in voices
            voiceWrap.style.display = cfg.builtinVoices ? '' : 'none';
        }
    }

    // Cloning section visibility
    if (cloningSection) {
        const showCloning = cfg.cloning && (!hasModeToggle || isClonedMode);
        if (showCloning) {
            cloningSection.classList.remove('hidden');
            const reqBadge = document.getElementById('tts-gen-ref-audio-req');
            const reqBadgeText = document.getElementById('tts-gen-ref-text-req');
            if (reqBadge) reqBadge.style.display = ['chatterbox', 'f5', 'vibevoice', 'higgs'].includes(modelId) ? '' : 'none';
            if (reqBadgeText) reqBadgeText.style.display = cfg.refTextRequired ? '' : 'none';
        } else {
            cloningSection.classList.add('hidden');
        }
    }
}

// ======================== Init / Polling ========================

export function initTTS() {
    if (!ttsState.initialized) {
        ttsState.initialized = true;
    }
    refreshTTSStatus();
    startTTSPolling();
}

export function startTTSPolling() {
    if (pollingInterval) return;
    pollingInterval = setInterval(refreshTTSStatus, 5000);
}

export function stopTTSPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

export async function refreshTTSStatus() {
    try {
        const resp = await fetch(`${API}/tts/status`);
        if (!resp.ok) return;
        const data = await resp.json();

        ttsState.moduleDownloaded = data.module_downloaded || false;
        ttsState.bootstrapped = data.bootstrapped || false;
        ttsState.installed = data.installed || false;
        ttsState.apiRunning = data.api_running || false;
        ttsState.workers = data.workers || [];
        ttsState.models = data.models || [];
        ttsState.devices = data.devices || [];

        // Update server badge on Workers tab
        const badge = document.getElementById('tts-server-badge');
        if (badge) {
            if (ttsState.apiRunning) {
                badge.textContent = 'Gateway Running';
                badge.className = 'tts-setup-badge ready';
            } else {
                badge.textContent = 'Gateway Stopped';
                badge.className = 'tts-setup-badge pending';
            }
        }

        renderOverview();

        if (ttsState.currentSubtab === 'workers') renderWorkers();
    } catch (e) {
        // Server not reachable, ignore
    }
}

// ======================== Subtab Switching ========================

export function switchTTSSubtab(subtab) {
    ttsState.currentSubtab = subtab;

    document.querySelectorAll('#tab-tts .tts-subtab').forEach(b => {
        b.classList.toggle('active', b.dataset.subtab === subtab);
    });
    document.querySelectorAll('#tab-tts .tts-subtab-content').forEach(p => {
        const id = p.id.replace('tts-subtab-', '');
        if (id === subtab) {
            p.classList.remove('hidden');
            p.classList.add('active');
        } else {
            p.classList.add('hidden');
            p.classList.remove('active');
        }
    });

    // Lazy-load data for specific subtabs
    if (subtab === 'workers') renderWorkers();
    if (subtab === 'generate') initGenerateSubtab();
    if (subtab === 'jobs') refreshTTSJobs();
    if (subtab === 'library') refreshTTSLibrary();
}

// ======================== Overview Subtab ========================

function renderOverview() {
    const grid = document.getElementById('tts-status-grid');
    if (!grid) return;

    const cards = [
        { label: 'Module', ok: ttsState.moduleDownloaded, text: ttsState.moduleDownloaded ? 'Downloaded' : 'Not Downloaded' },
        { label: 'Bootstrap', ok: ttsState.bootstrapped, text: ttsState.bootstrapped ? 'Ready' : 'Not Done' },
        { label: 'Installed', ok: ttsState.installed, text: ttsState.installed ? 'Installed' : 'Not Installed' },
        { label: 'Server', ok: ttsState.apiRunning, text: ttsState.apiRunning ? `Running (:${8100})` : 'Stopped' },
    ];

    grid.innerHTML = cards.map(c => `
        <div class="tts-status-card ${c.ok ? 'ok' : 'off'}">
            <div class="tts-status-label">${c.label}</div>
            <div class="tts-status-value">${c.text}</div>
        </div>
    `).join('');

    // Update button states
    const btnDownload = document.getElementById('tts-btn-download');
    const btnBootstrap = document.getElementById('tts-btn-bootstrap');
    const btnStart = document.getElementById('tts-btn-start');
    const btnStop = document.getElementById('tts-btn-stop');
    const btnUpdate = document.getElementById('tts-btn-update');
    const btnFullInstall = document.getElementById('tts-btn-full-install');

    if (btnDownload) {
        btnDownload.disabled = ttsState.moduleDownloaded;
        btnDownload.textContent = ttsState.moduleDownloaded ? 'Downloaded' : 'Download Module';
    }
    if (btnBootstrap) {
        btnBootstrap.disabled = !ttsState.moduleDownloaded || ttsState.installed;
        btnBootstrap.textContent = ttsState.installed ? 'Installed' : 'Bootstrap';
    }
    if (btnStart) {
        btnStart.disabled = !ttsState.installed || ttsState.apiRunning;
    }
    if (btnStop) {
        btnStop.disabled = !ttsState.apiRunning;
    }
    if (btnUpdate) {
        btnUpdate.disabled = !ttsState.moduleDownloaded;
    }
    if (btnFullInstall) {
        btnFullInstall.disabled = ttsState.apiRunning;
        if (ttsState.apiRunning) {
            btnFullInstall.textContent = 'TTS Server Running';
        } else if (ttsState.installed) {
            btnFullInstall.textContent = 'Start Server';
        } else {
            btnFullInstall.textContent = 'Set Up Everything';
        }
    }
}

// ======================== Installation ========================

export async function ttsFullInstall() {
    const toast = showToast('Starting TTS setup...', 'loading', 0);

    try {
        // Step 1: Download if needed
        if (!ttsState.moduleDownloaded) {
            updateToast(toast, 'Step 1/3: Cloning TTS server from GitHub...', 'loading');
            const dlResp = await fetch(`${API}/tts/module/download`, { method: 'POST' });
            const dl = await dlResp.json();
            if (!dl.success) {
                updateToast(toast, `Download failed: ${dl.error}`, 'error');
                return;
            }
            await refreshTTSStatus();
        }

        // Step 2: Bootstrap if needed
        if (!ttsState.installed) {
            updateToast(toast, 'Step 2/3: Bootstrapping (Python, venvs, models)... This may take 15-30 minutes.', 'loading');
            const bsResp = await fetch(`${API}/tts/module/bootstrap`, { method: 'POST' });
            const bs = await bsResp.json();
            if (!bs.success) {
                updateToast(toast, `Bootstrap failed: ${bs.error}`, 'error');
                return;
            }
            await refreshTTSStatus();
        }

        // Step 3: Start server
        if (!ttsState.apiRunning) {
            updateToast(toast, 'Step 3/3: Starting TTS API server...', 'loading');
            const startResp = await fetch(`${API}/tts/server/start`, { method: 'POST' });
            const start = await startResp.json();
            if (!start.success) {
                updateToast(toast, `Server start failed: ${start.error}`, 'error');
                return;
            }
            await refreshTTSStatus();
        }

        updateToast(toast, 'TTS server is ready!', 'success');
        log('TTS setup complete', 'success');
    } catch (e) {
        updateToast(toast, `Setup failed: ${e.message}`, 'error');
        log(`TTS setup error: ${e.message}`, 'error');
    }
}

export async function downloadTTSModule() {
    const toast = showToast('Cloning TTS server from GitHub...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/module/download`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'TTS module downloaded!', 'success');
            await refreshTTSStatus();
        } else {
            updateToast(toast, `Download failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Download error: ${e.message}`, 'error');
    }
}

export async function bootstrapTTSModule() {
    const toast = showToast('Bootstrapping TTS (this may take 15-30 minutes)...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/module/bootstrap`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'Bootstrap complete!', 'success');
            await refreshTTSStatus();
        } else {
            updateToast(toast, `Bootstrap failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Bootstrap error: ${e.message}`, 'error');
    }
}

export async function startTTSServer() {
    const toast = showToast('Starting TTS server...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/server/start`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'TTS server started!', 'success');
            await refreshTTSStatus();
        } else {
            updateToast(toast, `Start failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Start error: ${e.message}`, 'error');
    }
}

export async function stopTTSServer() {
    const toast = showToast('Stopping TTS server...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/server/stop`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'TTS server stopped', 'success');
            await refreshTTSStatus();
        } else {
            updateToast(toast, `Stop failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Stop error: ${e.message}`, 'error');
    }
}

export async function updateTTSModule() {
    const toast = showToast('Updating TTS module...', 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/module/update`, { method: 'POST' });
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

// ======================== Workers Subtab ========================

function populateSpawnDropdown() {
    const modelSelect = document.getElementById('tts-spawn-model');
    if (!modelSelect) return;
    let options = '';
    if (ttsState.models.length > 0) {
        // API is running — use runtime model list
        options = ttsState.models.map(m => `<option value="${m.id}">${m.name}</option>`).join('');
    } else if (ttsState._modelInfo && ttsState._modelInfo.length > 0) {
        // API not running — show installed models from model-info
        options = ttsState._modelInfo
            .filter(m => m.env_installed && m.weights_downloaded !== false)
            .map(m => `<option value="${m.id}">${m.display}</option>`).join('');
    }
    if (options) {
        modelSelect.innerHTML = '<option value="">Select model...</option>' + options;
    }
}

async function renderWorkers() {
    const list = document.getElementById('tts-workers-list');
    if (!list) return;

    // Render model setup first (populates _modelInfo for spawn dropdown)
    if (!ttsState._modelInfoLoaded) await renderModelSetup();

    // Populate spawn dropdown from runtime or local model info
    populateSpawnDropdown();

    // Render model status grid (runtime — requires API)
    renderModelStatusGrid();

    // Render whisper section
    if (ttsState.apiRunning) refreshWhisperStatus();

    if (!ttsState.apiRunning) {
        list.innerHTML = '<div class="empty-state">TTS server not running. Start it from the Overview tab.</div>';
        return;
    }

    if (ttsState.workers.length === 0) {
        list.innerHTML = '<div class="empty-state">No workers running. Spawn one above or generate speech to auto-spawn.</div>';
        return;
    }

    list.innerHTML = ttsState.workers.map(w => `
        <div class="tts-worker-row">
            <div class="tts-worker-info">
                <span class="tts-worker-model">${w.model || 'unknown'}</span>
                <span class="tts-worker-detail">Port ${w.port} | ${w.device || 'cpu'}</span>
            </div>
            <div class="tts-worker-status">
                <span class="status-dot ${w.status === 'ready' ? 'online' : w.status === 'busy' ? 'loading' : 'offline'}"></span>
                <span>${w.status || 'unknown'}</span>
            </div>
            <div class="tts-worker-vram">
                ${w.vram_used_mb ? `${Math.round(w.vram_used_mb)}/${Math.round(w.vram_total_mb || 0)} MB` : '-'}
            </div>
            <div class="tts-worker-actions">
                <button class="btn-danger btn-small" onclick="killTTSWorker('${w.worker_id}')">Kill</button>
            </div>
        </div>
    `).join('');
}

// ======================== Model Setup (Install Envs + Download Weights) ========================

async function renderModelSetup() {
    const container = document.getElementById('tts-model-setup');
    if (!container) return;

    if (!ttsState.moduleDownloaded) {
        container.innerHTML = '<div class="empty-state">TTS module not downloaded yet. Use the Overview tab to download it first.</div>';
        return;
    }

    if (!ttsState.installed) {
        container.innerHTML = '<div class="empty-state">TTS module not bootstrapped. Run bootstrap from the Overview tab first.</div>';
        return;
    }

    try {
        const resp = await fetch(`${API}/tts/model-info`);
        const data = await resp.json();
        const models = data.models || [];
        ttsState._modelInfo = models;
        ttsState._modelInfoLoaded = true;

        // Group by env for display
        const envGroups = {};
        for (const m of models) {
            if (!envGroups[m.env_name]) {
                envGroups[m.env_name] = {
                    display: m.env_display,
                    installed: m.env_installed,
                    models: [],
                };
            }
            envGroups[m.env_name].models.push(m);
        }

        // Check overall status
        const anyNotInstalled = Object.values(envGroups).some(g => !g.installed);
        const anyWeightsMissing = models.some(m => m.weights_downloaded === false);

        // Guide callout
        const guide = `<div class="tts-setup-guide">
            <span class="tts-setup-guide-icon">&#9432;</span>
            <div class="tts-setup-guide-text">
                <strong>How model setup works:</strong> Each model needs two things &mdash;
                <strong>1)</strong> a Python environment (shared by related models), then
                <strong>2)</strong> model weights downloaded from HuggingFace.
                Click <strong>Install</strong> on any model to do both steps automatically,
                or use <strong>Install All</strong> above to set up everything at once.
            </div>
        </div>`;

        const groups = Object.entries(envGroups).map(([envName, group]) => {
            // Determine overall group status
            const allWeightsReady = group.models.every(m => m.weights_downloaded !== false);
            const groupAllReady = group.installed && allWeightsReady;

            let envBadge, envBtn;
            if (group.installed) {
                envBadge = '<span class="tts-setup-badge ready">Environment Ready</span>';
                envBtn = '';
            } else {
                envBadge = '<span class="tts-setup-badge pending">Needs Setup</span>';
                envBtn = `<button class="btn-accent btn-small" onclick="installTTSEnv('${envName}')">Install Env</button>`;
            }

            const modelRows = group.models.map(m => {
                const isReady = m.env_installed && m.weights_downloaded !== false;
                let weightsBadge = '';
                let weightsBtn = '';
                let iconClass = '';

                if (m.weights_downloaded === null) {
                    // Auto-download on first use
                    weightsBadge = '<span class="tts-setup-badge auto">Auto-download</span>';
                    iconClass = m.env_installed ? 'ready' : '';
                } else if (m.weights_downloaded) {
                    weightsBadge = '<span class="tts-setup-check">&#10003;</span><span class="tts-setup-badge ready">Ready</span>';
                    iconClass = 'ready';
                } else {
                    weightsBadge = '<span class="tts-setup-badge needs-download">Needs Weights</span>';
                    weightsBtn = group.installed
                        ? `<button class="btn-secondary btn-small" onclick="downloadTTSWeights('${m.id}')">Download${m.weights_size ? ' ' + m.weights_size : ''}</button>`
                        : `<button class="btn-secondary btn-small" disabled title="Install environment first">Download</button>`;
                }

                const installBtn = (!group.installed || (m.weights_downloaded === false))
                    ? `<button class="btn-accent btn-small" onclick="installTTSModel('${m.id}')" title="Install env + download weights">Install</button>`
                    : '';

                return `
                    <div class="tts-setup-model-row${isReady ? ' model-ready' : ''}">
                        <div class="tts-setup-model-icon${iconClass ? ' ' + iconClass : ''}">
                            ${isReady ? '&#10003;' : '&#9834;'}
                        </div>
                        <div class="tts-setup-model-info">
                            <span class="tts-setup-model-name">${m.display}</span>
                            <span class="tts-setup-model-desc">${m.desc}</span>
                        </div>
                        <div class="tts-setup-model-status">
                            ${weightsBadge}
                        </div>
                        <div class="tts-setup-model-actions">
                            ${weightsBtn}
                            ${installBtn}
                        </div>
                    </div>
                `;
            }).join('');

            return `
                <div class="tts-setup-env-group${groupAllReady ? ' all-ready' : ''}">
                    <div class="tts-setup-env-header">
                        <span class="tts-setup-env-name">${group.display}</span>
                        ${envBadge}
                        ${envBtn}
                    </div>
                    <div class="tts-setup-model-list">
                        ${modelRows}
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = guide + groups;

    } catch (e) {
        container.innerHTML = `<div class="empty-state">Failed to load model info: ${e.message}</div>`;
    }
}

export async function refreshTTSModelInfo() {
    ttsState._modelInfoLoaded = false;
    await renderModelSetup();
    populateSpawnDropdown();
    showToast('Model info refreshed', 'success');
}

export async function installTTSEnv(envName) {
    const toast = showToast(`Installing environment ${envName}... This may take 10-20 minutes.`, 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/environments/${envName}/install`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, `Environment installed: ${data.message}`, 'success');
            ttsState._modelInfoLoaded = false;
            await renderModelSetup();
            populateSpawnDropdown();
        } else {
            updateToast(toast, `Install failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Install error: ${e.message}`, 'error');
    }
}

export async function downloadTTSWeights(modelId) {
    const toast = showToast(`Downloading ${modelId} weights... This may take a while.`, 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/model-weights/${modelId}/download`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, data.message, 'success');
            ttsState._modelInfoLoaded = false;
            await renderModelSetup();
            populateSpawnDropdown();
        } else {
            updateToast(toast, `Download failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Download error: ${e.message}`, 'error');
    }
}

export async function installTTSModel(modelId) {
    // Full install: env + weights (like the tkinter "Install" button)
    const models = ttsState._modelInfo || [];
    const model = models.find(m => m.id === modelId);
    if (!model) { showToast('Model not found', 'error'); return; }

    const toast = showToast(`Installing ${model.display}... (env + weights)`, 'loading', 0);

    try {
        // Step 1: Install env if not already
        if (!model.env_installed) {
            updateToast(toast, `Step 1: Installing ${model.env_display} environment...`, 'loading');
            const envResp = await fetch(`${API}/tts/environments/${model.env_name}/install`, { method: 'POST' });
            const envData = await envResp.json();
            if (!envData.success) {
                updateToast(toast, `Env install failed: ${envData.error}`, 'error');
                return;
            }
        }

        // Step 2: Download weights if needed
        if (model.weights_downloaded === false) {
            updateToast(toast, `Step 2: Downloading ${model.display} weights (${model.weights_size || '?'})...`, 'loading');
            const wResp = await fetch(`${API}/tts/model-weights/${modelId}/download`, { method: 'POST' });
            const wData = await wResp.json();
            if (!wData.success) {
                updateToast(toast, `Download failed: ${wData.error}`, 'error');
                return;
            }
        }

        updateToast(toast, `${model.display} installed successfully!`, 'success');
        ttsState._modelInfoLoaded = false;
        await renderModelSetup();
        populateSpawnDropdown();
    } catch (e) {
        updateToast(toast, `Install error: ${e.message}`, 'error');
    }
}

export async function installAllTTSModels() {
    if (!confirm('Install ALL model environments and download all weights? This may take 30-60+ minutes and use significant disk space.')) return;

    const models = ttsState._modelInfo || [];
    if (models.length === 0) {
        await renderModelSetup();
    }

    const toast = showToast('Installing all models...', 'loading', 0);

    // Get unique envs that need installing
    const envsDone = new Set();
    let envCount = 0;
    let weightCount = 0;

    for (const m of (ttsState._modelInfo || [])) {
        // Install env
        if (!m.env_installed && !envsDone.has(m.env_name)) {
            envsDone.add(m.env_name);
            envCount++;
            updateToast(toast, `Installing env: ${m.env_display} (${envCount})...`, 'loading');
            try {
                const resp = await fetch(`${API}/tts/environments/${m.env_name}/install`, { method: 'POST' });
                const data = await resp.json();
                if (!data.success) {
                    log(`Env ${m.env_name} failed: ${data.error}`, 'error');
                }
            } catch (e) {
                log(`Env ${m.env_name} error: ${e.message}`, 'error');
            }
        }

        // Download weights
        if (m.weights_downloaded === false) {
            weightCount++;
            updateToast(toast, `Downloading: ${m.display} (${weightCount})...`, 'loading');
            try {
                const resp = await fetch(`${API}/tts/model-weights/${m.id}/download`, { method: 'POST' });
                const data = await resp.json();
                if (!data.success) {
                    log(`Weights ${m.id} failed: ${data.error}`, 'error');
                }
            } catch (e) {
                log(`Weights ${m.id} error: ${e.message}`, 'error');
            }
        }
    }

    updateToast(toast, `All models installed! (${envCount} envs, ${weightCount} weight downloads)`, 'success');
    ttsState._modelInfoLoaded = false;
    renderModelSetup();
}

// ======================== Model Status Grid ========================

async function renderModelStatusGrid() {
    const grid = document.getElementById('tts-models-grid');
    if (!grid) return;

    if (!ttsState.apiRunning) {
        grid.innerHTML = '<div class="empty-state">TTS server not running.</div>';
        return;
    }

    // Fetch per-model status
    try {
        const resp = await fetch(`${API}/tts/models/status`);
        const data = await resp.json();
        const modelStatus = data.models || data;

        grid.innerHTML = Object.entries(modelStatus).map(([name, info]) => {
            const loaded = info.loaded || false;
            const workerCount = (info.workers || []).length;
            return `
                <div class="tts-model-card ${loaded ? 'loaded' : 'unloaded'}">
                    <div class="tts-model-card-header">
                        <span class="tts-model-name">${name}</span>
                        <span class="tts-model-badge ${loaded ? 'active' : ''}">${loaded ? `${workerCount} worker${workerCount !== 1 ? 's' : ''}` : 'Unloaded'}</span>
                    </div>
                    <div class="tts-model-card-actions">
                        ${!loaded
                            ? `<button class="btn-accent btn-small" onclick="loadTTSModel('${name}')">Load</button>`
                            : `<button class="btn-danger btn-small" onclick="unloadTTSModel('${name}')">Unload</button>
                               <button class="btn-secondary btn-small" onclick="scaleTTSModel('${name}')">Scale</button>`
                        }
                    </div>
                </div>
            `;
        }).join('');
    } catch (e) {
        grid.innerHTML = '<div class="empty-state">Failed to load model status.</div>';
    }
}

export async function refreshTTSModelStatus() {
    await renderModelStatusGrid();
    showToast('Model status refreshed', 'success');
}

export async function scaleTTSModel(model) {
    const count = prompt(`Scale ${model} to how many workers?`, '2');
    if (!count || isNaN(parseInt(count))) return;

    const device = document.getElementById('tts-spawn-device')?.value || 'cuda:1';
    const toast = showToast(`Scaling ${model} to ${count} workers...`, 'loading', 0);

    try {
        const resp = await fetch(`${API}/tts/models/${model}/scale`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ count: parseInt(count), device }),
        });
        const data = await resp.json();
        updateToast(toast, `${model} scaled to ${count} workers`, 'success');
        await refreshTTSStatus();
    } catch (e) {
        updateToast(toast, `Scale error: ${e.message}`, 'error');
    }
}

export async function spawnTTSWorker() {
    const model = document.getElementById('tts-spawn-model')?.value;
    const device = document.getElementById('tts-spawn-device')?.value || 'cuda:1';

    if (!model) {
        showToast('Select a model first', 'error');
        return;
    }

    const toast = showToast(`Spawning ${model} worker on ${device}...`, 'loading', 0);
    try {
        // Auto-start gateway if not running
        if (!ttsState.apiRunning) {
            updateToast(toast, 'Starting TTS gateway server...', 'loading');
            const startResp = await fetch(`${API}/tts/server/start`, { method: 'POST' });
            const startData = await startResp.json();
            if (!startData.success) {
                updateToast(toast, `Gateway start failed: ${startData.error}`, 'error');
                return;
            }
            await refreshTTSStatus();
            updateToast(toast, `Spawning ${model} worker on ${device}...`, 'loading');
        }

        const resp = await fetch(`${API}/tts/workers/spawn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, device }),
        });
        const data = await resp.json();
        if (data.status === 'spawned' || data.worker_id) {
            updateToast(toast, `Worker spawned: ${model} on ${device}`, 'success');
            await refreshTTSStatus();
        } else {
            updateToast(toast, `Spawn failed: ${data.detail || JSON.stringify(data)}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Spawn error: ${e.message}`, 'error');
    }
}

export async function restartTTSServer() {
    const toast = showToast('Restarting TTS server...', 'loading', 0);
    try {
        if (ttsState.apiRunning) {
            updateToast(toast, 'Stopping TTS server...', 'loading');
            await fetch(`${API}/tts/server/stop`, { method: 'POST' });
            // Brief pause for clean shutdown
            await new Promise(r => setTimeout(r, 2000));
        }
        updateToast(toast, 'Starting TTS server...', 'loading');
        const resp = await fetch(`${API}/tts/server/start`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            updateToast(toast, 'TTS server restarted!', 'success');
            await refreshTTSStatus();
        } else {
            updateToast(toast, `Restart failed: ${data.error}`, 'error');
        }
    } catch (e) {
        updateToast(toast, `Restart error: ${e.message}`, 'error');
    }
}

export async function killTTSWorker(workerId) {
    const toast = showToast('Killing worker...', 'loading', 0);
    try {
        await fetch(`${API}/tts/workers/${workerId}`, { method: 'DELETE' });
        updateToast(toast, 'Worker killed', 'success');
        await refreshTTSStatus();
    } catch (e) {
        updateToast(toast, `Kill error: ${e.message}`, 'error');
    }
}

export async function loadTTSModel(model) {
    const toast = showToast(`Loading ${model}...`, 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/models/${model}/load`, { method: 'POST' });
        const data = await resp.json();
        updateToast(toast, `${model} loaded`, 'success');
        await refreshTTSStatus();
    } catch (e) {
        updateToast(toast, `Load error: ${e.message}`, 'error');
    }
}

export async function unloadTTSModel(model) {
    const toast = showToast(`Unloading ${model}...`, 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/models/${model}/unload`, { method: 'POST' });
        const data = await resp.json();
        updateToast(toast, `${model} unloaded`, 'success');
        await refreshTTSStatus();
    } catch (e) {
        updateToast(toast, `Unload error: ${e.message}`, 'error');
    }
}

// ======================== Whisper Management ========================

async function refreshWhisperStatus() {
    const container = document.getElementById('tts-whisper-status');
    if (!container) return;

    if (!ttsState.apiRunning) {
        container.innerHTML = '<div class="empty-state">TTS server not running.</div>';
        return;
    }

    try {
        const resp = await fetch(`${API}/tts/whisper`);
        if (!resp.ok) {
            container.innerHTML = '<div class="empty-state">Whisper status unavailable.</div>';
            return;
        }
        const data = await resp.json();

        const available = data.available_models || {};
        const loaded = data.loaded || [];
        const enabled = data.enabled !== false;

        container.innerHTML = `
            <div class="tts-whisper-info">
                <span class="tts-whisper-enabled">Verification: <strong>${enabled ? 'Enabled' : 'Disabled'}</strong></span>
                <span>Loaded: <strong>${loaded.length > 0 ? loaded.join(', ') : 'None'}</strong></span>
            </div>
            <div class="tts-whisper-models">
                ${Object.entries(available).map(([size, info]) => {
                    const isLoaded = loaded.includes(size);
                    return `
                        <div class="tts-whisper-model-row">
                            <span class="tts-whisper-model-name">${size}</span>
                            <span class="tts-whisper-model-info">${info.params || ''} ${info.vram || ''} ${info.note || ''}</span>
                            ${isLoaded
                                ? `<button class="btn-danger btn-small" onclick="unloadTTSWhisper('${size}')">Unload</button>`
                                : `<button class="btn-accent btn-small" onclick="loadTTSWhisper('${size}')">Load</button>`
                            }
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    } catch (e) {
        container.innerHTML = `<div class="empty-state">Failed to load Whisper status: ${e.message}</div>`;
    }
}

export async function loadTTSWhisper(size) {
    const toast = showToast(`Loading Whisper ${size}...`, 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/whisper/${size}/load`, { method: 'POST' });
        const data = await resp.json();
        updateToast(toast, `Whisper ${size} loaded`, 'success');
        refreshWhisperStatus();
    } catch (e) {
        updateToast(toast, `Load error: ${e.message}`, 'error');
    }
}

export async function unloadTTSWhisper(size) {
    const toast = showToast(`Unloading Whisper ${size}...`, 'loading', 0);
    try {
        const resp = await fetch(`${API}/tts/whisper/${size}/unload`, { method: 'POST' });
        const data = await resp.json();
        updateToast(toast, `Whisper ${size} unloaded`, 'success');
        refreshWhisperStatus();
    } catch (e) {
        updateToast(toast, `Unload error: ${e.message}`, 'error');
    }
}

// ======================== Generate Subtab ========================

function initGenerateSubtab() {
    // Populate model dropdown from available models
    const modelSelect = document.getElementById('tts-gen-model');
    if (modelSelect && ttsState.models.length > 0) {
        const currentVal = modelSelect.value;
        modelSelect.innerHTML = '<option value="">Select model...</option>' +
            ttsState.models.map(m => `<option value="${m.id}">${m.name}</option>`).join('');
        if (currentVal) modelSelect.value = currentVal;
    }
    // If a model is already selected (browser restore or re-entry), trigger voice/param load
    if (modelSelect && modelSelect.value) {
        onTTSModelSelect();
    }
    // Schedule a second visibility sync after browser form restoration completes
    // (browsers restore <select> values asynchronously after DOM load, bypassing onchange)
    setTimeout(() => {
        const m = document.getElementById('tts-gen-model');
        if (m && m.value) updateCloningVisibility(m.value);
    }, 100);
}

/** Populate voice dropdown from hardcoded list, optionally overlay from API. */
function _populateVoiceDropdown(voiceSelect, voices) {
    if (!voices || voices.length === 0) {
        voiceSelect.innerHTML = '<option value="">Default</option>';
    } else {
        voiceSelect.innerHTML = '<option value="">Default</option>' +
            voices.map(v => `<option value="${v}">${v}</option>`).join('');
    }
}

export async function onTTSModelSelect() {
    const model = document.getElementById('tts-gen-model')?.value;
    const voiceSelect = document.getElementById('tts-gen-voice');

    // Render model-specific parameters
    renderModelParams(model);

    if (!model || !voiceSelect) return;

    const cfg = TTS_MODEL_CONFIG[model];

    // Immediately populate from hardcoded voices (always available)
    if (cfg && cfg.voices && cfg.voices.length > 0) {
        _populateVoiceDropdown(voiceSelect, cfg.voices);
        ttsState.voices[model] = cfg.voices;
    } else if (cfg && !cfg.builtinVoices) {
        voiceSelect.innerHTML = '<option value="">No built-in voices (use reference audio)</option>';
    } else {
        voiceSelect.innerHTML = '<option value="">Default</option>';
    }

    // If API is running, try to fetch fresh voices (may have more than hardcoded)
    if (ttsState.apiRunning) {
        try {
            const resp = await fetch(`${API}/tts/voices/${model}`);
            const data = await resp.json();
            const apiVoices = data.voices || [];
            // Only override if API returned more voices than hardcoded
            if (apiVoices.length > (cfg?.voices?.length || 0)) {
                const prev = voiceSelect.value;
                _populateVoiceDropdown(voiceSelect, apiVoices);
                if (prev) voiceSelect.value = prev;
                ttsState.voices[model] = apiVoices;
            }
        } catch (e) {
            // Hardcoded voices already populated, no action needed
        }
    }
}

export async function generateTTS() {
    const model = document.getElementById('tts-gen-model')?.value;
    if (!model) {
        showToast('Select a model first', 'error');
        return;
    }

    const text = document.getElementById('tts-gen-text')?.value?.trim();
    if (!text) {
        showToast('Enter text to synthesize', 'error');
        return;
    }

    const voice = document.getElementById('tts-gen-voice')?.value || undefined;
    const format = document.getElementById('tts-gen-format')?.value || 'wav';
    const de_reverb = parseFloat(document.getElementById('tts-gen-dereverb')?.value);
    const de_ess = parseFloat(document.getElementById('tts-gen-deess')?.value);
    const device = document.getElementById('tts-gen-device')?.value || undefined;
    const auto_retry = parseInt(document.getElementById('tts-gen-retry')?.value) || 3;
    const save_path = document.getElementById('tts-gen-save-path')?.value?.trim() || undefined;
    const verify_whisper = document.getElementById('tts-gen-verify-whisper')?.checked || false;
    const whisper_sel = document.getElementById('tts-gen-whisper-model')?.value || '';
    const whisper_model = verify_whisper && whisper_sel ? whisper_sel : undefined;
    const skip_post_process = document.getElementById('tts-gen-skip-postprocess')?.checked || false;

    // Collect model-specific dynamic parameters
    const cfg = TTS_MODEL_CONFIG[model];
    const dynamicParams = {};
    if (cfg) {
        for (const p of cfg.params) {
            const el = document.getElementById(`tts-gen-p-${p.id}`);
            if (!el) continue;
            if (p.type === 'select') {
                dynamicParams[p.id] = el.value;
            } else {
                const val = el.value;
                if (val !== '' && val !== undefined) {
                    dynamicParams[p.id] = parseFloat(val);
                }
            }
        }
    }

    const body = {
        text, output_format: format, de_reverb, de_ess,
        auto_retry, verify_whisper, skip_post_process,
        ...dynamicParams,
    };
    if (voice) body.voice = voice;
    if (device) body.device = device;
    if (save_path) body.save_path = save_path;
    if (whisper_model) body.whisper_model = whisper_model;

    // Handle reference audio for voice cloning
    const refInput = document.getElementById('tts-gen-ref-audio');
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
    const refText = document.getElementById('tts-gen-ref-text')?.value?.trim();
    if (refText) body.reference_text = refText;

    // Show status
    const resultDiv = document.getElementById('tts-gen-result');
    const statusDiv = document.getElementById('tts-gen-status');
    const playerDiv = document.getElementById('tts-gen-player');
    if (resultDiv) resultDiv.classList.remove('hidden');
    if (statusDiv) statusDiv.innerHTML = '<div class="tts-gen-loading">Generating speech... This may take a while for long text.</div>';
    if (playerDiv) playerDiv.innerHTML = '';

    const btn = document.getElementById('tts-btn-generate');
    if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }

    ttsState.activeGeneration = model;

    // Helper to update status text
    const setStatus = (msg) => { if (statusDiv) statusDiv.innerHTML = `<div class="tts-gen-loading">${msg}</div>`; };
    const setError = (msg) => {
        if (statusDiv) statusDiv.innerHTML = `<div class="tts-gen-error">${msg}</div>`;
        if (btn) { btn.disabled = false; btn.textContent = 'Generate'; }
    };

    try {
        // Step 1: Auto-install model env + weights if needed
        if (!ttsState._modelInfoLoaded) {
            const infoResp = await fetch(`${API}/tts/model-info`);
            const infoData = await infoResp.json();
            ttsState._modelInfo = infoData.models || [];
            ttsState._modelInfoLoaded = true;
        }
        const modelInfo = (ttsState._modelInfo || []).find(m => m.id === model);
        if (modelInfo) {
            if (!modelInfo.env_installed) {
                setStatus(`Installing ${modelInfo.env_display} environment... This may take 10-20 minutes.`);
                const envResp = await fetch(`${API}/tts/environments/${modelInfo.env_name}/install`, { method: 'POST' });
                const envData = await envResp.json();
                if (!envData.success) { setError(`Env install failed: ${envData.error}`); return; }
            }
            if (modelInfo.weights_downloaded === false) {
                setStatus(`Downloading ${modelInfo.display} weights (${modelInfo.weights_size || '?'})...`);
                const wResp = await fetch(`${API}/tts/model-weights/${model}/download`, { method: 'POST' });
                const wData = await wResp.json();
                if (!wData.success) { setError(`Weight download failed: ${wData.error}`); return; }
            }
        }

        // Step 2: Auto-start gateway if not running
        if (!ttsState.apiRunning) {
            setStatus('Starting TTS gateway...');
            const startResp = await fetch(`${API}/tts/server/start`, { method: 'POST' });
            const startData = await startResp.json();
            if (!startData.success) { setError(`Gateway start failed: ${startData.error}`); return; }
            await refreshTTSStatus();
        }

        // Step 3: Auto-spawn worker if none running for this model
        const hasWorker = ttsState.workers.some(w => w.model === model);
        if (!hasWorker) {
            setStatus(`Spawning ${model} worker on ${device || 'cuda:1'}... This may take a few minutes.`);
            const spawnResp = await fetch(`${API}/tts/workers/spawn`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, device: device || 'cuda:1' }),
            });
            const spawnData = await spawnResp.json();
            if (!spawnData.worker_id && spawnData.status !== 'spawned') {
                setError(`Worker spawn failed: ${spawnData.detail || JSON.stringify(spawnData)}`);
                return;
            }
            await refreshTTSStatus();
        }

        // Step 4: Auto-load whisper model if verification enabled
        if (verify_whisper && whisper_model) {
            setStatus(`Checking Whisper ${whisper_model} model...`);
            try {
                const wResp = await fetch(`${API}/tts/whisper`);
                if (wResp.ok) {
                    const wData = await wResp.json();
                    const loaded = wData.loaded || [];
                    if (!loaded.includes(whisper_model)) {
                        setStatus(`Loading Whisper ${whisper_model} model... (auto-downloads if needed)`);
                        const loadResp = await fetch(`${API}/tts/whisper/${whisper_model}/load`, { method: 'POST' });
                        const loadData = await loadResp.json();
                        if (loadData.error) {
                            setError(`Whisper load failed: ${loadData.error}`);
                            return;
                        }
                    }
                }
            } catch (e) { /* whisper check failed, continue anyway */ }
        }

        setStatus('Generating speech... This may take a while for long text.');

        const resp = await fetch(`${API}/tts/generate/${model}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await resp.json();

        if (data.status === 'completed' && data.audio_base64) {
            // Decode and play audio
            const audioBytes = atob(data.audio_base64);
            const audioArray = new Uint8Array(audioBytes.length);
            for (let i = 0; i < audioBytes.length; i++) {
                audioArray[i] = audioBytes.charCodeAt(i);
            }

            const mimeType = { wav: 'audio/wav', mp3: 'audio/mpeg', ogg: 'audio/ogg', flac: 'audio/flac', m4a: 'audio/mp4' }[format] || 'audio/wav';
            const blob = new Blob([audioArray], { type: mimeType });
            const url = URL.createObjectURL(blob);

            if (statusDiv) {
                statusDiv.innerHTML = `
                    <div class="tts-gen-success">
                        Generated successfully! Duration: ${data.duration_sec?.toFixed(1) || '?'}s |
                        ${data.chunks || 1} chunk(s) | ${data.sample_rate || '?'} Hz |
                        Job: ${data.job_id?.substring(0, 8) || '-'}
                    </div>
                `;
            }

            if (playerDiv) {
                playerDiv.innerHTML = `
                    <audio controls autoplay class="tts-audio-player">
                        <source src="${url}" type="${mimeType}">
                    </audio>
                    <div class="tts-gen-actions">
                        <a href="${url}" download="${data.filename || `tts_${model}.${format}`}" class="btn-secondary btn-small">Download</a>
                    </div>
                `;
            }

            log(`TTS generated: ${model}, ${data.duration_sec?.toFixed(1)}s, ${data.chunks} chunks`, 'success');
        } else if (data.status === 'completed' && data.saved_to) {
            // Saved to file, no inline audio
            if (statusDiv) {
                statusDiv.innerHTML = `
                    <div class="tts-gen-success">
                        Saved to: ${data.saved_to}<br>
                        Duration: ${data.duration_sec?.toFixed(1) || '?'}s | ${data.chunks || 1} chunk(s)
                    </div>
                `;
            }
        } else {
            const errMsg = data.detail || data.error || data.message || JSON.stringify(data);
            if (statusDiv) statusDiv.innerHTML = `<div class="tts-gen-error">Generation failed: ${errMsg}</div>`;
            log(`TTS generation failed: ${errMsg}`, 'error');
        }
    } catch (e) {
        if (statusDiv) statusDiv.innerHTML = `<div class="tts-gen-error">Error: ${e.message}</div>`;
        log(`TTS generation error: ${e.message}`, 'error');
    } finally {
        ttsState.activeGeneration = null;
        if (btn) { btn.disabled = false; btn.textContent = 'Generate Speech'; }
    }
}

// ======================== Jobs Subtab ========================

export async function refreshTTSJobs() {
    const list = document.getElementById('tts-jobs-list');
    if (!list) return;

    if (!ttsState.apiRunning) {
        list.innerHTML = '<div class="empty-state">TTS server not running.</div>';
        return;
    }

    try {
        const resp = await fetch(`${API}/tts/jobs`);
        const data = await resp.json();
        const jobs = Array.isArray(data) ? data : (data.jobs || []);
        ttsState.jobs = jobs;
        renderJobs();
    } catch (e) {
        list.innerHTML = `<div class="empty-state">Failed to load jobs: ${e.message}</div>`;
    }
}

function renderJobs() {
    const list = document.getElementById('tts-jobs-list');
    if (!list) return;

    if (ttsState.jobs.length === 0) {
        list.innerHTML = '<div class="empty-state">No jobs found. Generate some speech first!</div>';
        return;
    }

    list.innerHTML = ttsState.jobs.map(job => {
        const progress = job.total_chunks > 0
            ? Math.round((job.chunks_completed / job.total_chunks) * 100)
            : 0;

        const statusClass = {
            completed: 'tts-job-done',
            running: 'tts-job-running',
            failed: 'tts-job-failed',
        }[job.status] || 'tts-job-pending';

        return `
            <div class="tts-job-row ${statusClass}" onclick="viewTTSJob('${job.job_id}')" style="cursor:pointer;" title="Click for details">
                <div class="tts-job-info">
                    <span class="tts-job-model">${job.model || 'unknown'}</span>
                    <span class="tts-job-id">${(job.job_id || '').substring(0, 12)}...</span>
                    <span class="tts-job-time">${job.timestamp || ''}</span>
                </div>
                <div class="tts-job-text" title="${(job.input_text || '').replace(/"/g, '&quot;')}">
                    ${(job.input_text || '').substring(0, 80)}${(job.input_text || '').length > 80 ? '...' : ''}
                </div>
                <div class="tts-job-progress">
                    <div class="tts-progress-bar">
                        <div class="tts-progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <span class="tts-progress-text">${job.chunks_completed || 0}/${job.total_chunks || 0} chunks (${progress}%)</span>
                </div>
                <div class="tts-job-status">
                    <span class="tts-job-badge ${statusClass}">${job.status || 'unknown'}</span>
                    ${job.status === 'running' ? `<button class="btn-danger btn-small" onclick="event.stopPropagation(); cancelTTSJob('${job.model}')">Cancel</button>` : ''}
                    ${job.status === 'failed' ? `<button class="btn-secondary btn-small" onclick="event.stopPropagation(); recoverTTSJob('${job.job_id}')">Recover</button>` : ''}
                </div>
            </div>
        `;
    }).join('');
}

export async function viewTTSJob(jobId) {
    const detailDiv = document.getElementById('tts-job-detail');
    const contentDiv = document.getElementById('tts-job-detail-content');
    const titleDiv = document.getElementById('tts-job-detail-title');
    if (!detailDiv || !contentDiv) return;

    detailDiv.classList.remove('hidden');
    contentDiv.innerHTML = '<div class="tts-gen-loading">Loading job details...</div>';

    try {
        const resp = await fetch(`${API}/tts/jobs/${jobId}`);
        const job = await resp.json();

        if (titleDiv) titleDiv.textContent = `Job: ${jobId.substring(0, 16)}...`;

        const chunks = job.chunks || [];
        const params = job.parameters || job.params || {};

        contentDiv.innerHTML = `
            <div class="tts-job-detail-grid">
                <div class="tts-job-detail-field"><strong>Status:</strong> ${job.status || 'unknown'}</div>
                <div class="tts-job-detail-field"><strong>Model:</strong> ${job.model || '-'}</div>
                <div class="tts-job-detail-field"><strong>Voice:</strong> ${params.voice || 'default'}</div>
                <div class="tts-job-detail-field"><strong>Format:</strong> ${params.output_format || 'wav'}</div>
                <div class="tts-job-detail-field"><strong>Device:</strong> ${params.device || 'auto'}</div>
                <div class="tts-job-detail-field"><strong>Speed:</strong> ${params.speed || 1.0}</div>
                <div class="tts-job-detail-field"><strong>Temperature:</strong> ${params.temperature || 0.65}</div>
                <div class="tts-job-detail-field"><strong>Chunks:</strong> ${job.chunks_completed || 0} / ${job.total_chunks || 0}</div>
                <div class="tts-job-detail-field"><strong>Duration:</strong> ${job.duration_sec ? job.duration_sec.toFixed(1) + 's' : '-'}</div>
                <div class="tts-job-detail-field"><strong>Created:</strong> ${job.timestamp || job.created_at || '-'}</div>
            </div>
            <div class="tts-job-detail-text">
                <strong>Input Text:</strong>
                <div class="tts-job-detail-textbox">${job.input_text || job.text || '-'}</div>
            </div>
            ${chunks.length > 0 ? `
                <div class="tts-job-detail-chunks">
                    <strong>Chunk Progress:</strong>
                    <div class="tts-chunk-list">
                        ${chunks.map((c, i) => `
                            <div class="tts-chunk-row ${c.status === 'completed' ? 'done' : c.status === 'failed' ? 'failed' : 'pending'}">
                                <span>Chunk ${i + 1}</span>
                                <span class="tts-chunk-status">${c.status || 'pending'}</span>
                                <span class="tts-chunk-text">${(c.text || '').substring(0, 50)}${(c.text || '').length > 50 ? '...' : ''}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            ${job.error ? `<div class="tts-gen-error" style="margin-top:8px;">${job.error}</div>` : ''}
            ${job.saved_to ? `<div style="margin-top:8px;"><strong>Saved to:</strong> ${job.saved_to}</div>` : ''}
        `;
    } catch (e) {
        contentDiv.innerHTML = `<div class="tts-gen-error">Failed to load job: ${e.message}</div>`;
    }
}

export function closeTTSJobDetail() {
    const detailDiv = document.getElementById('tts-job-detail');
    if (detailDiv) detailDiv.classList.add('hidden');
}

export async function cancelTTSJob(model) {
    try {
        await fetch(`${API}/tts/generate/${model}/cancel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        showToast('Cancellation requested', 'success');
        setTimeout(refreshTTSJobs, 1000);
    } catch (e) {
        showToast(`Cancel failed: ${e.message}`, 'error');
    }
}

export async function recoverTTSJob(jobId) {
    try {
        await fetch(`${API}/tts/jobs/${jobId}/recover`, { method: 'POST' });
        showToast('Recovery started', 'success');
        setTimeout(refreshTTSJobs, 2000);
    } catch (e) {
        showToast(`Recovery failed: ${e.message}`, 'error');
    }
}

// ======================== Library Subtab ========================

export async function refreshTTSLibrary() {
    const grid = document.getElementById('tts-library-grid');
    if (!grid) return;

    try {
        const resp = await fetch(`${API}/tts/library`);
        const data = await resp.json();
        ttsState.library.items = data.items || [];

        // Populate model filter dropdown
        const modelFilter = document.getElementById('tts-library-model-filter');
        if (modelFilter && modelFilter.options.length <= 1) {
            const models = [...new Set(ttsState.library.items.map(i => i.model))];
            modelFilter.innerHTML = '<option value="">All Models</option>' +
                models.map(m => `<option value="${m}">${m}</option>`).join('');
        }

        renderLibrary();
    } catch (e) {
        grid.innerHTML = `<div class="empty-state">Failed to load library: ${e.message}</div>`;
    }
}

export function filterTTSLibrary() {
    const model = document.getElementById('tts-library-model-filter')?.value || '';
    const query = document.getElementById('tts-library-search')?.value?.toLowerCase() || '';
    const sort = document.getElementById('tts-library-sort')?.value || 'newest';

    ttsState.library.filters.model = model;
    ttsState.library.filters.query = query;
    ttsState.library.sort = sort;

    renderLibrary();
}

function renderLibrary() {
    const grid = document.getElementById('tts-library-grid');
    if (!grid) return;

    let items = [...ttsState.library.items];

    // Apply filters
    const { model, query } = ttsState.library.filters;
    if (model) items = items.filter(i => i.model === model);
    if (query) items = items.filter(i => (i.text || '').toLowerCase().includes(query) || (i.voice || '').toLowerCase().includes(query));

    // Sort
    if (ttsState.library.sort === 'oldest') {
        items.sort((a, b) => (a.timestamp || '').localeCompare(b.timestamp || ''));
    } else if (ttsState.library.sort === 'longest') {
        items.sort((a, b) => (b.duration_sec || 0) - (a.duration_sec || 0));
    }
    // 'newest' is already default from server (reverse sorted)

    if (items.length === 0) {
        grid.innerHTML = '<div class="empty-state">No completed audio files yet. Generate some speech first!</div>';
        return;
    }

    grid.innerHTML = items.map(item => `
        <div class="tts-library-card" data-job="${item.job_id}">
            <div class="tts-library-card-header">
                <span class="tts-library-model">${item.model}</span>
                <span class="tts-library-voice">${item.voice || 'default'}</span>
                <span class="tts-library-duration">${item.duration_sec ? item.duration_sec.toFixed(1) + 's' : '-'}</span>
            </div>
            <div class="tts-library-text">${(item.text || '').substring(0, 120)}${(item.text || '').length > 120 ? '...' : ''}</div>
            <div class="tts-library-player">
                <audio controls preload="none" class="tts-audio-player">
                    <source src="${API}/tts/library/${item.job_id}/audio" type="audio/${item.format === 'mp3' ? 'mpeg' : item.format || 'wav'}">
                </audio>
            </div>
            <div class="tts-library-meta">
                <span>${item.format?.toUpperCase() || 'WAV'} | ${item.chunks || 0} chunks | ${item.sample_rate || '?'} Hz</span>
                <span>${item.timestamp ? new Date(item.timestamp).toLocaleDateString() : ''}</span>
            </div>
            <div class="tts-library-actions">
                <a href="${API}/tts/library/${item.job_id}/audio" download class="btn-secondary btn-small">Download</a>
                <button class="btn-danger btn-small" onclick="deleteTTSLibraryItem('${item.job_id}')">Delete</button>
            </div>
        </div>
    `).join('');
}

export async function deleteTTSLibraryItem(jobId) {
    if (!confirm('Delete this audio file and its job data?')) return;

    try {
        const resp = await fetch(`${API}/tts/library/${jobId}`, { method: 'DELETE' });
        const data = await resp.json();
        if (data.success) {
            showToast('Deleted', 'success');
            ttsState.library.items = ttsState.library.items.filter(i => i.job_id !== jobId);
            renderLibrary();
        } else {
            showToast(`Delete failed: ${data.detail || data.error}`, 'error');
        }
    } catch (e) {
        showToast(`Delete error: ${e.message}`, 'error');
    }
}
