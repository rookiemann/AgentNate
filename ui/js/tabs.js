/**
 * Tab switching + lazy initialization for AgentNate UI
 */

import { marketplaceState } from './state.js';

let previousTab = 'chat';

export function switchTab(tabId) {
    // Stop auto-refresh when leaving specific tabs
    if (previousTab === 'executions' && tabId !== 'executions') {
        import('./executions.js').then(ex => ex.stopAutoRefresh());
    }
    if (previousTab === 'comfyui' && tabId !== 'comfyui') {
        import('./comfyui.js').then(cu => cu.stopComfyUIPolling());
    }
    if (previousTab === 'tts' && tabId !== 'tts') {
        import('./tts.js').then(t => t.stopTTSPolling());
    }
    if (previousTab === 'music' && tabId !== 'music') {
        import('./music.js').then(m => m.stopMusicPolling());
    }

    // Update tab buttons
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tabId);
    });

    // Update tab panels
    document.querySelectorAll('.tab-panel').forEach(p => {
        p.classList.toggle('active', p.id === `tab-${tabId}`);
    });

    // Lazy initialization for specific tabs
    if (tabId === 'workflows') {
        if (!marketplaceState.categoriesLoaded) {
            import('./workflows.js').then(wf => wf.loadMarketplaceCategories());
        }
    }

    if (tabId === 'arena') {
        import('./arena.js').then(arena => arena.populateArenaModels());
    }

    if (tabId === 'gpu') {
        import('./gpu.js').then(gpu => gpu.initGpuDashboard());
    }

    if (tabId === 'executions') {
        import('./executions.js').then(ex => ex.initExecutions());
    }

    if (tabId === 'comfyui') {
        import('./comfyui.js').then(cu => cu.initComfyUI());
    }

    if (tabId === 'tts') {
        import('./tts.js').then(t => t.initTTS());
    }

    if (tabId === 'music') {
        import('./music.js').then(m => m.initMusic());
    }

    previousTab = tabId;
}
