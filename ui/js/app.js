/**
 * AgentNate UI - Entry Point
 * Imports all modules and wires functions to window for HTML onclick handlers.
 */

// Core
import { state, API } from './state.js';
import { log, clearLogs, updateConnectionStatus, apiFetch, updateDebugStatus, updateContextDisplay } from './utils.js';

// Chat
import { connectWebSocket, sendMessage, addMessage, handleInputKeydown, autoResizeInput, clearChat, updateChatUI, cancelInference, panelSendMessage, clearPanelChat } from './chat.js';

// Agent
import { toggleAgentMode, toggleAutonomous, onPersonaChange, toggleAgentAdvanced, loadPersonas, toggleAdditionalInstructions, updateInstructionsButtonState, fillAgentPrompt, stopAgent, togglePanelAgentMode, stopPanelAgent, sendPanelAgentMessage, fillPanelAgentPrompt } from './agent.js';

// Panels
import { initPanelSystem, createNewPanel, switchToPanel, closePanel, renamePanel, onPanelModelChange, onPanelPersonaChange, onPanelRoutingChange, togglePanelAgentAdvanced, togglePanelInstructions, togglePanelAutonomous, toggleManualOverride, refreshAllPanelModelDropdowns, refreshAllPanelRoutingDropdowns, updateAllPanelInputStates } from './panels.js';

// Models
import { refreshLoadedModels, renderLoadedModels, showLoadModal, hideLoadModal, loadAvailableModels, onProviderChange, onModelSelectChange, loadSelectedModel, selectModel, unloadModel, getLoadedModels, cancelModelLoad, dismissErrorLoad, sendWarmupMessage } from './models.js';

// Model Settings
import { initModelSettings, openModelSettings, closeModelSettingsPanel, saveModelSettings, resetToDefaults, toggleAdvancedParams } from './model-settings.js';

// Images
import { triggerImageUpload, handleImageSelect, addPendingImage, removePendingImage, openImageModal, updateVisionUI } from './images.js';

// PDF
import { triggerPdfUpload, triggerPanelPdfUpload, handlePdfSelect, handlePanelPdfSelect, removePdf, clearPdfSession } from './pdf.js';

// Prompts
import { initSystemPrompts, openPromptModal, closePromptModal, switchPromptTab, selectCategory as selectPromptCategory, selectPrompt, createNewPrompt, editPrompt, closePromptEditor, savePromptFromEditor, deleteCurrentPrompt, generatePromptWithAI, saveGeneratedPrompt, useGeneratedPromptOnce } from './prompts.js';

// Presets
import { initModelPresets, loadFromPreset, openPresetsModal, closePresetsModal, confirmDeletePreset, updateContextLengthDisplay, togglePresetNameInput } from './presets.js';

// n8n
import { refreshQueueStatus, openMainAdminTab, toggleWorker, setRunCount, clearRunCounter, closeWorker, openWorkerTab, refreshN8nInstances, stopAllN8n, spawnN8n, stopN8n, openN8nTab, closeN8nTab, retryN8nTab, executeWorker, activateWorker, deactivateWorker, changeWorkerMode, setLoopTarget, enqueueRuns, pauseWorker, toggleParallel } from './n8n.js';

// Settings
import { showSettings, hideSettings, saveSettings, resetSettings, checkOpenRouterBalance, addSearchKey, removeSearchKey, validateSearchKey, toggleSearchSection } from './settings.js';

// Arena
import { updateArenaModels, populateArenaModels, runArenaComparison, handleArenaKeydown, setArenaMode, runDebate, voteDebateWinner, stopComparison, stopDebate, runAutoJudge } from './arena.js';

// GPU
import { initGpuDashboard, toggleGpuAutoRefresh, fetchGpuStats } from './gpu.js';

// Executions
import { initExecutions, refreshExecutions, clearExecutionHistory, applyExecutionFilters, toggleExecutionDetail, stopAutoRefresh as stopExecAutoRefresh } from './executions.js';

// Workflows
import { switchWorkflowSubtab, showDeployedWorkflows, refreshDeployedWorkflows, spawnWorkerFromDeployed, deleteDeployedWorkflow, updateBulkDeleteBar, selectAllDeployed, deselectAllDeployed, deleteSelectedWorkflows, openParamEditor, closeParamEditor, openAdvancedEditor, saveWorkflowParams, loadMarketplaceCategories, selectMarketplaceCategory, searchMarketplace, previewWorkflowById, previewWorkflow, closeWorkflowPreviewModal, copyWorkflowJson, downloadWorkflowJson, deployAndRunWorkflow, quickDeployWorkflowById, quickDeployWorkflow, loadMoreWorkflows, hideTemplateModal, useTemplate, showRecipeModal, showTemplateModal, createFromRecipe, quickAddTemplate, selectCategory as selectTemplateCategory, filterTemplates, generateWorkflow, showWorkflowPreview, closeWorkflowPreview, deployWorkflow, copyWorkflow, downloadWorkflow, stopAllWorkers, openMainAdminWorkflow, toggleImportZone, handleImportFile, pasteWorkflowFromClipboard, validateImportedWorkflow, deployImportedWorkflow, reviewImportWithAgent, showInspectionPanel, closeInspectionPanel, configureAndDeploy, openAgentForWorkflow } from './workflows.js';

// Conversations
import { openConversationHistory, closeConversationHistory, loadConversation, renameConversation, deleteConversation, clearUntitledConversations, saveCurrentConversation, savePanelConversation, closeSaveConversation, confirmSaveConversation } from './conversations.js';

// ComfyUI
import { initComfyUI, refreshComfyUIStatus, startComfyUIPolling, stopComfyUIPolling, switchComfyUISubtab, fullInstall, downloadModule, bootstrapModule, installComfyUI, updateComfyUI, startAPIServer, stopAPIServer, addInstance, removeInstance, startInstance, stopInstance, startAllInstances, stopAllInstances, openComfyUIAdmin, downloadModel, filterModelsByCategory, searchModels, installNode, removeNode, updateAllNodes, addExternalDir, removeExternalDir } from './comfyui.js';

// TTS
import { initTTS, refreshTTSStatus, startTTSPolling, stopTTSPolling, switchTTSSubtab, ttsFullInstall, downloadTTSModule, bootstrapTTSModule, startTTSServer, stopTTSServer, restartTTSServer, updateTTSModule, spawnTTSWorker, killTTSWorker, loadTTSModel, unloadTTSModel, scaleTTSModel, refreshTTSModelStatus, refreshTTSModelInfo, installTTSEnv, downloadTTSWeights, installTTSModel, installAllTTSModels, onTTSModelSelect, generateTTS, refreshTTSJobs, cancelTTSJob, recoverTTSJob, viewTTSJob, closeTTSJobDetail, loadTTSWhisper, unloadTTSWhisper, refreshTTSLibrary, filterTTSLibrary, deleteTTSLibraryItem } from './tts.js';

// Music
import { initMusic, refreshMusicStatus, startMusicPolling, stopMusicPolling, switchMusicSubtab, musicFullInstall, downloadMusicModule, bootstrapMusicModule, startMusicServer, stopMusicServer, updateMusicModule, refreshModelsSubtab, installMusicModel, uninstallMusicModel, spawnMusicWorker, killMusicWorker, onMusicModelSelect, onMusicPresetSelect, generateMusic, refreshMusicLibrary, filterMusicLibrary, deleteMusicLibraryItem } from './music.js';

// GGUF
import { searchGGUF, handleGGUFSearchKeydown, showGGUFFiles, downloadGGUF, cancelGGUFDownload } from './gguf.js';

// Gallery
import { initGallery, galleryApplyFilters, galleryToggleFavoriteFilter, galleryChangeSort, galleryPage, toggleFavorite, rateGeneration, saveGenerationMeta, deleteGeneration, scanExistingImages, openGenerationDetail } from './gallery.js';

// Onboarding
import { checkOnboarding, showWelcomeModal, dismissOnboarding, openSettingsFromOnboarding, updateWelcomeMessage } from './onboarding.js';

// Tabs
import { switchTab } from './tabs.js';

// ==================== Health & Shutdown ====================

async function refreshHealth() {
    if (state.shutdownReason) return;

    try {
        const resp = await apiFetch(`${API}/models/health/all`);
        const health = await resp.json();
        renderProviderStatus(health);
    } catch (e) {
        console.error('Failed to refresh health:', e);
    }

    // Also refresh loaded models to catch API-side loads/unloads
    if (!state.loadInProgress) {
        try {
            await refreshLoadedModels();
        } catch (e) { /* ignore */ }
    }
}

function renderProviderStatus(health) {
    const container = document.getElementById('provider-status');
    if (!container) return;

    const providers = [
        { key: 'llama_cpp', name: 'llama.cpp' },
        { key: 'vllm', name: 'vLLM' },
        { key: 'lm_studio', name: 'LM Studio' },
        { key: 'ollama', name: 'Ollama' },
        { key: 'openrouter', name: 'OpenRouter' },
    ];

    container.innerHTML = providers.map(p => {
        const status = health[p.key]?.status || 'offline';
        return `
            <div class="provider-item">
                <span class="status-dot ${status}"></span>
                <span>${p.name}</span>
            </div>
        `;
    }).join('');
}

async function shutdownSystem() {
    const confirmed = confirm(
        'SHUTDOWN AGENTNATE?\n\n' +
        'This will:\n' +
        '\u2022 Unload ALL loaded models\n' +
        '\u2022 Stop ALL n8n instances\n' +
        '\u2022 Stop ComfyUI instances & API server\n' +
        '\u2022 Stop TTS & Music servers\n' +
        '\u2022 Kill any orphaned processes\n' +
        '\u2022 Exit the server\n\n' +
        'Are you sure?'
    );

    if (!confirmed) return;

    const btn = document.getElementById('shutdown-btn');
    if (btn) {
        btn.classList.add('loading');
        btn.disabled = true;
        btn.textContent = 'Shutting down...';
    }

    log('Initiating system shutdown...', 'warning');

    state.shutdownReason = 'Shutdown in progress...';

    if (state.healthInterval) {
        clearInterval(state.healthInterval);
        state.healthInterval = null;
    }
    if (state.n8nInterval) {
        clearInterval(state.n8nInterval);
        state.n8nInterval = null;
    }
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }

    try {
        const resp = await fetch(`${API}/system/shutdown`, { method: 'POST' });
        const result = await resp.json();

        if (result.success) {
            const summary = `${result.results.models_unloaded} models unloaded, ${result.results.n8n_stopped} n8n stopped`;
            log(`Shutdown complete: ${summary}`, 'success');

            state.shutdownReason = `Graceful shutdown completed. ${summary}`;
            updateConnectionStatus(false, state.shutdownReason);

            if (btn) btn.style.display = 'none';
        } else {
            log('Shutdown failed: ' + result.error, 'error');
            alert('Shutdown failed: ' + result.error);
            resetAfterFailedShutdown(btn);
        }
    } catch (e) {
        if (e.message.includes('Failed to fetch') || e.message.includes('NetworkError')) {
            log('Server has shut down', 'success');
            state.shutdownReason = 'Server has shut down successfully.';
            updateConnectionStatus(false, state.shutdownReason);
            if (btn) btn.style.display = 'none';
        } else {
            log('Shutdown error: ' + e.message, 'error');
            alert('Shutdown error: ' + e.message);
            resetAfterFailedShutdown(btn);
        }
    }
}

function resetAfterFailedShutdown(btn) {
    state.shutdownReason = null;
    state.connectionFailures = 0;

    if (!state.healthInterval) {
        state.healthInterval = setInterval(refreshHealth, 30000);
    }

    if (!state.ws) {
        connectWebSocket();
    }

    if (btn) {
        btn.classList.remove('loading');
        btn.disabled = false;
        btn.textContent = 'Shutdown';
    }
}

// ==================== Expose to window for HTML handlers ====================

Object.assign(window, {
    // Tabs
    switchTab,

    // Chat
    sendMessage,
    clearChat,
    handleInputKeydown,
    autoResizeInput,
    cancelInference,
    panelSendMessage,
    clearPanelChat,

    // Agent
    toggleAgentMode,
    toggleAutonomous,
    onPersonaChange,
    toggleAgentAdvanced,
    toggleAdditionalInstructions,
    updateInstructionsButtonState,
    fillAgentPrompt,
    stopAgent,
    togglePanelAgentMode,
    stopPanelAgent,
    sendPanelAgentMessage,
    fillPanelAgentPrompt,

    // Panels
    createNewPanel,
    switchToPanel,
    closePanel,
    renamePanel,
    onPanelModelChange,
    onPanelPersonaChange,
    onPanelRoutingChange,
    togglePanelAgentAdvanced,
    togglePanelInstructions,
    togglePanelAutonomous,
    toggleManualOverride,

    // Models
    showLoadModal,
    hideLoadModal,
    loadSelectedModel,
    onProviderChange,
    onModelSelectChange,
    selectModel,
    unloadModel,
    cancelModelLoad,
    dismissErrorLoad,

    // Model Settings
    openModelSettings,
    closeModelSettingsPanel,
    saveModelSettings,
    resetToDefaults,
    toggleAdvancedParams,

    // Images
    triggerImageUpload,
    handleImageSelect,
    removePendingImage,
    openImageModal,

    // PDF
    triggerPdfUpload,
    triggerPanelPdfUpload,
    handlePdfSelect,
    handlePanelPdfSelect,
    removePdf,

    // Prompts
    openPromptModal,
    closePromptModal,
    switchPromptTab,
    selectCategory: selectPromptCategory,
    selectPrompt,
    createNewPrompt,
    editPrompt,
    closePromptEditor,
    savePromptFromEditor,
    deleteCurrentPrompt,
    generatePromptWithAI,
    saveGeneratedPrompt,
    useGeneratedPromptOnce,

    // Presets
    loadFromPreset,
    openPresetsModal,
    closePresetsModal,
    confirmDeletePreset,
    togglePresetNameInput,

    // n8n
    openMainAdminTab,
    toggleWorker,
    setRunCount,
    clearRunCounter,
    closeWorker,
    openWorkerTab,
    spawnN8n,
    stopN8n,
    closeN8nTab,
    retryN8nTab,
    stopAllN8n,
    executeWorker,
    activateWorker,
    deactivateWorker,
    changeWorkerMode,
    setLoopTarget,
    enqueueRuns,
    pauseWorker,
    toggleParallel,

    // Settings
    showSettings,
    hideSettings,
    saveSettings,
    resetSettings,
    checkOpenRouterBalance,
    addSearchKey,
    removeSearchKey,
    validateSearchKey,
    toggleSearchSection,

    // Arena / LLM Lab
    updateArenaModels,
    runArenaComparison,
    handleArenaKeydown,
    setArenaMode,
    runDebate,
    voteDebateWinner,
    stopComparison,
    stopDebate,
    runAutoJudge,

    // GPU
    toggleGpuAutoRefresh,

    // Executions
    refreshExecutions,
    clearExecutionHistory,
    applyExecutionFilters,
    toggleExecutionDetail,

    // Workflows - marketplace
    switchWorkflowSubtab,
    refreshDeployedWorkflows,
    spawnWorkerFromDeployed,
    deleteDeployedWorkflow,
    updateBulkDeleteBar,
    selectAllDeployed,
    deselectAllDeployed,
    deleteSelectedWorkflows,
    openParamEditor,
    closeParamEditor,
    openAdvancedEditor,
    saveWorkflowParams,
    selectMarketplaceCategory,
    searchMarketplace,
    previewWorkflowById,
    closeWorkflowPreviewModal,
    copyWorkflowJson,
    downloadWorkflowJson,
    deployAndRunWorkflow,
    quickDeployWorkflowById,
    loadMoreWorkflows,
    stopAllWorkers,
    showInspectionPanel,
    closeInspectionPanel,
    configureAndDeploy,
    openAgentForWorkflow,

    // Workflows - import & suite
    toggleImportZone,
    handleImportFile,
    pasteWorkflowFromClipboard,
    validateImportedWorkflow,
    deployImportedWorkflow,
    reviewImportWithAgent,

    // Workflows - legacy templates
    hideTemplateModal,
    useTemplate,
    showRecipeModal,
    showTemplateModal,
    createFromRecipe,
    quickAddTemplate,
    filterTemplates,
    generateWorkflow,

    // Conversations
    openConversationHistory,
    closeConversationHistory,
    loadConversation,
    renameConversation,
    deleteConversation,
    clearUntitledConversations,
    saveCurrentConversation,
    savePanelConversation,
    closeSaveConversation,
    confirmSaveConversation,

    // ComfyUI
    switchComfyUISubtab,
    fullInstall,
    downloadModule,
    bootstrapModule,
    installComfyUI,
    updateComfyUI,
    startAPIServer,
    stopAPIServer,
    addInstance,
    removeInstance,
    startInstance,
    stopInstance,
    startAllInstances,
    stopAllInstances,
    openComfyUIAdmin,
    downloadModel,
    filterModelsByCategory,
    searchModels,
    installNode,
    removeNode,
    updateAllNodes,
    addExternalDir,
    removeExternalDir,

    // TTS
    switchTTSSubtab,
    ttsFullInstall,
    downloadTTSModule,
    bootstrapTTSModule,
    startTTSServer,
    stopTTSServer,
    restartTTSServer,
    updateTTSModule,
    spawnTTSWorker,
    killTTSWorker,
    loadTTSModel,
    unloadTTSModel,
    scaleTTSModel,
    refreshTTSModelStatus,
    refreshTTSModelInfo,
    installTTSEnv,
    downloadTTSWeights,
    installTTSModel,
    installAllTTSModels,
    onTTSModelSelect,
    generateTTS,
    refreshTTSJobs,
    cancelTTSJob,
    recoverTTSJob,
    viewTTSJob,
    closeTTSJobDetail,
    loadTTSWhisper,
    unloadTTSWhisper,
    refreshTTSLibrary,
    filterTTSLibrary,
    deleteTTSLibraryItem,

    // Music
    switchMusicSubtab,
    musicFullInstall,
    downloadMusicModule,
    bootstrapMusicModule,
    startMusicServer,
    stopMusicServer,
    updateMusicModule,
    refreshModelsSubtab,
    installMusicModel,
    uninstallMusicModel,
    spawnMusicWorker,
    killMusicWorker,
    onMusicModelSelect,
    onMusicPresetSelect,
    generateMusic,
    refreshMusicLibrary,
    filterMusicLibrary,
    deleteMusicLibraryItem,

    // GGUF
    searchGGUF,
    handleGGUFSearchKeydown,
    showGGUFFiles,
    downloadGGUF,
    cancelGGUFDownload,

    // Gallery
    initGallery,
    galleryApplyFilters,
    galleryToggleFavoriteFilter,
    galleryChangeSort,
    galleryPage,
    toggleFavorite,
    rateGeneration,
    saveGenerationMeta,
    deleteGeneration,
    scanExistingImages,
    openGenerationDetail,

    // Health & Shutdown
    refreshHealth,
    shutdownSystem,

    // Onboarding
    showWelcomeModal,
    dismissOnboarding,
    openSettingsFromOnboarding,

    // Logs
    clearLogs,
});

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', async () => {
    log('AgentNate UI initializing...', 'info');

    // Initialize system prompts
    initSystemPrompts();

    // Initialize model presets (from server)
    await initModelPresets();

    // Initialize model settings
    initModelSettings();

    // Load agent personas
    await loadPersonas();

    // Initialize multi-panel chat system (creates first panel)
    initPanelSystem();

    await refreshHealth();
    await refreshLoadedModels();
    await refreshN8nInstances();

    // Update panel dropdowns now that models are loaded
    refreshAllPanelModelDropdowns();
    updateAllPanelInputStates();

    connectWebSocket();

    // Auto-load preset if configured
    try {
        const settingsResp = await fetch(`${API}/settings/ui`);
        const settingsData = await settingsResp.json();
        const autoPresetId = settingsData?.settings?.auto_load_preset;
        if (autoPresetId && state.modelPresets?.find(p => p.id === autoPresetId)) {
            log('Auto-loading preset...', 'info');
            loadFromPreset(autoPresetId);
        }
    } catch (e) {
        console.error('Auto-load preset check failed:', e);
    }

    // Refresh health every 30s
    state.healthInterval = setInterval(refreshHealth, 30000);

    // Refresh n8n instances every 10s (they can be spawned via agent tools)
    state.n8nInterval = setInterval(refreshN8nInstances, 10000);

    // Context length slider event listener
    const contextSlider = document.getElementById('context-length');
    if (contextSlider) {
        contextSlider.addEventListener('input', updateContextLengthDisplay);
    }

    // Keyboard shortcut: Ctrl+Shift+D to toggle debug bar
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.shiftKey && e.key === 'D') {
            e.preventDefault();
            const debugBar = document.getElementById('debug-status');
            if (debugBar) {
                debugBar.classList.toggle('hidden');
                if (!debugBar.classList.contains('hidden')) {
                    updateDebugStatus();
                }
            }
        }

        // Escape key: close the topmost open modal
        if (e.key === 'Escape') {
            // Ordered by likely z-index / overlay priority (topmost first)
            const modalMap = [
                ['prompt-editor-modal',           closePromptEditor],
                ['save-conversation-modal',       closeSaveConversation],
                ['workflow-inspection-modal',      closeInspectionPanel],
                ['param-editor-modal',            closeParamEditor],
                ['workflow-preview-modal',         closeWorkflowPreviewModal],
                ['template-modal',                hideTemplateModal],
                ['prompt-modal',                  closePromptModal],
                ['presets-modal',                 closePresetsModal],
                ['conversation-modal',            closeConversationHistory],
                ['load-modal',                    hideLoadModal],
                ['settings-modal',                hideSettings],
                ['welcome-modal',                 dismissOnboarding],
            ];

            for (const [id, closeFn] of modalMap) {
                const el = document.getElementById(id);
                if (el && !el.classList.contains('hidden') && window.getComputedStyle(el).display !== 'none') {
                    e.preventDefault();
                    closeFn();
                    break;
                }
            }

            // Also close model settings panel if open
            const settingsPanel = document.getElementById('model-settings-panel');
            if (settingsPanel && !settingsPanel.classList.contains('hidden')) {
                e.preventDefault();
                closeModelSettingsPanel();
            }
        }
    });

    // Show onboarding for first-time users
    checkOnboarding();

    log('AgentNate UI ready', 'success');
});
