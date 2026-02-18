/**
 * Shared state for AgentNate UI
 * All modules import the same object reference - mutations are shared automatically.
 */

// Main application state
export const state = {
    currentModel: null,
    currentModelHasVision: false,
    models: {},
    gpus: [],
    messages: [],           // Legacy — kept for backward compat, active panel mirrors here
    pendingImages: [],      // Legacy — active panel mirrors here
    isGenerating: false,    // Legacy — active panel mirrors here
    currentRequestId: null,
    ws: null,
    n8nTabs: [],
    spawningN8n: false,
    pendingLoads: {},
    errorLoads: {},
    loadInProgress: false,
    unloadingModels: {},
    serverConnected: true,
    lastError: null,
    connectionFailures: 0,
    shutdownReason: null,
    healthInterval: null,
    n8nInterval: null,
    // System Prompt State
    activePromptId: 'default',
    activePromptContent: null,
    customPrompts: [],
    selectedCategory: 'all',
    editingPromptId: null,
    generatingPrompt: false,
    // Model Presets State
    modelPresets: [],
    // Model Settings State
    modelSettings: {},
    settingsPanelOpen: false,
    settingsPanelModelId: null,
    // Agent Mode State (legacy — per-panel overrides these)
    agentMode: false,
    autonomous: true,
    selectedPersonaId: 'auto',
    showAdditionalInstructions: false,
    agentAdvancedOpen: false,
    agentWelcomeShown: false,
    agentAbortId: null,
    agentPlan: null,
    agentThinking: null,
    agentWorkingMemory: null,
    inThinkingBlock: false,
    conversationId: null,
    savedConversations: [],
    // Model Routing State
    routingEnabled: false,
    activeRoutingPresetName: null,
    // PDF RAG State
    pendingPdfs: [],
    loadedPdfFiles: [],
    pdfSessionId: 'default',
    embeddingProvider: null,
    // Multi-Panel Chat State
    panels: {},              // { panelId: PanelState }
    activePanelId: null,     // Currently visible panel
    panelCounter: 0,         // Auto-increment for panel naming
};

/**
 * Create an isolated state object for a chat panel.
 * Each panel is a fully independent chat environment.
 */
export function createPanelState(panelId, name) {
    return {
        panelId,
        name,                                // User-renamable: "Chat 1", "Research", etc.
        messages: [],                        // Conversation history
        conversationId: null,                // Backend conv_id
        agentMode: false,                    // Per-panel agent toggle
        autonomous: true,                    // Per-panel autonomous toggle (default ON)
        selectedPersonaId: 'auto',           // Per-panel persona (auto-routed by backend)
        instanceId: null,                    // Per-panel model override (null = use sidebar global)
        routingPresetId: null,               // Per-panel routing preset (null = use global)
        isGenerating: false,                 // Per-panel generation flag
        agentAbortId: null,                  // Per-panel SSE abort tracker
        currentRequestId: null,              // Per-panel WS request tracker
        pendingImages: [],                   // Per-panel image uploads
        loadedPdfFiles: [],                  // Per-panel PDF context
        additionalInstructions: '',          // Per-panel agent instructions
        agentPlan: null,                     // Per-panel agent plan display
        agentThinking: null,                 // Per-panel thinking block
        agentAdvancedOpen: false,            // Per-panel advanced settings visibility
        showAdditionalInstructions: false,   // Per-panel instructions panel visibility
    };
}

/** Get the currently active panel state, or null if none. */
export function getActivePanel() {
    return state.panels[state.activePanelId] || null;
}

/** Build a DOM ID scoped to a specific panel. */
export function getPanelDomId(panelId, suffix) {
    return `panel-${panelId}-${suffix}`;
}

// API Base (derive from current UI origin so non-8000 ports work)
const DEFAULT_API_ORIGIN = 'http://localhost:8000';
const RUNTIME_API_ORIGIN = (
    typeof window !== 'undefined'
    && window.location
    && /^https?:$/.test(window.location.protocol)
)
    ? window.location.origin
    : DEFAULT_API_ORIGIN;

export const API_ORIGIN = RUNTIME_API_ORIGIN;
export const API = `${API_ORIGIN}/api`;

// n8n Queue system state
export const queueState = {
    mainRunning: false,
    mainPort: 5678,
    workers: [],
};

// LLM Lab state
export const arenaState = {
    model1: null,
    model2: null,
    isRunning: false,
    history: [],
    mode: 'compare',
    // Cancel tracking
    activeRequestIds: [],   // WebSocket request_ids for compare cancel
    abortController: null,  // AbortController for debate SSE cancel
};

// GPU Dashboard state
export const gpuState = {
    autoRefresh: true,
    refreshInterval: null,
    history: {
        utilization: [],
        memory: []
    },
    maxHistoryPoints: 60,
    initialized: false,
};

// Workflow marketplace state (legacy template-based)
export const workflowState = {
    templates: null,
    currentWorkflow: null,
    currentCategory: 'all',
    searchQuery: '',
    selectedTemplate: null,
};

// Real marketplace state
export const marketplaceState = {
    categories: [],
    categoriesLoaded: false,
    selectedCategory: 'all',
    workflows: [],
    searchQuery: '',
    previewWorkflow: null,
    loading: false,
    currentSubtab: 'marketplace',
    deployedWorkflows: [],
};

// Executions tab state
export const executionsState = {
    initialized: false,
    autoRefreshInterval: null,
    filters: {
        workflowId: null,
        status: null,
        since: null,
    },
    executions: [],
    queued: [],
    stats: {},
    workflows: [],
    expandedIds: new Set(),
};

// ComfyUI module state
export const comfyuiState = {
    initialized: false,
    currentSubtab: 'overview',
    moduleDownloaded: false,
    bootstrapped: false,
    comfyuiInstalled: false,
    apiRunning: false,
    instances: [],
    gpus: [],
    activeJob: null,
    models: { registry: [], local: [], categories: [] },
    nodes: { registry: [], installed: [] },
    externalDirs: [],
    gallery: {
        generations: [], total: 0, page: 0, pageSize: 24,
        filters: { query: '', checkpoint: '', tags: '', favorite: false, minRating: 0 },
        sort: 'newest', selectedId: null, stats: {},
    },
};

// TTS module state
export const musicState = {
    initialized: false,
    currentSubtab: 'overview',
    moduleDownloaded: false,
    bootstrapped: false,
    installed: false,
    apiRunning: false,
    models: [],
    workers: [],
    devices: [],
    installStatus: {},       // { modelId: { installed, downloading, ... } }
    installJobs: [],
    selectedModel: null,
    modelParams: {},         // { modelId: paramSchema }
    modelPresets: {},        // { modelId: [preset1, ...] }
    activeGeneration: null,
    clap: { running: false },
    library: {
        items: [],
        filters: { model: '', query: '' },
        sort: 'newest',
        selectedId: null,
    },
};

export const ttsState = {
    initialized: false,
    currentSubtab: 'overview',
    moduleDownloaded: false,
    bootstrapped: false,
    installed: false,
    apiRunning: false,
    models: [],
    workers: [],
    devices: [],
    jobs: [],
    voices: {},
    selectedModel: null,
    activeGeneration: null,
    library: {
        items: [],
        filters: { model: '', voice: '', query: '' },
        sort: 'newest',
        selectedId: null,
    },
};

// Workflow parameter editor state
export const paramEditorState = {
    workflowId: null,
    workflowData: null,
    editableParams: []
};

// Logs
export const logs = [];
export const MAX_LOGS = 500;

// Settings
export let currentSettings = {};
export function setCurrentSettings(val) { currentSettings = val; }

// Storage keys
export const STORAGE_KEY_CUSTOM_PROMPTS = 'agentNate_customPrompts';
export const STORAGE_KEY_ACTIVE_PROMPT = 'agentNate_activePromptId';
export const MODEL_SETTINGS_KEY = 'agentNate_modelSettings';
