/**
 * Onboarding — First-time user experience for AgentNate.
 *
 * - Detects first launch via localStorage flag
 * - Shows welcome modal guiding user to Settings
 * - Provides dynamic welcome message in chat based on system state
 */

import { state } from './state.js';
import { showSettings, switchSettingsTab } from './settings.js';

const ONBOARDING_KEY = 'agentNateOnboardingComplete';

// ==================== First-Run Detection ====================

export function checkOnboarding() {
    if (!localStorage.getItem(ONBOARDING_KEY)) {
        // Skip onboarding if a model is already loaded (e.g. OpenRouter persists across reloads)
        if (state.currentModel) {
            localStorage.setItem(ONBOARDING_KEY, 'true');
            return;
        }
        showWelcomeModal();
    }
}

// ==================== Welcome Modal ====================

export function showWelcomeModal() {
    document.getElementById('welcome-modal').classList.remove('hidden');
}

export function dismissOnboarding() {
    localStorage.setItem(ONBOARDING_KEY, 'true');
    document.getElementById('welcome-modal').classList.add('hidden');
}

export function openSettingsFromOnboarding() {
    dismissOnboarding();
    showSettings();
    // Small delay to ensure settings modal is rendered before switching tab
    setTimeout(() => switchSettingsTab('providers'), 50);
}

// ==================== Dynamic Welcome Message ====================

export function updateWelcomeMessage() {
    const el = document.getElementById('welcome-content');
    if (!el) return;

    // Don't update if there are chat messages (welcome is only shown when empty)
    const container = document.getElementById('chat-messages');
    if (!container) return;
    const msgs = container.querySelectorAll('.message');
    if (msgs.length > 0) return;

    if (state.currentModel) {
        // Model loaded — ready to chat
        el.innerHTML = `
            <h2>AgentNate</h2>
            <p>Type a message below to start chatting.</p>
            <p class="welcome-hint">Tip: Toggle <strong>Agent</strong> mode for tools like web search, code execution, and file management.</p>
        `;
    } else {
        // No model — guide user
        el.innerHTML = `
            <h2>AgentNate</h2>
            <p class="welcome-subtitle">Your local AI platform</p>
            <div class="welcome-steps">
                <div class="welcome-step">
                    <span class="welcome-step-num">1</span>
                    <span>Open <strong>Settings</strong> (bottom-left) and configure a model provider</span>
                </div>
                <div class="welcome-step">
                    <span class="welcome-step-num">2</span>
                    <span>Click the <strong>+</strong> button under Models to load a model</span>
                </div>
                <div class="welcome-step">
                    <span class="welcome-step-num">3</span>
                    <span>Start chatting</span>
                </div>
            </div>
        `;
    }
}
