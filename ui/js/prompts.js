/**
 * System prompts - library, custom, AI generation for AgentNate UI
 */

import { state, API, STORAGE_KEY_CUSTOM_PROMPTS, STORAGE_KEY_ACTIVE_PROMPT } from './state.js';
import { log, escapeHtml } from './utils.js';
import { PROMPT_CATEGORIES, SYSTEM_PROMPTS_LIBRARY, PROMPT_GENERATOR_SYSTEM_PROMPT } from '../prompts-data.js';

export function initSystemPrompts() {
    loadCustomPrompts();

    const savedPromptId = localStorage.getItem(STORAGE_KEY_ACTIVE_PROMPT);
    if (savedPromptId) {
        state.activePromptId = savedPromptId;
    } else {
        state.activePromptId = 'default';
    }

    const prompt = getPromptById(state.activePromptId);
    if (prompt) {
        state.activePromptContent = prompt.content;
    } else {
        state.activePromptId = 'default';
        const defaultPrompt = getPromptById('default');
        state.activePromptContent = defaultPrompt ? defaultPrompt.content : null;
    }

    updatePromptDisplay();

    log('System prompts initialized: ' + state.activePromptId, 'info');
    console.log('[SystemPrompt] Initialized with:', state.activePromptId);
}

export function loadCustomPrompts() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY_CUSTOM_PROMPTS);
        if (saved) {
            state.customPrompts = JSON.parse(saved);
            console.log('[SystemPrompt] Loaded', state.customPrompts.length, 'custom prompts');
        }
    } catch (e) {
        console.error('[SystemPrompt] Failed to load custom prompts:', e);
        state.customPrompts = [];
    }
}

export function saveCustomPrompts() {
    try {
        localStorage.setItem(STORAGE_KEY_CUSTOM_PROMPTS, JSON.stringify(state.customPrompts));
        console.log('[SystemPrompt] Saved', state.customPrompts.length, 'custom prompts');
    } catch (e) {
        console.error('[SystemPrompt] Failed to save custom prompts:', e);
    }
}

export function getPromptById(promptId) {
    if (SYSTEM_PROMPTS_LIBRARY) {
        const libraryPrompt = SYSTEM_PROMPTS_LIBRARY.find(p => p.id === promptId);
        if (libraryPrompt) return libraryPrompt;
    }

    const customPrompt = state.customPrompts.find(p => p.id === promptId);
    if (customPrompt) return customPrompt;

    return null;
}

export function getActiveSystemPrompt() {
    return state.activePromptContent || null;
}

export function selectPrompt(promptId) {
    const prompt = getPromptById(promptId);
    if (!prompt) {
        console.warn('[SystemPrompt] Prompt not found:', promptId);
        return;
    }

    state.activePromptId = promptId;
    state.activePromptContent = prompt.content;

    localStorage.setItem(STORAGE_KEY_ACTIVE_PROMPT, promptId);

    updatePromptDisplay();
    closePromptModal();

    log('System prompt changed to: ' + prompt.name, 'info');
    console.log('[SystemPrompt] Selected:', promptId, '-', prompt.content.substring(0, 50) + '...');
}

export function updatePromptDisplay() {
    const nameEl = document.getElementById('current-prompt-name');
    const iconEl = document.getElementById('current-prompt-icon');

    const prompt = getPromptById(state.activePromptId);
    if (prompt) {
        if (nameEl) nameEl.textContent = prompt.name;
        if (iconEl) iconEl.textContent = prompt.icon || '\u{1F916}';
    } else {
        if (nameEl) nameEl.textContent = 'No Prompt';
        if (iconEl) iconEl.textContent = '\u274C';
    }
}

export function openPromptModal() {
    const modal = document.getElementById('prompt-modal');
    if (modal) {
        modal.classList.remove('hidden');
        renderPromptCategories();
        renderLibraryPrompts();
    }
}

export function closePromptModal() {
    const modal = document.getElementById('prompt-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

export function switchPromptTab(tabId) {
    document.querySelectorAll('.prompt-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    document.querySelectorAll('.prompt-tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    const activeContent = document.getElementById(`prompt-tab-${tabId}`);
    if (activeContent) {
        activeContent.classList.remove('hidden');
    }

    if (tabId === 'library') {
        renderPromptCategories();
        renderLibraryPrompts();
    } else if (tabId === 'custom') {
        renderCustomPrompts();
    }
}

export function renderPromptCategories() {
    const container = document.getElementById('prompt-categories');
    if (!container || !PROMPT_CATEGORIES) return;

    container.innerHTML = PROMPT_CATEGORIES.map(cat => `
        <button class="category-btn ${state.selectedCategory === cat.id ? 'active' : ''}"
                data-category="${cat.id}"
                onclick="selectCategory('${cat.id}')">
            <span class="category-icon">${cat.icon}</span>
            ${cat.name}
        </button>
    `).join('');
}

export function selectCategory(categoryId) {
    state.selectedCategory = categoryId;
    renderPromptCategories();
    renderLibraryPrompts();
}

export function renderLibraryPrompts() {
    const container = document.getElementById('library-prompt-list');
    if (!container || !SYSTEM_PROMPTS_LIBRARY) return;

    let prompts = SYSTEM_PROMPTS_LIBRARY;

    if (state.selectedCategory !== 'all') {
        prompts = prompts.filter(p => p.category === state.selectedCategory);
    }

    if (prompts.length === 0) {
        container.innerHTML = '<div class="empty-state">No prompts in this category</div>';
        return;
    }

    container.innerHTML = prompts.map(prompt => `
        <div class="prompt-card ${state.activePromptId === prompt.id ? 'active' : ''}"
             onclick="selectPrompt('${prompt.id}')">
            <div class="prompt-card-header">
                <span class="prompt-card-icon">${prompt.icon}</span>
                <span class="prompt-card-name">${escapeHtml(prompt.name)}</span>
            </div>
            <div class="prompt-card-description">${escapeHtml(prompt.description)}</div>
        </div>
    `).join('');
}

export function renderCustomPrompts() {
    const container = document.getElementById('custom-prompt-list');
    if (!container) return;

    if (state.customPrompts.length === 0) {
        container.innerHTML = '<div class="empty-state">No custom prompts yet. Create one!</div>';
        return;
    }

    container.innerHTML = state.customPrompts.map(prompt => `
        <div class="prompt-card ${state.activePromptId === prompt.id ? 'active' : ''}">
            <div class="prompt-card-header">
                <span class="prompt-card-icon">${prompt.icon || '\u{1F4DD}'}</span>
                <span class="prompt-card-name">${escapeHtml(prompt.name)}</span>
            </div>
            <div class="prompt-card-description">${escapeHtml(prompt.description || prompt.content.substring(0, 100) + '...')}</div>
            <div class="prompt-card-actions">
                <button class="btn-secondary" onclick="event.stopPropagation(); editPrompt('${prompt.id}')">Edit</button>
                <button class="btn-primary" onclick="event.stopPropagation(); selectPrompt('${prompt.id}')">Use</button>
            </div>
        </div>
    `).join('');
}

export function createNewPrompt() {
    state.editingPromptId = null;

    document.getElementById('prompt-editor-title').textContent = 'Create New Prompt';
    document.getElementById('edit-prompt-name').value = '';
    document.getElementById('edit-prompt-category').value = 'custom';
    document.getElementById('edit-prompt-icon').value = '\u{1F4DD}';
    document.getElementById('edit-prompt-content').value = '';
    document.getElementById('delete-prompt-btn').style.display = 'none';

    document.getElementById('prompt-editor-modal').classList.remove('hidden');
}

export function editPrompt(promptId) {
    const prompt = state.customPrompts.find(p => p.id === promptId);
    if (!prompt) return;

    state.editingPromptId = promptId;

    document.getElementById('prompt-editor-title').textContent = 'Edit Prompt';
    document.getElementById('edit-prompt-name').value = prompt.name;
    document.getElementById('edit-prompt-category').value = prompt.category || 'custom';
    document.getElementById('edit-prompt-icon').value = prompt.icon || '\u{1F4DD}';
    document.getElementById('edit-prompt-content').value = prompt.content;
    document.getElementById('delete-prompt-btn').style.display = 'inline-block';

    document.getElementById('prompt-editor-modal').classList.remove('hidden');
}

export function closePromptEditor() {
    document.getElementById('prompt-editor-modal').classList.add('hidden');
    state.editingPromptId = null;
}

export function savePromptFromEditor() {
    const name = document.getElementById('edit-prompt-name').value.trim();
    const category = document.getElementById('edit-prompt-category').value;
    const icon = document.getElementById('edit-prompt-icon').value.trim() || '\u{1F4DD}';
    const content = document.getElementById('edit-prompt-content').value.trim();

    if (!name || !content) {
        alert('Please enter a name and prompt content.');
        return;
    }

    if (state.editingPromptId) {
        const idx = state.customPrompts.findIndex(p => p.id === state.editingPromptId);
        if (idx >= 0) {
            state.customPrompts[idx] = {
                ...state.customPrompts[idx],
                name,
                category,
                icon,
                content,
                description: content.substring(0, 100) + '...'
            };

            if (state.activePromptId === state.editingPromptId) {
                state.activePromptContent = content;
                updatePromptDisplay();
            }
        }
    } else {
        const newPrompt = {
            id: 'custom-' + Date.now(),
            name,
            category,
            icon,
            content,
            description: content.substring(0, 100) + '...',
            isBuiltIn: false
        };
        state.customPrompts.push(newPrompt);
    }

    saveCustomPrompts();
    closePromptEditor();
    renderCustomPrompts();
    log('Prompt saved: ' + name, 'success');
}

export function deleteCurrentPrompt() {
    if (!state.editingPromptId) return;

    if (!confirm('Are you sure you want to delete this prompt?')) return;

    const idx = state.customPrompts.findIndex(p => p.id === state.editingPromptId);
    if (idx >= 0) {
        const deletedName = state.customPrompts[idx].name;
        state.customPrompts.splice(idx, 1);
        saveCustomPrompts();

        if (state.activePromptId === state.editingPromptId) {
            selectPrompt('default');
        }

        log('Prompt deleted: ' + deletedName, 'info');
    }

    closePromptEditor();
    renderCustomPrompts();
}

export async function generatePromptWithAI() {
    const descriptionInput = document.getElementById('prompt-description');
    const description = descriptionInput.value.trim();

    if (!description) {
        alert('Please describe what you want the assistant to do.');
        return;
    }

    if (!state.currentModel) {
        alert('Please load a model first to generate prompts with AI.');
        return;
    }

    if (state.generatingPrompt) {
        return;
    }

    state.generatingPrompt = true;
    const generateBtn = document.getElementById('generate-prompt-btn');
    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating...';

    console.log('[SystemPrompt] Generating prompt for:', description);
    log('Generating system prompt with AI...', 'info');

    try {
        const metaPrompt = PROMPT_GENERATOR_SYSTEM_PROMPT ||
            'You are a system prompt engineer. Write a concise, effective system prompt based on the user description. Output only the system prompt, no explanation.';

        const messages = [
            { role: 'system', content: metaPrompt },
            { role: 'user', content: description }
        ];

        const response = await fetch(`${API}/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                instance_id: state.currentModel,
                messages: messages,
                max_tokens: 1024,
                temperature: 0.7,
                stream: false
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        const generatedPrompt = data.content.trim();

        document.getElementById('generated-prompt-text').value = generatedPrompt;
        document.getElementById('generated-prompt-name').value = '';
        document.getElementById('generated-prompt-result').classList.remove('hidden');

        log('System prompt generated successfully', 'success');
        console.log('[SystemPrompt] Generated:', generatedPrompt.substring(0, 100) + '...');

    } catch (error) {
        console.error('[SystemPrompt] Generation failed:', error);
        log('Failed to generate prompt: ' + error.message, 'error');
        alert('Failed to generate prompt: ' + error.message);
    } finally {
        state.generatingPrompt = false;
        generateBtn.disabled = false;
        generateBtn.textContent = '\u2728 Generate with AI';
    }
}

export function saveGeneratedPrompt() {
    const name = document.getElementById('generated-prompt-name').value.trim();
    const content = document.getElementById('generated-prompt-text').value.trim();

    if (!name) {
        alert('Please enter a name for this prompt.');
        return;
    }

    if (!content) {
        alert('Generated prompt is empty.');
        return;
    }

    const newPrompt = {
        id: 'custom-' + Date.now(),
        name,
        category: 'custom',
        icon: '\u2728',
        content,
        description: content.substring(0, 100) + '...',
        isBuiltIn: false
    };

    state.customPrompts.push(newPrompt);
    saveCustomPrompts();

    selectPrompt(newPrompt.id);

    document.getElementById('prompt-description').value = '';
    document.getElementById('generated-prompt-result').classList.add('hidden');

    log('Generated prompt saved: ' + name, 'success');
}

export function useGeneratedPromptOnce() {
    const content = document.getElementById('generated-prompt-text').value.trim();

    if (!content) {
        alert('Generated prompt is empty.');
        return;
    }

    state.activePromptId = 'temporary';
    state.activePromptContent = content;

    const nameEl = document.getElementById('current-prompt-name');
    const iconEl = document.getElementById('current-prompt-icon');
    if (nameEl) nameEl.textContent = 'Temporary Prompt';
    if (iconEl) iconEl.textContent = '\u2728';

    document.getElementById('prompt-description').value = '';
    document.getElementById('generated-prompt-result').classList.add('hidden');
    closePromptModal();

    log('Using temporary generated prompt', 'info');
}
