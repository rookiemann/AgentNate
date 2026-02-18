/**
 * Image upload and preview for AgentNate UI
 */

import { state } from './state.js';
import { log, escapeHtml, updateDebugStatus } from './utils.js';

export function triggerImageUpload() {
    document.getElementById('image-input').click();
}

export function handleImageSelect(event) {
    const files = event.target.files;
    if (!files.length) return;

    Array.from(files).forEach(file => {
        if (!file.type.startsWith('image/')) {
            log('Skipped non-image file: ' + file.name, 'warning');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            log('Image too large (max 10MB): ' + file.name, 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUri = e.target.result;
            addPendingImage(dataUri, file.name);
        };
        reader.onerror = () => {
            log('Failed to read image: ' + file.name, 'error');
        };
        reader.readAsDataURL(file);
    });

    event.target.value = '';
}

export function addPendingImage(dataUri, filename) {
    state.pendingImages.push({ dataUri, filename });
    renderImagePreviews();
    log('Image attached: ' + filename, 'info');
}

export function removePendingImage(index) {
    const removed = state.pendingImages.splice(index, 1);
    renderImagePreviews();
    if (removed.length > 0) {
        log('Image removed: ' + removed[0].filename, 'info');
    }
}

export function renderImagePreviews() {
    const area = document.getElementById('image-preview-area');
    if (!area) return;

    if (state.pendingImages.length === 0) {
        area.classList.add('hidden');
        area.innerHTML = '';
        updateDebugStatus();
        return;
    }

    area.classList.remove('hidden');
    area.innerHTML = state.pendingImages.map((img, i) => `
        <div class="image-preview-item">
            <img src="${img.dataUri}" alt="${escapeHtml(img.filename)}">
            <button class="remove-image-btn" onclick="removePendingImage(${i})" title="Remove">&times;</button>
            <span class="image-filename">${escapeHtml(img.filename)}</span>
        </div>
    `).join('');
    updateDebugStatus();
}

export function updateVisionUI() {
    const uploadBtn = document.getElementById('image-upload-btn');
    if (!uploadBtn) return;

    if (state.currentModelHasVision) {
        uploadBtn.classList.remove('hidden');
    } else {
        uploadBtn.classList.add('hidden');
        if (state.pendingImages.length > 0) {
            state.pendingImages = [];
            renderImagePreviews();
            log('Pending images cleared (model does not support vision)', 'info');
        }
    }
    updateDebugStatus();
}

export function openImageModal(src) {
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.onclick = () => modal.remove();
    modal.innerHTML = `
        <div class="image-modal-content">
            <img src="${src}" alt="Full size image">
            <div class="image-modal-hint">Click anywhere to close</div>
        </div>
    `;
    document.body.appendChild(modal);
}
