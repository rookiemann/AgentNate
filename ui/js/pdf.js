/**
 * PDF upload and RAG retrieval for AgentNate UI
 */

import { state, API, getPanelDomId } from './state.js';
import { log, escapeHtml, updateDebugStatus } from './utils.js';

export function triggerPdfUpload() {
    document.getElementById('pdf-input')?.click();
}

// Panel-aware wrappers
export function triggerPanelPdfUpload(panelId) {
    const input = document.getElementById(getPanelDomId(panelId, 'pdf-input'));
    if (input) input.click();
}

export async function handlePanelPdfSelect(event, panelId) {
    // Delegate to the same upload logic — PDFs are a shared resource
    return handlePdfSelect(event);
}

export async function handlePdfSelect(event) {
    const files = event.target.files;
    if (!files.length) return;

    for (const file of Array.from(files)) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            log('Skipped non-PDF file: ' + file.name, 'warning');
            continue;
        }

        if (file.size > 50 * 1024 * 1024) {
            log('PDF too large (max 50MB): ' + file.name, 'error');
            continue;
        }

        await uploadPdf(file);
    }

    event.target.value = '';
}

export async function uploadPdf(file) {
    log('Uploading PDF: ' + file.name, 'info');

    const pendingEntry = {
        filename: file.name,
        status: 'uploading',
        size: file.size
    };
    state.pendingPdfs.push(pendingEntry);
    renderPdfPreviews();

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', state.pdfSessionId);

        const response = await fetch(`${API}/chat/upload-pdf`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        const pendingIdx = state.pendingPdfs.findIndex(p => p.filename === file.name && p.status === 'uploading');
        if (pendingIdx >= 0) {
            state.pendingPdfs.splice(pendingIdx, 1);
        }

        if (result.success) {
            state.loadedPdfFiles.push({
                filename: result.filename,
                pageCount: result.page_count,
                chunkCount: result.chunk_count,
                tokenEstimate: result.token_estimate,
                fallbackMode: result.fallback_mode || false
            });
            state.embeddingProvider = result.embedding_provider;

            renderPdfPreviews();

            const modeMsg = result.fallback_mode ? ' (full-text mode)' : ' (RAG mode)';
            log(`PDF ready: ${result.filename} - ${result.page_count} pages, ${result.chunk_count} chunks${modeMsg}`, 'info');

            if (result.fallback_mode && result.error) {
                log('Note: ' + result.error, 'warning');
            }
        } else {
            log('PDF error: ' + result.error, 'error');
            renderPdfPreviews();
        }
    } catch (e) {
        const pendingIdx = state.pendingPdfs.findIndex(p => p.filename === file.name && p.status === 'uploading');
        if (pendingIdx >= 0) {
            state.pendingPdfs.splice(pendingIdx, 1);
        }
        renderPdfPreviews();
        log('Failed to upload PDF: ' + (e.message || 'Network error'), 'error');
    }
}

export function removePdf(filename) {
    const idx = state.loadedPdfFiles.findIndex(p => p.filename === filename);
    if (idx >= 0) {
        state.loadedPdfFiles.splice(idx, 1);
        renderPdfPreviews();
        log('PDF removed: ' + filename, 'info');
    }
}

export function renderPdfPreviews() {
    const area = document.getElementById('pdf-preview-area');
    if (!area) return;

    const allPdfs = [...state.pendingPdfs, ...state.loadedPdfFiles.map(p => ({ ...p, status: 'ready' }))];

    if (allPdfs.length === 0) {
        area.classList.add('hidden');
        area.innerHTML = '';
        updateDebugStatus();
        return;
    }

    area.classList.remove('hidden');
    area.innerHTML = allPdfs.map((pdf, i) => {
        if (pdf.status === 'uploading') {
            return `
                <div class="pdf-preview-item uploading">
                    <div class="pdf-icon">📄</div>
                    <div class="pdf-info">
                        <span class="pdf-filename">${escapeHtml(pdf.filename)}</span>
                        <span class="pdf-meta">Uploading...</span>
                    </div>
                    <div class="pdf-spinner"></div>
                </div>
            `;
        } else {
            const modeLabel = pdf.fallbackMode ? '(full-text)' : '(RAG)';
            return `
                <div class="pdf-preview-item">
                    <div class="pdf-icon">📄</div>
                    <div class="pdf-info">
                        <span class="pdf-filename">${escapeHtml(pdf.filename)}</span>
                        <span class="pdf-meta">${pdf.pageCount} pages, ~${pdf.tokenEstimate?.toLocaleString() || '?'} tokens ${modeLabel}</span>
                    </div>
                    <button class="remove-pdf-btn" onclick="removePdf('${escapeHtml(pdf.filename)}')" title="Remove">&times;</button>
                </div>
            `;
        }
    }).join('');
    updateDebugStatus();
}

export async function retrievePdfContext(query) {
    if (state.loadedPdfFiles.length === 0) {
        return null;
    }

    try {
        const response = await fetch(`${API}/chat/retrieve-context`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                session_id: state.pdfSessionId,
                top_k: 5
            })
        });

        const result = await response.json();

        if (!result.success || !result.chunks || result.chunks.length === 0) {
            return null;
        }

        const contextParts = result.chunks.map(chunk => {
            if (chunk.fallback) {
                return `[Document: ${chunk.filename}]
${chunk.text}`;
            } else {
                return `[${chunk.filename} - Page ${chunk.page}] (relevance: ${Math.round(chunk.score * 100)}%)
${chunk.text}`;
            }
        });

        return contextParts.join('\n\n---\n\n');
    } catch (e) {
        console.error('Failed to retrieve PDF context:', e);
        return null;
    }
}

export async function clearPdfSession() {
    if (state.loadedPdfFiles.length === 0) {
        return;
    }

    try {
        await fetch(`${API}/chat/clear-pdf-session?session_id=${state.pdfSessionId}`, {
            method: 'POST'
        });
        log('PDF session cleared, embedding model unloaded', 'info');
    } catch (e) {
        console.error('Failed to clear PDF session:', e);
    }

    state.loadedPdfFiles = [];
    state.pendingPdfs = [];
    state.embeddingProvider = null;
    renderPdfPreviews();
}
