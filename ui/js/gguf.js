/**
 * GGUF Model Search & Download - Search HuggingFace and download GGUF models
 */

import { API } from './state.js';
import { log, escapeHtml } from './utils.js';

let _activeDownloads = {};  // download_id -> poll interval
let _currentRepoId = null;

// ==================== Search ====================

export async function searchGGUF() {
    const input = document.getElementById('gguf-search-input');
    const query = input?.value.trim();
    if (!query) return;

    const resultsEl = document.getElementById('gguf-search-results');
    const filesEl = document.getElementById('gguf-files-section');
    if (filesEl) filesEl.classList.add('hidden');

    resultsEl.innerHTML = '<div class="gguf-loading">Searching HuggingFace...</div>';

    try {
        const resp = await fetch(`${API}/gguf/search?query=${encodeURIComponent(query)}&limit=10`);
        const data = await resp.json();

        if (!data.success || !data.results?.length) {
            resultsEl.innerHTML = '<div class="gguf-empty">No GGUF models found. Try a different search.</div>';
            return;
        }

        resultsEl.innerHTML = data.results.map(r => {
            const downloads = r.downloads >= 1000 ? `${(r.downloads / 1000).toFixed(1)}k` : r.downloads;
            return `
                <div class="gguf-repo-card" onclick="showGGUFFiles('${escapeHtml(r.repo_id)}')">
                    <div class="gguf-repo-name">${escapeHtml(r.repo_id)}</div>
                    <div class="gguf-repo-meta">
                        <span title="Downloads">${downloads} downloads</span>
                        ${r.author ? `<span>by ${escapeHtml(r.author)}</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');
    } catch (e) {
        resultsEl.innerHTML = `<div class="gguf-error">Search failed: ${escapeHtml(e.message)}</div>`;
    }
}

export function handleGGUFSearchKeydown(e) {
    if (e.key === 'Enter') searchGGUF();
}

// ==================== File Listing ====================

export async function showGGUFFiles(repoId) {
    _currentRepoId = repoId;
    const filesEl = document.getElementById('gguf-files-section');
    const listEl = document.getElementById('gguf-files-list');

    filesEl.classList.remove('hidden');
    listEl.innerHTML = '<div class="gguf-loading">Loading files...</div>';

    // Update header
    const headerEl = document.getElementById('gguf-files-header');
    if (headerEl) headerEl.textContent = repoId;

    try {
        const [owner, name] = repoId.split('/');
        const resp = await fetch(`${API}/gguf/files/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`);
        const data = await resp.json();

        if (!data.success || !data.files?.length) {
            listEl.innerHTML = '<div class="gguf-empty">No GGUF files found in this repo.</div>';
            return;
        }

        listEl.innerHTML = data.files.map(f => {
            const sizeGB = f.size_gb?.toFixed(2) || '?';
            const quant = f.quant || '';
            const isRecommended = quant.includes('Q4_K_M');
            return `
                <div class="gguf-file-row ${isRecommended ? 'recommended' : ''}">
                    <div class="gguf-file-info">
                        <div class="gguf-file-name">${escapeHtml(f.filename)}</div>
                        <div class="gguf-file-meta">
                            <span>${sizeGB} GB</span>
                            ${quant ? `<span class="gguf-quant">${escapeHtml(quant)}</span>` : ''}
                            ${isRecommended ? '<span class="gguf-badge">Recommended</span>' : ''}
                        </div>
                    </div>
                    <button class="btn-primary btn-sm" onclick="downloadGGUF('${escapeHtml(repoId)}', '${escapeHtml(f.filename)}')">
                        Download
                    </button>
                </div>
            `;
        }).join('');
    } catch (e) {
        listEl.innerHTML = `<div class="gguf-error">Failed to list files: ${escapeHtml(e.message)}</div>`;
    }
}

// ==================== Download ====================

export async function downloadGGUF(repoId, filename) {
    const downloadsEl = document.getElementById('gguf-downloads');
    downloadsEl.classList.remove('hidden');

    // Add a placeholder card immediately
    const cardId = `gguf-dl-${Date.now()}`;
    const listEl = document.getElementById('gguf-downloads-list');
    listEl.insertAdjacentHTML('afterbegin', `
        <div id="${cardId}" class="gguf-download-card">
            <div class="gguf-dl-name">${escapeHtml(filename)}</div>
            <div class="gguf-dl-status">Starting...</div>
            <div class="gguf-progress-bar"><div class="gguf-progress-fill" style="width: 0%"></div></div>
        </div>
    `);

    try {
        const resp = await fetch(`${API}/gguf/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId, filename })
        });
        const data = await resp.json();

        if (!data.success) {
            _updateDownloadCard(cardId, { status: 'failed', error: data.error || 'Download failed' });
            return;
        }

        if (data.status === 'already_exists') {
            _updateDownloadCard(cardId, { status: 'completed', pct: 100, msg: 'Already downloaded' });
            log(`${filename} already exists in models directory`, 'success');
            return;
        }

        const downloadId = data.download_id;
        _pollDownloadStatus(downloadId, cardId, filename);
    } catch (e) {
        _updateDownloadCard(cardId, { status: 'failed', error: e.message });
    }
}

function _pollDownloadStatus(downloadId, cardId, filename) {
    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`${API}/gguf/downloads/${downloadId}`);
            const data = await resp.json();

            if (!data.success) {
                _updateDownloadCard(cardId, { status: 'failed', error: data.error || 'Status check failed' });
                clearInterval(interval);
                delete _activeDownloads[downloadId];
                return;
            }

            const pct = data.progress_pct || 0;
            const speed = data.speed_mbps ? `${data.speed_mbps.toFixed(1)} MB/s` : '';
            const eta = data.eta_human || '';

            if (data.status === 'downloading') {
                _updateDownloadCard(cardId, {
                    status: 'downloading',
                    pct,
                    msg: `${pct.toFixed(1)}% ${speed} ${eta ? `ETA: ${eta}` : ''}`.trim()
                });
            } else if (data.status === 'completed') {
                _updateDownloadCard(cardId, { status: 'completed', pct: 100, msg: 'Complete!' });
                log(`Downloaded: ${filename}`, 'success');
                clearInterval(interval);
                delete _activeDownloads[downloadId];
                // Refresh model list if llama_cpp provider is selected
                _refreshModelListIfLlamaCpp();
            } else if (data.status === 'failed' || data.status === 'cancelled') {
                _updateDownloadCard(cardId, { status: 'failed', error: data.error || data.status });
                clearInterval(interval);
                delete _activeDownloads[downloadId];
            }
        } catch (e) {
            console.error('Download poll error:', e);
        }
    }, 3000);

    _activeDownloads[downloadId] = { interval, cardId };
}

export async function cancelGGUFDownload(downloadId) {
    try {
        await fetch(`${API}/gguf/downloads/${downloadId}`, { method: 'DELETE' });
        const entry = _activeDownloads[downloadId];
        if (entry) {
            clearInterval(entry.interval);
            _updateDownloadCard(entry.cardId, { status: 'failed', error: 'Cancelled' });
            delete _activeDownloads[downloadId];
        }
    } catch (e) {
        console.error('Cancel download error:', e);
    }
}

// ==================== Helpers ====================

function _updateDownloadCard(cardId, { status, pct, msg, error }) {
    const card = document.getElementById(cardId);
    if (!card) return;

    const statusEl = card.querySelector('.gguf-dl-status');
    const fillEl = card.querySelector('.gguf-progress-fill');

    if (status === 'downloading') {
        statusEl.textContent = msg || 'Downloading...';
        if (fillEl && pct != null) fillEl.style.width = `${pct}%`;
    } else if (status === 'completed') {
        statusEl.textContent = msg || 'Complete!';
        statusEl.classList.add('gguf-dl-complete');
        if (fillEl) { fillEl.style.width = '100%'; fillEl.classList.add('complete'); }
    } else if (status === 'failed') {
        statusEl.textContent = error || 'Failed';
        statusEl.classList.add('gguf-dl-error');
        if (fillEl) fillEl.classList.add('error');
    }
}

async function _refreshModelListIfLlamaCpp() {
    const providerSelect = document.getElementById('load-provider');
    if (providerSelect?.value === 'llama_cpp') {
        try {
            const { state } = await import('./state.js');
            delete state.models['llama_cpp'];
            const { onProviderChange } = await import('./models.js');
            onProviderChange();
        } catch (e) {
            console.warn('Could not auto-refresh model list:', e);
        }
    }
}

export function cleanupGGUFDownloads() {
    for (const [id, entry] of Object.entries(_activeDownloads)) {
        clearInterval(entry.interval);
    }
    _activeDownloads = {};
}
