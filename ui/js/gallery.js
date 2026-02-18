/**
 * Gallery Module — Browse, search, and manage ComfyUI generations.
 *
 * Subtab under ComfyUI tab. Displays a grid of generated images with
 * filters, detail modal, favorites, tags, lineage, and action buttons.
 */

import { API, comfyuiState } from './state.js';
import { escapeHtml, showToast, apiFetch } from './utils.js';

// ======================== Initialization ========================

let galleryInitialized = false;

export function initGallery() {
    if (galleryInitialized) {
        loadGallery();
        return;
    }
    galleryInitialized = true;
    renderGalleryFilters();
    loadGalleryStats();
    loadGallery();
}

// ======================== Data Loading ========================

let _galleryLoadId = 0;          // Monotonic counter — discard stale responses
let _galleryAbort = null;        // AbortController for in-flight fetch

export async function loadGallery() {
    const g = comfyuiState.gallery;
    const params = new URLSearchParams();
    if (g.filters.query) params.set('query', g.filters.query);
    if (g.filters.checkpoint) params.set('checkpoint', g.filters.checkpoint);
    if (g.filters.tags) params.set('tags', g.filters.tags);
    if (g.filters.favorite) params.set('favorite', 'true');
    if (g.filters.minRating > 0) params.set('min_rating', g.filters.minRating);
    params.set('sort', g.sort);
    params.set('limit', g.pageSize);
    params.set('offset', g.page * g.pageSize);

    // Cancel any in-flight fetch and record this load's ID
    if (_galleryAbort) _galleryAbort.abort();
    _galleryAbort = new AbortController();
    const myId = ++_galleryLoadId;

    try {
        const resp = await fetch(`${API}/comfyui/media/generations?${params}`, {
            signal: _galleryAbort.signal,
        });
        // Discard if a newer load was triggered while we were fetching
        if (myId !== _galleryLoadId) return;
        const data = await resp.json();
        if (myId !== _galleryLoadId) return;
        g.generations = data.generations || [];
        g.total = data.total || 0;
        renderGalleryGrid();
        renderPagination();
    } catch (e) {
        if (e.name === 'AbortError') return;  // Expected — newer load superseded us
        console.error('Failed to load gallery:', e);
    }
}

export async function loadGalleryStats() {
    try {
        const resp = await fetch(`${API}/comfyui/media/stats`);
        const stats = await resp.json();
        comfyuiState.gallery.stats = stats;
        renderStatsBar(stats);
    } catch (e) {
        console.error('Failed to load gallery stats:', e);
    }
}

// ======================== Rendering ========================

function renderStatsBar(stats) {
    const el = document.getElementById('gallery-stats-bar');
    if (!el) return;

    el.innerHTML = `
        <div class="gallery-stats">
            <span class="stat-item"><strong>${stats.total_generations || 0}</strong> generations</span>
            <span class="stat-item"><strong>${stats.total_files || 0}</strong> files</span>
            <span class="stat-item"><strong>${stats.total_mb || 0}</strong> MB</span>
            <span class="stat-item"><strong>${stats.favorites_count || 0}</strong> favorites</span>
            <button class="btn-secondary btn-small" onclick="scanExistingImages()">Scan Existing</button>
        </div>
    `;
}

export function renderGalleryFilters() {
    const el = document.getElementById('gallery-filters');
    if (!el) return;
    const g = comfyuiState.gallery;

    el.innerHTML = `
        <div class="gallery-filter-bar">
            <input type="text" class="input-field gallery-search" placeholder="Search prompts..."
                   value="${escapeHtml(g.filters.query)}"
                   onkeyup="if(event.key==='Enter') galleryApplyFilters()"
                   id="gallery-search-input" />
            <input type="text" class="input-field gallery-checkpoint-filter" placeholder="Checkpoint..."
                   value="${escapeHtml(g.filters.checkpoint)}"
                   id="gallery-checkpoint-input" />
            <input type="text" class="input-field gallery-tags-filter" placeholder="Tags..."
                   value="${escapeHtml(g.filters.tags)}"
                   id="gallery-tags-input" />
            <label class="gallery-fav-toggle">
                <input type="checkbox" ${g.filters.favorite ? 'checked' : ''}
                       onchange="galleryToggleFavoriteFilter(this.checked)" />
                Favorites
            </label>
            <select class="input-field gallery-sort" onchange="galleryChangeSort(this.value)" id="gallery-sort-select">
                <option value="newest" ${g.sort === 'newest' ? 'selected' : ''}>Newest</option>
                <option value="oldest" ${g.sort === 'oldest' ? 'selected' : ''}>Oldest</option>
                <option value="rating" ${g.sort === 'rating' ? 'selected' : ''}>Rating</option>
                <option value="favorites" ${g.sort === 'favorites' ? 'selected' : ''}>Favorites</option>
            </select>
            <button class="btn-primary btn-small" onclick="galleryApplyFilters()">Search</button>
        </div>
    `;
}

function renderGalleryGrid() {
    const el = document.getElementById('gallery-grid');
    if (!el) return;
    const g = comfyuiState.gallery;

    if (g.generations.length === 0) {
        el.innerHTML = `<div class="gallery-empty">
            <p>No generations found.</p>
            <p class="text-muted">Generate images via the agent or use "Scan Existing" to catalog output files.</p>
        </div>`;
        return;
    }

    el.innerHTML = g.generations.map(gen => {
        const thumbUrl = gen.primary_filename
            ? `${API}/comfyui/images/${encodeURIComponent(gen.primary_filename)}${gen.primary_subfolder ? '?subfolder=' + encodeURIComponent(gen.primary_subfolder) : ''}`
            : '';
        const prompt = gen.prompt_text || 'No prompt';
        const truncPrompt = prompt.length > 60 ? prompt.substring(0, 57) + '...' : prompt;
        const checkpoint = gen.checkpoint ? gen.checkpoint.replace('.safetensors', '').replace('.gguf', '') : '';
        const dims = gen.width && gen.height ? `${gen.width}x${gen.height}` : '';
        const favClass = gen.favorite ? 'active' : '';
        const stars = gen.rating || 0;
        const mt = gen.primary_media_type || 'image';

        let thumbContent;
        if (!thumbUrl) {
            thumbContent = '<div class="gallery-no-thumb">No media</div>';
        } else if (mt === 'video') {
            thumbContent = `<video src="${thumbUrl}" muted loop preload="metadata" onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"></video>
                <div class="gallery-media-badge">VIDEO</div>`;
        } else if (mt === 'audio') {
            thumbContent = `<div class="gallery-audio-thumb"><span class="audio-icon">&#9835;</span><div class="gallery-media-badge">AUDIO</div></div>`;
        } else {
            thumbContent = `<img src="${thumbUrl}" loading="lazy" alt="Generated image" />`;
        }

        return `
            <div class="gallery-card" onclick="openGenerationDetail('${gen.id}')">
                <div class="gallery-thumb">
                    ${thumbContent}
                </div>
                <div class="gallery-card-info">
                    <span class="gallery-prompt" title="${escapeHtml(prompt)}">${escapeHtml(truncPrompt)}</span>
                    <div class="gallery-card-meta">
                        ${checkpoint ? `<span class="gallery-checkpoint">${escapeHtml(checkpoint)}</span>` : ''}
                        ${dims ? `<span class="gallery-dims">${dims}</span>` : ''}
                    </div>
                    <div class="gallery-card-actions">
                        <button class="gallery-fav-btn ${favClass}" onclick="event.stopPropagation(); toggleFavorite('${gen.id}', ${gen.favorite ? 0 : 1})" title="Favorite">&#9829;</button>
                        <span class="gallery-stars">${renderStars(stars)}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function renderStars(rating) {
    let html = '';
    for (let i = 1; i <= 5; i++) {
        html += `<span class="star ${i <= rating ? 'filled' : ''}">${i <= rating ? '\u2605' : '\u2606'}</span>`;
    }
    return html;
}

function renderPagination() {
    const el = document.getElementById('gallery-pagination');
    if (!el) return;
    const g = comfyuiState.gallery;
    const totalPages = Math.ceil(g.total / g.pageSize);
    if (totalPages <= 1) { el.innerHTML = ''; return; }

    let html = '<div class="gallery-pagination">';
    if (g.page > 0) html += `<button class="btn-secondary btn-small" onclick="galleryPage(${g.page - 1})">Prev</button>`;
    html += `<span class="page-info">Page ${g.page + 1} of ${totalPages}</span>`;
    if (g.page < totalPages - 1) html += `<button class="btn-secondary btn-small" onclick="galleryPage(${g.page + 1})">Next</button>`;
    html += '</div>';
    el.innerHTML = html;
}

// ======================== Detail Modal ========================

let _detailLoading = false;
export async function openGenerationDetail(genId) {
    if (_detailLoading) return;
    _detailLoading = true;
    // Remove any existing modal immediately to prevent stacking
    const existing = document.querySelector('.gallery-detail-modal');
    if (existing) existing.remove();
    try {
        const resp = await fetch(`${API}/comfyui/media/generations/${genId}`);
        const data = await resp.json();
        if (!data.generation) {
            showToast('Generation not found', 'error');
            return;
        }
        renderDetailModal(data.generation, data.lineage || []);
    } catch (e) {
        showToast('Failed to load generation details', 'error');
    } finally {
        _detailLoading = false;
    }
}

function renderDetailModal(gen, lineage) {
    // Remove existing modal if any
    const existing = document.querySelector('.gallery-detail-modal');
    if (existing) existing.remove();

    const files = gen.files || [];
    const primaryFile = files[0];
    const mediaUrl = primaryFile
        ? `${API}/comfyui/images/${encodeURIComponent(primaryFile.filename)}${primaryFile.subfolder ? '?subfolder=' + encodeURIComponent(primaryFile.subfolder) : ''}`
        : '';
    const primaryMt = primaryFile?.media_type || 'image';

    let primaryMediaHtml;
    if (!mediaUrl) {
        primaryMediaHtml = '<div class="gallery-no-image">No media</div>';
    } else if (primaryMt === 'video') {
        primaryMediaHtml = `<video src="${mediaUrl}" controls loop preload="metadata"></video>`;
    } else if (primaryMt === 'audio') {
        primaryMediaHtml = `<div class="gallery-audio-detail"><span class="audio-icon-large">&#9835;</span><audio src="${mediaUrl}" controls preload="metadata"></audio></div>`;
    } else {
        primaryMediaHtml = `<img src="${mediaUrl}" alt="Generated image" onclick="openImageModal('${mediaUrl}')" />`;
    }

    const modal = document.createElement('div');
    modal.className = 'gallery-detail-modal';
    modal.onclick = (e) => { if (e.target === modal) modal.remove(); };

    modal.innerHTML = `
        <div class="gallery-detail-content">
            <button class="gallery-detail-close" onclick="this.closest('.gallery-detail-modal').remove()">&times;</button>
            <div class="gallery-detail-layout">
                <div class="gallery-detail-image">
                    ${primaryMediaHtml}
                    ${files.length > 1 ? `<div class="gallery-file-count">${files.length} files</div>` : ''}
                </div>
                <div class="gallery-detail-info">
                    <div class="gallery-detail-section">
                        <h3>Generation Parameters</h3>
                        <div class="gallery-params">
                            ${paramRow('Checkpoint', gen.checkpoint)}
                            ${paramRow('Prompt', gen.prompt_text)}
                            ${paramRow('Negative', gen.negative_prompt)}
                            ${paramRow('Seed', gen.seed)}
                            ${paramRow('Steps', gen.steps)}
                            ${paramRow('CFG', gen.cfg)}
                            ${paramRow('Sampler', gen.sampler)}
                            ${paramRow('Scheduler', gen.scheduler)}
                            ${paramRow('Denoise', gen.denoise)}
                            ${paramRow('Size', gen.width && gen.height ? `${gen.width}x${gen.height}` : null)}
                            ${paramRow('Type', gen.workflow_type)}
                            ${paramRow('Created', gen.created_at)}
                        </div>
                    </div>

                    <div class="gallery-detail-section">
                        <h3>User Metadata</h3>
                        <div class="gallery-metadata-form">
                            <label>Title</label>
                            <input type="text" class="input-field" id="gen-detail-title" value="${escapeHtml(gen.title || '')}" placeholder="Name this generation..." />
                            <label>Tags</label>
                            <input type="text" class="input-field" id="gen-detail-tags" value="${escapeHtml(gen.tags || '')}" placeholder="tag1, tag2, ..." />
                            <label>Rating</label>
                            <div class="gallery-detail-rating" id="gen-detail-rating">${renderClickableStars(gen.rating || 0, gen.id)}</div>
                            <label>Notes</label>
                            <textarea class="input-field" id="gen-detail-notes" rows="3" placeholder="Notes...">${escapeHtml(gen.notes || '')}</textarea>
                            <div class="gallery-detail-btn-row">
                                <button class="btn-primary btn-small" onclick="saveGenerationMeta('${gen.id}')">Save</button>
                                <button class="btn-secondary btn-small gallery-fav-detail ${gen.favorite ? 'active' : ''}" onclick="toggleFavorite('${gen.id}', ${gen.favorite ? 0 : 1}); this.classList.toggle('active');">
                                    &#9829; ${gen.favorite ? 'Favorited' : 'Favorite'}
                                </button>
                            </div>
                        </div>
                    </div>

                    ${lineage.length > 1 ? `
                    <div class="gallery-detail-section">
                        <h3>Lineage</h3>
                        <div class="gallery-lineage">
                            ${lineage.map(l => `
                                <div class="lineage-item ${l.id === gen.id ? 'current' : ''}" onclick="openGenerationDetail('${l.id}')">
                                    <span class="lineage-type">${escapeHtml(l.workflow_type || '?')}</span>
                                    <span class="lineage-prompt">${escapeHtml((l.prompt_text || '').substring(0, 40))}</span>
                                </div>
                            `).join('<span class="lineage-arrow">&rarr;</span>')}
                        </div>
                    </div>` : ''}

                    <div class="gallery-detail-section">
                        <h3>Actions</h3>
                        <div class="gallery-detail-actions">
                            <button class="btn-secondary btn-small" onclick="deleteGeneration('${gen.id}')">Delete</button>
                        </div>
                    </div>

                    ${primaryFile ? `
                    <div class="gallery-detail-section">
                        <h3>File Info</h3>
                        <div class="gallery-params">
                            ${paramRow('Filename', primaryFile.filename)}
                            ${paramRow('Type', primaryFile.media_type)}
                            ${paramRow('Size', primaryFile.file_size ? formatBytes(primaryFile.file_size) : null)}
                            ${paramRow('Dimensions', primaryFile.width && primaryFile.height ? `${primaryFile.width}x${primaryFile.height}` : null)}
                            ${paramRow('Duration', primaryFile.duration ? primaryFile.duration + 's' : null)}
                            ${paramRow('Format', primaryFile.format)}
                        </div>
                    </div>` : ''}
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
}

function paramRow(label, value) {
    if (value === null || value === undefined || value === '') return '';
    return `<div class="param-row"><span class="param-label">${escapeHtml(label)}</span><span class="param-value">${escapeHtml(String(value))}</span></div>`;
}

function renderClickableStars(rating, genId) {
    let html = '';
    for (let i = 1; i <= 5; i++) {
        html += `<span class="star clickable ${i <= rating ? 'filled' : ''}" onclick="event.stopPropagation(); rateGeneration('${genId}', ${i})">${i <= rating ? '\u2605' : '\u2606'}</span>`;
    }
    return html;
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ======================== Actions ========================

export function galleryApplyFilters() {
    const g = comfyuiState.gallery;
    const queryEl = document.getElementById('gallery-search-input');
    const checkpointEl = document.getElementById('gallery-checkpoint-input');
    const tagsEl = document.getElementById('gallery-tags-input');
    if (queryEl) g.filters.query = queryEl.value.trim();
    if (checkpointEl) g.filters.checkpoint = checkpointEl.value.trim();
    if (tagsEl) g.filters.tags = tagsEl.value.trim();
    g.page = 0;
    loadGallery();
}

export function galleryToggleFavoriteFilter(checked) {
    comfyuiState.gallery.filters.favorite = checked;
    comfyuiState.gallery.page = 0;
    loadGallery();
}

export function galleryChangeSort(value) {
    comfyuiState.gallery.sort = value;
    comfyuiState.gallery.page = 0;
    loadGallery();
}

export function galleryPage(page) {
    const g = comfyuiState.gallery;
    const totalPages = Math.max(1, Math.ceil(g.total / g.pageSize));
    g.page = Math.max(0, Math.min(page, totalPages - 1));
    // Re-render pagination buttons immediately so rapid clicks
    // always see the correct Prev/Next targets (no stale onclick values)
    renderPagination();
    loadGallery();
}

export async function toggleFavorite(genId, newValue) {
    try {
        await fetch(`${API}/comfyui/media/generations/${genId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ favorite: newValue }),
        });
        loadGallery();
        loadGalleryStats();
    } catch (e) {
        showToast('Failed to update favorite', 'error');
    }
}

export async function rateGeneration(genId, rating) {
    try {
        await fetch(`${API}/comfyui/media/generations/${genId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rating }),
        });
        // Update stars in detail modal
        const ratingEl = document.getElementById('gen-detail-rating');
        if (ratingEl) ratingEl.innerHTML = renderClickableStars(rating, genId);
    } catch (e) {
        showToast('Failed to update rating', 'error');
    }
}

export async function saveGenerationMeta(genId) {
    const title = document.getElementById('gen-detail-title')?.value || '';
    const tags = document.getElementById('gen-detail-tags')?.value || '';
    const notes = document.getElementById('gen-detail-notes')?.value || '';
    try {
        await fetch(`${API}/comfyui/media/generations/${genId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, tags, notes }),
        });
        showToast('Saved', 'success');
        loadGallery();
    } catch (e) {
        showToast('Failed to save', 'error');
    }
}

export async function deleteGeneration(genId) {
    if (!confirm('Delete this generation record? (Files on disk are not removed)')) return;
    try {
        await fetch(`${API}/comfyui/media/generations/${genId}`, { method: 'DELETE' });
        showToast('Deleted', 'success');
        const modal = document.querySelector('.gallery-detail-modal');
        if (modal) modal.remove();
        loadGallery();
        loadGalleryStats();
    } catch (e) {
        showToast('Failed to delete', 'error');
    }
}

export async function scanExistingImages() {
    try {
        showToast('Scanning output directory...', 'info');
        const resp = await fetch(`${API}/comfyui/media/scan`, { method: 'POST' });
        const data = await resp.json();
        const count = data.new_files || 0;
        showToast(count > 0 ? `Added ${count} files to gallery` : 'No new files found', count > 0 ? 'success' : 'info');
        loadGalleryStats();
        loadGallery();
    } catch (e) {
        showToast('Scan failed', 'error');
    }
}
