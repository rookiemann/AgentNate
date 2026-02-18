/**
 * GPU dashboard, charts, auto-refresh for AgentNate UI
 */

import { state, API, gpuState } from './state.js';
import { log, escapeHtml } from './utils.js';

// Load GPU history from localStorage on module load
try {
    const saved = localStorage.getItem('gpuHistory');
    if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.timestamp && (Date.now() - parsed.timestamp) < 300000) {
            gpuState.history = parsed.history;
        }
    }
} catch (e) {
    console.warn('Failed to load GPU history from localStorage:', e);
}

export { gpuState };

export function initGpuDashboard() {
    if (!gpuState.initialized) {
        gpuState.initialized = true;
        fetchGpuStats();
    }

    if (gpuState.autoRefresh && !gpuState.refreshInterval) {
        startGpuAutoRefresh();
    }
}

export function toggleGpuAutoRefresh() {
    const checkbox = document.getElementById('gpu-auto-refresh');
    gpuState.autoRefresh = checkbox.checked;

    if (gpuState.autoRefresh) {
        startGpuAutoRefresh();
    } else {
        stopGpuAutoRefresh();
    }
}

export function startGpuAutoRefresh() {
    if (gpuState.refreshInterval) return;

    gpuState.refreshInterval = setInterval(() => {
        const gpuTab = document.getElementById('tab-gpu');
        if (gpuTab && gpuTab.classList.contains('active')) {
            fetchGpuStats();
        }
    }, 2000);
}

export function stopGpuAutoRefresh() {
    if (gpuState.refreshInterval) {
        clearInterval(gpuState.refreshInterval);
        gpuState.refreshInterval = null;
    }
}

export async function fetchGpuStats() {
    try {
        const resp = await fetch(`${API}/system/gpu`);
        const data = await resp.json();

        if (data.success) {
            renderGpuCards(data.gpus);
            updateGpuHistory(data.gpus);
            renderGpuCharts();
            renderGpuModelsMap(data.gpus);
            updateLastRefresh();
            updateDriverInfo(data.driver_version, data.cuda_version);
        }
    } catch (e) {
        console.error('Failed to fetch GPU stats:', e);
    }
}

function updateDriverInfo(driverVersion, cudaVersion) {
    const container = document.getElementById('gpu-driver-info');
    if (!container) return;

    let html = '';
    if (driverVersion) {
        html += `<span>Driver: <span class="version">${escapeHtml(driverVersion)}</span></span>`;
    }
    if (cudaVersion) {
        html += `<span>CUDA: <span class="version">${escapeHtml(cudaVersion)}</span></span>`;
    }
    container.innerHTML = html || '<span>Driver info unavailable</span>';
}

function renderGpuCards(gpus) {
    const container = document.getElementById('gpu-cards');

    if (!gpus || gpus.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <p>No NVIDIA GPUs detected</p>
                <p style="font-size: 12px; margin-top: 8px;">Make sure nvidia-smi is available</p>
            </div>
        `;
        return;
    }

    container.innerHTML = gpus.map(gpu => {
        const memPercent = gpu.memory_percent || 0;
        const utilPercent = gpu.utilization_percent || 0;
        const tempWarn = parseInt(localStorage.getItem('gpuTempWarn')) || 80;
        const tempClass = gpu.temperature_c > tempWarn ? 'high' : gpu.temperature_c > (tempWarn - 20) ? 'medium' : 'low';
        const memWarn = parseInt(localStorage.getItem('gpuWarnThreshold')) || 80;
        const memCrit = parseInt(localStorage.getItem('gpuCritThreshold')) || 95;
        const memClass = memPercent >= memCrit ? 'critical' : memPercent >= memWarn ? 'warning' : '';
        const utilClass = utilPercent > 90 ? 'high' : '';

        const modelsHtml = gpu.models_loaded && gpu.models_loaded.length > 0
            ? gpu.models_loaded.map(m => `
                <div class="gpu-model-chip ${m.busy ? 'busy' : ''}">
                    <span class="model-status"></span>
                    ${escapeHtml(m.model)}
                </div>
            `).join('')
            : '<span style="color: var(--text-muted); font-size: 12px;">No models loaded</span>';

        return `
            <div class="gpu-card">
                <div class="gpu-card-header">
                    <div>
                        <div class="gpu-name">${escapeHtml(gpu.name)}</div>
                    </div>
                    <div>
                        <span class="gpu-index">GPU ${gpu.index}</span>
                        ${gpu.pstate ? `<span class="gpu-pstate">${gpu.pstate}</span>` : ''}
                    </div>
                </div>

                <div class="gpu-stats">
                    <div class="gpu-stat">
                        <div class="gpu-stat-label">Temperature</div>
                        <div class="gpu-stat-value ${tempClass}">
                            ${gpu.temperature_c !== null ? gpu.temperature_c : '--'}
                            <span class="gpu-stat-unit">\u00B0C</span>
                        </div>
                    </div>
                    <div class="gpu-stat">
                        <div class="gpu-stat-label">Power</div>
                        <div class="gpu-stat-value">
                            ${gpu.power_draw_w !== null ? Math.round(gpu.power_draw_w) : '--'}
                            <span class="gpu-stat-unit">W</span>
                        </div>
                    </div>
                    <div class="gpu-stat">
                        <div class="gpu-stat-label">Fan Speed</div>
                        <div class="gpu-stat-value">
                            ${gpu.fan_speed_percent !== null ? gpu.fan_speed_percent : '--'}
                            <span class="gpu-stat-unit">%</span>
                        </div>
                    </div>
                    <div class="gpu-stat">
                        <div class="gpu-stat-label">Memory Used</div>
                        <div class="gpu-stat-value">
                            ${Math.round(gpu.memory_used_mb / 1024 * 10) / 10}
                            <span class="gpu-stat-unit">GB</span>
                        </div>
                    </div>
                </div>

                <div class="gpu-meters">
                    <div class="gpu-meter">
                        <div class="gpu-meter-header">
                            <span class="gpu-meter-label">Memory Usage</span>
                            <span class="gpu-meter-value">${Math.round(memPercent)}% (${Math.round(gpu.memory_used_mb)}/${Math.round(gpu.memory_total_mb)} MB)</span>
                        </div>
                        <div class="gpu-meter-bar">
                            <div class="gpu-meter-fill memory ${memClass}" style="width: ${memPercent}%"></div>
                        </div>
                    </div>
                    <div class="gpu-meter">
                        <div class="gpu-meter-header">
                            <span class="gpu-meter-label">GPU Utilization</span>
                            <span class="gpu-meter-value">${utilPercent}%</span>
                        </div>
                        <div class="gpu-meter-bar">
                            <div class="gpu-meter-fill utilization ${utilClass}" style="width: ${utilPercent}%"></div>
                        </div>
                    </div>
                </div>

                <div class="gpu-models">
                    <div class="gpu-models-label">Loaded Models</div>
                    <div class="gpu-model-list">
                        ${modelsHtml}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function updateGpuHistory(gpus) {
    const timestamp = Date.now();

    gpus.forEach((gpu, idx) => {
        if (!gpuState.history.utilization[idx]) {
            gpuState.history.utilization[idx] = [];
            gpuState.history.memory[idx] = [];
        }

        gpuState.history.utilization[idx].push({
            time: timestamp,
            value: gpu.utilization_percent || 0
        });

        gpuState.history.memory[idx].push({
            time: timestamp,
            value: gpu.memory_percent || 0
        });

        if (gpuState.history.utilization[idx].length > gpuState.maxHistoryPoints) {
            gpuState.history.utilization[idx].shift();
        }
        if (gpuState.history.memory[idx].length > gpuState.maxHistoryPoints) {
            gpuState.history.memory[idx].shift();
        }
    });

    if (!gpuState._saveCounter) gpuState._saveCounter = 0;
    gpuState._saveCounter++;
    if (gpuState._saveCounter >= 10) {
        gpuState._saveCounter = 0;
        try {
            localStorage.setItem('gpuHistory', JSON.stringify({
                timestamp: Date.now(),
                history: gpuState.history
            }));
        } catch (e) {
            // Ignore localStorage errors
        }
    }
}

function renderGpuCharts() {
    renderChart('util-canvas', gpuState.history.utilization, 'GPU Utilization', ['#4ade80', '#22d3ee', '#a78bfa', '#f472b6']);
    renderChart('mem-canvas', gpuState.history.memory, 'Memory Usage', ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']);
}

function renderChart(canvasId, historyData, label, colors) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();

    canvas.width = rect.width;
    canvas.height = rect.height;

    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 20, right: 20, bottom: 25, left: 40 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    ctx.fillStyle = '#16213e';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartHeight / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();

        ctx.fillStyle = '#6a6a7a';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`${100 - i * 25}%`, padding.left - 5, y + 3);
    }

    historyData.forEach((gpuData, gpuIdx) => {
        if (!gpuData || gpuData.length < 2) return;

        const color = colors[gpuIdx % colors.length];

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        gpuData.forEach((point, i) => {
            const x = padding.left + (i / (gpuState.maxHistoryPoints - 1)) * chartWidth;
            const y = padding.top + chartHeight - (point.value / 100) * chartHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        if (gpuData.length > 0) {
            const lastPoint = gpuData[gpuData.length - 1];
            const x = padding.left + ((gpuData.length - 1) / (gpuState.maxHistoryPoints - 1)) * chartWidth;
            const y = padding.top + chartHeight - (lastPoint.value / 100) * chartHeight;

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = color;
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`GPU ${gpuIdx}: ${Math.round(lastPoint.value)}%`, padding.left + 5 + gpuIdx * 80, padding.top - 5);
        }
    });

    ctx.fillStyle = '#6a6a7a';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Last 2 minutes', width / 2, height - 5);
}

function renderGpuModelsMap(gpus) {
    const container = document.getElementById('gpu-models-map');

    if (!gpus || gpus.length === 0) {
        container.innerHTML = '<div class="empty-state">No GPUs available</div>';
        return;
    }

    container.innerHTML = gpus.map(gpu => {
        const modelsHtml = gpu.models_loaded && gpu.models_loaded.length > 0
            ? gpu.models_loaded.map(m => `
                <div class="gpu-slot-model ${m.busy ? 'busy' : ''}">
                    <span class="status"></span>
                    <span>${escapeHtml(m.model)}</span>
                    <span style="margin-left: auto; font-size: 10px; color: var(--text-muted);">${m.instance_id}</span>
                </div>
            `).join('')
            : '<div class="gpu-slot-empty">No models</div>';

        return `
            <div class="gpu-slot">
                <div class="gpu-slot-header">GPU ${gpu.index}: ${escapeHtml(gpu.name)}</div>
                <div class="gpu-slot-models">
                    ${modelsHtml}
                </div>
            </div>
        `;
    }).join('');
}

function updateLastRefresh() {
    const el = document.getElementById('gpu-last-update');
    if (el) {
        el.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
    }
}
