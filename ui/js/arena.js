/**
 * Model comparison + debate mode for AgentNate UI
 */

import { state, API, arenaState } from './state.js';
import { log, escapeHtml } from './utils.js';
import { SYSTEM_PROMPTS_LIBRARY } from '../prompts-data.js';
import { getPromptById } from './prompts.js';

export function updateArenaModels() {
    arenaState.model1 = document.getElementById('arena-model-1').value;
    arenaState.model2 = document.getElementById('arena-model-2').value;

    const select1 = document.getElementById('arena-model-1');
    const select2 = document.getElementById('arena-model-2');

    const name1 = select1.options[select1.selectedIndex]?.text || 'Model 1';
    const name2 = select2.options[select2.selectedIndex]?.text || 'Model 2';

    document.getElementById('arena-name-1').textContent = name1;
    document.getElementById('arena-name-2').textContent = name2;

    const canCompare = arenaState.model1 && arenaState.model2 && !arenaState.isRunning;
    document.getElementById('arena-send-btn').disabled = !canCompare;

    updateDebateButton();
}

export async function populateArenaModels() {
    try {
        const resp = await fetch(`${API}/models/loaded`);
        const models = await resp.json();

        const select1 = document.getElementById('arena-model-1');
        const select2 = document.getElementById('arena-model-2');

        const options = '<option value="">Select a model...</option>' +
            models.map(m => `<option value="${m.id}">${m.name} (${m.provider})</option>`).join('');

        select1.innerHTML = options;
        select2.innerHTML = options;

        populateArenaPrompts();
        updateArenaModels();
    } catch (e) {
        console.error('Failed to load models for arena:', e);
    }
}

function populateArenaPrompts() {
    const select = document.getElementById('arena-system-prompt');
    if (!select || !SYSTEM_PROMPTS_LIBRARY) return;

    let html = '<option value="">None (raw model output)</option>';
    for (const prompt of SYSTEM_PROMPTS_LIBRARY) {
        html += `<option value="${prompt.id}">${escapeHtml(prompt.name)}</option>`;
    }
    // Also add custom prompts from state
    if (state.customPrompts) {
        for (const prompt of state.customPrompts) {
            html += `<option value="${prompt.id}">${escapeHtml(prompt.name)}</option>`;
        }
    }
    select.innerHTML = html;
}

function getArenaSystemPrompt() {
    const select = document.getElementById('arena-system-prompt');
    if (!select || !select.value) return null;
    const prompt = getPromptById(select.value);
    return prompt ? prompt.content : null;
}

export async function runArenaComparison() {
    const prompt = document.getElementById('arena-input').value.trim();

    if (!prompt || !arenaState.model1 || !arenaState.model2 || arenaState.isRunning) return;

    arenaState.isRunning = true;
    arenaState.activeRequestIds = [];
    const btn = document.getElementById('arena-send-btn');
    btn.disabled = false;
    btn.innerHTML = '<span>Stop</span>';
    btn.classList.add('btn-stop');
    btn.onclick = stopComparison;

    const content1 = document.getElementById('arena-content-1');
    const content2 = document.getElementById('arena-content-2');
    const time1 = document.getElementById('arena-time-1');
    const time2 = document.getElementById('arena-time-2');

    content1.textContent = '';
    content2.textContent = '';
    time1.textContent = '';

    // Hide judge section from previous run
    const judgeSection = document.getElementById('arena-judge-section');
    if (judgeSection) {
        judgeSection.classList.add('hidden');
        document.getElementById('arena-judge-verdict').classList.add('hidden');
        document.getElementById('arena-judge-verdict').textContent = '';
        document.getElementById('arena-judge-status').textContent = '';
    }
    time2.textContent = '';

    document.getElementById('arena-response-1').classList.remove('winner');
    document.getElementById('arena-response-2').classList.remove('winner');
    document.getElementById('arena-response-1').classList.add('loading');
    document.getElementById('arena-response-2').classList.add('loading');

    const streamState = {
        model1: { content: '', startTime: Date.now(), endTime: null, done: false },
        model2: { content: '', startTime: Date.now(), endTime: null, done: false }
    };

    await Promise.all([
        streamArenaResponse(arenaState.model1, prompt, 'model1', streamState, content1, time1),
        streamArenaResponse(arenaState.model2, prompt, 'model2', streamState, content2, time2)
    ]);

    document.getElementById('arena-response-1').classList.remove('loading');
    document.getElementById('arena-response-2').classList.remove('loading');

    if (streamState.model1.endTime && streamState.model2.endTime) {
        const time1Val = streamState.model1.endTime - streamState.model1.startTime;
        const time2Val = streamState.model2.endTime - streamState.model2.startTime;

        if (time1Val < time2Val) {
            document.getElementById('arena-response-1').classList.add('winner');
        } else if (time2Val < time1Val) {
            document.getElementById('arena-response-2').classList.add('winner');
        }

        addArenaHistory(prompt,
            { status: 'fulfilled', value: { content: streamState.model1.content, time: time1Val } },
            { status: 'fulfilled', value: { content: streamState.model2.content, time: time2Val } }
        );

        // Show Auto-Judge button
        const judgeSection = document.getElementById('arena-judge-section');
        if (judgeSection) {
            judgeSection.classList.remove('hidden');
            document.getElementById('arena-judge-verdict').classList.add('hidden');
            document.getElementById('arena-judge-verdict').textContent = '';
            document.getElementById('arena-judge-status').textContent = '';
        }
    }

    resetCompareButton();
}

function resetCompareButton() {
    arenaState.isRunning = false;
    arenaState.activeRequestIds = [];
    const btn = document.getElementById('arena-send-btn');
    btn.disabled = false;
    btn.innerHTML = '<span>Compare</span>';
    btn.classList.remove('btn-stop');
    btn.onclick = runArenaComparison;
    updateArenaModels();
}

export function stopComparison() {
    if (!arenaState.isRunning) return;
    // Send cancel for each active WebSocket request
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        for (const reqId of arenaState.activeRequestIds) {
            state.ws.send(JSON.stringify({ action: 'cancel', request_id: reqId }));
        }
    }
    // The streamArenaResponse promises will resolve via 'cancelled' or timeout
}

export async function fetchArenaResponse(instanceId, prompt) {
    const startTime = Date.now();

    const resp = await fetch(`${API}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            instance_id: instanceId,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 500,
            temperature: 0.7
        })
    });

    const data = await resp.json();
    const endTime = Date.now();

    if (data.error) {
        throw new Error(data.error);
    }

    return {
        content: data.content,
        time: endTime - startTime
    };
}

async function streamArenaResponse(instanceId, prompt, modelKey, streamState, contentEl, timeEl) {
    return new Promise((resolve) => {
        const requestId = `arena-${modelKey}-${Date.now()}`;
        arenaState.activeRequestIds.push(requestId);

        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
            contentEl.innerHTML = '<span style="color: var(--error)">WebSocket not connected</span>';
            streamState[modelKey].done = true;
            streamState[modelKey].endTime = Date.now();
            resolve();
            return;
        }

        const onMessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.request_id !== requestId) return;

            if (data.type === 'token' && data.content) {
                streamState[modelKey].content += data.content;
                contentEl.textContent = streamState[modelKey].content;
                contentEl.scrollTop = contentEl.scrollHeight;
            } else if (data.type === 'done' || data.type === 'cancelled') {
                streamState[modelKey].endTime = Date.now();
                streamState[modelKey].done = true;
                const elapsed = streamState[modelKey].endTime - streamState[modelKey].startTime;
                timeEl.textContent = `${elapsed}ms`;
                state.ws.removeEventListener('message', onMessage);
                resolve();
            } else if (data.type === 'error') {
                contentEl.innerHTML = `<span style="color: var(--error)">Error: ${data.error}</span>`;
                streamState[modelKey].endTime = Date.now();
                streamState[modelKey].done = true;
                state.ws.removeEventListener('message', onMessage);
                resolve();
            }
        };

        state.ws.addEventListener('message', onMessage);

        const messages = [];
        const sysPrompt = getArenaSystemPrompt();
        if (sysPrompt) {
            messages.push({ role: 'system', content: sysPrompt });
        }
        messages.push({ role: 'user', content: prompt });

        state.ws.send(JSON.stringify({
            action: 'chat',
            instance_id: instanceId,
            messages: messages,
            request_id: requestId,
            params: {
                max_tokens: 500,
                temperature: 0.7
            }
        }));

        setTimeout(() => {
            if (!streamState[modelKey].done) {
                contentEl.innerHTML += '<span style="color: var(--warning)"> (timeout)</span>';
                streamState[modelKey].endTime = Date.now();
                streamState[modelKey].done = true;
                state.ws.removeEventListener('message', onMessage);
                resolve();
            }
        }, 60000);
    });
}

export async function fetchModelResponse(instanceId, prompt) {
    const startTime = Date.now();

    const resp = await fetch(`${API}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            instance_id: instanceId,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 500,
            temperature: 0.7
        })
    });

    const data = await resp.json();
    const endTime = Date.now();

    if (data.error) {
        throw new Error(data.error);
    }

    return {
        content: data.content,
        time: endTime - startTime
    };
}

function addArenaHistory(prompt, result1, result2) {
    const historyContainer = document.getElementById('arena-history');

    const select1 = document.getElementById('arena-model-1');
    const select2 = document.getElementById('arena-model-2');
    const name1 = select1.options[select1.selectedIndex]?.text || 'Model 1';
    const name2 = select2.options[select2.selectedIndex]?.text || 'Model 2';

    const historyItem = document.createElement('div');
    historyItem.className = 'arena-history-item';
    historyItem.innerHTML = `
        <div class="arena-history-prompt">"${escapeHtml(prompt.substring(0, 100))}${prompt.length > 100 ? '...' : ''}"</div>
        <div class="arena-history-results">
            <div class="arena-history-result">
                <div class="model-name">${escapeHtml(name1)}</div>
                <div class="time">${result1.status === 'fulfilled' ? result1.value.time + 'ms' : 'Error'}</div>
            </div>
            <div class="arena-history-result">
                <div class="model-name">${escapeHtml(name2)}</div>
                <div class="time">${result2.status === 'fulfilled' ? result2.value.time + 'ms' : 'Error'}</div>
            </div>
        </div>
    `;

    historyContainer.insertBefore(historyItem, historyContainer.firstChild);

    while (historyContainer.children.length > 10) {
        historyContainer.removeChild(historyContainer.lastChild);
    }
}

export function handleArenaKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        runArenaComparison();
    }
}

export function setArenaMode(mode) {
    arenaState.mode = mode;

    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    const compareInput = document.getElementById('arena-compare-input');
    const debateInput = document.getElementById('arena-debate-input');
    const debateTranscript = document.getElementById('debate-transcript');
    const arenaResponses = document.querySelector('.arena-responses');
    const arenaHistory = document.getElementById('arena-history');
    const promptRow = document.getElementById('arena-prompt-row');

    if (mode === 'compare') {
        compareInput.classList.remove('hidden');
        debateInput.classList.add('hidden');
        debateTranscript.classList.add('hidden');
        if (arenaResponses) arenaResponses.style.display = 'grid';
        arenaHistory.classList.remove('hidden');
        if (promptRow) promptRow.style.display = '';
    } else {
        compareInput.classList.add('hidden');
        debateInput.classList.remove('hidden');
        if (arenaResponses) arenaResponses.style.display = 'none';
        arenaHistory.classList.add('hidden');
        if (promptRow) promptRow.style.display = 'none';
    }

    updateDebateButton();
}

export function updateDebateButton() {
    const btn = document.getElementById('debate-send-btn');
    const canDebate = arenaState.model1 && arenaState.model2 && !arenaState.isRunning;
    btn.disabled = !canDebate;
}

export async function runDebate() {
    const topic = document.getElementById('debate-topic').value.trim();
    const rounds = parseInt(document.getElementById('debate-rounds').value);
    const position = document.getElementById('debate-position').value;

    if (!topic || !arenaState.model1 || !arenaState.model2 || arenaState.isRunning) return;

    arenaState.isRunning = true;
    arenaState.abortController = new AbortController();
    const btn = document.getElementById('debate-send-btn');
    btn.disabled = false;
    btn.innerHTML = '<span>Stop Debate</span>';
    btn.classList.add('btn-stop');
    btn.onclick = stopDebate;

    const transcript = document.getElementById('debate-transcript');
    const turnsContainer = document.getElementById('debate-turns');
    const statusEl = document.getElementById('debate-status');

    transcript.classList.remove('hidden');
    turnsContainer.innerHTML = '';
    statusEl.textContent = 'Connecting...';
    statusEl.className = 'running';

    let currentContentEl = null;
    let turnCount = 0;
    let debateMeta = null;

    try {
        const resp = await fetch(`${API}/chat/debate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: arenaState.abortController.signal,
            body: JSON.stringify({
                topic: topic,
                model1_id: arenaState.model1,
                model2_id: arenaState.model2,
                model1_position: position,
                rounds: rounds,
                max_tokens_per_turn: 400
            })
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                let data;
                try {
                    data = JSON.parse(line.slice(6));
                } catch { continue; }

                switch (data.type) {
                    case 'debate_start':
                        debateMeta = data;
                        turnsContainer.innerHTML = `
                            <div style="margin-bottom: 20px; padding: 12px 16px; background: var(--bg-tertiary); border-radius: var(--radius);">
                                <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">TOPIC</div>
                                <div style="font-size: 15px; color: var(--text-primary); font-weight: 500;">"${escapeHtml(data.topic)}"</div>
                                <div style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">
                                    <span style="color: var(--success);">${escapeHtml(data.model1_name)}</span> (${escapeHtml(data.model1_position)}) vs
                                    <span style="color: var(--error);">${escapeHtml(data.model2_name)}</span> (${escapeHtml(data.model2_position)})
                                </div>
                            </div>
                        `;
                        statusEl.textContent = 'In progress...';
                        break;

                    case 'turn_start': {
                        turnCount++;
                        statusEl.textContent = `Round ${data.round} \u2014 ${escapeHtml(data.model_name)} (${data.position})...`;
                        const turnDiv = document.createElement('div');
                        turnDiv.className = `debate-turn ${data.position}`;
                        turnDiv.innerHTML = `
                            <div class="debate-turn-header">
                                <div>
                                    <span class="debate-turn-model">${escapeHtml(data.model_name)}</span>
                                    <span class="debate-turn-round">Round ${data.round}</span>
                                </div>
                                <span class="debate-turn-position">${escapeHtml(data.position)}</span>
                            </div>
                            <div class="debate-turn-content"></div>
                        `;
                        turnsContainer.appendChild(turnDiv);
                        currentContentEl = turnDiv.querySelector('.debate-turn-content');
                        break;
                    }

                    case 'token':
                        if (currentContentEl && data.content) {
                            currentContentEl.textContent += data.content;
                            turnsContainer.scrollTop = turnsContainer.scrollHeight;
                        }
                        break;

                    case 'turn_end':
                        currentContentEl = null;
                        break;

                    case 'complete':
                        statusEl.textContent = `Complete (${turnCount} turns)`;
                        statusEl.className = 'complete';
                        if (debateMeta) {
                            const votingDiv = document.createElement('div');
                            votingDiv.style.cssText = 'margin-top: 24px; padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius); text-align: center;';
                            votingDiv.innerHTML = `
                                <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">Who won the debate?</div>
                                <div style="display: flex; gap: 12px; justify-content: center;">
                                    <button class="btn-primary" onclick="voteDebateWinner('${escapeHtml(debateMeta.model1_name)}')" style="background: var(--success);">
                                        ${escapeHtml(debateMeta.model1_name)}
                                    </button>
                                    <button class="btn-secondary" onclick="voteDebateWinner('tie')">
                                        Tie
                                    </button>
                                    <button class="btn-primary" onclick="voteDebateWinner('${escapeHtml(debateMeta.model2_name)}')" style="background: var(--error);">
                                        ${escapeHtml(debateMeta.model2_name)}
                                    </button>
                                </div>
                            `;
                            turnsContainer.appendChild(votingDiv);
                        }
                        break;

                    case 'error':
                        turnsContainer.innerHTML += `<div class="debate-loading" style="color: var(--error)">Error: ${escapeHtml(data.error)}</div>`;
                        statusEl.textContent = 'Failed';
                        statusEl.className = '';
                        break;
                }
            }
        }

    } catch (e) {
        if (e.name === 'AbortError') {
            statusEl.textContent = `Stopped (${turnCount} turns)`;
            statusEl.className = '';
        } else {
            turnsContainer.innerHTML += `<div class="debate-loading" style="color: var(--error)">Error: ${escapeHtml(e.message)}</div>`;
            statusEl.textContent = 'Failed';
            statusEl.className = '';
        }
    }

    resetDebateButton();
}

function resetDebateButton() {
    arenaState.isRunning = false;
    arenaState.abortController = null;
    const btn = document.getElementById('debate-send-btn');
    btn.disabled = false;
    btn.innerHTML = '<span>Start Debate</span>';
    btn.classList.remove('btn-stop');
    btn.onclick = runDebate;
    updateDebateButton();
}

export function stopDebate() {
    if (arenaState.abortController) {
        arenaState.abortController.abort();
    }
}

// ==================== Auto-Judge ====================

export async function runAutoJudge() {
    const content1 = document.getElementById('arena-content-1')?.textContent?.trim();
    const content2 = document.getElementById('arena-content-2')?.textContent?.trim();
    const promptInput = document.getElementById('arena-input');
    const originalPrompt = promptInput ? promptInput.value.trim() || '(prompt not available)' : '(prompt not available)';

    if (!content1 || !content2) return;

    const select1 = document.getElementById('arena-model-1');
    const select2 = document.getElementById('arena-model-2');
    const name1 = select1?.options[select1.selectedIndex]?.text || 'Model A';
    const name2 = select2?.options[select2.selectedIndex]?.text || 'Model B';

    const judgeBtn = document.getElementById('arena-judge-btn');
    const statusEl = document.getElementById('arena-judge-status');
    const verdictEl = document.getElementById('arena-judge-verdict');

    judgeBtn.disabled = true;
    judgeBtn.textContent = 'Judging...';
    statusEl.textContent = '';
    verdictEl.textContent = '';
    verdictEl.classList.remove('hidden');

    try {
        const resp = await fetch(`${API}/chat/judge`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: originalPrompt,
                response_a: content1,
                response_b: content2,
                model_a_name: name1,
                model_b_name: name2,
            })
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                let data;
                try { data = JSON.parse(line.slice(6)); } catch { continue; }

                if (data.type === 'judge_start') {
                    statusEl.textContent = `Judge: ${data.judge_model}`;
                } else if (data.type === 'token' && data.content) {
                    verdictEl.textContent += data.content;
                    verdictEl.scrollTop = verdictEl.scrollHeight;
                } else if (data.type === 'complete') {
                    statusEl.textContent += ' — Done';
                } else if (data.type === 'error') {
                    verdictEl.textContent = `Error: ${data.error}`;
                }
            }
        }
    } catch (e) {
        verdictEl.textContent = `Error: ${e.message}`;
    }

    judgeBtn.disabled = false;
    judgeBtn.textContent = 'Auto-Judge';
}

export function voteDebateWinner(winner) {
    log(`Debate winner selected: ${winner}`, 'info');

    const container = document.getElementById('debate-turns');
    const votingSection = container.querySelector('div:last-child');
    if (votingSection) {
        votingSection.innerHTML = `
            <div style="text-align: center; color: var(--success);">
                <span style="font-size: 20px;">&#10003;</span>
                <div style="margin-top: 8px;">Winner: <strong>${escapeHtml(winner)}</strong></div>
            </div>
        `;
    }
}
