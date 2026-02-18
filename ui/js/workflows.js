/**
 * Workflow marketplace, deploy, param editor, legacy templates for AgentNate UI
 */

import { state, API, queueState, marketplaceState, workflowState, paramEditorState } from './state.js';
import { log, escapeHtml, showToast, updateToast, debugLog, renderSimpleMarkdown } from './utils.js';
import { refreshQueueStatus, openN8nTab, closeN8nTab, openMainAdminTab, renderWorkersFullList, stopAllWorkers, updateWorkersCount } from './n8n.js';

// Re-export worker functions that are referenced from HTML
export { renderWorkersFullList, stopAllWorkers, updateWorkersCount };

// ==================== Workflow Subtabs ====================

export function switchWorkflowSubtab(subtab) {
    marketplaceState.currentSubtab = subtab;

    document.querySelectorAll('.workflows-subtabs .subtab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.subtab === subtab);
    });

    document.querySelectorAll('.subtab-content').forEach(panel => {
        panel.classList.add('hidden');
    });
    const activePanel = document.getElementById(`subtab-${subtab}`);
    if (activePanel) activePanel.classList.remove('hidden');

    if (subtab === 'deployed') {
        refreshDeployedWorkflows();
    } else if (subtab === 'workers') {
        renderWorkersFullList();
    }
}

export function showDeployedWorkflows() {
    switchWorkflowSubtab('deployed');
}

// ==================== Workflow Import ====================

let importState = {
    validatedJson: null,
    summary: null,
};

export function toggleImportZone() {
    const zone = document.getElementById('import-zone');
    if (zone) zone.classList.toggle('hidden');
}

export function handleImportFile(event) {
    const files = Array.from(event.target.files);
    if (!files.length) return;

    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    const oversized = files.find(f => f.size > MAX_FILE_SIZE);
    if (oversized) {
        showToast(`File too large: ${oversized.name} (${(oversized.size / 1024 / 1024).toFixed(1)}MB, max 10MB)`, 'error');
        event.target.value = '';
        return;
    }

    if (files.length === 1) {
        // Single file: put in textarea for review
        const reader = new FileReader();
        reader.onload = (e) => {
            const textarea = document.getElementById('import-json-input');
            if (textarea) textarea.value = e.target.result;
            validateImportedWorkflow();
        };
        reader.readAsText(files[0]);
    } else {
        // Multiple files: batch deploy all
        batchImportWorkflows(files);
    }
    event.target.value = '';
}

async function batchImportWorkflows(files) {
    const resultsDiv = document.getElementById('import-results');
    if (resultsDiv) {
        resultsDiv.classList.remove('hidden');
        resultsDiv.innerHTML = `<div class="loading">Importing ${files.length} workflows...</div>`;
    }

    const results = [];

    for (const file of files) {
        try {
            const text = await file.text();
            const parsed = JSON.parse(text);

            const resp = await fetch(`${API}/workflows/import/deploy`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ workflow_json: parsed, mode: 'standby' }),
            });
            const data = await resp.json();

            results.push({
                name: parsed.name || file.name,
                success: data.success,
                error: data.error || null,
                port: data.worker?.port || null,
            });
        } catch (e) {
            results.push({
                name: file.name,
                success: false,
                error: e.message,
            });
        }
    }

    const succeeded = results.filter(r => r.success).length;
    let html = `<div class="import-batch-results">
        <strong>Batch Import: ${succeeded}/${results.length} deployed</strong>
        <div class="import-batch-list">`;

    for (const r of results) {
        if (r.success) {
            html += `<div class="import-batch-row success">&#10003; ${escapeHtml(r.name)}</div>`;
        } else {
            html += `<div class="import-batch-row error">&#10007; ${escapeHtml(r.name)}: ${escapeHtml(r.error)}</div>`;
        }
    }

    html += '</div></div>';
    if (resultsDiv) resultsDiv.innerHTML = html;

    if (succeeded > 0) {
        showToast(`Imported ${succeeded}/${results.length} workflows`, 'success');
        await refreshDeployedWorkflows();
    }
}

export async function pasteWorkflowFromClipboard() {
    try {
        const text = await navigator.clipboard.readText();
        const textarea = document.getElementById('import-json-input');
        if (textarea) textarea.value = text;
        validateImportedWorkflow();
    } catch (e) {
        log('Clipboard access denied - paste manually into the text area', 'warning');
    }
}

export async function validateImportedWorkflow() {
    debugLog('validateImportedWorkflow', 'starting');
    const textarea = document.getElementById('import-json-input');
    const resultsDiv = document.getElementById('import-results');
    if (!textarea || !resultsDiv) return;

    const raw = textarea.value.trim();
    if (!raw) {
        resultsDiv.classList.add('hidden');
        return;
    }

    let parsed;
    try {
        parsed = JSON.parse(raw);
    } catch (e) {
        resultsDiv.classList.remove('hidden');
        resultsDiv.innerHTML = `
            <div class="import-error">
                <strong>Invalid JSON</strong>
                <p>${escapeHtml(e.message)}</p>
            </div>`;
        importState.validatedJson = null;
        return;
    }

    resultsDiv.classList.remove('hidden');
    resultsDiv.innerHTML = '<div class="loading">Validating...</div>';

    try {
        const resp = await fetch(`${API}/workflows/import`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workflow_json: parsed }),
        });
        const data = await resp.json();

        importState.validatedJson = data.fixed_json;
        importState.summary = data.summary;

        let html = '';

        if (data.valid) {
            const s = data.summary;
            html += `
                <div class="import-success">
                    <div class="import-summary">
                        <strong>${escapeHtml(s.name)}</strong>
                        <span>${s.node_count} nodes</span>
                        <span>Trigger: ${escapeHtml(s.trigger_type)}</span>
                        ${s.integrations.length ? `<span>${s.integrations.map(i => escapeHtml(i)).join(', ')}</span>` : ''}
                    </div>
                    ${data.warnings.length ? `<div class="import-warnings">${data.warnings.map(w => `<div>&#9888; ${escapeHtml(w)}</div>`).join('')}</div>` : ''}
                    <div class="import-actions">
                        <button class="btn-primary" onclick="deployImportedWorkflow('once')">Deploy &amp; Run Once</button>
                        <button class="btn-secondary" onclick="deployImportedWorkflow('loop')">Deploy &amp; Loop</button>
                        <button class="btn-secondary" onclick="deployImportedWorkflow('standby')">Deploy Standby</button>
                        ${state.currentModel ? `<button class="btn-secondary" onclick="reviewImportWithAgent()">Review with Agent</button>` : ''}
                    </div>
                </div>`;
        } else {
            html += `
                <div class="import-error">
                    <strong>Validation Failed</strong>
                    ${data.errors.map(e => `<div>&#10007; ${escapeHtml(e)}</div>`).join('')}
                    ${data.warnings.length ? data.warnings.map(w => `<div>&#9888; ${escapeHtml(w)}</div>`).join('') : ''}
                    ${state.currentModel ? `<div class="import-actions"><button class="btn-secondary" onclick="reviewImportWithAgent()">Review with Agent</button></div>` : ''}
                </div>`;
        }

        resultsDiv.innerHTML = html;

    } catch (e) {
        resultsDiv.innerHTML = `<div class="import-error"><strong>Validation request failed</strong><p>${escapeHtml(e.message)}</p></div>`;
    }
}

export async function deployImportedWorkflow(mode = 'once') {
    debugLog('deployImportedWorkflow', `mode=${mode} name=${importState.summary?.name || 'unknown'}`);
    if (!importState.validatedJson) {
        log('No validated workflow to deploy', 'error');
        return;
    }

    const resultsDiv = document.getElementById('import-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = '<div class="loading">Deploying workflow...</div>';
    }

    try {
        const loopTarget = mode === 'loop' ? parseInt(prompt('Loop target (empty for infinite):', '') || '0', 10) || null : null;

        const resp = await fetch(`${API}/workflows/import/deploy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_json: importState.validatedJson,
                mode,
                loop_target: loopTarget,
            }),
        });
        const data = await resp.json();

        if (data.success) {
            log(`Workflow deployed! Worker on port ${data.worker?.port}`, 'success');
            showToast(`Deployed: ${importState.summary?.name || 'Workflow'}`, 'success');

            // Clear import state
            importState.validatedJson = null;
            importState.summary = null;
            const textarea = document.getElementById('import-json-input');
            if (textarea) textarea.value = '';
            if (resultsDiv) resultsDiv.classList.add('hidden');

            // Collapse import zone
            const zone = document.getElementById('import-zone');
            if (zone) zone.classList.add('hidden');

            // Refresh lists
            await refreshQueueStatus();
            await refreshDeployedWorkflows();
        } else {
            log(`Deploy failed: ${data.error}`, 'error');
            if (resultsDiv) {
                resultsDiv.innerHTML = `<div class="import-error"><strong>Deploy failed</strong><p>${escapeHtml(data.error)}</p></div>`;
            }
        }
    } catch (e) {
        log(`Deploy error: ${e.message}`, 'error');
        if (resultsDiv) {
            resultsDiv.innerHTML = `<div class="import-error"><strong>Deploy error</strong><p>${escapeHtml(e.message)}</p></div>`;
        }
    }
}

export function reviewImportWithAgent() {
    const textarea = document.getElementById('import-json-input');
    const raw = textarea?.value?.trim();
    if (!raw) return;

    // Switch to chat tab and pre-fill with a review request
    import('./tabs.js').then(tabs => tabs.switchTab('chat'));

    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.value = `Please review this n8n workflow JSON for issues, missing parameters, or improvements. Fix any problems and return the corrected JSON:\n\n\`\`\`json\n${raw}\n\`\`\``;
        chatInput.dispatchEvent(new Event('input'));
        chatInput.focus();
    }
}

// Drag-and-drop support for import zone
function initImportDropZone() {
    const dropArea = document.getElementById('import-drop-area');
    if (!dropArea) return;

    ['dragenter', 'dragover'].forEach(evt => {
        dropArea.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropArea.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropArea.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropArea.classList.remove('drag-over');
        });
    });

    dropArea.addEventListener('drop', (e) => {
        const files = Array.from(e.dataTransfer?.files || []).filter(f => f.name.endsWith('.json'));
        if (files.length > 1) {
            batchImportWorkflows(files);
        } else if (files.length === 1) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                const textarea = document.getElementById('import-json-input');
                if (textarea) textarea.value = ev.target.result;
                validateImportedWorkflow();
            };
            reader.readAsText(files[0]);
        } else {
            const text = e.dataTransfer?.getData('text');
            if (text) {
                const textarea = document.getElementById('import-json-input');
                if (textarea) textarea.value = text;
                validateImportedWorkflow();
            }
        }
    });
}

// Initialize drop zone when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initImportDropZone);
} else {
    // DOM already loaded (module loaded after DOMContentLoaded)
    setTimeout(initImportDropZone, 0);
}

// ==================== Deployed Workflows ====================

export async function refreshDeployedWorkflows() {
    const container = document.getElementById('deployed-workflows-list');
    if (!container) return;

    container.innerHTML = '<div class="loading">Loading deployed workflows...</div>';

    try {
        const resp = await fetch(`${API}/n8n/workflows`);
        const data = await resp.json();

        marketplaceState.deployedWorkflows = data.workflows || [];
        updateDeployedCount();
        renderDeployedWorkflows();

    } catch (e) {
        console.error('Failed to load deployed workflows:', e);
        container.innerHTML = '<div class="empty-state">Failed to load workflows</div>';
    }
}

export function renderDeployedWorkflows() {
    const container = document.getElementById('deployed-workflows-list');
    if (!container) return;

    const workflows = marketplaceState.deployedWorkflows;

    if (workflows.length === 0) {
        container.innerHTML = '<div class="empty-state">No workflows deployed yet.<br>Go to Marketplace and click "Run" on any workflow to get started.</div>';
        updateBulkDeleteBar();
        return;
    }

    container.innerHTML = workflows.map(wf => `
        <div class="deployed-workflow-item" data-wf-id="${wf.id}">
            <label class="deployed-workflow-checkbox">
                <input type="checkbox" onchange="updateBulkDeleteBar()" value="${wf.id}" />
            </label>
            <div class="deployed-workflow-info">
                <div class="deployed-workflow-name">${escapeHtml(wf.name || 'Untitled')}</div>
                <div class="deployed-workflow-meta">
                    <span>${wf.node_count || 0} nodes</span>
                    <span>${wf.active ? 'Active' : 'Inactive'}</span>
                    <span>Updated: ${wf.updated_at ? new Date(wf.updated_at).toLocaleDateString() : 'Unknown'}</span>
                </div>
            </div>
            <div class="deployed-workflow-actions">
                <button class="btn-secondary btn-small" onclick="openParamEditor('${wf.id}', '${escapeHtml(wf.name || 'Workflow')}')">Edit</button>
                <button class="btn-primary btn-small" onclick="spawnWorkerFromDeployed('${wf.id}', '${escapeHtml(wf.name || 'Workflow')}')">Run</button>
                <button class="btn-danger btn-small" onclick="deleteDeployedWorkflow('${wf.id}', '${escapeHtml(wf.name || 'Workflow')}')" title="Remove">&times;</button>
            </div>
        </div>
    `).join('');
    updateBulkDeleteBar();
}

export function updateBulkDeleteBar() {
    const checked = document.querySelectorAll('#deployed-workflows-list input[type="checkbox"]:checked');
    let bar = document.getElementById('bulk-delete-bar');

    if (checked.length === 0) {
        if (bar) bar.classList.add('hidden');
        return;
    }

    if (!bar) {
        bar = document.createElement('div');
        bar.id = 'bulk-delete-bar';
        bar.className = 'bulk-delete-bar';
        const container = document.getElementById('deployed-workflows-list');
        container.parentNode.insertBefore(bar, container);
    }

    bar.classList.remove('hidden');
    bar.innerHTML = `
        <span>${checked.length} selected</span>
        <button class="btn-secondary btn-small" onclick="selectAllDeployed()">Select All</button>
        <button class="btn-secondary btn-small" onclick="deselectAllDeployed()">Deselect</button>
        <button class="btn-danger btn-small" onclick="deleteSelectedWorkflows()">Delete Selected</button>
    `;
}

export function selectAllDeployed() {
    document.querySelectorAll('#deployed-workflows-list input[type="checkbox"]').forEach(cb => cb.checked = true);
    updateBulkDeleteBar();
}

export function deselectAllDeployed() {
    document.querySelectorAll('#deployed-workflows-list input[type="checkbox"]').forEach(cb => cb.checked = false);
    updateBulkDeleteBar();
}

export async function deleteSelectedWorkflows() {
    const checked = document.querySelectorAll('#deployed-workflows-list input[type="checkbox"]:checked');
    const ids = Array.from(checked).map(cb => cb.value);

    if (ids.length === 0) return;
    if (!confirm(`Delete ${ids.length} workflow${ids.length > 1 ? 's' : ''}?`)) return;

    const toast = showToast(`Deleting ${ids.length} workflows...`, 'loading', 0);

    try {
        const resp = await fetch(`${API}/n8n/workflows/batch-delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids }),
        });
        const data = await resp.json();

        if (data.errors?.length > 0) {
            const errMsgs = data.errors.map(e => e.error || e.id).join(', ');
            updateToast(toast, `Deleted ${data.deleted}/${ids.length}. Errors: ${errMsgs}`, 'error', 8000);
        } else {
            updateToast(toast, `Deleted ${data.deleted} workflow${data.deleted > 1 ? 's' : ''}`, 'success');
        }
    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }

    await refreshDeployedWorkflows();
}

export async function spawnWorkerFromDeployed(workflowId, workflowName) {
    debugLog('spawnWorkerFromDeployed', `id=${workflowId} name="${workflowName}"`);
    const toast = showToast(`Starting "${workflowName}"...`, 'loading', 0);

    try {
        const resp = await fetch(`${API}/n8n/workers/spawn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_id: workflowId,
                mode: 'once'
            })
        });

        const data = await resp.json();

        if (data.worker) {
            updateToast(toast, `Worker started on port ${data.worker.port}`, 'success');
            await refreshQueueStatus();
            openN8nTab(data.worker.port, workflowId);
        } else {
            updateToast(toast, 'Failed: ' + (data.error || 'Unknown error'), 'error');
        }

    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

export async function deleteDeployedWorkflow(workflowId, workflowName) {
    if (!confirm(`Remove "${workflowName}" from deployed workflows?`)) return;

    try {
        const resp = await fetch(`${API}/n8n/workflow/${workflowId}`, { method: 'DELETE' });
        const data = await resp.json();

        if (data.success) {
            showToast(`Removed "${workflowName}"`, 'success');
            await refreshDeployedWorkflows();
        } else {
            showToast('Failed: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

export function openMainAdminWorkflow(workflowId) {
    openMainAdminTab();
}

// ==================== Workflow Parameter Editor ====================

const KNOWN_PARAM_OPTIONS = {
    'method': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'],
    'requestMethod': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'],
    'httpMethod': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'],
    'httpRequestMethod': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'],
    'authentication': ['none', 'basicAuth', 'digestAuth', 'headerAuth', 'oAuth1', 'oAuth2', 'queryAuth'],
    'authenticationType': ['none', 'basicAuth', 'digestAuth', 'headerAuth', 'oAuth1', 'oAuth2', 'queryAuth'],
    'authType': ['none', 'basic', 'bearer', 'apiKey', 'oauth2', 'digest'],
    'genericAuthType': ['httpBasicAuth', 'httpDigestAuth', 'httpHeaderAuth', 'httpQueryAuth', 'oAuth1Api', 'oAuth2Api', 'none'],
    'responseFormat': ['json', 'text', 'binary', 'file', 'autodetect'],
    'responseDataType': ['json', 'text', 'binary', 'file'],
    'outputFormat': ['json', 'csv', 'xml', 'html', 'text', 'binary'],
    'format': ['json', 'csv', 'xml', 'html', 'text', 'yaml', 'toml'],
    'dataType': ['string', 'number', 'boolean', 'array', 'object', 'binary', 'json'],
    'returnDataType': ['auto', 'json', 'text', 'binary'],
    'contentType': ['application/json', 'application/x-www-form-urlencoded', 'multipart/form-data', 'text/plain', 'text/html', 'text/xml', 'application/xml'],
    'mimeType': ['application/json', 'application/xml', 'text/plain', 'text/html', 'text/csv', 'application/pdf', 'image/png', 'image/jpeg'],
    'bodyContentType': ['json', 'form-urlencoded', 'multipart-form-data', 'raw'],
    'operation': ['create', 'read', 'update', 'delete', 'upsert', 'get', 'getAll', 'getMany', 'search', 'append', 'lookup', 'list', 'find', 'execute', 'send', 'upload', 'download'],
    'action': ['create', 'read', 'update', 'delete', 'get', 'list', 'search', 'send', 'receive'],
    'resource': ['user', 'users', 'contact', 'contacts', 'message', 'messages', 'channel', 'channels', 'file', 'files', 'folder', 'folders', 'sheet', 'row', 'record', 'task', 'project', 'email', 'event', 'calendar', 'deal', 'lead', 'opportunity', 'ticket', 'issue', 'comment', 'note', 'attachment', 'webhook', 'workflow'],
    'operation_filter': ['equal', 'notEqual', 'contains', 'notContains', 'startsWith', 'endsWith', 'regex', 'greater', 'greaterEqual', 'less', 'lessEqual', 'isEmpty', 'isNotEmpty', 'between', 'exists', 'notExists'],
    'condition': ['equal', 'notEqual', 'contains', 'notContains', 'startsWith', 'endsWith', 'regex', 'greater', 'greaterEqual', 'less', 'lessEqual', 'isEmpty', 'isNotEmpty'],
    'operator': ['equals', 'notEquals', 'contains', 'notContains', 'startsWith', 'endsWith', 'matchesRegex', 'greaterThan', 'lessThan', 'greaterThanOrEqual', 'lessThanOrEqual', 'isNull', 'isNotNull'],
    'filterType': ['contains', 'equals', 'notEquals', 'startsWith', 'endsWith', 'greaterThan', 'lessThan', 'isEmpty', 'isNotEmpty', 'regex'],
    'returnAll': ['true', 'false'],
    'rawData': ['true', 'false'],
    'resolveData': ['true', 'false'],
    'simplify': ['true', 'false'],
    'download': ['true', 'false'],
    'binaryData': ['true', 'false'],
    'continueOnFail': ['true', 'false'],
    'alwaysOutputData': ['true', 'false'],
    'encoding': ['utf-8', 'utf-16', 'ascii', 'base64', 'binary', 'hex', 'latin1', 'iso-8859-1'],
    'responseEncoding': ['utf-8', 'utf-16', 'ascii', 'base64', 'binary', 'latin1'],
    'unit': ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'],
    'timeUnit': ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months'],
    'interval': ['seconds', 'minutes', 'hours', 'days'],
    'triggerTimes': ['everyMinute', 'everyHour', 'everyDay', 'everyWeek', 'everyMonth', 'custom'],
    'cronExpression': ['* * * * *', '0 * * * *', '0 0 * * *', '0 0 * * 0', '0 0 1 * *'],
    'timezone': ['UTC', 'America/New_York', 'America/Los_Angeles', 'America/Chicago', 'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Asia/Tokyo', 'Asia/Shanghai', 'Australia/Sydney'],
    'mode': ['append', 'merge', 'mergeByKey', 'mergeByPosition', 'multiplex', 'chooseBranch', 'combine', 'passThrough', 'wait', 'keepKeyMatches', 'removeKeyMatches'],
    'mergeMode': ['append', 'merge', 'mergeByIndex', 'mergeByKey', 'multiplex', 'passThrough'],
    'combineMode': ['mergeByFields', 'mergeByPosition', 'multiplex', 'append'],
    'joinMode': ['inner', 'left', 'right', 'outer', 'cross'],
    'outputKey': ['data', 'result', 'output', 'response', 'body', 'json', 'binary'],
    'inputKey': ['data', 'json', 'binary', 'body', 'payload'],
    'propertyName': ['data', 'json', 'binary', 'result', 'output', 'response'],
    'httpMethod_webhook': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD'],
    'responseMode': ['onReceived', 'lastNode', 'responseNode'],
    'responseCode': ['200', '201', '204', '301', '302', '400', '401', '403', '404', '500', '502', '503'],
    'emailProvider': ['gmail', 'outlook', 'yahoo', 'smtp', 'imap', 'sendgrid', 'mailchimp', 'mailgun', 'ses'],
    'emailFormat': ['text', 'html', 'both'],
    'importance': ['low', 'normal', 'high'],
    'priority': ['low', 'normal', 'high', 'urgent'],
    'fileOperation': ['read', 'write', 'append', 'delete', 'copy', 'move', 'rename', 'list', 'create'],
    'binaryPropertyName': ['data', 'file', 'binary', 'attachment'],
    'sortOrder': ['ascending', 'descending', 'asc', 'desc'],
    'sortBy': ['name', 'date', 'size', 'type', 'id', 'createdAt', 'updatedAt'],
    'type_sort': ['string', 'number', 'date', 'boolean'],
    'model': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'gemini-pro', 'llama-2', 'mistral'],
    'temperature': ['0', '0.1', '0.2', '0.3', '0.5', '0.7', '0.8', '0.9', '1.0'],
    'responseType_ai': ['text', 'json', 'structured'],
    'databaseType': ['mysql', 'postgres', 'mssql', 'sqlite', 'mongodb', 'redis', 'oracle', 'mariadb'],
    'queryType': ['select', 'insert', 'update', 'delete', 'upsert', 'raw'],
    'parseMode': ['HTML', 'Markdown', 'MarkdownV2', 'None'],
    'chatType': ['private', 'group', 'supergroup', 'channel'],
    'messageType': ['text', 'photo', 'video', 'audio', 'document', 'sticker', 'location', 'contact'],
    'channelType': ['public', 'private', 'dm', 'group'],
    'messageFormat': ['plain', 'blocks', 'attachments', 'markdown'],
    'sheetDataOption': ['RAW', 'USER_ENTERED'],
    'valueInputOption': ['RAW', 'USER_ENTERED'],
    'valueRenderOption': ['FORMATTED_VALUE', 'UNFORMATTED_VALUE', 'FORMULA'],
    'fieldType': ['singleLineText', 'multilineText', 'number', 'checkbox', 'singleSelect', 'multipleSelects', 'date', 'email', 'url', 'phone', 'attachment', 'link', 'formula', 'rollup', 'lookup'],
    'transformOperation': ['rename', 'remove', 'set', 'move', 'copy', 'convert', 'split', 'join', 'trim', 'uppercase', 'lowercase', 'capitalize'],
    'stringCase': ['uppercase', 'lowercase', 'capitalize', 'titleCase', 'camelCase', 'snakeCase', 'kebabCase'],
    'errorHandling': ['stopWorkflow', 'continueOnFail', 'continueRegularOutput', 'continueErrorOutput'],
    'onError': ['stop', 'continue', 'continueRegularOutput', 'continueErrorOutput'],
    'paginationMode': ['none', 'offset', 'cursor', 'link', 'page'],
    'limitType': ['all', 'limit', 'first', 'last'],
    'retryOnFail': ['true', 'false'],
    'maxRetries': ['0', '1', '2', '3', '5', '10'],
    'waitBetweenRetries': ['0', '1000', '2000', '5000', '10000'],
};

const PARAM_OPTION_LABELS = {
    'GET': 'GET - Retrieve data',
    'POST': 'POST - Create/Send data',
    'PUT': 'PUT - Update/Replace data',
    'DELETE': 'DELETE - Remove data',
    'PATCH': 'PATCH - Partial update',
    'HEAD': 'HEAD - Get headers only',
    'OPTIONS': 'OPTIONS - Get allowed methods',
    'none': 'None - No authentication',
    'basicAuth': 'Basic Auth - Username/Password',
    'oAuth2': 'OAuth 2.0 - Token-based',
    'apiKey': 'API Key - Key in header/query',
    'headerAuth': 'Header Auth - Custom header',
    'json': 'JSON - JavaScript Object Notation',
    'text': 'Text - Plain text',
    'binary': 'Binary - Raw bytes',
    'autodetect': 'Auto-detect - Infer from response',
};

export async function openParamEditor(workflowId, workflowName) {
    debugLog('openParamEditor', `id=${workflowId} name="${workflowName}"`);
    const modal = document.getElementById('param-editor-modal');
    const title = document.getElementById('param-editor-title');
    const form = document.getElementById('param-editor-form');

    if (!modal || !form) return;

    title.textContent = `Edit: ${workflowName}`;
    form.innerHTML = '<div class="loading">Loading workflow parameters...</div>';
    modal.classList.remove('hidden');

    try {
        const resp = await fetch(`${API}/n8n/workflow/${workflowId}`);
        const data = await resp.json();

        if (!data.workflow) {
            form.innerHTML = '<div class="param-empty">Failed to load workflow</div>';
            return;
        }

        paramEditorState.workflowId = workflowId;
        paramEditorState.workflowData = data.workflow;

        renderParamForm(data.workflow);

    } catch (e) {
        form.innerHTML = `<div class="param-empty">Error: ${e.message}</div>`;
    }
}

function renderParamForm(workflow) {
    const form = document.getElementById('param-editor-form');
    const nodes = workflow.nodes || [];

    const editableNodes = nodes.filter(node => {
        const skip = ['n8n-nodes-base.start', 'n8n-nodes-base.noOp', 'n8n-nodes-base.stickyNote'];
        return !skip.includes(node.type) && node.parameters && Object.keys(node.parameters).length > 0;
    });

    if (editableNodes.length === 0) {
        form.innerHTML = '<div class="param-empty">No editable parameters found in this workflow</div>';
        return;
    }

    paramEditorState.editableParams = [];

    let html = '';
    editableNodes.forEach((node, nodeIndex) => {
        const params = node.parameters;
        const paramEntries = Object.entries(params).filter(([key, value]) => {
            const skipKeys = ['options', 'additionalFields', 'filters', 'updateFields', 'credentials'];
            if (skipKeys.includes(key)) return false;
            return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean';
        });

        if (paramEntries.length === 0) return;

        const nodeIcon = getNodeIcon(node.type);
        html += `<div class="param-node-section">`;
        html += `<div class="param-node-title">${nodeIcon} ${escapeHtml(node.name)} <span class="node-type">${node.type.split('.').pop()}</span></div>`;

        paramEntries.forEach(([key, value]) => {
            const paramId = `param-${nodeIndex}-${key}`;
            paramEditorState.editableParams.push({ nodeIndex, nodeName: node.name, key, paramId, originalType: typeof value });

            html += `<div class="param-group">`;
            html += `<label for="${paramId}">${formatParamName(key)}</label>`;

            const knownOptions = findKnownOptions(key, value);

            if (knownOptions) {
                html += `<select id="${paramId}" data-node="${nodeIndex}" data-key="${key}" class="param-dropdown">`;
                knownOptions.forEach(option => {
                    const selected = String(value).toLowerCase() === String(option).toLowerCase() ? 'selected' : '';
                    const label = PARAM_OPTION_LABELS[option] || option;
                    html += `<option value="${escapeHtml(option)}" ${selected}>${escapeHtml(label)}</option>`;
                });
                if (!knownOptions.some(opt => String(opt).toLowerCase() === String(value).toLowerCase())) {
                    html += `<option value="${escapeHtml(String(value))}" selected>${escapeHtml(String(value))} (current)</option>`;
                }
                html += `</select>`;
                html += `<span class="param-hint">Select from common options or keep current value</span>`;
            } else if (typeof value === 'boolean') {
                html += `<select id="${paramId}" data-node="${nodeIndex}" data-key="${key}" class="param-dropdown">
                    <option value="true" ${value ? 'selected' : ''}>Yes / Enabled</option>
                    <option value="false" ${!value ? 'selected' : ''}>No / Disabled</option>
                </select>`;
            } else if (typeof value === 'string' && value.length > 80) {
                html += `<textarea id="${paramId}" data-node="${nodeIndex}" data-key="${key}" rows="3">${escapeHtml(value)}</textarea>`;
            } else if (typeof value === 'string' && (value.includes('{{') || value.includes('{{'))) {
                html += `<input type="text" id="${paramId}" data-node="${nodeIndex}" data-key="${key}" value="${escapeHtml(String(value))}" class="param-expression">`;
                html += `<span class="param-hint">Expression detected - edit carefully</span>`;
            } else {
                const inputType = typeof value === 'number' ? 'number' : 'text';
                html += `<input type="${inputType}" id="${paramId}" data-node="${nodeIndex}" data-key="${key}" value="${escapeHtml(String(value))}">`;
            }

            html += `</div>`;
        });

        html += `</div>`;
    });

    form.innerHTML = html || '<div class="param-empty">No editable parameters found</div>';
}

function findKnownOptions(paramKey, currentValue) {
    if (KNOWN_PARAM_OPTIONS[paramKey]) {
        return KNOWN_PARAM_OPTIONS[paramKey];
    }

    const lowerKey = paramKey.toLowerCase();
    for (const [key, options] of Object.entries(KNOWN_PARAM_OPTIONS)) {
        if (key.toLowerCase() === lowerKey) {
            return options;
        }
    }

    const partialMatches = [
        { pattern: /method$/i, options: KNOWN_PARAM_OPTIONS['method'] },
        { pattern: /operation$/i, options: KNOWN_PARAM_OPTIONS['operation'] },
        { pattern: /format$/i, options: KNOWN_PARAM_OPTIONS['format'] },
        { pattern: /encoding$/i, options: KNOWN_PARAM_OPTIONS['encoding'] },
        { pattern: /mode$/i, options: KNOWN_PARAM_OPTIONS['mode'] },
        { pattern: /type$/i, options: null },
        { pattern: /auth/i, options: KNOWN_PARAM_OPTIONS['authentication'] },
        { pattern: /sort.*order/i, options: KNOWN_PARAM_OPTIONS['sortOrder'] },
        { pattern: /priority/i, options: KNOWN_PARAM_OPTIONS['priority'] },
        { pattern: /timezone/i, options: KNOWN_PARAM_OPTIONS['timezone'] },
    ];

    for (const { pattern, options } of partialMatches) {
        if (pattern.test(paramKey) && options) {
            return options;
        }
    }

    return null;
}

function getNodeIcon(nodeType) {
    const icons = {
        'httpRequest': '\u{1F310}',
        'webhook': '\u{1FA9D}',
        'code': '\u{1F4BB}',
        'function': '\u26A1',
        'if': '\u{1F500}',
        'switch': '\u{1F500}',
        'merge': '\u{1F517}',
        'set': '\u270F\uFE0F',
        'gmail': '\u{1F4E7}',
        'slack': '\u{1F4AC}',
        'discord': '\u{1F3AE}',
        'telegram': '\u2708\uFE0F',
        'googleSheets': '\u{1F4CA}',
        'airtable': '\u{1F4CB}',
        'notion': '\u{1F4DD}',
        'postgres': '\u{1F418}',
        'mysql': '\u{1F42C}',
        'mongodb': '\u{1F343}',
        'redis': '\u{1F534}',
        'ftp': '\u{1F4C1}',
        'ssh': '\u{1F510}',
        'cron': '\u23F0',
        'schedule': '\u{1F4C5}',
        'wait': '\u23F3',
        'filter': '\u{1F50D}',
        'sort': '\u2195\uFE0F',
        'splitInBatches': '\u{1F4E6}',
        'aggregate': '\u{1F4CA}',
        'openAi': '\u{1F916}',
        'anthropic': '\u{1F9E0}',
    };

    const typeName = nodeType.split('.').pop().toLowerCase();
    for (const [key, icon] of Object.entries(icons)) {
        if (typeName.includes(key.toLowerCase())) {
            return icon;
        }
    }
    return '\u2699\uFE0F';
}

function formatParamName(key) {
    return key
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .trim();
}

export function closeParamEditor() {
    const modal = document.getElementById('param-editor-modal');
    if (modal) modal.classList.add('hidden');
    paramEditorState.workflowId = null;
    paramEditorState.workflowData = null;
    paramEditorState.editableParams = [];
}

export function openAdvancedEditor() {
    if (paramEditorState.workflowId) {
        closeParamEditor();
        openMainAdminTab();
        showToast('Opening full n8n editor...', 'info');
    }
}

export async function saveWorkflowParams() {
    debugLog('saveWorkflowParams', `workflowId=${paramEditorState.workflowId}`);
    if (!paramEditorState.workflowId || !paramEditorState.workflowData) {
        showToast('No workflow loaded', 'error');
        return;
    }

    const toast = showToast('Saving changes...', 'loading', 0);

    try {
        const workflow = { ...paramEditorState.workflowData };
        const nodes = [...(workflow.nodes || [])];

        paramEditorState.editableParams.forEach(({ nodeIndex, key, paramId }) => {
            const input = document.getElementById(paramId);
            if (!input) return;

            let value = input.value;
            const originalValue = nodes[nodeIndex].parameters[key];

            if (typeof originalValue === 'boolean') {
                value = value === 'true';
            } else if (typeof originalValue === 'number') {
                value = parseFloat(value) || 0;
            }

            nodes[nodeIndex] = {
                ...nodes[nodeIndex],
                parameters: {
                    ...nodes[nodeIndex].parameters,
                    [key]: value
                }
            };
        });

        workflow.nodes = nodes;

        const resp = await fetch(`${API}/n8n/workflow/${paramEditorState.workflowId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workflow })
        });

        const data = await resp.json();

        if (data.success) {
            updateToast(toast, 'Workflow saved!', 'success');
            closeParamEditor();
            refreshDeployedWorkflows();
        } else {
            updateToast(toast, 'Save failed: ' + (data.error || 'Unknown error'), 'error');
        }

    } catch (e) {
        updateToast(toast, 'Error: ' + e.message, 'error');
    }
}

export function updateDeployedCount() {
    const badge = document.getElementById('deployed-count');
    if (badge) {
        badge.textContent = marketplaceState.deployedWorkflows.length;
    }
}

// ==================== Real Marketplace ====================

export async function loadMarketplaceCategories() {
    if (marketplaceState.loading) return;
    marketplaceState.loading = true;

    const categoriesContainer = document.getElementById('marketplace-categories');
    const statsEl = document.getElementById('marketplace-stats');

    try {
        const resp = await fetch(`${API}/marketplace/categories`);
        const data = await resp.json();

        marketplaceState.categories = data.categories || [];
        marketplaceState.categoriesLoaded = true;
        marketplaceState.totalWorkflows = data.total_workflows || 0;

        let html = `
            <div class="category-item active" data-category="all" onclick="selectMarketplaceCategory('all')">
                <span class="category-icon">&#128218;</span>
                <span class="category-name">All Workflows</span>
                <span class="category-count">${marketplaceState.totalWorkflows}</span>
            </div>
        `;

        marketplaceState.categories.forEach(cat => {
            const icon = getCategoryIcon(cat.name);
            html += `
                <div class="category-item" data-category="${escapeHtml(cat.name)}" onclick="selectMarketplaceCategory('${escapeHtml(cat.name)}')">
                    <span class="category-icon">${icon}</span>
                    <span class="category-name">${escapeHtml(cat.name)}</span>
                    <span class="category-count">${cat.count || 0}</span>
                </div>
            `;
        });

        categoriesContainer.innerHTML = html;

        if (statsEl) {
            statsEl.textContent = `${marketplaceState.totalWorkflows.toLocaleString()} workflows available`;
        }

        await loadCategoryWorkflows('all');

    } catch (e) {
        console.error('Failed to load marketplace categories:', e);
        categoriesContainer.innerHTML = `
            <div class="category-item active" data-category="all">
                <span class="category-icon">&#128218;</span>
                <span class="category-name">All Workflows</span>
            </div>
            <div class="loading-categories" style="color: var(--error);">Failed to load categories</div>
        `;
    } finally {
        marketplaceState.loading = false;
    }
}

export function formatCategoryName(rawName) {
    const nameMap = {
        'activecampaign': 'ActiveCampaign',
        'apitemplateio': 'APITemplate.io',
        'airtable': 'Airtable',
        'asana': 'Asana',
        'aws': 'AWS',
        'baserow': 'Baserow',
        'clickup': 'ClickUp',
        'discord': 'Discord',
        'dropbox': 'Dropbox',
        'facebook': 'Facebook',
        'github': 'GitHub',
        'gitlab': 'GitLab',
        'gmail': 'Gmail',
        'google': 'Google',
        'googlesheets': 'Google Sheets',
        'googledrive': 'Google Drive',
        'hubspot': 'HubSpot',
        'http': 'HTTP/API',
        'instagram': 'Instagram',
        'jira': 'Jira',
        'linkedin': 'LinkedIn',
        'mailchimp': 'Mailchimp',
        'mattermost': 'Mattermost',
        'microsoft': 'Microsoft',
        'monday': 'Monday.com',
        'mysql': 'MySQL',
        'notion': 'Notion',
        'openai': 'OpenAI',
        'outlook': 'Outlook',
        'pipedrive': 'Pipedrive',
        'postgres': 'PostgreSQL',
        'quickbooks': 'QuickBooks',
        'redis': 'Redis',
        'salesforce': 'Salesforce',
        'sendgrid': 'SendGrid',
        'shopify': 'Shopify',
        'slack': 'Slack',
        'stripe': 'Stripe',
        'telegram': 'Telegram',
        'trello': 'Trello',
        'twilio': 'Twilio',
        'twitter': 'Twitter/X',
        'webhook': 'Webhooks',
        'wordpress': 'WordPress',
        'youtube': 'YouTube',
        'zapier': 'Zapier',
        'zendesk': 'Zendesk',
        'zoom': 'Zoom',
    };

    const lower = rawName.toLowerCase();
    if (nameMap[lower]) {
        return nameMap[lower];
    }

    return rawName
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

export function getCategoryIcon(category) {
    const icons = {
        // Marketplace categories
        'ai': '&#129302;',
        'multimodal ai': '&#127912;',
        'marketing': '&#128227;',
        'ai summarization': '&#128221;',
        'content creation': '&#9999;',
        'sales': '&#128176;',
        'it ops': '&#128295;',
        'document ops': '&#128196;',
        'ai chatbot': '&#128488;',
        'other': '&#128300;',
        'engineering': '&#9881;',
        'hr': '&#128101;',
        'finance': '&#128178;',
        'support': '&#127384;',
        'productivity': '&#9889;',
        'data': '&#128202;',
        'devops': '&#128640;',
        'security': '&#128274;',
        'education': '&#127891;',
        'communication': '&#128172;',
        'design': '&#127912;',
        'analytics': '&#128200;',
        'automation': '&#9881;',
        'building ai': '&#129520;',
        // Service-level icons
        'discord': '&#128172;',
        'gmail': '&#128231;',
        'slack': '&#128172;',
        'airtable': '&#128202;',
        'google': '&#127760;',
        'notion': '&#128221;',
        'telegram': '&#9992;',
        'twitter': '&#128038;',
        'youtube': '&#127909;',
        'openai': '&#129302;',
        'github': '&#128736;',
        'trello': '&#128203;',
        'shopify': '&#128722;',
        'stripe': '&#128179;',
        'webhook': '&#128279;',
        'http': '&#127760;',
        'email': '&#128231;',
        'mysql': '&#128451;',
        'postgres': '&#128451;',
        'api': '&#128268;',
        'aws': '&#9729;',
        'microsoft': '&#128187;',
        'outlook': '&#128231;',
        'salesforce': '&#9729;',
        'hubspot': '&#128200;',
        'jira': '&#128203;',
        'asana': '&#128203;',
        'clickup': '&#128203;',
        'monday': '&#128203;',
        'zoom': '&#127909;',
        'dropbox': '&#128193;',
        'drive': '&#128193;',
        'sheets': '&#128202;',
        'facebook': '&#128101;',
        'instagram': '&#128247;',
        'linkedin': '&#128101;',
        'twilio': '&#128222;',
        'sendgrid': '&#128231;',
        'mailchimp': '&#128231;',
        'wordpress': '&#128221;',
        'default': '&#128196;',
    };

    const lower = category.toLowerCase();
    // Exact match first (for marketplace categories like "AI", "IT Ops")
    if (icons[lower]) return icons[lower];
    // Partial match fallback (for service names in workflow tags)
    for (const [key, icon] of Object.entries(icons)) {
        if (key !== 'default' && lower.includes(key)) {
            return icon;
        }
    }
    return icons.default;
}

export async function selectMarketplaceCategory(category) {
    marketplaceState.selectedCategory = category;

    document.querySelectorAll('.marketplace-categories .category-item').forEach(el => {
        el.classList.toggle('active', el.dataset.category === category);
    });

    await loadCategoryWorkflows(category);
}

export async function loadCategoryWorkflows(category) {
    const gridContainer = document.getElementById('workflows-grid');
    gridContainer.innerHTML = '<div class="loading">Loading workflows...</div>';

    try {
        let url = `${API}/marketplace/workflows?limit=100`;
        if (category && category !== 'all') {
            url += `&category=${encodeURIComponent(category)}`;
        }

        const resp = await fetch(url);
        const data = await resp.json();

        marketplaceState.workflows = data.workflows || [];
        marketplaceState.workflowsTotal = data.total || 0;
        marketplaceState.workflowsOffset = 0;

        renderWorkflowsGrid();

    } catch (e) {
        console.error('Failed to load workflows:', e);
        gridContainer.innerHTML = '<div class="empty-state">Failed to load workflows</div>';
    }
}

export function renderWorkflowsGrid() {
    const gridContainer = document.getElementById('workflows-grid');
    const query = marketplaceState.searchQuery.toLowerCase();

    let workflows = marketplaceState.workflows;

    // Only apply client-side filter if results aren't already from API search
    if (query && !marketplaceState._apiSearchActive) {
        workflows = workflows.filter(w =>
            (w.name || '').toLowerCase().includes(query) ||
            (w.description || '').toLowerCase().includes(query) ||
            (w.integrations || []).join(' ').toLowerCase().includes(query)
        );
    }

    // Update stats counter to reflect search results vs total
    const statsEl = document.getElementById('marketplace-stats');
    const searchInput = document.getElementById('marketplace-search');
    if (statsEl) {
        if (query && marketplaceState._apiSearchActive) {
            statsEl.innerHTML = `${workflows.length} results for &ldquo;${escapeHtml(marketplaceState.searchQuery)}&rdquo;`;
            statsEl.classList.remove('searching');
            if (searchInput) searchInput.classList.remove('searching');
        } else if (query) {
            statsEl.innerHTML = `<span class="search-spinner"></span> Searching marketplace...`;
            statsEl.classList.add('searching');
            if (searchInput) searchInput.classList.add('searching');
        } else {
            statsEl.textContent = `${marketplaceState.totalWorkflows.toLocaleString()} workflows available`;
            statsEl.classList.remove('searching');
            if (searchInput) searchInput.classList.remove('searching');
        }
    }

    if (workflows.length === 0) {
        gridContainer.innerHTML = '<div class="empty-state">No workflows found</div>';
        return;
    }

    gridContainer.innerHTML = workflows.map(w => {
        const triggerIcon = getTriggerIcon(w.trigger_type);
        const triggerLabel = formatTriggerType(w.trigger_type);
        const complexityClass = (w.complexity || 'medium').toLowerCase();
        const description = w.description ? escapeHtml(w.description).substring(0, 100) + (w.description.length > 100 ? '...' : '') : '';
        const integrations = (w.integrations || []).slice(0, 3);

        return `
            <div class="workflow-card" onclick="previewWorkflowById('${escapeHtml(w.id)}')">
                <div class="workflow-card-content">
                    <div class="workflow-card-header">
                        <div class="workflow-card-title">${escapeHtml(w.name || 'Untitled')}</div>
                        <span class="complexity-badge ${complexityClass}">${w.complexity || 'medium'}</span>
                    </div>
                    ${description ? `<div class="workflow-card-description">${description}</div>` : ''}
                    <div class="workflow-card-meta">
                        <span class="workflow-card-trigger">${triggerIcon} ${triggerLabel}</span>
                        <span class="workflow-card-nodes">${w.node_count || 0} nodes</span>
                        ${integrations.length > 0 ? `<span class="workflow-card-integrations">${integrations.join(', ')}</span>` : ''}
                    </div>
                </div>
                <div class="workflow-card-actions">
                    <button class="btn-deploy" onclick="event.stopPropagation(); quickDeployWorkflowById('${escapeHtml(w.id)}')" title="Auto-starts n8n servers and runs this workflow">
                        Run
                    </button>
                </div>
            </div>
        `;
    }).join('');

    if (marketplaceState.workflowsTotal > marketplaceState.workflows.length) {
        gridContainer.innerHTML += `
            <div class="load-more-container">
                <button class="btn-secondary" onclick="loadMoreWorkflows()">
                    Load More (${marketplaceState.workflows.length} of ${marketplaceState.workflowsTotal})
                </button>
            </div>
        `;
    }
}

export function formatTriggerType(trigger) {
    const labels = {
        'webhook': 'Webhook',
        'schedule': 'Scheduled',
        'email': 'Email Trigger',
        'manual': 'Manual',
    };
    return labels[trigger] || trigger;
}

export function getTriggerIcon(triggerType) {
    const icons = {
        'webhook': '&#128279;',
        'schedule': '&#128197;',
        'email': '&#128231;',
        'manual': '&#9654;',
    };
    return icons[triggerType] || '&#128196;';
}

let _searchDebounce = null;

export function searchMarketplace() {
    const input = document.getElementById('marketplace-search');
    const query = input ? input.value.trim() : '';
    marketplaceState.searchQuery = query;
    marketplaceState._apiSearchActive = false;

    // Instant client-side filter for responsiveness
    renderWorkflowsGrid();

    // Debounced backend search for queries >= 2 chars
    clearTimeout(_searchDebounce);
    if (query.length >= 2) {
        _searchDebounce = setTimeout(() => searchMarketplaceAPI(query), 400);
    } else if (query.length === 0) {
        // Cleared search — restore original category listing
        loadCategoryWorkflows(marketplaceState.selectedCategory || 'all');
    }
}

async function searchMarketplaceAPI(query) {
    try {
        const category = marketplaceState.selectedCategory;
        let url = `${API}/marketplace/search?q=${encodeURIComponent(query)}&limit=100`;
        if (category && category !== 'all') {
            url += `&category=${encodeURIComponent(category)}`;
        }
        const resp = await fetch(url);
        if (!resp.ok) return; // Silently fail, client-side filter still showing

        const data = await resp.json();
        const results = data.results || [];

        // Only update if the query hasn't changed while we were fetching
        if (marketplaceState.searchQuery === query) {
            marketplaceState.workflows = results;
            marketplaceState.workflowsTotal = data.count || results.length;
            marketplaceState.workflowsOffset = 0;
            marketplaceState._apiSearchActive = true;
            renderWorkflowsGrid();
        }
    } catch (e) {
        console.error('Marketplace search failed:', e);
        // Client-side filter results already showing — no disruption
    }
}

// Simplified getNodeIcon for marketplace preview (uses HTML entities)
function getNodeIconSimple(nodeType) {
    const type = (nodeType || '').toLowerCase();
    if (type.includes('webhook')) return '&#128279;';
    if (type.includes('http')) return '&#127760;';
    if (type.includes('schedule') || type.includes('cron')) return '&#128197;';
    if (type.includes('email') || type.includes('gmail') || type.includes('imap')) return '&#128231;';
    if (type.includes('discord')) return '&#128172;';
    if (type.includes('slack')) return '&#128172;';
    if (type.includes('telegram')) return '&#9992;';
    if (type.includes('openai') || type.includes('ai') || type.includes('llm')) return '&#129302;';
    if (type.includes('if') || type.includes('switch')) return '&#10140;';
    if (type.includes('code') || type.includes('function')) return '&#128187;';
    if (type.includes('set')) return '&#128295;';
    if (type.includes('merge')) return '&#128256;';
    if (type.includes('split')) return '&#128257;';
    if (type.includes('database') || type.includes('postgres') || type.includes('mysql')) return '&#128451;';
    if (type.includes('file') || type.includes('drive')) return '&#128193;';
    if (type.includes('sheet')) return '&#128202;';
    return '&#9881;';
}

export async function previewWorkflowById(workflowId) {
    const modal = document.getElementById('workflow-preview-modal');
    const nameEl = document.getElementById('preview-workflow-name');
    const metaEl = document.getElementById('preview-workflow-meta');
    const jsonEl = document.getElementById('preview-workflow-json');
    const nodesListEl = document.getElementById('preview-nodes-list');
    const descSection = document.getElementById('preview-description-section');
    const descEl = document.getElementById('preview-workflow-description');
    const integrationsSection = document.getElementById('preview-integrations-section');
    const integrationsList = document.getElementById('preview-integrations-list');

    if (!modal) return;

    modal.classList.remove('hidden');
    if (jsonEl) jsonEl.textContent = 'Loading...';
    if (nodesListEl) nodesListEl.innerHTML = '<div class="loading">Loading...</div>';
    if (descSection) descSection.classList.add('hidden');
    if (integrationsSection) integrationsSection.classList.add('hidden');

    try {
        const resp = await fetch(`${API}/marketplace/workflow/${encodeURIComponent(workflowId)}`);
        const data = await resp.json();

        marketplaceState.previewWorkflow = {
            id: workflowId,
            ...data.workflow
        };

        const workflow = data.workflow;
        const meta = workflow.metadata || {};
        const triggerLabel = formatTriggerType(meta.trigger_type || 'manual');

        nameEl.textContent = meta.name || 'Untitled Workflow';
        metaEl.textContent = `${meta.category || 'Unknown'} | ${triggerLabel} | ${meta.node_count || 0} nodes | ${meta.complexity || 'medium'} complexity`;

        if (meta.description && descSection && descEl) {
            descEl.innerHTML = renderSimpleMarkdown(meta.description);
            descSection.classList.remove('hidden');
        }

        if (meta.integrations?.length > 0 && integrationsSection && integrationsList) {
            integrationsList.innerHTML = meta.integrations.map(int => {
                const icon = getCategoryIcon(int);
                return `<span class="integration-tag">${icon} ${escapeHtml(int)}</span>`;
            }).join('');
            integrationsSection.classList.remove('hidden');
        }

        if (nodesListEl && workflow.json?.nodes) {
            const nodes = workflow.json.nodes;
            nodesListEl.innerHTML = nodes.map(node => {
                const nodeIcon = getNodeIconSimple(node.type);
                const nodeName = node.name || node.type.split('.').pop();
                const nodeType = node.type.replace('n8n-nodes-base.', '');
                return `
                    <div class="node-item">
                        <span class="node-icon">${nodeIcon}</span>
                        <span class="node-name">${escapeHtml(nodeName)}</span>
                        <span class="node-type">${escapeHtml(nodeType)}</span>
                    </div>
                `;
            }).join('');
        }

        if (jsonEl) jsonEl.textContent = JSON.stringify(workflow.json, null, 2);

    } catch (e) {
        console.error('Failed to load workflow:', e);
        if (jsonEl) jsonEl.textContent = 'Failed to load workflow: ' + e.message;
        if (nodesListEl) nodesListEl.innerHTML = '<div class="error">Failed to load</div>';
    }
}

export async function previewWorkflow(category, filename) {
    const wf = marketplaceState.workflows.find(w =>
        w.category === category && (w.filename === filename || w.id?.includes(filename))
    );
    if (wf?.id) {
        await previewWorkflowById(wf.id);
    } else {
        log('Workflow not found', 'error');
    }
}

export function closeWorkflowPreviewModal() {
    const modal = document.getElementById('workflow-preview-modal');
    if (modal) modal.classList.add('hidden');
    marketplaceState.previewWorkflow = null;
}

export function copyWorkflowJson() {
    if (!marketplaceState.previewWorkflow?.json) return;

    navigator.clipboard.writeText(JSON.stringify(marketplaceState.previewWorkflow.json, null, 2))
        .then(() => log('Workflow JSON copied to clipboard', 'success'))
        .catch(e => log('Failed to copy: ' + e.message, 'error'));
}

export function downloadWorkflowJson() {
    if (!marketplaceState.previewWorkflow?.json) return;

    const blob = new Blob([JSON.stringify(marketplaceState.previewWorkflow.json, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = marketplaceState.previewWorkflow.filename || 'workflow.json';
    a.click();
    URL.revokeObjectURL(url);
}

export async function deployAndRunWorkflow(mode = 'once') {
    if (!marketplaceState.previewWorkflow?.json) {
        showToast('No workflow to deploy', 'error');
        return;
    }

    const workflowName = marketplaceState.previewWorkflow.metadata?.name || 'Workflow';
    debugLog('deployAndRunWorkflow', `mode=${mode} name="${workflowName}"`);
    const toast = showToast(`Deploying "${workflowName}"...`, 'loading', 0);

    const deployBtns = document.querySelectorAll('.deploy-dropdown button, .btn-deploy');
    deployBtns.forEach(btn => btn.disabled = true);

    try {
        const resp = await fetch(`${API}/n8n/deploy-and-run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_json: marketplaceState.previewWorkflow.json,
                mode: mode
            })
        });

        const data = await resp.json();

        if (data.worker) {
            updateToast(toast, `Deployed! Opening workflow...`, 'success');
            closeWorkflowPreviewModal();

            await refreshQueueStatus();

            setTimeout(() => {
                openN8nTab(data.worker.port, data.workflow_id);
            }, 500);
        } else {
            updateToast(toast, 'Deploy failed: ' + (data.error || 'Unknown error'), 'error');
        }

    } catch (e) {
        updateToast(toast, 'Deploy error: ' + e.message, 'error');
    } finally {
        deployBtns.forEach(btn => btn.disabled = false);
    }
}

export async function quickDeployWorkflowById(workflowId) {
    debugLog('quickDeployWorkflowById', `id=${workflowId}`);
    log(`Fetching workflow...`, 'info');

    try {
        const resp = await fetch(`${API}/marketplace/workflow/${encodeURIComponent(workflowId)}`);
        const data = await resp.json();

        if (!data.workflow?.json) {
            log('Failed to load workflow JSON', 'error');
            return;
        }

        marketplaceState.previewWorkflow = {
            id: workflowId,
            ...data.workflow
        };

        // Pre-deploy inspection: check for missing credentials & placeholders
        log('Inspecting workflow...', 'info');
        try {
            const inspectResp = await fetch(`${API}/workflows/inspect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ workflow_json: data.workflow.json })
            });
            const inspection = await inspectResp.json();

            if (inspection.success && inspection.summary?.ready_to_deploy) {
                // Clean workflow — fast-path deploy
                debugLog('quickDeploy', 'Workflow is clean, fast-path deploy');
                await deployAndRunWorkflow('once');
            } else if (inspection.success) {
                // Issues found — show inspection panel
                debugLog('quickDeploy', `Needs config: ${inspection.summary?.credentials_missing || 0} creds missing, ${inspection.summary?.placeholders_found || 0} placeholders`);
                showInspectionPanel(data.workflow, inspection);
            } else {
                // Inspection failed — deploy anyway (fallback to old behavior)
                debugLog('quickDeploy', 'Inspection failed, deploying anyway');
                await deployAndRunWorkflow('once');
            }
        } catch (inspectErr) {
            // Inspection endpoint not available — deploy anyway
            debugLog('quickDeploy', `Inspect error: ${inspectErr.message}, deploying anyway`);
            await deployAndRunWorkflow('once');
        }

    } catch (e) {
        log('Failed to load workflow: ' + e.message, 'error');
    }
}


// ==================== Workflow Inspection Panel ====================

export function showInspectionPanel(workflow, inspection) {
    const modal = document.getElementById('workflow-inspection-modal');
    if (!modal) return;

    const name = workflow.metadata?.name || 'Workflow';
    const summary = inspection.summary || {};
    const credentials = inspection.credentials || [];
    const placeholders = inspection.placeholders || [];

    // Store inspection data for later use
    marketplaceState.currentInspection = inspection;

    // Header
    document.getElementById('inspection-workflow-name').textContent = `Configure: ${name}`;
    document.getElementById('inspection-workflow-meta').textContent =
        `${summary.node_count || 0} nodes | ${summary.trigger_type || 'manual'} trigger`;

    // Summary
    const summaryEl = document.getElementById('inspection-summary');
    const missingCount = summary.credentials_missing || 0;
    const needsSetupCount = summary.credentials_needs_setup || 0;
    const placeholderCount = summary.placeholders_found || 0;
    const issueCount = missingCount + needsSetupCount;
    summaryEl.innerHTML = `
        ${issueCount > 0 ? `<div class="stat"><strong>${issueCount}</strong> credentials need setup</div>` : ''}
        ${(summary.credentials_available || 0) > 0 ? `<div class="stat"><strong>${summary.credentials_available}</strong> available locally</div>` : ''}
        ${(summary.credentials_pre_configured || 0) > 0 ? `<div class="stat"><strong>${summary.credentials_pre_configured}</strong> verified</div>` : ''}
        ${placeholderCount > 0 ? `<div class="stat"><strong>${placeholderCount}</strong> placeholders to fill</div>` : ''}
    `;

    // Credentials section
    const credsEl = document.getElementById('inspection-credentials');
    if (credentials.length > 0) {
        credsEl.innerHTML = '<h5>Credentials</h5>' + credentials.map((cred, i) => {
            let control = '';
            if (cred.status === 'available' && cred.available_credentials?.length) {
                const options = cred.available_credentials.map(c =>
                    `<option value="${c.id}">${c.name}</option>`
                ).join('');
                control = `<select data-cred-type="${cred.credential_type}" data-cred-idx="${i}">${options}</select>`;
            } else if (cred.status === 'pre-configured') {
                control = '<span class="inspection-badge ready">Verified</span>';
            } else if (cred.status === 'needs_setup') {
                control = '<span class="inspection-badge missing">Needs setup in n8n</span>';
            } else {
                control = '<span class="inspection-badge missing">Not configured in n8n</span>';
            }

            return `<div class="inspection-row">
                <span class="node-name">${cred.node_name}</span>
                <span class="cred-type">${cred.credential_type}</span>
                ${control}
            </div>`;
        }).join('');
    } else {
        credsEl.innerHTML = '';
    }

    // Placeholders section
    const placeholdersEl = document.getElementById('inspection-placeholders');
    if (placeholders.length > 0) {
        placeholdersEl.innerHTML = '<h5>Placeholders</h5>' + placeholders.map((ph, i) => {
            const paramName = ph.param_path?.split('.').pop() || 'value';
            return `<div class="inspection-row">
                <span class="node-name">${ph.node_name}</span>
                <input type="text"
                    data-ph-node="${ph.node_name}"
                    data-ph-param="${paramName}"
                    data-ph-idx="${i}"
                    placeholder="${ph.current_value || paramName}"
                    value="" />
            </div>`;
        }).join('');
    } else {
        placeholdersEl.innerHTML = '';
    }

    modal.classList.remove('hidden');
}

export function closeInspectionPanel() {
    const modal = document.getElementById('workflow-inspection-modal');
    if (modal) modal.classList.add('hidden');
    marketplaceState.currentInspection = null;
}

export async function configureAndDeploy() {
    const inspection = marketplaceState.currentInspection;
    const workflow = marketplaceState.previewWorkflow;
    if (!inspection || !workflow?.json) {
        log('No workflow to configure', 'error');
        return;
    }

    // Collect credential selections
    const credentialMap = {};
    document.querySelectorAll('#inspection-credentials select').forEach(sel => {
        const credType = sel.dataset.credType;
        if (credType && sel.value) {
            credentialMap[credType] = sel.value;
        }
    });

    // Collect placeholder values
    const paramOverrides = {};
    document.querySelectorAll('#inspection-placeholders input').forEach(input => {
        const nodeName = input.dataset.phNode;
        const paramName = input.dataset.phParam;
        if (nodeName && paramName && input.value.trim()) {
            if (!paramOverrides[nodeName]) paramOverrides[nodeName] = {};
            paramOverrides[nodeName][paramName] = input.value.trim();
        }
    });

    // Configure the workflow
    const hasChanges = Object.keys(credentialMap).length > 0 || Object.keys(paramOverrides).length > 0;

    if (hasChanges) {
        log('Configuring workflow...', 'info');
        try {
            const resp = await fetch(`${API}/workflows/configure`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    workflow_json: workflow.json,
                    credential_map: credentialMap,
                    param_overrides: paramOverrides
                })
            });
            const result = await resp.json();

            if (result.success && result.workflow_json) {
                marketplaceState.previewWorkflow.json = result.workflow_json;
                log(`Applied ${result.changes_count || 0} changes`, 'success');
            } else {
                log('Configure failed: ' + (result.error || 'Unknown error'), 'error');
                return;
            }
        } catch (e) {
            log('Configure error: ' + e.message, 'error');
            return;
        }
    }

    // Deploy
    closeInspectionPanel();
    await deployAndRunWorkflow('once');
}

export function openAgentForWorkflow() {
    const inspection = marketplaceState.currentInspection;
    const workflow = marketplaceState.previewWorkflow;
    if (!inspection || !workflow) return;

    const name = workflow.metadata?.name || 'Unknown workflow';
    const summary = inspection.summary || {};

    // Build a descriptive prompt for the agent
    let prompt = `I have a marketplace workflow "${name}" that needs configuration before deployment.\n\n`;

    const missingCreds = (inspection.credentials || []).filter(c => c.status === 'missing');
    if (missingCreds.length > 0) {
        prompt += `Missing credentials:\n`;
        missingCreds.forEach(c => {
            prompt += `- ${c.node_name} needs "${c.credential_type}"\n`;
        });
        prompt += '\n';
    }

    const placeholders = inspection.placeholders || [];
    if (placeholders.length > 0) {
        prompt += `Placeholders to fill:\n`;
        placeholders.forEach(p => {
            prompt += `- ${p.node_name}: ${p.param_path} (current: "${p.current_value}")\n`;
        });
        prompt += '\n';
    }

    prompt += 'Please help me configure the missing credentials and placeholder values, then deploy the workflow.';

    // Close inspection panel
    closeInspectionPanel();

    // Switch to Chat tab
    import('./tabs.js').then(tabs => tabs.switchTab('chat'));

    // Enable agent mode + autonomous on the active panel
    setTimeout(async () => {
        const { state: s, getPanelDomId } = await import('./state.js');
        const pid = s.activePanelId;
        if (!pid) return;

        const agentToggle = document.getElementById(getPanelDomId(pid, 'agent-toggle'));
        const autoToggle = document.getElementById(getPanelDomId(pid, 'autonomous-toggle'));
        if (agentToggle && !agentToggle.checked) {
            agentToggle.checked = true;
            agentToggle.dispatchEvent(new Event('change', { bubbles: true }));
        }
        if (autoToggle && !autoToggle.checked) {
            autoToggle.checked = true;
            autoToggle.dispatchEvent(new Event('change', { bubbles: true }));
        }

        // Select workflow_builder persona
        const personaSelect = document.getElementById(getPanelDomId(pid, 'persona-select'));
        if (personaSelect) {
            personaSelect.value = 'workflow_builder';
            personaSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }

        // Fill prompt
        import('./agent.js').then(agent => agent.fillPanelAgentPrompt(pid, prompt));
    }, 200);
}

export async function quickDeployWorkflow(category, filename) {
    const wf = marketplaceState.workflows.find(w =>
        w.category === category && (w.filename === filename || w.id?.includes(filename))
    );
    if (wf?.id) {
        await quickDeployWorkflowById(wf.id);
    } else {
        log('Workflow not found', 'error');
    }
}

export async function loadMoreWorkflows() {
    const currentCount = marketplaceState.workflows.length;
    const category = marketplaceState.selectedCategory;

    let url = `${API}/marketplace/workflows?limit=100&offset=${currentCount}`;
    if (category && category !== 'all') {
        url += `&category=${encodeURIComponent(category)}`;
    }

    try {
        const resp = await fetch(url);
        const data = await resp.json();

        marketplaceState.workflows = [...marketplaceState.workflows, ...(data.workflows || [])];
        renderWorkflowsGrid();

    } catch (e) {
        log('Failed to load more workflows: ' + e.message, 'error');
    }
}

// ==================== Legacy Template System ====================

const QUICK_RECIPES = [
    {
        id: 'webhook_llm_respond',
        name: 'Webhook + AI Response',
        description: 'Receive HTTP requests, process with your local AI, and send back intelligent responses.',
        icon: '&#128279;',
        category: 'recipes',
        nodes: ['webhook', 'local_llm_chat', 'respond_webhook'],
        config: { webhook_path: 'ai-chat' }
    },
    {
        id: 'scheduled_summary',
        name: 'Daily AI Summary',
        description: 'Fetch data on a schedule, generate an AI summary, and send to Discord or Slack.',
        icon: '&#128197;',
        category: 'recipes',
        nodes: ['schedule', 'http_request', 'summarize', 'discord_webhook'],
        config: { cron: '0 9 * * *' }
    },
    {
        id: 'discord_ai_bot',
        name: 'Discord AI Bot',
        description: 'Listen for Discord messages via webhook and respond with AI-generated content.',
        icon: '&#128172;',
        category: 'recipes',
        nodes: ['webhook', 'local_llm_chat', 'discord_webhook'],
        config: { webhook_path: 'discord-bot' }
    },
    {
        id: 'sentiment_classifier',
        name: 'Sentiment Classifier',
        description: 'Analyze incoming text sentiment and route to different actions based on result.',
        icon: '&#128200;',
        category: 'recipes',
        nodes: ['webhook', 'classify', 'if_condition'],
        config: { categories: 'positive, negative, neutral' }
    },
    {
        id: 'email_summarizer',
        name: 'Email Summarizer',
        description: 'Monitor inbox for new emails, summarize them with AI, and forward digests.',
        icon: '&#128231;',
        category: 'recipes',
        nodes: ['email_imap', 'summarize', 'slack_webhook'],
        config: {}
    },
    {
        id: 'json_extractor',
        name: 'Data Extractor',
        description: 'Receive unstructured text, extract structured JSON data using AI.',
        icon: '&#128203;',
        category: 'recipes',
        nodes: ['webhook', 'extract_json', 'respond_webhook'],
        config: { schema: 'name, email, phone, company' }
    }
];

const CATEGORY_ICONS = {
    triggers: '&#128279;',
    ai: '&#129302;',
    actions: '&#128640;',
    data: '&#128202;',
    recipes: '&#9889;',
    examples: '&#128218;'
};

const TEMPLATE_ICONS = {
    webhook: '&#128279;',
    schedule: '&#128197;',
    manual: '&#9654;',
    email_imap: '&#128231;',
    local_llm_chat: '&#129302;',
    summarize: '&#128196;',
    classify: '&#127991;',
    extract_json: '&#128203;',
    discord_webhook: '&#128172;',
    slack_webhook: '&#128172;',
    http_request: '&#127760;',
    respond_webhook: '&#10132;',
    set_field: '&#128295;',
    code: '&#128187;',
    if_condition: '&#10140;',
    parse_json: '&#128196;'
};

export async function loadWorkflowTemplates() {
    try {
        const resp = await fetch(`${API}/workflows/templates`);
        const data = await resp.json();
        workflowState.templates = data.templates;
        renderMarketplace();
    } catch (e) {
        console.error('Failed to load templates:', e);
        const grid = document.getElementById('template-grid');
        if (grid) {
            grid.innerHTML =
                '<div class="empty-templates"><div class="empty-icon">&#128533;</div><h4>Failed to load templates</h4><p>Check if the server is running</p></div>';
        }
    }
}

function renderMarketplace() {
    const container = document.getElementById('template-grid');
    if (!container) return;
    const category = workflowState.currentCategory;
    const query = workflowState.searchQuery.toLowerCase();

    const aiPanel = document.getElementById('ai-generator-panel');
    if (category === 'custom') {
        if (aiPanel) aiPanel.classList.remove('hidden');
        container.style.display = 'none';
        return;
    } else {
        if (aiPanel) aiPanel.classList.add('hidden');
        container.style.display = 'grid';
    }

    let cards = [];

    if (category === 'all' || category === 'recipes') {
        QUICK_RECIPES.forEach(recipe => {
            if (!query || recipe.name.toLowerCase().includes(query) || recipe.description.toLowerCase().includes(query)) {
                cards.push(renderRecipeCard(recipe));
            }
        });
    }

    if (workflowState.templates) {
        for (const [cat, templates] of Object.entries(workflowState.templates)) {
            if (category !== 'all' && category !== cat && category !== 'recipes') continue;

            for (const template of templates) {
                if (query && !template.name.toLowerCase().includes(query) &&
                    !(template.description || '').toLowerCase().includes(query)) {
                    continue;
                }

                cards.push(renderTemplateCard(template, cat));
            }
        }
    }

    if (cards.length === 0) {
        container.innerHTML = `
            <div class="empty-templates">
                <div class="empty-icon">&#128269;</div>
                <h4>No templates found</h4>
                <p>Try a different search or category</p>
            </div>
        `;
    } else {
        container.innerHTML = cards.join('');
    }
}

function renderRecipeCard(recipe) {
    return `
        <div class="marketplace-card recipe-card" onclick="showRecipeModal('${recipe.id}')">
            <div class="card-header">
                <div class="card-icon recipe">${recipe.icon}</div>
                <div class="card-title">
                    <h5>${escapeHtml(recipe.name)}</h5>
                    <span class="category-tag">Quick Recipe</span>
                </div>
            </div>
            <div class="card-description">${escapeHtml(recipe.description)}</div>
            <div class="card-footer">
                <span class="card-stats">${recipe.nodes.length} nodes</span>
                <button class="card-action" onclick="event.stopPropagation(); createFromRecipe('${recipe.id}')">Use</button>
            </div>
        </div>
    `;
}

function renderTemplateCard(template, category) {
    const icon = TEMPLATE_ICONS[template.type] || CATEGORY_ICONS[category] || '&#128196;';
    const iconClass = category === 'triggers' ? 'trigger' :
                      category === 'ai' ? 'ai' :
                      category === 'actions' ? 'action' :
                      category === 'data' ? 'data' : '';

    return `
        <div class="marketplace-card" onclick="showTemplateModal('${category}', '${template.type}')">
            <div class="card-header">
                <div class="card-icon ${iconClass}">${icon}</div>
                <div class="card-title">
                    <h5>${escapeHtml(template.name)}</h5>
                    <span class="category-tag">${escapeHtml(category)}</span>
                </div>
            </div>
            <div class="card-description">${escapeHtml(template.description || 'No description available')}</div>
            <div class="card-footer">
                <span class="card-stats">Single node</span>
                <button class="card-action" onclick="event.stopPropagation(); quickAddTemplate('${category}', '${template.type}')">Add</button>
            </div>
        </div>
    `;
}

export function selectCategory(category) {
    workflowState.currentCategory = category;

    document.querySelectorAll('.marketplace-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.category === category);
    });

    renderMarketplace();
}

export function filterTemplates() {
    workflowState.searchQuery = document.getElementById('template-search').value;
    renderMarketplace();
}

export function showRecipeModal(recipeId) {
    const recipe = QUICK_RECIPES.find(r => r.id === recipeId);
    if (!recipe) return;

    workflowState.selectedTemplate = { type: 'recipe', data: recipe };

    document.getElementById('modal-template-icon').innerHTML = recipe.icon;
    document.getElementById('modal-template-name').textContent = recipe.name;
    document.getElementById('modal-template-category').textContent = 'Quick Recipe';
    document.getElementById('modal-template-description').textContent = recipe.description;

    let configHtml = '';
    if (recipe.config.webhook_path !== undefined) {
        configHtml += `
            <div class="config-field">
                <label>Webhook Path</label>
                <input type="text" id="config-webhook-path" value="${recipe.config.webhook_path}" placeholder="my-webhook">
            </div>
        `;
    }
    if (recipe.config.cron !== undefined) {
        configHtml += `
            <div class="config-field">
                <label>Schedule (Cron)</label>
                <input type="text" id="config-cron" value="${recipe.config.cron}" placeholder="0 9 * * *">
            </div>
        `;
    }
    if (recipe.config.categories !== undefined) {
        configHtml += `
            <div class="config-field">
                <label>Categories</label>
                <input type="text" id="config-categories" value="${recipe.config.categories}" placeholder="positive, negative, neutral">
            </div>
        `;
    }
    if (recipe.config.schema !== undefined) {
        configHtml += `
            <div class="config-field">
                <label>Data Schema</label>
                <input type="text" id="config-schema" value="${recipe.config.schema}" placeholder="name, email, phone">
            </div>
        `;
    }
    configHtml += `
        <div class="config-field">
            <label>Workflow Name</label>
            <input type="text" id="config-name" value="${recipe.name}" placeholder="My Workflow">
        </div>
    `;

    document.getElementById('modal-template-config').innerHTML = configHtml;
    document.getElementById('modal-template-preview').textContent = `Nodes: ${recipe.nodes.join(' \u2192 ')}`;

    document.getElementById('template-modal').classList.remove('hidden');
}

export function showTemplateModal(category, templateType) {
    const templates = workflowState.templates[category];
    const template = templates.find(t => t.type === templateType);
    if (!template) return;

    workflowState.selectedTemplate = { type: 'template', category, data: template };

    const icon = TEMPLATE_ICONS[templateType] || CATEGORY_ICONS[category] || '&#128196;';

    document.getElementById('modal-template-icon').innerHTML = icon;
    document.getElementById('modal-template-name').textContent = template.name;
    document.getElementById('modal-template-category').textContent = category;
    document.getElementById('modal-template-description').textContent = template.description || 'No description available';

    let configHtml = '';
    if (templateType === 'webhook') {
        configHtml = `
            <div class="config-field">
                <label>Webhook Path</label>
                <input type="text" id="config-webhook-path" value="my-webhook" placeholder="my-webhook">
            </div>
            <div class="config-field">
                <label>HTTP Method</label>
                <select id="config-method">
                    <option value="POST">POST</option>
                    <option value="GET">GET</option>
                </select>
            </div>
        `;
    } else if (templateType === 'schedule') {
        configHtml = `
            <div class="config-field">
                <label>Cron Expression</label>
                <input type="text" id="config-cron" value="0 9 * * *" placeholder="0 9 * * *">
            </div>
        `;
    } else if (templateType === 'classify') {
        configHtml = `
            <div class="config-field">
                <label>Input Field</label>
                <input type="text" id="config-input-field" value="text" placeholder="text">
            </div>
            <div class="config-field">
                <label>Categories</label>
                <input type="text" id="config-categories" value="positive, negative, neutral" placeholder="category1, category2">
            </div>
        `;
    } else if (templateType === 'extract_json') {
        configHtml = `
            <div class="config-field">
                <label>Data Schema</label>
                <input type="text" id="config-schema" value="name, email, phone" placeholder="field1, field2">
            </div>
        `;
    } else if (templateType === 'discord_webhook' || templateType === 'slack_webhook') {
        configHtml = `
            <div class="config-field">
                <label>Webhook URL (or use $env variable)</label>
                <input type="text" id="config-webhook-url" value="" placeholder="https://discord.com/api/webhooks/...">
            </div>
        `;
    }

    document.getElementById('modal-template-config').innerHTML = configHtml || '<p style="color: var(--text-muted)">No configuration needed</p>';
    document.getElementById('modal-template-preview').textContent = 'Single node template - configure and add to your workflow';

    document.getElementById('template-modal').classList.remove('hidden');
}

export function hideTemplateModal() {
    document.getElementById('template-modal').classList.add('hidden');
    workflowState.selectedTemplate = null;
}

export async function useTemplate() {
    const selected = workflowState.selectedTemplate;
    if (!selected) return;

    hideTemplateModal();

    if (selected.type === 'recipe') {
        const recipe = selected.data;
        const name = document.getElementById('config-name')?.value || recipe.name;

        try {
            const config = {
                webhook_path: document.getElementById('config-webhook-path')?.value,
                cron: document.getElementById('config-cron')?.value,
                categories: document.getElementById('config-categories')?.value,
                schema: document.getElementById('config-schema')?.value
            };

            const resp = await fetch(`${API}/workflows/quick`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    template: recipe.id,
                    name: name,
                    webhook_path: config.webhook_path,
                    config: config
                })
            });

            const data = await resp.json();

            if (data.success) {
                workflowState.currentWorkflow = data.workflow;
                showWorkflowPreview(data.workflow);
                log(`Created workflow: ${data.workflow.name}`, 'success');
            } else {
                alert('Failed to create workflow: ' + (data.error || 'Unknown error'));
            }
        } catch (e) {
            alert('Error: ' + e.message);
        }
    } else {
        await quickAddTemplate(selected.category, selected.data.type);
    }
}

export async function quickAddTemplate(category, templateType) {
    const name = `${templateType} Workflow`;

    try {
        const resp = await fetch(`${API}/workflows/quick`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                template: templateType,
                name: name,
                category: category
            })
        });

        const data = await resp.json();

        if (data.success) {
            workflowState.currentWorkflow = data.workflow;
            showWorkflowPreview(data.workflow);
            log(`Created workflow from ${templateType}`, 'success');
        } else {
            alert('Failed to create workflow: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

export async function createFromRecipe(recipeId) {
    const recipe = QUICK_RECIPES.find(r => r.id === recipeId);
    if (!recipe) return;

    try {
        const resp = await fetch(`${API}/workflows/quick`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                template: recipeId,
                name: recipe.name,
                webhook_path: recipe.config.webhook_path
            })
        });

        const data = await resp.json();

        if (data.success) {
            workflowState.currentWorkflow = data.workflow;
            showWorkflowPreview(data.workflow);
            log(`Created workflow: ${data.workflow.name}`, 'success');
        } else {
            alert('Failed to create workflow: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

export async function generateWorkflow() {
    const description = document.getElementById('workflow-description').value.trim();
    const triggerType = document.getElementById('workflow-trigger').value;
    const workflowName = document.getElementById('workflow-name')?.value || 'Generated Workflow';

    if (!description) {
        alert('Please enter a workflow description');
        return;
    }

    const btn = document.querySelector('.generate-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating...';

    try {
        const resp = await fetch(`${API}/workflows/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description: description,
                trigger_type: triggerType,
                name: workflowName
            })
        });

        const data = await resp.json();

        if (data.success && data.workflow) {
            workflowState.currentWorkflow = data.workflow;
            showWorkflowPreview(data.workflow);
            log(`Generated workflow: ${data.workflow.name}`, 'success');
        } else if (data.errors && data.errors.length > 0) {
            alert('Workflow generation had issues:\n' + data.errors.join('\n'));
            if (data.raw_response) {
                console.log('Raw LLM response:', data.raw_response);
            }
        } else {
            alert('Failed to generate workflow: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon-inline">&#10024;</span> Generate with AI';
    }
}

export function showWorkflowPreview(workflow) {
    const previewSection = document.getElementById('workflow-preview');
    const jsonEl = document.getElementById('workflow-json');
    const nameEl = document.getElementById('preview-name');
    const statsEl = document.getElementById('preview-nodes');

    if (!previewSection) return;

    nameEl.textContent = workflow.name || 'Untitled Workflow';
    statsEl.textContent = `${(workflow.nodes || []).length} nodes`;
    jsonEl.textContent = JSON.stringify(workflow, null, 2);
    previewSection.classList.remove('hidden');

    updateDeployPortDropdown();
}

export function closeWorkflowPreview() {
    const el = document.getElementById('workflow-preview');
    if (el) el.classList.add('hidden');
}

async function updateDeployPortDropdown() {
    const select = document.getElementById('deploy-port');
    if (!select) return;

    try {
        const resp = await fetch(`${API}/n8n/list`);
        const data = await resp.json();

        select.innerHTML = '<option value="">Select n8n instance...</option>';

        if (data.instances && data.instances.length > 0) {
            data.instances.forEach(inst => {
                if (inst.is_running) {
                    select.innerHTML += `<option value="${inst.port}">n8n :${inst.port}</option>`;
                }
            });
        } else {
            select.innerHTML += '<option value="" disabled>No n8n instances running</option>';
        }
    } catch (e) {
        console.error('Failed to get n8n instances:', e);
    }
}

export async function deployWorkflow() {
    debugLog('deployWorkflow', `legacy deploy`);
    const port = document.getElementById('deploy-port').value;

    if (!port) {
        alert('Please select an n8n instance');
        return;
    }

    if (!workflowState.currentWorkflow) {
        alert('No workflow to deploy');
        return;
    }

    const btn = document.querySelector('.preview-actions .btn-primary');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Deploying...';

    try {
        const resp = await fetch(`${API}/workflows/deploy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow: workflowState.currentWorkflow,
                n8n_port: parseInt(port),
                activate: true
            })
        });

        const data = await resp.json();

        if (data.success) {
            alert(`Workflow deployed successfully!\n\nWorkflow ID: ${data.workflow_id || 'N/A'}\nWebhook URL: ${data.webhook_url || 'N/A'}`);
            log(`Deployed workflow to n8n:${port}`, 'success');
            closeWorkflowPreview();
        } else {
            alert('Failed to deploy: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon-inline">&#128640;</span> Deploy to n8n';
    }
}

export function copyWorkflow() {
    if (!workflowState.currentWorkflow) return;

    const json = JSON.stringify(workflowState.currentWorkflow, null, 2);
    navigator.clipboard.writeText(json).then(() => {
        log('Workflow JSON copied to clipboard', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy to clipboard');
    });
}

export function downloadWorkflow() {
    if (!workflowState.currentWorkflow) return;

    const json = JSON.stringify(workflowState.currentWorkflow, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `${workflowState.currentWorkflow.name || 'workflow'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    log('Workflow downloaded', 'success');
}
