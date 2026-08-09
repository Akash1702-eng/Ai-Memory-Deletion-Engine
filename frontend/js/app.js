/* ══════════════════════════════════════════════════════════════════════════════
   AI Memory Engine — Full Client-Side Application
   ══════════════════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    const API_BASE = 'http://127.0.0.1:8001/api';

    // ── State ────────────────────────────────────────────────────────────────
    let chatHistory = [];
    let currentPanel = null;
    let modelVersion = 'v1.0';
    let memories = [];
    let selectedForgetIds = new Set();
    let forgetTexts = [];
    let retainTexts = [];

    // ── Chart Instances ──────────────────────────────────────────────────────
    const _charts = {};

    const C = {
        purple: 'rgba(102, 126, 234, 1)',
        purpleFill: 'rgba(102, 126, 234, 0.15)',
        green: 'rgba(52, 211, 153, 1)',
        greenFill: 'rgba(52, 211, 153, 0.15)',
        red: 'rgba(248, 113, 113, 1)',
        redFill: 'rgba(248, 113, 113, 0.15)',
        orange: 'rgba(251, 191, 36, 1)',
        orangeFill: 'rgba(251, 191, 36, 0.15)',
        blue: 'rgba(96, 165, 250, 1)',
        blueFill: 'rgba(96, 165, 250, 0.15)',
        pink: 'rgba(240, 147, 251, 1)',
        pinkFill: 'rgba(240, 147, 251, 0.15)',
        text: 'rgba(232, 232, 240, 0.7)',
        grid: 'rgba(255, 255, 255, 0.05)',
    };

    // ── Utilities ────────────────────────────────────────────────────────────
    function esc(str) {
        if (str == null) return '';
        const div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    }

    function showToast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast alert-${type}`;
        const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
        toast.innerHTML = `<span>${icons[type] || ''} ${esc(message)}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('toast-out');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    function showCustomAlert(message, title = 'Notice', icon = '⚠️', customDetails = null) {
        const modal = document.getElementById('custom-alert-modal');
        const iconEl = document.getElementById('custom-alert-icon');
        const titleEl = document.getElementById('custom-alert-title');
        const messageEl = document.getElementById('custom-alert-message');
        const gridEl = document.getElementById('custom-alert-details');

        if (iconEl) iconEl.textContent = icon;
        if (titleEl) titleEl.textContent = title;

        let mainMsg = message;
        let detailsArr = [];

        if (customDetails && typeof customDetails === 'object') {
            detailsArr = Object.entries(customDetails).map(([k, v]) => ({ label: k, value: v }));
        } else if (typeof message === 'string' && message.includes('\n\n')) {
            const parts = message.split('\n\n');
            mainMsg = parts[0];
            const rawDetails = parts.slice(1).join('\n').split('\n');
            rawDetails.forEach(line => {
                const idx = line.indexOf(':');
                if (idx !== -1) {
                    const label = line.substring(0, idx).trim();
                    const value = line.substring(idx + 1).trim();
                    if (label && value) {
                        detailsArr.push({ label, value });
                    }
                }
            });
        }

        if (messageEl) messageEl.textContent = mainMsg;

        if (gridEl) {
            if (detailsArr.length > 0) {
                gridEl.style.display = 'grid';
                gridEl.innerHTML = detailsArr.map(d => `
                    <div class="alert-detail-item">
                        <span class="detail-label">${esc(d.label)}</span>
                        <span class="detail-value ${d.label.toLowerCase().includes('remaining') || d.label.toLowerCase().includes('days') ? 'highlight-amber' : ''}">${esc(d.value)}</span>
                    </div>
                `).join('');
            } else {
                gridEl.style.display = 'none';
                gridEl.innerHTML = '';
            }
        }

        if (modal) {
            modal.classList.add('open');
        } else {
            alert(message);
        }
    }

    function closeCustomAlert() {
        const modal = document.getElementById('custom-alert-modal');
        if (modal) modal.classList.remove('open');
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeCustomAlert();
    });

    // ── API Client ───────────────────────────────────────────────────────────
    async function api(endpoint, data, timeout = 120000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeout);
        try {
            const res = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
                signal: controller.signal,
            });
            clearTimeout(timer);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (e) {
            clearTimeout(timer);
            if (e.name === 'AbortError') throw new Error('Request timed out');
            throw e;
        }
    }

    async function apiGet(endpoint) {
        // Normalize endpoint to use full backend URL
        const url = endpoint.startsWith('http') ? endpoint : `http://127.0.0.1:8001${endpoint}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    }

    async function apiUpload(endpoint, formData, timeout = 60000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeout);
        try {
            const res = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                body: formData,
                signal: controller.signal,
            });
            clearTimeout(timer);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (e) {
            clearTimeout(timer);
            if (e.name === 'AbortError') throw new Error('Upload timed out');
            throw e;
        }
    }

    // ── Chat Rendering ───────────────────────────────────────────────────────
    function removeWelcome() {
        const el = document.getElementById('welcome-state');
        if (el) el.remove();
    }

    function addMessage(role, content, extra = null) {
        removeWelcome();
        const container = document.getElementById('chat-messages');
        const msg = document.createElement('div');
        msg.className = `message ${role}`;

        const avatarIcon = role === 'user' ? '👤' : '🧠';
        const roleName = role === 'user' ? 'You' : 'AI Assistant';

        let html = `
            <div class="message-avatar">${avatarIcon}</div>
            <div class="message-body">
                <div class="message-role">${roleName}</div>
                <div class="message-content">${esc(content)}</div>
        `;
        if (extra) html += extra;
        html += `</div>`;
        msg.innerHTML = html;
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
        chatHistory.push({ role, content });
        return msg;
    }

    function addSystemMessage(content, extra = null) {
        removeWelcome();
        const container = document.getElementById('chat-messages');
        const msg = document.createElement('div');
        msg.className = 'message assistant';

        let html = `
            <div class="message-avatar">⚙️</div>
            <div class="message-body">
                <div class="message-role">System</div>
                <div class="message-content">${content}</div>
        `;
        if (extra) html += extra;
        html += `</div>`;
        msg.innerHTML = html;
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
        return msg;
    }

    function showTyping() {
        removeWelcome();
        const container = document.getElementById('chat-messages');
        const el = document.createElement('div');
        el.className = 'typing-indicator';
        el.id = 'typing';
        el.innerHTML = `
            <div class="message-avatar">🧠</div>
            <div class="message-body">
                <div class="typing-dots"><span></span><span></span><span></span></div>
            </div>
        `;
        container.appendChild(el);
        container.scrollTop = container.scrollHeight;
    }

    function hideTyping() {
        const el = document.getElementById('typing');
        if (el) el.remove();
    }

    // ── Render Chart ─────────────────────────────────────────────────────────
    function renderChart(canvasId, config) {
        if (_charts[canvasId]) {
            _charts[canvasId].destroy();
            delete _charts[canvasId];
        }
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        const defaults = {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: C.text, font: { family: "'Inter', sans-serif", size: 11 } } },
                tooltip: {
                    backgroundColor: 'rgba(10, 10, 18, 0.95)',
                    titleColor: '#e8e8f0',
                    bodyColor: '#e8e8f0',
                    borderColor: 'rgba(102, 126, 234, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 10,
                },
            },
        };

        if (config.options && config.options.scales) {
            for (const axis of Object.keys(config.options.scales)) {
                config.options.scales[axis] = {
                    ticks: { color: C.text },
                    grid: { color: C.grid },
                    ...config.options.scales[axis],
                };
            }
        }

        config.options = {
            ...defaults,
            ...config.options,
            plugins: { ...defaults.plugins, ...(config.options?.plugins || {}) },
        };

        const chart = new Chart(canvas.getContext('2d'), config);
        _charts[canvasId] = chart;
        return chart;
    }

    // ── Send Chat Message ────────────────────────────────────────────────────
    async function sendMessage(text) {
        if (isUserExpired()) {
            showSubscriptionPage(true);
            return;
        }
        if (!text) text = document.getElementById('chat-input').value.trim();
        if (!text) return;

        document.getElementById('chat-input').value = '';
        autoResizeInput();

        addMessage('user', text);

        const sendBtn = document.getElementById('btn-send');
        sendBtn.disabled = true;
        showTyping();

        try {
            const historySlice = chatHistory.slice(0, -1).slice(-6);
            const result = await api('/chat', {
                message: text,
                model_type: 'auto',
                history: historySlice,
            });
            hideTyping();
            const statusExtra = result.status_note
                ? `<div style="margin-top:8px;padding:8px 12px;border-radius:var(--radius-sm);font-size:11px;background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.25);color:var(--text-secondary);">${esc(result.status_note)}</div>`
                : null;
            addMessage('assistant', result.answer || 'Sorry, I couldn\'t process that.', statusExtra);
            modelVersion = result.model_version || modelVersion;
            updateVersionBadge();
        } catch (err) {
            hideTyping();
            addMessage('assistant', `Error: ${err.message}`);
            showToast(err.message, 'error');
        } finally {
            sendBtn.disabled = false;
            document.getElementById('chat-input').focus();
        }
    }


    // ── CSV Upload ───────────────────────────────────────────────────────────
    async function uploadCSV(file) {
        const resultEl = document.getElementById('upload-result');
        const previewEl = document.getElementById('upload-preview');

        resultEl.innerHTML = '<div class="alert alert-info">⏳ Uploading and processing CSV...</div>';

        try {
            const formData = new FormData();
            formData.append('file', file);
            const result = await apiUpload('/upload-csv', formData);

            showToast(result.message, 'success');

            // Preview table
            let previewHtml = `<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px;">
                Preview (first ${Math.min(result.preview.length, 10)} records)
            </div>`;
            previewHtml += '<div style="max-height:200px;overflow-y:auto;">';
            result.preview.forEach(r => {
                previewHtml += `
                    <div style="padding:8px 10px;margin-bottom:4px;background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius-sm);font-size:12px;">
                        <strong style="color:var(--text-primary);">Q:</strong> ${esc(r.question || r.instruction || '')}<br/>
                        <strong style="color:var(--text-primary);">A:</strong> ${esc(r.answer || r.output || '')}
                        ${r.category ? `<span class="memory-item-cat" style="margin-left:8px;">${esc(r.category)}</span>` : ''}
                    </div>
                `;
            });
            previewHtml += '</div>';
            previewEl.innerHTML = previewHtml;

            resultEl.innerHTML = `
                <div class="alert alert-success">
                    ✅ ${esc(result.message)}<br/>
                    <small>Train: ${result.split_sizes.train} · Val: ${result.split_sizes.val} · Test: ${result.split_sizes.test}</small>
                </div>
            `;

            addSystemMessage(`📁 CSV uploaded: ${result.total_records} records across ${result.categories.length} categories.`);
        } catch (err) {
            showToast(`Upload failed: ${err.message}`, 'error');
            resultEl.innerHTML = `<div class="alert alert-error">❌ ${esc(err.message)}</div>`;
        }
    }

    async function loadBuiltinData() {
        const resultEl = document.getElementById('upload-result');
        resultEl.innerHTML = '<div class="alert alert-info">⏳ Loading built-in dataset...</div>';

        try {
            const result = await api('/memories/load-builtin', {});
            showToast(result.message, 'success');
            resultEl.innerHTML = `<div class="alert alert-success">✅ ${esc(result.message)}</div>`;
            addSystemMessage(`📦 Built-in dataset loaded.`);
        } catch (err) {
            showToast(`Failed: ${err.message}`, 'error');
            resultEl.innerHTML = `<div class="alert alert-error">❌ ${esc(err.message)}</div>`;
        }
    }

    // ── Fine-Tuning (Kaggle GPU) ───────────────────────────────────────────
    let _trainPollTimer = null;

    async function startTraining() {
        const btn = document.getElementById('btn-train');
        const resultEl = document.getElementById('train-result');
        const epochs = parseInt(document.getElementById('train-epochs').value) || 3;
        const batchSize = parseInt(document.getElementById('train-batch').value) || 4;
        const lr = parseFloat(document.getElementById('train-lr').value) || 2e-4;

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Submitting to Kaggle...';
        resultEl.innerHTML = '<div class="alert alert-info">⏳ Submitting fine-tuning notebook to Kaggle GPU...</div>';

        addSystemMessage('🔧 Submitting LoRA fine-tuning to Kaggle GPU...', `
            <div class="result-card">
                <div class="result-card-title">⏳ Training Configuration → Kaggle GPU</div>
                <div class="metrics-row">
                    <div class="metric-card"><div class="metric-value">${epochs}</div><div class="metric-label">Epochs</div></div>
                    <div class="metric-card"><div class="metric-value">${batchSize}</div><div class="metric-label">Batch Size</div></div>
                    <div class="metric-card"><div class="metric-value">${lr}</div><div class="metric-label">Learning Rate</div></div>
                </div>
            </div>
        `);

        try {
            const result = await api('/train-model', { epochs, batch_size: batchSize, learning_rate: lr });
            showToast(result.message || 'Job submitted to Kaggle!', 'info');

            resultEl.innerHTML = `
                <div class="alert alert-info">
                    🚀 ${esc(result.message)}<br/>
                    <small>Kernel: ${esc(result.kernel_slug)} · Status: ${esc(result.status)}</small>
                </div>
                <div id="train-poll-status" style="margin-top:8px;"></div>
            `;

            btn.innerHTML = '<span class="spinner"></span> Training on Kaggle...';
            addSystemMessage(`🚀 Fine-tuning notebook submitted to Kaggle. Kernel: ${result.kernel_slug}. Polling for status...`);

            // Start polling
            startTrainPolling();
        } catch (err) {
            showToast(`Submission failed: ${err.message}`, 'error');
            resultEl.innerHTML = `<div class="alert alert-error">❌ ${esc(err.message)}</div>`;
            addSystemMessage(`❌ Kaggle submission failed: ${err.message}`);
            btn.disabled = false;
            btn.innerHTML = '🚀 Start Fine-Tuning';
        }
    }

    function startTrainPolling() {
        if (_trainPollTimer) clearInterval(_trainPollTimer);
        _trainPollTimer = setInterval(pollTrainingStatus, 15000);
        // Also poll immediately
        pollTrainingStatus();
    }

    async function pollTrainingStatus() {
        const btn = document.getElementById('btn-train');
        const resultEl = document.getElementById('train-result');
        const pollEl = document.getElementById('train-poll-status');

        try {
            const status = await apiGet('/api/train-model/status');

            if (pollEl) {
                const statusEmoji = { queued: '🕐', running: '⚡', complete: '✅', error: '❌', cancelled: '🚫' }[status.status] || '❓';
                pollEl.innerHTML = `
                    <div style="padding:8px 12px;background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius-sm);font-size:12px;">
                        ${statusEmoji} <strong>Kaggle Status:</strong> ${esc(status.status).toUpperCase()} · ${esc(status.message)}
                    </div>
                `;
            }

            if (status.status === 'complete' && status.has_results) {
                clearInterval(_trainPollTimer);
                _trainPollTimer = null;

                const result = status.results;
                showToast('Fine-tuning complete on Kaggle GPU!', 'success');

                const chartId = 'chart-train-' + Date.now();
                const lossHistory = result.loss_history || [];
                const trainLoss = result.training_loss || result.final_train_loss || lossHistory[lossHistory.length - 1]?.train_loss || 0;
                const valLoss = result.validation_loss || result.final_val_loss || lossHistory[lossHistory.length - 1]?.val_loss || 0;

                const extraHtml = `
                    <div class="result-card">
                        <div class="result-card-title">✅ Fine-Tuning Complete (Kaggle GPU)</div>
                        <div class="metrics-row">
                            <div class="metric-card"><div class="metric-value">${trainLoss.toFixed(4)}</div><div class="metric-label">Train Loss</div></div>
                            <div class="metric-card"><div class="metric-value">${valLoss.toFixed(4)}</div><div class="metric-label">Val Loss</div></div>
                            <div class="metric-card"><div class="metric-value">${result.duration_seconds ? result.duration_seconds.toFixed(1) + 's' : 'N/A'}</div><div class="metric-label">Duration</div></div>
                            <div class="metric-card"><div class="metric-value">${result.epochs || 'N/A'}</div><div class="metric-label">Epochs</div></div>
                        </div>
                        <div class="chart-container">
                            <div class="chart-title">Training Loss Curve</div>
                            <canvas id="${chartId}"></canvas>
                        </div>
                    </div>
                `;
                addSystemMessage('Fine-tuning completed on Kaggle GPU! The adapter has been downloaded.', extraHtml);

                requestAnimationFrame(() => {
                    if (lossHistory.length > 0) {
                        renderChart(chartId, {
                            type: 'line',
                            data: {
                                labels: lossHistory.map(h => `Epoch ${h.epoch}`),
                                datasets: [
                                    { label: 'Training Loss', data: lossHistory.map(h => h.train_loss), borderColor: C.blue, backgroundColor: C.blueFill, fill: true, tension: 0.3, pointRadius: 5, borderWidth: 2.5 },
                                    { label: 'Validation Loss', data: lossHistory.map(h => h.val_loss), borderColor: C.orange, backgroundColor: C.orangeFill, fill: true, tension: 0.3, pointRadius: 5, borderWidth: 2.5 },
                                ],
                            },
                            options: { scales: { x: { title: { display: true, text: 'Epoch', color: C.text } }, y: { title: { display: true, text: 'Loss', color: C.text } } } },
                        });
                    }
                });

                resultEl.innerHTML = `<div class="alert alert-success">✅ Fine-tuning complete! Adapter downloaded from HF Hub.</div>`;
                btn.disabled = false;
                btn.innerHTML = '🚀 Start Fine-Tuning';
            } else if (status.status === 'error' || status.status === 'cancelled') {
                clearInterval(_trainPollTimer);
                _trainPollTimer = null;
                resultEl.innerHTML = `<div class="alert alert-error">❌ Kaggle job ${status.status}: ${esc(status.message)}</div>`;
                addSystemMessage(`❌ Kaggle training ${status.status}: ${status.message}`);
                btn.disabled = false;
                btn.innerHTML = '🚀 Start Fine-Tuning';
            } else if (status.status === 'none') {
                clearInterval(_trainPollTimer);
                _trainPollTimer = null;
                btn.disabled = false;
                btn.innerHTML = '🚀 Start Fine-Tuning';
            }
        } catch (err) {
            // Silently retry on poll errors
            console.warn('Training poll error:', err.message);
        }
    }

    // ── Memory Selection ─────────────────────────────────────────────────────
    async function loadMemories() {
        const listEl = document.getElementById('memory-list');
        listEl.innerHTML = '<p class="form-hint">Loading memories...</p>';

        try {
            const result = await apiGet('/api/memories');
            memories = result.records || [];
            selectedForgetIds.clear();

            // Populate category filter
            const filterEl = document.getElementById('memory-category-filter');
            filterEl.innerHTML = '<option value="all">All Categories</option>';
            (result.categories || []).forEach(cat => {
                filterEl.innerHTML += `<option value="${esc(cat)}">${esc(cat)}</option>`;
            });

            renderMemoryList();
            updateMemoryStats();
        } catch (err) {
            listEl.innerHTML = `<p class="form-hint" style="color:var(--error);">Failed to load: ${esc(err.message)}</p>`;
        }
    }

    function renderMemoryList(filter = 'all') {
        const listEl = document.getElementById('memory-list');
        const filtered = filter === 'all' ? memories : memories.filter(m => m.category === filter);

        if (filtered.length === 0) {
            listEl.innerHTML = '<p class="form-hint">No records found. Upload a CSV or load built-in data first.</p>';
            return;
        }

        listEl.innerHTML = filtered.map(m => `
            <div class="memory-item ${selectedForgetIds.has(m.id) ? 'selected' : ''}"
                 data-id="${m.id}" onclick="App.toggleMemory(${m.id})">
                <input type="checkbox" ${selectedForgetIds.has(m.id) ? 'checked' : ''} onclick="event.stopPropagation(); App.toggleMemory(${m.id})" />
                <div class="memory-item-text">
                    <div class="memory-item-q">${esc(m.question)}</div>
                    <div class="memory-item-a">${esc(m.answer)}</div>
                </div>
                <span class="memory-item-cat">${esc(m.category)}</span>
            </div>
        `).join('');
    }

    function toggleMemory(id) {
        if (selectedForgetIds.has(id)) {
            selectedForgetIds.delete(id);
        } else {
            selectedForgetIds.add(id);
        }
        const filter = document.getElementById('memory-category-filter').value;
        renderMemoryList(filter);
        updateMemoryStats();
    }

    function selectAllVisible() {
        const filter = document.getElementById('memory-category-filter').value;
        const filtered = filter === 'all' ? memories : memories.filter(m => m.category === filter);
        filtered.forEach(m => selectedForgetIds.add(m.id));
        renderMemoryList(filter);
        updateMemoryStats();
    }

    function deselectAll() {
        selectedForgetIds.clear();
        const filter = document.getElementById('memory-category-filter').value;
        renderMemoryList(filter);
        updateMemoryStats();
    }

    function updateMemoryStats() {
        const total = memories.length;
        const forgetCount = selectedForgetIds.size;
        const retainCount = total - forgetCount;

        document.getElementById('memory-stats').innerHTML = `
            <span class="memory-stat">Total: <strong>${total}</strong></span>
            <span class="memory-stat forget-count">Forget: <strong>${forgetCount}</strong></span>
            <span class="memory-stat retain-count">Retain: <strong>${retainCount}</strong></span>
        `;
    }

    async function createForgetSet() {
        if (selectedForgetIds.size === 0) {
            showToast('Select at least one memory to forget.', 'warning');
            return;
        }

        const resultEl = document.getElementById('forget-set-result');
        resultEl.innerHTML = '<div class="alert alert-info">⏳ Creating forget/retain sets...</div>';

        try {
            const result = await api('/memories/forget-set', {
                forget_ids: Array.from(selectedForgetIds),
            });

            forgetTexts = result.forget_texts || [];
            retainTexts = result.retain_texts || [];

            // Auto-populate unlearning panel
            document.getElementById('unlearn-texts').value = forgetTexts.join('\n');
            document.getElementById('unlearn-retain-texts').value = retainTexts.slice(0, 10).join('\n');

            // Generate test queries from forget set
            const forgetRecords = memories.filter(m => selectedForgetIds.has(m.id));
            const testQueries = forgetRecords.map(m => m.question).slice(0, 5);
            document.getElementById('unlearn-test-queries').value = testQueries.join('\n');

            resultEl.innerHTML = `
                <div class="alert alert-success">✅ ${esc(result.message)}</div>
                <div class="alert alert-warning" style="margin-top:8px;">
                    <strong>⚠️ Important:</strong> The selected data has been removed from the <strong>dataset</strong> only.
                    The fine-tuned model still retains this knowledge in its weights.
                    <br/><br/>
                    <strong>→ To actually make the model forget</strong>, go to the <strong>Unlearn panel</strong> and
                    run <strong>Gradient Ascent Unlearning</strong>. The texts have been auto-populated for you.
                </div>
            `;
            showToast(result.message, 'success');

            addSystemMessage(
                `🗂️ Forget/retain sets created: ${result.forget_count} to forget, ${result.retain_count} to retain.`,
                `<div class="result-card">
                    <div class="result-card-title">📋 Pipeline Status</div>
                    <div class="metrics-row">
                        <div class="metric-card"><div class="metric-value" style="color:var(--success)">✅</div><div class="metric-label">Data Removed from Dataset</div></div>
                        <div class="metric-card"><div class="metric-value" style="color:var(--warning)">⚠️</div><div class="metric-label">Model Still Knows</div></div>
                        <div class="metric-card"><div class="metric-value" style="color:var(--text-dim)">⏳</div><div class="metric-label">Unlearning Pending</div></div>
                    </div>
                    <div style="font-size:12px;color:var(--text-muted);margin-top:12px;padding:10px;background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius-sm);">
                        <strong>What happened:</strong> ${result.forget_count} record(s) have been marked for deletion in the dataset database.
                        The fine-tuned model's weights are unchanged — it will still answer questions about this data correctly.<br/><br/>
                        <strong>Next step:</strong> Go to <strong>🧹 Unlearn</strong> panel → Run Gradient Ascent. This modifies the model's weights
                        via gradient ascent to make it genuinely forget this information.<br/><br/>
                        <strong>After unlearning:</strong> The model will respond with "I don't know" to questions about forgotten data.
                    </div>
                </div>`
            );
        } catch (err) {
            resultEl.innerHTML = `<div class="alert alert-error">❌ ${esc(err.message)}</div>`;
            showToast(err.message, 'error');
        }
    }

    // ── Unlearning (Kaggle GPU) ────────────────────────────────────────────
    let _unlearnPollTimer = null;

    async function startUnlearning() {
        const btn = document.getElementById('btn-unlearn');
        const resultEl = document.getElementById('unlearn-result');
        const textsRaw = document.getElementById('unlearn-texts').value.trim();
        const retainRaw = document.getElementById('unlearn-retain-texts').value.trim();
        const queriesRaw = document.getElementById('unlearn-test-queries').value.trim();
        const epochs = parseInt(document.getElementById('unlearn-epochs').value) || 5;
        const lr = parseFloat(document.getElementById('unlearn-lr').value) || 1e-4;

        const forgetList = textsRaw.split('\n').map(t => t.trim()).filter(Boolean);
        const retainList = retainRaw.split('\n').map(t => t.trim()).filter(Boolean);
        const testQueries = queriesRaw.split('\n').map(t => t.trim()).filter(Boolean);

        if (forgetList.length === 0) {
            showToast('Provide at least one text to unlearn.', 'error');
            return;
        }

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Submitting to Kaggle...';
        resultEl.innerHTML = '<div class="alert alert-info">⏳ Submitting gradient ascent notebook to Kaggle GPU...</div>';

        addSystemMessage('🧹 Submitting Gradient Ascent to Kaggle GPU...', `
            <div class="result-card">
                <div class="result-card-title">⏳ Unlearning Configuration → Kaggle GPU</div>
                <div class="metrics-row">
                    <div class="metric-card"><div class="metric-value">${forgetList.length}</div><div class="metric-label">Forget Texts</div></div>
                    <div class="metric-card"><div class="metric-value">${retainList.length}</div><div class="metric-label">Retain Texts</div></div>
                    <div class="metric-card"><div class="metric-value">${testQueries.length}</div><div class="metric-label">Test Queries</div></div>
                    <div class="metric-card"><div class="metric-value">${epochs}</div><div class="metric-label">Epochs</div></div>
                </div>
            </div>
        `);

        try {
            const payload = {
                forget_texts: forgetList,
                epochs,
                learning_rate: lr,
            };
            if (retainList.length > 0) payload.retain_texts = retainList;
            if (testQueries.length > 0) payload.test_queries = testQueries;

            const result = await api('/run-unlearning', payload);
            showToast(result.message || 'Job submitted to Kaggle!', 'info');

            resultEl.innerHTML = `
                <div class="alert alert-info">
                    🚀 ${esc(result.message)}<br/>
                    <small>Kernel: ${esc(result.kernel_slug)} · Status: ${esc(result.status)}</small>
                </div>
                <div id="unlearn-poll-status" style="margin-top:8px;"></div>
            `;

            btn.innerHTML = '<span class="spinner"></span> Unlearning on Kaggle...';
            addSystemMessage(`🚀 Unlearning notebook submitted to Kaggle. Kernel: ${result.kernel_slug}. Polling for status...`);

            startUnlearnPolling();
        } catch (err) {
            showToast(`Submission failed: ${err.message}`, 'error');
            resultEl.innerHTML = `<div class="alert alert-error">❌ ${esc(err.message)}</div>`;
            addSystemMessage(`❌ Kaggle submission failed: ${err.message}`);
            btn.disabled = false;
            btn.innerHTML = '🧠 Run Gradient Ascent';
        }
    }

    function startUnlearnPolling() {
        if (_unlearnPollTimer) clearInterval(_unlearnPollTimer);
        _unlearnPollTimer = setInterval(pollUnlearningStatus, 15000);
        pollUnlearningStatus();
    }

    async function pollUnlearningStatus() {
        const btn = document.getElementById('btn-unlearn');
        const resultEl = document.getElementById('unlearn-result');
        const pollEl = document.getElementById('unlearn-poll-status');

        try {
            const status = await apiGet('/api/run-unlearning/status');

            if (pollEl) {
                const statusEmoji = { queued: '🕐', running: '⚡', complete: '✅', error: '❌', cancelled: '🚫' }[status.status] || '❓';
                pollEl.innerHTML = `
                    <div style="padding:8px 12px;background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius-sm);font-size:12px;">
                        ${statusEmoji} <strong>Kaggle Status:</strong> ${esc(status.status).toUpperCase()} · ${esc(status.message)}
                    </div>
                `;
            }

            if (status.status === 'complete' && status.has_results) {
                clearInterval(_unlearnPollTimer);
                _unlearnPollTimer = null;

                const result = status.results;
                showToast('Unlearning complete on Kaggle GPU!', 'success');

                const lossBefore = result.loss_before || 0;
                const lossAfter = result.loss_after || 0;
                const delta = (lossAfter - lossBefore).toFixed(4);
                const lossChartId = 'chart-ga-' + Date.now();
                const barChartId = 'chart-ba-' + Date.now();

                // Comparison section removed — the before/after cards added visual
                // noise without actionable insight; the loss metrics and charts
                // already convey unlearning effectiveness.
                let comparisonHtml = '';

                const miaHtml = result.mia_accuracy != null ? `
                    <div class="metrics-row" style="margin-top:12px">
                        <div class="metric-card"><div class="metric-value">${(result.mia_accuracy * 100).toFixed(1)}%</div><div class="metric-label">MIA Accuracy</div></div>
                        <div class="metric-card"><div class="metric-value">${result.mia_f1?.toFixed(4) || 'N/A'}</div><div class="metric-label">MIA F1</div></div>
                        <div class="metric-card"><div class="metric-value">${result.forget_success_rate != null ? (result.forget_success_rate * 100).toFixed(0) + '%' : 'N/A'}</div><div class="metric-label">Forget Rate</div></div>
                    </div>
                ` : '';

                const extraHtml = `
                    <div class="result-card">
                        <div class="result-card-title">✅ Gradient Ascent Unlearning Complete (Kaggle GPU)</div>
                        <div class="metrics-row">
                            <div class="metric-card"><div class="metric-value">${lossBefore.toFixed(4)}</div><div class="metric-label">Loss Before</div></div>
                            <div class="metric-card"><div class="metric-value">${lossAfter.toFixed(4)}</div><div class="metric-label">Loss After</div></div>
                            <div class="metric-card"><div class="metric-value positive">+${delta}</div><div class="metric-label">Loss Δ</div></div>
                            <div class="metric-card"><div class="metric-value">${result.duration_seconds ? result.duration_seconds.toFixed(1) + 's' : 'N/A'}</div><div class="metric-label">Duration</div></div>
                        </div>
                        ${miaHtml}
                        <div class="chart-container">
                            <div class="chart-title">Gradient Ascent Loss Curve</div>
                            <canvas id="${lossChartId}"></canvas>
                        </div>
                        <div class="chart-container" style="margin-top:12px">
                            <div class="chart-title">Before vs After Loss</div>
                            <canvas id="${barChartId}"></canvas>
                        </div>
                        ${comparisonHtml}
                        <div style="margin-top:16px;padding:12px;background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);border-radius:var(--radius-sm);">
                            <div class="result-card-title">✅ Pipeline Status — Unlearning Complete</div>
                            <div class="metrics-row" style="margin-top:8px">
                                <div class="metric-card"><div class="metric-value" style="color:var(--success)">✅</div><div class="metric-label">Data Removed from Dataset</div></div>
                                <div class="metric-card"><div class="metric-value" style="color:var(--success)">✅</div><div class="metric-label">Model Weights Updated</div></div>
                                <div class="metric-card"><div class="metric-value" style="color:var(--success)">✅</div><div class="metric-label">Unlearning Complete</div></div>
                            </div>
                            <div style="font-size:12px;color:var(--text-muted);margin-top:8px;">
                                The model has now genuinely forgotten the selected information via gradient ascent.
                                Ask the model questions about the forgotten data — it will respond with "I don't know".
                            </div>
                        </div>
                    </div>
                `;

                addSystemMessage(
                    `Unlearning complete! Loss increased from ${lossBefore.toFixed(4)} to ${lossAfter.toFixed(4)} (+${delta}).`,
                    extraHtml
                );

                requestAnimationFrame(() => {
                    const lossCurve = result.loss_curve || [];
                    if (lossCurve.length > 0) {
                        renderChart(lossChartId, {
                            type: 'line',
                            data: {
                                labels: lossCurve.map(p => `Epoch ${p.epoch}`),
                                datasets: [
                                    { label: 'Forget Loss (↑)', data: lossCurve.map(p => p.forget_loss), borderColor: C.purple, backgroundColor: C.purpleFill, fill: true, tension: 0.3, pointRadius: 5, borderWidth: 2.5 },
                                    ...(lossCurve[0].retain_loss != null ? [{ label: 'Retain Loss', data: lossCurve.map(p => p.retain_loss), borderColor: C.green, backgroundColor: C.greenFill, fill: true, tension: 0.3, pointRadius: 5, borderWidth: 2.5 }] : []),
                                ],
                            },
                            options: { scales: { x: { title: { display: true, text: 'Epoch', color: C.text } }, y: { title: { display: true, text: 'Loss', color: C.text } } } },
                        });
                    }
                    renderChart(barChartId, {
                        type: 'bar',
                        data: {
                            labels: ['Loss Before', 'Loss After'],
                            datasets: [{ label: 'Loss', data: [lossBefore, lossAfter], backgroundColor: [C.greenFill, C.redFill], borderColor: [C.green, C.red], borderWidth: 2, borderRadius: 8, barPercentage: 0.5 }],
                        },
                        options: { scales: { y: { title: { display: true, text: 'Loss', color: C.text }, beginAtZero: true } }, plugins: { legend: { display: false } } },
                    });
                });

                resultEl.innerHTML = `<div class="alert alert-success">✅ Unlearning complete! Adapter downloaded from HF Hub.</div>`;
                btn.disabled = false;
                btn.innerHTML = '🧠 Run Gradient Ascent';
            } else if (status.status === 'error' || status.status === 'cancelled') {
                clearInterval(_unlearnPollTimer);
                _unlearnPollTimer = null;
                resultEl.innerHTML = `<div class="alert alert-error">❌ Kaggle job ${status.status}: ${esc(status.message)}</div>`;
                addSystemMessage(`❌ Kaggle unlearning ${status.status}: ${status.message}`);
                btn.disabled = false;
                btn.innerHTML = '🧠 Run Gradient Ascent';
            } else if (status.status === 'none') {
                clearInterval(_unlearnPollTimer);
                _unlearnPollTimer = null;
                btn.disabled = false;
                btn.innerHTML = '🧠 Run Gradient Ascent';
            }
        } catch (err) {
            console.warn('Unlearning poll error:', err.message);
        }
    }

    // ── Evaluation ───────────────────────────────────────────────────────────
    async function startEvaluation() {
        const btn = document.getElementById('btn-evaluate');
        const resultEl = document.getElementById('eval-result');
        const testRaw = document.getElementById('eval-test-queries').value.trim();

        const testQueries = testRaw.split('\n').map(t => t.trim()).filter(Boolean);
        if (testQueries.length === 0) {
            showToast('Provide at least one test query.', 'error');
            return;
        }

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Evaluating...';
        resultEl.innerHTML = '<div class="alert alert-info">⏳ Running before vs after evaluation (this may take a moment)...</div>';

        addSystemMessage('📊 Running Before vs After evaluation (Fine-Tuned → Unlearned)...');

        try {
            const result = await api('/evaluation/before-after', {
                test_queries: testQueries,
            }, 300000);
            showToast(result.message || 'Evaluation complete!', 'success');

            const comparisons = result.comparisons || [];
            const summary = result.summary || {};
            const mode = result.mode || 'before_after';
            const isFinetuneOnly = mode === 'finetuned_only';

            // MIA comparison (only in before_after mode)
            let miaHtml = '';
            if (!isFinetuneOnly && (result.mia_before || result.mia_after)) {
                const miaBefore = result.mia_before || {};
                const miaAfter = result.mia_after || {};
                miaHtml = `
                    <div class="result-card-title" style="margin-top:16px">🔍 Membership Inference Attack (MIA)</div>
                    <div class="form-hint" style="margin-bottom:8px">Lower MIA accuracy after unlearning = better forgetting (ideal: ~50%)</div>
                    <div class="metrics-row">
                        <div class="metric-card"><div class="metric-value">${miaBefore.accuracy != null ? (miaBefore.accuracy * 100).toFixed(1) + '%' : 'N/A'}</div><div class="metric-label">MIA Before</div></div>
                        <div class="metric-card"><div class="metric-value">${miaAfter.accuracy != null ? (miaAfter.accuracy * 100).toFixed(1) + '%' : 'N/A'}</div><div class="metric-label">MIA After</div></div>
                        <div class="metric-card"><div class="metric-value">${miaBefore.f1 != null ? miaBefore.f1.toFixed(4) : 'N/A'}</div><div class="metric-label">F1 Before</div></div>
                        <div class="metric-card"><div class="metric-value">${miaAfter.f1 != null ? miaAfter.f1.toFixed(4) : 'N/A'}</div><div class="metric-label">F1 After</div></div>
                    </div>
                `;
            }

            // Summary metrics section
            let summaryHtml = '';
            if (isFinetuneOnly) {
                summaryHtml = `
                    <div class="result-card-title">📊 Fine-Tuned Model Evaluation</div>
                    <div class="form-hint" style="margin-bottom:8px">No unlearning performed — showing current fine-tuned model knowledge.</div>
                    <div class="metrics-row">
                        <div class="metric-card"><div class="metric-value">${(summary.avg_loss_before || 0).toFixed(4)}</div><div class="metric-label">Avg Loss</div></div>
                        <div class="metric-card"><div class="metric-value">${summary.total_queries || 0}</div><div class="metric-label">Queries Tested</div></div>
                    </div>
                `;
            } else {
                const lossChartId = 'chart-eval-loss-' + Date.now();
                const confChartId = 'chart-eval-conf-' + Date.now();
                summaryHtml = `
                    <div class="result-card-title">📊 Before vs After Unlearning Evaluation</div>
                    <div class="metrics-row">
                        <div class="metric-card"><div class="metric-value">${(summary.avg_loss_before || 0).toFixed(4)}</div><div class="metric-label">Avg Loss Before</div></div>
                        <div class="metric-card"><div class="metric-value">${(summary.avg_loss_after || 0).toFixed(4)}</div><div class="metric-label">Avg Loss After</div></div>
                        <div class="metric-card"><div class="metric-value positive">+${(summary.avg_loss_increase || 0).toFixed(4)}</div><div class="metric-label">Avg Loss Δ</div></div>
                        <div class="metric-card"><div class="metric-value">${summary.forget_success_rate != null ? (summary.forget_success_rate * 100).toFixed(0) + '%' : 'N/A'}</div><div class="metric-label">Forget Rate</div></div>
                        <div class="metric-card"><div class="metric-value">${summary.queries_forgotten || 0}/${summary.total_queries || 0}</div><div class="metric-label">Queries Forgotten</div></div>
                    </div>
                    ${miaHtml}
                    <div class="chart-container" style="margin-top:16px">
                        <div class="chart-title">Per-Query Loss: Before vs After</div>
                        <canvas id="${lossChartId}"></canvas>
                    </div>
                    <div class="chart-container" style="margin-top:12px">
                        <div class="chart-title">Per-Query Confidence: Before vs After</div>
                        <canvas id="${confChartId}"></canvas>
                    </div>
                `;

                // Render charts after DOM update
                requestAnimationFrame(() => {
                    if (comparisons.length > 0) {
                        const labels = comparisons.map(c => {
                            const q = c.query || '';
                            return q.length > 18 ? q.slice(0, 18) + '…' : q;
                        });

                        renderChart(lossChartId, {
                            type: 'bar',
                            data: {
                                labels,
                                datasets: [
                                    {
                                        label: 'Before (Fine-Tuned)',
                                        data: comparisons.map(c => c.before?.loss || 0),
                                        backgroundColor: C.greenFill,
                                        borderColor: C.green,
                                        borderWidth: 2, borderRadius: 6,
                                    },
                                    {
                                        label: 'After (Unlearned)',
                                        data: comparisons.map(c => c.after?.loss || 0),
                                        backgroundColor: C.redFill,
                                        borderColor: C.red,
                                        borderWidth: 2, borderRadius: 6,
                                    },
                                ],
                            },
                            options: {
                                scales: {
                                    x: { ticks: { maxRotation: 45, minRotation: 20 } },
                                    y: { title: { display: true, text: 'Loss (↑ after = good)', color: C.text }, beginAtZero: true },
                                },
                            },
                        });

                        renderChart(confChartId, {
                            type: 'bar',
                            data: {
                                labels,
                                datasets: [
                                    {
                                        label: 'Before (Fine-Tuned)',
                                        data: comparisons.map(c => c.before?.confidence || 0),
                                        backgroundColor: C.greenFill,
                                        borderColor: C.green,
                                        borderWidth: 2, borderRadius: 6,
                                    },
                                    {
                                        label: 'After (Unlearned)',
                                        data: comparisons.map(c => c.after?.confidence || 0),
                                        backgroundColor: C.orangeFill,
                                        borderColor: C.orange,
                                        borderWidth: 2, borderRadius: 6,
                                    },
                                ],
                            },
                            options: {
                                scales: {
                                    x: { ticks: { maxRotation: 45, minRotation: 20 } },
                                    y: { title: { display: true, text: 'Confidence (↓ after = good)', color: C.text }, beginAtZero: true },
                                },
                            },
                        });
                    }
                });
            }

            const extraHtml = `
                <div class="result-card">
                    ${summaryHtml}
                </div>
            `;

            addSystemMessage(result.message, extraHtml);
            resultEl.innerHTML = `<div class="alert alert-success">✅ ${esc(result.message)}</div>`;
        } catch (err) {
            showToast(`Evaluation failed: ${err.message}`, 'error');
            resultEl.innerHTML = `<div class="alert alert-error">❌ ${esc(err.message)}</div>`;
            addSystemMessage(`❌ Evaluation failed: ${err.message}`);
        } finally {
            btn.disabled = false;
            btn.innerHTML = '📊 Run Before vs After Evaluation';
        }
    }

    // ── Hugging Face ─────────────────────────────────────────────────────────
    async function hfUpload() {
        const resultEl = document.getElementById('hf-result');
        const adapterType = document.getElementById('hf-adapter-type').value;
        const repoId = document.getElementById('hf-repo-id').value.trim() || null;

        resultEl.innerHTML = '<div class="alert alert-info">⏳ Uploading adapter to Hugging Face Hub...</div>';

        try {
            const result = await api('/hf-upload', { adapter_type: adapterType, repo_id: repoId }, 120000);
            showToast(result.message, 'success');
            resultEl.innerHTML = `
                <div class="alert alert-success">
                    ✅ ${esc(result.message)}<br/>
                    <a href="${esc(result.url)}" target="_blank" style="color:var(--success);">${esc(result.url)}</a>
                </div>
            `;
            addSystemMessage(`🤗 Adapter uploaded to ${result.url}`);
        } catch (err) {
            showToast(`Upload failed: ${err.message}`, 'error');
            resultEl.innerHTML = `<div class="alert alert-error">❌ ${esc(err.message)}</div>`;
        }
    }

    async function hfDownload() {
        const resultEl = document.getElementById('hf-result');
        const adapterType = document.getElementById('hf-adapter-type').value;
        const repoId = document.getElementById('hf-repo-id').value.trim() || null;

        resultEl.innerHTML = '<div class="alert alert-info">⏳ Downloading adapter from Hugging Face Hub...</div>';

        try {
            const result = await api('/hf-download', { adapter_type: adapterType, repo_id: repoId }, 120000);
            showToast(result.message, 'success');
            resultEl.innerHTML = `<div class="alert alert-success">✅ ${esc(result.message)}</div>`;
            addSystemMessage(`🤗 Adapter downloaded from ${result.repo_id}`);
        } catch (err) {
            showToast(`Download failed: ${err.message}`, 'error');
            resultEl.innerHTML = `<div class="alert alert-error">❌ ${esc(err.message)}</div>`;
        }
    }

    // ── Panel Management ─────────────────────────────────────────────────────
    function openPanel(name) {
        if (isUserExpired()) {
            showSubscriptionPage(true);
            return;
        }
        closePanel();
        const panel = document.getElementById(`panel-${name}`);
        if (panel) {
            panel.classList.add('open');
            currentPanel = name;
        }

        document.querySelectorAll('.nav-item[data-action]').forEach(el => {
            el.classList.toggle('active', el.dataset.action === name);
        });

        // Auto-load data for memory panel
        if (name === 'memories') loadMemories();
    }

    function closePanel() {
        document.querySelectorAll('.action-panel').forEach(p => p.classList.remove('open'));
        document.querySelectorAll('.nav-item[data-action]').forEach(el => el.classList.remove('active'));
        currentPanel = null;
    }

    // ── Version Badge ────────────────────────────────────────────────────────
    function updateVersionBadge() {
        document.getElementById('header-version').textContent = `${modelVersion} — Qwen2.5`;
    }

    // ── New Chat ─────────────────────────────────────────────────────────────
    function newChat() {
        chatHistory = [];
        const container = document.getElementById('chat-messages');
        container.innerHTML = `
            <div class="welcome-state" id="welcome-state">
                <div class="welcome-icon">🧠</div>
                <h2>AI Memory Engine</h2>
                <p>
                    Fine-tune <strong>Qwen2.5</strong> on personal data, then use <strong>Gradient Ascent</strong>
                    to make it forget specific information. Chat uses <strong>auto mode</strong> — 
                    it combines model knowledge with stored memories for the best answer.
                </p>
                <div class="welcome-actions">
                    <button class="welcome-action-btn" onclick="App.openPanel('upload')">📁 Upload CSV</button>
                    <button class="welcome-action-btn" onclick="App.openPanel('finetune')">🔧 Fine-Tune</button>
                    <button class="welcome-action-btn" onclick="App.openPanel('unlearn')">🧹 Unlearn</button>
                    <button class="welcome-action-btn" onclick="App.sendMessage('What is my name?')">💬 Ask a Question</button>
                </div>
            </div>
        `;
        closePanel();
        showToast('Chat cleared.', 'info');
    }

    // ── Auto-resize textarea ─────────────────────────────────────────────────
    function autoResizeInput() {
        const textarea = document.getElementById('chat-input');
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
    }

    // ── Health Check ─────────────────────────────────────────────────────────
    async function checkHealth() {
        try {
            const health = await apiGet('http://127.0.0.1:8001/health');
            modelVersion = health.model_version || 'v1.0';
            updateVersionBadge();
            document.getElementById('status-text').textContent = `Model: ${health.model || 'Qwen2.5'}`;
            document.getElementById('status-dot').classList.add('online');
        } catch {
            document.getElementById('status-text').textContent = 'Backend: Offline';
            document.getElementById('status-dot').classList.remove('online');
        }
    }

    // ── Initialization ───────────────────────────────────────────────────────
    function init() {
        // Send button
        document.getElementById('btn-send').addEventListener('click', () => sendMessage());

        // Enter to send
        document.getElementById('chat-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Auto-resize textarea
        document.getElementById('chat-input').addEventListener('input', autoResizeInput);

        // New chat
        document.getElementById('btn-new-chat').addEventListener('click', newChat);


        // Mobile menu
        document.getElementById('mobile-toggle').addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('open');
        });

        // Sidebar nav actions
        document.querySelectorAll('.nav-item[data-action]').forEach(item => {
            item.addEventListener('click', () => {
                openPanel(item.dataset.action);
                if (window.innerWidth <= 768) {
                    document.getElementById('sidebar').classList.remove('open');
                }
            });
        });

        // Quick prompts
        document.querySelectorAll('.quick-prompt').forEach(item => {
            item.addEventListener('click', () => {
                sendMessage(item.dataset.prompt);
                if (window.innerWidth <= 768) {
                    document.getElementById('sidebar').classList.remove('open');
                }
            });
        });

        // Unlearn epochs slider
        document.getElementById('unlearn-epochs').addEventListener('input', (e) => {
            document.getElementById('unlearn-epoch-val').textContent = e.target.value;
        });

        // CSV dropzone
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('csv-file-input');

        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) uploadCSV(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) uploadCSV(fileInput.files[0]);
        });

        // Load built-in data
        document.getElementById('btn-load-builtin').addEventListener('click', loadBuiltinData);

        // Memory category filter
        document.getElementById('memory-category-filter').addEventListener('change', (e) => {
            renderMemoryList(e.target.value);
        });

        // Health check
        checkHealth();

        // Compute Target Toggle Switch
        initComputeToggle();

        // Check authentication state & payment callback
        checkAuthState();
        checkPaymentCallback();

        // Header Account Button click
        const btnAccount = document.getElementById('btn-account-header');
        if (btnAccount) {
            btnAccount.addEventListener('click', () => {
                openAuthModal('login');
            });
        }

        // Focus input
        document.getElementById('chat-input').focus();
    }

    // ── Compute Target Toggle ────────────────────────────────────────────────
    let currentComputeTarget = 'kaggle';

    async function initComputeToggle() {
        const toggle = document.getElementById('compute-toggle');
        const label = document.getElementById('compute-toggle-label');
        if (!toggle) return;

        try {
            const res = await apiGet('/api/settings/compute_target');
            if (res && res.compute_target) {
                currentComputeTarget = res.compute_target;
                toggle.checked = (currentComputeTarget === 'kaggle');
                label.textContent = toggle.checked ? '⚡ Kaggle GPU' : '💻 Local Machine';
            }
        } catch (e) {
            console.warn('Could not fetch compute target:', e);
        }

        toggle.addEventListener('change', async (e) => {
            const newTarget = e.target.checked ? 'kaggle' : 'local';
            label.textContent = e.target.checked ? '⚡ Kaggle GPU' : '💻 Local Machine';
            try {
                const res = await api('/settings/compute_target', { target: newTarget });
                currentComputeTarget = res.compute_target;
                showToast(`Compute target switched to ${res.compute_target.toUpperCase()}`, 'success');
            } catch (err) {
                showToast(err.message, 'error');
            }
        });
    }

    // ── Landing Page, Auth & Trial System ────────────────────────────────────
    let currentUser = null;
    let authMode = 'signup';

    function isUserExpired(user = currentUser) {
        if (!user) return true;
        const sub = user.subscription;
        if (!sub) return true;

        if (sub.active === false || sub.status === 'expired') return true;

        if (sub.expires_at) {
            try {
                const expDate = new Date(sub.expires_at);
                if (!isNaN(expDate.getTime()) && expDate <= new Date()) {
                    return true;
                }
            } catch (e) {}
        }

        if (typeof sub.days_left === 'number' && sub.days_left <= 0) {
            return true;
        }

        return false;
    }

    async function checkAuthState() {
        const stored = localStorage.getItem('md_engine_user');
        const overlay = document.getElementById('landing-overlay');
        const headerAccountBtn = document.getElementById('btn-account-header');
        const logoutBtn = document.getElementById('btn-logout');

        if (stored) {
            try {
                currentUser = JSON.parse(stored);

                // Fetch fresh status live from backend
                const userId = currentUser.user_id || currentUser.id;
                if (userId && !String(userId).startsWith('guest_')) {
                    try {
                        const fresh = await apiGet(`/api/auth/me?user_id=${userId}`);
                        if (fresh && fresh.subscription) {
                            currentUser.subscription = fresh.subscription;
                            currentUser.email = fresh.email || currentUser.email;
                            localStorage.setItem('md_engine_user', JSON.stringify(currentUser));
                        }
                    } catch (err) {
                        console.warn('Could not refresh subscription status live:', err);
                    }
                }

                const expired = isUserExpired(currentUser);

                if (expired) {
                    if (overlay) overlay.classList.add('hidden');
                    if (headerAccountBtn) {
                        headerAccountBtn.textContent = `⚠️ ${currentUser.email} (Expired)`;
                        headerAccountBtn.className = 'btn-account btn-account-expired';
                        headerAccountBtn.title = 'Subscription expired. Click to choose a plan.';
                        headerAccountBtn.onclick = () => showSubscriptionPage(true);
                    }
                    if (logoutBtn) logoutBtn.style.display = 'inline-block';

                    // Directly display subscription page modal!
                    showSubscriptionPage(true);
                    return;
                }

                // Active subscription / active free trial
                if (overlay) overlay.classList.add('hidden');
                if (headerAccountBtn) {
                    const days = currentUser.subscription?.days_left || 7;
                    const isPaid = currentUser.subscription?.status === 'active';
                    const labelText = isPaid ? '⭐ Premium' : `${days}d left`;
                    headerAccountBtn.textContent = `👤 ${currentUser.email} (${labelText})`;
                    headerAccountBtn.className = 'btn-account';
                    headerAccountBtn.title = 'Click to view or upgrade subscription plans';
                    headerAccountBtn.onclick = () => showSubscriptionPage(false);
                }
                if (logoutBtn) logoutBtn.style.display = 'inline-block';

                const subModal = document.getElementById('sub-modal');
                if (subModal && !window.location.hash.includes('payment_success')) {
                    subModal.classList.remove('open');
                }
                return;
            } catch (e) {
                console.error('Auth state check error:', e);
                localStorage.removeItem('md_engine_user');
            }
        }

        if (logoutBtn) logoutBtn.style.display = 'none';
        if (overlay) overlay.classList.remove('hidden');
    }

    function showSubscriptionPage(isForced = false) {
        const authModal = document.getElementById('auth-modal');
        if (authModal) authModal.classList.remove('open');

        const overlay = document.getElementById('landing-overlay');
        if (overlay) overlay.classList.add('hidden');

        const subModal = document.getElementById('sub-modal');
        const noticeEl = document.getElementById('sub-modal-notice');
        const closeBtn = document.getElementById('sub-modal-close');

        const forced = isForced || isUserExpired();

        if (noticeEl) {
            if (forced) {
                noticeEl.innerHTML = `<span style="color:#f87171; font-weight:700;">⚠️ Access Expired!</span> Your 7-day free trial or subscription has expired. Please select a plan below to activate unlimited access:`;
            } else {
                noticeEl.innerHTML = `Select a plan to activate or extend your premium access for unlimited AI Fine-Tuning & Gradient Ascent Machine Unlearning:`;
            }
        }

        if (closeBtn) {
            closeBtn.style.display = forced ? 'none' : 'flex';
        }

        const homeBtn = document.getElementById('sub-modal-home-btn');
        if (homeBtn) {
            homeBtn.style.display = forced ? 'block' : 'none';
        }

        if (subModal) subModal.classList.add('open');
    }

    function goToHomeScreen() {
        localStorage.removeItem('md_engine_user');
        currentUser = null;

        const subModal = document.getElementById('sub-modal');
        if (subModal) subModal.classList.remove('open');

        const authModal = document.getElementById('auth-modal');
        if (authModal) authModal.classList.remove('open');

        const customAlert = document.getElementById('custom-alert-modal');
        if (customAlert) customAlert.classList.remove('open');

        const overlay = document.getElementById('landing-overlay');
        if (overlay) overlay.classList.remove('hidden');

        checkAuthState();
        showToast('Returned to Home Screen.', 'info');
    }

    function logout() {
        localStorage.removeItem('md_engine_user');
        currentUser = null;

        const subModal = document.getElementById('sub-modal');
        if (subModal) subModal.classList.remove('open');

        const authModal = document.getElementById('auth-modal');
        if (authModal) authModal.classList.remove('open');

        const customAlert = document.getElementById('custom-alert-modal');
        if (customAlert) customAlert.classList.remove('open');

        checkAuthState();
        showToast('Logged out successfully.', 'info');
    }

    function startFreeTrial() {
        const demoEmail = `user_${Math.floor(Math.random() * 89999 + 10000)}@trial.com`;
        openAuthModal('signup');
        document.getElementById('auth-email').value = demoEmail;
        document.getElementById('auth-password').value = 'trial1234';
    }

    function openAuthModal(mode = 'signup') {
        authMode = mode;
        const modal = document.getElementById('auth-modal');
        const title = document.getElementById('auth-modal-title');
        const btn = document.getElementById('btn-auth-submit');
        const link = document.getElementById('auth-toggle-link');
        const otpGroup = document.getElementById('otp-step-group');

        if (otpGroup) otpGroup.style.display = 'none';

        if (mode === 'signup') {
            title.textContent = 'Sign Up for 7-Day Free Trial';
            btn.textContent = 'Send Verification OTP';
            link.textContent = 'Already have an account? Log In';
        } else {
            title.textContent = 'Log In to MD Engine';
            btn.textContent = 'Log In';
            link.textContent = "Don't have an account? Sign Up";
        }

        if (modal) modal.classList.add('open');
    }

    async function sendOtpForSignup() {
        const email = document.getElementById('auth-email').value;
        if (!email) {
            showToast('Please enter your email address first.', 'warning');
            return;
        }
        showToast('Sending verification OTP code...', 'info');
        try {
            const res = await api('/auth/send_otp', { email });
            if (res.already_registered) {
                showToast(res.message, 'warning');
                openAuthModal('login');
                document.getElementById('auth-email').value = email;
                return;
            }
            showToast(res.message, 'success');
            const otpGroup = document.getElementById('otp-step-group');
            if (otpGroup) otpGroup.style.display = 'block';
            document.getElementById('btn-auth-submit').textContent = 'Verify OTP & Complete Sign Up';
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    function closeAuthModal() {
        const modal = document.getElementById('auth-modal');
        if (modal) modal.classList.remove('open');

        const stored = localStorage.getItem('md_engine_user');
        if (currentUser || stored) {
            const overlay = document.getElementById('landing-overlay');
            if (overlay) overlay.classList.add('hidden');
        }
    }

    function closeSubModal() {
        if (isUserExpired()) {
            showCustomAlert('Your 7-day free trial or subscription date has expired.\n\nPlease select a subscription plan below to unlock the AI Memory Deletion Engine.', 'Access Expired', '🔒');
            showSubscriptionPage(true);
            return;
        }

        const subModal = document.getElementById('sub-modal');
        if (subModal) subModal.classList.remove('open');

        const authModal = document.getElementById('auth-modal');
        if (authModal) authModal.classList.remove('open');

        const overlay = document.getElementById('landing-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }

    function toggleAuthMode() {
        openAuthModal(authMode === 'signup' ? 'login' : 'signup');
    }

    async function handleAuthSubmit(e) {
        e.preventDefault();
        const email = document.getElementById('auth-email').value;
        const password = document.getElementById('auth-password').value;
        const otpInput = document.getElementById('auth-otp');

        if (authMode === 'login') {
            try {
                const res = await api('/auth/login', { email, password });
                currentUser = res;
                localStorage.setItem('md_engine_user', JSON.stringify(res));

                closeAuthModal();
                const overlay = document.getElementById('landing-overlay');
                if (overlay) overlay.classList.add('hidden');

                showToast(res.message || 'Access granted!', 'success');
                checkAuthState();
            } catch (err) {
                showToast(err.message, 'error');
            }
            return;
        }

        // Signup mode — 2-step OTP flow
        const otp = otpInput ? otpInput.value.trim() : '';

        if (!otp) {
            // Step 1: Send OTP
            await sendOtpForSignup();
            return;
        }

        // Step 2: Verify OTP & Register
        try {
            const res = await api('/auth/verify_otp_signup', { email, password, otp });
            currentUser = res;
            localStorage.setItem('md_engine_user', JSON.stringify(res));

            closeAuthModal();
            const overlay = document.getElementById('landing-overlay');
            if (overlay) overlay.classList.add('hidden');

            showToast(res.message || 'Email verified! 7-day trial activated.', 'success');
            checkAuthState();
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    function checkoutPlan(planId) {
        const user = currentUser || JSON.parse(localStorage.getItem('md_engine_user') || '{}');
        const sub = user.subscription || {};

        const planNames = {
            'plan_1m': '1 Month Plan',
            'plan_6m': '6 Months Plan',
            'plan_12m': '12 Months Plan',
        };

        // Prompt user if they already have an active subscription for the same plan
        if (sub.status === 'active' && sub.plan === planId) {
            const planName = planNames[planId] || 'Subscription Plan';
            const daysLeft = sub.days_left || 0;
            let formattedEnd = '';

            if (sub.expires_at) {
                try {
                    const d = new Date(sub.expires_at);
                    formattedEnd = d.toLocaleDateString('en-US', { day: '2-digit', month: 'short', year: 'numeric' });
                } catch (e) {
                    formattedEnd = sub.expires_at;
                }
            }

            const endNotice = formattedEnd ? `\nEnd Date: ${formattedEnd}` : '';
            const msg = `You already have an active subscription for the ${planName}!\n\nDays Remaining: ${daysLeft} days${endNotice}`;

            showToast(`Active Subscription: ${planName} (${daysLeft} days remaining)`, 'warning', 6000);
            showCustomAlert(msg, 'Active Subscription', '⚠️');
            return;
        }

        const userId = user.user_id || user.id || ('guest_' + Date.now());
        const email = user.email || '';
        const returnUrl = encodeURIComponent('http://127.0.0.1:8001/#payment_success');
        const infinityFreeUrl = `https://onlineclothier.infinityfreeapp.com/payment_accept.php?plan_id=${planId}&user_id=${encodeURIComponent(userId)}&email=${encodeURIComponent(email)}&return_url=${returnUrl}`;
        window.location.href = infinityFreeUrl;
    }

    async function checkPaymentCallback() {
        const hash = window.location.hash || '';
        const search = window.location.search || '';
        const queryStr = search || (hash.includes('?') ? hash.substring(hash.indexOf('?')) : '');
        const urlParams = new URLSearchParams(queryStr);

        const paymentId = urlParams.get('payment_id');
        const planId = urlParams.get('plan_id');
        const userId = urlParams.get('user_id');
        const email = urlParams.get('email');

        if (paymentId && planId && (userId || email)) {
            showToast('Verifying payment & activating subscription...', 'info');
            try {
                const res = await api('/auth/subscribe', {
                    user_id: userId || '',
                    email: email || '',
                    plan_id: planId,
                    payment_id: paymentId,
                });

                if (res.subscription) {
                    if (currentUser) {
                        currentUser.subscription = res.subscription;
                        currentUser.email = res.email || currentUser.email;
                    } else {
                        currentUser = { user_id: res.user_id, email: res.email, subscription: res.subscription };
                    }
                    localStorage.setItem('md_engine_user', JSON.stringify(currentUser));
                }

                showToast(res.message || '🎉 Subscription Activated!', 'success');
                checkAuthState();
                window.history.replaceState({}, document.title, window.location.pathname);
            } catch (err) {
                showToast('Subscription update failed: ' + err.message, 'error');
            }
        }
    }

    // ── Expose Public API ────────────────────────────────────────────────────
    window.App = {
        sendMessage,
        startTraining,
        startUnlearning,
        startEvaluation,
        openPanel,
        closePanel,
        newChat,
        toggleMemory,
        selectAllVisible,
        deselectAll,
        createForgetSet,
        hfUpload,
        hfDownload,
        startFreeTrial,
        openAuthModal,
        closeAuthModal,
        closeSubModal,
        toggleAuthMode,
        handleAuthSubmit,
        checkoutPlan,
        logout,
        sendOtpForSignup,
        showCustomAlert,
        closeCustomAlert,
        isUserExpired,
        goToHomeScreen,
    };

    document.addEventListener('DOMContentLoaded', init);
})();
