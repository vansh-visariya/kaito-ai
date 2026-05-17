/* ============================================================
   Kaito-AI — app.js
   Features:
     1. SSE streaming (POST /api/chat/stream) with fetch ReadableStream
     2. marked.js + highlight.js markdown rendering with copy buttons
     3. Source citations rendered under RAG answers
   ============================================================ */

const API = '';   // same-origin; set to 'http://localhost:8000' for dev

// ── Configure marked.js with highlight.js ─────────────────────────────────
const { markedHighlight } = globalThis.markedHighlight;
marked.use(markedHighlight({
  langPrefix: 'hljs language-',
  highlight(code, lang) {
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language }).value;
  },
}));
marked.use({ gfm: true, breaks: true });

// ── DOM refs ──────────────────────────────────────────────────────────────
const configOverlay = document.getElementById('config-overlay');
const configForm = document.getElementById('config-form');
const configError = document.getElementById('config-error');
const configSubmit = document.getElementById('config-submit');
const appEl = document.getElementById('app');

const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const mobSidebarToggle = document.getElementById('mob-sidebar-toggle');
const newChatBtn = document.getElementById('new-chat-btn');
const threadListEl = document.getElementById('thread-list');
const docListEl = document.getElementById('doc-list');
const clearDocsBtn = document.getElementById('clear-docs-btn');
const cleanThreadsBtn = document.getElementById('clean-threads-btn');
const settingsBtn = document.getElementById('settings-btn');
const modeBadge = document.getElementById('mode-badge');

const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const pdfUpload = document.getElementById('pdf-upload');
const uploadIndicator = document.getElementById('upload-indicator');
const uploadFilename = document.getElementById('upload-filename');
const cancelUpload = document.getElementById('cancel-upload');
const uploadOverlay = document.getElementById('upload-overlay');
const toastEl = document.getElementById('toast');
const topbarLabel = document.getElementById('topbar-thread-label');
const topbarMode = document.getElementById('topbar-mode');

// ── State ─────────────────────────────────────────────────────────────────
let currentThreadId = null;
let pendingFiles = [];
let isStreaming = false;

// ── Generic API helper (JSON only) ────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'API error');
  return data;
}

// ── Toast ─────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info', ms = 2800) {
  toastEl.textContent = msg;
  toastEl.className = `toast ${type}`;
  toastEl.classList.remove('hidden');
  clearTimeout(toastEl._timer);
  toastEl._timer = setTimeout(() => toastEl.classList.add('hidden'), ms);
}

// ── Misc helpers ──────────────────────────────────────────────────────────
function setLoading(btn, loading) {
  const txt = btn.querySelector('.btn-text');
  const spin = btn.querySelector('.btn-spinner');
  if (txt) txt.classList.toggle('hidden', loading);
  if (spin) spin.classList.toggle('hidden', !loading);
  btn.disabled = loading;
}

function autoResize() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

// ── Markdown rendering ────────────────────────────────────────────────────
function renderMarkdown(text) {
  return marked.parse(text || '');
}

/** Add copy-to-clipboard buttons to every <pre><code> block inside el. */
function addCopyButtons(el) {
  el.querySelectorAll('pre code').forEach(block => {
    if (block.parentElement.querySelector('.copy-btn')) return; // already added
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(block.innerText).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
      });
    });
    block.parentElement.appendChild(btn);
  });
}

// ── Source citations ──────────────────────────────────────────────────────
/** Append a sources bar under a bubble element. */
function renderSources(sources, bubble) {
  if (!sources || !sources.length) return;
  const bar = document.createElement('div');
  bar.className = 'sources';
  bar.innerHTML = '<span class="sources-label">Sources</span>';
  sources.forEach(s => {
    const chip = document.createElement('span');
    chip.className = 'source-chip';
    chip.textContent = `📄 ${s.file} · p.${s.page}`;
    chip.title = `${s.file}, page ${s.page}`;
    bar.appendChild(chip);
  });
  bubble.appendChild(bar);
}

// ── Mode UI ───────────────────────────────────────────────────────────────
function updateModeUI(mode) {
  const isRag = mode === 'rag';
  const cls = isRag ? 'mode-rag' : 'mode-search';
  modeBadge.className = `mode-badge ${cls}`;
  modeBadge.innerHTML = `<span class="mode-icon">${isRag ? '📄' : '🔍'}</span><span class="mode-label">${isRag ? 'RAG Mode' : 'Search Mode'}</span>`;
  topbarMode.className = `topbar-mode ${cls}`;
  topbarMode.textContent = isRag ? '📄 RAG' : '🔍 Search';
}

// ── Messages ──────────────────────────────────────────────────────────────
function appendMessage(role, content, sources = []) {
  welcomeEl.classList.add('hidden');

  const wrapper = document.createElement('div');
  wrapper.className = `msg-wrapper ${role}`;

  if (role === 'assistant') {
    const av = document.createElement('div');
    av.className = 'avatar ai';
    av.textContent = 'K';
    wrapper.appendChild(av);
  }

  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.innerHTML = renderMarkdown(content);
  addCopyButtons(bubble);
  renderSources(sources, bubble);
  wrapper.appendChild(bubble);

  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function showTypingIndicator() {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg-wrapper assistant';
  wrapper.id = 'typing-indicator';

  const av = document.createElement('div');
  av.className = 'avatar ai';
  av.textContent = 'K';

  const bubble = document.createElement('div');
  bubble.className = 'bubble assistant typing-bubble';
  bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';

  wrapper.appendChild(av);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrapper;
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function clearMessages() {
  messagesEl.innerHTML = '';
  messagesEl.appendChild(welcomeEl);
  welcomeEl.classList.remove('hidden');
}

function renderHistory(messages) {
  clearMessages();
  if (!messages || !messages.length) return;
  // History messages don't carry sources (they're from the DB not the live stream)
  messages.forEach(m => appendMessage(m.role, m.content));
}

// ── Thread list ───────────────────────────────────────────────────────────
async function loadThreads() {
  try {
    const { threads } = await api('/api/threads');
    threadListEl.innerHTML = '';

    if (!threads.length) {
      threadListEl.innerHTML = '<p class="empty-hint">No conversations yet</p>';
      return;
    }

    threads.forEach(t => {
      const item = document.createElement('div');
      item.className = `thread-item${t.active ? ' active' : ''}`;
      item.dataset.id = t.id;
      item.innerHTML = `
        <span class="thread-preview">${escapeHtml(t.preview)}</span>
        <button class="thread-delete" title="Delete" data-id="${t.id}">×</button>
      `;
      item.addEventListener('click', e => {
        if (e.target.closest('.thread-delete')) return;
        selectThread(t.id);
      });
      item.querySelector('.thread-delete').addEventListener('click', e => {
        e.stopPropagation();
        deleteThread(t.id);
      });
      threadListEl.appendChild(item);
    });
  } catch (err) {
    console.error('loadThreads:', err);
  }
}

async function selectThread(tid) {
  try {
    const data = await api('/api/threads/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: tid }),
    });
    currentThreadId = data.thread_id;
    topbarLabel.textContent = 'Conversation';
    renderHistory(data.messages);
    updateModeUI(data.mode);
    await loadThreads();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function deleteThread(tid) {
  try {
    const data = await api(`/api/threads/${tid}`, { method: 'DELETE' });
    currentThreadId = data.current_thread_id;
    showToast('Thread deleted', 'success');
    await loadThreads();
    const hist = await api(`/api/chat/${currentThreadId}/history`);
    renderHistory(hist.messages);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Documents ─────────────────────────────────────────────────────────────
async function loadDocuments() {
  try {
    const { documents } = await api('/api/documents');
    docListEl.innerHTML = '';
    if (!documents.length) {
      docListEl.innerHTML = '<p class="empty-hint">No documents uploaded</p>';
      clearDocsBtn.classList.add('hidden');
      return;
    }
    clearDocsBtn.classList.remove('hidden');
    documents.forEach(name => {
      const item = document.createElement('div');
      item.className = 'doc-item';
      item.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span class="doc-name">${escapeHtml(name)}</span>
        <button class="doc-delete" data-name="${escapeHtml(name)}" title="Delete document">×</button>`;
      
      item.querySelector('.doc-delete').addEventListener('click', e => {
        e.stopPropagation();
        deleteDocument(name);
      });
      docListEl.appendChild(item);
    });
  } catch (err) {
    console.error('loadDocuments:', err);
  }
}

async function deleteDocument(name) {
  try {
    const data = await api(`/api/documents/${encodeURIComponent(name)}`, { method: 'DELETE' });
    showToast(`Deleted ${name}`, 'success');
    updateModeUI(data.mode);
    if (data.thread_id && data.thread_id !== currentThreadId) {
      currentThreadId = data.thread_id;
      clearMessages();
    }
    await loadDocuments();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Config form ───────────────────────────────────────────────────────────
configForm.addEventListener('submit', async e => {
  e.preventDefault();
  configError.classList.add('hidden');
  setLoading(configSubmit, true);

  const payload = {
    username: document.getElementById('username').value.trim(),
    groq_api_key: document.getElementById('groq-key').value.trim(),
    model_name: document.getElementById('model-name').value.trim() || 'openai/gpt-oss-20b',
    tavily_api_key: document.getElementById('tavily-key').value.trim(),
    langchain_api_key: document.getElementById('langchain-key').value.trim() || null,
  };

  try {
    const data = await api('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    currentThreadId = data.current_thread_id;
    configOverlay.classList.add('hidden');
    appEl.classList.remove('hidden');
    await loadThreads();
    await loadDocuments();
    showToast('Connected ✓', 'success');
  } catch (err) {
    configError.textContent = err.message;
    configError.classList.remove('hidden');
  } finally {
    setLoading(configSubmit, false);
  }
});

// ── Visibility toggles ────────────────────────────────────────────────────
document.querySelectorAll('.toggle-visibility').forEach(btn => {
  btn.addEventListener('click', () => {
    const inp = document.getElementById(btn.dataset.target);
    inp.type = inp.type === 'password' ? 'text' : 'password';
  });
});

// ── New chat ──────────────────────────────────────────────────────────────
newChatBtn.addEventListener('click', async () => {
  try {
    const data = await api('/api/threads/new', { method: 'POST' });
    currentThreadId = data.thread_id;
    topbarLabel.textContent = 'New Conversation';
    clearMessages();
    updateModeUI(data.mode);
    await loadThreads();
  } catch (err) {
    showToast(err.message, 'error');
  }
});

// ── Chat input events ─────────────────────────────────────────────────────
chatInput.addEventListener('input', () => {
  autoResize();
  sendBtn.disabled = !chatInput.value.trim() && !pendingFiles.length;
});

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) handleSend();
  }
});

sendBtn.addEventListener('click', handleSend);

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chatInput.value = chip.dataset.prompt;
    sendBtn.disabled = false;
    autoResize();
    chatInput.focus();
  });
});

// ── SSE streaming send ────────────────────────────────────────────────────
async function handleSend() {
  if (isStreaming) return;
  const text = chatInput.value.trim();
  if (!text && !pendingFiles.length) return;

  isStreaming = true;
  sendBtn.disabled = true;
  chatInput.value = '';
  autoResize();

  try {
    // Upload files first if any are pending
    if (pendingFiles.length) {
      await uploadFiles(pendingFiles);
      pendingFiles = [];
      uploadIndicator.classList.add('hidden');
      pdfUpload.value = '';
    }

    if (!text) { isStreaming = false; sendBtn.disabled = false; return; }

    // User bubble
    appendMessage('user', text);

    // Create empty assistant bubble with streaming cursor
    welcomeEl.classList.add('hidden');
    const wrapper = document.createElement('div');
    wrapper.className = 'msg-wrapper assistant';

    const av = document.createElement('div');
    av.className = 'avatar ai';
    av.textContent = 'K';

    const bubble = document.createElement('div');
    bubble.className = 'bubble assistant streaming-cursor';

    wrapper.appendChild(av);
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Open SSE stream via fetch (POST, not EventSource which only does GET)
    const response = await fetch(API + '/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, thread_id: currentThreadId }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || 'Stream failed');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep the incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let evt;
        try { evt = JSON.parse(raw); } catch { continue; }

        if (evt.type === 'token') {
          fullText += evt.token;
          // Re-render markdown on each token — fast enough with marked
          bubble.innerHTML = renderMarkdown(fullText);
          messagesEl.scrollTop = messagesEl.scrollHeight;

        } else if (evt.type === 'done') {
          bubble.classList.remove('streaming-cursor');
          // Final render + post-processing
          bubble.innerHTML = renderMarkdown(fullText);
          addCopyButtons(bubble);
          renderSources(evt.sources || [], bubble);
          currentThreadId = evt.thread_id;
          updateModeUI(evt.mode);
          await loadThreads();

        } else if (evt.type === 'error') {
          bubble.classList.remove('streaming-cursor');
          bubble.innerHTML = `<span style="color:#f87171">⚠️ ${escapeHtml(evt.message)}</span>`;
        }
      }
    }

  } catch (err) {
    removeTypingIndicator();
    appendMessage('assistant', `⚠️ Error: ${err.message}`);
  } finally {
    isStreaming = false;
    sendBtn.disabled = !chatInput.value.trim();
  }
}

// ── PDF Upload ────────────────────────────────────────────────────────────
pdfUpload.addEventListener('change', () => {
  const files = Array.from(pdfUpload.files);
  if (!files.length) return;
  pendingFiles = files;
  uploadFilename.textContent = files.length === 1 ? files[0].name : `${files.length} PDFs selected`;
  uploadIndicator.classList.remove('hidden');
  sendBtn.disabled = false;
});

cancelUpload.addEventListener('click', () => {
  pendingFiles = [];
  pdfUpload.value = '';
  uploadIndicator.classList.add('hidden');
  sendBtn.disabled = !chatInput.value.trim();
});

async function uploadFiles(files) {
  uploadOverlay.classList.remove('hidden');
  try {
    const form = new FormData();
    files.forEach(f => form.append('files', f));
    const data = await api('/api/documents/upload', { method: 'POST', body: form });
    currentThreadId = data.thread_id;
    topbarLabel.textContent = 'Document Analysis';
    updateModeUI(data.mode);
    showToast(`✅ ${data.uploaded.length} document(s) uploaded`, 'success');
    await loadDocuments();
    await loadThreads();
  } finally {
    uploadOverlay.classList.add('hidden');
  }
}

// ── Clear docs ────────────────────────────────────────────────────────────
clearDocsBtn.addEventListener('click', async () => {
  try {
    const data = await api('/api/documents', { method: 'DELETE' });
    currentThreadId = data.thread_id;
    updateModeUI('search');
    clearMessages();
    showToast('All documents cleared', 'success');
    await loadDocuments();
    await loadThreads();
  } catch (err) {
    showToast(err.message, 'error');
  }
});

// ── Clean empty threads ───────────────────────────────────────────────────
cleanThreadsBtn.addEventListener('click', async () => {
  try {
    const data = await api('/api/threads', { method: 'DELETE' });
    showToast(`Cleaned ${data.count} empty thread(s)`, 'success');
    await loadThreads();
  } catch (err) {
    showToast(err.message, 'error');
  }
});

// ── Reconfigure ───────────────────────────────────────────────────────────
settingsBtn.addEventListener('click', () => {
  appEl.classList.add('hidden');
  configOverlay.classList.remove('hidden');
  configError.classList.add('hidden');
});

// ── Sidebar toggles ───────────────────────────────────────────────────────
sidebarToggle.addEventListener('click', () => sidebar.classList.toggle('collapsed'));
mobSidebarToggle.addEventListener('click', () => sidebar.classList.toggle('mobile-open'));

document.addEventListener('click', e => {
  if (window.innerWidth <= 700 && sidebar.classList.contains('mobile-open')) {
    if (!sidebar.contains(e.target) && e.target !== mobSidebarToggle) {
      sidebar.classList.remove('mobile-open');
    }
  }
});

// ── Check if already configured (page reload) ─────────────────────────────
(async () => {
  try {
    const status = await api('/api/config/status');
    if (status.configured) {
      currentThreadId = status.current_thread_id;
      configOverlay.classList.add('hidden');
      appEl.classList.remove('hidden');
      updateModeUI(status.mode);
      await loadThreads();
      await loadDocuments();
      if (currentThreadId) {
        const hist = await api(`/api/chat/${currentThreadId}/history`);
        renderHistory(hist.messages);
      }
    }
  } catch (_) {
    // Not configured yet — show config modal
  }
})();
