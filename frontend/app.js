/* ============================================================
   Kaito-AI — app.js
   Features:
     1. Login / Register authentication
     2. SSE streaming (POST /api/chat/stream) with fetch ReadableStream
     3. marked.js + highlight.js markdown rendering with copy buttons
     4. Source citations rendered under RAG answers
     5. Multi-thread management per user account
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
const authOverlay = document.getElementById('auth-overlay');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const loginError = document.getElementById('login-error');
const registerError = document.getElementById('register-error');
const loginSubmit = document.getElementById('login-submit');
const registerSubmit = document.getElementById('register-submit');
const tabLogin = document.getElementById('tab-login');
const tabRegister = document.getElementById('tab-register');
const appEl = document.getElementById('app');

const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const mobSidebarToggle = document.getElementById('mob-sidebar-toggle');
const newChatBtn = document.getElementById('new-chat-btn');
const threadListEl = document.getElementById('thread-list');
const docListEl = document.getElementById('doc-list');
const clearDocsBtn = document.getElementById('clear-docs-btn');
const cleanThreadsBtn = document.getElementById('clean-threads-btn');
const logoutBtn = document.getElementById('logout-btn');

const userProfile = document.getElementById('user-profile');
const userAvatar = document.getElementById('user-avatar');
const userDisplayName = document.getElementById('user-display-name');
const userDisplayEmail = document.getElementById('user-display-email');

const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const pdfUpload = document.getElementById('pdf-upload');
const uploadIndicator = document.getElementById('upload-indicator');
const uploadFilename = document.getElementById('upload-filename');
const cancelUpload = document.getElementById('cancel-upload');
const uploadOverlay = document.getElementById('upload-overlay');
const toastEl = document.getElementById('toast');
const topbarLabel = document.getElementById('topbar-thread-label');
const topbarDocsBadge = document.getElementById('topbar-docs-badge');

// ── State ─────────────────────────────────────────────────────────────────
let currentThreadId = null;
let pendingFiles = [];
let isStreaming = false;
let currentUsername = '';
let currentEmail = '';
let currentAbortController = null;
let messageCount = 0;

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

// ── User profile UI ──────────────────────────────────────────────────────
function updateUserProfile(username, email) {
  currentUsername = username;
  currentEmail = email || '';
  userDisplayName.textContent = username;
  userDisplayEmail.textContent = email || '';
  userAvatar.textContent = (username || 'U').charAt(0).toUpperCase();
}

// ── Document badge ───────────────────────────────────────────────────────
function updateDocsBadge(hasDocs) {
  if (hasDocs) {
    topbarDocsBadge.classList.remove('hidden');
  } else {
    topbarDocsBadge.classList.add('hidden');
  }
}

// ── Messages ──────────────────────────────────────────────────────────────
function appendMessage(role, content, sources = []) {
  welcomeEl.classList.add('hidden');

  const wrapper = document.createElement('div');
  wrapper.className = `msg-wrapper ${role}`;
  wrapper.dataset.index = messageCount++;

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

  if (role === 'user') {
    const editBtn = document.createElement('button');
    editBtn.className = 'btn-edit';
    editBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>`;
    editBtn.title = 'Edit Message';
    
    editBtn.addEventListener('click', () => {
      const currentText = content;
      bubble.innerHTML = '';
      
      const editWrapper = document.createElement('div');
      editWrapper.className = 'edit-wrapper';
      
      const textarea = document.createElement('textarea');
      textarea.value = currentText;
      
      const actions = document.createElement('div');
      actions.className = 'edit-actions';
      
      const cancelBtn = document.createElement('button');
      cancelBtn.className = 'btn-edit-cancel';
      cancelBtn.textContent = 'Cancel';
      
      const saveBtn = document.createElement('button');
      saveBtn.className = 'btn-edit-save';
      saveBtn.textContent = 'Save & Regenerate';
      
      cancelBtn.addEventListener('click', () => {
        bubble.innerHTML = renderMarkdown(currentText);
        addCopyButtons(bubble);
        bubble.appendChild(editBtn);
      });
      
      saveBtn.addEventListener('click', async () => {
        const newText = textarea.value.trim();
        if (!newText) return;
        
        try {
          const data = await api('/api/threads/branch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_id: currentThreadId, edit_index: parseInt(wrapper.dataset.index) })
          });
          
          currentThreadId = data.thread_id;
          chatInput.value = newText;
          autoResize();
          await loadThreads();
          
          const hist = await api(`/api/chat/${currentThreadId}/history`);
          renderHistory(hist.messages);
          
          handleSend();
        } catch(e) {
          showToast(e.message, 'error');
        }
      });
      
      actions.appendChild(cancelBtn);
      actions.appendChild(saveBtn);
      editWrapper.appendChild(textarea);
      editWrapper.appendChild(actions);
      bubble.appendChild(editWrapper);
      
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 180) + 'px';
      textarea.focus();
    });
    
    bubble.appendChild(editBtn);
  }

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
  messageCount = 0;
}

function renderHistory(messages) {
  clearMessages();
  if (!messages || !messages.length) return;
  // History messages might carry sources
  messages.forEach(m => appendMessage(m.role, m.content, m.sources));
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
      updateDocsBadge(false);
      return;
    }
    clearDocsBtn.classList.remove('hidden');
    updateDocsBadge(true);
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
    updateDocsBadge(data.has_documents);
    await loadDocuments();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Auth Tabs ─────────────────────────────────────────────────────────────
tabLogin.addEventListener('click', () => {
  tabLogin.classList.add('active');
  tabRegister.classList.remove('active');
  loginForm.classList.remove('hidden');
  registerForm.classList.add('hidden');
  loginError.classList.add('hidden');
  registerError.classList.add('hidden');
});

tabRegister.addEventListener('click', () => {
  tabRegister.classList.add('active');
  tabLogin.classList.remove('active');
  registerForm.classList.remove('hidden');
  loginForm.classList.add('hidden');
  loginError.classList.add('hidden');
  registerError.classList.add('hidden');
});

// ── Login form ────────────────────────────────────────────────────────────
loginForm.addEventListener('submit', async e => {
  e.preventDefault();
  loginError.classList.add('hidden');
  setLoading(loginSubmit, true);

  const payload = {
    email: document.getElementById('login-email').value.trim(),
    password: document.getElementById('login-password').value,
  };

  try {
    const data = await api('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    currentThreadId = data.current_thread_id;
    updateUserProfile(data.username, payload.email);
    authOverlay.classList.add('hidden');
    appEl.classList.remove('hidden');
    await loadThreads();
    await loadDocuments();
    showToast(`Welcome back, ${data.username}!`, 'success');
  } catch (err) {
    loginError.textContent = err.message;
    loginError.classList.remove('hidden');
  } finally {
    setLoading(loginSubmit, false);
  }
});

// ── Register form ─────────────────────────────────────────────────────────
registerForm.addEventListener('submit', async e => {
  e.preventDefault();
  registerError.classList.add('hidden');

  const password = document.getElementById('reg-password').value;
  const confirm = document.getElementById('reg-confirm').value;

  if (password !== confirm) {
    registerError.textContent = 'Passwords do not match.';
    registerError.classList.remove('hidden');
    return;
  }

  setLoading(registerSubmit, true);

  const payload = {
    username: document.getElementById('reg-username').value.trim(),
    email: document.getElementById('reg-email').value.trim(),
    password: password,
  };

  try {
    const data = await api('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    currentThreadId = data.current_thread_id;
    updateUserProfile(data.username, payload.email);
    authOverlay.classList.add('hidden');
    appEl.classList.remove('hidden');
    await loadThreads();
    await loadDocuments();
    showToast(`Welcome, ${data.username}! Account created.`, 'success');
  } catch (err) {
    registerError.textContent = err.message;
    registerError.classList.remove('hidden');
  } finally {
    setLoading(registerSubmit, false);
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

stopBtn.addEventListener('click', () => {
  if (currentAbortController) {
    currentAbortController.abort();
  }
});

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
  sendBtn.classList.add('hidden');
  stopBtn.classList.remove('hidden');
  stopBtn.disabled = false;
  chatInput.value = '';
  autoResize();

  currentAbortController = new AbortController();

  try {
    // Upload files first if any are pending
    if (pendingFiles.length) {
      await uploadFiles(pendingFiles);
      pendingFiles = [];
      uploadIndicator.classList.add('hidden');
      pdfUpload.value = '';
    }

    if (!text) { isStreaming = false; sendBtn.classList.remove('hidden'); stopBtn.classList.add('hidden'); return; }

    // User bubble
    appendMessage('user', text);

    // Create empty assistant bubble with streaming cursor
    welcomeEl.classList.add('hidden');
    const wrapper = document.createElement('div');
    wrapper.className = 'msg-wrapper assistant';
    wrapper.dataset.index = messageCount++;

    const av = document.createElement('div');
    av.className = 'avatar ai';
    av.textContent = 'K';

    const bubble = document.createElement('div');
    bubble.className = 'bubble assistant streaming-cursor';

    wrapper.appendChild(av);
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Open SSE stream via fetch
    const response = await fetch(API + '/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, thread_id: currentThreadId }),
      signal: currentAbortController.signal,
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
          // Re-render markdown on each token
          bubble.innerHTML = renderMarkdown(fullText);
          messagesEl.scrollTop = messagesEl.scrollHeight;

        } else if (evt.type === 'done') {
          bubble.classList.remove('streaming-cursor');
          // Final render + post-processing
          bubble.innerHTML = renderMarkdown(fullText);
          addCopyButtons(bubble);
          renderSources(evt.sources || [], bubble);
          currentThreadId = evt.thread_id;
          await loadThreads();

        } else if (evt.type === 'error') {
          bubble.classList.remove('streaming-cursor');
          bubble.innerHTML = `<span style="color:#f87171">⚠️ ${escapeHtml(evt.message)}</span>`;
        }
      }
    }

  } catch (err) {
    removeTypingIndicator();
    if (err.name === 'AbortError') {
      const bubble = messagesEl.lastElementChild.querySelector('.bubble.assistant');
      if (bubble) {
        bubble.classList.remove('streaming-cursor');
        bubble.innerHTML += `<br/><br/><span style="color:#f87171">⚠️ Generation stopped by user.</span>`;
      } else {
        appendMessage('assistant', `⚠️ Generation stopped by user.`);
      }
    } else {
      appendMessage('assistant', `⚠️ Error: ${err.message}`);
    }
  } finally {
    isStreaming = false;
    currentAbortController = null;
    sendBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
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
    updateDocsBadge(true);
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
    updateDocsBadge(false);
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
    showToast(`Cleaned ${data.deleted.length} empty thread(s)`, 'success');
    await loadThreads();
  } catch (err) {
    showToast(err.message, 'error');
  }
});

// ── Logout ────────────────────────────────────────────────────────────────
logoutBtn.addEventListener('click', async () => {
  try {
    await api('/api/auth/logout', { method: 'POST' });
  } catch (_) {
    // Ignore errors during logout
  }
  appEl.classList.add('hidden');
  authOverlay.classList.remove('hidden');
  loginError.classList.add('hidden');
  registerError.classList.add('hidden');
  // Reset state
  currentThreadId = null;
  currentUsername = '';
  currentEmail = '';
  showToast('Logged out', 'success');
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

// ── Check if already authenticated (page reload) ──────────────────────────
(async () => {
  try {
    const status = await api('/api/auth/status');
    if (status.authenticated) {
      currentThreadId = status.current_thread_id;
      updateUserProfile(status.username, '');
      updateDocsBadge(status.has_documents);
      authOverlay.classList.add('hidden');
      appEl.classList.remove('hidden');
      await loadThreads();
      await loadDocuments();
      if (currentThreadId) {
        const hist = await api(`/api/chat/${currentThreadId}/history`);
        renderHistory(hist.messages);
      }
    }
  } catch (_) {
    // Not authenticated — show auth modal
  }
})();
