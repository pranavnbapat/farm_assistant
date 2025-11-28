// static/js/chat.js

// -----------------------------
// Auth + header wiring
// -----------------------------
const LS_TOKEN = 'fa_access_token';
const LS_EMAIL = 'fa_email';

const FA_ENV = window.FA_ENV || 'local';

// Backend (Django) bases – used for sessions + logging
const BACKEND_BASES = {
    local: 'http://127.0.0.1:8000',
    dev:   'https://backend-admin.dev.farmbook.ugent.be',
    prd:   'https://backend-admin.prd.farmbook.ugent.be',
};

const BACKEND_BASE   = BACKEND_BASES[FA_ENV] || BACKEND_BASES.local;
const SESSIONS_URL   = `${BACKEND_BASE}/chat/sessions/`;
const LOG_TURN_URL   = `${BACKEND_BASE}/chat/log-turn/`;

// Current auth token and email from login page
let authToken = localStorage.getItem(LS_TOKEN);
const authEmail = localStorage.getItem(LS_EMAIL);

// If no token, send user back to login
if (!authToken) {
    window.location.href = "/";
}

// Decode JWT to get a reasonable display name (first_name / given_name / name)
function getDisplayNameFromToken(token, fallbackEmail) {
    if (!token) return fallbackEmail || '';

    try {
        const payloadBase64 = token.split('.')[1];
        if (!payloadBase64) return fallbackEmail || '';

        // Convert base64url -> base64 and pad so atob works
        const padded = payloadBase64
            .replace(/-/g, '+')
            .replace(/_/g, '/')
            .padEnd(Math.ceil(payloadBase64.length / 4) * 4, '=');

        const payload = JSON.parse(atob(padded));

        const firstName =
            payload.first_name ||
            payload.given_name ||
            (payload.name ? String(payload.name).split(' ')[0] : null);

        return firstName || (fallbackEmail ? fallbackEmail.split('@')[0] : '');
    } catch (err) {
        console.warn('Could not decode JWT for display name', err);
        return fallbackEmail || '';
    }
}

// Put "Welcome {first_name}" + email in the header
const chatUserNameSpan = document.getElementById("chat-user-name");
if (chatUserNameSpan) {
    chatUserNameSpan.textContent = getDisplayNameFromToken(authToken, authEmail);
}
const chatUserEmailSpan = document.getElementById("chat-user-email");
if (chatUserEmailSpan && authEmail) {
    chatUserEmailSpan.textContent = authEmail;
}

// Logout button clears storage and redirects
const logoutBtn = document.getElementById("logout-btn");
if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
        localStorage.removeItem(LS_TOKEN);
        localStorage.removeItem(LS_EMAIL);
        window.location.href = "/";
    });
}

// Helper to build headers for Django backend calls
function backendHeaders() {
    const h = { 'Content-Type': 'application/json' };
    if (authToken) {
        h['Authorization'] = 'Bearer ' + authToken;
    }
    return h;
}

// If Django returns a refreshed access_token, update localStorage + memory
function maybeUpdateTokenFromResponse(data) {
    if (data && data.access_token) {
        authToken = data.access_token;
        localStorage.setItem(LS_TOKEN, authToken);
    }
}

// Show the app layout (remove the "hidden" class)
const appLayout = document.getElementById('app-layout');
if (appLayout) {
    appLayout.classList.remove('hidden');
}


// Current ChatSession.session_uuid (string or null)
let currentSessionUuid = null;
// Last user prompt (for logging)
let lastUserQuestion = '';

// Load all sessions for the sidebar
async function loadSessions() {
    if (!authToken) return;

    try {
        const res = await fetch(SESSIONS_URL, {
            method: 'GET',
            headers: backendHeaders(),
        });

        const data = await res.json();
        maybeUpdateTokenFromResponse(data);

        if (!res.ok || data.status !== 'success') {
            console.warn('Failed to load sessions', data);
            return;
        }
        renderSessionList(data.sessions || []);
    } catch (err) {
        console.error('Error loading sessions', err);
    }
}

function renderSessionList(sessions) {
    const listEl = document.getElementById('session-list');
    if (!listEl) return;
    listEl.innerHTML = '';

    sessions.forEach(sess => {
        const li = document.createElement('li');
        li.className = 'session-item';
        li.dataset.uuid = sess.session_uuid;
        li.textContent = sess.title || '(untitled)';
        li.addEventListener('click', () => openSession(sess.session_uuid));
        listEl.appendChild(li);
    });
}

// Create a new session (called on first question or when user presses "+ New")
async function createNewSession(initialTitle = '') {
    if (!authToken) return null;

    try {
        const res = await fetch(SESSIONS_URL, {
            method: 'POST',
            headers: backendHeaders(),
            body: JSON.stringify({
                title: initialTitle.slice(0, 120),
                metadata: { created_from: 'farm-assistant-ui' },
            }),
        });

        const data = await res.json();
        maybeUpdateTokenFromResponse(data);

        if (!res.ok || data.status !== 'success') {
            console.warn('Failed to create session', data);
            return null;
        }

        currentSessionUuid = data.session_uuid;
        // Reload list so the new chat appears at the top
        loadSessions().catch(console.error);
        return currentSessionUuid;
    } catch (err) {
        console.error('Error creating session', err);
        return null;
    }
}

// Ensure a session exists before logging the first turn
async function ensureSessionExists(firstUserText) {
    if (currentSessionUuid) return currentSessionUuid;
    return await createNewSession(firstUserText || '');
}

// Load one session and render its messages
async function openSession(sessionUuid) {
    if (!authToken) return;
    currentSessionUuid = sessionUuid;

    try {
        const res = await fetch(`${SESSIONS_URL}${sessionUuid}/`, {
            method: 'GET',
            headers: backendHeaders(),
        });

        const data = await res.json();
        maybeUpdateTokenFromResponse(data);

        if (!res.ok || data.status !== 'success') {
            console.warn('Failed to load session', data);
            return;
        }

        const chatEl = document.getElementById('chat');
        if (!chatEl) return;
        chatEl.innerHTML = '';

        (data.messages || []).forEach(m => {
            addMessage(m.role === 'user' ? 'you' : 'assistant', m.content);
        });
    } catch (err) {
        console.error('Error loading session detail', err);
    }
}

// "+ New" button
const newChatBtn = document.getElementById('new-chat');
if (newChatBtn) {
    newChatBtn.addEventListener('click', async () => {
        const chatEl = document.getElementById('chat');
        if (chatEl) chatEl.innerHTML = '';
        currentSessionUuid = await createNewSession('');
        const inputEl = document.getElementById('question');
        if (inputEl) inputEl.focus();
    });
}

// Log each completed Q&A turn to Django
async function logChatTurnToBackend(userText, assistantText, latencyMs) {
    if (!authToken) return;
    if (!userText || !assistantText) return;

    try {
        const res = await fetch(LOG_TURN_URL, {
            method: 'POST',
            headers: backendHeaders(),
            body: JSON.stringify({
                session_uuid: currentSessionUuid,   // may be null; backend will create
                user_message: userText,
                assistant_message: assistantText,
                meta: {
                    model: selModel ? selModel.value : null,
                    latency_ms: latencyMs,
                },
            }),
        });

        const data = await res.json();
        maybeUpdateTokenFromResponse(data);

        if (!res.ok || data.status !== 'success') {
            console.warn('Failed to log chat turn', data);
            return;
        }

        // Backend may create a session if we passed null
        if (!currentSessionUuid && data.session_uuid) {
            currentSessionUuid = data.session_uuid;
            loadSessions().catch(console.error);
        }
    } catch (err) {
        console.error('Error logging chat turn', err);
    }
}

// -----------------------------
// Core chat streaming logic
// -----------------------------

const chat = document.getElementById('chat');
const form = document.getElementById('qform');
const input = document.getElementById('question');
const btnSend = document.getElementById('send');
const btnStop = document.getElementById('stop');
const selModel = document.getElementById('model');

let es = null;
let answerNode = null;
let statusNode = null;
let cancelled = false;
let pendingSources = [];

let clientStartTs = 0;
let serverTimingMs = null;
let statusLeft = null;
let statusRight = null;

function formatDuration(ms) {
    if (ms == null || !isFinite(ms)) return '';
    if (ms < 1000) return '<1s';
    const secs = Math.floor(ms / 1000);
    const s = secs % 60;
    const m = Math.floor(secs / 60) % 60;
    const h = Math.floor(secs / 3600);
    if (h > 0 && m > 0 && s > 0) return `${h}h ${m}m ${s}s`;
    if (h > 0 && m > 0) return `${h}h ${m}m`;
    if (h > 0) return `${h}h`;
    if (m > 0 && s > 0) return `${m}m ${s}s`;
    if (m > 0) return `${m}m`;
    return `${s}s`;
}

function insertThinkTime(container, ms) {
    const line = document.createElement('div');
    line.className = 'muted';
    line.style.marginTop = '10px';
    line.style.marginBottom = '6px';

    container.parentElement.appendChild(line);
    return line;
}

function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, text) {
    const box = document.createElement('div');
    box.className = `msg ${role}`;
    const who = document.createElement('div');
    who.className = 'role';
    who.textContent = role === 'you' ? 'You' : 'Assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text || '';
    box.appendChild(who);
    box.appendChild(bubble);
    chat.appendChild(box);
    scrollToBottom();

    return bubble;
}

function setStatus(text, thinking = false) {
    if (!statusNode) {
        statusNode = document.createElement('div');
        statusNode.className = 'status';

        statusLeft = document.createElement('span');
        statusLeft.className = 'left';

        statusRight = document.createElement('span');
        statusRight.className = 'right';

        statusNode.appendChild(statusLeft);
        statusNode.appendChild(statusRight);
        answerNode.parentElement.appendChild(statusNode);
    }

    statusLeft.textContent = text;

    if (thinking) statusNode.classList.add('blink');
    else statusNode.classList.remove('blink');
}

function renderSourcesInline(container, items) {
    if (!items || !items.length) return;

    const title = document.createElement('div');
    title.className = 'sources-title';
    title.textContent = 'References';

    const ul = document.createElement('ul');
    ul.className = 'sources-list';

    items.forEach(s => {
        const li = document.createElement('li');

        const sup = document.createElement('sup');
        sup.className = 'cite';
        sup.textContent = String(s.n || '?') + ' ';
        li.appendChild(sup);

        const a = document.createElement('a');
        a.href = s.url || '#';
        a.textContent = s.title || s.id || s.url || '(untitled)';
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        li.appendChild(a);

        const meta = document.createElement('span');
        const lic = s.license ? ` • ${s.license}` : '';
        meta.className = 'muted';
        meta.textContent = lic;

        ul.appendChild(li);
    });

    const holder = document.createElement('div');
    holder.appendChild(title);
    holder.appendChild(ul);
    answerNode.parentElement.appendChild(holder);
}

function escapeHTML(s) {
    return s.replace(/[&<>"']/g, m => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[m]));
}

function renderCitationsToSuperscript(node) {
    const txt = node.textContent;
    const html = escapeHTML(txt).replace(/\[(\d+)\]/g, '<sup class="cite">$1</sup>');
    node.innerHTML = html;
}

function startStream(q) {
    if (es) {
        es.close();
        es = null;
    }
    cancelled = false;
    pendingSources = [];
    statusNode = null;
    serverTimingMs = null;
    clientStartTs = Date.now();

    btnSend.disabled = true;
    btnStop.classList.remove('hidden');
    chat.setAttribute('aria-busy', 'true');

    addMessage('you', q);
    answerNode = addMessage('assistant', '');

    if (window.onAssistantStreamStart) {
        window.onAssistantStreamStart();
    }

    const url = `/ask/stream?q=${encodeURIComponent(q)}&page=1&max_tokens=-1&model=${encodeURIComponent(selModel.value)}`;
    es = new EventSource(url);

    es.addEventListener('status', (e) => {
        try {
            const obj = JSON.parse(e.data);
            const isThinking = (obj.stage === 'LLM');
            setStatus(`${obj.stage}: ${obj.message}`, isThinking);
        } catch {
            setStatus(e.data, false);
        }
    });

    es.addEventListener('token', (e) => {
        answerNode.textContent += e.data;
        scrollToBottom();
        if (window.onAssistantToken) {
            window.onAssistantToken(e.data);
        }
    });

    es.addEventListener('sources', (e) => {
        try {
            pendingSources = JSON.parse(e.data) || [];
        } catch {
            pendingSources = [];
        }
    });

    es.addEventListener('timing', (e) => {
        try {
            const t = JSON.parse(e.data);
            serverTimingMs = (t && typeof t.total_ms === 'number') ? t.total_ms : null;
        } catch {
            serverTimingMs = null;
        }
    });

    es.addEventListener('stats', () => { /* noop */ });

    es.addEventListener('done', () => {
        setStatus(cancelled ? 'Stopped.' : 'Completed.');

        renderCitationsToSuperscript(answerNode);

        const ms = (serverTimingMs != null ? serverTimingMs : (Date.now() - clientStartTs));
        if (statusRight) statusRight.textContent = `Thought for ${formatDuration(ms)}`;

        insertThinkTime(answerNode, ms);
        renderSourcesInline(answerNode, pendingSources);

        const assistantText = answerNode ? answerNode.textContent : '';
        // Fire-and-forget log to backend
        logChatTurnToBackend(lastUserQuestion, assistantText, ms);

        if (window.onAssistantStreamEnd) {
            window.onAssistantStreamEnd();
        }
        cleanup();
    });

    es.addEventListener('error', () => {
        setStatus('Error or connection closed.');
        if (window.onAssistantStreamEnd) {
            window.onAssistantStreamEnd();
        }
        cleanup();
    });
}

function cleanup() {
    if (es) {
        es.close();
        es = null;
    }
    btnSend.disabled = false;
    btnStop.classList.add('hidden');
    chat.setAttribute('aria-busy', 'false');
    if (statusNode) statusNode.classList.remove('blink');
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
    scrollToBottom();
}

// Submit handler: ensure session exists, then stream
if (form) {
    form.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        const q = input.value.trim();
        if (!q) return;

        lastUserQuestion = q;

        // Make sure there is a ChatSession row for this conversation
        await ensureSessionExists(q);

        startStream(q);
        input.value = '';
        input.focus();
    });
}

// Stop button cancels current stream
if (btnStop) {
    btnStop.addEventListener('click', () => {
        cancelled = true;
        cleanup();
    });
}

// Convenience: Enter sends, Shift+Enter adds newline
if (input && form) {
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            form.requestSubmit();
        }
    });
}

// Initial load of sessions when the page opens
loadSessions().catch(console.error);
