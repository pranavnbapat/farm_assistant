// static/js/chat.js

// -----------------------------
// Auth + header wiring
// -----------------------------
const LS_TOKEN = 'fa_access_token';
const LS_EMAIL = 'fa_email';

const FA_ENV = window.FA_ENV || 'local';

// Use proxy endpoints through FastAPI (avoids CORS issues)
// The FastAPI backend will forward these to the Django backend
const SESSIONS_URL   = '/proxy/chat/sessions/';
const LOG_TURN_URL   = '/proxy/chat/log-turn/';

// Current auth token and email from login page / login.js
let authToken    = localStorage.getItem(LS_TOKEN);
let refreshToken = localStorage.getItem('fa_refresh_token') || null;

// Email: prefer the new key 'fa_user_email', fall back to legacy 'fa_email'
const storedUserEmail = localStorage.getItem('fa_user_email') || localStorage.getItem(LS_EMAIL);
const authEmail = storedUserEmail || '';

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

// Centralised forced-logout handler
function forceLogout(reason) {
    console.warn('Forcing logout:', reason);

    // Best-effort: tell Django to invalidate all tokens for this user
    try {
        const email =
            localStorage.getItem('fa_user_email') ||
            localStorage.getItem('fa_email');

        if (email) {
            fetch('/proxy/logout/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email }),
            }).catch(() => {
                // Ignore network/logging errors here – logout must not break the UI
            });
        }
    } catch (e) {
        console.warn('Backend logout failed', e);
    }

    // Clear local tokens
    authToken    = null;
    refreshToken = null;

    localStorage.removeItem('fa_access_token');
    localStorage.removeItem('fa_refresh_token');
    localStorage.removeItem('fa_user_uuid');
    localStorage.removeItem('fa_user_email');
    localStorage.removeItem('fa_email');

    // Reset UI bits if present
    const chatEl = document.getElementById('chat');
    if (chatEl) chatEl.innerHTML = '';

    const sessList = document.getElementById('session-list');
    if (sessList) sessList.innerHTML = '';

    // Go back to login
    window.location.href = "/";
}

// Logout button uses forceLogout
const logoutBtn = document.getElementById("logout-btn");
if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
        forceLogout('user_clicked_logout');
    });
}

// Helper to build headers for Django backend calls
function backendHeaders() {
    const h = { 'Content-Type': 'application/json' };

    // Send the current access token for normal auth
    if (authToken) {
        h['Authorization'] = 'Bearer ' + authToken;
    }

    // Send refresh token so Django can transparently refresh if needed
    if (refreshToken) {
        h['X-Refresh-Token'] = refreshToken;
    }

    return h;
}

// If Django returns a refreshed access_token (from validate_authorization_header),
// update localStorage + in-memory variables.
function maybeUpdateTokenFromResponse(data) {
    if (!data || typeof data !== 'object') return;

    if (data.access_token) {
        authToken = data.access_token;
        localStorage.setItem('fa_access_token', authToken);
    }

    // Optional: if backend also returns a new refresh token in future
    if (data.refresh_token) {
        refreshToken = data.refresh_token;
        localStorage.setItem('fa_refresh_token', refreshToken);
    }
}

// Show the app layout (remove the "hidden" class)
const appLayout = document.getElementById('app-layout');
if (appLayout) {
    appLayout.classList.remove('hidden');
}


// -----------------------------
// Sessions + logging
// -----------------------------

// Current ChatSession.session_uuid (string or null)
let currentSessionUuid = null;
// Last user prompt (for logging)
let lastUserQuestion = '';

// Shared helper: handle 401/400 auth failures from Django
function handleAuthFailure(res, data, contextLabel) {
    if (res.status === 401 || res.status === 400) {
        // Access token invalid AND refresh token invalid/absent
        forceLogout(
            `${contextLabel}_auth_failed: ` + (data && data.message ? data.message : '')
        );
        return true;
    }
    return false;
}

// Load all sessions for the sidebar
async function loadSessions() {
    if (!authToken) return;

    try {
        const res = await fetch(SESSIONS_URL, {
            method: 'GET',
            headers: backendHeaders(),
        });

        const data = await res.json().catch(() => ({}));
        maybeUpdateTokenFromResponse(data);

        if (!res.ok) {
             // 1) deal with auth failures (401 / 400) in a single place
            if (handleAuthFailure(res, data, 'loadSessions')) return;

             // 2) non-auth HTTP error
            console.warn('Failed to load sessions (HTTP)', res.status, data);
            return;
        }

        // 3) HTTP OK but API reported failure
        if (data.status !== 'success') {
            console.warn('Failed to load sessions (status)', data);
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

        // Text span so we can add a delete button next to it
        const titleSpan = document.createElement('span');
        titleSpan.textContent = sess.title || '(untitled)';
        titleSpan.addEventListener('click', () => openSession(sess.session_uuid));

        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.textContent = '🗑';
        delBtn.title = 'Delete this chat';
        delBtn.className = 'session-delete-btn';
        delBtn.style.marginLeft = '8px';

        delBtn.addEventListener('click', async (e) => {
            e.stopPropagation();  // don’t trigger openSession
            const ok = window.confirm('Delete this chat?');
            if (!ok) return;

            try {
                const res = await fetch(`${SESSIONS_URL}${sess.session_uuid}/`, {
                    method: 'DELETE',
                    headers: backendHeaders(),
                });

                const data = await res.json().catch(() => ({}));
                maybeUpdateTokenFromResponse(data);

                if (!res.ok) {
                    if (handleAuthFailure(res, data, 'deleteSession')) return;
                    console.warn('Failed to delete session (HTTP)', res.status, data);
                    return;
                }

                if (data.status !== 'success') {
                    console.warn('Failed to delete session (status)', data);
                    return;
                }

                // If we were viewing this session, clear it
                if (currentSessionUuid === sess.session_uuid) {
                    currentSessionUuid = null;
                    const chatEl = document.getElementById('chat');
                    if (chatEl) chatEl.innerHTML = '';
                }

                // Reload the list
                loadSessions().catch(console.error);
            } catch (err) {
                console.error('Error deleting session', err);
            }
        });

        li.appendChild(titleSpan);
        li.appendChild(delBtn);
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

        const data = await res.json().catch(() => ({}));
        maybeUpdateTokenFromResponse(data);

        if (!res.ok) {
            if (handleAuthFailure(res, data, 'createNewSession')) return null;
            console.warn('Failed to create session (HTTP)', res.status, data);
            return null;
        }

        if (data.status !== 'success') {
            console.warn('Failed to create session (status)', data);
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

        const data = await res.json().catch(() => ({}));
        maybeUpdateTokenFromResponse(data);

        if (!res.ok) {
            if (handleAuthFailure(res, data, 'openSession')) return;
            console.warn('Failed to load session (HTTP)', res.status, data);
            return;
        }

        if (data.status !== 'success') {
            console.warn('Failed to load session (status)', data);
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
    console.log('Logging chat turn:', { userText: userText?.substring(0, 50), assistantText: assistantText?.substring(0, 50), session: currentSessionUuid });
    
    if (!authToken) {
        console.warn('Cannot log: no auth token');
        return;
    }
    if (!userText || !assistantText) {
        console.warn('Cannot log: missing text', { userText, assistantText });
        return;
    }

    console.log('Sending to:', LOG_TURN_URL);
    
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

        const data = await res.json().catch(() => ({}));
        maybeUpdateTokenFromResponse(data);

        if (!res.ok) {
            if (handleAuthFailure(res, data, 'logChatTurn')) return;
            console.warn('Failed to log chat turn (HTTP)', res.status, data);
            return;
        }

        if (data.status !== 'success') {
            console.warn('Failed to log chat turn (status)', data);
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

// Auto-resize textarea
function autoResizeTextarea() {
    if (!input) return;
    input.style.height = 'auto';
    const newHeight = Math.min(input.scrollHeight, 200); // max-height: 200px
    input.style.height = newHeight + 'px';
}

if (input) {
    input.addEventListener('input', autoResizeTextarea);
}

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

// Add a small "speak" icon for a given assistant bubble.
function attachSpeakButton(bubble) {
    if (!bubble || !window.speechSynthesis) return;

    const container = bubble.parentElement; // .msg wrapper

    // Avoid duplicates on the same message
    const existing = container.querySelector('.speak-btn');
    if (existing) existing.remove();

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'speak-btn';
    btn.textContent = '🔊';
    btn.title = 'Listen to this answer';

    btn.style.marginLeft = '8px';
    btn.style.cursor = 'pointer';
    btn.style.border = 'none';
    btn.style.background = 'transparent';
    btn.style.fontSize = '14px';
    btn.style.padding = '0';

    btn.addEventListener('click', () => {
        const text = bubble.textContent || '';
        if (!text.trim()) return;

        const utter = new SpeechSynthesisUtterance(text);

        // Prefer an en-GB voice if available
        const voices = window.speechSynthesis.getVoices() || [];
        const gbVoices = voices.filter(v =>
            v.lang && v.lang.toLowerCase().startsWith('en-gb')
        );
        if (gbVoices.length) {
            utter.voice = gbVoices[0];
        }

        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utter);
    });

    if (statusLeft) {
        statusLeft.appendChild(document.createTextNode(' '));
        statusLeft.appendChild(btn);
    } else if (statusNode) {
        statusNode.appendChild(btn);
    } else {
        container.appendChild(btn);
    }
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

    const chatBackendBase = window.CHAT_BACKEND_URL || '';

    const params = new URLSearchParams({
        q,
        page: '1',
        max_tokens: '-1',
        model: selModel.value,
    });

    if (currentSessionUuid) {
        params.append('session_id', currentSessionUuid);
    }

    // Add auth token as query param since SSE doesn't support custom headers
    console.log('DEBUG: authToken exists?', !!authToken, 'token length:', authToken ? authToken.length : 0);
    if (authToken) {
        params.append('auth_token', 'Bearer ' + authToken);
    }
    
    const url = `${chatBackendBase}/ask/stream?` + params.toString();
    console.log('DEBUG: SSE URL (without token):', url.replace(/auth_token=Bearer%20[^&]+/, 'auth_token=***'));
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
            const parsed = JSON.parse(e.data);
            // null or empty array means explicitly clear sources
            pendingSources = parsed || [];
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

        // Add a speak icon for this completed assistant answer
        attachSpeakButton(answerNode);

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
        input.style.height = 'auto';
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

