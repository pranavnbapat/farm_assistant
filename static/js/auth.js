// static/js/auth.js

// These are for compatibility with older code; keep them but derive from authToken
let accessToken = authToken;
let refreshToken = null;
let userUuid = null;

// Current chat session (Django ChatSession.session_uuid)
let currentSessionUuid = null;

// For logging the last Q&A turn
let lastUserQuestion = '';


// Simple storage keys
const LS_ACCESS = 'fa_access_token';
const LS_REFRESH = 'fa_refresh_token';
const LS_USER_UUID = 'fa_user_uuid';
const LS_USER_EMAIL = 'fa_user_email';

// DOM elements used in this file
const loginForm = document.getElementById('login-form');
const loginStatus = document.getElementById('login-status');
const authInfo = document.getElementById('auth-info');
const authUserEmail = document.getElementById('auth-user-email');
const logoutBtn = document.getElementById('logout-btn');
const appLayout = document.getElementById('app-layout');

// --- Helpers ---------------------------------------------------------

function restoreAuth() {
    accessToken = localStorage.getItem(LS_ACCESS);
    refreshToken = localStorage.getItem(LS_REFRESH);
    userUuid = localStorage.getItem(LS_USER_UUID);
    const email = localStorage.getItem(LS_USER_EMAIL);

    if (accessToken && userUuid) {
        // Show "logged in" view
        if (loginForm) loginForm.classList.add('hidden');
        if (authInfo) authInfo.classList.remove('hidden');
        if (authUserEmail && email) authUserEmail.textContent = email;

        if (appLayout) appLayout.classList.remove('hidden');

        // Load existing chat sessions
        loadSessions().catch(console.error);
    } else {
        // Not logged in
        if (appLayout) appLayout.classList.add('hidden');
        if (loginForm) loginForm.classList.remove('hidden');
        if (authInfo) authInfo.classList.add('hidden');
    }
}

// Build headers with auth token
function authHeaders() {
    const h = { 'Content-Type': 'application/json' };
    if (accessToken) {
        h['Authorization'] = 'Bearer ' + accessToken;
    }
    return h;
}

// Expose helpers globally so chat.js can use them if needed
window.faAuth = {
    authHeaders,
    get accessToken() { return accessToken; },
    get userUuid() { return userUuid; },
    get currentSessionUuid() { return currentSessionUuid; },
    setCurrentSession(uuid) { currentSessionUuid = uuid; },
};

// --- Login -----------------------------------------------------------

if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('login-email').value.trim();
        const password = document.getElementById('login-password').value;

        loginStatus.textContent = 'Logging in…';

        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            const data = await res.json();

            if (!res.ok || data.status !== 'success') {
                loginStatus.textContent = data.message || 'Login failed';
                return;
            }

            // Expecting access_token, refresh_token, uuid in response
            accessToken = data.access_token;
            refreshToken = data.refresh_token;
            userUuid = data.uuid;

            localStorage.setItem(LS_ACCESS, accessToken);
            localStorage.setItem(LS_REFRESH, refreshToken || '');
            localStorage.setItem(LS_USER_UUID, userUuid);
            localStorage.setItem(LS_USER_EMAIL, email);

            loginStatus.textContent = 'Logged in';

            // Switch UI: hide login, show app
            if (loginForm) loginForm.classList.add('hidden');
            if (authInfo) authInfo.classList.remove('hidden');
            if (authUserEmail) authUserEmail.textContent = email;
            if (appLayout) appLayout.classList.remove('hidden');

            await loadSessions();
        } catch (err) {
            console.error(err);
            loginStatus.textContent = 'Error during login';
        }
    });
}

// --- Logout ----------------------------------------------------------

if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
        // Clear tokens
        accessToken = null;
        refreshToken = null;
        userUuid = null;

        localStorage.removeItem(LS_ACCESS);
        localStorage.removeItem(LS_REFRESH);
        localStorage.removeItem(LS_USER_UUID);
        localStorage.removeItem(LS_USER_EMAIL);

        // Reset UI
        if (authInfo) authInfo.classList.add('hidden');
        if (loginForm) loginForm.classList.remove('hidden');
        if (appLayout) appLayout.classList.add('hidden');

        // Clear chat + sessions
        const chatEl = document.getElementById('chat');
        if (chatEl) chatEl.innerHTML = '';
        const sessList = document.getElementById('session-list');
        if (sessList) sessList.innerHTML = '';

        loginStatus.textContent = 'Logged out';
    });
}

// --- Sessions sidebar ------------------------------------------------

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

// Ensure we have a session before logging the first turn
async function ensureSessionExists(firstUserText) {
    if (currentSessionUuid) return currentSessionUuid;
    return await createNewSession(firstUserText || '');
}

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

// "New chat" button
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

// Run initial restore on page load
restoreAuth();

// --- Turn logging (Django log_chat_turn) -------------------------------

async function logChatTurnToBackend(userText, assistantText, latencyMs) {
    if (!authToken) return;
    if (!userText || !assistantText) return;

    try {
        const res = await fetch(LOG_TURN_URL, {
            method: 'POST',
            headers: backendHeaders(),
            body: JSON.stringify({
                session_uuid: currentSessionUuid,   // may be null; backend will create if needed
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
            // Keep the sidebar in sync
            loadSessions().catch(console.error);
        }
    } catch (err) {
        console.error('Error logging chat turn', err);
    }
}
