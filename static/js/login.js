// static/js/login.js

// Storage keys for later use (chat page)
const LS_TOKEN = 'fa_access_token';
const LS_EMAIL = 'fa_email';

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("login-form");
    const statusEl = document.getElementById("login-status");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        statusEl.textContent = "Logging in...";

        const email = document.getElementById("login-email").value.trim();
        const password = document.getElementById("login-password").value;

        try {
            const resp = await fetch("/api/login", {
                // <-- always talk to our own FastAPI app
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ email, password }),
            });

            if (!resp.ok) {
                const text = await resp.text();
                console.error("Login error:", resp.status, text);
                statusEl.textContent = "Login failed";
                return;
            }

            const data = await resp.json();
            console.log("Login response:", data);

            const token = data.access_token || data.token;
            if (!token) {
                console.error("No token in login response", data);
                statusEl.textContent = "Login failed (no token)";
                return;
            }

            // Store token + email so chat.js can use them
            localStorage.setItem(LS_TOKEN, token);
            localStorage.setItem(LS_EMAIL, data.email || email);

            // Go to chat page
            window.location.href = "/chat";
        } catch (err) {
            console.error("Login exception:", err);
            statusEl.textContent = "Network error, please try again.";
        }
    });
});
