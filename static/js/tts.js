// static/js/tts.js

/**
 * Called by chat.js when a new assistant reply starts streaming.
 * Currently just resets any ongoing speech, if the browser supports it.
 */
window.onAssistantStreamStart = function () {
    if (window.speechSynthesis) {
        // Stop anything that might still be speaking
        window.speechSynthesis.cancel();
    }
};

/**
 * Called by chat.js for each streamed token.
 * We are not doing streaming TTS any more, so this is a no-op.
 */
window.onAssistantToken = function (_chunk) {
    // Intentionally empty: per-message "🔊" playback lives in chat.js
};

/**
 * Called by chat.js when the assistant finishes (done or error).
 * Make sure any partial speech is stopped.
 */
window.onAssistantStreamEnd = function () {
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
};
