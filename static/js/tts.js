// --- TTS controls (browser speech) ---
const speakToggle = document.getElementById('speak-toggle');
const voiceSelect = document.getElementById('voice-select');
const playLastBtn = document.getElementById('play-last');
const audioEl     = document.getElementById('live-audio');

const synth = window.speechSynthesis;
const voiceProfiles = [
    { id: 'en-gb-neutral', label: 'English (UK) — Neutral', lang: 'en-GB', prefer: /UK|English/i, rate: 1.0, pitch: 1.0 },
    { id: 'en-gb-male',    label: 'English (UK) — Male',    lang: 'en-GB', prefer: /Male|Daniel|UK English Male/i, rate: 1.0, pitch: 1.0 },
    { id: 'en-gb-female',  label: 'English (UK) — Female',  lang: 'en-GB', prefer: /Serena|Martha|UK English Female/i, rate: 1.0, pitch: 1.05 },
];

let currentVoice = null;
let currentProfile = voiceProfiles[0];
let sentenceBuffer = '';
let lastAnswerText = '';

// Restore user prefs
voiceSelect.value = localStorage.getItem('voiceProfile') || voiceSelect.value;
speakToggle.checked = localStorage.getItem('speakEnabled') === '1';

voiceSelect.addEventListener('change', (e) => {
    localStorage.setItem('voiceProfile', e.target.value);
    selectVoice();
});
speakToggle.addEventListener('change', () => {
    localStorage.setItem('speakEnabled', speakToggle.checked ? '1' : '0');
    if (!speakToggle.checked) synth.cancel(); // stop any ongoing speech
});

function selectVoice() {
    const voices = synth.getVoices();
    const profileId = voiceSelect.value || 'en-gb-neutral';
    currentProfile = voiceProfiles.find(p => p.id === profileId) || voiceProfiles[0];

    // Find best matching voice for this profile
    let v = voices.find(v => v.lang === currentProfile.lang && currentProfile.prefer.test(v.name))
          || voices.find(v => v.lang === currentProfile.lang)
          || voices[0];
    currentVoice = v || null;
}

// Some browsers load voices asynchronously
synth.onvoiceschanged = () => { selectVoice(); };
selectVoice();

function configureUtterance(u) {
    if (currentVoice) u.voice = currentVoice;
    u.rate  = currentProfile.rate;
    u.pitch = currentProfile.pitch;
}

// Speak a single sentence or block
function speakText(text) {
    if (!speakToggle.checked) return;
    const s = text.trim();
    if (!s) return;
    const u = new SpeechSynthesisUtterance(s);
    configureUtterance(u);
    synth.speak(u);
}

// Called when a new assistant reply starts
function onAssistantStreamStart() {
    lastAnswerText = '';
    sentenceBuffer = '';
    if (speakToggle.checked) synth.cancel(); // reset any previous speech
}

// Called for each streamed token chunk
function onAssistantToken(chunk) {
    lastAnswerText += chunk;

    if (!speakToggle.checked) return;

    sentenceBuffer += chunk;
    const parts = sentenceBuffer.split(/([.!?]+[\s]+)/);
    for (let i = 0; i + 1 < parts.length; i += 2) {
        const sentence = (parts[i] + parts[i+1]).trim();
        if (sentence) speakText(sentence);
    }
    sentenceBuffer = (parts.length % 2 === 1) ? parts[parts.length - 1] : '';
}

// Called when the assistant finishes (done or error)
function onAssistantStreamEnd() {
    if (speakToggle.checked && sentenceBuffer.trim()) {
        speakText(sentenceBuffer);
    }
    sentenceBuffer = '';
}

// "Listen" button: read the whole last answer once more
playLastBtn.addEventListener('click', () => {
    if (!lastAnswerText.trim()) return;
    synth.cancel();
    speakText(lastAnswerText);
});

onAssistantStreamStart();

