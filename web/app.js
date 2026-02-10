/**
 * MIA – WebUI Client
 *
 * Conexión WebSocket bidireccional con el pipeline de MIA.
 * Recibe: status, subtítulos, métricas, logs, mouth, emotion.
 * Envía: comandos (mute, pause, toggle RAG/vision, chat text, TTS config).
 */

// ── Config ────────────────────────────────
const WS_URL = `ws://${location.hostname || "localhost"}:8765`;
const RECONNECT_DELAY = 3000;

// ── State ─────────────────────────────────
let ws = null;
let logFilter = "all";
let conversationActive = false;

const state = {
    muted: false,
    paused: false,
    rag: true,
    vision: false,
};

// ── DOM refs ──────────────────────────────
const $ = (sel) => document.querySelector(sel);
const statusBadge = $("#status-badge");
const wsIndicator = $("#ws-indicator");
const chatMessages = $("#chat-messages");
const chatInput = $("#chat-input");
const logOutput = $("#log-output");

// ── WebSocket ─────────────────────────────

function connect() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        wsIndicator.className = "ws-dot online";
        addLog("Conectado al pipeline", "info");
        updateBadge("listening");
    };

    ws.onclose = () => {
        wsIndicator.className = "ws-dot offline";
        updateBadge("desconectado");
        addLog("Desconectado — reconectando...", "error");
        setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = () => {
        addLog("Error de conexión WebSocket", "error");
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleMessage(data);
        } catch {
            addLog(`Mensaje no JSON: ${event.data}`, "debug");
        }
    };
}

function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
    }
}

// ── Message handler ───────────────────────

function handleMessage(data) {
    switch (data.type) {
        case "status":
            updateBadge(data.value);
            break;

        case "subtitle":
            addChatMessage(data.text, data.role || "assistant");
            break;

        case "mouth":
            // Could be used for animation
            break;

        case "emotion":
            addLog(`Emoción: ${data.value}`, "debug");
            break;

        case "metrics":
            updateMetrics(data);
            break;

        case "audio_level":
            updateAudioLevel(data.value);
            break;

        case "log":
            addLog(data.text, data.level || "info");
            break;

        case "config_state":
            syncState(data);
            break;

        // ── Conversation Turn Protocol ──

        case "control":
            handleControl(data.action);
            break;

        case "full-text":
            // Placeholder text (e.g. "Thinking...")
            addLog(`Texto: ${data.text}`, "debug");
            break;

        case "user-input-transcription":
            addChatMessage(data.text, "user");
            break;

        case "audio-response":
            // Audio chunk from TTS — decode and queue for playback
            if (data.audio) {
                queueAudioChunk(data.audio);
            }
            if (data.display_text) {
                addLog(`🔊 ${data.display_text}`, "debug");
            }
            break;

        case "backend-synth-complete":
            // Backend finished sending all audio — wait for playback queue to drain
            addLog("Backend synth complete, waiting for audio playback...", "debug");
            waitForAudioQueueDrain().then(() => {
                send({ type: "frontend-playback-complete" });
                addLog("Frontend playback complete", "debug");
            });
            break;

        case "force-new-message":
            // Frontend should start a fresh message bubble
            break;

        case "interrupt-signal":
            addLog("Conversación interrumpida", "info");
            conversationActive = false;
            clearAudioQueue();
            break;

        case "error":
            addLog(`Error: ${data.message}`, "error");
            break;

        default:
            addLog(`Tipo desconocido: ${data.type}`, "debug");
    }
}

function handleControl(action) {
    switch (action) {
        case "conversation-chain-start":
            conversationActive = true;
            addLog("Conversación iniciada", "debug");
            break;
        case "conversation-chain-end":
            conversationActive = false;
            addLog("Conversación finalizada", "debug");
            break;
    }
}

// ── UI Updates ────────────────────────────

function updateBadge(status) {
    statusBadge.textContent = status;
    statusBadge.className = "badge";
    if (status === "listening") statusBadge.classList.add("badge-listening");
    else if (status === "thinking") statusBadge.classList.add("badge-thinking");
    else if (status === "speaking") statusBadge.classList.add("badge-speaking");
    else statusBadge.classList.add("badge-offline");
}

function addChatMessage(text, role) {
    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateMetrics(data) {
    if (data.stt != null) $("#metric-stt").textContent = Math.round(data.stt);
    if (data.rag != null) $("#metric-rag").textContent = Math.round(data.rag);
    if (data.llm != null) $("#metric-llm").textContent = Math.round(data.llm);
    if (data.tts != null) $("#metric-tts").textContent = Math.round(data.tts);
    if (data.total != null) $("#metric-total").textContent = Math.round(data.total);
}

function updateAudioLevel(value) {
    const pct = Math.min(100, Math.max(0, value * 100));
    $("#audio-level").style.width = pct + "%";
}

function addLog(text, level = "info") {
    const line = document.createElement("div");
    line.className = `log-line ${level}`;
    const time = new Date().toLocaleTimeString("es-CL", { hour12: false });
    line.textContent = `[${time}] ${text}`;
    line.dataset.level = level;

    logOutput.appendChild(line);
    logOutput.scrollTop = logOutput.scrollHeight;

    // Keep max 200 lines
    while (logOutput.children.length > 200) {
        logOutput.removeChild(logOutput.firstChild);
    }

    applyLogFilter();
}

function applyLogFilter() {
    for (const line of logOutput.children) {
        if (logFilter === "all" || line.dataset.level === logFilter) {
            line.style.display = "";
        } else {
            line.style.display = "none";
        }
    }
}

function syncState(data) {
    if (data.muted != null) {
        state.muted = data.muted;
        updateToggle("btn-mute", !state.muted, "mic-state", state.muted ? "mute" : "activo");
    }
    if (data.paused != null) {
        state.paused = data.paused;
        updateToggle("btn-pause", !state.paused, "pipeline-state", state.paused ? "pausa" : "activo");
    }
    if (data.rag != null) {
        state.rag = data.rag;
        updateToggle("btn-rag", state.rag, "rag-state", state.rag ? "activo" : "inactivo");
    }
    if (data.vision != null) {
        state.vision = data.vision;
        updateToggle("btn-vision", state.vision, "vision-state", state.vision ? "activo" : "inactivo");
    }
}

function updateToggle(btnId, active, stateId, label) {
    const btn = $(`#${btnId}`);
    const st = $(`#${stateId}`);
    btn.dataset.active = active ? "true" : "false";
    st.textContent = label;
}

// ── Event Listeners ───────────────────────

// Control buttons
$("#btn-mute").addEventListener("click", () => {
    state.muted = !state.muted;
    updateToggle("btn-mute", !state.muted, "mic-state", state.muted ? "mute" : "activo");
    send({ type: "command", action: "mute", value: state.muted });
});

$("#btn-pause").addEventListener("click", () => {
    state.paused = !state.paused;
    updateToggle("btn-pause", !state.paused, "pipeline-state", state.paused ? "pausa" : "activo");
    send({ type: "command", action: "pause", value: state.paused });
});

$("#btn-rag").addEventListener("click", () => {
    state.rag = !state.rag;
    updateToggle("btn-rag", state.rag, "rag-state", state.rag ? "activo" : "inactivo");
    send({ type: "command", action: "toggle_rag", value: state.rag });
});

$("#btn-vision").addEventListener("click", () => {
    state.vision = !state.vision;
    updateToggle("btn-vision", state.vision, "vision-state", state.vision ? "activo" : "inactivo");
    send({ type: "command", action: "toggle_vision", value: state.vision });
});

// Chat
function sendChat() {
    const text = chatInput.value.trim();
    if (!text) return;
    addChatMessage(text, "user");
    send({ type: "text-input", text });
    chatInput.value = "";
}

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
});

$("#btn-send").addEventListener("click", sendChat);
$("#btn-clear-chat").addEventListener("click", () => {
    chatMessages.innerHTML = "";
});

// TTS sliders
$("#tts-rate").addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    const label = (val >= 0 ? "+" : "") + val + "%";
    $("#tts-rate-val").textContent = label;
    send({ type: "command", action: "set_tts_rate", value: label });
});

$("#tts-pitch").addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    const label = (val >= 0 ? "+" : "") + val + "Hz";
    $("#tts-pitch-val").textContent = label;
    send({ type: "command", action: "set_tts_pitch", value: label });
});

// Log filters
document.querySelectorAll(".log-filter").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".log-filter").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        logFilter = btn.dataset.level;
        applyLogFilter();
    });
});

// ── Audio Playback System ─────────────────
let audioCtx = null;
const audioQueue = [];
let isPlayingAudio = false;
let drainResolvers = [];

function getAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
        audioCtx.resume();
    }
    return audioCtx;
}

function queueAudioChunk(base64Audio) {
    audioQueue.push(base64Audio);
    if (!isPlayingAudio) {
        playNextInQueue();
    }
}

async function playNextInQueue() {
    if (audioQueue.length === 0) {
        isPlayingAudio = false;
        // Resolve all drain waiters
        const resolvers = drainResolvers.splice(0);
        resolvers.forEach(r => r());
        return;
    }

    isPlayingAudio = true;
    const base64Audio = audioQueue.shift();

    try {
        const ctx = getAudioContext();
        const binary = atob(base64Audio);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }

        const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        source.onended = () => playNextInQueue();
        source.start(0);
    } catch (err) {
        addLog(`Error playing audio: ${err.message}`, "error");
        playNextInQueue(); // Skip broken chunk
    }
}

function waitForAudioQueueDrain() {
    if (!isPlayingAudio && audioQueue.length === 0) {
        return Promise.resolve();
    }
    return new Promise(resolve => {
        drainResolvers.push(resolve);
    });
}

function clearAudioQueue() {
    audioQueue.length = 0;
    isPlayingAudio = false;
    drainResolvers.splice(0).forEach(r => r());
}

// ── Init ──────────────────────────────────
addLog("Iniciando WebUI...", "info");
connect();
