/**
 * MIA WebUI — Real-time control panel
 * WebSocket client + UI controller
 */

(() => {
    "use strict";

    // ── Elements ──
    const $ = (sel) => document.querySelector(sel);
    const chatMessages = $("#chatMessages");
    const statusBadge = $("#statusBadge");
    const statusText = $(".status-text");

    const togglePause = $("#togglePause");
    const toggleProactive = $("#toggleProactive");
    const toggleText = $("#toggleText");
    const toggleRag = $("#toggleRag");
    const sliderIdle = $("#sliderIdle");
    const sliderSilence = $("#sliderSilence");
    const sliderEnergy = $("#sliderEnergy");
    const idleValue = $("#idleValue");
    const silenceValue = $("#silenceValue");
    const energyValue = $("#energyValue");
    const btnClearHistory = $("#btnClearHistory");
    const btnForceSpeak = $("#btnForceSpeak");
    const btnReconnect = $("#btnReconnect");
    const audioMeterFill = $("#audioMeterFill");
    const idleGroup = $("#idleGroup");

    // Terminal
    const tabChat = $("#tabChat");
    const tabTerminal = $("#tabTerminal");
    const tabMemory = $("#tabMemory");
    const terminalPanel = $("#terminalPanel");
    const terminalOutput = $("#terminalOutput");
    const memoryPanel = $("#memoryPanel");
    const btnRefreshMemory = $("#btnRefreshMemory");
    const MAX_LOG_LINES = 500;
    let memoryInitialized = false;

    // Stats
    const statSpeakers = $("#statSpeakers .stat-value");
    const statHistory = $("#statHistory .stat-value");
    const statRag = $("#statRag .stat-value");

    // ── WebSocket ──
    let ws = null;
    let reconnectTimer = null;
    let hasMessages = false;

    function connect() {
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${proto}//${location.host}/ws`);

        ws.onopen = () => {
            setStatus("listening", "Listening");
            if (reconnectTimer) {
                clearInterval(reconnectTimer);
                reconnectTimer = null;
            }
        };

        ws.onclose = () => {
            setStatus("disconnected", "Disconnected");
            if (!reconnectTimer) {
                reconnectTimer = setInterval(connect, 3000);
            }
        };

        ws.onerror = () => ws.close();

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleEvent(msg);
            } catch (e) {
                console.error("Invalid message:", e);
            }
        };
    }

    function sendCommand(command, value) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ command, value }));
        }
    }

    // ── Event handlers ──

    function handleEvent(msg) {
        switch (msg.type) {
            case "state":
                applyState(msg.data);
                break;
            case "chat_message":
                addChatMessage(msg.data);
                break;
            case "state_change":
                applyStateChange(msg.data);
                break;
            case "stats":
                updateStats(msg.data);
                break;
            case "audio_level":
                updateAudioMeter(msg.data);
                break;
            case "log":
                appendLogLine(msg.data);
                break;
        }
    }

    function applyState(state) {
        if (!state) return;

        // Toggles
        if (state.paused !== undefined) togglePause.checked = state.paused;
        if (state.proactive !== undefined) toggleProactive.checked = state.proactive;
        if (state.text_responses !== undefined) toggleText.checked = state.text_responses;
        if (state.rag_enabled !== undefined) toggleRag.checked = state.rag_enabled;

        // Sliders
        if (state.idle_seconds !== undefined) {
            sliderIdle.value = state.idle_seconds;
            idleValue.textContent = state.idle_seconds + "s";
        }
        if (state.silence_ms !== undefined) {
            sliderSilence.value = state.silence_ms;
            silenceValue.textContent = state.silence_ms + "ms";
        }
        if (state.min_energy_rms !== undefined) {
            sliderEnergy.value = state.min_energy_rms;
            energyValue.textContent = parseFloat(state.min_energy_rms).toFixed(3);
        }

        // Status
        if (state.bot_state) {
            setStatus(state.bot_state, formatState(state.bot_state));
        }

        // Idle group visibility
        idleGroup.style.opacity = toggleProactive.checked ? "1" : "0.4";

        // Stats
        if (state.stats) updateStats(state.stats);

        // Chat history
        if (state.chat_history && state.chat_history.length > 0) {
            state.chat_history.forEach(msg => addChatMessage(msg));
        }
    }

    function applyStateChange(data) {
        if (!data) return;
        if (data.key === "paused") togglePause.checked = data.value;
        if (data.key === "proactive") {
            toggleProactive.checked = data.value;
            idleGroup.style.opacity = data.value ? "1" : "0.4";
        }
        if (data.key === "text_responses") toggleText.checked = data.value;
        if (data.key === "rag_enabled") toggleRag.checked = data.value;
        if (data.key === "idle_seconds") {
            sliderIdle.value = data.value;
            idleValue.textContent = data.value + "s";
        }
        if (data.key === "silence_ms") {
            sliderSilence.value = data.value;
            silenceValue.textContent = data.value + "ms";
        }
        if (data.key === "min_energy_rms") {
            sliderEnergy.value = data.value;
            energyValue.textContent = parseFloat(data.value).toFixed(3);
        }
        if (data.key === "bot_state") {
            setStatus(data.value, formatState(data.value));
        }
    }

    // ── Chat rendering ──

    function addChatMessage(data) {
        if (!hasMessages) {
            chatMessages.innerHTML = "";
            hasMessages = true;
        }

        const div = document.createElement("div");
        const isMia = data.speaker === "MIA";
        const isSystem = data.speaker === "SYSTEM";

        div.className = `chat-msg ${isMia ? "msg-mia" : isSystem ? "msg-system" : "msg-user"}`;

        const time = data.time || new Date().toLocaleTimeString("es-CL", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        });

        div.innerHTML = `<span class="msg-time">${time}</span>`
            + `<span class="msg-speaker">${escapeHtml(data.speaker)}</span>`
            + `<span class="msg-text">${escapeHtml(data.text)}</span>`;

        chatMessages.appendChild(div);

        // Auto-scroll
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Status ──

    function setStatus(state, label) {
        statusBadge.dataset.state = state;
        statusText.textContent = label;
    }

    function formatState(state) {
        const map = {
            listening: "Listening",
            paused: "Paused",
            processing: "Processing",
            speaking: "Speaking",
            proactive: "Proactive",
            disconnected: "Disconnected",
        };
        return map[state] || state;
    }

    // ── Stats ──

    function updateStats(data) {
        if (!data) return;
        if (data.speakers !== undefined) statSpeakers.textContent = data.speakers;
        if (data.history !== undefined) statHistory.textContent = data.history;
        if (data.rag_docs !== undefined) statRag.textContent = data.rag_docs;
    }

    // ── Control bindings ──

    togglePause.addEventListener("change", () => {
        sendCommand("toggle_pause", togglePause.checked);
    });

    toggleProactive.addEventListener("change", () => {
        sendCommand("toggle_proactive", toggleProactive.checked);
        idleGroup.style.opacity = toggleProactive.checked ? "1" : "0.4";
    });

    toggleText.addEventListener("change", () => {
        sendCommand("toggle_text_responses", toggleText.checked);
    });

    toggleRag.addEventListener("change", () => {
        sendCommand("toggle_rag", toggleRag.checked);
    });

    sliderIdle.addEventListener("input", () => {
        idleValue.textContent = sliderIdle.value + "s";
    });
    sliderIdle.addEventListener("change", () => {
        sendCommand("set_idle_seconds", parseInt(sliderIdle.value));
    });

    sliderSilence.addEventListener("input", () => {
        silenceValue.textContent = sliderSilence.value + "ms";
    });
    sliderSilence.addEventListener("change", () => {
        sendCommand("set_silence_ms", parseInt(sliderSilence.value));
    });

    sliderEnergy.addEventListener("input", () => {
        energyValue.textContent = parseFloat(sliderEnergy.value).toFixed(3);
    });
    sliderEnergy.addEventListener("change", () => {
        sendCommand("set_min_energy", parseFloat(sliderEnergy.value));
    });

    btnClearHistory.addEventListener("click", () => {
        if (confirm("¿Limpiar todo el historial de chat?")) {
            sendCommand("clear_history", true);
            chatMessages.innerHTML = "";
            hasMessages = false;
            chatMessages.innerHTML = `
                <div class="chat-empty">
                    <span class="chat-empty-icon">🎙️</span>
                    <p>Historial limpiado</p>
                </div>`;
        }
    });

    btnForceSpeak.addEventListener("click", () => {
        sendCommand("force_speak", true);
    });

    btnReconnect.addEventListener("click", () => {
        sendCommand("reconnect_voice", true);
    });

    // ── Audio meter ──

    let meterDecayTimer = null;

    function updateAudioMeter(data) {
        if (!data || data.rms === undefined) return;
        const rms = data.rms;
        const pct = Math.min(100, rms * 1000);
        audioMeterFill.style.width = pct + "%";

        audioMeterFill.classList.remove("level-medium", "level-high");
        if (rms > 0.05) {
            audioMeterFill.classList.add("level-high");
        } else if (rms > 0.02) {
            audioMeterFill.classList.add("level-medium");
        }

        clearTimeout(meterDecayTimer);
        meterDecayTimer = setTimeout(() => {
            audioMeterFill.style.width = "0%";
            audioMeterFill.classList.remove("level-medium", "level-high");
        }, 500);
    }

    // ── Tab switching ──

    function switchTab(tab) {
        tabChat.classList.remove("active");
        tabTerminal.classList.remove("active");
        tabMemory.classList.remove("active");
        chatMessages.style.display = "none";
        terminalPanel.style.display = "none";
        memoryPanel.style.display = "none";

        if (tab === "chat") {
            tabChat.classList.add("active");
            chatMessages.style.display = "";
        } else if (tab === "terminal") {
            tabTerminal.classList.add("active");
            terminalPanel.style.display = "";
            terminalPanel.scrollTop = terminalPanel.scrollHeight;
        } else if (tab === "memory") {
            tabMemory.classList.add("active");
            memoryPanel.style.display = "";
            if (!memoryInitialized && typeof Memory3D !== "undefined") {
                Memory3D.init("memoryCanvas");
                Memory3D.start();
                memoryInitialized = true;
            } else if (memoryInitialized) {
                Memory3D.onResize();
            }
        }
    }

    tabChat.addEventListener("click", () => switchTab("chat"));
    tabTerminal.addEventListener("click", () => switchTab("terminal"));
    tabMemory.addEventListener("click", () => switchTab("memory"));

    if (btnRefreshMemory) {
        btnRefreshMemory.addEventListener("click", () => {
            if (memoryInitialized) Memory3D.start();
        });
    }

    // ── Terminal log ──

    function appendLogLine(data) {
        if (!data || !data.text) return;
        const line = document.createElement("div");
        line.className = `terminal-line log-${data.level || "INFO"}`;
        line.textContent = data.text;
        terminalOutput.appendChild(line);

        // Trim old lines
        while (terminalOutput.children.length > MAX_LOG_LINES) {
            terminalOutput.removeChild(terminalOutput.firstChild);
        }

        // Auto-scroll if visible
        if (terminalPanel.style.display !== "none") {
            terminalPanel.scrollTop = terminalPanel.scrollHeight;
        }
    }

    // ── Init ──
    connect();
})();
