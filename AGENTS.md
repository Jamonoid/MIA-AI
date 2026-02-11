# AGENTS.md

Guía para agentes (IA o humanos) que contribuyen al proyecto **MIA-AI**: un asistente VTuber con baja latencia, memoria RAG, animación de avatar en tiempo real, e integración con Discord y VTube Studio.

---

## Objetivo del proyecto

Pipeline de voz-a-avatar con el menor retardo percibido posible:

```
Mic/Discord → VAD → STT → RAG → LLM (stream) → TTS (chunked) → Audio + Lipsync → Avatar
```

**Interfaces de entrada:**
- Micrófono local (VAD → STT)
- WebUI (chat escrito)
- Discord voice channel (multi-speaker)

**Interfaces de salida:**
- Audio local (speakers) + WebUI (Web Audio API)
- VTube Studio (WebSocket Plugin API: expresiones + lipsync + blink)
- Discord voice channel (TTS → FFmpeg → voice)

---

## Principios no negociables

1. **Nada pesado en el loop principal**: STT/LLM/TTS en hilos o `run_in_executor`
2. **Streaming end-to-end**: no esperar texto completo para hablar
3. **Configuración en YAML**: una sola fuente de verdad (`config.yaml`)
4. **Módulos pequeños y desacoplados**: una responsabilidad por archivo
5. **Tests obligatorios** para cambios en módulos core *(33 tests actualmente)*

**Metas de latencia:**

| Etapa | Objetivo |
|-------|----------|
| Primer token LLM | < 300 ms |
| Primera salida de voz | < 900 ms |
| Lipsync | 50–100 Hz |
| RAG retrieval | < 50 ms |

---

## Estructura del repositorio

```
MIA-AI/
  config.yaml                    Configuración central (gitignored)
  config.example.yaml            Ejemplo de referencia
  pyproject.toml                 Dependencias y build
  .env                           Secrets (Discord token, etc.) — gitignored
  .vts_token                     Token de autenticación VTube Studio — gitignored

  src/mia/
    main.py                      Punto de entrada
    config.py                    YAML → dataclasses tipados (12 secciones)
    pipeline.py                  Orquestador asíncrono principal

    # ── Audio ──
    audio_io.py                  AudioCapture (mic) + AudioPlayer (cola)
    vad.py                       EnergyVAD: RMS + pre-roll buffer

    # ── Modelos ──
    stt_whispercpp.py            STT (faster-whisper, GPU)
    llm_llamacpp.py              LLM local (llama-cpp-python)
    llm_lmstudio.py              LLM vía LM Studio (API OpenAI)
    llm_openrouter.py            LLM vía OpenRouter (nube)
    tts_edge.py                  TTS online (Microsoft Edge) + chunk_text

    # ── Memoria ──
    rag_memory.py                ChromaDB + sentence-transformers

    # ── Avatar ──
    lipsync.py                   RMS → mouth_open (0..1)
    vtube_studio.py              WebSocket Plugin API hacia VTube Studio
    vtube_osc.py                 (legacy) OSC hacia VTube Studio

    # ── Comunicación ──
    ws_server.py                 WebSocket + HTTP server para WebUI
    discord_bot.py               Bot Discord con voice (py-cord)
    discord_sink.py              ContinuousVoiceSink para captura de audio Discord

    # ── Conversaciones ──
    conversations/
      __init__.py                Exports del paquete
      types.py                   Type aliases, dataclasses, WebSocketSend
      message_handler.py         Sincronización frontend↔backend (asyncio.Event)
      tts_manager.py             TTS paralelo + entrega ordenada por seq number
      conversation_handler.py    Entry point: triggers → asyncio.Task
      single_conversation.py     Flujo de 14 pasos de un turno + lipsync
      conversation_utils.py      Helpers (señales, cleanup, finalización)

  prompts/                       Prompt modular (archivos .md concatenados) — gitignored
  web/                           WebUI (HTML/CSS/JS vanilla)
  data/chroma_db/                Vector store persistente (auto-generado)
  tests/                         Tests unitarios (33 tests)
```

---

## Configuración

- Toda la configuración va en `config.yaml` (tipada en `config.py`).
- **Nunca hardcodear** rutas, modelos ni parámetros.
- Si necesitas un parámetro nuevo:
  1. Agregar a `config.yaml` y `config.example.yaml`
  2. Agregar dataclass en `config.py` (o campo nuevo en dataclass existente)
  3. Documentar en AGENTS.md
  4. Añadir test si aplica

### Secciones de config.yaml

| Sección | Dataclass | Descripción |
|---------|-----------|-------------|
| `prompt` | `PromptConfig` | System prompt fallback + carpeta modular |
| `models.llm` | `LLMConfig` | Backend, modelo, parámetros de generación |
| `models.stt` | `STTConfig` | Modelo, idioma, device |
| `models.tts` | `TTSConfig` | Backend, voz, chunk size, Edge params |
| `audio` | `AudioConfig` | Sample rates, chunk_ms |
| `vad` | `VADConfig` | Energy threshold, silence/speech durations |
| `rag` | `RAGConfig` | Embedding model, persist dir, top_k |
| `lipsync` | `LipsyncConfig` | Smoothing, RMS range |
| `vtube_studio` | `VTubeStudioConfig` | WebSocket URL, token, expresiones, parámetros |
| `websocket` | `WebSocketConfig` | Host, port, enabled, WebUI dir |
| `performance` | `PerformanceConfig` | VAD sensitivity, lipsync smoothing |
| `discord` | `DiscordConfig` | Voice receive, canales, ajustes de VAD |

---

## Setup y ejecución

```bash
# 1. Entorno + dependencias base
uv venv
uv pip install -e ".[dev]"

# 2. CUDA PyTorch (obligatorio para GPU)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Dependencias ML
uv pip install faster-whisper
uv pip install TTS
# Re-instalar CUDA torch (TTS sobreescribe con CPU):
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. WebSocket para VTube Studio
uv pip install websockets

# 5. Discord (opcional)
uv pip install "py-cord[voice]>=2.6.0" "python-dotenv>=1.0.0"

# 6. Ejecutar
.venv/Scripts/python -m mia.main
```

> **Nunca usar `uv run mia`** — re-resuelve dependencias y puede sobreescribir CUDA torch.

---

## Arquitectura del pipeline

### Flujo principal (`pipeline.py`)

```python
async def run(self):
    while self._running:
        await self._listen_and_respond()
```

`_listen_and_respond()`:
1. Chequea si hay conversación activa (`is_busy`) → si sí, `sleep(0.1)` y retorna
2. Captura audio del mic → `_capture_speech()` con VAD + pre-roll buffer
3. Delega a `ConversationHandler.handle_trigger()` como `asyncio.Task`
4. Si no hay WebSocket → fallback legacy (STT → LLM → TTS secuencial)

### Sistema de turnos de conversación

El paquete `conversations/` implementa control de turnos basado en `asyncio.Task`:

```
WebSocket/Discord → conversation_handler → asyncio.Task
                                             ↓
                                    single_conversation (14 pasos)
                                             ↓
                                    LLM stream → TTSTaskManager (paralelo)
                                             ↓
                                    Audio ordenado → WebSocket/Discord
                                             ↓
                                    Lipsync → VTube Studio (send_mouth)
                                             ↓
                                    wait "frontend-playback-complete"
                                             ↓
                                    Turno completo
```

#### Componentes

| Módulo | Responsabilidad |
|--------|----------------|
| `types.py` | `ConversationMetadata`, `GroupConversationState`, `WebSocketSend` |
| `message_handler.py` | Sincronización request↔response con `asyncio.Event` |
| `tts_manager.py` | TTS paralelo: sequence numbers + buffer de reordenamiento + lipsync callback |
| `conversation_handler.py` | Recibe triggers, concurrency guard (1 turno/cliente), despacha tasks |
| `single_conversation.py` | Flujo completo: STT → RAG → LLM → TTS → lipsync → sync → historial |
| `conversation_utils.py` | Señales de inicio/fin, cleanup, finalización de turno |

#### Reglas de implementación

- **Un turno por cliente a la vez**: `is_busy()` antes de crear nueva task
- **TTS paralelo, entrega ordenada**: cada chunk recibe sequence number; sender loop envía en orden
- **Lipsync en callback**: `on_audio_ready` callback en TTSTaskManager procesa sub-chunks de 20ms con timing preciso
- **Sincronización**: `backend-synth-complete` → bloquea hasta `frontend-playback-complete`
- **Cleanup siempre**: `try/except CancelledError/finally` en todo flujo
- **Interrupciones**: cancelar task → guardar respuesta parcial → `[Interrupted by user]`

#### Protocolo WebSocket

**Backend → Frontend:**

| Tipo | Propósito |
|------|-----------|
| `control: conversation-chain-start` | AI procesando |
| `audio-response` | Chunk de audio WAV base64 + display text |
| `backend-synth-complete` | No hay más audio |
| `force-new-message` | Nueva burbuja de chat |
| `control: conversation-chain-end` | Turno completo |
| `interrupt-signal` | Conversación interrumpida |
| `user-input-transcription` | Texto transcrito del usuario |

**Frontend → Backend:**

| Tipo | Propósito |
|------|-----------|
| `text-input` | Chat escrito |
| `mic-audio-end` | Usuario terminó de hablar |
| `frontend-playback-complete` | Audio reproducido |
| `interrupt` | Usuario quiere interrumpir |

---

## VTube Studio (WebSocket Plugin API)

MIA se conecta a VTube Studio mediante su WebSocket Plugin API (`ws://localhost:8001`).

### Funcionalidades

| Feature | Implementación |
|---------|---------------|
| **Autenticación** | Token persistido en `.vts_token`, solicitud automática en primer uso |
| **Expresiones** | El LLM genera tags `[emotion]` → se parsean → `ExpressionActivationRequest` |
| **Lipsync** | Inyección de `MouthOpen` en sub-chunks de 20ms con timing preciso |
| **Parpadeo** | Loop automático con `EyeOpenLeft`/`EyeOpenRight` cada ~3.5s |

### Configuración (`vtube_studio` en config.yaml)

```yaml
vtube_studio:
  enabled: true
  ws_url: "ws://localhost:8001"
  token_file: ".vts_token"
  mouth_param: "MouthOpen"           # Nombre INPUT del tracking param
  eye_l_param: "EyeOpenLeft"         # Nombre INPUT del tracking param
  eye_r_param: "EyeOpenRight"        # Nombre INPUT del tracking param
  expressions:                       # emotion → nombre de archivo de expresión
    neutral: "00_IdleFace.exp3.json"
    happy: "01_HappyFace.exp3.json"
    # ... etc
```

### Nombres de parámetros

**Importante:** `InjectParameterDataRequest` usa nombres de **input/tracking** (ej: `MouthOpen`), NO nombres de **output/Live2D** (ej: `ParamMouthOpenY`). Se pueden consultar en VTube Studio → Parameter Settings.

### Concurrencia

- `asyncio.Lock` serializa todas las llamadas WebSocket (evita que blink loop robe respuestas de expression calls)
- `send_mouth()` es fire-and-forget para no bloquear el flujo de lipsync

---

## Expresiones faciales (LLM → VTS)

El LLM genera tags de emoción al inicio de cada respuesta (ej: `[happy] ¡Hola!`).

**Flujo:**
1. `expressions.md` instruye al LLM a prefixar con `[emotion]`
2. `_parse_emotion()` extrae el tag y limpia el texto
3. `vts.set_expression(emotion)` activa la expresión en VTube Studio
4. El texto limpio (sin tag) se envía a TTS

**Emociones disponibles:** neutral, happy, sad, angry, surprised, scared, ashamed, pout, cry, super_happy

---

## Discord

### Arquitectura

- **Bot**: `discord_bot.py` — maneja conexión, comandos, voice channels
- **Audio capture**: `discord_sink.py` — `ContinuousVoiceSink` captura audio por speaker
- **Voice pipeline**: escucha → silence detection → STT → LLM → TTS → FFmpeg → play

### Configuración

```bash
# Instalar
uv pip install "py-cord[voice]>=2.6.0" "python-dotenv>=1.0.0"
# ffmpeg debe estar instalado en el sistema
```

`.env`:
```
DISCORD_BOT_TOKEN=tu_token_aquí
```

`config.yaml`:
```yaml
discord:
  enabled: true
  voice_receive: true
```

---

## Reglas por componente

### Audio / VAD
- Chunks pequeños (20 ms)
- Pre-roll buffer de 5 chunks (~100 ms) para no perder inicio de palabras
- VAD configurable: `energy_threshold`, `silence_duration_ms`, `min_speech_duration_ms`

### STT
- Forzar idioma (`es`) para evitar auto-detección
- `run_in_executor` para no bloquear el event loop

### LLM
- Streaming obligatorio
- Contexto moderado (2048) para rendimiento
- RAG context se inyecta antes del mensaje del usuario

### RAG Memory
- ChromaDB local, persistente en `./data/chroma_db/`
- Embeddings: `all-MiniLM-L6-v2` (~80 MB)
- Ingesta al finalizar cada turno: pares `(user_msg, assistant_msg)`
- Retrieval: top-K por similitud coseno
- Si `rag.enabled: false`, no cargar ChromaDB ni embeddings

### TTS
- **Edge TTS**: único backend activo. Online (Microsoft), sin GPU
- Parámetros: `edge_voice`, `edge_rate`, `edge_pitch`
- `chunk_text()` divide texto largo en chunks de 120–160 chars antes de sintetizar
- Expone `.synthesize(text) → np.ndarray`
- Para explorar voces: `edge-tts --list-voices`

### Lipsync
- RMS con suavizado exponencial
- Actualización 50–100 Hz sin bloquear el hilo principal
- Timing preciso basado en reloj (no sleep acumulativo) para mantener sync

### Prompt modular
- Carpeta `prompts/` con archivos `.md` concatenados alfabéticamente
- Editar personalidad sin tocar código
- `prompt.system` como fallback si la carpeta no existe o está vacía
- **Gitignored** — contiene personalidad del personaje (privada)

### WebUI
- Vanilla HTML/CSS/JS, sin build step
- Servido desde `web/` por `ws_server.py` (aiohttp)
- Comunicación bidireccional por WebSocket

---

## Tests

```bash
.venv/Scripts/python -m pytest -v
```

33 tests:
- `test_config.py` — carga y validación YAML
- `test_prompt.py` — prompt modular + fallback
- `test_rag.py` — ingesta, retrieval, score threshold
- `test_message_handler.py` — sincronización (7 tests)
- `test_tts_manager.py` — TTS paralelo + ordenamiento (5 tests)

Agregar tests al modificar módulos core.

---

## Gestión de dependencias

### Tabla de restricciones

| Paquete | Restricción | Motivo |
|---------|-------------|--------|
| numpy | Verificar tras instalar | Binary incompatibility con extensiones C |
| py-cord | >=2.6.0 | No coexiste con discord.py |
| faster-whisper | Manual install | CTranslate2, no depende de PyTorch |
| websockets | Requerido para VTube Studio | WebSocket Plugin API |

### ⚠️ Problema frecuente: numpy binary incompatibility

Al instalar paquetes nuevos, `uv` puede actualizar numpy a una versión más nueva que las extensiones C compiladas (chromadb/hnswlib, onnxruntime, etc.). Esto causa:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

**Solución**: reinstalar los paquetes que dependen de numpy contra la nueva versión:

```bash
uv pip install --force-reinstall chromadb chroma-hnswlib
```

### Orden de instalación seguro

1. **Crear venv**: `python -m venv .venv`
2. **Instalar el proyecto**: `uv pip install -e .`
3. **Instalar faster-whisper**: `uv pip install faster-whisper`
4. **Instalar websockets**: `uv pip install websockets`
5. **(Opcional) llama-cpp-python**: `uv pip install llama-cpp-python`
6. **(Opcional) Discord**: `uv pip install "py-cord[voice]>=2.6.0" python-dotenv`
7. **Verificar**: `.venv/Scripts/python -m mia.main`

### Uso de uv vs pip

- Usar `uv pip install` para instalar paquetes (más rápido, resuelve mejor)
- Usar `uv pip install --force-reinstall` cuando hay conflictos de binary compatibility
- **No usar** `uv run` para ejecutar MIA — puede reinstalar paquetes y romper numpy
- Siempre ejecutar con: `.venv/Scripts/python -m mia.main`
- Para tests: `.venv/Scripts/python -m pytest tests/ -v`

### Discord dependencies

```bash
uv pip install "py-cord[voice]>=2.6.0" "python-dotenv>=1.0.0"
```

- `py-cord` reemplaza `discord.py` — ambos no pueden coexistir
- Requiere `ffmpeg` instalado en el sistema para audio de Discord
- Token del bot en `.env` (archivo gitignored)

---

## Perfilado

Todo cambio al pipeline debe medir tiempos por etapa:
- `stt_ms`, `rag_retrieval_ms`, `llm_first_token_ms`, `tts_first_audio_ms`, `end_to_end_ms`
- Reportar en logs nivel INFO
- No loguear por chunk (demasiado overhead)

---

## Definition of Done

Un cambio está listo si:
- No aumenta latencia percibida
- Configurable vía YAML
- Sin regresiones en tests
- Código claro, modular, sin dependencias innecesarias
- No rompe CUDA torch
