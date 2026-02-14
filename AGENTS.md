# AGENTS.md – Guía para agentes IA que trabajan en MIA

## Visión general

MIA es un **bot de Discord con voz e inteligencia artificial**. Se conecta a servidores de Discord, escucha a los usuarios en voice channels, y responde con voz sintetizada y texto. Incluye un **WebUI** de control en `http://localhost:8080` y un **modo proactivo** donde MIA puede hablar sola tras un período de silencio.

## Arquitectura

```
main.py → pipeline.py → discord_bot.py ↔ discord_sink.py
                ↓            ↓      ↑
            config.py    STT / LLM / TTS / RAG
                             ↕
                        web_server.py → webui/ (HTML/CSS/JS)
```

### Flujo de voz (Discord voice channel)
1. `discord_sink.py` recibe audio PCM de todos los usuarios
2. Cuando detecta silencio grupal (`group_silence_ms`), concatena el audio
3. `discord_bot.py` envía el audio al STT → texto
4. Texto → LLM (con contexto RAG si hay) → respuesta
5. Respuesta → limpieza (`_clean_llm_output`: quita prefijo `MIA:`) → TTS → audio
6. Audio → FFmpeg → reproduce en Discord voice channel
7. Cada paso emite eventos vía **event bus** → WebSocket → WebUI

### Filtros STT (anti-alucinación)
Antes de enviar audio al STT, se aplican filtros en este orden:
1. **Duración**: `< 500ms` a 16kHz → descartado
2. **Energía RMS**: si RMS < umbral (configurable desde WebUI) → descartado
3. **STT**: audio → Whisper → texto
4. **Blacklist**: si texto está en `stt_hallucinations` (config.yaml) → descartado

### Flujo de texto (text channel)
1. Usuario menciona al bot o usa `/mia <texto>`
2. `discord_bot.py` envía texto → LLM → respuesta
3. Respuesta → limpieza (`_clean_llm_output`) → envía como mensaje de texto

### Flujo proactivo (modo continuo)
1. Un timer en `discord_bot.py` monitorea el silencio (`_proactive_loop`)
2. Si pasan `idle_seconds` sin voz, pregunta al LLM si tiene algo que decir
3. Si el LLM responde `NO`, se queda callada. Si responde con un mensaje, lo reproduce
4. El timer se resetea con cada actividad de voz

### WebUI (panel de control)
1. `web_server.py` levanta un server aiohttp en el puerto configurado
2. Sirve archivos estáticos de `webui/` (HTML/CSS/JS)
3. WebSocket bidireccional: eventos del bot → frontend, comandos del frontend → bot
4. `pipeline.py` conecta el event bus del bot con el WebSocket del server

## Estructura de archivos

```
src/mia/
├── main.py              # Entry point, carga .env, config, y logging (consola + archivo)
├── pipeline.py          # Carga STT/LLM/TTS/RAG, inicia Discord bot + WebUI server
├── config.py            # Dataclasses tipadas desde config.yaml
├── discord_bot.py       # Bot de Discord (slash commands, voice, text, event bus, proactive)
├── discord_sink.py      # Recepción de audio grupal desde Discord
├── web_server.py        # Servidor aiohttp embebido (HTTP + WebSocket)
├── webui/
│   ├── index.html       # Layout: chat panel + controles sidebar
│   ├── style.css        # Dark theme premium con glassmorphism
│   └── app.js           # WebSocket client + UI controller
├── stt_whispercpp.py    # STT con faster-whisper (CTranslate2)
├── llm_openrouter.py    # LLM vía OpenRouter API
├── llm_lmstudio.py      # LLM vía LM Studio (local)
├── llm_llamacpp.py      # LLM vía llama.cpp (local)
├── tts_edge.py          # TTS con Microsoft Edge TTS
├── rag_memory.py        # RAG con ChromaDB + sentence-transformers
└── __init__.py

prompts/
├── 01_identity.md       # Personalidad de MIA
├── 02_behavior.md       # Reglas de comportamiento
├── 05_discord.md        # Reglas para Discord (no prefijear "MIA:", usar nombres)
└── 06_proactive.md      # Prompt para habla proactiva

logs/                    # Logs por sesión (auto-creado, gitignored)
└── mia_YYYY-MM-DD_HH-MM-SS.log
```

## Convenciones

- **Idioma del código**: docstrings y comentarios en español
- **Config**: todo configurable via `config.yaml`, tipado con dataclasses en `config.py`
- **Prompts**: archivos `.md` modulares en `prompts/`, cargados alfabéticamente
- **Secrets**: en `.env` (gitignored), cargados con `python-dotenv`
- **LLM cleanup**: `_clean_llm_output()` quita prefijo `MIA:` del output del LLM
- **Event bus**: `discord_bot.on_event(callback)` registra listeners, `_emit()` notifica
- **Logging**: consola + archivo en `logs/` con timestamps, creado automáticamente por `main.py`

## Config sections

| Sección | Dataclass | Descripción |
|---------|-----------|-------------|
| `prompt` | `PromptConfig` | System prompt y directorio de prompts |
| `models.llm` | `LLMConfig` | Backend, modelo, temperatura, tokens |
| `models.stt` | `STTConfig` | Whisper model size, idioma, device, `stt_hallucinations` |
| `models.tts` | `TTSConfig` | Edge TTS voice, rate, pitch |
| `rag` | `RAGConfig` | ChromaDB, embeddings, thresholds |
| `discord` | `DiscordConfig` | Silence detection, text responses |
| `webui` | `WebUIConfig` | Puerto del WebUI (default: 8080) |
| `proactive` | `ProactiveConfig` | Modo proactivo: enabled, idle_seconds, prompt_file |

## Secrets requeridos

| Variable | Dónde | Obligatorio |
|----------|-------|-------------|
| `DISCORD_BOT_TOKEN` | `.env` | ✅ Siempre |
| `OPENROUTER_API_KEY` | `.env` o `config.yaml` | Solo si backend = openrouter |

## Tests

```bash
.venv\Scripts\python -m pytest tests/ -v
```

Tests existentes: `test_config.py`, `test_prompt.py`, `test_rag.py`, `test_webui.py`

## Ejecutar

```bash
.venv\Scripts\python -m mia.main
```

Esto inicia el bot de Discord + WebUI en `http://localhost:8080`. Los logs se guardan en `logs/`.

## Notas para agentes

- `discord_sink.py` tiene su propio sistema de detección de silencio (no usar `vad.py`)
- Los prompts se cargan de `prompts/` en orden alfabético y se concatenan con `---`
- `_clean_llm_output()` en `discord_bot.py` quita prefijo `MIA:` con regex `_NAME_PREFIX_RE`
- El WebUI se conecta vía WebSocket y usa un patrón command/event desacoplado
- El modo proactivo usa `_proactive_loop()` — el LLM decide si hablar (responde `NO` si no tiene nada que decir)
- Los speakers fantasma (nombre `User_*`) se filtran en `_on_group_silence`
- STT hallucinations se configuran en `config.yaml` → `models.stt.stt_hallucinations`
- Min energy RMS se ajusta en vivo solo desde el WebUI (default: 0.01)
