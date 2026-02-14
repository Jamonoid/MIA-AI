# AGENTS.md – Guía para agentes IA que trabajan en MIA

## Visión general

MIA es un **bot de Discord con voz e inteligencia artificial**. Se conecta a servidores de Discord, escucha a los usuarios en voice channels, y responde con voz sintetizada y texto. No usa interfaz local — todo corre como un servicio headless.

## Arquitectura

```
main.py → pipeline.py → discord_bot.py ↔ discord_sink.py
                ↓            ↓
            config.py    STT / LLM / TTS / RAG
```

### Flujo de voz (Discord voice channel)
1. `discord_sink.py` recibe audio PCM de todos los usuarios
2. Cuando detecta silencio grupal (`group_silence_ms`), concatena el audio
3. `discord_bot.py` envía el audio al STT → texto
4. Texto → LLM (con contexto RAG si hay) → respuesta
5. Respuesta → strip emotion tags `[emotion]` → TTS → audio
6. Audio → FFmpeg → reproduce en Discord voice channel

### Flujo de texto (text channel)
1. Usuario menciona al bot o usa `/mia <texto>`
2. `discord_bot.py` envía texto → LLM → respuesta
3. Respuesta → strip emotion tags → envía como mensaje de texto

## Estructura de archivos

```
src/mia/
├── main.py              # Entry point, carga .env y config
├── pipeline.py          # Carga STT/LLM/TTS/RAG, inicia Discord bot
├── config.py            # Dataclasses tipadas desde config.yaml
├── discord_bot.py       # Bot de Discord (slash commands, voice, text)
├── discord_sink.py      # Recepción de audio grupal desde Discord
├── stt_whispercpp.py    # STT con faster-whisper (CTranslate2)
├── llm_openrouter.py    # LLM vía OpenRouter API
├── llm_lmstudio.py      # LLM vía LM Studio (local)
├── llm_llamacpp.py      # LLM vía llama.cpp (local)
├── tts_edge.py          # TTS con Microsoft Edge TTS
├── rag_memory.py        # RAG con ChromaDB + sentence-transformers
└── __init__.py
```

## Convenciones

- **Idioma del código**: docstrings y comentarios en español
- **Config**: todo configurable via `config.yaml`, tipado con dataclasses en `config.py`
- **Prompts**: archivos `.md` modulares en `prompts/`, cargados alfabéticamente
- **Secrets**: en `.env` (gitignored), cargados con `python-dotenv`
- **LLM tags**: el LLM puede generar `[emotion]` al inicio — `_strip_emotion_tags()` los limpia antes de TTS/texto

## Config sections

| Sección | Archivo | Descripción |
|---------|---------|-------------|
| `prompt` | `config.py:PromptConfig` | System prompt y directorio de prompts |
| `models.llm` | `config.py:LLMConfig` | Backend, modelo, temperatura, tokens |
| `models.stt` | `config.py:STTConfig` | Whisper model size, idioma, device |
| `models.tts` | `config.py:TTSConfig` | Edge TTS voice, rate, pitch |
| `rag` | `config.py:RAGConfig` | ChromaDB, embeddings, thresholds |
| `discord` | `config.py:DiscordConfig` | Silence detection, text responses |

## Secrets requeridos

| Variable | Dónde | Obligatorio |
|----------|-------|-------------|
| `DISCORD_BOT_TOKEN` | `.env` | ✅ Siempre |
| `OPENROUTER_API_KEY` | `.env` o `config.yaml` | Solo si backend = openrouter |

## Tests

```bash
.venv\Scripts\python -m pytest tests/ -v
```

Tests existentes: `test_config.py`, `test_prompt.py`, `test_rag.py`

## Ejecutar

```bash
.venv\Scripts\python -m mia.main
```

## Notas para agentes

- **No crear archivos de audio local, VAD, lipsync, VTube Studio, WebUI.** Este branch es Discord-only.
- `discord_sink.py` tiene su propio sistema de detección de silencio (no usar `vad.py`)
- Los prompts se cargan de `prompts/` en orden alfabético y se concatenan con `---`
- El `_strip_emotion_tags()` en `discord_bot.py` usa regex `\[\w+\]\s*` para limpiar tags
