# MIA-AI

Asistente VTuber local con baja latencia, memoria conversacional RAG, y animación de avatar en tiempo real. Se integra con VTube Studio vía OSC y ofrece un WebUI para monitoreo y control. Soporta entrada de audio por micrófono local y Discord (próximamente).

**Python:** 3.11+
**Estado:** Alpha
**Licencia:** [AGPL-3.0](LICENSE)

---

## ¿Qué es MIA?

MIA es un pipeline de voz-a-avatar que convierte lo que dices en respuestas habladas con animación facial sincronizada:

```
Mic → VAD → STT → RAG → LLM (stream) → TTS (chunked) → Audio + Lipsync → Avatar
```

### Características

- **Baja latencia**: streaming end-to-end, TTS paralelo con entrega ordenada
- **Memoria RAG**: ChromaDB con embeddings para contexto conversacional persistente
- **3 backends LLM**: llama-cpp-python (local), LM Studio, OpenRouter (nube)
- **2 backends TTS**: XTTS v2 (local, clonación de voz) o Edge TTS (online, sin GPU)
- **Prompt modular**: archivos `.md` en `prompts/` que se concatenan automáticamente
- **WebUI**: panel de control con subtítulos, métricas, chat escrito, logs filtrables
- **VTube Studio**: lipsync + blink automático vía OSC
- **Sistema de turnos**: conversaciones como `asyncio.Task` con interrupciones y sincronización frontend↔backend
- **Discord** *(próximamente)*: escucha en voice channels, identifica quién habla, responde por voz

---

## Requisitos

### Con GPU (recomendado)

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| RAM | 16 GB | 32 GB |
| VRAM | 6 GB | 12 GB+ |
| GPU | NVIDIA GTX 1060 | RTX 3060+ |
| Disco | ~10 GB | ~15 GB |

### Sin GPU

Con `backend: "edge"` para TTS y un LLM remoto (LM Studio u OpenRouter), MIA funciona sin GPU dedicada. Solo necesita conexión a internet.

---

## Instalación

### 1. Clonar y crear entorno

```bash
git clone https://github.com/Jamonoid/MIA-AI.git
cd MIA-AI
uv venv
uv pip install -e ".[dev]"
```

### 2. Instalar PyTorch con CUDA

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verificar:
```bash
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"  # True
```

### 3. Instalar dependencias ML

```bash
uv pip install faster-whisper
uv pip install TTS
# Re-instalar CUDA torch (TTS sobreescribe con CPU):
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Para LLM local (opcional):
```bash
uv pip install llama-cpp-python
```

### 4. Configurar

```bash
cp config.example.yaml config.yaml
# Editar config.yaml con tu configuración
```

### 5. Ejecutar

```bash
.venv/Scripts/python -m mia.main
```

> **Nota:** No usar `uv run mia` — re-resuelve dependencias y puede sobreescribir CUDA torch con la versión CPU.

---

## Estructura del proyecto

```
src/mia/
    main.py                 Punto de entrada
    config.py               YAML → dataclasses tipados
    pipeline.py             Orquestador asíncrono principal

    audio_io.py             Captura de mic + cola de reproducción
    vad.py                  Detección de actividad vocal (energía + pre-roll)
    stt_whispercpp.py       STT (faster-whisper)
    llm_llamacpp.py         LLM local (llama-cpp-python)
    llm_lmstudio.py         LLM vía LM Studio (API OpenAI)
    llm_openrouter.py       LLM vía OpenRouter (nube)
    tts_xtts.py             TTS con chunking (XTTS v2)
    tts_edge.py             TTS con Microsoft Edge (edge-tts)
    rag_memory.py           Memoria conversacional (ChromaDB)

    lipsync.py              RMS → mouth_open (0..1)
    vtube_osc.py            OSC hacia VTube Studio
    ws_server.py            WebSocket + servidor HTTP para WebUI

    conversations/          Sistema de turnos de conversación
      types.py              Type aliases y dataclasses compartidas
      message_handler.py    Sincronización frontend↔backend
      tts_manager.py        TTS paralelo con entrega ordenada
      conversation_handler.py   Entry point: triggers → tasks
      single_conversation.py    Flujo completo de un turno
      conversation_utils.py     Helpers (señales, cleanup)

prompts/                    Prompt modular (archivos .md)
web/                        WebUI (HTML/CSS/JS)
data/chroma_db/             Vector store persistente
tests/                      Tests unitarios (33 tests)
config.yaml                 Configuración central
```

---

## Configuración

Toda la configuración en `config.yaml`. Ver `config.example.yaml` como referencia.

### Backends de LLM

| Backend | Uso | Configuración |
|---------|-----|---------------|
| **llamacpp** | LLM local con GGUF | `backend: "llamacpp"`, `path: "./models/..."` |
| **lmstudio** | LM Studio (API OpenAI) | `backend: "lmstudio"`, `base_url: "http://localhost:1234/v1"` |
| **openrouter** | OpenRouter (nube) | `backend: "openrouter"`, `api_key: "sk-or-..."` |

### Backends de TTS

| Backend | Requiere GPU | Configuración |
|---------|-------------|---------------|
| **xtts** | Sí (~2 GB VRAM) | `backend: "xtts"`, necesita WAV de referencia |
| **edge** | No (online) | `backend: "edge"`, `edge_voice: "es-CL-CatalinaNeural"` |

Listar voces Edge disponibles: `edge-tts --list-voices`

### Prompt modular

En vez de un string en `prompt.system`, MIA carga archivos `.md` de la carpeta `prompts/`:

```
prompts/
  personality.md    → Personalidad y estilo
  expressions.md    → Instrucciones de expresiones faciales
```

Se concatenan alfabéticamente para armar el system prompt. Si la carpeta no existe, usa el fallback `prompt.system`.

---

## WebUI

Panel de control accesible en `http://localhost:8080`:

- Subtítulos en tiempo real (usuario + MIA)
- Chat escrito (modo texto)
- Métricas de latencia (STT, RAG, LLM, TTS)
- Mute/unmute, pausa del pipeline
- Toggle RAG on/off
- Configuración TTS (voz, velocidad, tono)
- Logs filtrables

---

## VTube Studio

1. Habilitar receptor OSC en VTube Studio (puerto 9000)
2. Los parámetros `MouthOpen` y `EyeBlink` se actualizan automáticamente

---

## Tests

```bash
.venv/Scripts/python -m pytest -v
```

33 tests cubriendo: configuración, prompts, chunking TTS, VAD, lipsync, RAG, MessageHandler, TTSTaskManager.

---

## Stack

| Componente | Tecnología |
|------------|------------|
| STT | faster-whisper (CTranslate2) |
| LLM | llama-cpp-python / LM Studio / OpenRouter |
| TTS | Coqui TTS (XTTS v2) / Edge TTS |
| VAD | Energía (RMS) con pre-roll buffer |
| Lipsync | RMS con suavizado exponencial |
| Avatar | VTube Studio (OSC) / WebSocket |
| Memoria | ChromaDB + sentence-transformers |
| Audio | sounddevice (PortAudio) / Web Audio API |
| WebUI | Vanilla HTML/CSS/JS |
| Config | YAML → dataclasses tipados |

---

## Problemas conocidos

| Problema | Solución |
|----------|----------|
| `TTS` instala torch CPU | Re-ejecutar paso 2 de instalación después |
| `transformers>=4.44` rompe TTS | Fijado a `<4.44` en pyproject.toml |
| `torch.load weights_only` | Parcheado automáticamente en `tts_xtts.py` |
| Windows symlinks (Hugging Face) | Usar `HF_HUB_DISABLE_SYMLINKS=1` |