# MIA-AI

Asistente VTuber con baja latencia, memoria conversacional RAG, y animación de avatar en tiempo real. Se integra con VTube Studio vía WebSocket Plugin API y soporta Discord voice channels. Incluye WebUI para monitoreo y control.

**Python:** 3.11+
**Estado:** Alpha
**Licencia:** [AGPL-3.0](LICENSE)

---

## ¿Qué es MIA?

MIA es un pipeline de voz-a-avatar que convierte lo que dices en respuestas habladas con animación facial sincronizada:

```
Mic/Discord → VAD → STT → RAG → LLM (stream) → TTS (chunked) → Audio + Lipsync → Avatar
```

### Características

- **Baja latencia**: streaming end-to-end, TTS paralelo con entrega ordenada
- **Memoria RAG**: ChromaDB con embeddings para contexto conversacional persistente
- **3 backends LLM**: llama-cpp-python (local), LM Studio, OpenRouter (nube)
- **TTS**: Edge TTS (online, sin GPU, múltiples voces)
- **Prompt modular**: archivos `.md` en `prompts/` que se concatenan automáticamente
- **WebUI**: panel de control con subtítulos, métricas, chat escrito, logs filtrables
- **VTube Studio**: expresiones faciales + lipsync + blink automático vía WebSocket Plugin API
- **Expresiones inteligentes**: el LLM genera tags de emoción que activan expresiones en el modelo Live2D
- **Sistema de turnos**: conversaciones como `asyncio.Task` con interrupciones y sincronización frontend↔backend
- **Discord**: escucha en voice channels, identifica quién habla, responde por voz

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
uv pip install websockets
```

Para LLM local (opcional):
```bash
uv pip install llama-cpp-python
```

Para Discord (opcional):
```bash
uv pip install "py-cord[voice]>=2.6.0" "python-dotenv>=1.0.0"
# ffmpeg debe estar instalado en el sistema
```

### 4. Configurar

```bash
cp config.example.yaml config.yaml
# Editar config.yaml con tu configuración
```

Para Discord, crear `.env`:
```
DISCORD_BOT_TOKEN=tu_token_aquí
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
    stt_whispercpp.py       STT (faster-whisper, GPU)
    llm_llamacpp.py         LLM local (llama-cpp-python)
    llm_lmstudio.py         LLM vía LM Studio (API OpenAI)
    llm_openrouter.py       LLM vía OpenRouter (nube)
    tts_edge.py             TTS con Microsoft Edge (edge-tts)
    rag_memory.py           Memoria conversacional (ChromaDB)

    lipsync.py              RMS → mouth_open (0..1)
    vtube_studio.py         WebSocket Plugin API hacia VTube Studio
    ws_server.py            WebSocket + servidor HTTP para WebUI
    discord_bot.py          Bot Discord con voice (py-cord)
    discord_sink.py         ContinuousVoiceSink para audio Discord

    conversations/          Sistema de turnos de conversación
      types.py              Type aliases y dataclasses compartidas
      message_handler.py    Sincronización frontend↔backend
      tts_manager.py        TTS paralelo con entrega ordenada + lipsync callback
      conversation_handler.py   Entry point: triggers → tasks
      single_conversation.py    Flujo completo de un turno
      conversation_utils.py     Helpers (señales, cleanup)

prompts/                    Prompt modular (archivos .md, gitignored)
web/                        WebUI (HTML/CSS/JS)
data/chroma_db/             Vector store persistente
tests/                      Tests unitarios (33 tests)
config.yaml                 Configuración central (gitignored)
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

### TTS

| Backend | Requiere GPU | Configuración |
|---------|-------------|---------------|
| **edge** | No (online) | `backend: "edge"`, `edge_voice: "es-CL-CatalinaNeural"` |

Listar voces Edge disponibles: `edge-tts --list-voices`

### Prompt modular

En vez de un string en `prompt.system`, MIA carga archivos `.md` de la carpeta `prompts/`:

```
prompts/
  expressions.md    → Instrucciones de expresiones faciales
  personality.md    → Personalidad y estilo
```

Se concatenan alfabéticamente para armar el system prompt. Si la carpeta no existe, usa el fallback `prompt.system`.

---

## VTube Studio

MIA se conecta a VTube Studio mediante la WebSocket Plugin API para controlar el modelo Live2D en tiempo real.

### Setup

1. Abrir VTube Studio
2. Ir a configuración → habilitar API (puerto 8001)
3. Iniciar MIA → aceptar popup de autenticación en VTube Studio
4. El token se guarda en `.vts_token` para sesiones futuras

### Expresiones

El LLM genera tags de emoción al inicio de cada respuesta:
```
[happy] ¡Hola! ¿Cómo estás?
```

MIA parsea el tag, activa la expresión correspondiente en VTube Studio, y envía solo el texto limpio a TTS.

Emociones soportadas: neutral, happy, sad, angry, surprised, scared, ashamed, pout, cry, super_happy

### Lipsync

MIA inyecta valores de `MouthOpen` en tiempo real procesando el audio TTS en sub-chunks de 20ms con timing basado en reloj para mantener sincronización.

### Configuración

```yaml
vtube_studio:
  enabled: true
  ws_url: "ws://localhost:8001"
  mouth_param: "MouthOpen"        # Nombre INPUT (tracking), no OUTPUT (Live2D)
  eye_l_param: "EyeOpenLeft"
  eye_r_param: "EyeOpenRight"
  expressions:
    neutral: "00_IdleFace.exp3.json"
    happy: "01_HappyFace.exp3.json"
    # ... etc (solo nombre de archivo, sin carpeta)
```

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

## Discord

MIA puede unirse a canales de voz de Discord, escuchar a los usuarios, y responder por voz.

### Setup

1. Crear bot en Discord Developer Portal
2. Instalar `py-cord[voice]` y `ffmpeg`
3. Agregar token en `.env`
4. Habilitar `discord.enabled: true` en `config.yaml`
5. Usar `/join` para que MIA entre al canal de voz

### Características

- Multi-speaker: identifica quién habla
- Silence detection: detecta fin de grupo de mensajes
- TTS → FFmpeg → voice playback

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
| STT | faster-whisper (CTranslate2, GPU) |
| LLM | llama-cpp-python / LM Studio / OpenRouter |
| TTS | Edge TTS (Microsoft) |
| VAD | Energía (RMS) con pre-roll buffer |
| Lipsync | RMS con suavizado exponencial |
| Avatar | VTube Studio (WebSocket Plugin API) |
| Memoria | ChromaDB + sentence-transformers |
| Audio | sounddevice (PortAudio) / Web Audio API |
| WebUI | Vanilla HTML/CSS/JS |
| Discord | py-cord (voice channels) |
| Config | YAML → dataclasses tipados |

---

## Problemas conocidos

| Problema | Solución |
|----------|----------|
| `TTS` instala torch CPU | Re-ejecutar instalación de CUDA torch después |
| numpy binary incompatibility | `uv pip install --force-reinstall chromadb chroma-hnswlib` |
| VTS no reacciona a expresiones | Verificar nombres de archivo (sin prefijo `expressions/`) |
| VTS no hace lipsync | Usar nombres INPUT (`MouthOpen`) no OUTPUT (`ParamMouthOpenY`) |
| Windows symlinks (Hugging Face) | Usar `HF_HUB_DISABLE_SYMLINKS=1` |