# AGENTS.md

Guía para agentes (IA o humanos) que contribuyen al proyecto **MIA-AI** (Live2D) con foco en **latencia mínima**, sin Unity, integrable con VTube Studio vía OSC o con frontend propio. Incluye memoria RAG para contexto conversacional persistente.

---

## Objetivo del proyecto

Construir un asistente VTuber local con el menor retardo percibido posible:

- Entrada: micrófono → VAD → STT (streaming)
- Contexto: RAG memory + búsqueda web (MCP/DuckDuckGo) + visión de pantalla
- Generación: LLM (streaming, con contexto inyectado)
- Salida: TTS (XTTS o Edge TTS, chunking) → reproducción de audio
- Animación: lipsync + expresiones → OSC a VTube Studio **o** WebSocket a frontend propio
- Control: WebUI para monitoreo y configuración en tiempo real

**Metas de latencia (referencia):**
- Primer token del LLM: < 300 ms (depende del modelo y GPU)
- Primera salida de voz: < 900 ms (XTTS suele ser el factor dominante)
- Lipsync: actualización 50–100 Hz sin bloqueos del hilo principal
- RAG retrieval: < 50 ms (búsqueda vectorial local)

---

## Principios no negociables (latencia)

1. **Nada pesado en el loop principal**: STT/LLM/TTS deben ejecutarse en hilos o procesos dedicados.
2. **Streaming end-to-end**: no esperar texto completo para hablar.
3. **Mensajes mínimos** hacia el avatar:
   - `mouth_open` (0..1), `blink` (0/1), `gaze` (x,y), `emotion` (enum)
4. **Evitar GC/allocations** en rutas calientes:
   - no construir strings grandes por frame
   - no logs en exceso en tiempo real
5. **Configuración en YAML** (una sola fuente de verdad) para:
   - modelos
   - prompt de personalidad
   - parámetros de performance
   - mapeo OSC

---

## Estructura recomendada del repositorio

```
mia/
  AGENTS.md
  config.yaml
  pyproject.toml
  src/mia/
    main.py
    config.py
    pipeline.py
    audio_io.py
    vad.py
    stt_whispercpp.py
    llm_llamacpp.py
    llm_lmstudio.py
    llm_openrouter.py
    tts_xtts.py
    tts_edge.py
    lipsync.py
    rag_memory.py
    vision.py             # captura de pantalla + LLM visión (OpenRouter)
    conversations/        # sistema de turnos de conversación
      __init__.py
      types.py            # type aliases y dataclasses compartidas
      message_handler.py  # sincronización frontend↔backend (asyncio.Event)
      tts_manager.py      # TTS paralelo con reordenamiento por seq number
      conversation_handler.py   # entry point: despacha triggers a tasks
      single_conversation.py    # flujo completo de conversación 1:1
      conversation_utils.py     # helpers compartidos
    tools/
      web_search.py       # búsqueda en internet (MCP + DuckDuckGo)
    vtube_osc.py
    ws_server.py          # WebSocket + servidor HTTP para WebUI
  prompts/                # prompt modular (archivos .md concatenados)
    personality.md
    expressions.md
    search.md
    vision.md
  web/                    # WebUI (vanilla HTML/CSS/JS)
    index.html
    style.css
    app.js
  data/
    chroma_db/            # vector store persistente (auto-generado)
  tests/
    test_config.py
    test_prompt.py
    test_rag.py
    test_message_handler.py
    test_tts_manager.py
```

> Un agente debe mantener los módulos pequeños y con responsabilidad única.

---

## Configuración (YAML)

- **No hardcodear** rutas de modelos ni prompts: todo va en `config.yaml`.
- La personalidad y las instrucciones del LLM se definen en la carpeta `prompts/` (ver sección **Prompt modular**).
- No agregar “lore” largo: mantener el prompt compacto para reducir tokens y latencia.

Si un agente necesita un parámetro nuevo:
1. Agregarlo a `config.yaml`
2. Leerlo en `config.py`
3. Documentarlo aquí en AGENTS.md
4. Añadir test si aplica

---

## Setup y ejecución

### 1) Crear entorno e instalar dependencias base
```bash
uv venv
uv pip install -e ".[dev]"
```

### 2) Instalar CUDA PyTorch (obligatorio para GPU)
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```
PyPI solo distribuye torch CPU. Sin este paso, STT/TTS/RAG corren en CPU.

### 3) Instalar dependencias ML
```bash
uv pip install faster-whisper
uv pip install TTS
# Re-instalar CUDA torch (TTS sobreescribe con CPU torch):
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4) Ejecutar
```bash
.venv/Scripts/python -m mia.main
```
**No usar `uv run mia`** — `uv run` re-resuelve dependencias y puede sobreescribir CUDA torch con la versión CPU de PyPI.

### Variables y rutas
- Modelos y voces se esperan bajo `./models` y `./voices` según `config.yaml`.
- El agente no debe asumir rutas absolutas.

---

## Reglas de implementación por componente

### Audio / VAD
- Audio en chunks pequeños (10–20 ms).
- VAD debe ser configurable y rápido.
- No bloquear el hilo de captura.

### STT
- Priorizar modelos `tiny`/`base` para baja latencia.
- Forzar idioma (`es`) para evitar auto-detección.

### LLM
- Preferir 7B–8B cuantizado.
- Streaming habilitado.
- Contexto moderado (ej. 2048) para evitar degradación de rendimiento.
- El prompt del LLM recibe fragmentos RAG **antes** del mensaje del usuario.

### RAG Memory
- **Motor**: ChromaDB (local, persistente en `./data/chroma_db/`).
- **Embeddings**: `sentence-transformers` con modelo ligero (`all-MiniLM-L6-v2` ~80 MB).
- **Ingesta**: al finalizar cada turno se almacenan pares `(user_msg, assistant_msg)` como documentos.
- **Retrieval**: buscar top-K fragmentos (configurable, default `k=3`) por similitud coseno.
- **Inyección en prompt**: insertar fragmentos relevantes como sección `## Contexto previo` antes del último mensaje del usuario.
- **Latencia**: la búsqueda vectorial debe completarse en < 50 ms; ejecutar en hilo separado si es necesario.
- **Limpieza**: respetar `rag.max_docs`; si se excede, eliminar los más antiguos.
- **Desactivación**: si `rag.enabled` es `false`, no cargar ChromaDB ni embeddings.

### TTS

Se soportan dos backends seleccionables vía `models.tts.backend` en `config.yaml`:

#### XTTS (`backend: "xtts"`)
-   Requiere GPU y modelo local (Coqui TTS).
-   Debe soportar **chunking** de texto:
    -   sintetizar 120–160 caracteres por chunk (configurable)
    -   reproducir mientras se genera el siguiente chunk
-   Evitar ajustes "quality-first" por defecto.
-   Manejar cola de salida (no desordenar chunks).
-   **torch.load compatibility**: torch 2.6+ usa `weights_only=True` por defecto, incompatible con XTTS. `tts_xtts.py` tiene un monkey-patch para forzar `weights_only=False` durante la carga.

#### Edge TTS (`backend: "edge"`)
-   Usa el servicio online de Microsoft Edge. **No requiere GPU, modelo local ni API key.**
-   Parámetros configurables en YAML:
    -   `edge_voice`: nombre de la voz (ej. `es-MX-DaliaNeural`). Listar disponibles: `edge-tts --list-voices`
    -   `edge_rate`: velocidad (ej. `"+10%"`, `"-10%"`)
    -   `edge_pitch`: tono (ej. `"+10Hz"`, `"-10Hz"`)
-   El audio se recibe como MP3 y se decodifica a PCM float32 (pydub → soundfile → ffmpeg fallback).
-   Latencia depende de la red; ideal para desarrollo o cuando no hay GPU disponible.

### Lipsync
-   Modo recomendado: **RMS** (simple, estable y rápido).
-   Visemas solo si el costo extra no afecta la latencia percibida.

### Integración avatar

#### OSC (VTube Studio)
- UDP, mensajes pequeños y frecuentes.
- Smoothing configurable (alpha).
- Respetar nombres exactos de parámetros del modelo (mapeo en YAML).

#### WebSocket (frontend propio)
- Protocolo mínimo:
  - `{"type":"mouth","value":0.42}`
  - `{"type":"emotion","value":"happy"}`
- Evitar enviar texto completo al frontend salvo para subtítulos.

### Prompt modular
- En vez de un string en `prompt.system`, MIA carga una carpeta `prompts/` con archivos `.md`.
- Cada archivo es un módulo del system prompt (personalidad, expresiones, herramientas, visión, etc.).
- Se concatenan en orden alfabético (o configurable) al armar el system prompt.
- Permite editar la personalidad sin tocar código ni reiniciar (hot-reload nice-to-have).
- Nuevos archivos `.md` en la carpeta se incluyen automáticamente.
- La carpeta se configura vía `prompt.dir` en `config.yaml` (default `./prompts/`).

### Búsqueda web (MCP + DuckDuckGo)
- El LLM puede solicitar búsquedas en internet emitiendo un tag/JSON especial en su respuesta.
- Módulo `tools/web_search.py` ejecuta la búsqueda vía servidor MCP con DuckDuckGo.
- Los resultados se inyectan como contexto y el LLM genera su respuesta final.
- Configurable en YAML: `tools.web_search.enabled`, `max_results`.
- Instrucciones para el LLM en `prompts/search.md`.
- No bloquear el pipeline: ejecutar en hilo separado.

### Vision LLM (captura de pantalla)
- Módulo `vision.py` captura la pantalla cada N segundos (configurable).
- Envía el screenshot a un LLM con visión vía **OpenRouter** (ej. `google/gemini-flash-1.5-8b`).
- La descripción devuelta se inyecta en el prompt como `## Lo que ves en pantalla`.
- Configurable: `vision.enabled`, `vision.interval_s`, `vision.model`, `vision.api_key`.
- **Toggle on/off desde el WebUI** vía comando WebSocket.
- Debe correr en hilo aparte para no bloquear el pipeline principal.
- Instrucciones para el LLM en `prompts/vision.md`.

### WebUI (panel de control)
- **Stack**: vanilla HTML/CSS/JS (sin frameworks ni build step).
- Servido como archivos estáticos desde `web/` por el mismo proceso Python.
- Comunicación bidireccional por WebSocket (el existente en `ws_server.py`).
- Funcionalidades:
  - Mute/unmute micrófono, pausa del pipeline
  - Subtítulos en tiempo real (usuario + MIA)
  - Historial de conversación
  - Caja de texto (modo chat escrito)
  - Métricas de latencia en vivo (STT, RAG, LLM, TTS)
  - Toggle de visión on/off
  - Toggle de RAG on/off
  - Selector de voz TTS / sliders de rate/pitch
  - Logs filtrables y panel de debug

---

## Estándares de código

- Python 3.11+
- Tipado gradual (anotaciones donde agreguen valor).
- Logging:
  - `INFO` para eventos de alto nivel
  - `DEBUG` solo para diagnóstico (apagado por defecto)
- No introducir dependencias pesadas sin justificación de latencia.

---

## Pruebas mínimas (obligatorias para cambios relevantes)

Agregar o actualizar tests en `tests/` cuando se modifique:
- carga/validación del YAML
- construcción del prompt
- serialización de mensajes OSC/WS
- chunking de texto para TTS
- ingesta y retrieval de RAG (`test_rag.py`)

Ejecutar:
```bash
.venv/Scripts/python -m pytest
```

---

## Perfilado y métricas

Todo agente que toque el pipeline debe:
- Medir tiempos por etapa:
  - `vad_ms`, `stt_ms`, `rag_retrieval_ms`, `llm_first_token_ms`, `tts_first_audio_ms`, `end_to_end_ms`
- Reportar en logs (nivel INFO) cada N interacciones.
- No imprimir por chunk (demasiado costo).

---

## Seguridad y comportamiento

- No inventar capacidades que no estén implementadas.
- Si `rag.enabled` es `false`, no simular memoria ni hacer referencia a conversaciones pasadas.
- Si `rag.enabled` es `true`, MIA puede referenciar contexto recuperado pero **nunca fabricar recuerdos** que no estén en los fragmentos.
- Si falta contexto, preguntar **una sola cosa** para reducir turnos.

### Sistema de turnos de conversación

El paquete `conversations/` implementa un sistema de control de turnos basado en `asyncio.Task`. La arquitectura está documentada en detalle en `conversation_turn_system_guide.md`.

#### Arquitectura general

```
WebSocket Message → conversation_handler → asyncio.Task
                                             ↓
                                    single_conversation
                                             ↓
                                    LLM stream → TTSTaskManager (parallel)
                                             ↓
                                    Ordered audio → WebSocket → Frontend
                                             ↓
                                    wait "frontend-playback-complete"
                                             ↓
                                    Turn complete
```

#### Componentes clave

| Módulo | Responsabilidad |
|---|---|
| `message_handler.py` | Sincronización request-response sobre WebSocket con `asyncio.Event` |
| `tts_manager.py` | TTS paralelo con sequence numbers + buffer de reordenamiento |
| `conversation_handler.py` | Entry point: recibe triggers, despacha tasks, maneja concurrency guard |
| `single_conversation.py` | Flujo completo de un turno de conversación (14 pasos) |
| `conversation_utils.py` | Helpers compartidos (señales, cleanup, finalización) |

#### Reglas de implementación

- **Un turno por cliente a la vez**: verificar `task.done()` antes de crear una nueva task.
- **TTS paralelo, entrega ordenada**: cada chunk de TTS recibe un sequence number; el sender loop bufferea y envía en orden.
- **Sincronización con frontend**: después de enviar todo el audio, el backend envía `backend-synth-complete` y **bloquea** hasta recibir `frontend-playback-complete`.
- **Cleanup siempre**: usar `try/except CancelledError/finally` en todo flujo de conversación.
- **Interrupciones**: al cancelar una task, guardar respuesta parcial en historial y marcar `[Interrupted by user]`.

#### Protocolo WebSocket (mensajes de conversación)

**Backend → Frontend:**
- `conversation-chain-start` — AI procesando
- `backend-synth-complete` — no hay más audio
- `force-new-message` — frontend inicia nueva burbuja
- `conversation-chain-end` — turno completo
- `audio-response` — chunk de audio + texto display
- `interrupt-signal` — conversación interrumpida

**Frontend → Backend:**
- `text-input` — usuario envió texto
- `mic-audio-end` — usuario terminó de hablar
- `frontend-playback-complete` — audio reproducido
- `interrupt` — usuario quiere interrumpir

---

## Tareas típicas para agentes

1. Mejorar streaming:
   - disminuir tamaño de chunks
   - paralelizar TTS/LLM
2. Reducir jitter de lipsync:
   - ajustar smoothing
   - evitar picos por GC
3. Mejorar estabilidad del prompt:
   - reducir tokens
   - reforzar estilo sin añadir texto largo
4. Optimizar RAG:
   - ajustar `k` y `max_docs` según latencia
   - experimentar con modelos de embedding más ligeros
   - implementar filtrado por relevancia mínima (score threshold)

---

## Gestión de dependencias

Versiones que causan problemas conocidos:

| Paquete        | Restricción      | Motivo                                              |
|----------------|------------------|-----------------------------------------------------|
| torch          | CUDA build       | PyPI solo tiene CPU. Instalar desde pytorch index   |
| torchaudio     | Misma versión/CUDA que torch | DLL mismatch si no coinciden           |
| transformers   | <4.44            | TTS 0.22 usa `BeamSearchScorer` (eliminado en 4.44) |
| TTS            | 0.22.0           | Sobreescribe torch con CPU al instalarse            |

**Regla**: después de instalar cualquier paquete que dependa de torch (TTS, sentence-transformers, etc.), verificar que torch sigue siendo CUDA:
```bash
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"  # debe ser True
```

---

## Definition of Done (DoD)

Un cambio se considera listo si:
- No sube latencia percibida en pruebas locales
- Configurable vía YAML
- Sin regressions en tests
- Código claro, modular y sin dependencias innecesarias
- No rompe CUDA torch (verificar después de cambios en dependencias)
