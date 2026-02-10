# AGENTS.md

Guía para agentes (IA o humanos) que contribuyen al proyecto **MIA-AI** (Live2D) con foco en **latencia mínima**, sin Unity, integrable con VTube Studio vía OSC o con frontend propio. Incluye memoria RAG para contexto conversacional persistente.

---

## Objetivo del proyecto

Construir un asistente VTuber local con el menor retardo percibido posible:

- Entrada: micrófono → VAD → STT (streaming)
- Contexto: RAG memory (recuperación de fragmentos relevantes)
- Generación: LLM (streaming, con contexto RAG inyectado)
- Salida: TTS (XTTS, chunking) → reproducción de audio
- Animación: lipsync + expresiones → OSC a VTube Studio **o** WebSocket a frontend propio

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
    lipsync.py
    rag_memory.py
    vtube_osc.py
    ws_server.py
  data/
    chroma_db/          # vector store persistente (auto-generado)
  tests/
    test_config.py
    test_prompt.py
    test_rag.py
```

> Un agente debe mantener los módulos pequeños y con responsabilidad única.

---

## Configuración (YAML)

- **No hardcodear** rutas de modelos ni prompts: todo va en `config.yaml`.
- La sección `prompt.system` define la personalidad.
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

### TTS (XTTS)
- Debe soportar **chunking** de texto:
  - sintetizar 120–160 caracteres por chunk (configurable)
  - reproducir mientras se genera el siguiente chunk
- Evitar ajustes “quality-first” por defecto.
- Manejar cola de salida (no desordenar chunks).
- **torch.load compatibility**: torch 2.6+ usa `weights_only=True` por defecto, incompatible con XTTS. `tts_xtts.py` tiene un monkey-patch para forzar `weights_only=False` durante la carga.

### Lipsync
- Modo recomendado: **RMS** (simple, estable y rápido).
- Visemas solo si el costo extra no afecta la latencia percibida.

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
