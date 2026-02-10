# MIA-AI

Pipeline de asistente VTuber local. Baja latencia, memoria RAG, integración con VTube Studio vía OSC.

**Python:** 3.11+  
**Estado:** Alpha

---

## Requisitos de hardware

### Local completo (llamacpp + STT + TTS XTTS)

Todo corre en tu PC. Necesita una GPU NVIDIA dedicada.

| Recurso  | Mínimo             | Recomendado             |
|----------|---------------------|-------------------------|
| RAM      | 16 GB               | 32 GB                   |
| VRAM     | 6 GB (8B Q4)        | 12 GB+ (modelos mayores)|
| GPU      | NVIDIA GTX 1060     | NVIDIA RTX 3060+        |
| Disco    | ~10 GB (modelos)    | ~15 GB                  |
| CPU      | 4 núcleos           | 8+ núcleos              |

Desglose por componente:
- **STT** (faster-whisper large-v3): ~3 GB VRAM, ~3 GB disco
- **LLM** (8B Q4_K_M): ~5 GB VRAM, ~5 GB disco
- **TTS** (XTTS v2): ~2 GB VRAM, ~2 GB disco
- **RAG** (MiniLM-L6): ~0.5 GB RAM, ~80 MB disco

### Modo sin GPU (TTS Edge)

Con `backend: "edge"` en la sección TTS, se usa el servicio online de Microsoft Edge, que **no requiere GPU ni modelo local**. Solo necesita conexión a internet.

### Modo CPU (sin GPU, XTTS)

Posible pero significativamente más lento. Configurar `device: "cpu"` en STT/TTS y `n_gpu_layers: 0` en LLM. Esperar 5-10x más latencia.

---

## Resumen

Pipeline de voz-a-avatar con streaming en cada etapa:

```
Mic -> VAD -> STT -> RAG -> LLM (stream) -> TTS (chunked) -> Audio
                                                |
                                          Lipsync -> OSC / WebSocket
```

### Metas de latencia

| Etapa                | Objetivo   |
|----------------------|------------|
| Primer token LLM     | < 300 ms   |
| Primera salida de voz | < 900 ms   |
| Tasa de lipsync       | 50-100 Hz  |
| Búsqueda RAG          | < 50 ms    |

---

## Estructura del proyecto

```
src/mia/
    main.py               Punto de entrada
    config.py              YAML -> dataclasses tipados
    pipeline.py            Orquestador asíncrono

    audio_io.py            Captura de mic + cola de reproducción
    vad.py                 Detección de actividad vocal (energía)
    stt_whispercpp.py      STT (faster-whisper)
    llm_llamacpp.py        LLM local (llama-cpp-python)
    llm_lmstudio.py        LLM vía LM Studio (API OpenAI)
    llm_openrouter.py      LLM vía OpenRouter (nube)
    tts_xtts.py            TTS con chunking (XTTS v2)
    tts_edge.py            TTS con Microsoft Edge (edge-tts)
    rag_memory.py          Memoria conversacional (ChromaDB)

    lipsync.py             RMS -> mouth_open (0..1)
    vtube_osc.py           OSC hacia VTube Studio
    ws_server.py           WebSocket broadcast (JSON)

models/
    stt/                   faster-whisper-large-v3 (descarga automática)
    tts/                   XTTS v2 (descarga automática)

voices/                    WAV de referencia para clonación de voz (~10s)
data/chroma_db/            Vector store persistente (auto-generado)
tests/                     Tests unitarios
config.yaml                Toda la configuración
```

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

PyPI solo distribuye torch CPU. Para soporte GPU:

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
```

**Importante:** `TTS` puede instalar torch CPU como dependencia, sobreescribiendo la versión CUDA. Después de instalar TTS, re-ejecutar el paso 2.

Para LLM local con llama-cpp-python (solo si usas `backend: "llamacpp"`):
```bash
uv pip install llama-cpp-python
```

### 4. Modelos

| Componente | Modelo                   | Ubicación                       |
|------------|--------------------------|---------------------------------|
| LLM        | Cualquier GGUF (ej. Llama 3) | `./models/llama-3-8b.gguf` |
| STT        | faster-whisper large-v3  | Descarga automática             |
| TTS        | XTTS v2                  | Descarga automática             |
| Voz        | WAV de referencia (~10s) | `./voices/female_01.wav`        |

### Problemas conocidos

- **Conflicto de versión de torch:** `TTS` instala torch CPU desde PyPI, sobreescribiendo CUDA torch. Siempre re-ejecutar el paso 2 después de instalar TTS.
- **Versión de transformers:** Coqui TTS 0.22 requiere `transformers<4.44` (fijado en pyproject.toml). Versiones más nuevas eliminan `BeamSearchScorer` que TTS necesita.
- **torch.load weights_only:** torch 2.6+ usa `weights_only=True` por defecto, incompatible con archivos de modelo XTTS. Parcheado en `tts_xtts.py`.

---

## Configuración

Toda la configuración en `config.yaml`:

```yaml
prompt:
  system: "Personalidad"

models:
  llm:
    backend: "llamacpp"            # llamacpp | lmstudio | openrouter
    path: "./models/llama-3-8b.gguf"
    context_size: 2048
    max_tokens: 512
    temperature: 0.7
    n_gpu_layers: -1
    # LM Studio / OpenRouter
    base_url: "http://localhost:1234/v1"
    model_name: "default"
    api_key: ""                    # Solo OpenRouter
  stt:
    model_size: "large-v3"
    language: "es"
  tts:
    backend: "edge"                # "xtts" | "edge"
    voice_path: "./voices/female_01.wav"
    chunk_size: 150
    # Edge TTS (solo si backend: "edge")
    edge_voice: "es-MX-DaliaNeural"
    edge_rate: "+0%"
    edge_pitch: "+0Hz"

rag:
  enabled: true
  top_k: 3
  max_docs: 5000

osc:
  ip: "127.0.0.1"
  port: 9000
```

---

## Backends de LLM

### llamacpp (por defecto)

Corre localmente usando llama-cpp-python. Requiere un archivo GGUF y `pip install llama-cpp-python`.

```yaml
backend: "llamacpp"
path: "./models/llama-3-8b.gguf"
```

### lmstudio

Usa el servidor local de LM Studio compatible con la API de OpenAI. No necesita compilación.

1. Instalar [LM Studio](https://lmstudio.ai/), cargar un modelo, iniciar el servidor local
2. Configurar:

```yaml
backend: "lmstudio"
base_url: "http://localhost:1234/v1"
```

### openrouter

Acceso en la nube a cientos de modelos vía [OpenRouter](https://openrouter.ai/).

1. Obtener una API key en openrouter.ai
2. Configurar:

```yaml
backend: "openrouter"
base_url: "https://openrouter.ai/api/v1"
model_name: "meta-llama/llama-3-8b-instruct"
api_key: "sk-or-..."       # o env OPENROUTER_API_KEY
```

---

## Backends de TTS

### xtts

Coqui TTS (XTTS v2). Síntesis local con clonación de voz. Requiere GPU y modelo (~2 GB VRAM).

```yaml
backend: "xtts"
voice_path: "./voices/female_01.wav"
chunk_size: 150
```

### edge

Microsoft Edge TTS. Servicio online, **no requiere GPU, modelo local ni API key**. Solo necesita conexión a internet.

```yaml
backend: "edge"
edge_voice: "es-MX-DaliaNeural"    # edge-tts --list-voices para ver todas
edge_rate: "+0%"                    # Velocidad: "+20%", "-10%", etc.
edge_pitch: "+0Hz"                  # Tono: "+10Hz", "-10Hz", etc.
```

Listar voces disponibles: `edge-tts --list-voices`

---

## Uso

```bash
.venv/Scripts/python -m mia.main
```

Nota: preferir ejecutar directamente con el python del venv. `uv run mia` puede re-resolver dependencias y sobreescribir CUDA torch con la versión CPU.

### VTube Studio

Habilitar el receptor OSC en VTube Studio en el puerto 9000. Los parámetros `MouthOpen` y `EyeBlink` se actualizan automáticamente.

### WebSocket

Servidor en `ws://127.0.0.1:8765`. Mensajes:

```json
{"type": "mouth", "value": 0.42}
{"type": "emotion", "value": "happy"}
{"type": "subtitle", "role": "assistant", "text": "Hola"}
{"type": "status", "value": "listening"}
```

---

## Memoria RAG

ChromaDB con embeddings `all-MiniLM-L6-v2`. Almacena pares de conversación, recupera contexto relevante por consulta e inyecta en el prompt del LLM. Persistente en `./data/chroma_db/`. Desactivar con `rag.enabled: false`.

---

## Tests

```bash
.venv/Scripts/python -m pytest -v
```

Cubre: carga de configuración, construcción de prompt, chunking de texto, VAD, lipsync, RAG.

---

## Stack

| Componente | Tecnología                              |
|------------|-----------------------------------------|
| STT        | faster-whisper (CTranslate2)            |
| LLM        | llama-cpp-python / LM Studio / OpenRouter |
| TTS        | Coqui TTS (XTTS v2) / Edge TTS         |
| VAD        | Basado en energía (RMS)                 |
| Lipsync    | RMS con suavizado exponencial           |
| Avatar     | VTube Studio (OSC) / WebSocket          |
| Memoria    | ChromaDB + sentence-transformers        |
| Audio      | sounddevice (PortAudio)                 |
| Config     | YAML -> dataclasses tipados             |

---

## Licencia

[GNU Affero General Public License v3.0](LICENSE)