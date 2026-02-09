<p align="center">
  <h1 align="center">MIA–AI
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License">
  <img src="https://img.shields.io/badge/VTube_Studio-OSC-purple" alt="VTube Studio">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status">
</p>

---

MIA es un pipeline de voz conversacional que convierte tu micrófono en un avatar interactivo:

```
🎤 Micrófono → VAD → STT → RAG Memory → LLM → TTS → 🔊 Audio
                                                  ↓
                                            Lipsync → VTube Studio / WebSocket
```

Todo funciona **localmente** y en **streaming** – el avatar empieza a hablar antes de que el LLM termine de generar texto.

### Metas de latencia

| Etapa | Objetivo |
|---|---|
| Primer token LLM | < 300 ms |
| Primera salida de voz | < 900 ms |
| Lipsync update rate | 50–100 Hz |
| RAG retrieval | < 50 ms |

---

## Arquitectura

```
MIA-AI/
├── config.yaml              # Toda la configuración (modelos, prompts, OSC, etc.)
├── pyproject.toml            # Dependencias y metadata del proyecto
├── AGENTS.md                 # Guía para contribuidores (IA o humanos)
│
├── src/mia/
│   ├── main.py               # Punto de entrada (uv run mia)
│   ├── config.py             # Carga tipada de YAML → dataclasses
│   ├── pipeline.py           # Orquestador async del pipeline completo
│   │
│   ├── audio_io.py           # Captura de mic + cola de reproducción
│   ├── vad.py                # Voice Activity Detection (RMS)
│   ├── stt_whispercpp.py     # Speech-to-Text (faster-whisper)
│   ├── llm_llamacpp.py       # LLM local (llama-cpp-python)
│   ├── llm_lmstudio.py       # LLM vía LM Studio (API OpenAI)
│   ├── tts_xtts.py           # Text-to-Speech con chunking (XTTS v2)
│   ├── rag_memory.py         # Memoria conversacional (ChromaDB)
│   │
│   ├── lipsync.py            # Sincronización labial (RMS → mouth_open)
│   ├── vtube_osc.py          # Control de VTube Studio vía OSC/UDP
│   └── ws_server.py          # WebSocket server para frontend propio
│
├── tests/                    # Tests unitarios
├── models/                   # Modelos GGUF (no incluidos)
├── voices/                   # Samples de voz para clonación (no incluidos)
└── data/chroma_db/           # Vector store persistente (auto-generado)
```

---

## Instalación

### Requisitos previos

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (gestor de paquetes rápido)
- **GPU con CUDA** (recomendado para STT/LLM/TTS)

### 1. Clonar y crear entorno

```bash
git clone https://github.com/Jamonoid/MIA-AI.git
cd MIA-AI
uv venv
uv pip install -e ".[dev]"
```

### 2. Instalar dependencias ML

Estas dependencias son pesadas y requieren compilación C++/CUDA:

```bash
# STT (faster-whisper descarga el modelo automáticamente)
pip install faster-whisper

# LLM
pip install llama-cpp-python

# TTS (requiere Visual Studio Build Tools en Windows)
pip install TTS
```

> ** Nota Windows:** Si `TTS` falla al compilar, instala [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) con el workload "C++ build tools".

### 3. Descargar modelos

Coloca los modelos en las rutas definidas en `config.yaml`:

```bash
mkdir models voices
```

| Modelo | Recomendado | Dónde |
|---|---|---|
| **LLM** | [Llama 3 8B Q4_K_M](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) | `./models/llama-3-8b.gguf` |
| **STT** | Whisper `base` (auto-descarga) | Automático |
| **TTS Voice** | WAV de referencia (~10s) | `./voices/female_01.wav` |

---

## ⚙️ Configuración

Todo se controla desde `config.yaml`:

```yaml
prompt:
  system: "Eres MIA, una asistente virtual inteligente y amigable."

models:
  llm:
    backend: "llamacpp"       # "llamacpp" | "lmstudio"
    path: "./models/llama-3-8b.gguf"
    context_size: 2048
    n_gpu_layers: -1          # -1 = todas las capas en GPU
    # LM Studio (solo si backend: "lmstudio")
    base_url: "http://localhost:1234/v1"
    model_name: "default"
  stt:
    model_size: "base"        # tiny | base | small | medium
    language: "es"
  tts:
    voice_path: "./voices/female_01.wav"
    chunk_size: 150            # caracteres por chunk TTS

rag:
  enabled: true
  top_k: 3
  max_docs: 5000

osc:
  ip: "127.0.0.1"
  port: 9000                  # Puerto de VTube Studio
  mapping:
    mouth_open: "MouthOpen"
    blink: "EyeBlink"

websocket:
  host: "127.0.0.1"
  port: 8765
  enabled: true
```

> Consulta [config.yaml](config.yaml) para ver todas las opciones disponibles.

---

## ▶️ Uso

### Ejecutar MIA

```bash
uv run mia
```

MIA se iniciará, cargará los modelos y comenzará a escuchar por el micrófono.

### Conectar con VTube Studio

1. Abre **VTube Studio**
2. Ve a **Settings → VTube Studio API → OSC Receiver**
3. Habilita OSC y configura el puerto `9000`
4. Los parámetros `MouthOpen` y `EyeBlink` se actualizarán automáticamente

### Conectar un frontend propio

MIA expone un servidor WebSocket en `ws://127.0.0.1:8765` que envía mensajes JSON:

```json
{"type": "mouth", "value": 0.42}
{"type": "emotion", "value": "happy"}
{"type": "subtitle", "role": "assistant", "text": "¡Hola!"}
{"type": "status", "value": "listening"}
```

### Usar LM Studio como backend de LLM

[LM Studio](https://lmstudio.ai/) es la forma más fácil de correr modelos locales — no requiere compilar `llama-cpp-python`.

1. **Descarga e instala** [LM Studio](https://lmstudio.ai/)
2. **Carga un modelo** desde la UI (ej. Llama 3 8B)
3. **Inicia el servidor local** → "Local Server" → Start
4. **Cambia el backend** en `config.yaml`:

```yaml
models:
  llm:
    backend: "lmstudio"
    base_url: "http://localhost:1234/v1"
```

5. **Ejecuta MIA:** `uv run mia`

> **💡 Ventajas:** No necesita compilación C++. Se puede cambiar de modelo desde la UI de LM Studio sin reiniciar MIA. GPU nativa.

---

## 🧪 Tests

```bash
uv run pytest -v
```

Los tests cubren:
- Carga y validación de config YAML
- Construcción de prompts (con RAG y sin RAG)
- Chunking de texto para TTS
- VAD (detección de silencio/habla)
- Lipsync (mapeo RMS → mouth_open)

---

## 🧠 Memoria RAG

MIA recuerda conversaciones pasadas gracias a un sistema RAG local:

- **Almacenamiento:** ChromaDB (persistente en `./data/chroma_db/`)
- **Embeddings:** `all-MiniLM-L6-v2` (~80 MB)
- **Funcionamiento:** Al final de cada turno, se almacena el par `(usuario, MIA)`. En cada nueva pregunta, se recuperan los fragmentos más relevantes y se inyectan en el prompt del LLM.
- **Desactivar:** Pon `rag.enabled: false` en `config.yaml`

---

## 📊 Stack tecnológico

| Componente | Tecnología |
|---|---|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) |
| LLM | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) o [LM Studio](https://lmstudio.ai/) |
| TTS | [Coqui TTS](https://github.com/coqui-ai/TTS) (XTTS v2) |
| VAD | Energy-based (RMS, zero deps) |
| Lipsync | RMS con smoothing exponencial |
| Avatar | VTube Studio vía OSC / WebSocket |
| Memoria | ChromaDB + sentence-transformers |
| Audio | sounddevice (PortAudio) |
| Config | YAML → dataclasses tipadas |

---

