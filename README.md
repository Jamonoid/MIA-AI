# MIA-AI

Local VTuber assistant pipeline. Low latency, RAG memory, VTube Studio integration via OSC.  

**Python:** 3.11+  
**Status:** Alpha

---

## Hardware requirements

### Full local (llamacpp + STT + TTS)

Everything runs on your machine. Needs a dedicated NVIDIA GPU.

| Resource | Minimum            | Recommended             |
|----------|--------------------|-------------------------|
| RAM      | 16 GB              | 32 GB                   |
| VRAM     | 6 GB (8B Q4)       | 12 GB+ (larger models)  |
| GPU      | NVIDIA GTX 1060    | NVIDIA RTX 3060+        |
| Disk     | ~10 GB (models)    | ~15 GB                  |
| CPU      | 4 cores            | 8+ cores                |

Breakdown by component:
- **STT** (faster-whisper large-v3): ~3 GB VRAM, ~3 GB disk
- **LLM** (8B Q4_K_M): ~5 GB VRAM, ~5 GB disk
- **TTS** (XTTS v2): ~2 GB VRAM, ~2 GB disk
- **RAG** (MiniLM-L6): ~0.5 GB RAM, ~80 MB disk

### CPU-only mode (no GPU)

Possible but significantly slower. Set `device: "cpu"` in STT/TTS config and `n_gpu_layers: 0` for LLM. Expect 5-10x higher latency.

---

## Overview

Voice-to-avatar pipeline with streaming at every stage:

```
Mic -> VAD -> STT -> RAG -> LLM (stream) -> TTS (chunked) -> Audio
                                                |
                                          Lipsync -> OSC / WebSocket
```

### Latency targets

| Stage              | Target     |
|--------------------|------------|
| LLM first token    | < 300 ms   |
| First voice output | < 900 ms   |
| Lipsync rate       | 50-100 Hz  |
| RAG retrieval      | < 50 ms    |

---

## Project structure

```
src/mia/
    main.py               Entry point
    config.py              YAML -> typed dataclasses
    pipeline.py            Async orchestrator

    audio_io.py            Mic capture + playback queue
    vad.py                 Energy-based voice activity detection
    stt_whispercpp.py      STT (faster-whisper)
    llm_llamacpp.py        LLM local (llama-cpp-python)
    llm_lmstudio.py        LLM via LM Studio (OpenAI API)
    llm_openrouter.py      LLM via OpenRouter (cloud)
    tts_xtts.py            TTS with chunking (XTTS v2)
    rag_memory.py          Conversational memory (ChromaDB)

    lipsync.py             RMS -> mouth_open (0..1)
    vtube_osc.py           OSC to VTube Studio
    ws_server.py           WebSocket broadcast (JSON)

models/
    stt/                   faster-whisper-large-v3 (auto-download)
    tts/                   XTTS v2 (auto-download)

voices/                    Reference WAV for voice cloning (~10s)
data/chroma_db/            Persistent vector store (auto-generated)
tests/                     Unit tests
config.yaml                All configuration
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/Jamonoid/MIA-AI.git
cd MIA-AI
uv venv
uv pip install -e ".[dev]"
```

### 2. Install CUDA PyTorch

PyPI ships CPU-only torch. For GPU support, install from the PyTorch index:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"  # True
```

### 3. Install ML dependencies

```bash
uv pip install faster-whisper
uv pip install TTS
```

**Important:** `TTS` may pull in CPU torch as a dependency, overwriting the CUDA version. After installing TTS, re-run the CUDA torch install from step 2.

For local LLM via llama-cpp-python (only if using `backend: "llamacpp"`):
```bash
uv pip install llama-cpp-python
```

### 4. Models

| Component | Model                    | Location                        |
|-----------|--------------------------|---------------------------------|
| LLM       | Any GGUF (e.g. Llama 3) | `./models/llama-3-8b.gguf`     |
| STT       | faster-whisper large-v3  | Auto-download on first run      |
| TTS       | XTTS v2                  | Auto-download on first run      |
| Voice     | WAV reference (~10s)     | `./voices/female_01.wav`        |

### Known issues

- **torch version conflict:** `TTS` installs CPU torch from PyPI, overwriting CUDA torch. Always re-run step 2 after installing TTS.
- **transformers version:** Coqui TTS 0.22 requires `transformers<4.44` (pinned in pyproject.toml). Newer versions remove `BeamSearchScorer` which TTS depends on.
- **torch.load weights_only:** torch 2.6+ defaults to `weights_only=True`, incompatible with XTTS model files. Patched in `tts_xtts.py`.

---

## Configuration

All settings in `config.yaml`:

```yaml
prompt:
  system: "Personality"

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
    api_key: ""                    # OpenRouter only
  stt:
    model_size: "large-v3"
    language: "es"
  tts:
    voice_path: "./voices/female_01.wav"
    chunk_size: 150

rag:
  enabled: true
  top_k: 3
  max_docs: 5000

osc:
  ip: "127.0.0.1"
  port: 9000
```

---

## LLM backends

### llamacpp (default)

Runs locally using llama-cpp-python. Requires a GGUF model file and `pip install llama-cpp-python`.

```yaml
backend: "llamacpp"
path: "./models/llama-3-8b.gguf"
```

### lmstudio

Uses LM Studio's local OpenAI-compatible server. No compilation needed.

1. Install [LM Studio](https://lmstudio.ai/), load a model, start local server
2. Configure:

```yaml
backend: "lmstudio"
base_url: "http://localhost:1234/v1"
```

### openrouter

Cloud access to hundreds of models via [OpenRouter](https://openrouter.ai/).

1. Get an API key from openrouter.ai
2. Configure:

```yaml
backend: "openrouter"
base_url: "https://openrouter.ai/api/v1"
model_name: "meta-llama/llama-3-8b-instruct"
api_key: "sk-or-..."       # or env OPENROUTER_API_KEY
```

---

## Usage

```bash
.venv/Scripts/python -m mia.main
```

Note: prefer running via venv python directly. `uv run mia` may re-resolve dependencies and overwrite CUDA torch with CPU torch.

### VTube Studio

Enable OSC receiver in VTube Studio on port 9000. Parameters `MouthOpen` and `EyeBlink` update automatically.

### WebSocket

Server at `ws://127.0.0.1:8765`. Messages:

```json
{"type": "mouth", "value": 0.42}
{"type": "emotion", "value": "happy"}
{"type": "subtitle", "role": "assistant", "text": "Hola"}
{"type": "status", "value": "listening"}
```

---

## RAG memory

ChromaDB with `all-MiniLM-L6-v2` embeddings. Stores conversation pairs, retrieves relevant context per query, injects into LLM prompt. Persistent in `./data/chroma_db/`. Disable with `rag.enabled: false`.

---

## Tests

```bash
.venv/Scripts/python -m pytest -v
```

Covers: config loading, prompt construction, text chunking, VAD, lipsync, RAG.

---

## Stack

| Component | Technology                          |
|-----------|-------------------------------------|
| STT       | faster-whisper (CTranslate2)        |
| LLM       | llama-cpp-python / LM Studio / OpenRouter |
| TTS       | Coqui TTS (XTTS v2)                |
| VAD       | Energy-based (RMS)                  |
| Lipsync   | RMS with exponential smoothing      |
| Avatar    | VTube Studio (OSC) / WebSocket      |
| Memory    | ChromaDB + sentence-transformers    |
| Audio     | sounddevice (PortAudio)             |
| Config    | YAML -> typed dataclasses           |

---

## License

[GNU Affero General Public License v3.0](LICENSE)