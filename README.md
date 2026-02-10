# MIA-AI

Local VTuber assistant pipeline. Low latency, RAG memory, VTube Studio integration via OSC.  

**Python:** 3.11+  
**Status:** Alpha

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

### 1. Clone and install

```bash
git clone https://github.com/Jamonoid/MIA-AI.git
cd MIA-AI
uv venv
uv pip install -e ".[dev]"
```

### 2. ML dependencies (optional, heavy)

```bash
pip install faster-whisper
pip install llama-cpp-python
pip install TTS
```

On Windows, `TTS` requires Visual Studio Build Tools (C++ workload).

### 3. Models

| Component | Model                    | Location                        |
|-----------|--------------------------|---------------------------------|
| LLM       | Any GGUF (e.g. Llama 3) | `./models/llama-3-8b.gguf`     |
| STT       | faster-whisper large-v3  | Auto-download                   |
| TTS       | XTTS v2                  | Auto-download                   |
| Voice     | WAV reference (~10s)     | `./voices/female_01.wav`        |

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
uv run mia
```

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
uv run pytest -v
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