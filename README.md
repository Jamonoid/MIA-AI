# MIA – Bot de Discord con IA

Bot de Discord con inteligencia artificial que escucha, piensa y habla. Se une a voice channels, escucha a los usuarios, y responde con voz sintetizada. También responde mensajes de texto.

## Features

- **Voz en Discord** — Escucha y habla en voice channels usando STT + TTS
- **LLM configurable** — OpenRouter, LM Studio, o llama.cpp local
- **Memoria RAG** — Recuerda conversaciones usando ChromaDB con vectorización curada por LLM
- **Prompts modulares** — Personalidad definida en archivos `.md`
- **Baja latencia** — faster-whisper (CTranslate2) + Edge TTS
- **Modo proactivo** — MIA habla espontáneamente después de silencio
- **WebUI** — Panel de control en tiempo real con chat, terminal y visualización 3D de memoria
- **Filtro TTS** — Limpia asteriscos, paréntesis y caracteres especiales antes de sintetizar voz

## Requisitos

- Python 3.11+
- FFmpeg (para audio en Discord)
- Token de bot de Discord

## Instalación

```bash
# Clonar
git clone https://github.com/tu-usuario/MIA-AI.git
cd MIA-AI

# Entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Dependencias
pip install -e ".[dev]"

# ML (instalación manual)
pip install faster-whisper
# pip install llama-cpp-python  # Solo si usas backend local
```

## Configuración

### 1. Token de Discord

Crea un archivo `.env` en la raíz del proyecto:

```env
DISCORD_BOT_TOKEN=tu_token_aquí
OPENROUTER_API_KEY=tu_api_key  # Solo si usas OpenRouter
```

### 2. Config

Copia y edita `config.example.yaml`:

```bash
cp config.example.yaml config.yaml
```

Secciones principales:

| Sección | Qué configurar |
|---------|---------------|
| `models.llm` | Backend (openrouter/lmstudio/llamacpp), modelo, temperatura |
| `models.stt` | Modelo Whisper, idioma, device (cpu/cuda) |
| `models.tts` | Voz de Edge TTS, velocidad, tono |
| `rag` | Activar/desactivar memoria, embedding model |
| `discord` | Tiempo de silencio grupal, respuestas en text channels |
| `webui` | Puerto del panel de control (default: 8080) |

### 3. Prompts

Los archivos en `prompts/` definen la personalidad de MIA. Se cargan en orden alfabético:

```
prompts/
├── 01_identity.md    # Quién es MIA
├── 02_behavior.md    # Cómo responde
├── 04_memory.md      # Uso del contexto RAG
├── 05_discord.md     # Comportamiento en Discord
└── 06_proactive.md   # Habla proactiva
```

## Uso

### Ejecutables rápidos (doble click)

| Archivo | Descripción |
|---------|-------------|
| `Iniciar MIA.bat` | Arranca el bot (Ctrl+C para cerrar) |
| `Vectorizar Memoria.bat` | Procesa sesiones guardadas con LLM |
| `Borrar Memoria.bat` | Limpia toda la memoria vectorizada |

### Línea de comandos

```bash
.venv\Scripts\python -m mia.main
```

### WebUI

Al iniciar MIA, se levanta un panel web en `http://localhost:8080` con:

- **Chat** — Mensajes en tiempo real
- **Terminal** — Logs del sistema
- **Memory** — Visualización 3D interactiva de la memoria vectorizada (Three.js)
- **Controles** — Pausar escucha, modo proactivo, sliders, RAG toggle

### Slash commands en Discord

| Comando | Descripción |
|---------|-------------|
| `/join` | MIA se une a tu voice channel |
| `/leave` | MIA sale del voice channel |
| `/mia <texto>` | Enviar texto a MIA |
| `/mute @usuario` | Server mute a un usuario |
| `/unmute @usuario` | Unmute a un usuario |
| `/move @usuario #canal` | Mover usuario a otro voice channel |
| `/nick @usuario <nombre>` | Cambiar apodo de un usuario |
| `/sound <nombre>` | Reproducir un sonido |

### Conversación por voz

1. Usa `/join` para que MIA entre a tu canal de voz
2. Habla normalmente — MIA escucha a todos
3. Cuando hay silencio grupal (1.5s por defecto), procesa todo lo dicho
4. Responde con voz sintetizada en el canal

## Sistema de Memoria

MIA usa un sistema de memoria en dos fases:

1. **Sesión activa** — El historial se guarda automáticamente como `.jsonl` al cerrar MIA
2. **Vectorización offline** — Un LLM cura qué es relevante y solo eso se vectoriza en ChromaDB

```
MIA sesión → data/chat_sessions/*.jsonl → vectorize_memory.py → ChromaDB
```

Esto evita acumular datos irrelevantes en la memoria — solo se guardan datos útiles como nombres, preferencias, eventos importantes.

## Arquitectura

```
Usuario habla en Discord
        ↓
discord_sink.py (captura audio grupal)
        ↓  silencio detectado
discord_bot.py
        ↓
    STT (faster-whisper) → texto
        ↓
    LLM (OpenRouter/local) → respuesta
        ↓
    TTS (Edge TTS) → audio
        ↓           ↓
    FFmpeg → Discord   WebUI → chat + subtítulos
```

## Tests

```bash
.venv\Scripts\python -m pytest tests/ -v
```

## Estructura del proyecto

```
MIA-AI/
├── src/mia/
│   ├── main.py              # Entry point
│   ├── pipeline.py          # Carga módulos e inicia bot
│   ├── config.py            # Configuración tipada
│   ├── discord_bot.py       # Bot de Discord
│   ├── discord_sink.py      # Audio receiver
│   ├── stt_whispercpp.py    # Speech-to-Text
│   ├── llm_openrouter.py    # LLM via API
│   ├── llm_lmstudio.py      # LLM via LM Studio
│   ├── llm_llamacpp.py      # LLM local
│   ├── tts_edge.py          # Text-to-Speech
│   ├── tts_filter.py        # Filtro de texto para TTS
│   ├── rag_memory.py        # Memoria conversacional (ChromaDB)
│   ├── web_server.py        # WebUI server (aiohttp)
│   └── webui/               # Frontend (HTML/CSS/JS/Three.js)
├── prompts/                 # Personalidad modular
├── data/
│   ├── chat_sessions/       # Sesiones guardadas (JSONL)
│   └── chroma_db/           # Memoria vectorizada
├── vectorize_memory.py      # Script de vectorización con LLM
├── clear_memory.py          # Script de limpieza de memoria
├── Iniciar MIA.bat          # Ejecutable Windows
├── Vectorizar Memoria.bat   # Ejecutable Windows
├── Borrar Memoria.bat       # Ejecutable Windows
├── config.yaml              # Configuración
├── .env                     # Secrets (gitignored)
└── tests/                   # Tests
```

## Licencia

GPL-3.0 — ver [LICENSE](LICENSE)