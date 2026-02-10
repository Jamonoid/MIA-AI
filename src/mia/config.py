"""
config.py – Carga y validación de configuración desde YAML.

Provee dataclasses tipadas para cada sección del config.yaml.
Única fuente de verdad para todos los parámetros del sistema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")


# ──────────────────────────────────────────────
# Dataclasses de configuración
# ──────────────────────────────────────────────


@dataclass
class PromptConfig:
    system: str = "Eres MIA, una asistente virtual inteligente y amigable."


@dataclass
class LLMConfig:
    backend: str = "llamacpp"  # "llamacpp" | "lmstudio" | "openrouter"
    path: str = "./models/llama-3-8b.gguf"
    context_size: int = 2048
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    n_gpu_layers: int = -1  # -1 = todas las capas en GPU
    # LM Studio / OpenRouter settings
    base_url: str = "http://localhost:1234/v1"
    model_name: str = "default"
    api_key: str = ""  # OpenRouter API key (o env OPENROUTER_API_KEY)


@dataclass
class STTConfig:
    model_size: str = "base"
    language: str = "es"
    device: str = "auto"  # "cpu", "cuda", "auto"
    compute_type: str = "int8"


@dataclass
class TTSConfig:
    backend: str = "xtts"  # "xtts" | "edge"
    voice_path: str = "./voices/female_01.wav"
    chunk_size: int = 150
    language: str = "es"
    device: str = "auto"
    # Edge TTS settings (solo si backend: "edge")
    edge_voice: str = "es-MX-DaliaNeural"
    edge_rate: str = "+0%"
    edge_pitch: str = "+0Hz"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 20  # tamaño de chunk en ms
    playback_sample_rate: int = 24000


@dataclass
class VADConfig:
    energy_threshold: float = 0.01
    silence_duration_ms: int = 800  # ms de silencio para cortar
    min_speech_duration_ms: int = 300


@dataclass
class RAGConfig:
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    persist_dir: str = "./data/chroma_db"
    top_k: int = 3
    max_docs: int = 5000
    score_threshold: float = 0.3


@dataclass
class LipsyncConfig:
    smoothing_alpha: float = 0.2
    rms_min: float = 0.0
    rms_max: float = 0.3


@dataclass
class OSCConfig:
    ip: str = "127.0.0.1"
    port: int = 9000
    mapping: dict[str, str] = field(default_factory=lambda: {
        "mouth_open": "MouthOpen",
        "blink": "EyeBlink",
    })


@dataclass
class WebSocketConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    enabled: bool = True


@dataclass
class PerformanceConfig:
    vad_sensitivity: float = 0.5
    lipsync_smoothing: float = 0.2


@dataclass
class MIAConfig:
    """Configuración raíz de MIA."""

    prompt: PromptConfig = field(default_factory=PromptConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    lipsync: LipsyncConfig = field(default_factory=LipsyncConfig)
    osc: OSCConfig = field(default_factory=OSCConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


# ──────────────────────────────────────────────
# Carga
# ──────────────────────────────────────────────


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Construye un dataclass desde un dict, ignorando keys desconocidas."""
    if not data:
        return cls()
    fieldnames = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in fieldnames}
    return cls(**filtered)


def load_config(path: Path | str | None = None) -> MIAConfig:
    """Carga config.yaml y devuelve MIAConfig tipado."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning("Config no encontrado en %s, usando defaults.", config_path)
        return MIAConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    logger.info("Config cargado desde %s", config_path)

    # Mapear secciones del YAML a secciones del config
    models = raw.get("models", {})

    cfg = MIAConfig(
        prompt=_dict_to_dataclass(PromptConfig, raw.get("prompt", {})),
        llm=_dict_to_dataclass(LLMConfig, models.get("llm", {})),
        stt=_dict_to_dataclass(STTConfig, models.get("stt", {})),
        tts=_dict_to_dataclass(TTSConfig, models.get("tts", {})),
        audio=_dict_to_dataclass(AudioConfig, raw.get("audio", {})),
        vad=_dict_to_dataclass(VADConfig, raw.get("vad", {})),
        rag=_dict_to_dataclass(RAGConfig, raw.get("rag", {})),
        lipsync=_dict_to_dataclass(LipsyncConfig, raw.get("lipsync", {})),
        osc=_dict_to_dataclass(OSCConfig, raw.get("osc", {})),
        websocket=_dict_to_dataclass(WebSocketConfig, raw.get("websocket", {})),
        performance=_dict_to_dataclass(PerformanceConfig, raw.get("performance", {})),
    )

    return cfg
