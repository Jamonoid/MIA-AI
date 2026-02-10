"""
tts_xtts.py – Text-to-Speech con XTTS v2 (Coqui TTS).

Chunking de texto para síntesis incremental.
Cada chunk se sintetiza y encola para reproducción inmediata.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import TTSConfig

logger = logging.getLogger(__name__)


def chunk_text(text: str, max_chars: int = 150) -> list[str]:
    """
    Divide texto en chunks respetando límites de oración/cláusula.

    Prioriza cortar en:  . ! ? ; , (en ese orden)
    Si no hay puntuación, corta en espacio más cercano al límite.
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Buscar el mejor punto de corte dentro del rango
        segment = remaining[:max_chars]
        cut_index = -1

        # Priorizar puntuación fuerte
        for sep in [". ", "! ", "? ", "; ", ", ", " "]:
            idx = segment.rfind(sep)
            if idx > max_chars // 3:  # No cortar demasiado pronto
                cut_index = idx + len(sep)
                break

        if cut_index <= 0:
            cut_index = max_chars  # Forzar corte

        chunk = remaining[:cut_index].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut_index:].strip()

    return chunks


class XTTS:
    """Motor TTS usando XTTS v2 de Coqui."""

    def __init__(self, config: TTSConfig) -> None:
        self.voice_path = config.voice_path
        self.chunk_size = config.chunk_size
        self.language = config.language
        self.device = config.device
        self._tts = None
        self._sample_rate: int = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        """Carga el modelo XTTS v2."""
        try:
            from TTS.api import TTS

            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Torch 2.6+ defaults weights_only=True, incompatible with Coqui TTS 0.22
            # This monkey-patch forces weights_only=False for torch.load calls made by TTS.
            import torch
            _original_load = torch.load
            torch.load = lambda *args, **kwargs: _original_load(
                *args, **{**kwargs, "weights_only": False}
            )

            logger.info("Cargando XTTS v2 en device=%s", device)

            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._tts.to(device)

            # Restaurar torch.load original
            torch.load = _original_load

            # Verificar que el archivo de voz existe
            voice = Path(self.voice_path)
            if not voice.exists():
                logger.warning(
                    "Archivo de voz no encontrado: %s. "
                    "TTS funcionará pero sin clonación de voz.",
                    self.voice_path,
                )

            logger.info("XTTS v2 cargado correctamente")
        except ImportError:
            logger.error(
                "Coqui TTS no instalado. Instalar con: pip install TTS"
            )
            raise

    def synthesize(self, text: str) -> np.ndarray:
        """
        Sintetiza un fragmento de texto a audio.

        Returns:
            Audio como np.ndarray float32, mono.
        """
        if self._tts is None:
            raise RuntimeError("TTS no cargado. Llamar load() primero.")

        t0 = time.perf_counter()

        voice_path = Path(self.voice_path)
        if voice_path.exists():
            wav = self._tts.tts(
                text=text,
                speaker_wav=str(voice_path),
                language=self.language,
            )
        else:
            wav = self._tts.tts(text=text, language=self.language)

        audio = np.array(wav, dtype=np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "TTS chunk: '%s...' → %d samples (%.1f ms)",
            text[:40],
            len(audio),
            elapsed_ms,
        )
        return audio

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Sintetiza texto en chunks, yielding audio incrementalmente.

        El pipeline puede empezar a reproducir el primer chunk
        mientras los siguientes se generan.
        """
        chunks = chunk_text(text, self.chunk_size)
        logger.info("TTS stream: %d chunks de texto", len(chunks))

        first_chunk = True
        t_start = time.perf_counter()

        for chunk in chunks:
            if not chunk:
                continue
            audio = self.synthesize(chunk)

            if first_chunk:
                elapsed = (time.perf_counter() - t_start) * 1000
                logger.info("TTS first audio: %.1f ms", elapsed)
                first_chunk = False

            yield audio
