"""
tts_edge.py – Text-to-Speech con Microsoft Edge TTS (edge-tts).

Backend ligero que usa el servicio online de Microsoft Edge.
No requiere GPU, modelos locales ni API key.
Soporta chunking de texto para síntesis incremental.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import TTSConfig

logger = logging.getLogger(__name__)


def _decode_mp3_bytes(mp3_data: bytes) -> tuple[np.ndarray, int]:
    """
    Decodifica bytes MP3 a PCM float32 usando el decodificador disponible.

    Returns:
        Tupla (audio_array float32, sample_rate).
    """
    try:
        # Intentar con pydub (si está instalado)
        from pydub import AudioSegment

        seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= 2**15  # int16 → float32 normalizado
        if seg.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        return samples, seg.frame_rate
    except ImportError:
        pass

    try:
        # Intentar con soundfile + io
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(mp3_data))
        audio = audio.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr
    except (ImportError, RuntimeError):
        pass

    # Fallback: subprocess con ffmpeg
    import subprocess

    proc = subprocess.run(
        [
            "ffmpeg", "-i", "pipe:0",
            "-f", "s16le", "-ac", "1", "-ar", "24000",
            "pipe:1",
        ],
        input=mp3_data,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"No se pudo decodificar MP3. Instalar pydub o ffmpeg.\n"
            f"ffmpeg stderr: {proc.stderr.decode(errors='replace')}"
        )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    pcm /= 32768.0
    return pcm, 24000


class EdgeTTS:
    """Motor TTS usando Microsoft Edge TTS (edge-tts)."""

    def __init__(self, config: TTSConfig) -> None:
        self.voice = config.edge_voice
        self.rate = config.edge_rate
        self.pitch = config.edge_pitch
        self.chunk_size = config.chunk_size
        self._sample_rate: int = 24000
        # Event loop dedicado para correr edge-tts async desde hilos sync
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        """Verifica que edge-tts está instalado e inicia el event loop interno."""
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise ImportError(
                "edge-tts no instalado. Instalar con: pip install edge-tts"
            )

        # Crear event loop en hilo dedicado para llamadas async
        import threading

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="edge-tts-loop"
        )
        self._loop_thread.start()

        logger.info(
            "Edge TTS listo (voice=%s, rate=%s, pitch=%s)",
            self.voice, self.rate, self.pitch,
        )

    async def _synthesize_async(self, text: str) -> bytes:
        """Sintetiza texto a bytes MP3 usando edge-tts."""
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])

        return b"".join(mp3_chunks)

    def synthesize(self, text: str) -> np.ndarray:
        """
        Sintetiza un fragmento de texto a audio.

        Returns:
            Audio como np.ndarray float32, mono.
        """
        if self._loop is None:
            raise RuntimeError("TTS no cargado. Llamar load() primero.")

        t0 = time.perf_counter()

        # Ejecutar coroutine en el loop dedicado
        future = asyncio.run_coroutine_threadsafe(
            self._synthesize_async(text), self._loop
        )
        mp3_data = future.result(timeout=30)

        if not mp3_data:
            logger.warning("Edge TTS: respuesta vacía para '%s...'", text[:40])
            return np.zeros(0, dtype=np.float32)

        audio, sr = _decode_mp3_bytes(mp3_data)
        self._sample_rate = sr

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
        from .tts_xtts import chunk_text

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
