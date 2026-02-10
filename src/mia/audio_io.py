"""
audio_io.py – Captura de micrófono y reproducción de audio.

AudioCapture: streaming de mic vía sounddevice con callback non-blocking.
AudioPlayer:  cola de reproducción para chunks de audio generados por TTS.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from .config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captura audio del micrófono en chunks pequeños (non-blocking)."""

    def __init__(self, config: AudioConfig) -> None:
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.chunk_samples = int(config.sample_rate * config.chunk_ms / 1000)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._running = False
        self.muted = False  # When True, drops all audio from mic

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("Audio capture status: %s", status)
        if self.muted:
            return  # Drop audio when muted
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # Descartar si el consumidor es lento

    def start(self) -> None:
        """Inicia la captura de audio."""
        if self._running:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_samples,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        logger.info(
            "AudioCapture iniciado – SR=%d, chunk=%d samples",
            self.sample_rate,
            self.chunk_samples,
        )

    def stop(self) -> None:
        """Detiene la captura."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("AudioCapture detenido")

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        """Lee un chunk de la cola. Retorna None si timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class AudioPlayer:
    """Reproduce audio en cola, chunk por chunk (non-blocking)."""

    def __init__(self, config: AudioConfig) -> None:
        self.sample_rate = config.playback_sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._playing_event = threading.Event()

    def start(self) -> None:
        """Inicia el hilo de reproducción."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        logger.info("AudioPlayer iniciado – SR=%d", self.sample_rate)

    def stop(self) -> None:
        """Detiene el hilo de reproducción."""
        self._running = False
        # Flush la cola
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("AudioPlayer detenido")

    def enqueue(self, audio: np.ndarray) -> None:
        """Agrega un chunk de audio a la cola de reproducción."""
        self._queue.put(audio)

    def is_playing(self) -> bool:
        """Retorna True si hay audio reproduciéndose o en cola."""
        return self._playing_event.is_set() or not self._queue.empty()

    def _playback_loop(self) -> None:
        """Hilo de reproducción: toma chunks de la cola y los reproduce."""
        while self._running:
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._playing_event.clear()
                continue

            self._playing_event.set()
            try:
                sd.play(chunk, samplerate=self.sample_rate)
                sd.wait()
            except Exception as exc:
                logger.error("Error de reproducción: %s", exc)

        self._playing_event.clear()
