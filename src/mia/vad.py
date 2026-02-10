"""
vad.py – Voice Activity Detection basado en energía (RMS).

Zero dependencias extra. Rápido y configurable.
Detecta inicio/fin de habla usando umbral de energía + duración mínima.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import VADConfig

logger = logging.getLogger(__name__)


class VADState:
    """Estado interno del VAD."""

    SILENCE = "silence"
    SPEECH = "speech"


class EnergyVAD:
    """Detección de actividad de voz basada en energía RMS."""

    def __init__(self, config: VADConfig, sample_rate: int = 16000) -> None:
        self.threshold = config.energy_threshold
        self.silence_duration = config.silence_duration_ms / 1000.0
        self.min_speech_duration = config.min_speech_duration_ms / 1000.0
        self.sample_rate = sample_rate

        self._state = VADState.SILENCE
        self._speech_start: float = 0.0
        self._last_speech: float = 0.0
        self._speech_buffer: list[np.ndarray] = []

        # Pre-roll: guardar los últimos N chunks de silencio para no perder
        # el inicio de las palabras (antes de cruzar el umbral de energía)
        self._pre_roll_size = 5  # ~100ms a 20ms/chunk
        self._pre_roll: list[np.ndarray] = []

    @staticmethod
    def rms(audio: np.ndarray) -> float:
        """Calcula RMS de un buffer de audio."""
        return float(np.sqrt(np.mean(audio**2)))

    def process(self, chunk: np.ndarray) -> tuple[str, np.ndarray | None]:
        """
        Procesa un chunk de audio.

        Retorna:
            (evento, audio_completo)
            - ("speech_start", None): se detectó inicio de habla
            - ("speech_end", audio_ndarray): se detectó fin de habla, retorna audio acumulado
            - ("silence", None): silencio continuo
            - ("speaking", None): habla continua
        """
        energy = self.rms(chunk)
        now = time.monotonic()

        if self._state == VADState.SILENCE:
            if energy >= self.threshold:
                self._state = VADState.SPEECH
                self._speech_start = now
                self._last_speech = now
                # Incluir pre-roll para no perder inicio de palabras
                self._speech_buffer = list(self._pre_roll) + [chunk]
                self._pre_roll = []
                logger.debug("VAD: speech_start (RMS=%.4f, pre_roll=%d chunks)", energy, len(self._speech_buffer) - 1)
                return "speech_start", None
            # Mantener buffer circular de pre-roll
            self._pre_roll.append(chunk)
            if len(self._pre_roll) > self._pre_roll_size:
                self._pre_roll.pop(0)
            return "silence", None

        else:  # SPEECH
            if energy >= self.threshold:
                self._last_speech = now
                self._speech_buffer.append(chunk)
                return "speaking", None
            else:
                self._speech_buffer.append(chunk)
                elapsed_silence = now - self._last_speech
                speech_duration = now - self._speech_start

                if elapsed_silence >= self.silence_duration:
                    self._state = VADState.SILENCE

                    if speech_duration >= self.min_speech_duration:
                        full_audio = np.concatenate(self._speech_buffer)
                        self._speech_buffer = []
                        logger.debug(
                            "VAD: speech_end (dur=%.2fs, samples=%d)",
                            speech_duration,
                            len(full_audio),
                        )
                        return "speech_end", full_audio

                    # Demasiado corto, descartar
                    self._speech_buffer = []
                    logger.debug("VAD: descartado (dur=%.2fs)", speech_duration)
                    return "silence", None

                return "speaking", None

    def reset(self) -> None:
        """Reinicia el estado del VAD."""
        self._state = VADState.SILENCE
        self._speech_buffer = []
        self._pre_roll = []
        self._speech_start = 0.0
        self._last_speech = 0.0
