"""
lipsync.py – Sincronización labial basada en RMS.

Calcula mouth_open (0..1) a partir de la energía del audio.
Smoothing configurable para evitar saltos bruscos.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import LipsyncConfig

logger = logging.getLogger(__name__)


class RMSLipsync:
    """Lipsync basado en energía RMS con smoothing exponencial."""

    def __init__(self, config: LipsyncConfig) -> None:
        self.alpha = config.smoothing_alpha
        self.rms_min = config.rms_min
        self.rms_max = config.rms_max
        self._current_value: float = 0.0

    def process(self, audio_chunk: np.ndarray) -> float:
        """
        Procesa un chunk de audio y retorna mouth_open (0..1).

        Usa smoothing exponencial:
            value = alpha * new + (1 - alpha) * prev
        """
        rms = float(np.sqrt(np.mean(audio_chunk**2)))

        # Normalizar al rango [0, 1]
        normalized = (rms - self.rms_min) / max(self.rms_max - self.rms_min, 1e-6)
        normalized = max(0.0, min(1.0, normalized))

        # Smoothing exponencial
        self._current_value = (
            self.alpha * normalized + (1.0 - self.alpha) * self._current_value
        )

        return self._current_value

    def reset(self) -> None:
        """Cierra la boca."""
        self._current_value = 0.0

    @property
    def value(self) -> float:
        """Valor actual de mouth_open."""
        return self._current_value
