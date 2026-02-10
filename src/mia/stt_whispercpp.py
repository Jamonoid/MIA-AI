"""
stt_whispercpp.py – Speech-to-Text con faster-whisper.

Wrapper asíncrono sobre faster-whisper (CTranslate2).
Recibe audio completo (post-VAD) y retorna transcripción.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import STTConfig

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Transcriptor de voz usando faster-whisper."""

    def __init__(self, config: STTConfig) -> None:
        self.model_size = config.model_size
        self.language = config.language
        self.device = config.device
        self.compute_type = config.compute_type
        self._model = None

    def load(self) -> None:
        """Carga el modelo de Whisper. Llamar una vez al inicio."""
        try:
            from faster_whisper import WhisperModel

            device = self.device
            compute = self.compute_type

            if device == "auto":
                # Usar CTranslate2 nativo (no PyTorch) para detectar CUDA
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
                else:
                    device = "cpu"
                    # int8 no soportado en CPU sin extensiones AVX
                    if compute in ("int8", "int8_float16"):
                        compute = "float32"
                        logger.info("CPU detectada, cambiando compute_type a float32")

            logger.info(
                "Cargando Whisper model=%s, device=%s, compute=%s",
                self.model_size,
                device,
                compute,
            )

            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute,
            )

            logger.info("Whisper STT cargado correctamente")
        except ImportError:
            logger.error(
                "faster-whisper no instalado. "
                "Instalar con: pip install faster-whisper"
            )
            raise

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio a texto.

        Args:
            audio: array float32, mono, normalizado [-1, 1]
            sample_rate: sample rate del audio (debe ser 16000 para Whisper)

        Returns:
            Texto transcrito
        """
        if self._model is None:
            raise RuntimeError("Modelo STT no cargado. Llamar load() primero.")

        t0 = time.perf_counter()

        # Whisper espera float32, mono, 16kHz
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,  # Greedy para mínima latencia
            best_of=1,
            vad_filter=False,  # Ya hicimos VAD nosotros
            without_timestamps=True,
        )

        # Concatenar todos los segmentos
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("STT: '%.60s...' (%.1f ms)", text, elapsed_ms)

        return text
