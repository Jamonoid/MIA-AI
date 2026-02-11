"""
tts_manager.py – TTS paralelo con entrega ordenada.

Genera audio para múltiples oraciones en paralelo, pero envía
los resultados al frontend en el orden correcto usando sequence
numbers y un buffer de reordenamiento.

Flujo:
    LLM Stream: Sentence1 → Sentence2 → Sentence3
                    ↓            ↓            ↓
    TTS Tasks:   Task(seq=0)  Task(seq=1)  Task(seq=2)    ← paralelo
                    ↓            ↓            ↓
    Payload Queue: (payload,0) (payload,2) (payload,1)     ← llegan desordenados
                                  ↓
    Sender Task:  Envía seq=0, bufferea seq=2, espera seq=1, luego envía 1, 2
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np

from .types import WebSocketSend

logger = logging.getLogger(__name__)

# Tipo del motor TTS — debe tener .synthesize(text) -> np.ndarray
TTSEngine = Any


def _audio_to_base64(audio: np.ndarray, sample_rate: int = 24000) -> str:
    """Convierte audio float32 a base64 WAV para enviar por WebSocket."""
    import io
    import struct

    # Convertir float32 a int16
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    raw_bytes = pcm.tobytes()

    # Construir WAV header
    buf = io.BytesIO()
    num_samples = len(pcm)
    data_size = num_samples * 2  # int16 = 2 bytes
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", 1))  # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))  # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw_bytes)

    return base64.b64encode(buf.getvalue()).decode("ascii")


class TTSTaskManager:
    """Gestor de TTS paralelo con entrega ordenada por sequence number.

    Uso:
        tts_mgr = TTSTaskManager(tts_engine, sample_rate=24000)

        # Por cada oración del LLM stream:
        await tts_mgr.speak(sentence, display_text, websocket_send)

        # Esperar a que todas terminen:
        await asyncio.gather(*tts_mgr.task_list)

        # Señalar fin:
        await tts_mgr.finish(websocket_send)
    """

    def __init__(
        self,
        tts_engine: TTSEngine,
        sample_rate: int = 24000,
        executor: Optional[ThreadPoolExecutor] = None,
        on_audio_ready: Optional[Callable[[np.ndarray], Any]] = None,
    ) -> None:
        self.tts_engine = tts_engine
        self.sample_rate = sample_rate
        self._executor = executor or ThreadPoolExecutor(max_workers=3)
        self._on_audio_ready = on_audio_ready

        self.task_list: list[asyncio.Task] = []
        self._payload_queue: asyncio.Queue = asyncio.Queue()
        self._sender_task: Optional[asyncio.Task] = None
        self._sequence_counter: int = 0
        self._next_sequence_to_send: int = 0

    async def speak(
        self,
        tts_text: str,
        display_text: str,
        websocket_send: WebSocketSend,
    ) -> None:
        """Encola una oración para TTS paralelo.

        Args:
            tts_text: Texto a sintetizar (puede diferir del display).
            display_text: Texto a mostrar en el frontend.
            websocket_send: Función async para enviar al WebSocket.
        """
        if not tts_text.strip():
            return

        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # Iniciar sender loop si no está corriendo
        if self._sender_task is None or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # Lanzar TTS en background
        task = asyncio.create_task(
            self._process_tts(tts_text, display_text, current_sequence)
        )
        self.task_list.append(task)

    async def _process_payload_queue(
        self, websocket_send: WebSocketSend
    ) -> None:
        """Sender loop: envía payloads en orden de sequence number.

        Bufferea payloads que llegan fuera de orden y los envía
        cuando el sequence number esperado está disponible.
        """
        buffered: dict[int, dict] = {}

        while True:
            try:
                payload, seq = await self._payload_queue.get()
            except asyncio.CancelledError:
                return

            buffered[seq] = payload

            # Enviar en orden
            while self._next_sequence_to_send in buffered:
                ordered_payload = buffered.pop(self._next_sequence_to_send)
                try:
                    await websocket_send(json.dumps(ordered_payload, ensure_ascii=False))
                except Exception:
                    logger.warning(
                        "Error enviando audio chunk seq=%d",
                        self._next_sequence_to_send,
                    )
                self._next_sequence_to_send += 1

            self._payload_queue.task_done()

    async def _process_tts(
        self,
        tts_text: str,
        display_text: str,
        sequence_number: int,
    ) -> None:
        """Genera audio para una oración y lo pone en la cola."""
        loop = asyncio.get_event_loop()

        try:
            # Ejecutar TTS sincrónico en thread pool
            audio = await loop.run_in_executor(
                self._executor,
                self.tts_engine.synthesize,
                tts_text,
            )

            if audio is None or len(audio) == 0:
                logger.warning(
                    "TTS devolvió audio vacío para seq=%d: '%s...'",
                    sequence_number,
                    tts_text[:40],
                )
                # Poner payload vacío para no romper la secuencia
                await self._payload_queue.put((
                    {
                        "type": "audio-response",
                        "display_text": display_text,
                        "audio": "",
                        "sequence": sequence_number,
                    },
                    sequence_number,
                ))
                return

            # Preparar payload
            audio_b64 = _audio_to_base64(audio, self.sample_rate)
            payload = {
                "type": "audio-response",
                "display_text": display_text,
                "audio": audio_b64,
                "sample_rate": self.sample_rate,
                "sequence": sequence_number,
            }

            await self._payload_queue.put((payload, sequence_number))

            # Lipsync callback (runs after audio is queued for frontend)
            if self._on_audio_ready is not None:
                try:
                    result = self._on_audio_ready(audio)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.debug("on_audio_ready error: %s", e)

            logger.debug(
                "TTS seq=%d listo: '%s...' (%d samples)",
                sequence_number,
                tts_text[:30],
                len(audio),
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "Error en TTS seq=%d: %s", sequence_number, e, exc_info=True
            )
            # Poner payload de error para no romper la secuencia
            await self._payload_queue.put((
                {
                    "type": "audio-response",
                    "display_text": display_text,
                    "audio": "",
                    "error": str(e),
                    "sequence": sequence_number,
                },
                sequence_number,
            ))

    async def finish(self, websocket_send: WebSocketSend) -> None:
        """Espera a que todas las tasks terminen y envía señal de fin.

        Llamar después de `await asyncio.gather(*tts_mgr.task_list)`.
        """
        # Esperar a que la cola se vacíe
        if self._payload_queue.qsize() > 0:
            await self._payload_queue.join()

        # Enviar señal de que no hay más audio
        await websocket_send(
            json.dumps({"type": "backend-synth-complete"})
        )

    def clear(self) -> None:
        """Limpia todo el estado del manager.

        Llamar en finally blocks y después de interrupciones.
        """
        # Cancelar tasks pendientes
        for task in self.task_list:
            if not task.done():
                task.cancel()
        self.task_list.clear()

        # Cancelar sender
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()
        self._sender_task = None

        # Reiniciar contadores
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        self._payload_queue = asyncio.Queue()

        logger.debug("TTSTaskManager limpiado")

    @property
    def pending_tasks(self) -> int:
        """Número de tasks TTS pendientes."""
        return sum(1 for t in self.task_list if not t.done())
