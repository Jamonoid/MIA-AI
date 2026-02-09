"""
pipeline.py – Orquestador principal del pipeline de MIA.

Conecta todos los módulos en un flujo asíncrono:
Mic → VAD → STT → RAG → LLM (stream) → TTS (chunked) → Audio + Lipsync → OSC/WS

Usa ThreadPoolExecutor para operaciones bloqueantes (STT, LLM, TTS)
y asyncio para coordinación.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import MIAConfig

logger = logging.getLogger(__name__)


class MIAPipeline:
    """Pipeline principal de MIA."""

    def __init__(self, config: MIAConfig) -> None:
        self.config = config
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._chat_history: list[dict[str, str]] = []

        # Componentes (se inicializan en load())
        self._audio_capture = None
        self._audio_player = None
        self._vad = None
        self._stt = None
        self._llm = None
        self._tts = None
        self._lipsync = None
        self._osc = None
        self._blink_ctrl = None
        self._ws = None
        self._rag = None

    # ──────────────────────────────────────────
    # Inicialización
    # ──────────────────────────────────────────

    def load(self) -> None:
        """Carga todos los módulos. Llamar antes de run()."""
        logger.info("═══ Cargando módulos MIA ═══")
        t0 = time.perf_counter()

        # Audio I/O
        from .audio_io import AudioCapture, AudioPlayer

        self._audio_capture = AudioCapture(self.config.audio)
        self._audio_player = AudioPlayer(self.config.audio)

        # VAD
        from .vad import EnergyVAD

        self._vad = EnergyVAD(self.config.vad, self.config.audio.sample_rate)

        # Lipsync
        from .lipsync import RMSLipsync

        self._lipsync = RMSLipsync(self.config.lipsync)

        # OSC / VTube Studio
        from .vtube_osc import VTubeBlinkController, VTubeOSC

        self._osc = VTubeOSC(self.config.osc)
        self._osc.connect()
        self._blink_ctrl = VTubeBlinkController(self._osc)

        # WebSocket
        from .ws_server import WSServer

        self._ws = WSServer(self.config.websocket)

        # RAG Memory
        from .rag_memory import RAGMemory

        self._rag = RAGMemory(
            {
                "rag": {
                    "enabled": self.config.rag.enabled,
                    "embedding_model": self.config.rag.embedding_model,
                    "persist_dir": self.config.rag.persist_dir,
                    "top_k": self.config.rag.top_k,
                    "max_docs": self.config.rag.max_docs,
                    "score_threshold": self.config.rag.score_threshold,
                }
            }
        )

        # STT (pesado – cargar modelo)
        from .stt_whispercpp import WhisperSTT

        self._stt = WhisperSTT(self.config.stt)
        self._stt.load()

        # LLM – seleccionar backend
        backend = self.config.llm.backend.lower()
        if backend == "lmstudio":
            from .llm_lmstudio import LMStudioLLM

            self._llm = LMStudioLLM(self.config.llm, self.config.prompt)
            self._llm.load()
            logger.info("LLM backend: LM Studio (%s)", self.config.llm.base_url)
        else:
            from .llm_llamacpp import LlamaLLM

            self._llm = LlamaLLM(self.config.llm, self.config.prompt)
            self._llm.load()
            logger.info("LLM backend: llama-cpp-python")

        # TTS (pesado – cargar modelo)
        from .tts_xtts import XTTS

        self._tts = XTTS(self.config.tts)
        self._tts.load()

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("═══ Módulos cargados en %.1f ms ═══", elapsed)

    # ──────────────────────────────────────────
    # Loop principal
    # ──────────────────────────────────────────

    async def run(self) -> None:
        """Ejecuta el pipeline principal."""
        self._running = True

        # Iniciar servicios de fondo
        self._audio_capture.start()
        self._audio_player.start()
        self._blink_ctrl.start()
        await self._ws.start()

        logger.info("🎤 MIA escuchando... (Ctrl+C para detener)")

        if self._ws.enabled:
            await self._ws.send_status("listening")

        try:
            while self._running:
                await self._listen_and_respond()
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def _listen_and_respond(self) -> None:
        """Un ciclo: escuchar → transcribir → generar → hablar."""
        loop = asyncio.get_event_loop()

        # ── 1. Captura + VAD ──
        audio_data = await self._capture_speech(loop)
        if audio_data is None:
            return

        t_start = time.perf_counter()

        # ── 2. STT ──
        text = await loop.run_in_executor(
            self._executor,
            self._stt.transcribe,
            audio_data,
            self.config.audio.sample_rate,
        )

        if not text or len(text.strip()) < 2:
            logger.debug("STT: texto vacío o muy corto, ignorando")
            return

        stt_ms = (time.perf_counter() - t_start) * 1000
        logger.info("📝 Usuario: %s", text)

        if self._ws.enabled:
            await self._ws.send_subtitle(text, role="user")
            await self._ws.send_status("thinking")

        # ── 3. RAG Retrieval ──
        t_rag = time.perf_counter()
        rag_context = ""
        if self._rag.enabled:
            rag_context = await loop.run_in_executor(
                self._executor,
                self._rag.build_context_block,
                text,
            )
        rag_ms = (time.perf_counter() - t_rag) * 1000

        # ── 4. LLM Streaming → TTS Chunking ──
        await self._generate_and_speak(text, rag_context, loop)

        # ── 5. Métricas ──
        end_to_end_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "📊 Métricas: stt=%.0fms rag=%.0fms total=%.0fms",
            stt_ms,
            rag_ms,
            end_to_end_ms,
        )

    async def _capture_speech(
        self, loop: asyncio.AbstractEventLoop
    ) -> np.ndarray | None:
        """Captura audio hasta detectar fin de habla."""
        while self._running:
            chunk = await loop.run_in_executor(
                None, self._audio_capture.read, 0.05
            )
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            event, audio = self._vad.process(chunk)

            if event == "speech_end" and audio is not None:
                return audio

            # Yield control al event loop
            await asyncio.sleep(0)

        return None

    async def _generate_and_speak(
        self,
        user_text: str,
        rag_context: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Genera respuesta con LLM en streaming y sintetiza/reproduce por chunks."""
        if self._ws.enabled:
            await self._ws.send_status("speaking")

        # Generar texto completo primero (el streaming real va token a token)
        full_response = ""
        text_buffer = ""
        chunk_size = self.config.tts.chunk_size

        # Ejecutar generación de LLM en hilo
        def _generate() -> str:
            return "".join(
                self._llm.generate_stream(
                    user_text, rag_context, self._chat_history
                )
            )

        full_response = await loop.run_in_executor(self._executor, _generate)
        logger.info("🤖 MIA: %s", full_response)

        if self._ws.enabled:
            await self._ws.send_subtitle(full_response, role="assistant")

        # Sintetizar y reproducir por chunks
        from .tts_xtts import chunk_text

        text_chunks = chunk_text(full_response, chunk_size)

        for tts_chunk_text in text_chunks:
            if not tts_chunk_text.strip():
                continue

            # Sintetizar en hilo
            audio_chunk = await loop.run_in_executor(
                self._executor, self._tts.synthesize, tts_chunk_text
            )

            # Reproducir
            self._audio_player.enqueue(audio_chunk)

            # Lipsync: procesar el audio en sub-chunks para actualizar boca
            await self._process_lipsync(audio_chunk, loop)

        # Esperar a que termine de reproducir
        while self._audio_player.is_playing():
            await asyncio.sleep(0.05)

        # Resetear boca
        self._lipsync.reset()
        self._osc.send_mouth(0.0)
        if self._ws.enabled:
            await self._ws.send_mouth(0.0)

        # ── Actualizar historial y RAG ──
        self._chat_history.append({"role": "user", "content": user_text})
        self._chat_history.append({"role": "assistant", "content": full_response})

        # Mantener historial compacto
        if len(self._chat_history) > 20:
            self._chat_history = self._chat_history[-12:]

        # Ingestar en RAG
        if self._rag.enabled:
            await loop.run_in_executor(
                self._executor,
                self._rag.ingest,
                user_text,
                full_response,
            )

        if self._ws.enabled:
            await self._ws.send_status("listening")

    async def _process_lipsync(
        self, audio: np.ndarray, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Procesa lipsync en sub-chunks del audio generado."""
        # Dividir audio en trozos pequeños para actualización suave
        sub_chunk_size = int(self.config.audio.playback_sample_rate * 0.02)  # 20ms
        num_sub_chunks = max(1, len(audio) // sub_chunk_size)

        for i in range(num_sub_chunks):
            start = i * sub_chunk_size
            end = min(start + sub_chunk_size, len(audio))
            sub = audio[start:end]

            mouth_value = self._lipsync.process(sub)
            self._osc.send_mouth(mouth_value)

            if self._ws.enabled:
                await self._ws.send_mouth(mouth_value)

            # Simular timing real (20ms por sub-chunk)
            await asyncio.sleep(0.02)

    # ──────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────

    async def shutdown(self) -> None:
        """Apaga todos los componentes limpiamente."""
        logger.info("Apagando MIA...")
        self._running = False

        if self._blink_ctrl:
            self._blink_ctrl.stop()
        if self._audio_capture:
            self._audio_capture.stop()
        if self._audio_player:
            self._audio_player.stop()
        if self._osc:
            self._osc.close()
        if self._ws:
            await self._ws.stop()

        self._executor.shutdown(wait=False)
        logger.info("MIA apagada correctamente ✓")
