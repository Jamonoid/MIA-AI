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
        self._conversation_handler = None
        self._discord_bot = None
        self._audio_mode = "frontend"  # "frontend" or "discord"

    # ──────────────────────────────────────────
    # Inicialización
    # ──────────────────────────────────────────

    def _load_modular_prompt(self) -> None:
        """Carga archivos .md de la carpeta de prompts y arma el system prompt."""
        from pathlib import Path

        prompt_dir = Path(self.config.prompt.dir)
        if not prompt_dir.is_dir():
            logger.info(
                "Carpeta de prompts no encontrada (%s), usando prompt.system",
                prompt_dir,
            )
            return

        md_files = sorted(prompt_dir.glob("*.md"))
        if not md_files:
            logger.info("Carpeta de prompts vacía, usando prompt.system")
            return

        parts: list[str] = []
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                parts.append(content)

        if parts:
            combined = "\n\n---\n\n".join(parts)
            self.config.prompt.system = combined
            logger.info(
                "Prompt modular: %d archivos cargados (%s)",
                len(parts),
                ", ".join(f.name for f in md_files),
            )
        else:
            logger.info("Archivos de prompt vacíos, usando prompt.system")


    def load(self) -> None:
        """Carga todos los módulos. Llamar antes de run()."""
        logger.info("═══ Cargando módulos MIA ═══")
        t0 = time.perf_counter()

        # ── Prompt modular ──
        self._load_modular_prompt()

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
        elif backend == "openrouter":
            from .llm_openrouter import OpenRouterLLM

            self._llm = OpenRouterLLM(self.config.llm, self.config.prompt)
            self._llm.load()
            logger.info("LLM backend: OpenRouter (%s)", self.config.llm.model_name)
        else:
            from .llm_llamacpp import LlamaLLM

            self._llm = LlamaLLM(self.config.llm, self.config.prompt)
            self._llm.load()
            logger.info("LLM backend: llama-cpp-python")

        # TTS – seleccionar backend
        tts_backend = self.config.tts.backend.lower()
        from .tts_edge import EdgeTTS

        self._tts = EdgeTTS(self.config.tts)
        self._tts.load()
        logger.info(
            "TTS backend: Edge TTS (voice=%s)", self.config.tts.edge_voice
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("═══ Módulos cargados en %.1f ms ═══", elapsed)

        # ── Conversation Handler (requiere LLM, TTS, STT, RAG ya cargados) ──
        from .conversations.conversation_handler import ConversationHandler

        self._conversation_handler = ConversationHandler(
            llm=self._llm,
            tts=self._tts,
            stt=self._stt,
            rag=self._rag,
            ws_server=self._ws,
            osc=self._osc,
            lipsync=self._lipsync,
            audio_player=self._audio_player,
            executor=self._executor,
            chat_history=self._chat_history,
            rag_enabled=self.config.rag.enabled,
            chunk_size=self.config.tts.chunk_size,
            audio_sample_rate=self.config.audio.sample_rate,
            playback_sample_rate=self.config.audio.playback_sample_rate,
        )

        # Registrar con WSServer para despacho de mensajes WebSocket
        self._ws.set_conversation_handler(self._conversation_handler)
        # Registrar callback para cambio de modo de audio
        self._ws._on_audio_mode_change = self.set_audio_mode
        logger.info("ConversationHandler registrado")

        # ── Discord Bot (config check only, bot created in run()) ──
        if self.config.discord.enabled:
            import os
            token = os.getenv("DISCORD_BOT_TOKEN", "")
            if not token:
                logger.warning(
                    "Discord habilitado pero DISCORD_BOT_TOKEN no encontrado. "
                    "Crear archivo .env con DISCORD_BOT_TOKEN=tu_token"
                )
            else:
                self._discord_token = token
                logger.info("Discord bot se creará al iniciar (para usar el event loop correcto)")

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

        # Crear e iniciar Discord bot DENTRO del async context
        # (py-cord almacena asyncio.get_event_loop() en Bot.__init__,
        #  así que debe crearse dentro de asyncio.run() para tener el loop correcto)
        discord_task = None
        if hasattr(self, '_discord_token') and self._discord_token:
            from .discord_bot import MIADiscordBot

            self._discord_bot = MIADiscordBot(
                conversation_handler=self._conversation_handler,
                stt=self._stt,
                tts=self._tts,
                llm=self._llm,
                rag=self._rag,
                audio_player=self._audio_player,
                lipsync=self._lipsync,
                osc=self._osc,
                ws_server=self._ws,
                executor=self._executor,
                chat_history=self._chat_history,
                config=self.config,
            )
            # Re-register audio mode callback now that bot exists
            self._ws._on_audio_mode_change = self.set_audio_mode
            logger.info("Discord bot inicializado ✓")

            discord_task = asyncio.create_task(
                self._discord_bot.start(self._discord_token)
            )
            logger.info("Discord bot iniciando...")

        logger.info("🎤 MIA escuchando... (Ctrl+C para detener)")

        if self._ws.enabled:
            await self._ws.send_status("listening")

        try:
            while self._running:
                await self._listen_and_respond()
        except asyncio.CancelledError:
            pass
        finally:
            if discord_task and not discord_task.done():
                discord_task.cancel()
            await self.shutdown()

    async def set_audio_mode(self, mode: str) -> None:
        """Switch audio mode between 'frontend' and 'discord'.

        - frontend: local mic → STT → LLM → TTS → browser audio
        - discord: Discord voice → STT → LLM → TTS → Discord voice
        """
        if mode not in ("frontend", "discord"):
            logger.warning("Audio mode desconocido: %s", mode)
            return

        old_mode = self._audio_mode
        self._audio_mode = mode
        logger.info("Audio mode: %s → %s", old_mode, mode)

        if mode == "discord":
            # Mute local mic (stop capturing)
            self._audio_capture.muted = True
            # Enable Discord voice receive
            if self._discord_bot:
                await self._discord_bot.enable_voice_receive()
        else:
            # Unmute local mic
            self._audio_capture.muted = False
            # Disable Discord voice receive
            if self._discord_bot:
                await self._discord_bot.disable_voice_receive()

    async def _listen_and_respond(self) -> None:
        """Un ciclo: escuchar → detectar fin de habla → delegar a conversation_handler."""
        loop = asyncio.get_event_loop()

        # ── Skip local mic capture in Discord mode ──
        if self._audio_mode == "discord":
            await asyncio.sleep(0.2)  # Yield, Discord bot handles voice
            return

        # ── 0. Si hay conversación activa, esperar sin capturar audio ──
        if self._conversation_handler and self._ws.enabled:
            # Buscar el primer cliente conectado
            client_uid = None
            for _ws, uid in self._ws._client_uids.items():
                client_uid = uid
                break

            if client_uid and self._conversation_handler.is_busy(client_uid):
                await asyncio.sleep(0.1)  # Yield sin capturar audio
                return

        # ── 1. Captura + VAD ──
        audio_data = await self._capture_speech(loop)
        if audio_data is None:
            return

        # ── 2. Delegar a ConversationHandler ──
        if self._conversation_handler and self._ws.enabled:
            # Buscar el primer cliente conectado para enviar
            client_uid = None
            websocket = None
            for ws, uid in self._ws._client_uids.items():
                client_uid = uid
                websocket = ws
                break

            if client_uid and websocket:
                async def _send(msg: str) -> None:
                    try:
                        await websocket.send(msg)
                    except Exception:
                        pass

                await self._conversation_handler.handle_trigger(
                    "mic-audio-end",
                    {"audio_data": audio_data},
                    client_uid,
                    _send,
                )
                return

        # ── Fallback: pipeline legacy (sin WebSocket) ──
        t_start = time.perf_counter()

        # STT
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

        # RAG Retrieval
        t_rag = time.perf_counter()
        rag_context = ""
        if self._rag.enabled:
            rag_context = await loop.run_in_executor(
                self._executor,
                self._rag.build_context_block,
                text,
            )
        rag_ms = (time.perf_counter() - t_rag) * 1000

        # LLM Streaming → TTS Chunking (legacy secuencial)
        await self._generate_and_speak(text, rag_context, loop)

        # Métricas
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
        from .tts_edge import chunk_text

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

        # Discord
        if self._discord_bot:
            try:
                await self._discord_bot.close()
            except Exception as e:
                logger.debug("Error cerrando Discord bot: %s", e)

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
