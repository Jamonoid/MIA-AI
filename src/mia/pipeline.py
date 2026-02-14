"""
pipeline.py – Pipeline simplificado de MIA (Discord-only).

Carga los módulos necesarios (STT, LLM, TTS, RAG) e inicia el bot
de Discord como única interfaz de interacción.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from .config import MIAConfig

logger = logging.getLogger(__name__)


class MIAPipeline:
    """Pipeline principal de MIA – Discord-only."""

    def __init__(self, config: MIAConfig) -> None:
        self.config = config
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Módulos (se cargan en load())
        self._stt: Any = None
        self._llm: Any = None
        self._tts: Any = None
        self._rag: Any = None
        self._discord_bot: Any = None
        self._discord_token: str = ""

        self._chat_history: list[dict[str, str]] = []

    # ──────────────────────────────────────────
    # Carga de módulos
    # ──────────────────────────────────────────

    def load(self) -> None:
        """Carga todos los módulos necesarios."""
        logger.info("═══ Cargando módulos MIA ═══")

        # Prompts modulares
        self._load_modular_prompt()

        # ── STT ──
        backend = self.config.stt.backend
        if backend in ("faster-whisper", "whisper.cpp"):
            from .stt_whispercpp import WhisperSTT
            self._stt = WhisperSTT(self.config.stt)
            self._stt.load()
            logger.info("STT: faster-whisper ✓")
        else:
            raise ValueError(f"STT backend no soportado: {backend}")

        # ── LLM ──
        llm_backend = self.config.llm.backend
        if llm_backend == "openrouter":
            from .llm_openrouter import OpenRouterLLM
            self._llm = OpenRouterLLM(self.config.llm, self.config.prompt)
            self._llm.load()
            logger.info("LLM: OpenRouter ✓")
        elif llm_backend == "lmstudio":
            from .llm_lmstudio import LMStudioLLM
            self._llm = LMStudioLLM(self.config.llm, self.config.prompt)
            self._llm.load()
            logger.info("LLM: LM Studio ✓")
        elif llm_backend == "llamacpp":
            from .llm_llamacpp import LlamaLLM
            self._llm = LlamaLLM(self.config.llm, self.config.prompt)
            self._llm.load()
            logger.info("LLM: llama.cpp ✓")
        else:
            raise ValueError(f"LLM backend no soportado: {llm_backend}")

        # ── TTS ──
        tts_backend = self.config.tts.backend
        if tts_backend == "edge":
            from .tts_edge import EdgeTTS
            self._tts = EdgeTTS(self.config.tts)
            self._tts.load()
            logger.info("TTS: Edge TTS ✓")
        else:
            raise ValueError(f"TTS backend no soportado: {tts_backend}")

        # ── RAG ──
        if self.config.rag.enabled:
            from .rag_memory import RAGMemory
            self._rag = RAGMemory(self.config.rag)
            logger.info("RAG: ✓")

        # ── Discord token ──
        token = os.getenv("DISCORD_BOT_TOKEN", "")
        if not token:
            logger.error(
                "DISCORD_BOT_TOKEN no encontrado en entorno. "
                "Crea un archivo .env con DISCORD_BOT_TOKEN=tu_token"
            )
            raise RuntimeError("DISCORD_BOT_TOKEN requerido para modo Discord-only")
        self._discord_token = token
        logger.info("Discord token detectado (%d chars) ✓", len(token))

        logger.info("═══ Módulos cargados ═══")

    # ──────────────────────────────────────────
    # Prompts modulares
    # ──────────────────────────────────────────

    def _load_modular_prompt(self) -> None:
        """Carga archivos .md de la carpeta de prompts y arma el system prompt."""
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

    # ──────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────

    async def run(self) -> None:
        """Inicia el bot de Discord y espera."""
        self._running = True

        logger.info("Discord: creando bot...")
        try:
            from .discord_bot import MIADiscordBot

            self._discord_bot = MIADiscordBot(
                stt=self._stt,
                tts=self._tts,
                llm=self._llm,
                rag=self._rag,
                executor=self._executor,
                chat_history=self._chat_history,
                config=self.config,
            )
            logger.info("Discord bot inicializado ✓")
        except ImportError as exc:
            logger.error(
                "No se pudo importar discord_bot: %s. "
                "Instala py-cord: uv pip install 'py-cord[voice]>=2.6.0'",
                exc,
            )
            return

        try:
            logger.info("🎤 MIA Discord-only – iniciando bot...")
            await self._discord_bot.start(self._discord_token)
        except Exception as exc:
            logger.error(
                "═══ DISCORD BOT ERROR ═══\n"
                "  Tipo: %s\n"
                "  Detalle: %s\n"
                "  ¿Token válido? Verifica DISCORD_BOT_TOKEN en .env",
                type(exc).__name__, exc,
            )

    # ──────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────

    async def shutdown(self) -> None:
        """Cierra todo limpiamente."""
        logger.info("Apagando MIA...")
        self._running = False

        if self._discord_bot:
            try:
                await self._discord_bot.close()
            except Exception:
                pass

        self._executor.shutdown(wait=False)
        logger.info("MIA apagada correctamente ✓")
