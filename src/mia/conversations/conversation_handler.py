"""
conversation_handler.py – Entry point del sistema de turnos.

Recibe triggers del WebSocket y despacha conversaciones como asyncio.Tasks.
Implementa el concurrency guard (un turno por cliente a la vez) y
manejo de interrupciones.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np

from .single_conversation import process_single_conversation
from .types import ConversationMetadata, WebSocketSend

logger = logging.getLogger(__name__)


class ConversationHandler:
    """Orquestador de conversaciones.

    Recibe triggers (text-input, mic-audio-end, ai-speak-signal),
    verifica que no haya una conversación activa para el cliente,
    y despacha la conversación como asyncio.Task.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tts: Any,
        stt: Any = None,
        rag: Any = None,
        ws_server: Any = None,
        osc: Any = None,
        lipsync: Any = None,
        audio_player: Any = None,
        executor: ThreadPoolExecutor,
        chat_history: list[dict[str, str]],
        rag_enabled: bool = False,
        chunk_size: int = 150,
        audio_sample_rate: int = 16000,
        playback_sample_rate: int = 24000,
    ) -> None:
        self._llm = llm
        self._tts = tts
        self._stt = stt
        self._rag = rag
        self._ws_server = ws_server
        self._osc = osc
        self._lipsync = lipsync
        self._audio_player = audio_player
        self._executor = executor
        self._chat_history = chat_history
        self._rag_enabled = rag_enabled
        self._chunk_size = chunk_size
        self._audio_sample_rate = audio_sample_rate
        self._playback_sample_rate = playback_sample_rate

        # Tasks activas: client_uid → asyncio.Task
        self._tasks: Dict[str, asyncio.Task] = {}
        # Respuestas parciales para restaurar tras interrupción
        self._partial_responses: Dict[str, str] = {}

    def is_busy(self, client_uid: str) -> bool:
        """Retorna True si hay una conversación activa para este cliente."""
        task = self._tasks.get(client_uid)
        return task is not None and not task.done()

    async def handle_trigger(
        self,
        msg_type: str,
        data: dict,
        client_uid: str,
        websocket_send: WebSocketSend,
    ) -> None:
        """Procesa un trigger de conversación.

        Args:
            msg_type: Tipo de trigger ("text-input", "mic-audio-end", "ai-speak-signal").
            data: Datos del mensaje.
            client_uid: ID del cliente.
            websocket_send: Función para enviar al cliente.
        """
        # ── 1. Determinar user_input ──
        if msg_type == "ai-speak-signal":
            user_input = "Please say something."
            metadata = ConversationMetadata(
                proactive_speak=True, skip_memory=True, skip_history=True
            )
        elif msg_type == "text-input":
            user_input = data.get("text", "")
            if not user_input.strip():
                logger.debug("text-input vacío, ignorando")
                return
            metadata = ConversationMetadata()
        elif msg_type == "mic-audio-end":
            user_input = data.get("audio_data")
            if user_input is None:
                logger.warning("mic-audio-end sin audio_data")
                return
            metadata = ConversationMetadata()
        else:
            logger.debug("Tipo de trigger desconocido: %s", msg_type)
            return

        # ── 2. Concurrency guard ──
        if self.is_busy(client_uid):
            logger.info(
                "Conversación en curso para %s, trigger '%s' ignorado",
                client_uid,
                msg_type,
            )
            await websocket_send(
                json.dumps({
                    "type": "error",
                    "message": "Ya hay una conversación en curso.",
                })
            )
            return

        # ── 3. Despachar como Task ──
        logger.info(
            "Iniciando conversación para %s (trigger=%s)",
            client_uid,
            msg_type,
        )

        task = asyncio.create_task(
            process_single_conversation(
                client_uid=client_uid,
                user_input=user_input,
                websocket_send=websocket_send,
                llm=self._llm,
                tts=self._tts,
                stt=self._stt,
                rag=self._rag,
                chat_history=self._chat_history,
                executor=self._executor,
                ws_server=self._ws_server,
                osc=self._osc,
                lipsync=self._lipsync,
                audio_player=self._audio_player,
                metadata=metadata,
                rag_enabled=self._rag_enabled,
                chunk_size=self._chunk_size,
                audio_sample_rate=self._audio_sample_rate,
                playback_sample_rate=self._playback_sample_rate,
            )
        )
        self._tasks[client_uid] = task

        # Callback para limpiar tasks terminadas
        task.add_done_callback(
            lambda t: self._on_task_done(client_uid, t)
        )

    async def handle_interrupt(
        self,
        client_uid: str,
        websocket_send: WebSocketSend,
    ) -> None:
        """Interrumpe la conversación activa de un cliente.

        1. Cancela la task activa
        2. Guarda respuesta parcial en historial
        3. Notifica al frontend
        """
        task = self._tasks.get(client_uid)
        if task is None or task.done():
            logger.debug("No hay conversación activa para %s", client_uid)
            return

        logger.info("Interrumpiendo conversación de %s", client_uid)

        # Cancelar la task
        task.cancel()

        # Esperar a que CancelledError se propague
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        # Guardar respuesta parcial si hay
        partial = self._partial_responses.pop(client_uid, "")
        if partial:
            self._chat_history.append({"role": "assistant", "content": partial})
            self._chat_history.append(
                {"role": "system", "content": "[Interrupted by user]"}
            )

        # Notificar al frontend
        await websocket_send(
            json.dumps({"type": "interrupt-signal"})
        )
        await websocket_send(
            json.dumps({
                "type": "control",
                "action": "conversation-chain-end",
            })
        )

        if self._ws_server:
            await self._ws_server.send_status("listening")

    def _on_task_done(self, client_uid: str, task: asyncio.Task) -> None:
        """Callback cuando una task termina."""
        # Limpiar la referencia
        if self._tasks.get(client_uid) is task:
            del self._tasks[client_uid]

        # Log de errores inesperados
        if not task.cancelled():
            exc = task.exception()
            if exc:
                logger.error(
                    "Conversación de %s terminó con error: %s",
                    client_uid,
                    exc,
                )

    def cleanup_all(self) -> None:
        """Cancela todas las conversaciones activas. Llamar en shutdown."""
        for uid, task in self._tasks.items():
            if not task.done():
                task.cancel()
                logger.info("Conversación de %s cancelada (shutdown)", uid)
        self._tasks.clear()
