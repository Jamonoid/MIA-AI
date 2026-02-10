"""
conversation_utils.py – Helpers compartidos para flujos de conversación.

Funciones stateless usadas tanto por single_conversation como
por futuras implementaciones de group_conversation.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .message_handler import message_handler
from .tts_manager import TTSTaskManager
from .types import WebSocketSend

logger = logging.getLogger(__name__)


async def send_conversation_start_signals(
    websocket_send: WebSocketSend,
) -> None:
    """Envía señales de inicio de turno al frontend.

    1. conversation-chain-start → frontend muestra indicador de thinking
    2. Thinking... → texto placeholder
    """
    await websocket_send(
        json.dumps({
            "type": "control",
            "action": "conversation-chain-start",
        })
    )
    await websocket_send(
        json.dumps({
            "type": "full-text",
            "text": "Thinking...",
        })
    )


async def send_user_transcription(
    websocket_send: WebSocketSend,
    text: str,
) -> None:
    """Envía la transcripción del usuario al frontend."""
    await websocket_send(
        json.dumps({
            "type": "user-input-transcription",
            "text": text,
        })
    )


async def finalize_conversation_turn(
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    client_uid: str,
    timeout: Optional[float] = 30.0,
) -> None:
    """Finaliza un turno de conversación.

    1. Envía backend-synth-complete (via tts_manager.finish)
    2. Espera frontend-playback-complete (bloquea hasta confirmación)
    3. Envía force-new-message + conversation-chain-end
    """
    # Señalar que no hay más audio
    await tts_manager.finish(websocket_send)

    # Esperar a que el frontend confirme que terminó de reproducir
    logger.debug("Esperando frontend-playback-complete de %s...", client_uid)
    response = await message_handler.wait_for_response(
        client_uid,
        "frontend-playback-complete",
        timeout=timeout,
    )

    if response is None:
        logger.warning(
            "Timeout o desconexión esperando playback-complete de %s",
            client_uid,
        )

    # Señales de fin de turno
    await websocket_send(
        json.dumps({"type": "force-new-message"})
    )
    await websocket_send(
        json.dumps({
            "type": "control",
            "action": "conversation-chain-end",
        })
    )


def cleanup_conversation(tts_manager: TTSTaskManager) -> None:
    """Limpia recursos de una conversación.

    Llamar SIEMPRE en finally blocks, incluso tras errores o cancelaciones.
    """
    tts_manager.clear()
    logger.debug("Recursos de conversación limpiados")
