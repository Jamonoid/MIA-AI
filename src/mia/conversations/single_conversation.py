"""
single_conversation.py – Flujo completo de una conversación individual.

Un turno de conversación humano ↔ AI con 14 pasos:
1. Enviar conversation-chain-start
2. Enviar "Thinking..."
3. Procesar input (ASR si audio, directo si texto)
4. RAG retrieval
5. LLM generate stream
6. Por cada chunk → TTS paralelo
7. Esperar todas las TTS
8. Enviar backend-synth-complete
9. Esperar frontend-playback-complete
10. Enviar force-new-message + conversation-chain-end
11. Guardar en historial + RAG
12. Cleanup
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .conversation_utils import (
    cleanup_conversation,
    finalize_conversation_turn,
    send_conversation_start_signals,
    send_user_transcription,
)
from .tts_manager import TTSTaskManager
from .types import ConversationMetadata, WebSocketSend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def process_single_conversation(
    *,
    client_uid: str,
    user_input: str | np.ndarray,
    websocket_send: WebSocketSend,
    llm: Any,
    tts: Any,
    stt: Any = None,
    rag: Any = None,
    chat_history: list[dict[str, str]],
    executor: ThreadPoolExecutor,
    ws_server: Any = None,
    osc: Any = None,
    lipsync: Any = None,
    audio_player: Any = None,
    metadata: Optional[ConversationMetadata] = None,
    rag_enabled: bool = False,
    chunk_size: int = 150,
    audio_sample_rate: int = 16000,
    playback_sample_rate: int = 24000,
) -> None:
    """Ejecuta un turno completo de conversación individual.

    Args:
        client_uid: ID del cliente WebSocket.
        user_input: Texto del usuario o audio numpy array.
        websocket_send: Función para enviar mensajes al cliente.
        llm: Motor LLM (con .generate_stream()).
        tts: Motor TTS (con .synthesize()).
        stt: Motor STT (con .transcribe()). Requerido si user_input es audio.
        rag: Motor RAG (con .build_context_block(), .ingest()).
        chat_history: Lista mutable de historial de chat.
        executor: ThreadPoolExecutor para operaciones bloqueantes.
        ws_server: WSServer para broadcast de estado/subtítulos.
        osc: VTubeOSC para lipsync.
        lipsync: RMSLipsync para cálculo de mouth values.
        audio_player: AudioPlayer para reproducción local.
        metadata: Flags opcionales (proactive, skip_memory, etc.).
        rag_enabled: Si RAG está habilitado.
        chunk_size: Tamaño de chunk para TTS.
        audio_sample_rate: Sample rate de audio de entrada.
        playback_sample_rate: Sample rate de audio de salida.
    """
    if metadata is None:
        metadata = ConversationMetadata()

    tts_manager = TTSTaskManager(
        tts, sample_rate=getattr(tts, "sample_rate", playback_sample_rate),
        executor=executor,
    )

    try:
        # ── 1-2. Señales de inicio ──
        await send_conversation_start_signals(websocket_send)

        if ws_server:
            await ws_server.send_status("thinking")

        # ── 3. Procesar input (ASR si es audio) ──
        loop = asyncio.get_event_loop()

        if isinstance(user_input, np.ndarray):
            if stt is None:
                raise RuntimeError("STT requerido para input de audio")
            user_text = await loop.run_in_executor(
                executor,
                stt.transcribe,
                user_input,
                audio_sample_rate,
            )
            if not user_text or len(user_text.strip()) < 2:
                logger.debug("STT: texto vacío o muy corto, ignorando")
                return
            await send_user_transcription(websocket_send, user_text)
        else:
            user_text = user_input

        logger.info("📝 Usuario: %s", user_text)

        if ws_server:
            await ws_server.send_subtitle(user_text, role="user")

        # ── 4. RAG Retrieval ──
        rag_context = ""
        if rag and rag_enabled:
            t_rag = time.perf_counter()
            rag_context = await loop.run_in_executor(
                executor,
                rag.build_context_block,
                user_text,
            )
            rag_ms = (time.perf_counter() - t_rag) * 1000
            logger.debug("RAG retrieval: %.1f ms", rag_ms)

        # ── 5-6. LLM Stream → TTS paralelo ──
        if ws_server:
            await ws_server.send_status("speaking")

        full_response = await _generate_with_parallel_tts(
            user_text=user_text,
            rag_context=rag_context,
            chat_history=chat_history,
            llm=llm,
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            executor=executor,
            chunk_size=chunk_size,
        )

        logger.info("🤖 MIA: %s", full_response)

        if ws_server:
            await ws_server.send_subtitle(full_response, role="assistant")

        # ── 7. Esperar todas las TTS ──
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)

        # ── 8-10. Finalizar turno (backend-synth-complete → wait playback → end) ──
        await finalize_conversation_turn(
            websocket_send, tts_manager, client_uid
        )

        # ── 11. Guardar en historial y RAG ──
        if not metadata.skip_history:
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": full_response})

            # Mantener historial compacto
            if len(chat_history) > 20:
                chat_history[:] = chat_history[-12:]

        if not metadata.skip_memory and rag and rag_enabled:
            await loop.run_in_executor(
                executor,
                rag.ingest,
                user_text,
                full_response,
            )

        if ws_server:
            await ws_server.send_status("listening")

    except asyncio.CancelledError:
        logger.info("Conversación cancelada (interrupción) para %s", client_uid)
        raise
    except Exception as e:
        logger.error("Error en conversación: %s", e, exc_info=True)
        try:
            await websocket_send(
                json.dumps({"type": "error", "message": str(e)})
            )
        except Exception:
            pass
        raise
    finally:
        # ── 12. Cleanup SIEMPRE ──
        cleanup_conversation(tts_manager)


async def _generate_with_parallel_tts(
    *,
    user_text: str,
    rag_context: str,
    chat_history: list[dict[str, str]],
    llm: Any,
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    executor: ThreadPoolExecutor,
    chunk_size: int,
) -> str:
    """Genera respuesta LLM y envía chunks a TTS en paralelo.

    Returns:
        Texto completo de la respuesta.
    """
    loop = asyncio.get_event_loop()

    # Importar chunk_text para dividir la respuesta
    from ..tts_xtts import chunk_text

    # Generar texto completo del LLM en hilo
    def _generate() -> str:
        return "".join(
            llm.generate_stream(user_text, rag_context, chat_history)
        )

    full_response = await loop.run_in_executor(executor, _generate)

    if not full_response.strip():
        return full_response

    # Dividir en chunks y lanzar TTS paralelo
    text_chunks = chunk_text(full_response, chunk_size)

    for chunk_text_str in text_chunks:
        if not chunk_text_str.strip():
            continue
        await tts_manager.speak(
            tts_text=chunk_text_str,
            display_text=chunk_text_str,
            websocket_send=websocket_send,
        )

    return full_response
