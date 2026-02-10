"""
message_handler.py – Sincronización frontend ↔ backend sobre WebSocket.

Implementa un patrón request-response usando asyncio.Event:
el backend puede esperar a que el frontend envíe un mensaje
específico (ej. "frontend-playback-complete") sin polling.

Uso típico:
    # Backend — bloquea hasta que el frontend confirme
    await message_handler.wait_for_response(
        client_uid, "frontend-playback-complete"
    )

    # Cuando llega el mensaje del frontend
    message_handler.handle_message(client_uid, {"type": "frontend-playback-complete"})
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Tipo clave para identificar una respuesta esperada: (response_type, request_id)
_ResponseKey = Tuple[str, Optional[str]]


class MessageHandler:
    """Sincronizador de mensajes request-response sobre WebSocket.

    Permite al backend registrar un 'wait' por un tipo de mensaje
    específico del frontend, y desbloquearse cuando llega.
    """

    def __init__(self) -> None:
        # client_uid → {(response_type, request_id) → Event}
        self._response_events: Dict[str, Dict[_ResponseKey, asyncio.Event]] = (
            defaultdict(dict)
        )
        # client_uid → {(response_type, request_id) → message_data}
        self._response_data: Dict[str, Dict[_ResponseKey, Any]] = defaultdict(
            dict
        )

    async def wait_for_response(
        self,
        client_uid: str,
        response_type: str,
        request_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """Bloquea hasta recibir un mensaje del tipo esperado.

        Args:
            client_uid: ID del cliente WebSocket.
            response_type: Tipo de mensaje esperado (ej. "frontend-playback-complete").
            request_id: ID opcional para correlacionar request/response.
            timeout: Segundos máximo de espera. None = sin límite.

        Returns:
            El mensaje recibido, o None si hay timeout o el cliente se desconecta.
        """
        event = asyncio.Event()
        key: _ResponseKey = (response_type, request_id)
        self._response_events[client_uid][key] = event

        logger.debug(
            "Esperando respuesta '%s' de %s (timeout=%s)",
            response_type,
            client_uid,
            timeout,
        )

        try:
            if timeout is not None:
                await asyncio.wait_for(event.wait(), timeout)
            else:
                await event.wait()
            return self._response_data[client_uid].pop(key, None)
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout esperando '%s' de %s", response_type, client_uid
            )
            return None
        finally:
            self._response_events[client_uid].pop(key, None)

    def handle_message(self, client_uid: str, message: dict) -> bool:
        """Procesa un mensaje entrante y desbloquea waiters si corresponde.

        Args:
            client_uid: ID del cliente que envió el mensaje.
            message: Mensaje JSON parseado.

        Returns:
            True si el mensaje desbloqueó un waiter, False si no había nadie esperando.
        """
        msg_type = message.get("type")
        if not msg_type:
            return False

        request_id = message.get("request_id")
        key: _ResponseKey = (msg_type, request_id)

        if (
            client_uid in self._response_events
            and key in self._response_events[client_uid]
        ):
            self._response_data[client_uid][key] = message
            self._response_events[client_uid][key].set()
            logger.debug(
                "Respuesta '%s' recibida de %s — waiter desbloqueado",
                msg_type,
                client_uid,
            )
            return True

        return False

    def cleanup_client(self, client_uid: str) -> None:
        """Limpia todos los eventos pendientes de un cliente.

        Desbloquea cualquier waiter pendiente (retornará None)
        para evitar coroutines colgadas cuando un cliente se desconecta.
        """
        if client_uid in self._response_events:
            pending = len(self._response_events[client_uid])
            for event in self._response_events[client_uid].values():
                event.set()  # Desbloquear waiters
            self._response_events.pop(client_uid)
            self._response_data.pop(client_uid, None)
            if pending:
                logger.info(
                    "Cleanup: %d waiters desbloqueados para %s",
                    pending,
                    client_uid,
                )

    @property
    def active_waiters(self) -> int:
        """Número total de waiters activos (para diagnóstico)."""
        return sum(
            len(events) for events in self._response_events.values()
        )


# Singleton — importar esta instancia desde cualquier módulo.
message_handler = MessageHandler()
