"""
ws_server.py – Servidor WebSocket para frontend propio.

Broadcast de estado del avatar y subtítulos a clientes conectados.
Protocolo JSON mínimo.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import WebSocketConfig

logger = logging.getLogger(__name__)


class WSServer:
    """Servidor WebSocket asíncrono para comunicación con frontend."""

    def __init__(self, config: WebSocketConfig) -> None:
        self.host = config.host
        self.port = config.port
        self.enabled = config.enabled
        self._clients: set[Any] = set()
        self._server = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Inicia el servidor WebSocket."""
        if not self.enabled:
            logger.info("WebSocket server desactivado")
            return

        try:
            import websockets

            self._server = await websockets.serve(
                self._handler,
                self.host,
                self.port,
            )
            logger.info("WebSocket server en ws://%s:%d", self.host, self.port)
        except ImportError:
            logger.error("websockets no instalado")
            self.enabled = False

    async def stop(self) -> None:
        """Detiene el servidor."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server detenido")

    async def _handler(self, websocket: Any, path: str = "/") -> None:
        """Maneja una conexión de cliente."""
        self._clients.add(websocket)
        client_info = f"{websocket.remote_address}"
        logger.info("WS cliente conectado: %s", client_info)

        try:
            async for message in websocket:
                # Por ahora solo recibimos pings/keepalives
                logger.debug("WS recibido de %s: %s", client_info, message)
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("WS cliente desconectado: %s", client_info)

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Envía un mensaje JSON a todos los clientes conectados."""
        if not self._clients:
            return

        message = json.dumps(data, ensure_ascii=False)
        disconnected = set()

        for client in self._clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)

        self._clients -= disconnected

    async def send_mouth(self, value: float) -> None:
        """Envía valor de mouth_open a los clientes."""
        await self.broadcast({"type": "mouth", "value": round(value, 3)})

    async def send_emotion(self, emotion: str) -> None:
        """Envía emoción a los clientes."""
        await self.broadcast({"type": "emotion", "value": emotion})

    async def send_subtitle(self, text: str, role: str = "assistant") -> None:
        """Envía subtítulo a los clientes."""
        await self.broadcast({"type": "subtitle", "role": role, "text": text})

    async def send_status(self, status: str) -> None:
        """Envía estado del pipeline a los clientes."""
        await self.broadcast({"type": "status", "value": status})
