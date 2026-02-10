"""
ws_server.py – Servidor WebSocket + HTTP para frontend propio.

Broadcast de estado del avatar y subtítulos a clientes conectados.
Protocolo JSON mínimo. Sirve archivos estáticos del WebUI.
Integrado con el sistema de turnos de conversación (conversations/).
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .config import WebSocketConfig
    from .conversations.conversation_handler import ConversationHandler

logger = logging.getLogger(__name__)


class WSServer:
    """Servidor WebSocket asíncrono + HTTP estático para el WebUI."""

    def __init__(self, config: WebSocketConfig) -> None:
        self.host = config.host
        self.port = config.port
        self.enabled = config.enabled
        self.webui_dir = Path(getattr(config, "webui_dir", "./web/"))
        self.webui_port = getattr(config, "webui_port", 8080)
        self._clients: set[Any] = set()
        self._server = None
        self._http_server = None
        self._command_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Conversation system
        self._conversation_handler: Optional[ConversationHandler] = None
        # websocket → client_uid mapping
        self._client_uids: dict[Any, str] = {}
        # client_uid → websocket mapping (para send_to_client)
        self._uid_to_ws: dict[str, Any] = {}

    async def start(self) -> None:
        """Inicia el servidor WebSocket y opcionalmente el HTTP para WebUI."""
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
            return

        # HTTP estático para WebUI
        await self._start_http()

    async def _start_http(self) -> None:
        """Inicia un servidor HTTP simple para servir el WebUI."""
        if not self.webui_dir.is_dir():
            logger.info("WebUI dir no encontrado (%s), HTTP desactivado", self.webui_dir)
            return

        from http import HTTPStatus
        from aiohttp import web

        async def handle_static(request: web.Request) -> web.Response:
            """Sirve archivos estáticos del WebUI."""
            path = request.match_info.get("path", "index.html")
            if not path or path == "/":
                path = "index.html"

            file_path = (self.webui_dir / path).resolve()

            # Seguridad: no salir del directorio
            try:
                file_path.relative_to(self.webui_dir.resolve())
            except ValueError:
                return web.Response(status=HTTPStatus.FORBIDDEN, text="Forbidden")

            if not file_path.is_file():
                # Fallback a index.html para SPA
                file_path = self.webui_dir / "index.html"
                if not file_path.is_file():
                    return web.Response(status=HTTPStatus.NOT_FOUND, text="Not found")

            content_type, _ = mimetypes.guess_type(str(file_path))
            content = file_path.read_bytes()

            return web.Response(
                body=content,
                content_type=content_type or "application/octet-stream",
            )

        app = web.Application()
        app.router.add_get("/", handle_static)
        app.router.add_get("/{path:.*}", handle_static)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.webui_port)
        await site.start()
        self._http_server = runner

        logger.info(
            "WebUI en http://%s:%d",
            self.host,
            self.webui_port,
        )

    async def stop(self) -> None:
        """Detiene ambos servidores."""
        # Cancelar conversaciones activas
        if self._conversation_handler:
            self._conversation_handler.cleanup_all()

        # Limpiar message_handler para desbloquear waiters
        from .conversations.message_handler import message_handler
        for uid in list(self._uid_to_ws.keys()):
            message_handler.cleanup_client(uid)

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server detenido")
        if self._http_server:
            await self._http_server.cleanup()
            logger.info("HTTP server detenido")

    def set_conversation_handler(
        self, handler: ConversationHandler
    ) -> None:
        """Registra el ConversationHandler para despacho de conversaciones."""
        self._conversation_handler = handler

    async def _handler(self, websocket: Any, path: str = "/") -> None:
        """Maneja una conexión de cliente WebSocket."""
        self._clients.add(websocket)
        client_uid = f"client-{uuid.uuid4().hex[:8]}"
        self._client_uids[websocket] = client_uid
        self._uid_to_ws[client_uid] = websocket
        client_info = f"{websocket.remote_address} ({client_uid})"
        logger.info("WS cliente conectado: %s", client_info)

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_command(data, websocket, client_uid)
                except json.JSONDecodeError:
                    logger.debug("WS mensaje no-JSON: %s", message)
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            self._client_uids.pop(websocket, None)
            self._uid_to_ws.pop(client_uid, None)

            # Limpiar waiters del message_handler
            from .conversations.message_handler import message_handler
            message_handler.cleanup_client(client_uid)

            logger.info("WS cliente desconectado: %s", client_info)

    async def _handle_command(
        self, data: dict[str, Any], websocket: Any, client_uid: str
    ) -> None:
        """Procesa mensajes del WebUI.

        Enruta a: conversation_handler (turnos), message_handler (sync),
        o command_queue (comandos del pipeline).
        """
        msg_type = data.get("type")

        # ── Mensajes de sincronización (para message_handler) ──
        if msg_type in ("frontend-playback-complete",):
            from .conversations.message_handler import message_handler
            message_handler.handle_message(client_uid, data)
            return

        # ── Triggers de conversación ──
        if msg_type in ("text-input", "mic-audio-end", "ai-speak-signal"):
            if self._conversation_handler:
                async def _send(msg: str) -> None:
                    try:
                        await websocket.send(msg)
                    except Exception:
                        pass

                await self._conversation_handler.handle_trigger(
                    msg_type, data, client_uid, _send
                )
            else:
                # Fallback: encolar como chat legacy
                if msg_type == "text-input":
                    await self._command_queue.put({
                        "type": "chat",
                        "text": data.get("text", ""),
                    })
            return

        # ── Interrupciones ──
        if msg_type == "interrupt":
            if self._conversation_handler:
                async def _send_interrupt(msg: str) -> None:
                    try:
                        await websocket.send(msg)
                    except Exception:
                        pass

                await self._conversation_handler.handle_interrupt(
                    client_uid, _send_interrupt
                )
            return

        # ── Comandos legacy (command, chat) ──
        if msg_type in ("command", "chat"):
            await self._command_queue.put(data)
            logger.debug("Comando encolado: %s", data)

    async def get_command(self) -> dict[str, Any]:
        """Obtiene el siguiente comando del WebUI (blocking)."""
        return await self._command_queue.get()

    def get_command_nowait(self) -> dict[str, Any] | None:
        """Obtiene un comando sin bloquear, o None."""
        try:
            return self._command_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

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

    async def send_to_client(
        self, client_uid: str, data: dict[str, Any]
    ) -> bool:
        """Envía un mensaje a un cliente específico por su UID.

        Returns:
            True si se envió, False si el cliente no existe.
        """
        ws = self._uid_to_ws.get(client_uid)
        if ws is None:
            return False
        try:
            await ws.send(json.dumps(data, ensure_ascii=False))
            return True
        except Exception:
            return False

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

    async def send_metrics(self, metrics: dict[str, float]) -> None:
        """Envía métricas de latencia a los clientes."""
        await self.broadcast({"type": "metrics", **metrics})

    async def send_log(self, text: str, level: str = "info") -> None:
        """Envía línea de log a los clientes."""
        await self.broadcast({"type": "log", "text": text, "level": level})

    async def send_audio_level(self, value: float) -> None:
        """Envía nivel de audio del micrófono (0-1)."""
        await self.broadcast({"type": "audio_level", "value": round(value, 3)})

    async def send_config_state(self, **kwargs: Any) -> None:
        """Envía estado actual de configuración (muted, paused, rag, vision)."""
        await self.broadcast({"type": "config_state", **kwargs})
