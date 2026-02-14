"""
web_server.py – WebUI server embebido para MIA.

Provee una interfaz web local con:
- Chat en tiempo real vía WebSocket
- Controles: pausa, modo proactivo, sliders, toggles
- Stats en vivo
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from aiohttp import web

logger = logging.getLogger(__name__)

WEBUI_DIR = Path(__file__).parent / "webui"


class MIAWebServer:
    """Servidor web embebido con WebSocket para control en tiempo real."""

    def __init__(self, port: int = 8080) -> None:
        self.port = port
        self._app = web.Application()
        self._ws_clients: list[web.WebSocketResponse] = []
        self._runner: web.AppRunner | None = None
        self._command_handler: Callable[[str, Any], Any] | None = None
        self._state_provider: Callable[[], dict] | None = None

        # Routes
        self._app.router.add_get("/ws", self._ws_handler)
        self._app.router.add_get("/api/state", self._api_state)
        # Static files (serve index.html at root)
        if WEBUI_DIR.is_dir():
            self._app.router.add_get("/", self._serve_index)
            self._app.router.add_static("/", WEBUI_DIR, show_index=False)

    def set_command_handler(
        self, handler: Callable[[str, Any], Any]
    ) -> None:
        """Registra handler para comandos del WebUI."""
        self._command_handler = handler

    def set_state_provider(
        self, provider: Callable[[], dict]
    ) -> None:
        """Registra proveedor de estado actual."""
        self._state_provider = provider

    # ──────────────────────────────────────────
    # HTTP handlers
    # ──────────────────────────────────────────

    async def _serve_index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEBUI_DIR / "index.html")

    async def _api_state(self, request: web.Request) -> web.Response:
        """Devuelve el estado actual como JSON."""
        state = {}
        if self._state_provider:
            state = self._state_provider()
        return web.json_response(state)

    # ──────────────────────────────────────────
    # WebSocket
    # ──────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.append(ws)
        logger.info("WebUI: client connected (%d total)", len(self._ws_clients))

        # Enviar estado inicial
        if self._state_provider:
            state = self._state_provider()
            await ws.send_json({"type": "state", "data": state})

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        cmd = data.get("command", "")
                        value = data.get("value")
                        if cmd and self._command_handler:
                            result = self._command_handler(cmd, value)
                            if asyncio.iscoroutine(result):
                                await result
                    except json.JSONDecodeError:
                        logger.warning("WebUI: invalid JSON from client")
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(
                        "WebUI: ws error %s", ws.exception()
                    )
        finally:
            self._ws_clients.remove(ws)
            logger.info(
                "WebUI: client disconnected (%d remaining)",
                len(self._ws_clients),
            )

        return ws

    async def broadcast(self, event_type: str, data: Any = None) -> None:
        """Envía un evento a todos los clientes WebSocket."""
        if not self._ws_clients:
            return

        message = json.dumps({"type": event_type, "data": data})
        dead: list[web.WebSocketResponse] = []

        for ws in self._ws_clients:
            try:
                await ws.send_str(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._ws_clients.remove(ws)

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    async def start(self) -> None:
        """Inicia el servidor HTTP."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        logger.info("WebUI: http://localhost:%d", self.port)

    async def stop(self) -> None:
        """Detiene el servidor."""
        # Cerrar todos los WebSocket
        for ws in list(self._ws_clients):
            await ws.close()
        self._ws_clients.clear()

        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        logger.info("WebUI: stopped")
