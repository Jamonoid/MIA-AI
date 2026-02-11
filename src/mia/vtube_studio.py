"""
vtube_studio.py – Integración con VTube Studio vía WebSocket Plugin API.

Se conecta a la API de plugins de VTube Studio (ws://localhost:8001)
para controlar expresiones faciales, lipsync y parpadeo del modelo Live2D.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import VTubeStudioConfig

logger = logging.getLogger(__name__)


class VTubeStudioClient:
    """Cliente WebSocket para la API de plugins de VTube Studio.

    Uses an asyncio.Lock to serialize WebSocket requests, ensuring
    correct request/response pairing despite concurrent callers
    (blink loop, expression changes, lipsync).
    """

    PLUGIN_NAME = "MIA-AI"
    PLUGIN_DEV = "Jamonoid"

    def __init__(self, config: VTubeStudioConfig) -> None:
        self._config = config
        self._ws: Any = None
        self._token: str | None = None
        self._authenticated = False
        self._connected = False
        self._running = False
        self._request_id = 0
        self._ws_lock = asyncio.Lock()  # Serialize ws send/recv

        # Expression mapping: emotion → expression filename
        self._expression_map: dict[str, str] = dict(config.expressions)
        self._current_expression: str | None = None
        self._available_expressions: set[str] = set()

        # Param names for lipsync/blink
        self._mouth_param = config.mouth_param
        self._eye_l_param = config.eye_l_param
        self._eye_r_param = config.eye_r_param

        # Token persistence
        self._token_file = Path(config.token_file)

        # Blink controller
        self._blink_task: asyncio.Task | None = None

    # ──────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to VTube Studio and authenticate."""
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets no instalado. Instalar con: uv pip install websockets"
            )
            return False

        try:
            self._ws = await websockets.connect(
                self._config.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            self._connected = True
            logger.info("VTS: conectado a %s", self._config.ws_url)
        except Exception as e:
            logger.warning("VTS: no se pudo conectar a %s: %s", self._config.ws_url, e)
            return False

        # Authenticate
        auth_ok = await self._authenticate()
        if not auth_ok:
            return False

        self._running = True
        self._authenticated = True

        # Query available expressions
        await self._query_expressions()

        # Start blink controller
        self._blink_task = asyncio.create_task(self._blink_loop())

        logger.info("VTS: autenticado y listo ✓")
        return True

    async def close(self) -> None:
        """Close connection cleanly."""
        self._running = False

        if self._blink_task:
            self._blink_task.cancel()
            try:
                await self._blink_task
            except asyncio.CancelledError:
                pass
            self._blink_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._connected = False
        self._authenticated = False
        logger.info("VTS: desconectado")

    # ──────────────────────────────────────────
    # Authentication
    # ──────────────────────────────────────────

    async def _authenticate(self) -> bool:
        """Authenticate with VTS using stored or new token."""
        # Try stored token first
        self._token = self._load_token()
        if self._token:
            ok = await self._auth_with_token(self._token)
            if ok:
                return True
            logger.info("VTS: token guardado inválido, solicitando nuevo...")

        # Request new token (triggers VTS popup)
        self._token = await self._request_token()
        if not self._token:
            return False

        # Authenticate with new token
        ok = await self._auth_with_token(self._token)
        if ok:
            self._save_token(self._token)
            return True

        return False

    async def _request_token(self) -> str | None:
        """Request a new auth token (triggers VTS popup)."""
        resp = await self._send_request(
            "AuthenticationTokenRequest",
            {
                "pluginName": self.PLUGIN_NAME,
                "pluginDeveloper": self.PLUGIN_DEV,
            },
        )
        if resp and resp.get("messageType") != "APIError":
            token = resp.get("data", {}).get("authenticationToken")
            if token:
                logger.info("VTS: token recibido ✓")
                return token

        error_msg = resp.get("data", {}).get("message", "Unknown") if resp else "No response"
        logger.warning("VTS: token request failed: %s", error_msg)
        return None

    async def _auth_with_token(self, token: str) -> bool:
        """Authenticate for this session using a token."""
        resp = await self._send_request(
            "AuthenticationRequest",
            {
                "pluginName": self.PLUGIN_NAME,
                "pluginDeveloper": self.PLUGIN_DEV,
                "authenticationToken": token,
            },
        )
        if resp:
            authed = resp.get("data", {}).get("authenticated", False)
            if authed:
                logger.info("VTS: sesión autenticada ✓")
            return authed
        return False

    def _load_token(self) -> str | None:
        if self._token_file.exists():
            try:
                return self._token_file.read_text().strip()
            except Exception:
                pass
        return None

    def _save_token(self, token: str) -> None:
        try:
            self._token_file.write_text(token)
        except Exception as e:
            logger.warning("VTS: no se pudo guardar token: %s", e)

    # ──────────────────────────────────────────
    # Expressions
    # ──────────────────────────────────────────

    async def _query_expressions(self) -> None:
        """Query available expressions from VTS and log them."""
        resp = await self._send_request(
            "ExpressionStateRequest",
            {"details": False, "expressionFile": ""},
        )
        if not resp or resp.get("messageType") == "APIError":
            logger.warning("VTS: could not query expressions")
            return

        expressions = resp.get("data", {}).get("expressions", [])
        self._available_expressions = {e["file"] for e in expressions}

        logger.info(
            "VTS: %d expresiones disponibles: %s",
            len(expressions),
            [e["file"] for e in expressions],
        )

        # Validate our mapping
        for emotion, filename in self._expression_map.items():
            if filename not in self._available_expressions:
                logger.warning(
                    "VTS: expression '%s' → '%s' NOT FOUND in model!",
                    emotion,
                    filename,
                )

    async def set_expression(self, emotion: str) -> None:
        """Activate an expression based on emotion name."""
        if not self._authenticated:
            return

        expression_file = self._expression_map.get(emotion)
        if not expression_file:
            logger.debug("VTS: no mapping for emotion '%s'", emotion)
            return

        # Deactivate current expression (if different and not neutral)
        if self._current_expression and self._current_expression != expression_file:
            resp = await self._send_request(
                "ExpressionActivationRequest",
                {
                    "expressionFile": self._current_expression,
                    "active": False,
                    "fadeTime": 0.3,
                },
            )
            if resp and resp.get("messageType") == "APIError":
                err = resp.get("data", {}).get("message", "")
                logger.warning("VTS: deactivate error: %s", err)

        # Activate new expression
        if expression_file != self._current_expression:
            resp = await self._send_request(
                "ExpressionActivationRequest",
                {
                    "expressionFile": expression_file,
                    "active": True,
                    "fadeTime": 0.5,
                },
            )
            if resp and resp.get("messageType") == "APIError":
                err = resp.get("data", {}).get("message", "")
                logger.warning(
                    "VTS: activate '%s' error: %s", expression_file, err
                )
            else:
                self._current_expression = expression_file
                logger.info("VTS: expression → %s (%s)", emotion, expression_file)

    # ──────────────────────────────────────────
    # Lipsync (parameter injection)
    # ──────────────────────────────────────────

    def send_mouth(self, value: float) -> None:
        """Send mouth open value (0.0-1.0). Fire-and-forget async."""
        if not self._authenticated or not self._ws:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._inject_param_fire_forget(
                    self._mouth_param, value
                ))
        except RuntimeError:
            pass

    async def _inject_param_fire_forget(self, param_id: str, value: float) -> None:
        """Inject a param value - fire and forget (don't wait for response)."""
        if not self._ws:
            return
        self._request_id += 1
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"ff-{self._request_id}",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": True,
                "mode": "set",
                "parameterValues": [
                    {"id": param_id, "value": value},
                ],
            },
        }
        try:
            async with self._ws_lock:
                await self._ws.send(json.dumps(payload))
                # Read the response but don't process it
                try:
                    await asyncio.wait_for(self._ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            pass

    async def _inject_params(self, params: list[dict[str, Any]]) -> None:
        """Inject multiple parameter values at once."""
        await self._send_request(
            "InjectParameterDataRequest",
            {
                "faceFound": True,
                "mode": "set",
                "parameterValues": params,
            },
        )

    # ──────────────────────────────────────────
    # Blink controller
    # ──────────────────────────────────────────

    async def _blink_loop(self) -> None:
        """Automatic blink loop using parameter injection."""
        import random

        while self._running:
            try:
                await asyncio.sleep(3.5 + random.uniform(-1.0, 1.5))
                if not self._running or not self._authenticated:
                    continue

                # Close eyes
                await self._inject_params([
                    {"id": self._eye_l_param, "value": 0.0},
                    {"id": self._eye_r_param, "value": 0.0},
                ])
                await asyncio.sleep(0.12)

                # Open eyes
                await self._inject_params([
                    {"id": self._eye_l_param, "value": 1.0},
                    {"id": self._eye_r_param, "value": 1.0},
                ])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("VTS blink error: %s", e)
                await asyncio.sleep(2.0)

    # ──────────────────────────────────────────
    # WebSocket communication
    # ──────────────────────────────────────────

    async def _send_request(
        self, message_type: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Send a request to VTS and wait for response.

        Uses a lock to ensure only one request/response pair at a time.
        """
        if not self._ws:
            return None

        self._request_id += 1
        req_id = f"mia-{self._request_id}"
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": req_id,
            "messageType": message_type,
            "data": data or {},
        }

        try:
            async with self._ws_lock:
                await self._ws.send(json.dumps(payload))
                raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
                resp = json.loads(raw)
                return resp
        except asyncio.TimeoutError:
            logger.warning("VTS: timeout for %s", message_type)
            return None
        except Exception as e:
            logger.warning("VTS: communication error: %s", e)
            self._connected = False
            self._authenticated = False
            return None
