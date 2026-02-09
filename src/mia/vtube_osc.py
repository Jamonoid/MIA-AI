"""
vtube_osc.py – Integración con VTube Studio vía OSC.

Envía parámetros del avatar (mouth_open, blink, emotion)
como mensajes OSC/UDP a VTube Studio.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OSCConfig

logger = logging.getLogger(__name__)


class VTubeOSC:
    """Controlador OSC para VTube Studio."""

    def __init__(self, config: OSCConfig) -> None:
        self.ip = config.ip
        self.port = config.port
        self.mapping = config.mapping
        self._client = None
        self._lock = threading.Lock()

        # Estado actual de los parámetros
        self._state: dict[str, float] = {
            "mouth_open": 0.0,
            "blink": 0.0,
        }

    def connect(self) -> None:
        """Inicializa el cliente OSC UDP."""
        try:
            from pythonosc.udp_client import SimpleUDPClient

            self._client = SimpleUDPClient(self.ip, self.port)
            logger.info("VTube OSC conectado a %s:%d", self.ip, self.port)
        except ImportError:
            logger.error(
                "python-osc no instalado. Instalar con: pip install python-osc"
            )
            raise

    def send_param(self, param_name: str, value: float) -> None:
        """
        Envía un parámetro a VTube Studio.

        Args:
            param_name: nombre interno (ej. "mouth_open")
            value: valor del parámetro
        """
        if self._client is None:
            return

        # Mapear nombre interno al nombre del modelo VTS
        osc_name = self.mapping.get(param_name, param_name)
        osc_address = f"/VMC/Ext/{osc_name}"

        with self._lock:
            self._state[param_name] = value
            try:
                self._client.send_message(osc_address, value)
            except Exception as exc:
                logger.debug("OSC send error: %s", exc)

    def send_mouth(self, value: float) -> None:
        """Atajo: envía mouth_open."""
        self.send_param("mouth_open", value)

    def send_blink(self, value: float) -> None:
        """Atajo: envía blink (0 o 1)."""
        self.send_param("blink", value)

    def send_emotion(self, emotion: str) -> None:
        """
        Envía emoción como parámetros OSC.

        Mapea emociones a combinaciones de parámetros:
        happy, sad, angry, surprised, neutral
        """
        emotion_map: dict[str, dict[str, float]] = {
            "happy": {"ParamBrowLeftY": 0.5, "ParamMouthSmile": 0.8},
            "sad": {"ParamBrowLeftY": -0.3, "ParamMouthSmile": -0.4},
            "angry": {"ParamBrowLeftY": -0.6, "ParamEyeOpenLeft": 1.2},
            "surprised": {"ParamBrowLeftY": 0.8, "ParamEyeOpenLeft": 1.3},
            "neutral": {"ParamBrowLeftY": 0.0, "ParamMouthSmile": 0.0},
        }

        params = emotion_map.get(emotion, emotion_map["neutral"])

        if self._client is None:
            return

        with self._lock:
            for param, val in params.items():
                osc_address = f"/VMC/Ext/{param}"
                try:
                    self._client.send_message(osc_address, val)
                except Exception as exc:
                    logger.debug("OSC emotion send error: %s", exc)

        logger.debug("OSC emotion: %s", emotion)

    def close(self) -> None:
        """Cierra la conexión OSC."""
        self._client = None
        logger.info("VTube OSC desconectado")


class VTubeBlinkController:
    """Controlador automático de parpadeo para VTube Studio."""

    def __init__(self, osc: VTubeOSC, interval: float = 4.0) -> None:
        self.osc = osc
        self.interval = interval
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Inicia el loop de parpadeo automático."""
        self._running = True
        self._thread = threading.Thread(target=self._blink_loop, daemon=True)
        self._thread.start()
        logger.info("Blink controller iniciado (interval=%.1fs)", self.interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _blink_loop(self) -> None:
        import random

        while self._running:
            time.sleep(self.interval + random.uniform(-1.0, 1.0))
            if not self._running:
                break
            # Parpadeo: cerrar y abrir rápidamente
            self.osc.send_blink(1.0)
            time.sleep(0.15)
            self.osc.send_blink(0.0)
