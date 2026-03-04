"""
dave_patch.py – Monkey-patch para py-cord: soporte protocolo DAVE.

Discord exige DAVE (Audio/Video E2EE) desde marzo 2026.
py-cord 2.7.x no lo soporta (gateway v8, sin DAVE).

Este patch aplica 4 correcciones:
  1. Gateway v8 → v9
  2. Identify con max_dave_protocol_version
  3. poll_event con soporte de mensajes BINARY (MLS)
  4. DAVE decrypt integrado en el audio pipeline

Requiere: pip install dave.py

Uso:
    from mia.dave_patch import apply_dave_patch
    apply_dave_patch()
"""

from __future__ import annotations

import logging
import struct
from typing import Any, Optional

logger = logging.getLogger(__name__)

_patched = False

# ── Importar dave.py ──
try:
    import dave as _dave_lib
    DAVE_AVAILABLE = True
    DAVE_MAX_VERSION = _dave_lib.get_max_supported_protocol_version()
except ImportError:
    DAVE_AVAILABLE = False
    DAVE_MAX_VERSION = 0
    logger.warning("dave.py no instalado – DAVE no funcionara")


# ═══════════════════════════════════════════════════════════════
# DaveState – gestiona sesion MLS y decryptors por SSRC
# ═══════════════════════════════════════════════════════════════

class DaveState:
    """Estado DAVE para una conexion de voz."""

    DISABLED_VERSION = 0
    NEW_MLS_GROUP_EPOCH = 1

    def __init__(self, voice_client: Any) -> None:
        self._vc = voice_client
        self._ws_ref: Any = None  # Referencia directa al DiscordVoiceWebSocket
        self._session: Optional[Any] = None
        self._protocol_version: int = 0
        self._decryptors: dict[int, Any] = {}  # ssrc → Decryptor
        self._recognized_users: set[int] = set()
        self._prepared_transitions: dict[int, int] = {}  # tid → version
        self._transient_keys: dict[int, Any] = {}  # version → SignatureKeyPair
        self._ready = False

        # Agregar nuestro propio user_id a reconocidos
        try:
            self._recognized_users.add(voice_client.user.id)
        except Exception:
            pass

    def _get_transient_key(self, version: int) -> Any:
        """Genera o reutiliza un SignatureKeyPair para una version."""
        if version not in self._transient_keys:
            self._transient_keys[version] = _dave_lib.SignatureKeyPair.generate(version)
        return self._transient_keys[version]

    def set_ws(self, ws: Any) -> None:
        """Actualiza la referencia al DiscordVoiceWebSocket."""
        self._ws_ref = ws

    def _get_ws(self) -> Any:
        """Obtiene el ws interno de aiohttp para enviar bytes."""
        if self._ws_ref is not None:
            return self._ws_ref.ws
        try:
            return self._vc.ws.ws
        except Exception:
            return None

    @property
    def active(self) -> bool:
        """True si DAVE esta listo para desencriptar audio."""
        return self._protocol_version > 0 and self._ready

    # ── Lifecycle ──

    async def reinit_state(self, protocol_version: int) -> None:
        """Llamado al recibir SESSION_DESCRIPTION con dave_protocol_version.
        Flujo de disnake: reinit → prepare_epoch(1, version) → key_package."""
        self._protocol_version = protocol_version
        self._ready = False
        logger.info("DAVE: reinit protocol_version=%d", protocol_version)

        if protocol_version == 0:
            self._session = None
            for dec in self._decryptors.values():
                dec.transition_to_passthrough_mode()
            return

        self._session = _dave_lib.Session(
            lambda source, reason: logger.error("DAVE MLS failure: %s - %s", source, reason)
        )
        self._session.set_protocol_version(protocol_version)
        self._decryptors.clear()

        # Disnake: reinit_state llama prepare_epoch(1, version) inmediatamente
        await self.prepare_epoch(self.NEW_MLS_GROUP_EPOCH, protocol_version)

    # ── Audio decrypt ──

    def get_or_create_decryptor(self, ssrc: int) -> Any:
        """Obtiene/crea un Decryptor para un SSRC."""
        if ssrc not in self._decryptors:
            self._decryptors[ssrc] = _dave_lib.Decryptor()
        return self._decryptors[ssrc]

    def dave_decrypt_audio(self, ssrc: int, data: bytes) -> bytes:
        """Desencripta audio DAVE E2EE. Retorna data original si DAVE no esta activo."""
        if not self.active:
            return data
        dec = self.get_or_create_decryptor(ssrc)
        try:
            result = dec.decrypt(data, _dave_lib.Codec.Opus)
            if result is not None:
                return bytes(result)
        except Exception:
            pass
        return data

    # ── MLS handlers (mensajes binarios) ──

    def handle_mls_external_sender(self, data: bytes) -> None:
        """Opcode 25: configura external sender en la sesion MLS."""
        if self._session:
            self._session.set_external_sender(data)
            logger.info("DAVE MLS: external_sender recibido (%d bytes)", len(data))

    async def handle_mls_proposals(self, data: bytes) -> None:
        """Opcode 28: procesa proposals MLS."""
        if not self._session:
            return
        try:
            recognized = {str(u) for u in self._recognized_users}
            result = self._session.process_proposals(data, recognized)
            if result is not None:
                # Enviar commit_welcome como binario (opcode 28)
                ws = self._get_ws()
                if ws:
                    out = struct.pack(">B", 28) + result
                    await ws.send_bytes(out)
                    logger.info("DAVE MLS: proposals procesadas, commit enviado")
        except Exception:
            logger.debug("DAVE MLS: error en proposals", exc_info=True)

    async def handle_mls_welcome(self, transition_id: int, data: bytes) -> None:
        """Opcode 30: procesa welcome MLS."""
        if not self._session:
            return
        try:
            recognized = {str(u) for u in self._recognized_users}
            roster = self._session.process_welcome(data, recognized)

            if roster is None:
                # Welcome invalido — reinicializar
                logger.warning("DAVE MLS: welcome fallido, reinicializando")
                ws_ref = self._ws_ref
                if ws_ref:
                    await ws_ref.send_as_json({"op": 31, "d": {"transition_id": transition_id}})
                version = self._session.get_protocol_version()
                await self.reinit_state(version)
                return

            logger.info("DAVE MLS: welcome procesado (tid=%d, roster=%d users)", transition_id, len(roster))

            # Preparar transicion (equivale a prepare_transition en disnake)
            await self.prepare_transition(transition_id, self._session.get_protocol_version())
        except Exception:
            logger.exception("DAVE MLS: error procesando welcome")

    async def handle_mls_announce_commit(self, transition_id: int, data: bytes) -> None:
        """Opcode 29: announce commit transition — process_commit."""
        if not self._session:
            return
        logger.info("DAVE MLS: announce_commit (tid=%d, %d bytes)", transition_id, len(data))
        try:
            result = self._session.process_commit(data)
            if result is _dave_lib.RejectType.ignored:
                logger.debug("DAVE MLS: commit ignorado")
                return
            elif result is _dave_lib.RejectType.failed:
                logger.warning("DAVE MLS: commit fallido, reinicializando")
                ws_ref = self._ws_ref
                if ws_ref:
                    await ws_ref.send_as_json({"op": 31, "d": {"transition_id": transition_id}})
                await self.reinit_state(self._session.get_protocol_version())
                return
            # Commit exitoso — preparar transicion
            logger.info("DAVE MLS: commit procesado exitosamente")
            await self.prepare_transition(transition_id, self._session.get_protocol_version())
        except Exception:
            logger.exception("DAVE MLS: error procesando commit")

    # ── Opcodes JSON ──

    async def prepare_transition(self, transition_id: int, protocol_version: int) -> None:
        """Prepara una transicion de protocolo."""
        logger.info("DAVE: prepare_transition tid=%d v=%d", transition_id, protocol_version)
        self._prepared_transitions[transition_id] = protocol_version

        if transition_id == 0:
            # Transicion ID 0 se ejecuta inmediatamente
            self.execute_transition(transition_id)
        else:
            # Enviar transition ready
            ws_ref = self._ws_ref
            if ws_ref:
                await ws_ref.send_as_json({"op": 23, "d": {"transition_id": transition_id}})

    def execute_transition(self, transition_id: int) -> None:
        """Ejecuta una transicion preparada."""
        version = self._prepared_transitions.pop(transition_id, None)
        if version is None:
            logger.error("DAVE: transition %d no estaba preparada", transition_id)
            return
        logger.info("DAVE: execute_transition tid=%d v=%d", transition_id, version)

        if version == self.DISABLED_VERSION:
            if self._session:
                self._session.reset()
            self._ready = False
        else:
            # Activar key ratchet para desencriptacion
            self._setup_ratchet(version)

    def _setup_ratchet(self, version: int) -> None:
        """Configura el key ratchet despues de una transicion exitosa."""
        if not self._session or not self._session.has_established_group():
            return
        try:
            user_id = str(self._vc.user.id)
            ratchet = self._session.get_key_ratchet(user_id)
            if ratchet:
                self._ready = True
                logger.info("DAVE: key ratchet activado para desencriptacion")
        except Exception:
            logger.debug("DAVE: error configurando ratchet", exc_info=True)

    async def prepare_epoch(self, epoch: int, protocol_version: int) -> None:
        """Prepara una nueva epoca MLS."""
        logger.info("DAVE: prepare_epoch epoch=%d v=%d", epoch, protocol_version)
        self._protocol_version = protocol_version

        if protocol_version == 0:
            self._session = None
            for dec in self._decryptors.values():
                dec.transition_to_passthrough_mode()
            self._ready = False
            return

        if epoch == self.NEW_MLS_GROUP_EPOCH:
            # Inicializar sesion MLS con channel_id, user_id, y transient key
            channel_id = self._vc.channel.id
            user_id = str(self._vc.user.id)
            transient_key = self._get_transient_key(protocol_version)

            self._session.init(
                protocol_version,
                channel_id,
                user_id,
                transient_key,
            )
            logger.info("DAVE MLS: session.init(v=%d, channel=%d, user=%s)",
                       protocol_version, channel_id, user_id)

            # Generar y enviar key package
            key_package = self._session.get_marshalled_key_package()
            if key_package:
                ws = self._get_ws()
                if ws:
                    out = struct.pack(">B", 26) + key_package
                    await ws.send_bytes(out)
                    logger.info("DAVE MLS: key_package enviado (%d bytes)", len(key_package))
                else:
                    logger.warning("DAVE MLS: ws no disponible para enviar key_package")
            else:
                logger.error("DAVE MLS: key_package es None despues de session.init")

    def add_recognized_user(self, user_id: int) -> None:
        self._recognized_users.add(user_id)
        logger.debug("DAVE: user connect %d (total: %d)", user_id, len(self._recognized_users))

    def remove_recognized_user(self, user_id: int) -> None:
        self._recognized_users.discard(user_id)
        logger.debug("DAVE: user disconnect %d", user_id)


# ═══════════════════════════════════════════════════════════════
# apply_dave_patch() – Los 4 parches
# ═══════════════════════════════════════════════════════════════

def apply_dave_patch() -> None:
    """Aplica los 4 monkey-patches DAVE a py-cord."""
    global _patched
    if _patched:
        return

    import aiohttp
    import asyncio
    import threading
    from discord.gateway import DiscordVoiceWebSocket
    from discord import utils as discord_utils
    from discord.errors import ConnectionClosed
    from discord.voice_client import VoiceClient

    # ────────────────────────────────────────────
    # PARCHE 1: Gateway v9 + inicializar DaveState
    # ────────────────────────────────────────────

    @classmethod
    async def patched_from_client(cls, client, *, resume=False, hook=None):
        """Reemplaza from_client: gateway v9 + DaveState."""
        gateway = f"wss://{client.endpoint}/?v=9"
        http = client._state.http
        socket = await http.ws_connect(gateway, compress=15)
        ws = cls(socket, loop=client.loop, hook=hook)
        ws.gateway = gateway
        ws._connection = client
        ws._max_heartbeat_timeout = 60.0
        ws.thread_id = threading.get_ident()

        # Inicializar DaveState
        if DAVE_AVAILABLE and not hasattr(client, 'dave'):
            client.dave = DaveState(client)
        elif not DAVE_AVAILABLE:
            client.dave = None

        if resume:
            await ws.resume()
        else:
            await ws.identify()

        return ws

    DiscordVoiceWebSocket.from_client = patched_from_client

    # ────────────────────────────────────────────
    # PARCHE 2: Identify con max_dave_protocol_version
    # ────────────────────────────────────────────

    async def patched_identify(self):
        """Identify con DAVE protocol version."""
        state = self._connection
        payload = {
            "op": self.IDENTIFY,
            "d": {
                "server_id": str(state.server_id),
                "user_id": str(state.user.id),
                "session_id": state.session_id,
                "token": state.token,
                "max_dave_protocol_version": DAVE_MAX_VERSION,
            },
        }
        await self.send_as_json(payload)
        logger.info("DAVE: identify (max_version=%d)", DAVE_MAX_VERSION)

    DiscordVoiceWebSocket.identify = patched_identify

    # ────────────────────────────────────────────
    # PARCHE 3: poll_event con BINARY + received_message con DAVE
    # ────────────────────────────────────────────

    _original_received = DiscordVoiceWebSocket.received_message

    async def patched_poll_event(self):
        """poll_event que soporta mensajes BINARY (MLS)."""
        msg = await asyncio.wait_for(self.ws.receive(), timeout=30.0)
        if msg.type is aiohttp.WSMsgType.TEXT:
            await self.received_message(discord_utils._from_json(msg.data))
        elif msg.type is aiohttp.WSMsgType.BINARY:
            await handle_binary_ws(self, msg.data)
        elif msg.type is aiohttp.WSMsgType.ERROR:
            raise ConnectionClosed(self.ws, shard_id=None) from msg.data
        elif msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        ):
            raise ConnectionClosed(self.ws, shard_id=None, code=self._close_code)

    DiscordVoiceWebSocket.poll_event = patched_poll_event

    async def handle_binary_ws(ws, data: bytes):
        """Procesa mensajes binarios del voice gateway.
        Formato recibido: [2 bytes seq BE][1 byte opcode][payload...]
        """
        if len(data) < 3:
            return

        # Actualizar seq
        ws.seq_ack = int.from_bytes(data[0:2], "big", signed=False)

        dave_state = getattr(ws._connection, 'dave', None)
        if dave_state is None:
            return

        op = data[2]
        payload = data[3:]

        if op == 25:  # DAVE_MLS_EXTERNAL_SENDER
            dave_state.handle_mls_external_sender(payload)
        elif op == 28:  # DAVE_MLS_PROPOSALS
            await dave_state.handle_mls_proposals(payload)
        elif op == 29:  # DAVE_MLS_ANNOUNCE_COMMIT_TRANSITION
            if len(payload) >= 2:
                tid = int.from_bytes(payload[0:2], "big", signed=False)
                await dave_state.handle_mls_announce_commit(tid, payload[2:])
        elif op == 30:  # DAVE_MLS_WELCOME
            if len(payload) >= 2:
                tid = int.from_bytes(payload[0:2], "big", signed=False)
                await dave_state.handle_mls_welcome(tid, payload[2:])
        else:
            logger.debug("DAVE binary: opcode=%d (%d bytes)", op, len(payload))

    # received_message parcheado: maneja opcodes DAVE JSON
    async def patched_received(self, msg):
        op = msg.get("op")
        data = msg.get("d")

        # Fijar seq_ack desde el nivel raiz del mensaje (no desde d)
        seq = msg.get("seq")
        if seq is not None:
            self.seq_ack = seq

        dave_state = getattr(self._connection, 'dave', None)

        # SESSION_DESCRIPTION (op 4): inicializar DAVE si hay dave_protocol_version
        if op == 4 and dave_state and data and "dave_protocol_version" in data:
            # Dejar que el original procese (secret_key, mode, etc)
            await _original_received(self, msg)
            # Guardar referencia al ws ANTES de reinit (self es DiscordVoiceWebSocket)
            dave_state.set_ws(self)
            await dave_state.reinit_state(data["dave_protocol_version"])
            return

        # Opcodes DAVE JSON
        if dave_state and data:
            if op == 18:  # CLIENTS_CONNECT
                if "user_ids" in data:
                    for uid in data["user_ids"]:
                        dave_state.add_recognized_user(int(uid))
                return
            elif op == 21:  # DAVE_PREPARE_TRANSITION
                await dave_state.prepare_transition(
                    data["transition_id"], data.get("protocol_version", 0)
                )
                return
            elif op == 22:  # DAVE_EXECUTE_TRANSITION
                dave_state.execute_transition(data["transition_id"])
                return
            elif op == 24:  # DAVE_MLS_PREPARE_EPOCH
                await dave_state.prepare_epoch(
                    data["epoch"], data.get("protocol_version", 0)
                )
                return

        # CLIENT_DISCONNECT (op 13) con DAVE tracking
        if dave_state and op == 13 and data:
            dave_state.remove_recognized_user(int(data.get("user_id", 0)))

        # Pasar al handler original para todos los demas opcodes
        await _original_received(self, msg)

    DiscordVoiceWebSocket.received_message = patched_received

    # ────────────────────────────────────────────
    # PARCHE 4: DAVE decrypt en audio pipeline
    # ────────────────────────────────────────────

    _original_decrypt_aead = VoiceClient._decrypt_aead_xchacha20_poly1305_rtpsize

    def patched_decrypt_aead(self, header, data):
        """Transport decrypt + DAVE E2EE decrypt."""
        # Paso 1: Transport decrypt (original)
        result = _original_decrypt_aead(self, header, data)

        # Paso 2: DAVE E2EE decrypt (si activo)
        dave_state = getattr(self, 'dave', None)
        if dave_state and dave_state.active:
            ssrc = struct.unpack_from(">I", header, 8)[0]
            result = dave_state.dave_decrypt_audio(ssrc, result)

        return result

    VoiceClient._decrypt_aead_xchacha20_poly1305_rtpsize = patched_decrypt_aead

    _patched = True
    logger.info(
        "DAVE patch aplicado: gateway v9 + dave.py (max_version=%d) + audio decrypt",
        DAVE_MAX_VERSION,
    )
