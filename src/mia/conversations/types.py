"""
types.py – Tipos compartidos para el sistema de turnos de conversación.

Define type aliases y dataclasses usados por todos los módulos
del paquete conversations/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
)

# ── Type aliases ──────────────────────────────

# Función para enviar un string por WebSocket a un cliente específico.
WebSocketSend = Callable[[str], Awaitable[None]]

# Función para broadcast a un grupo: (member_uids, data_dict, exclude_uid?)
BroadcastFunc = Callable[[List[str], dict, Optional[str]], Awaitable[None]]


# ── Dataclasses ───────────────────────────────


@dataclass
class BroadcastContext:
    """Lleva info de broadcast para que las utilidades puedan
    enviar mensajes a un grupo sin acoplar dependencias."""

    broadcast_func: Optional[BroadcastFunc] = None
    group_members: Optional[List[str]] = None
    current_client_uid: Optional[str] = None


@dataclass
class ConversationMetadata:
    """Flags opcionales que modifican el comportamiento de un turno."""

    proactive_speak: bool = False
    skip_memory: bool = False
    skip_history: bool = False


@dataclass
class GroupConversationState:
    """Estado mutable de una conversación grupal.

    Se registra en un dict a nivel de clase para que los handlers de
    interrupción puedan buscarlo por group_id sin recibirlo como parámetro.
    """

    _states: ClassVar[Dict[str, "GroupConversationState"]] = {}

    group_id: str
    conversation_history: List[str] = field(default_factory=list)
    memory_index: Dict[str, int] = field(default_factory=dict)
    group_queue: List[str] = field(default_factory=list)
    session_emoji: str = ""
    current_speaker_uid: Optional[str] = None

    def __post_init__(self) -> None:
        GroupConversationState._states[self.group_id] = self

    @classmethod
    def get_state(cls, group_id: str) -> Optional["GroupConversationState"]:
        return cls._states.get(group_id)

    @classmethod
    def remove_state(cls, group_id: str) -> None:
        cls._states.pop(group_id, None)
