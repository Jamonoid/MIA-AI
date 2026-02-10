"""
conversations – Sistema de control de turnos de conversación.

Maneja quién habla cuándo en conversaciones real-time via WebSocket.
Soporta: turnos individuales, interrupciones, TTS paralelo,
y sincronización frontend-backend.
"""

from .conversation_handler import ConversationHandler
from .message_handler import message_handler
from .tts_manager import TTSTaskManager
from .types import BroadcastContext, WebSocketSend

__all__ = [
    "ConversationHandler",
    "message_handler",
    "TTSTaskManager",
    "BroadcastContext",
    "WebSocketSend",
]
