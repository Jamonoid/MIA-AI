"""
llm_openrouter.py – Generación de texto vía OpenRouter (API OpenAI-compatible).

OpenRouter es un servicio que unifica acceso a múltiples LLMs
(OpenAI, Anthropic, Mistral, Llama, etc.) bajo una sola API
compatible con OpenAI: https://openrouter.ai/api/v1

Requiere una API key (gratis para modelos seleccionados).
La key se lee de config o de la variable de entorno OPENROUTER_API_KEY.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LLMConfig, PromptConfig

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLM:
    """Generador de texto usando OpenRouter (API OpenAI-compatible)."""

    def __init__(self, config: LLMConfig, prompt_config: PromptConfig) -> None:
        self.base_url = config.base_url or OPENROUTER_BASE_URL
        self.model = config.model_name
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.system_prompt = prompt_config.system
        self._client = None

    def load(self) -> None:
        """Inicializa el cliente OpenAI apuntando a OpenRouter."""
        if not self.api_key:
            raise ValueError(
                "OpenRouter requiere API key. "
                "Configúrala en config.yaml (models.llm.api_key) "
                "o en la variable de entorno OPENROUTER_API_KEY"
            )

        try:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/Jamonoid/MIA-AI",
                    "X-Title": "MIA – Minimalist Interactive Avatar",
                },
            )

            logger.info(
                "OpenRouter conectado – modelo: %s, url: %s",
                self.model,
                self.base_url,
            )

        except ImportError:
            logger.error(
                "openai no instalado. Instalar con: pip install openai"
            )
            raise

    def _build_messages(
        self,
        user_message: str,
        rag_context: str = "",
        chat_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Construye la lista de mensajes para la API de chat."""
        messages: list[dict[str, str]] = []

        # System message (con RAG context si hay)
        system_content = self.system_prompt
        if rag_context:
            system_content += f"\n\n{rag_context}"
        messages.append({"role": "system", "content": system_content})

        # Chat history (últimos N turnos)
        if chat_history:
            for msg in chat_history[-6:]:  # Máximo 3 pares
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # Mensaje actual
        messages.append({"role": "user", "content": user_message})

        return messages

    def build_prompt(
        self,
        user_message: str,
        rag_context: str = "",
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Construye prompt como string (para compatibilidad con tests)."""
        messages = self._build_messages(user_message, rag_context, chat_history)
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def generate_stream(
        self,
        user_message: str,
        rag_context: str = "",
        chat_history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """
        Genera texto en streaming usando la API de OpenRouter.

        Yields:
            Tokens parciales del LLM.
        """
        if self._client is None:
            raise RuntimeError(
                "Cliente OpenRouter no inicializado. Llamar load() primero."
            )

        messages = self._build_messages(user_message, rag_context, chat_history)

        t0 = time.perf_counter()
        first_token = True

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token:
                        elapsed = (time.perf_counter() - t0) * 1000
                        logger.info("LLM first token: %.1f ms", elapsed)
                        first_token = False
                    yield delta.content

        except Exception as exc:
            logger.error("OpenRouter error: %s", exc)
            yield f"[Error de conexión con OpenRouter: {exc}]"
            return

        elapsed_total = (time.perf_counter() - t0) * 1000
        logger.info("LLM generación completa: %.1f ms", elapsed_total)

    def generate(
        self,
        user_message: str,
        rag_context: str = "",
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Genera respuesta completa (no-streaming). Para testing."""
        return "".join(
            self.generate_stream(user_message, rag_context, chat_history)
        )
