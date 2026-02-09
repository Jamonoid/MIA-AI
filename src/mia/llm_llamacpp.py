"""
llm_llamacpp.py – Generación de texto con llama-cpp-python.

Wrapper de streaming sobre llama.cpp.
Construye prompt con system, contexto RAG y chat history.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LLMConfig, PromptConfig

logger = logging.getLogger(__name__)


class LlamaLLM:
    """Generador de texto usando llama-cpp-python."""

    def __init__(self, config: LLMConfig, prompt_config: PromptConfig) -> None:
        self.model_path = config.path
        self.context_size = config.context_size
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.n_gpu_layers = config.n_gpu_layers
        self.system_prompt = prompt_config.system
        self._model = None

    def load(self) -> None:
        """Carga el modelo GGUF. Llamar una vez al inicio."""
        try:
            from llama_cpp import Llama

            logger.info(
                "Cargando LLM desde %s (ctx=%d, gpu_layers=%d)",
                self.model_path,
                self.context_size,
                self.n_gpu_layers,
            )

            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )

            logger.info("LLM cargado correctamente")
        except ImportError:
            logger.error(
                "llama-cpp-python no instalado. "
                "Instalar con: pip install llama-cpp-python"
            )
            raise

    def build_prompt(
        self,
        user_message: str,
        rag_context: str = "",
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Construye el prompt completo para el LLM.

        Formato ChatML:
        <|im_start|>system
        {system_prompt}

        {rag_context}
        <|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        """
        parts: list[str] = []

        # System message
        system_content = self.system_prompt
        if rag_context:
            system_content += f"\n\n{rag_context}"

        parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")

        # Chat history (últimos N turnos)
        if chat_history:
            for msg in chat_history[-6:]:  # Máximo 3 pares user/assistant
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Mensaje actual del usuario
        parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def generate_stream(
        self,
        user_message: str,
        rag_context: str = "",
        chat_history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """
        Genera texto en streaming, token por token.

        Yields:
            Tokens parciales del LLM.
        """
        if self._model is None:
            raise RuntimeError("Modelo LLM no cargado. Llamar load() primero.")

        prompt = self.build_prompt(user_message, rag_context, chat_history)

        t0 = time.perf_counter()
        first_token = True

        stream = self._model.create_completion(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
            stop=["<|im_end|>", "<|im_start|>"],
        )

        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if first_token:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("LLM first token: %.1f ms", elapsed)
                first_token = False
            yield token

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
