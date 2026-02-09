"""Tests de construcción de prompt."""

import pytest


class TestPromptBuilding:
    def test_basic_prompt(self):
        from mia.config import LLMConfig, PromptConfig
        from mia.llm_llamacpp import LlamaLLM

        llm = LlamaLLM(LLMConfig(), PromptConfig(system="Soy MIA"))
        prompt = llm.build_prompt("Hola")

        assert "<|im_start|>system" in prompt
        assert "Soy MIA" in prompt
        assert "<|im_start|>user" in prompt
        assert "Hola" in prompt
        assert "<|im_start|>assistant" in prompt

    def test_prompt_with_rag_context(self):
        from mia.config import LLMConfig, PromptConfig
        from mia.llm_llamacpp import LlamaLLM

        llm = LlamaLLM(LLMConfig(), PromptConfig(system="Soy MIA"))
        rag = "## Contexto previo\n[1] conversación anterior"
        prompt = llm.build_prompt("¿Qué hablamos ayer?", rag_context=rag)

        assert "Contexto previo" in prompt
        assert "conversación anterior" in prompt

    def test_prompt_with_history(self):
        from mia.config import LLMConfig, PromptConfig
        from mia.llm_llamacpp import LlamaLLM

        llm = LlamaLLM(LLMConfig(), PromptConfig(system="Test"))
        history = [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "¡Hola!"},
        ]
        prompt = llm.build_prompt("¿Cómo estás?", chat_history=history)

        assert "Hola" in prompt
        assert "¡Hola!" in prompt
        assert "¿Cómo estás?" in prompt

    def test_prompt_trims_long_history(self):
        from mia.config import LLMConfig, PromptConfig
        from mia.llm_llamacpp import LlamaLLM

        llm = LlamaLLM(LLMConfig(), PromptConfig(system="Test"))
        # 20 mensajes – debe recortar a los últimos 6
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(20)
        ]
        prompt = llm.build_prompt("Último", chat_history=history)

        # Los primeros mensajes no deberían estar
        assert "msg0" not in prompt
        assert "msg5" not in prompt
        # Los últimos sí
        assert "msg19" in prompt
