"""Tests para el módulo RAG memory."""

import pytest


@pytest.fixture
def sample_config():
    """Config mínima para RAG testing."""
    return {
        "rag": {
            "enabled": True,
            "embedding_model": "all-MiniLM-L6-v2",
            "persist_dir": "./data/test_chroma_db",
            "top_k": 3,
            "max_docs": 100,
            "score_threshold": 0.3,
        }
    }


@pytest.fixture
def disabled_config():
    """Config con RAG desactivado."""
    return {"rag": {"enabled": False}}


class TestRAGMemoryDisabled:
    """Verifica que RAG no hace nada cuando está desactivado."""

    def test_retrieve_returns_empty(self, disabled_config):
        from mia.rag_memory import RAGMemory

        mem = RAGMemory(disabled_config)
        assert mem.retrieve("hola") == []

    def test_build_context_returns_empty(self, disabled_config):
        from mia.rag_memory import RAGMemory

        mem = RAGMemory(disabled_config)
        assert mem.build_context_block("hola") == ""

    def test_ingest_does_not_raise(self, disabled_config):
        from mia.rag_memory import RAGMemory

        mem = RAGMemory(disabled_config)
        mem.ingest("hola", "hola, ¿cómo estás?")  # no debe lanzar error


class TestRAGMemoryEnabled:
    """Verifica ingesta y retrieval con RAG activado."""

    def test_ingest_and_retrieve(self, sample_config, tmp_path):
        from mia.rag_memory import RAGMemory

        sample_config["rag"]["persist_dir"] = str(tmp_path / "chroma")
        mem = RAGMemory(sample_config)

        mem.ingest("¿Cuál es tu color favorito?", "Me gusta el azul.")
        mem.ingest("¿Qué música te gusta?", "Me encanta el jazz.")

        results = mem.retrieve("color favorito")
        assert len(results) > 0
        assert any("azul" in r for r in results)

    def test_context_block_format(self, sample_config, tmp_path):
        from mia.rag_memory import RAGMemory

        sample_config["rag"]["persist_dir"] = str(tmp_path / "chroma")
        mem = RAGMemory(sample_config)

        mem.ingest("Hola", "¡Hola! ¿En qué te ayudo?")
        block = mem.build_context_block("Hola")
        assert "## Contexto previo" in block

    def test_max_docs_enforcement(self, sample_config, tmp_path):
        from mia.rag_memory import RAGMemory

        sample_config["rag"]["persist_dir"] = str(tmp_path / "chroma")
        sample_config["rag"]["max_docs"] = 2
        mem = RAGMemory(sample_config)

        mem.ingest("msg1", "resp1")
        mem.ingest("msg2", "resp2")
        mem.ingest("msg3", "resp3")

        assert mem._collection.count() <= 2
