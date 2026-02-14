"""Tests de configuración (Discord-only)."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_yaml(tmp_path):
    """Crea un config.yaml temporal para testing."""
    config = {
        "prompt": {"system": "Test MIA"},
        "models": {
            "llm": {"path": "./models/test.gguf", "context_size": 1024},
            "stt": {"model_size": "tiny", "language": "es"},
            "tts": {"voice_path": "./voices/test.wav", "chunk_size": 100},
        },
        "rag": {"enabled": False},
        "discord": {"enabled": True, "group_silence_ms": 2000},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


class TestConfigLoad:
    def test_load_valid_config(self, sample_yaml):
        from mia.config import load_config

        cfg = load_config(sample_yaml)
        assert cfg.prompt.system == "Test MIA"
        assert cfg.llm.context_size == 1024
        assert cfg.stt.model_size == "tiny"
        assert cfg.stt.language == "es"
        assert cfg.tts.chunk_size == 100
        assert cfg.rag.enabled is False

    def test_load_missing_config_uses_defaults(self, tmp_path):
        from mia.config import load_config

        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.prompt.system is not None
        assert cfg.discord.group_silence_ms == 1500  # default

    def test_load_partial_config(self, tmp_path):
        """Config parcial debe rellenar con defaults."""
        path = tmp_path / "partial.yaml"
        path.write_text("prompt:\n  system: 'Parcial'", encoding="utf-8")

        from mia.config import load_config

        cfg = load_config(path)
        assert cfg.prompt.system == "Parcial"
        assert cfg.llm.context_size == 2048  # default

    def test_discord_config(self, sample_yaml):
        from mia.config import load_config

        cfg = load_config(sample_yaml)
        assert cfg.discord.enabled is True
        assert cfg.discord.group_silence_ms == 2000

    def test_unknown_keys_ignored(self, tmp_path):
        """Keys desconocidas en YAML no deben causar error."""
        config = {
            "prompt": {"system": "Test"},
            "unknown_section": {"key": "value"},
            "discord": {"enabled": True, "unknown_key": 123},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config), encoding="utf-8")

        from mia.config import load_config

        cfg = load_config(path)
        assert cfg.prompt.system == "Test"
        assert cfg.discord.enabled is True
