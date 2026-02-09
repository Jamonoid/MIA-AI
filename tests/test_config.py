"""Tests de configuración."""

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
        "audio": {"sample_rate": 16000, "channels": 1, "chunk_ms": 20},
        "vad": {"energy_threshold": 0.02, "silence_duration_ms": 500},
        "rag": {"enabled": False},
        "osc": {"ip": "127.0.0.1", "port": 9000},
        "websocket": {"host": "127.0.0.1", "port": 8765, "enabled": False},
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
        assert cfg.audio.sample_rate == 16000

    def test_load_partial_config(self, tmp_path):
        """Config parcial debe rellenar con defaults."""
        path = tmp_path / "partial.yaml"
        path.write_text("prompt:\n  system: 'Parcial'", encoding="utf-8")

        from mia.config import load_config

        cfg = load_config(path)
        assert cfg.prompt.system == "Parcial"
        assert cfg.llm.context_size == 2048  # default

    def test_osc_mapping(self, sample_yaml):
        from mia.config import load_config

        cfg = load_config(sample_yaml)
        assert cfg.osc.ip == "127.0.0.1"
        assert cfg.osc.port == 9000


class TestChunkText:
    def test_short_text_no_split(self):
        from mia.tts_xtts import chunk_text

        result = chunk_text("Hola mundo", 150)
        assert result == ["Hola mundo"]

    def test_split_at_period(self):
        from mia.tts_xtts import chunk_text

        text = "Primera oración. Segunda oración. Tercera oración que es más larga."
        chunks = chunk_text(text, 40)
        assert len(chunks) >= 2
        assert all(len(c) <= 45 for c in chunks)  # Margen por corte

    def test_empty_text(self):
        from mia.tts_xtts import chunk_text

        assert chunk_text("", 100) == []
        assert chunk_text("   ", 100) == []


class TestVAD:
    def test_silence_detection(self):
        import numpy as np

        from mia.config import VADConfig
        from mia.vad import EnergyVAD

        cfg = VADConfig(energy_threshold=0.1)
        vad = EnergyVAD(cfg, sample_rate=16000)

        # Audio silencioso
        silent = np.zeros(320, dtype=np.float32)
        event, audio = vad.process(silent)
        assert event == "silence"
        assert audio is None

    def test_speech_detection(self):
        import numpy as np

        from mia.config import VADConfig
        from mia.vad import EnergyVAD

        cfg = VADConfig(energy_threshold=0.01, min_speech_duration_ms=0)
        vad = EnergyVAD(cfg, sample_rate=16000)

        # Audio con energía
        loud = np.random.randn(320).astype(np.float32) * 0.5
        event, _ = vad.process(loud)
        assert event in ("speech_start", "speaking")


class TestLipsync:
    def test_silence_returns_zero(self):
        import numpy as np

        from mia.config import LipsyncConfig
        from mia.lipsync import RMSLipsync

        cfg = LipsyncConfig()
        lip = RMSLipsync(cfg)

        silent = np.zeros(480, dtype=np.float32)
        val = lip.process(silent)
        assert val == 0.0

    def test_loud_returns_high(self):
        import numpy as np

        from mia.config import LipsyncConfig
        from mia.lipsync import RMSLipsync

        cfg = LipsyncConfig(smoothing_alpha=1.0)  # Sin smoothing
        lip = RMSLipsync(cfg)

        loud = np.ones(480, dtype=np.float32) * 0.5
        val = lip.process(loud)
        assert val > 0.5
