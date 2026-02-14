"""Tests para WebUI, proactive config, clean_llm_output, y web_server."""

from pathlib import Path

import pytest
import yaml


# ──────────────────────────────────────────
# Config: WebUI + Proactive
# ──────────────────────────────────────────


class TestWebUIConfig:
    def test_webui_defaults(self):
        from mia.config import WebUIConfig

        cfg = WebUIConfig()
        assert cfg.enabled is True
        assert cfg.port == 8080

    def test_webui_from_yaml(self, tmp_path):
        config = {
            "webui": {"enabled": False, "port": 9090},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config), encoding="utf-8")

        from mia.config import load_config

        cfg = load_config(path)
        assert cfg.webui.enabled is False
        assert cfg.webui.port == 9090

    def test_webui_missing_uses_defaults(self, tmp_path):
        """Config sin sección webui usa defaults."""
        config = {"prompt": {"system": "Test"}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config), encoding="utf-8")

        from mia.config import load_config

        cfg = load_config(path)
        assert cfg.webui.enabled is True
        assert cfg.webui.port == 8080



class TestFullConfig:
    def test_all_sections_load(self, tmp_path):
        """Todas las secciones se cargan correctamente."""
        config = {
            "prompt": {"system": "Test MIA"},
            "models": {
                "llm": {"context_size": 1024},
                "stt": {"model_size": "tiny"},
                "tts": {"voice_path": "./test.wav"},
            },
            "rag": {"enabled": False},
            "discord": {"group_silence_ms": 2000},
            "webui": {"port": 3000},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config), encoding="utf-8")

        from mia.config import load_config

        cfg = load_config(path)
        assert cfg.prompt.system == "Test MIA"
        assert cfg.llm.context_size == 1024
        assert cfg.discord.group_silence_ms == 2000
        assert cfg.webui.port == 3000


# ──────────────────────────────────────────
# TTS Filter
# ──────────────────────────────────────────


class TestTTSFilter:
    def test_strips_asterisks(self):
        from mia.tts_filter import tts_filter
        assert tts_filter("hola *risas* mundo") == "hola mundo"

    def test_strips_double_asterisks(self):
        from mia.tts_filter import tts_filter
        assert tts_filter("**negrita** y texto") == "y texto"

    def test_strips_parentheses(self):
        from mia.tts_filter import tts_filter
        assert tts_filter("hola (acotación) mundo") == "hola mundo"

    def test_strips_brackets(self):
        from mia.tts_filter import tts_filter
        assert tts_filter("hola [acción] mundo") == "hola mundo"

    def test_strips_angle_brackets(self):
        from mia.tts_filter import tts_filter
        assert tts_filter("hola <meta> mundo") == "hola mundo"

    def test_empty_string(self):
        from mia.tts_filter import tts_filter
        assert tts_filter("") == ""

    def test_clean_text_unchanged(self):
        from mia.tts_filter import tts_filter
        assert tts_filter("hola mundo") == "hola mundo"

    def test_nested_parentheses(self):
        from mia.tts_filter import tts_filter
        assert tts_filter("hola (a (b) c) mundo") == "hola mundo"


# ──────────────────────────────────────────
# _clean_llm_output
# ──────────────────────────────────────────


class TestCleanLLMOutput:
    """Tests para _clean_llm_output (antes _strip_emotion_tags)."""

    @staticmethod
    def _clean(text: str) -> str:
        from mia.discord_bot import MIADiscordBot

        return MIADiscordBot._clean_llm_output(text)

    def test_strips_mia_prefix(self):
        assert self._clean("MIA: hola mundo") == "hola mundo"

    def test_strips_mia_lowercase(self):
        assert self._clean("mia: hola mundo") == "hola mundo"

    def test_strips_mia_mixed_case(self):
        assert self._clean("Mia: hola mundo") == "hola mundo"

    def test_no_prefix_unchanged(self):
        assert self._clean("hola mundo") == "hola mundo"

    def test_empty_string(self):
        assert self._clean("") == ""

    def test_only_mia_prefix(self):
        assert self._clean("MIA:") == ""

    def test_preserves_mia_in_text(self):
        """MIA dentro del texto no se toca."""
        assert self._clean("le dije a MIA que sí") == "le dije a MIA que sí"

    def test_strips_whitespace(self):
        assert self._clean("  MIA:   hola  ") == "hola"

    def test_no_emotion_tags_stripped(self):
        """Emotion tags ya no se stripean (Live2D removed)."""
        result = self._clean("[happy] hola")
        assert "[happy]" in result


# ──────────────────────────────────────────
# WebServer
# ──────────────────────────────────────────


class TestWebServer:
    def test_import(self):
        from mia.web_server import MIAWebServer

        server = MIAWebServer(port=9999)
        assert server.port == 9999

    def test_set_command_handler(self):
        from mia.web_server import MIAWebServer

        server = MIAWebServer()
        handler = lambda cmd, val: None
        server.set_command_handler(handler)
        assert server._command_handler is handler

    def test_set_state_provider(self):
        from mia.web_server import MIAWebServer

        server = MIAWebServer()
        provider = lambda: {"status": "ok"}
        server.set_state_provider(provider)
        assert server._state_provider is provider

    def test_webui_dir_exists(self):
        from mia.web_server import WEBUI_DIR

        assert WEBUI_DIR.is_dir()
        assert (WEBUI_DIR / "index.html").is_file()
        assert (WEBUI_DIR / "style.css").is_file()
        assert (WEBUI_DIR / "app.js").is_file()


# ──────────────────────────────────────────
# Proactive prompt file
# ──────────────────────────────────────────


class TestProactivePrompt:
    def test_prompt_file_exists(self):
        prompt_path = Path("prompts/06_proactive.md")
        assert prompt_path.is_file()

    def test_prompt_has_no_keyword(self):
        """El prompt proactivo debe contener la palabra NO como opción."""
        content = Path("prompts/06_proactive.md").read_text(encoding="utf-8")
        assert "NO" in content

    def test_prompt_no_emotion_references(self):
        """No debe haber referencias a emotion tags en los prompts."""
        for md_file in Path("prompts").glob("*.md"):
            content = md_file.read_text(encoding="utf-8").lower()
            assert "emotion" not in content, f"{md_file.name} contiene 'emotion'"
