"""
Tests para conversations/tts_manager.py

Verifica que TTSTaskManager genera TTS en paralelo
pero entrega los payloads en orden correcto.
"""

import asyncio
import json
import random

import numpy as np
import pytest

from mia.conversations.tts_manager import TTSTaskManager


class FakeTTSEngine:
    """Motor TTS falso con delays artificiales para probar ordering."""

    def __init__(self, delays: dict[int, float] | None = None):
        self.delays = delays or {}
        self._call_count = 0
        self.sample_rate = 24000

    def synthesize(self, text: str) -> np.ndarray:
        idx = self._call_count
        self._call_count += 1

        # Delay artificial — los chunks con delays más largos terminan después
        import time
        delay = self.delays.get(idx, 0)
        if delay:
            time.sleep(delay)

        # Generar audio falso (sinusoidal corto)
        duration = 0.05  # 50ms
        t = np.linspace(0, duration, int(self.sample_rate * duration), dtype=np.float32)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)


class PayloadCollector:
    """Recolecta payloads enviados para verificar orden."""

    def __init__(self):
        self.payloads: list[dict] = []

    async def send(self, message: str) -> None:
        data = json.loads(message)
        self.payloads.append(data)


@pytest.mark.asyncio
async def test_ordered_delivery_5_tasks():
    """5 TTS tasks con delays que terminan fuera de orden → entrega en orden."""
    # Task 0: lento (0.2s), Task 1: rápido (0s), Task 2: medio (0.1s),
    # Task 3: rápido (0s), Task 4: lento (0.15s)
    engine = FakeTTSEngine(delays={0: 0.2, 1: 0, 2: 0.1, 3: 0, 4: 0.15})
    collector = PayloadCollector()
    mgr = TTSTaskManager(engine, sample_rate=24000)

    # Encolar 5 tasks
    sentences = [f"Sentence {i}" for i in range(5)]
    for i, s in enumerate(sentences):
        await mgr.speak(s, s, collector.send)

    # Esperar a que todas terminen
    await asyncio.gather(*mgr.task_list)
    await mgr.finish(collector.send)

    # Verificar orden
    audio_payloads = [p for p in collector.payloads if p["type"] == "audio-response"]
    assert len(audio_payloads) == 5

    for i, payload in enumerate(audio_payloads):
        assert payload["sequence"] == i, (
            f"Payload en posición {i} tiene sequence={payload['sequence']}"
        )
        assert payload["display_text"] == f"Sentence {i}"

    # Verificar señal de fin
    assert collector.payloads[-1]["type"] == "backend-synth-complete"

    mgr.clear()


@pytest.mark.asyncio
async def test_empty_text_skipped():
    """Texto vacío no genera task."""
    engine = FakeTTSEngine()
    collector = PayloadCollector()
    mgr = TTSTaskManager(engine, sample_rate=24000)

    await mgr.speak("", "", collector.send)
    await mgr.speak("   ", "   ", collector.send)

    assert len(mgr.task_list) == 0
    mgr.clear()


@pytest.mark.asyncio
async def test_clear_resets_state():
    """clear() reinicia contadores y cancela tasks."""
    engine = FakeTTSEngine(delays={0: 5.0})  # Task larga
    collector = PayloadCollector()
    mgr = TTSTaskManager(engine, sample_rate=24000)

    await mgr.speak("Long running", "Long running", collector.send)
    assert len(mgr.task_list) == 1

    mgr.clear()

    assert len(mgr.task_list) == 0
    assert mgr._sequence_counter == 0
    assert mgr._next_sequence_to_send == 0
    assert mgr.pending_tasks == 0


@pytest.mark.asyncio
async def test_single_task():
    """Una sola task funciona correctamente."""
    engine = FakeTTSEngine()
    collector = PayloadCollector()
    mgr = TTSTaskManager(engine, sample_rate=24000)

    await mgr.speak("Hola mundo", "Hola mundo", collector.send)
    await asyncio.gather(*mgr.task_list)
    await mgr.finish(collector.send)

    audio_payloads = [p for p in collector.payloads if p["type"] == "audio-response"]
    assert len(audio_payloads) == 1
    assert audio_payloads[0]["display_text"] == "Hola mundo"
    assert audio_payloads[0]["sequence"] == 0
    assert audio_payloads[0]["audio"] != ""  # No vacío
    assert collector.payloads[-1]["type"] == "backend-synth-complete"

    mgr.clear()


@pytest.mark.asyncio
async def test_audio_payload_has_base64():
    """El payload contiene audio codificado en base64 WAV."""
    import base64

    engine = FakeTTSEngine()
    collector = PayloadCollector()
    mgr = TTSTaskManager(engine, sample_rate=24000)

    await mgr.speak("Test", "Test", collector.send)
    await asyncio.gather(*mgr.task_list)
    await mgr.finish(collector.send)

    audio_payloads = [p for p in collector.payloads if p["type"] == "audio-response"]
    audio_b64 = audio_payloads[0]["audio"]

    # Debe ser base64 válido
    raw = base64.b64decode(audio_b64)
    # Debe ser WAV (empieza con RIFF)
    assert raw[:4] == b"RIFF"
    assert raw[8:12] == b"WAVE"

    mgr.clear()
