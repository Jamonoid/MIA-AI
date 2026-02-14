"""
discord_sink.py – Custom voice sink para recibir audio per-user de Discord.

Usa un Sink personalizado que recibe PCM en tiempo real via write().
Implementa detección de silencio grupal: cuando TODOS los usuarios
que han hablado se callan por >= group_silence_ms, dispara un callback
con el audio acumulado de cada speaker.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Any, Callable, Coroutine

import numpy as np

try:
    import discord
    from discord.sinks.core import Sink, Filters, default_filters, AudioData
except ImportError:
    raise ImportError(
        "py-cord[voice] is required for Discord integration. "
        "Install with: uv pip install 'py-cord[voice]>=2.6.0'"
    )

logger = logging.getLogger(__name__)

# Tipo del callback: async func(speakers: dict[user_id, SpeakerData])
SpeakerData = dict[str, Any]  # {"name": str, "audio": np.ndarray}
GroupSilenceCallback = Callable[
    [dict[int, SpeakerData]], Coroutine[Any, Any, None]
]


class ContinuousVoiceSink(Sink):
    """Custom Sink that receives PCM audio continuously via write().

    Instead of start/stop recording cycles, this sink stays active
    and accumulates audio per-user. A background task monitors for
    group silence and triggers the callback.

    The write() method is called from py-cord's recv_audio thread,
    so we buffer data and process it from the async monitor task.
    """

    def __init__(
        self,
        *,
        group_silence_ms: int = 1500,
        energy_threshold: float = 0.008,
        on_group_silence: GroupSilenceCallback | None = None,
        on_audio_level: Callable[[float], None] | None = None,
        sample_rate: int = 48000,  # Discord native rate
        target_rate: int = 16000,  # STT rate
        **kwargs: Any,
    ) -> None:
        if "filters" not in kwargs:
            kwargs["filters"] = default_filters
        super().__init__(**kwargs)

        self.group_silence_ms = group_silence_ms
        self._energy_threshold = energy_threshold
        self._on_group_silence = on_group_silence
        self._on_audio_level = on_audio_level
        self._sample_rate = sample_rate
        self._target_rate = target_rate
        self._bot_user_id: int | None = None  # Filtrar audio del bot
        self._last_level_emit: float = 0.0  # Throttle level events

        # Per-user accumulated audio (PCM int16 bytes)
        self._user_buffers: dict[int, bytearray] = {}
        self._user_names: dict[int, str] = {}
        self._user_last_speech: dict[int, float] = {}
        self._active_conversation = False
        self._lock = asyncio.Lock()

        # Monitor task
        self._monitor_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def init(self, vc: Any) -> None:
        """Called by py-cord when recording starts."""
        super().init(vc)
        self._loop = asyncio.get_event_loop()
        # Start the silence monitor
        self._monitor_task = self._loop.create_task(self._silence_monitor())
        logger.info("ContinuousVoiceSink: started (monitoring for silence)")

    def format_audio(self, audio: Any) -> None:
        """No-op: py-cord llama esto en cleanup, pero nosotros manejamos el audio."""
        pass

    @Filters.container
    def write(self, data: bytes, user: int) -> None:
        """Called from recv_audio thread with decoded PCM int16 stereo 48kHz."""
        # Ignorar audio del bot mismo
        if self._bot_user_id and user == self._bot_user_id:
            return

        # Also call parent write to maintain audio_data for cleanup
        if user not in self.audio_data:
            import io
            self.audio_data[user] = AudioData(io.BytesIO())
        self.audio_data[user].write(data)

        # Accumulate in our buffer
        if user not in self._user_buffers:
            self._user_buffers[user] = bytearray()
        self._user_buffers[user].extend(data)

        # Check energy (PCM int16, stereo, 48kHz)
        try:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(samples) == 0:
                return

            rms = float(np.sqrt(np.mean(samples ** 2)))

            # Emit audio level every ~200ms
            if self._on_audio_level:
                now_lvl = time.monotonic()
                if now_lvl - self._last_level_emit >= 0.2:
                    self._last_level_emit = now_lvl
                    try:
                        self._on_audio_level(rms)
                    except Exception:
                        pass

            if rms >= self._energy_threshold:
                now = time.monotonic()
                self._user_last_speech[user] = now
                self._active_conversation = True

                # Resolve username
                if user not in self._user_names and self.vc:
                    member = None
                    # Primero buscar en el voice channel (siempre tiene los miembros actualizados)
                    if self.vc.channel:
                        for m in self.vc.channel.members:
                            if m.id == user:
                                member = m
                                break
                    # Fallback al guild cache
                    if not member:
                        member = self.vc.guild.get_member(user)
                    name = member.display_name if member else f"User_{user}"
                    self._user_names[user] = name

        except Exception:
            pass  # Don't crash the recv_audio thread

    async def _silence_monitor(self) -> None:
        """Async task that checks for group silence periodically."""
        while True:
            try:
                await asyncio.sleep(0.2)  # Check every 200ms

                if not self._active_conversation:
                    continue

                if not self._user_last_speech:
                    continue

                now = time.monotonic()
                all_silent = True
                for uid, last in self._user_last_speech.items():
                    elapsed_ms = (now - last) * 1000
                    if elapsed_ms < self.group_silence_ms:
                        all_silent = False
                        break

                if all_silent and self._user_buffers:
                    # Group silence detected!
                    logger.info(
                        "ContinuousVoiceSink: group silence, %d speakers",
                        len(self._user_buffers),
                    )

                    # Build speaker data from buffers
                    speakers = self._build_speaker_data()

                    # Reset state for next conversation
                    self._user_buffers.clear()
                    self._user_last_speech.clear()
                    self._active_conversation = False

                    # Trigger callback
                    if self._on_group_silence and speakers:
                        try:
                            await self._on_group_silence(speakers)
                        except Exception:
                            logger.exception(
                                "ContinuousVoiceSink: error in callback"
                            )

            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("ContinuousVoiceSink: monitor error")
                await asyncio.sleep(1.0)

    def _build_speaker_data(self) -> dict[int, SpeakerData]:
        """Convert raw PCM buffers to numpy arrays for STT."""
        speakers: dict[int, SpeakerData] = {}
        for user_id, buf in self._user_buffers.items():
            if not buf:
                continue

            try:
                # PCM int16 stereo 48kHz → float32 mono 16kHz
                samples = np.frombuffer(bytes(buf), dtype=np.int16).astype(
                    np.float32
                ) / 32768.0

                # Stereo to mono (2 channels interleaved)
                if len(samples) % 2 == 0:
                    samples = samples.reshape(-1, 2).mean(axis=1)

                # Resample 48kHz → 16kHz
                if self._sample_rate != self._target_rate:
                    duration = len(samples) / self._sample_rate
                    target_len = int(duration * self._target_rate)
                    if target_len > 0:
                        indices = np.linspace(
                            0, len(samples) - 1, target_len
                        )
                        samples = np.interp(
                            indices, np.arange(len(samples)), samples
                        ).astype(np.float32)

                if len(samples) > 0:
                    speakers[user_id] = {
                        "name": self._user_names.get(
                            user_id, f"User_{user_id}"
                        ),
                        "audio": samples,
                    }
            except Exception as e:
                logger.debug("Error processing audio for user %d: %s", user_id, e)

        return speakers

    async def stop(self) -> None:
        """Stop the monitor task."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("ContinuousVoiceSink: stopped")


class GroupVoiceSink:
    """Manages continuous voice recording with group silence detection.

    Uses ContinuousVoiceSink as the Sink implementation and a single
    continuous recording session (no start/stop cycles).
    """

    def __init__(
        self,
        voice_client: discord.VoiceClient,
        group_silence_ms: int = 1500,
        sample_rate: int = 16000,
        energy_threshold: float = 0.008,
        on_group_silence: GroupSilenceCallback | None = None,
        on_audio_level: Callable[[float], None] | None = None,
        bot_user_id: int | None = None,
    ) -> None:
        self._vc = voice_client
        self._sink: ContinuousVoiceSink | None = None
        self._group_silence_ms = group_silence_ms
        self._energy_threshold = energy_threshold
        self._on_group_silence = on_group_silence
        self._on_audio_level = on_audio_level
        self._bot_user_id = bot_user_id
        self._running = False

    async def start(self) -> None:
        """Start continuous recording."""
        if self._running:
            return

        if not self._vc.is_connected():
            logger.warning("GroupVoiceSink: not connected, can't start")
            return

        self._sink = ContinuousVoiceSink(
            group_silence_ms=self._group_silence_ms,
            energy_threshold=self._energy_threshold,
            on_group_silence=self._on_group_silence,
            on_audio_level=self._on_audio_level,
        )
        self._sink._bot_user_id = self._bot_user_id

        # Dummy async callback (py-cord requires it)
        async def _on_stop(sink: ContinuousVoiceSink, *args: Any) -> None:
            logger.debug("Recording stopped by py-cord")

        try:
            self._vc.start_recording(self._sink, _on_stop)
            self._running = True
            logger.info("GroupVoiceSink: started continuous recording")
        except Exception as e:
            logger.error("GroupVoiceSink: failed to start recording: %s", e)

    async def stop(self) -> None:
        """Stop continuous recording."""
        if not self._running:
            return

        self._running = False

        # Stop the monitor task first
        if self._sink:
            await self._sink.stop()

        # Stop py-cord recording
        try:
            if self._vc.recording:
                self._vc.stop_recording()
        except Exception:
            pass

        self._sink = None
        logger.info("GroupVoiceSink: stopped")

    def reset(self) -> None:
        """Reset buffers (called after processing)."""
        if self._sink:
            self._sink._user_buffers.clear()
            self._sink._user_last_speech.clear()
            self._sink._active_conversation = False
