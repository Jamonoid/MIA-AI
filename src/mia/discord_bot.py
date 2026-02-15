"""
discord_bot.py – Bot de Discord para MIA (Discord-only).

Gestiona la conexión a Discord, voice channels, y la interacción
con MIA. Soporta:
- Slash commands: /join, /leave, /mia, /mute, /move, /nick, /sound
- Voice receive con detección de silencio grupal
- Text channel responses (cuando la mencionan)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .tts_filter import tts_filter

try:
    import discord
    from discord import FFmpegPCMAudio
    from discord.ext import commands
except ImportError:
    raise ImportError(
        "py-cord[voice] is required. Install: uv pip install 'py-cord[voice]>=2.6.0'"
    )

from .discord_sink import GroupVoiceSink, SpeakerData

logger = logging.getLogger(__name__)

_NAME_PREFIX_RE = re.compile(r"^\s*(?:MIA|Mia|mia)\s*:\s*")


class MIADiscordBot:
    """Bot de Discord para MIA (Discord-only).

    Features:
    - /join, /leave: Voice channel management
    - /mia <texto>: Enviar texto a MIA
    - /mute, /unmute, /move, /nick, /sound: Moderation
    - Responde cuando la mencionan en text channels
    - Voice receive con detección de silencio grupal
    """

    def __init__(
        self,
        *,
        stt: Any,
        tts: Any,
        llm: Any,
        rag: Any = None,
        executor: ThreadPoolExecutor,
        chat_history: list[dict[str, str]],
        config: Any,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._llm = llm
        self._rag = rag
        self._executor = executor
        self._chat_history = chat_history
        self._session_log: list[dict[str, str]] = []  # Full session, never trimmed
        self._config = config

        self._voice_sink: Optional[GroupVoiceSink] = None
        self._voice_client: Optional[discord.VoiceClient] = None
        self._text_responses_enabled = config.discord.text_channel_responses
        self._processing = False  # Guard para evitar respuestas simultáneas
        self._voice_receive_enabled = True  # Siempre activo en Discord-only

        # ── Event bus (for WebUI) ──
        self._event_callbacks: list[Callable] = []
        self._bot_state: str = "listening"  # listening|paused|processing|speaking|proactive

        # ── Proactive mode (defaults; controlled from WebUI) ──
        self._proactive_mode: bool = False
        self._proactive_idle_seconds: int = 30
        self._proactive_prompt: str = ""
        self._proactive_timer_task: Optional[asyncio.Task] = None
        self._last_voice_activity: float = time.monotonic()
        self._load_proactive_prompt()

        # ── STT filters ──
        self._stt_hallucinations: set[str] = {
            h.lower().strip() for h in config.stt.stt_hallucinations
        }
        self._min_energy_rms: float = 0.01  # configurable solo desde WebUI

        # Configurar intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.presences = True
        intents.voice_states = True

        self.bot = commands.Bot(
            command_prefix="!",
            intents=intents,
        )

        self._setup_events()
        self._setup_commands()

    # ──────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────

    @staticmethod
    def _clean_llm_output(text: str) -> str:
        """Limpia el output del LLM: quita prefijo 'MIA:' si aparece."""
        return _NAME_PREFIX_RE.sub("", text).strip()

    def _load_proactive_prompt(self) -> None:
        """Carga el prompt proactivo desde archivo."""
        prompt_dir = Path(getattr(self._config.prompt, "dir", "./prompts/"))
        path = prompt_dir / "06_proactive.md"
        if path.is_file():
            self._proactive_prompt = path.read_text(encoding="utf-8").strip()
            logger.info("Proactive prompt cargado desde %s", path)
        else:
            self._proactive_prompt = (
                "Nadie ha hablado en un rato. Inicia conversación de forma natural."
            )

    # ──────────────────────────────────────────
    # Event bus (WebUI integration)
    # ──────────────────────────────────────────

    def on_event(self, callback: Callable) -> None:
        """Registra un callback para eventos. Firma: callback(event_type, data)"""
        self._event_callbacks.append(callback)

    async def _emit(self, event_type: str, data: Any = None) -> None:
        """Emite un evento a todos los listeners."""
        for cb in self._event_callbacks:
            try:
                result = cb(event_type, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.debug("Event callback error", exc_info=True)

    async def _set_bot_state(self, state: str) -> None:
        """Cambia el estado del bot y notifica."""
        self._bot_state = state
        await self._emit("state_change", {"key": "bot_state", "value": state})

    def get_state(self) -> dict:
        """Devuelve estado actual para WebUI."""
        rag_docs = 0
        if self._rag:
            try:
                rag_docs = self._rag._collection.count() if hasattr(self._rag, "_collection") else 0
            except Exception:
                rag_docs = 0

        vc_members = 0
        if self._voice_client and self._voice_client.channel:
            vc_members = len(self._voice_client.channel.members)

        return {
            "paused": not self._voice_receive_enabled,
            "proactive": self._proactive_mode,
            "text_responses": self._text_responses_enabled,
            "rag_enabled": self._config.rag.enabled,
            "idle_seconds": self._proactive_idle_seconds,
            "silence_ms": self._config.discord.group_silence_ms,
            "min_energy_rms": self._min_energy_rms,
            "bot_state": self._bot_state,
            "stats": {
                "speakers": vc_members,
                "history": len(self._chat_history),
                "rag_docs": rag_docs,
            },
            "chat_history": [
                {
                    "speaker": msg["role"].replace("assistant", "MIA").replace("user", "User"),
                    "text": msg["content"],
                }
                for msg in self._chat_history
            ],
        }

    async def handle_webui_command(self, command: str, value: Any) -> None:
        """Procesa comandos del WebUI."""
        if command == "toggle_pause":
            if value:
                await self.disable_voice_receive()
                await self._set_bot_state("paused")
            else:
                await self.enable_voice_receive()
                await self._set_bot_state("listening")

        elif command == "toggle_proactive":
            self._proactive_mode = bool(value)
            if self._proactive_mode:
                self._last_voice_activity = time.monotonic()  # Fresh start
                self._start_proactive_timer()
            else:
                self._stop_proactive_timer()
            await self._emit("state_change", {"key": "proactive", "value": self._proactive_mode})
            logger.info("Proactive mode: %s", "ON" if self._proactive_mode else "OFF")

        elif command == "toggle_text_responses":
            self._text_responses_enabled = bool(value)
            await self._emit("state_change", {"key": "text_responses", "value": self._text_responses_enabled})

        elif command == "toggle_rag":
            self._config.rag.enabled = bool(value)
            await self._emit("state_change", {"key": "rag_enabled", "value": self._config.rag.enabled})

        elif command == "set_idle_seconds":
            self._proactive_idle_seconds = max(10, min(120, int(value)))
            await self._emit("state_change", {"key": "idle_seconds", "value": self._proactive_idle_seconds})
            logger.info("Proactive idle: %ds", self._proactive_idle_seconds)

        elif command == "set_silence_ms":
            ms = max(500, min(5000, int(value)))
            self._config.discord.group_silence_ms = ms
            if self._voice_sink:
                self._voice_sink._group_silence_ms = ms
                # Also update the running inner sink
                if self._voice_sink._sink:
                    self._voice_sink._sink.group_silence_ms = ms
            await self._emit("state_change", {"key": "silence_ms", "value": ms})

        elif command == "clear_history":
            self._chat_history.clear()
            await self._emit("state_change", {"key": "history_cleared", "value": True})
            logger.info("Chat history cleared")

        elif command == "set_min_energy":
            self._min_energy_rms = max(0.001, min(0.05, float(value)))
            await self._emit("state_change", {"key": "min_energy_rms", "value": self._min_energy_rms})
            logger.info("Min energy RMS: %.3f", self._min_energy_rms)

        elif command == "force_speak":
            if not self._processing and self._voice_client and self._voice_client.is_connected():
                logger.info("WebUI: force speak triggered")
                asyncio.ensure_future(self._trigger_proactive_speech())
            else:
                logger.info("WebUI: force speak ignored (processing or disconnected)")

        elif command == "reconnect_voice":
            logger.info("WebUI: reconnect voice triggered")
            asyncio.ensure_future(self._reconnect_voice())

    # ──────────────────────────────────────────
    # Voice reconnect
    # ──────────────────────────────────────────

    async def _reconnect_voice(self) -> None:
        """Desconecta y reconecta al mismo voice channel."""
        try:
            channel = None
            if self._voice_client and self._voice_client.channel:
                channel = self._voice_client.channel

            # Detener escucha y desconectar
            await self._stop_voice_receive()
            if self._voice_client:
                try:
                    await self._voice_client.disconnect(force=True)
                except Exception:
                    pass
                self._voice_client = None

            if not channel:
                logger.warning("Reconnect: no hay canal previo")
                await self._emit("chat_message", {"speaker": "SYSTEM", "text": "❌ No hay canal de voz previo"})
                return

            # Reconectar
            await self._set_bot_state("processing")
            self._voice_client = await channel.connect()
            await self._start_voice_receive()
            await self._set_bot_state("listening")
            logger.info("Reconnect: reconectada a '%s'", channel.name)
            await self._emit("chat_message", {"speaker": "SYSTEM", "text": f"🌀 Reconectada a {channel.name}"})

        except Exception:
            logger.exception("Reconnect: error")
            await self._emit("chat_message", {"speaker": "SYSTEM", "text": "❌ Error reconectando"})
            await self._set_bot_state("listening")

    # ──────────────────────────────────────────
    # Proactive mode
    # ──────────────────────────────────────────

    def _start_proactive_timer(self) -> None:
        """Inicia el timer para habla proactiva."""
        self._stop_proactive_timer()
        self._last_voice_activity = time.monotonic()
        self._proactive_timer_task = asyncio.ensure_future(self._proactive_loop())

    def _stop_proactive_timer(self) -> None:
        """Detiene el timer proactivo."""
        if self._proactive_timer_task and not self._proactive_timer_task.done():
            self._proactive_timer_task.cancel()
            self._proactive_timer_task = None

    async def _proactive_loop(self) -> None:
        """Loop que chequea si debe hablar proactivamente."""
        try:
            while self._proactive_mode:
                # Intervalo de check — más rápido si idle_seconds es bajo
                check_interval = max(2, self._proactive_idle_seconds // 5)
                await asyncio.sleep(check_interval)

                if not self._proactive_mode:
                    break
                if self._processing:
                    continue
                if not self._voice_client or not self._voice_client.is_connected():
                    continue

                elapsed = time.monotonic() - self._last_voice_activity
                if elapsed >= self._proactive_idle_seconds:
                    logger.info("Proactive: %.0fs sin voz, hablando...", elapsed)
                    await self._trigger_proactive_speech()
                    self._last_voice_activity = time.monotonic()  # Reset

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Proactive loop error")

    async def _trigger_proactive_speech(self) -> None:
        """Pregunta al LLM si tiene algo que decir. Solo habla si dice que sí."""
        try:
            self._processing = True
            loop = asyncio.get_event_loop()

            # Paso 1: Preguntar al LLM si quiere hablar
            decision_input = f"[SISTEMA]: {self._proactive_prompt}"

            def _decide() -> str:
                return "".join(
                    self._llm.generate_stream(
                        decision_input, "", self._chat_history
                    )
                )

            raw = await loop.run_in_executor(self._executor, _decide)
            raw = self._clean_llm_output(raw)

            # Si el LLM decide no hablar
            if not raw or raw.upper().startswith("NO"):
                logger.info("Proactive: LLM decidió no hablar")
                return

            # Paso 2: El LLM quiere hablar — reproducir
            logger.info("Proactive MIA: %s", raw)
            await self._set_bot_state("proactive")
            await self._stop_voice_receive()

            # TTS
            def _synthesize() -> "np.ndarray":
                return self._tts.synthesize(tts_filter(raw))

            audio = await loop.run_in_executor(self._executor, _synthesize)
            if audio is not None and len(audio) > 0:
                await self._play_in_discord(audio)

            # Historial
            self._chat_history.append({"role": "assistant", "content": raw})
            self._session_log.append({"role": "assistant", "content": raw})
            if len(self._chat_history) > 20:
                self._chat_history[:] = self._chat_history[-12:]

            await self._emit("chat_message", {
                "speaker": "MIA",
                "text": raw,
                "proactive": True,
            })

        except Exception:
            logger.exception("Proactive speech error")
        finally:
            self._processing = False
            try:
                if self._voice_client and self._voice_client.is_connected():
                    await self._start_voice_receive()
            except Exception:
                logger.exception("Proactive: error reiniciando voice receive")
            await self._set_bot_state("listening")

    # ──────────────────────────────────────────
    # Events
    # ──────────────────────────────────────────

    def _setup_events(self) -> None:
        """Registra event handlers del bot."""

        @self.bot.event
        async def on_ready() -> None:
            logger.info(
                "Discord bot conectado como %s (ID: %s)",
                self.bot.user.name if self.bot.user else "?",
                self.bot.user.id if self.bot.user else "?",
            )
            # Sincronizar slash commands PER GUILD (instantáneo)
            # Los globales tardan hasta 1 hora en aparecer
            for guild in self.bot.guilds:
                try:
                    await self.bot.sync_commands(guild_ids=[guild.id])
                    logger.info(
                        "Discord: commands sincronizados en '%s' (ID: %s)",
                        guild.name, guild.id,
                    )
                except Exception as e:
                    logger.warning(
                        "Discord: error sincronizando commands en '%s': %s",
                        guild.name, e,
                    )

        @self.bot.event
        async def on_message(message: discord.Message) -> None:
            # Ignorar bots
            if message.author.bot:
                return

            # Solo responder si text responses están habilitadas
            if not self._text_responses_enabled:
                return

            # Responder cuando la mencionan o en DMs
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = (
                self.bot.user is not None
                and self.bot.user.mentioned_in(message)
            )

            if not (is_dm or is_mentioned):
                return

            # Extraer texto limpio (sin la mención)
            content = message.content
            if self.bot.user and not is_dm:
                content = content.replace(f"<@{self.bot.user.id}>", "")
                content = content.replace(f"<@!{self.bot.user.id}>", "")
            content = content.strip()

            if not content:
                await message.reply("¿Sí? 👀", mention_author=False)
                return

            speaker_name = message.author.display_name
            labeled_input = f"[{speaker_name}]: {content}"

            logger.info("Discord text: %s", labeled_input)

            async with message.channel.typing():
                response = await self._generate_text_response(labeled_input)

            if response:
                # Discord tiene límite de 2000 chars
                if len(response) > 1900:
                    # Dividir en chunks
                    for i in range(0, len(response), 1900):
                        chunk = response[i:i + 1900]
                        if i == 0:
                            await message.reply(chunk, mention_author=False)
                        else:
                            await message.channel.send(chunk)
                else:
                    await message.reply(response, mention_author=False)
            else:
                await message.reply(
                    "Hmm, no se me ocurre nada 🤔",
                    mention_author=False,
                )

            # Procesar prefix commands (! commands como fallback)
            await self.bot.process_commands(message)


    # ──────────────────────────────────────────
    # Slash Commands
    # ──────────────────────────────────────────

    def _setup_commands(self) -> None:
        """Registra slash commands."""

        # ── /join ──
        @self.bot.slash_command(
            name="join",
            description="MIA se une a tu canal de voz 🎙️",
        )
        async def join(ctx: discord.ApplicationContext) -> None:
            if not ctx.author.voice:
                await ctx.respond(
                    "❌ Primero entra a un canal de voz.",
                    ephemeral=True,
                )
                return

            # Defer so Discord waits while we connect (avoids 3s timeout)
            await ctx.defer()

            try:
                channel = ctx.author.voice.channel

                if ctx.voice_client:
                    if ctx.voice_client.is_connected() and ctx.voice_client.channel == channel:
                        await ctx.followup.send("Ya estoy aquí 😊")
                        return
                    # Stale or different channel — disconnect first
                    try:
                        await asyncio.wait_for(
                            ctx.voice_client.disconnect(force=True),
                            timeout=3.0,
                        )
                    except (Exception, asyncio.TimeoutError):
                        try:
                            ctx.voice_client.cleanup()
                        except Exception:
                            pass

                # Connect fresh
                self._voice_client = await channel.connect()

                # Start voice receive only if enabled by audio mode toggle
                if self._voice_receive_enabled:
                    await self._start_voice_receive()

                members = [
                    m.display_name
                    for m in channel.members
                    if not m.bot
                ]
                member_list = ", ".join(members) if members else "nadie"

                listen_status = (
                    "🎧 Escuchando voz activo"
                    if self._voice_receive_enabled
                    else "💤 Escucha de voz inactiva (actívala desde el panel)"
                )

                await ctx.followup.send(
                    f"🎙️ Conectada a **{channel.name}**\n"
                    f"👥 Presentes: {member_list}\n"
                    f"{listen_status}",
                )
                logger.info(
                    "Discord: conectada a '%s' (%d miembros)",
                    channel.name, len(members),
                )
            except Exception as e:
                logger.exception("Discord /join error")
                try:
                    await ctx.followup.send(f"❌ Error conectando: {e}")
                except Exception:
                    pass

        # ── /leave ──
        @self.bot.slash_command(
            name="leave",
            description="MIA se va del canal de voz 👋",
        )
        async def leave(ctx: discord.ApplicationContext) -> None:
            if not ctx.voice_client:
                await ctx.respond(
                    "No estoy en ningún canal.", ephemeral=True
                )
                return

            await self._stop_voice_receive()
            await ctx.voice_client.disconnect()
            self._voice_client = None
            await ctx.respond("👋 ¡Nos vemos!")
            logger.info("Discord: desconectada del voice channel")

        # ── /mia ──
        @self.bot.slash_command(
            name="mia",
            description="Envía un mensaje a MIA ✉️",
        )
        async def mia_text(
            ctx: discord.ApplicationContext,
            texto: discord.Option(str, description="Tu mensaje"),  # type: ignore
        ) -> None:
            speaker_name = ctx.author.display_name
            labeled_input = f"[{speaker_name}]: {texto}"

            logger.info("Discord /mia: %s", labeled_input)
            await ctx.defer()  # "MIA está pensando..."

            response = await self._generate_text_response(labeled_input)
            if response:
                await ctx.followup.send(f"🤖 {response}")
            else:
                await ctx.followup.send("No pude pensar en nada 😅")

        # ── /mute ──
        @self.bot.slash_command(
            name="mute",
            description="MIA mutea a alguien en el voice channel 🔇",
        )
        async def mute_user(
            ctx: discord.ApplicationContext,
            usuario: discord.Option(  # type: ignore
                discord.Member,
                description="A quién mutear",
            ),
        ) -> None:
            try:
                await usuario.edit(mute=True)
                await ctx.respond(
                    f"🔇 **{usuario.display_name}** muteado. Silencio, por favor.",
                )
            except discord.Forbidden:
                await ctx.respond(
                    "❌ No tengo permisos para mutear.", ephemeral=True
                )

        # ── /unmute ──
        @self.bot.slash_command(
            name="unmute",
            description="MIA desmutea a alguien 🔊",
        )
        async def unmute_user(
            ctx: discord.ApplicationContext,
            usuario: discord.Option(  # type: ignore
                discord.Member,
                description="A quién desmutear",
            ),
        ) -> None:
            try:
                await usuario.edit(mute=False)
                await ctx.respond(
                    f"🔊 **{usuario.display_name}** desmuteado. Ya puedes hablar.",
                )
            except discord.Forbidden:
                await ctx.respond(
                    "❌ No tengo permisos para desmutear.", ephemeral=True
                )

        # ── /move ──
        @self.bot.slash_command(
            name="move",
            description="MIA mueve a alguien a otro voice channel 🚀",
        )
        async def move_user(
            ctx: discord.ApplicationContext,
            usuario: discord.Option(  # type: ignore
                discord.Member,
                description="A quién mover",
            ),
            canal: discord.Option(  # type: ignore
                discord.VoiceChannel,
                description="A dónde moverlo",
            ),
        ) -> None:
            try:
                await usuario.move_to(canal)
                await ctx.respond(
                    f"🚀 **{usuario.display_name}** movido a **{canal.name}**",
                )
            except discord.Forbidden:
                await ctx.respond(
                    "❌ No tengo permisos para mover miembros.",
                    ephemeral=True,
                )
            except discord.HTTPException:
                await ctx.respond(
                    "❌ No pude mover a ese usuario. ¿Está en un canal de voz?",
                    ephemeral=True,
                )

        # ── /nick ──
        @self.bot.slash_command(
            name="nick",
            description="MIA cambia el apodo de alguien ✏️",
        )
        async def change_nick(
            ctx: discord.ApplicationContext,
            usuario: discord.Option(  # type: ignore
                discord.Member,
                description="A quién cambiarle el apodo",
            ),
            apodo: discord.Option(  # type: ignore
                str,
                description="Nuevo apodo",
            ),
        ) -> None:
            old_nick = usuario.display_name
            try:
                await usuario.edit(nick=apodo)
                await ctx.respond(
                    f"✏️ **{old_nick}** ahora se llama **{apodo}**",
                )
            except discord.Forbidden:
                await ctx.respond(
                    "❌ No tengo permisos para cambiar apodos.",
                    ephemeral=True,
                )

        # ── /deafen ──
        @self.bot.slash_command(
            name="deafen",
            description="MIA ensordece a alguien (no escucha nada) 🙉",
        )
        async def deafen_user(
            ctx: discord.ApplicationContext,
            usuario: discord.Option(  # type: ignore
                discord.Member,
                description="A quién ensordecer",
            ),
        ) -> None:
            try:
                await usuario.edit(deafen=True)
                await ctx.respond(
                    f"🙉 **{usuario.display_name}** ensordecido. No escucha nada.",
                )
            except discord.Forbidden:
                await ctx.respond(
                    "❌ No tengo permisos para ensordecer.",
                    ephemeral=True,
                )

        # ── /undeafen ──
        @self.bot.slash_command(
            name="undeafen",
            description="MIA des-ensordece a alguien 👂",
        )
        async def undeafen_user(
            ctx: discord.ApplicationContext,
            usuario: discord.Option(  # type: ignore
                discord.Member,
                description="A quién des-ensordecer",
            ),
        ) -> None:
            try:
                await usuario.edit(deafen=False)
                await ctx.respond(
                    f"👂 **{usuario.display_name}** puede escuchar de nuevo.",
                )
            except discord.Forbidden:
                await ctx.respond(
                    "❌ No tengo permisos.", ephemeral=True
                )

        # ── /status ──
        @self.bot.slash_command(
            name="status",
            description="Muestra el estado de MIA 📊",
        )
        async def bot_status(ctx: discord.ApplicationContext) -> None:
            vc = ctx.voice_client
            vc_status = (
                f"🎙️ En **{vc.channel.name}** "
                f"({'grabando' if self._voice_sink else 'idle'})"
                if vc
                else "No conectada a voz"
            )
            text_status = (
                "✅ Activado" if self._text_responses_enabled else "❌ Desactivado"
            )
            history_len = len(self._chat_history)

            embed = discord.Embed(
                title="📊 Estado de MIA",
                color=discord.Color.gold(),
            )
            embed.add_field(name="Voice", value=vc_status, inline=False)
            embed.add_field(
                name="Text responses", value=text_status, inline=True
            )
            embed.add_field(
                name="Historial", value=f"{history_len} mensajes", inline=True
            )
            embed.add_field(
                name="TTS Backend",
                value=self._config.tts.backend.upper(),
                inline=True,
            )
            embed.add_field(
                name="LLM Backend",
                value=self._config.llm.backend.upper(),
                inline=True,
            )

            await ctx.respond(embed=embed)

        # ── Prefix commands (fallback para cuando slash no aparecen) ──

        @self.bot.command(name="join")
        async def prefix_join(ctx: commands.Context) -> None:
            """!join — fallback si /join no aparece."""
            if not ctx.author.voice:
                await ctx.send("❌ Primero entra a un canal de voz.")
                return

            channel = ctx.author.voice.channel
            if ctx.voice_client:
                if ctx.voice_client.channel == channel:
                    await ctx.send("Ya estoy aquí 😊")
                    return
                await ctx.voice_client.move_to(channel)
                self._voice_client = ctx.voice_client
            else:
                self._voice_client = await channel.connect()

            await self._start_voice_receive()
            await ctx.send(f"🎙️ Conectada a **{channel.name}**. ¡Los escucho!")
            logger.info("Discord: conectada a '%s' (prefix cmd)", channel.name)

        @self.bot.command(name="leave")
        async def prefix_leave(ctx: commands.Context) -> None:
            """!leave — fallback si /leave no aparece."""
            if not ctx.voice_client:
                await ctx.send("No estoy en ningún canal.")
                return

            await self._stop_voice_receive()
            await ctx.voice_client.disconnect()
            self._voice_client = None
            await ctx.send("👋 ¡Nos vemos!")

        @self.bot.command(name="mia")
        async def prefix_mia(ctx: commands.Context, *, texto: str = "") -> None:
            """!mia <texto> — fallback si /mia no aparece."""
            if not texto:
                await ctx.send("Uso: `!mia <tu mensaje>`")
                return

            speaker_name = ctx.author.display_name
            labeled_input = f"[{speaker_name}]: {texto}"

            async with ctx.typing():
                response = await self._generate_text_response(labeled_input)

            if response:
                await ctx.send(f"🤖 {response}")
            else:
                await ctx.send("No pude pensar en nada 😅")

    # ──────────────────────────────────────────
    # Voice receive
    # ──────────────────────────────────────────

    async def enable_voice_receive(self) -> None:
        """Habilita la recepción de audio (llamado por audio mode toggle)."""
        self._voice_receive_enabled = True
        if self._voice_client and self._voice_client.is_connected():
            await self._start_voice_receive()
        await self._set_bot_state("listening")
        logger.info("Discord: voice receive HABILITADO")

    async def disable_voice_receive(self) -> None:
        """Deshabilita la recepción de audio (llamado por audio mode toggle)."""
        self._voice_receive_enabled = False
        await self._stop_voice_receive()
        await self._set_bot_state("paused")
        logger.info("Discord: voice receive DESHABILITADO")

    def _on_audio_level(self, rms: float) -> None:
        """Callback del sink — emite nivel RMS al WebUI (throttled)."""
        # Called from recv_audio thread — use bot's loop to schedule safely
        try:
            if self._event_callbacks:
                loop = self.bot.loop
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda r=rms: asyncio.ensure_future(
                            self._emit("audio_level", {"rms": round(r, 4)})
                        )
                    )
        except Exception:
            pass

    async def _start_voice_receive(self) -> None:
        """Inicia la recepción de audio del voice channel."""
        if not self._voice_receive_enabled:
            return
        if not self._voice_client or not self._voice_client.is_connected():
            return

        # Detener sink anterior si existe
        await self._stop_voice_receive()

        self._voice_sink = GroupVoiceSink(
            voice_client=self._voice_client,
            group_silence_ms=self._config.discord.group_silence_ms,
            on_group_silence=self._on_group_silence,
            on_audio_level=self._on_audio_level,
            bot_user_id=self.bot.user.id if self.bot.user else None,
        )
        await self._voice_sink.start()
        logger.info("Discord: voice receive activado")

    async def _stop_voice_receive(self) -> None:
        """Detiene la recepción de audio."""
        if self._voice_sink:
            await self._voice_sink.stop()
            self._voice_sink = None

    async def _on_group_silence(
        self, speakers: dict[int, SpeakerData]
    ) -> None:
        """Callback cuando se detecta silencio grupal.

        Transcribe el audio de cada speaker, construye prompt
        multi-speaker, y genera respuesta con voz.
        """
        if self._processing:
            logger.debug("Discord: ya procesando, ignorando trigger")
            return

        if not speakers:
            return

        self._processing = True
        try:
            # Dejar de escuchar mientras procesamos
            await self._stop_voice_receive()
            await self._set_bot_state("processing")

            # Reset proactive timer — alguien habló
            self._last_voice_activity = time.monotonic()

            # STT per speaker
            loop = asyncio.get_event_loop()
            speaker_texts: list[tuple[str, str]] = []  # (name, text)

            for user_id, speaker_data in speakers.items():
                name = speaker_data["name"]
                audio = speaker_data["audio"]

                # Filtrar usuarios fantasma (IDs que Discord no puede resolver)
                if name.startswith("User_"):
                    logger.debug("Discord: ignorando speaker fantasma %s", name)
                    continue

                if len(audio) < 8000:  # < 500ms a 16kHz — evita alucinaciones de Whisper
                    continue

                # Filtro de energía RMS
                rms = float(np.sqrt(np.mean(audio ** 2)))
                if rms < self._min_energy_rms:
                    logger.debug("STT skip [%s]: energía %.4f < %.4f", name, rms, self._min_energy_rms)
                    continue

                text = await loop.run_in_executor(
                    self._executor,
                    self._stt.transcribe,
                    audio,
                    16000,
                )

                if not text or len(text.strip()) < 2:
                    continue

                # Filtro de alucinaciones de Whisper
                if text.strip().lower().rstrip(".!?¡¿") in self._stt_hallucinations:
                    logger.info("STT hallucination filtered [%s]: '%s'", name, text.strip())
                    continue

                speaker_texts.append((name, text.strip()))
                logger.info("Discord STT [%s]: %s", name, text.strip())
                # Emit chat event for each speaker
                await self._emit("chat_message", {
                    "speaker": name,
                    "text": text.strip(),
                })

            if not speaker_texts:
                logger.debug("Discord: sin texto útil transcrito")
                return

            # Construir prompt multi-speaker
            if len(speaker_texts) == 1:
                name, text = speaker_texts[0]
                labeled_input = f"[{name}]: {text}"
            else:
                labeled_input = "\n".join(
                    f"[{name}]: {text}" for name, text in speaker_texts
                )

            logger.info("Discord multi-speaker:\n%s", labeled_input)

            # Generar respuesta con voz
            await self._set_bot_state("speaking")
            response = await self._generate_and_speak(labeled_input)

            if response:
                logger.info("Discord MIA: %s", response)
                await self._emit("chat_message", {
                    "speaker": "MIA",
                    "text": response,
                })

        except Exception:
            logger.exception("Discord: error procesando group silence")
        finally:
            self._processing = False
            self._last_voice_activity = time.monotonic()
            # Reanudar escucha después de todo el pipeline
            try:
                if self._voice_client and self._voice_client.is_connected():
                    await self._start_voice_receive()
            except Exception:
                logger.exception("Discord: error reiniciando voice receive")
            await self._set_bot_state("listening")
            # Emit stats update
            await self._emit("stats", self.get_state().get("stats", {}))

    # ──────────────────────────────────────────
    # Response generation + dual audio
    # ──────────────────────────────────────────

    async def _generate_text_response(self, user_text: str) -> str:
        """Genera respuesta de texto (sin audio)."""
        loop = asyncio.get_event_loop()

        # RAG
        rag_context = ""
        if self._rag and self._config.rag.enabled:
            rag_context = await loop.run_in_executor(
                self._executor,
                self._rag.build_context_block,
                user_text,
            )

        # LLM
        def _generate() -> str:
            return "".join(
                self._llm.generate_stream(
                    user_text, rag_context, self._chat_history
                )
            )

        response = await loop.run_in_executor(self._executor, _generate)

        clean_response = self._clean_llm_output(response)

        # Guardar en historial
        if clean_response.strip():
            self._chat_history.append({"role": "user", "content": user_text})
            self._chat_history.append(
                {"role": "assistant", "content": clean_response}
            )
            self._session_log.append({"role": "user", "content": user_text})
            self._session_log.append(
                {"role": "assistant", "content": clean_response}
            )
            if len(self._chat_history) > 20:
                self._chat_history[:] = self._chat_history[-12:]

        return clean_response

    async def _generate_and_speak(self, user_text: str) -> str:
        """Genera respuesta y reproduce en Discord.

        Returns:
            Texto limpio de la respuesta.
        """
        loop = asyncio.get_event_loop()

        # RAG
        rag_context = ""
        if self._rag and self._config.rag.enabled:
            rag_context = await loop.run_in_executor(
                self._executor,
                self._rag.build_context_block,
                user_text,
            )

        # LLM
        def _generate() -> str:
            return "".join(
                self._llm.generate_stream(
                    user_text, rag_context, self._chat_history
                )
            )

        raw_response = await loop.run_in_executor(self._executor, _generate)

        clean_response = self._clean_llm_output(raw_response)
        logger.info("🤖 MIA: %s", clean_response[:80])

        if not clean_response.strip():
            return clean_response

        # TTS
        def _synthesize() -> np.ndarray:
            return self._tts.synthesize(tts_filter(clean_response))

        audio_data = await loop.run_in_executor(self._executor, _synthesize)

        if audio_data is not None and len(audio_data) > 0:
            await self._play_in_discord(audio_data)

        # Historial
        self._chat_history.append({"role": "user", "content": user_text})
        self._chat_history.append(
            {"role": "assistant", "content": clean_response}
        )
        self._session_log.append({"role": "user", "content": user_text})
        self._session_log.append(
            {"role": "assistant", "content": clean_response}
        )
        if len(self._chat_history) > 20:
            self._chat_history[:] = self._chat_history[-12:]

        return clean_response

    async def _play_discord_audio(self, audio: np.ndarray) -> None:
        """Reproduce audio en Discord."""
        if self._voice_client and self._voice_client.is_connected():
            await self._play_in_discord(audio)

    async def _play_in_discord(self, audio: np.ndarray) -> None:
        """Reproduce audio en el voice channel vía FFmpeg."""
        try:
            # Convertir a WAV temporal
            sr = getattr(self._tts, "sample_rate", 24000)
            wav_path = await self._numpy_to_temp_wav(audio, sr)

            if wav_path and self._voice_client and self._voice_client.is_connected():
                source = FFmpegPCMAudio(wav_path)
                # Use the bot's loop explicitly to avoid 'different loop' error
                bot_loop = self.bot.loop or asyncio.get_event_loop()
                done_event = asyncio.Event()

                def after_play(error: Exception | None) -> None:
                    if error:
                        logger.error("Discord playback error: %s", error)
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
                    if bot_loop.is_running():
                        bot_loop.call_soon_threadsafe(done_event.set)

                self._voice_client.play(source, after=after_play)
                try:
                    await asyncio.wait_for(done_event.wait(), timeout=60.0)
                except asyncio.TimeoutError:
                    logger.warning("Discord: playback timeout")

        except Exception:
            logger.exception("Discord: error playing audio")

    @staticmethod
    async def _numpy_to_temp_wav(
        audio: np.ndarray, sample_rate: int
    ) -> Optional[str]:
        """Convierte numpy array a archivo WAV temporal."""
        try:
            import wave

            if audio.dtype != np.int16:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio

            fd, path = tempfile.mkstemp(suffix=".wav")
            try:
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())
            finally:
                os.close(fd)

            return path
        except Exception as e:
            logger.error("Error creating temp WAV: %s", e)
            return None

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    async def start(self, token: str) -> None:
        """Inicia el bot de Discord."""
        logger.info(
            "Discord bot starting... (token: %s...%s, %d chars)",
            token[:4] if len(token) > 8 else "????",
            token[-4:] if len(token) > 8 else "????",
            len(token),
        )
        # Py-cord's Bot.__init__ stores asyncio.get_event_loop() which may
        # differ from the loop running inside asyncio.run(). We must update
        # the loop reference (same as __aenter__ does) so VoiceClient,
        # recv_audio and run_coroutine_threadsafe all use the correct loop.
        loop = asyncio.get_running_loop()
        self.bot.loop = loop
        self.bot.http.loop = loop
        self.bot._connection.loop = loop
        logger.info("Discord: llamando bot.start()...")
        await self.bot.start(token)
        logger.info("Discord: bot.start() retornó (no debería pasar normalmente)")

    async def close(self) -> None:
        """Cierra el bot limpiamente."""
        await self._stop_voice_receive()
        if self._voice_client and self._voice_client.is_connected():
            await self._voice_client.disconnect()
        await self.bot.close()
        logger.info("Discord bot closed")

    @property
    def text_responses_enabled(self) -> bool:
        return self._text_responses_enabled

    @text_responses_enabled.setter
    def text_responses_enabled(self, value: bool) -> None:
        self._text_responses_enabled = value
        logger.info("Discord text responses: %s", "ON" if value else "OFF")
