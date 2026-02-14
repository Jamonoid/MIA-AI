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
from typing import Any, Optional

import numpy as np

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

_EMOTION_RE = re.compile(r"\[\w+\]\s*")
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
        self._config = config

        self._voice_sink: Optional[GroupVoiceSink] = None
        self._voice_client: Optional[discord.VoiceClient] = None
        self._text_responses_enabled = config.discord.text_channel_responses
        self._processing = False  # Guard para evitar respuestas simultáneas
        self._voice_receive_enabled = True  # Siempre activo en Discord-only

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
    def _strip_emotion_tags(text: str) -> str:
        """Elimina tags de emoción [tag] y prefijo 'MIA:' del texto."""
        text = _EMOTION_RE.sub("", text).strip()
        text = _NAME_PREFIX_RE.sub("", text).strip()
        return text

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
        logger.info("Discord: voice receive HABILITADO")

    async def disable_voice_receive(self) -> None:
        """Deshabilita la recepción de audio (llamado por audio mode toggle)."""
        self._voice_receive_enabled = False
        await self._stop_voice_receive()
        logger.info("Discord: voice receive DESHABILITADO")

    async def _start_voice_receive(self) -> None:
        """Inicia la recepción de audio del voice channel."""
        if not self._voice_client or not self._voice_client.is_connected():
            return

        # Detener sink anterior si existe
        await self._stop_voice_receive()

        self._voice_sink = GroupVoiceSink(
            voice_client=self._voice_client,
            group_silence_ms=self._config.discord.group_silence_ms,
            on_group_silence=self._on_group_silence,
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

                text = await loop.run_in_executor(
                    self._executor,
                    self._stt.transcribe,
                    audio,
                    16000,
                )

                if text and len(text.strip()) >= 2:
                    speaker_texts.append((name, text.strip()))
                    logger.info("Discord STT [%s]: %s", name, text.strip())

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
            response = await self._generate_and_speak(labeled_input)

            if response:
                logger.info("Discord MIA: %s", response)

        except Exception:
            logger.exception("Discord: error procesando group silence")
        finally:
            self._processing = False
            # Reanudar escucha después de todo el pipeline
            if self._voice_client and self._voice_client.is_connected():
                await self._start_voice_receive()

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

        # Strip emotion tags
        clean_response = self._strip_emotion_tags(response)

        # Guardar en historial
        if clean_response.strip():
            self._chat_history.append({"role": "user", "content": user_text})
            self._chat_history.append(
                {"role": "assistant", "content": clean_response}
            )
            if len(self._chat_history) > 20:
                self._chat_history[:] = self._chat_history[-12:]

            # RAG ingest
            if self._rag and self._config.rag.enabled:
                await loop.run_in_executor(
                    self._executor,
                    self._rag.ingest,
                    user_text,
                    clean_response,
                )

        return clean_response

    async def _generate_and_speak(self, user_text: str) -> str:
        """Genera respuesta y reproduce en Discord.

        Returns:
            Texto limpio de la respuesta (sin emotion tags).
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

        # Strip emotion tags antes de TTS
        clean_response = self._strip_emotion_tags(raw_response)
        logger.info("🤖 MIA: %s", clean_response[:80])

        if not clean_response.strip():
            return clean_response

        # TTS
        def _synthesize() -> np.ndarray:
            return self._tts.synthesize(clean_response)

        audio_data = await loop.run_in_executor(self._executor, _synthesize)

        if audio_data is not None and len(audio_data) > 0:
            await self._play_in_discord(audio_data)

        # Historial
        self._chat_history.append({"role": "user", "content": user_text})
        self._chat_history.append(
            {"role": "assistant", "content": clean_response}
        )
        if len(self._chat_history) > 20:
            self._chat_history[:] = self._chat_history[-12:]

        # RAG ingest
        if self._rag and self._config.rag.enabled:
            await loop.run_in_executor(
                self._executor,
                self._rag.ingest,
                user_text,
                clean_response,
            )

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
