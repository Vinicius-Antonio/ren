"""Bot de voz do Discord para o Hermes 2.0 (transmissão de voz bidirecional com DAVE E2EE e controle por comandos)."""

from __future__ import annotations

import asyncio
import logging
import threading
import unicodedata
from typing import Any, List, Optional

from src.config import Settings
from src.discord_bot.audio import (
    HermesDiscordAudioSource,
    HermesDiscordVoiceReceiver,
    ensure_opus_loaded,
)
from src.gateway.ws_server import GatewayServer

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(no_accents.split()).strip()


class DiscordVoiceBot:
    """Gerencia a presença e transmissão/recepção de voz do Hermes em canais de voz do Discord."""

    def __init__(
        self,
        settings: Settings,
        gateway: Optional[GatewayServer] = None,
    ) -> None:
        self.settings = settings
        self.gateway = gateway
        self.token = settings.discord_bot_token

        self._audio_source = HermesDiscordAudioSource(startup_frames=6)
        self._receiver = HermesDiscordVoiceReceiver()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[Any] = None
        self._ready_event = threading.Event()
        self._lock = threading.Lock()
        self._active_voice_channel_name: str = ""
        self._active_guild_name: str = ""
        self._active_guild_id: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self._loop is not None and self._loop.is_running()

    @property
    def is_in_call(self) -> bool:
        return bool(self._active_voice_channel_name)

    @property
    def is_speaking(self) -> bool:
        return self.is_in_call and self._audio_source.is_speaking_active

    @property
    def active_channel(self) -> str:
        return self._active_voice_channel_name

    @property
    def active_guild(self) -> str:
        return self._active_guild_name

    def start(self) -> bool:
        """Inicia o bot do Discord em uma thread daemon separada."""
        if not self.token:
            logger.info("DISCORD_BOT_TOKEN não configurado. Módulo Discord em modo inativo.")
            return False

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._ready_event.clear()
            self._thread = threading.Thread(
                target=self._run_thread,
                name="hermes-discord-bot",
                daemon=True,
            )
            self._thread.start()

        # Aguarda brevemente a conexão inicial (até 6s)
        if not self._ready_event.wait(timeout=6.0):
            return False
        client = self._client
        return bool(client is not None and client.is_ready())

    def _sync_voice_playback_state(self) -> None:
        """Sincroniza pause/resume do AudioPlayer para emitir op=5 SPEAKING e OPUS_SILENCE corretamente."""
        client = self._client
        if client is None:
            return
        want_speak = self._audio_source.is_speaking_active
        for vc in list(client.voice_clients):
            if not vc.is_connected():
                continue
            try:
                if want_speak:
                    if vc.is_paused():
                        vc.resume()
                    elif not vc.is_playing():
                        vc.play(self._audio_source)
                else:
                    if vc.is_playing() and not vc.is_paused():
                        vc.pause()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Falha ao sincronizar estado de fala no Discord: %s", exc)

    def _run_thread(self) -> None:
        try:
            import discord

            from src.discord_bot.dave_patch import apply_dave_compat_patch

            ensure_opus_loaded()
            apply_dave_compat_patch()

            intents = discord.Intents.default()
            intents.guilds = True
            intents.voice_states = True
            intents.members = False

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            client = discord.Client(intents=intents)
            self._client = client

            async def _speaking_watchdog() -> None:
                while not client.is_closed():
                    await asyncio.sleep(0.08)
                    if self.is_in_call:
                        self._sync_voice_playback_state()

            @client.event
            async def on_ready() -> None:
                user_tag = str(client.user)
                logger.info("Bot do Discord Hermes conectado como: %s", user_tag)
                self._ready_event.set()
                loop.create_task(_speaking_watchdog(), name="discord-speaking-watchdog")
                if self.gateway is not None:
                    self.gateway.emit_ui_widget(
                        "discord",
                        {"status": "ONLINE", "user": user_tag},
                    )

            @client.event
            async def on_voice_state_update(
                member: Any, before: Any, after: Any
            ) -> None:
                # Se o próprio bot foi desconectado externamente
                if member == client.user and after.channel is None:
                    with self._lock:
                        self._active_voice_channel_name = ""
                        self._active_guild_name = ""
                    self._active_guild_id = None
                    self._receiver.set_voice_client(None)
                    self._receiver.clear()
                    self._audio_source.clear()
                    logger.info("Bot do Hermes foi desconectado do canal de voz.")
                    if self.gateway is not None:
                        self.gateway.emit_ui_widget(
                            "discord",
                            {"status": "DISCONNECTED", "action": "left"},
                        )

            loop.run_until_complete(client.start(self.token))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Erro na execução do bot do Discord: %s", exc)
        finally:
            self._ready_event.set()
            with self._lock:
                self._client = None
                self._loop = None
                self._active_voice_channel_name = ""
                self._active_guild_name = ""
                self._active_guild_id = None

    def stop(self) -> None:
        """Desconecta de todos os canais e encerra o cliente do Discord."""
        with self._lock:
            client = self._client
            loop = self._loop

        if client is not None and loop is not None and loop.is_running():
            async def _close_all() -> None:
                for vc in list(client.voice_clients):
                    try:
                        await vc.disconnect(force=True)
                    except Exception:  # noqa: BLE001
                        pass
                await client.close()

            asyncio.run_coroutine_threadsafe(_close_all(), loop)

    def feed_audio(self, pcm_24k_mono: bytes) -> None:
        """Enfileira áudio falado pelo Hermes para reprodução imediata no canal de voz do Discord."""
        if not self.is_in_call:
            return
        self._audio_source.feed(pcm_24k_mono)

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._sync_voice_playback_state)

    def end_audio_turn(self) -> None:
        """Fecha o turno de saída sem descartar o áudio ainda no buffer."""
        self._audio_source.end_turn()

    def publish_text(self, content: str) -> bool:
        """Agenda uma mensagem no loop do Discord sem bloquear o áudio do Gemini."""
        loop = self._loop
        if not self.is_in_call or loop is None or not loop.is_running():
            return False
        if not self.settings.discord_text_channel_id or not content.strip():
            return False
        future = asyncio.run_coroutine_threadsafe(self._publish_text(content), loop)

        def report_result(done: Any) -> None:
            try:
                done.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao publicar no chat do Discord: %s", exc)

        future.add_done_callback(report_result)
        return True

    async def _publish_text(self, content: str) -> None:
        import discord

        channel = await self._resolve_text_channel()
        safe_text = content.strip()
        for start in range(0, len(safe_text), 1900):
            await channel.send(
                safe_text[start : start + 1900],
                allowed_mentions=discord.AllowedMentions.none(),
            )

    async def _resolve_text_channel(self) -> Any:
        """Confere canal, servidor e permissões antes de publicar."""
        client = self._client
        channel_id = self.settings.discord_text_channel_id
        if client is None or channel_id is None:
            raise RuntimeError("Canal de texto não configurado")
        channel = client.get_channel(channel_id)
        if channel is None:
            channel = await client.fetch_channel(channel_id)
        guild = getattr(channel, "guild", None)
        if guild is None or guild.id != self._active_guild_id:
            raise ValueError("Canal de texto não pertence ao servidor da call ativa")
        member = guild.me
        if member is not None:
            permissions = channel.permissions_for(member)
            if not permissions.view_channel or not permissions.send_messages:
                raise PermissionError("Bot sem permissão para ver ou enviar mensagens nesse canal")
        return channel

    def get_incoming_chunk(self, max_bytes: int) -> bytes:
        """Retorna áudio PCM 16 kHz mono recebido dos membros humanos do Discord."""
        if not self.is_in_call:
            return b""
        return self._receiver.read_chunk(max_bytes)

    def get_speaker_chunk(self, speaker_id: int, max_bytes: int) -> bytes:
        if not self.is_in_call:
            return b""
        return self._receiver.read_speaker_chunk(speaker_id, max_bytes)

    def get_wake_chunks(self, max_bytes: int) -> dict[int, bytes]:
        if not self.is_in_call:
            return {}
        return self._receiver.read_wake_chunks(max_bytes)

    def speaker_name(self, speaker_id: int) -> str:
        return self._receiver.speaker_name(speaker_id)

    def clear_wake_audio(self) -> None:
        self._receiver.clear_wake_audio()

    def reset_call_input(self) -> None:
        self._receiver.reset_call_input()

    def has_incoming_audio(self) -> bool:
        if not self.is_in_call:
            return False
        return self._receiver.has_audio()

    @property
    def last_speech_time(self) -> float:
        return self._receiver.last_speech_time if self.is_in_call else 0.0

    @property
    def last_speaker_name(self) -> str:
        return self._receiver.last_speaker_name if self.is_in_call else ""

    @property
    def last_speaker_username(self) -> str:
        return self._receiver.last_speaker_username if self.is_in_call else ""

    @property
    def last_speaker_id(self) -> Optional[int]:
        return self._receiver.last_speaker_id if self.is_in_call else None

    # ----------------------------------------------------------- Ações de Voz
    def join_call(self, channel_name: Optional[str] = None) -> str:
        """Entra na chamada de voz do Discord (método thread-safe chamado por tools do Gemini)."""
        if not self.token:
            return (
                "O bot do Discord ainda não está configurado. "
                "Para eu entrar na call, adicione o 'DISCORD_BOT_TOKEN' no arquivo .env."
            )

        if not self.is_running:
            self.start()
            if not self._ready_event.wait(timeout=5.0):
                return "Não foi possível conectar o bot ao Discord no momento. Verifique seu token e conexão de internet."

        loop = self._loop
        if loop is None or not loop.is_running():
            return "O serviço do Discord não está ativo no momento."

        fut = asyncio.run_coroutine_threadsafe(
            self._async_join_call(channel_name), loop
        )
        try:
            return fut.result(timeout=14.0)
        except Exception as exc:  # noqa: BLE001
            return f"Erro ao tentar entrar na call do Discord: {exc}"

    async def _async_join_call(self, channel_hint: Optional[str]) -> str:
        client = self._client
        if client is None or not client.is_ready():
            return "O bot do Discord ainda está conectando aos servidores. Tente novamente em alguns segundos."

        if not client.guilds:
            return (
                "O bot do Hermes ainda não foi adicionado a nenhum servidor do Discord! "
                "Convide-o para o seu servidor através do Portal de Desenvolvedores do Discord."
            )

        target_channel: Optional[Any] = None
        target_guild: Optional[Any] = None
        channel_has_humans: bool = False

        # A conta configurada tem prioridade sobre a heurística de primeiro canal ocupado.
        if not channel_hint and self.settings.discord_user_id:
            for guild in client.guilds:
                state = getattr(guild, "_voice_states", {}).get(self.settings.discord_user_id)
                channel = getattr(state, "channel", None)
                if channel is not None:
                    target_channel = channel
                    target_guild = guild
                    channel_has_humans = True
                    break

        # 1. Se informou nome de canal específico, procura em todos os servidores
        if channel_hint and target_channel is None:
            norm_hint = _norm(channel_hint)
            if norm_hint in (
                "discord",
                "call",
                "call do discord",
                "chamada",
                "chamada de voz",
                "voz",
                "canal de voz",
                "a call",
                "na call",
            ):
                norm_hint = ""

            if norm_hint:
                for guild in client.guilds:
                    for vc in guild.voice_channels:
                        if norm_hint in _norm(vc.name):
                            target_channel = vc
                            target_guild = guild
                            break
                    if target_channel:
                        break

        # 2. Se não encontrou ou não especificou, procura o canal onde pessoas reais estão conectadas
        if target_channel is None:
            for guild in client.guilds:
                for vc in guild.voice_channels:
                    has_humans = False
                    for u_id, v_state in getattr(guild, "_voice_states", {}).items():
                        if v_state.channel and v_state.channel.id == vc.id:
                            if client.user and u_id != client.user.id:
                                has_humans = True
                                break
                    if not has_humans:
                        human_members = [m for m in vc.members if not m.bot]
                        if human_members:
                            has_humans = True
                    if has_humans:
                        target_channel = vc
                        target_guild = guild
                        channel_has_humans = True
                        break
                if target_channel is not None:
                    break

        # 3. Fallback: canal padrão configurado no .env (ex.: "Geral")
        if target_channel is None:
            default_name = _norm(self.settings.discord_default_channel)
            for guild in client.guilds:
                for vc in guild.voice_channels:
                    if default_name in _norm(vc.name):
                        target_channel = vc
                        target_guild = guild
                        break
                if target_channel is None and guild.voice_channels:
                    target_channel = guild.voice_channels[0]
                    target_guild = guild
                    break

        if target_channel is None:
            return "Não encontrei nenhum canal de voz disponível no Discord para entrar."

        if not channel_has_humans and target_guild is not None:
            for u_id, v_state in getattr(target_guild, "_voice_states", {}).items():
                if v_state.channel and v_state.channel.id == target_channel.id:
                    if client.user and u_id != client.user.id:
                        channel_has_humans = True
                        break

        import discord.ext.voice_recv as voice_recv

        voice_client: Optional[Any] = None
        for vc in client.voice_clients:
            if vc.guild == target_guild:
                voice_client = vc
                break

        if voice_client is not None:
            if not isinstance(voice_client, voice_recv.VoiceRecvClient):
                try:
                    await voice_client.disconnect(force=True)
                except Exception:  # noqa: BLE001
                    pass
                voice_client = None

        if voice_client is not None:
            if voice_client.channel != target_channel:
                await voice_client.move_to(target_channel)
        else:
            voice_client = await target_channel.connect(
                cls=voice_recv.VoiceRecvClient,
                reconnect=True,
                timeout=12.0,
                self_deaf=False,
                self_mute=False,
            )

        # Associa o VoiceRecvClient ao receptor e inicia a escuta da fala dos membros na call
        self._receiver.clear()
        self._receiver.set_voice_client(voice_client)
        if hasattr(voice_client, "is_listening") and not voice_client.is_listening():
            try:
                voice_client.listen(self._receiver)
            except Exception as exc:
                logger.warning("Erro ao registrar receptor de voz no Discord: %s", exc)

        # Aguarda brevemente (até 1.2s) a conclusão do handshake criptográfico DAVE E2EE (MLS)
        # quando há membros na sala, garantindo que a primeira fala do Hermes já saia audível!
        if channel_has_humans:
            conn = getattr(voice_client, "_connection", None)
            if conn is not None and getattr(conn, "dave_protocol_version", 0) > 0:
                for _ in range(12):
                    if getattr(conn, "can_encrypt", False):
                        break
                    await asyncio.sleep(0.1)

        # Prepara o AudioPlayer em estado pausado (pronto para dar resume() assim que o Hermes falar)
        self._audio_source.clear()
        if not voice_client.is_playing():
            try:
                voice_client.play(self._audio_source)
                if not self._audio_source.is_speaking_active:
                    voice_client.pause()
            except Exception as exc:
                logger.warning("Erro ao iniciar playback de voz no Discord: %s", exc)

        with self._lock:
            self._active_voice_channel_name = str(target_channel.name)
            self._active_guild_name = str(target_guild.name)
            self._active_guild_id = int(target_guild.id)

        if self.settings.discord_text_channel_id:
            try:
                await self._resolve_text_channel()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chat do Discord indisponível; voz continua ativa: %s", exc)

        if self.gateway is not None:
            self.gateway.emit_ui_widget(
                "discord",
                {
                    "status": "CONNECTED",
                    "channel": str(target_channel.name),
                    "guild": str(target_guild.name),
                },
            )

        return (
            f"Conectado com sucesso ao canal de voz '{target_channel.name}' "
            f"no servidor '{target_guild.name}' do Discord. Já estou na call e ouvindo vocês!"
        )

    def leave_call(self) -> str:
        """Sai de todos os canais de voz do Discord (método thread-safe)."""
        if not self.is_running:
            return "O Hermes não está conectado ao Discord no momento."

        loop = self._loop
        if loop is None or not loop.is_running():
            return "O serviço do Discord não está ativo."

        fut = asyncio.run_coroutine_threadsafe(self._async_leave_call(), loop)
        try:
            return fut.result(timeout=6.0)
        except Exception as exc:  # noqa: BLE001
            return f"Erro ao desconectar do Discord: {exc}"

    async def _async_leave_call(self) -> str:
        client = self._client
        if client is None:
            return "Bot não inicializado."

        left_channels: List[str] = []
        for vc in list(client.voice_clients):
            c_name = vc.channel.name if vc.channel else "Voz"
            left_channels.append(c_name)
            try:
                if hasattr(vc, "stop_listening"):
                    vc.stop_listening()
                if hasattr(vc, "stop"):
                    vc.stop()
                await vc.disconnect(force=True)
            except Exception:  # noqa: BLE001
                pass

        with self._lock:
            self._active_voice_channel_name = ""
            self._active_guild_name = ""
            self._active_guild_id = None

        self._receiver.set_voice_client(None)
        self._receiver.clear()
        self._audio_source.clear()

        if self.gateway is not None:
            self.gateway.emit_ui_widget(
                "discord",
                {"status": "DISCONNECTED", "action": "left"},
            )

        if left_channels:
            names = ", ".join(f"'{n}'" for n in left_channels)
            return f"Desconectado da chamada de voz {names} no Discord. Até logo!"
        return "O Hermes já não estava em nenhum canal de voz no Discord."

    def get_status(self) -> str:
        """Retorna o status atual da conexão com o Discord."""
        if not self.token:
            return "Discord não configurado (DISCORD_BOT_TOKEN vazio no .env)."
        if not self.is_running:
            return "Discord configurado, mas o bot está offline."
        with self._lock:
            ch = self._active_voice_channel_name
            g = self._active_guild_name
        if ch:
            return f"Conectado e transmitindo áudio no canal '{ch}' do servidor '{g}'."
        return "Online no Discord, mas não está em nenhum canal de voz no momento."
