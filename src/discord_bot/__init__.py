"""Módulo Discord Voice do Hermes 2.0."""

from src.discord_bot.audio import (
    HermesDiscordAudioSource,
    convert_24k_mono_to_48k_stereo,
    ensure_opus_loaded,
)
from src.discord_bot.bot import DiscordVoiceBot

__all__ = [
    "DiscordVoiceBot",
    "HermesDiscordAudioSource",
    "convert_24k_mono_to_48k_stereo",
    "ensure_opus_loaded",
]

