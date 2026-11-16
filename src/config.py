"""Configuração central do Hermes (variáveis de ambiente + constantes de áudio)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --- Especificações fixas de áudio (seção 3 do AGENTS.md) ---
CHANNELS: int = 1
INPUT_RATE: int = 16_000
OUTPUT_RATE: int = 24_000
CHUNK_SIZE: int = 1024
INPUT_MIME_TYPE: str = f"audio/pcm;rate={INPUT_RATE}"


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "sim", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Variável {name} inválida: {raw!r}") from exc


def _get_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


_VALID_THINKING_LEVELS = frozenset(
    {"THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"}
)
_VALID_SAFETY_THRESHOLDS = frozenset(
    {
        "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        "BLOCK_LOW_AND_ABOVE",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_NONE",
        "OFF",
    }
)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    gemini_voice: str
    picovoice_access_key: Optional[str]
    wake_word_model_path: Path
    wake_word_builtin: str
    wake_word_sensitivity: float
    vad_silence_threshold_rms: float
    vad_silence_timeout_sec: float
    half_duplex: bool
    sound_fx_enabled: bool
    wake_word_language_model_path: Path = PROJECT_ROOT / "models" / "porcupine_params_pt.pv"
    vosk_model_path: Path = PROJECT_ROOT / "models" / "vosk-model-small-pt-0.3"
    silero_model_path: Path = PROJECT_ROOT / "models" / "silero_vad.onnx"
    barge_in_enabled: bool = True
    gateway_enabled: bool = True
    gateway_port: int = 8765
    continuous_vision_enabled: bool = True
    discord_bot_token: Optional[str] = None
    discord_user_id: Optional[int] = None
    discord_text_channel_id: Optional[int] = None
    discord_default_channel: str = "Geral"
    obsidian_vault_path: Optional[Path] = None
    thinking_level: str = "LOW"
    safety_threshold: str = "BLOCK_ONLY_HIGH"

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "Settings":
        if load_dotenv_file:
            from dotenv import dotenv_values

            for k, v in dotenv_values(PROJECT_ROOT / ".env").items():
                if v is not None and v.strip() != "":
                    os.environ[k] = v.strip()

        model_path = Path(_get_str("WAKE_WORD_MODEL_PATH", "models/ren.ppn"))
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        lang_model_path = Path(
            _get_str("WAKE_WORD_LANGUAGE_MODEL_PATH", "models/porcupine_params_pt.pv")
        )
        if not lang_model_path.is_absolute():
            lang_model_path = PROJECT_ROOT / lang_model_path

        silero_path = Path(_get_str("SILERO_MODEL_PATH", "models/silero_vad.onnx"))
        if not silero_path.is_absolute():
            silero_path = PROJECT_ROOT / silero_path

        vosk_path = Path(
            _get_str("VOSK_MODEL_PATH", "models/vosk-model-small-pt-0.3")
        )
        if not vosk_path.is_absolute():
            vosk_path = PROJECT_ROOT / vosk_path

        api_key = _get_str("GEMINI_API_KEY") or _get_str("GOOGLE_API_KEY")
        if api_key and "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
            os.environ.pop("GOOGLE_API_KEY", None)

        obsidian_raw = _get_str("OBSIDIAN_VAULT_PATH")
        obsidian_path = Path(obsidian_raw).expanduser() if obsidian_raw else None

        return cls(
            gemini_api_key=api_key,
            gemini_model=_get_str(
                "GEMINI_MODEL", "gemini-3.8-live-extended-thinking"
            ),
            gemini_voice=_get_str("GEMINI_VOICE", "Aoede"),
            picovoice_access_key=_get_str("PICOVOICE_ACCESS_KEY") or None,
            wake_word_model_path=model_path,
            wake_word_builtin=_get_str("WAKE_WORD_BUILTIN", "jarvis"),
            wake_word_sensitivity=_get_float("WAKE_WORD_SENSITIVITY", 0.6),
            vad_silence_threshold_rms=_get_float("VAD_SILENCE_THRESHOLD_RMS", 500.0),
            vad_silence_timeout_sec=_get_float("VAD_SILENCE_TIMEOUT_SEC", 6.0),
            half_duplex=_get_bool("HALF_DUPLEX", True),
            sound_fx_enabled=_get_bool("SOUND_FX_ENABLED", True),
            wake_word_language_model_path=lang_model_path,
            vosk_model_path=vosk_path,
            silero_model_path=silero_path,
            barge_in_enabled=_get_bool("BARGE_IN_ENABLED", True),
            gateway_enabled=_get_bool("GATEWAY_ENABLED", True),
            gateway_port=int(_get_float("GATEWAY_PORT", 8765)),
            continuous_vision_enabled=_get_bool("CONTINUOUS_VISION_ENABLED", True),
            discord_bot_token=_get_str("DISCORD_BOT_TOKEN") or None,
            discord_user_id=int(_get_str("DISCORD_USER_ID")) if _get_str("DISCORD_USER_ID") else None,
            discord_text_channel_id=int(_get_str("DISCORD_TEXT_CHANNEL_ID")) if _get_str("DISCORD_TEXT_CHANNEL_ID") else None,
            discord_default_channel=_get_str("DISCORD_DEFAULT_CHANNEL", "Geral"),
            obsidian_vault_path=obsidian_path,
            thinking_level=_get_str("THINKING_LEVEL", "LOW").upper(),
            safety_threshold=_get_str("SAFETY_THRESHOLD", "BLOCK_ONLY_HIGH").upper(),
        )

    def validate(self) -> None:
        """Falha cedo com mensagem clara se faltar algo obrigatório."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY não definida. Copie .env.example para .env e preencha."
            )
        if not 0.0 <= self.wake_word_sensitivity <= 1.0:
            raise ValueError("WAKE_WORD_SENSITIVITY deve estar entre 0 e 1.")
        if self.vad_silence_timeout_sec <= 0:
            raise ValueError("VAD_SILENCE_TIMEOUT_SEC deve ser positivo.")
        if self.discord_text_channel_id is not None and self.discord_text_channel_id <= 0:
            raise ValueError("DISCORD_TEXT_CHANNEL_ID deve ser um ID numérico positivo.")
        if self.thinking_level.upper() not in _VALID_THINKING_LEVELS:
            raise ValueError(
                f"THINKING_LEVEL inválido ({self.thinking_level!r}). "
                f"Use um de: {sorted(_VALID_THINKING_LEVELS)}."
            )
        if self.safety_threshold.upper() not in _VALID_SAFETY_THRESHOLDS:
            raise ValueError(
                f"SAFETY_THRESHOLD inválido ({self.safety_threshold!r}). "
                f"Use um de: {sorted(_VALID_SAFETY_THRESHOLDS)}."
            )
        if self.obsidian_vault_path is not None and not self.obsidian_vault_path.is_dir():
            raise ValueError(
                f"OBSIDIAN_VAULT_PATH não encontrado ou não é um diretório: {self.obsidian_vault_path}"
            )

