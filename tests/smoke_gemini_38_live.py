"""Teste manual curto do modelo Live alternativo; não altera o .env."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace

from google import genai
from google.genai import types

from src.config import Settings
from src.gemini.session import GeminiLiveSession


async def main() -> None:
    settings = replace(Settings.from_env(), gemini_model="gemini-3.8-live")
    client = genai.Client(api_key=settings.gemini_api_key, http_options={"api_version": "v1beta"})
    model = "gemini-3.8-live"
    config = types.LiveConnectConfig(response_modalities=[types.Modality.AUDIO])
    if "--full" in sys.argv:
        hermes = GeminiLiveSession(settings, stream_manager=None, client=client)  # type: ignore[arg-type]
        config = hermes.build_config()
    if "--no-language" in sys.argv and config.speech_config is not None:
        config.speech_config.language_code = None
    if "--no-tools" in sys.argv:
        config.tools = None
    if "--search-only" in sys.argv:
        config.tools = [types.Tool(google_search=types.GoogleSearch())]
    if "--functions-only" in sys.argv:
        config.tools = [tool for tool in config.tools or [] if tool.function_declarations]
    if "--no-prompt" in sys.argv:
        config.system_instruction = None
    if "--no-speech" in sys.argv:
        config.speech_config = None
    if "--no-realtime" in sys.argv:
        config.realtime_input_config = None
    if "--no-transcription" in sys.argv:
        config.input_audio_transcription = None
        config.output_audio_transcription = None
    audio_bytes = 0
    transcription = ""
    async with client.aio.live.connect(model=model, config=config) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part.from_text(text="Responda apenas: Olá, teste de voz.")]),
            turn_complete=True,
        )
        async with asyncio.timeout(30):
            async for message in session.receive():
                content = getattr(message, "server_content", None)
                if content is None:
                    continue
                for part in getattr(getattr(content, "model_turn", None), "parts", None) or []:
                    inline = getattr(part, "inline_data", None)
                    audio_bytes += len(getattr(inline, "data", b"") or b"")
                output = getattr(content, "output_transcription", None)
                transcription += str(getattr(output, "text", "") or "")
                if getattr(content, "turn_complete", False):
                    break
    print(f"Modelo: {model}")
    print(f"Áudio PCM recebido: {audio_bytes} bytes")
    print(f"Transcrição: {transcription.strip()[:120]}")
    if audio_bytes == 0:
        raise RuntimeError("A sessão terminou sem áudio")


if __name__ == "__main__":
    asyncio.run(main())
