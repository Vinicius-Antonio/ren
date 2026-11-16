"""Ferramentas de controle real do Windows: Spotify, YouTube, teclas de mídia/volume, apps, clipboard e status."""

from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# Códigos de teclas virtuais Win32 para controle global de mídia e volume
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_RETURN = 0x0D
VK_TAB = 0x09
KEYEVENTF_KEYUP = 0x0002

_SITE_SHORTCUTS: Dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "whatsapp": "https://web.whatsapp.com",
    "linkedin": "https://www.linkedin.com",
    "spotify": "https://open.spotify.com",
    "netflix": "https://www.netflix.com",
    "chatgpt": "https://chatgpt.com",
}


def _press_vk(vk_code: int, times: int = 1) -> None:
    """Dispara uma tecla virtual global no Windows via user32.dll."""
    user32 = getattr(ctypes, "windll", None)
    if user32 is None:
        return
    for _ in range(max(1, times)):
        user32.user32.keybd_event(vk_code, 0, 0, 0)
        user32.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
        if times > 1:
            time.sleep(0.03)


def _find_spotify_exe() -> Path | None:
    appdata = os.getenv("APPDATA", "")
    localappdata = os.getenv("LOCALAPPDATA", "")
    candidates = [
        Path(appdata) / "Spotify" / "Spotify.exe",
        Path(localappdata) / "Microsoft" / "WindowsApps" / "Spotify.exe",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _start_uri(uri: str) -> None:
    startfile = getattr(os, "startfile", None)
    if startfile is not None:
        startfile(uri)  # noqa: S606
    else:
        webbrowser.open(uri)


def _find_first_youtube_video_id(query: str) -> str | None:
    """Busca o primeiro videoId do YouTube para tocar a música direto sem clique."""
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", errors="ignore")
        match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        return match.group(1) if match else None
    except Exception:  # noqa: BLE001
        logger.debug("Falha ao buscar videoId no YouTube", exc_info=True)
        return None


def _auto_play_spotify_search() -> None:
    """Após o Spotify abrir a busca (`spotify:search:...`), envia Tab + Enter para tocar."""
    time.sleep(2.2)
    try:
        _press_vk(VK_TAB, times=1)
        time.sleep(0.15)
        _press_vk(VK_RETURN, times=1)
    except Exception:  # noqa: BLE001
        pass


def play_music(query: str, platform: str = "spotify") -> str:
    """Toca uma música, artista ou playlist no Spotify Desktop ou no YouTube."""
    cleaned_query = query.strip()
    plat = platform.strip().lower()

    if not cleaned_query:
        _press_vk(VK_MEDIA_PLAY_PAUSE)
        return "Comando Play/Pause enviado para o player de mídia."

    if "youtube" in plat:
        vid = _find_first_youtube_video_id(cleaned_query)
        if vid:
            watch_url = f"https://www.youtube.com/watch?v={vid}&autoplay=1"
            webbrowser.open(watch_url)
            return f"Tocando '{cleaned_query}' diretamente no YouTube ({watch_url})."
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(cleaned_query)}"
        webbrowser.open(search_url)
        return f"Pesquisa por '{cleaned_query}' aberta no YouTube."

    # Preferência: Spotify Desktop se instalado
    spotify_exe = _find_spotify_exe()
    if spotify_exe is not None:
        uri = f"spotify:search:{urllib.parse.quote(cleaned_query)}"
        _start_uri(uri)
        threading.Thread(target=_auto_play_spotify_search, daemon=True).start()
        return (
            f"Spotify aberto buscando e iniciando '{cleaned_query}'. "
            "Se não iniciar sozinho em 2 segundos, basta dar Enter na faixa."
        )

    # Fallback: se não houver Spotify Desktop, toca direto no YouTube ou Spotify Web
    vid = _find_first_youtube_video_id(cleaned_query)
    if vid:
        watch_url = f"https://www.youtube.com/watch?v={vid}&autoplay=1"
        webbrowser.open(watch_url)
        return f"Spotify Desktop não encontrado; tocando '{cleaned_query}' no YouTube."

    web_url = f"https://open.spotify.com/search/{urllib.parse.quote(cleaned_query)}"
    webbrowser.open(web_url)
    return f"Aberta busca por '{cleaned_query}' no Spotify Web."


def control_media(action: str, steps: int = 5) -> str:
    """Controla reprodução de mídia (play/pause, próxima, anterior) e volume do Windows."""
    act = action.strip().lower()
    if act in {"play", "pause", "play_pause", "pausar", "continuar", "retomar", "tocar"}:
        _press_vk(VK_MEDIA_PLAY_PAUSE)
        return "Reprodução alternada (Play/Pause)."
    if act in {"next", "proxima", "próxima", "pular", "avançar"}:
        _press_vk(VK_MEDIA_NEXT_TRACK)
        return "Pulou para a próxima faixa."
    if act in {"previous", "anterior", "voltar"}:
        _press_vk(VK_MEDIA_PREV_TRACK)
        return "Voltou para a faixa anterior."
    if act in {"stop", "parar"}:
        _press_vk(VK_MEDIA_STOP)
        return "Reprodução parada."
    if act in {"volume_up", "aumentar", "aumentar_volume", "mais_alto"}:
        _press_vk(VK_VOLUME_UP, times=max(1, min(steps, 25)))
        return f"Volume aumentado ({steps} níveis)."
    if act in {"volume_down", "abaixar", "diminuir", "abaixar_volume", "mais_baixo"}:
        _press_vk(VK_VOLUME_DOWN, times=max(1, min(steps, 25)))
        return f"Volume reduzido ({steps} níveis)."
    if act in {"mute", "mutar", "silenciar", "desmutar"}:
        _press_vk(VK_VOLUME_MUTE)
        return "Áudio mutado/desmutado."
    return f"Ação de mídia desconhecida: '{action}'."


def open_website_or_search(target: str, search_query: str = "") -> str:
    """Abre um site ou pesquisa no YouTube/Google no navegador padrão."""
    key = target.strip().lower()
    query = search_query.strip()

    if "youtube" in key and query:
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
    elif ("google" in key or not key) and query:
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
    elif key in _SITE_SHORTCUTS:
        url = _SITE_SHORTCUTS[key]
    elif key.startswith(("http://", "https://")):
        url = key
    elif "." in key and " " not in key:
        url = f"https://{key}"
    elif query:
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(f'{target} {query}')}"
    else:
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(target)}"

    webbrowser.open(url)
    return f"Aberto no navegador: {url}"


def open_application(app_name: str) -> str:
    """Abre um aplicativo do Windows (Spotify, Calculadora, Bloco de Notas, VS Code, etc.)."""
    key = app_name.strip().lower()

    if "spotify" in key:
        sp = _find_spotify_exe()
        if sp is not None:
            subprocess.Popen([str(sp)], shell=False)  # noqa: S603
            return "Aplicativo Spotify iniciado."
        _start_uri("spotify:")
        return "Comando para abrir Spotify enviado."

    if "config" in key or "ajustes" in key:
        _start_uri("ms-settings:")
        return "Configurações do Windows abertas."

    app_commands: Dict[str, list[str]] = {
        "calculadora": ["calc.exe"],
        "calc": ["calc.exe"],
        "bloco de notas": ["notepad.exe"],
        "notepad": ["notepad.exe"],
        "explorador": ["explorer.exe"],
        "arquivos": ["explorer.exe"],
        "paint": ["mspaint.exe"],
        "gerenciador de tarefas": ["taskmgr.exe"],
        "cmd": ["cmd.exe"],
        "terminal": ["wt.exe"],
        "powershell": ["powershell.exe"],
        "vscode": ["code"],
        "vs code": ["code"],
    }

    for name, cmd in app_commands.items():
        if name in key or key in name:
            try:
                subprocess.Popen(cmd, shell=False)  # noqa: S603
                return f"Aplicativo '{name}' iniciado com sucesso."
            except Exception as exc:  # noqa: BLE001
                return f"Erro ao abrir '{name}': {exc}"

    return (
        f"Aplicativo '{app_name}' não encontrado nos atalhos rápidos. "
        "Tente: Spotify, Calculadora, Bloco de Notas, Explorador de Arquivos, Paint, Terminal, VS Code."
    )


def read_clipboard() -> str:
    """Lê o texto atual da área de transferência (Ctrl+C)."""
    try:
        import tkinter as tk

        r = tk.Tk()
        r.withdraw()
        try:
            text = r.clipboard_get()
        except Exception:  # noqa: BLE001
            text = ""
        finally:
            r.destroy()
        cleaned = text.strip()
        if not cleaned:
            return "A área de transferência está vazia."
        return cleaned[:3000]
    except Exception as exc:  # noqa: BLE001
        return f"Não foi possível ler a área de transferência: {exc}"


def write_clipboard(text: str) -> str:
    """Copia um texto para a área de transferência do usuário (pronto para Ctrl+V)."""
    try:
        import tkinter as tk

        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return "Texto copiado para a área de transferência (pronto para colar com Ctrl+V)."
    except Exception as exc:  # noqa: BLE001
        return f"Não foi possível copiar para a área de transferência: {exc}"


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_pc_status() -> str:
    """Retorna o uso real de memória RAM e espaço em disco no Windows."""
    parts = []
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            total_gb = stat.ullTotalPhys / (1024**3)
            avail_gb = stat.ullAvailPhys / (1024**3)
            used_gb = total_gb - avail_gb
            parts.append(
                f"Memória RAM: {used_gb:.1f} GB em uso de {total_gb:.1f} GB "
                f"({stat.dwMemoryLoad}% ocupada, {avail_gb:.1f} GB livres)"
            )
    except Exception:  # noqa: BLE001
        pass

    try:
        disk = shutil.disk_usage("C:\\")
        free_gb = disk.free / (1024**3)
        total_disk_gb = disk.total / (1024**3)
        used_pct = ((disk.total - disk.free) / disk.total) * 100.0
        parts.append(
            f"Disco C: {free_gb:.1f} GB livres de {total_disk_gb:.0f} GB ({used_pct:.0f}% ocupado)"
        )
    except Exception:  # noqa: BLE001
        pass

    return " | ".join(parts) if parts else "Status do sistema indisponível."
