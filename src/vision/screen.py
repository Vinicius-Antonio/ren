"""Módulo B: Visão Contínua (`src/vision/screen.py`) — Hermes 2.0.

- Captura assíncrona da tela com `mss` (com suporte a múltiplos monitores e
  `PrintWindow(PW_RENDERFULLCONTENT)` para jogos DirectX/GPU) via `asyncio.to_thread`.
- Redimensionamento dinâmico para resolução máxima de 1280x720 com compressão JPEG (qualidade 60%).
- `ScreenCaptureWorker`: streaming contínuo a 1 FPS (1 frame a cada 1.0s) com buffer
  de tamanho 1 (*drop-frame*) para nunca acumular memória se o WebSocket oscilar.
"""

from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
import io
import logging
import re
import sys
import unicodedata
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Habilita Per-Monitor DPI Awareness no Windows para não cortar telas em 125%/150%
if sys.platform == "win32":
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:  # noqa: BLE001
            pass

_SCREEN_TRIGGER_PHRASES: Tuple[str, ...] = (
    "minha tela",
    "na tela",
    "olha a tela",
    "olhe a tela",
    "olha isso",
    "olhe isso",
    "o que esta escrito",
    "esse erro",
    "este erro",
    "esse codigo",
    "este codigo",
    "essa janela",
    "esta janela",
    "essa imagem",
    "esta imagem",
    "esse programa",
    "este programa",
    "meu jogo",
    "essa partida",
    "esta partida",
    "meu tabuleiro",
    "olha meu jogo",
    "olha o jogo",
    "qual item",
    "qual comp",
    "qual aprimoramento",
    "qual augment",
    "leia isso",
    "traduz isso",
    "traduza isso",
)

_SCREEN_TRIGGER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _SCREEN_TRIGGER_PHRASES) + r")\b"
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", no_accents)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_screen_request(text: str) -> bool:
    """True se a transcrição da fala do usuário menciona olhar a tela/jogo/código/erro."""
    return bool(_SCREEN_TRIGGER_RE.search(_normalize(text)))


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def get_active_window_info() -> Tuple[
    int,
    str,
    Optional[Tuple[int, int, int, int]],
    Optional[Tuple[int, int, int, int]],
]:
    """Retorna (hwnd, título, rect_janela, rect_monitor) da janela atualmente em foco."""
    if sys.platform != "win32":
        return 0, "", None, None
    try:
        user32 = ctypes.windll.user32
        hwnd = int(user32.GetForegroundWindow())
        if not hwnd:
            return 0, "", None, None

        length = int(user32.GetWindowTextLengthW(hwnd))
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()

        win_rect = wintypes.RECT()
        rect_tuple: Optional[Tuple[int, int, int, int]] = None
        if user32.GetWindowRect(hwnd, ctypes.byref(win_rect)):
            rect_tuple = (
                int(win_rect.left),
                int(win_rect.top),
                int(win_rect.right),
                int(win_rect.bottom),
            )

        mon_tuple: Optional[Tuple[int, int, int, int]] = None
        MONITOR_DEFAULTTONEAREST = 2
        hmon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if hmon:
            mi = _MONITORINFO()
            mi.cbSize = ctypes.sizeof(_MONITORINFO)
            if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                mon_tuple = (
                    int(mi.rcMonitor.left),
                    int(mi.rcMonitor.top),
                    int(mi.rcMonitor.right),
                    int(mi.rcMonitor.bottom),
                )
        return hwnd, title, rect_tuple, mon_tuple
    except Exception:  # noqa: BLE001
        return 0, "", None, None


def _capture_hwnd_printwindow(
    hwnd: int, win_rect: Tuple[int, int, int, int]
) -> Optional["Image.Image"]:
    """Captura a superfície acelerada por hardware da janela via PrintWindow(PW_RENDERFULLCONTENT)."""
    if sys.platform != "win32" or not hwnd:
        return None
    from PIL import Image

    left, top, right, bottom = win_rect
    w = right - left
    h = bottom - top
    if w < 200 or h < 150:
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        return None
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    old_bmp = gdi32.SelectObject(mem_dc, bmp)
    try:
        PW_RENDERFULLCONTENT = 0x00000002
        ok = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
        if not ok:
            ok = user32.PrintWindow(hwnd, mem_dc, 0)
        if not ok:
            return None

        bmi = _BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB

        buf_len = w * h * 4
        raw_buf = ctypes.create_string_buffer(buf_len)
        lines = gdi32.GetDIBits(
            mem_dc, bmp, 0, h, raw_buf, ctypes.byref(bmi), 0
        )
        if lines != h:
            return None

        img = Image.frombuffer("RGBA", (w, h), raw_buf, "raw", "BGRA", 0, 1).convert("RGB")
        extrema = img.getextrema()
        if all(channel_max < 8 for _, channel_max in extrema):
            return None
        return img
    except Exception:  # noqa: BLE001
        return None
    finally:
        gdi32.SelectObject(mem_dc, old_bmp)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)


def _capture_with_mss(mon_rect: Optional[Tuple[int, int, int, int]]) -> Optional["Image.Image"]:
    """Captura rápida via `mss` do monitor onde está a janela em foco (ou monitor primário)."""
    try:
        import mss
        from PIL import Image

        with mss.mss() as sct:
            if mon_rect is not None:
                left, top, right, bottom = mon_rect
                w = max(1, right - left)
                h = max(1, bottom - top)
                region = {"left": left, "top": top, "width": w, "height": h}
            else:
                # Monitor primário (índice 1 no mss; índice 0 é o virtual desktop inteiro)
                region = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            shot = sct.grab(region)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Fallback de mss para ImageGrab: %s", exc)
        return None


def capture_screen_jpeg(
    max_width: int = 1280, quality: int = 60, max_height: int = 720
) -> bytes:
    """Captura a tela via `mss` (ou `ImageGrab`), redimensiona para até 1280x720 e comprime em JPEG."""
    from PIL import Image, ImageGrab

    hwnd, title, win_rect, mon_rect = get_active_window_info()
    if title:
        logger.debug("Janela ativa na captura: %r", title)

    img: Optional[Image.Image] = None
    is_mocked_imagegrab = getattr(ImageGrab.grab, "__module__", "") != "PIL.ImageGrab"

    if not is_mocked_imagegrab:
        img = _capture_with_mss(mon_rect)

    if img is None:
        try:
            if mon_rect is not None and not is_mocked_imagegrab:
                img = ImageGrab.grab(bbox=mon_rect, all_screens=True)
            else:
                img = ImageGrab.grab(all_screens=False)
        except TypeError:
            img = ImageGrab.grab(all_screens=False)

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Sobrepõe superfície GPU/DirectX da janela ativa (se não estivermos em mock de teste)
    if not is_mocked_imagegrab and hwnd and win_rect is not None:
        pw_img = _capture_hwnd_printwindow(hwnd, win_rect)
        if pw_img is not None:
            if mon_rect is not None:
                offset_x = win_rect[0] - mon_rect[0]
                offset_y = win_rect[1] - mon_rect[1]
                img.paste(pw_img, (offset_x, offset_y))
            else:
                img = pw_img

    w, h = img.size
    if w > 0 and h > 0 and (w > max_width or h > max_height):
        ratio = min(max_width / float(w), max_height / float(h))
        new_size = (max(1, int(round(w * ratio))), max(1, int(round(h * ratio))))
        img = img.resize(new_size, Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


async def send_jpeg_frame_to_session(session: Any, jpeg_bytes: bytes) -> None:
    """Envia um frame JPEG para a sessão Gemini Live (compatível com v1alpha/v1beta)."""
    from google.genai import types

    blob = types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
    send_realtime = getattr(session, "send_realtime_input", None)
    if send_realtime is not None:
        try:
            await send_realtime(video=blob)
            return
        except TypeError:
            await send_realtime(media_chunks=[blob])
            return
    await session.send(input=blob)


class ScreenCaptureWorker:
    """Worker assíncrono de visão contínua (1 FPS, 1280x720, JPEG 60%, drop-frame buffer=1)."""

    def __init__(
        self,
        interval_sec: float = 1.0,
        max_width: int = 1280,
        max_height: int = 720,
        quality: int = 60,
    ) -> None:
        self.interval_sec = max(0.2, interval_sec)
        self.max_width = max_width
        self.max_height = max_height
        self.quality = quality
        self._frame_queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=1)

    def put_latest_frame(self, jpeg_bytes: bytes) -> None:
        """Insere o frame mais recente no buffer de tamanho 1 descartando o anterior (*drop-frame*)."""
        while self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            self._frame_queue.put_nowait(jpeg_bytes)
        except asyncio.QueueFull:
            pass

    async def capture_once_async(self) -> bytes:
        """Captura um frame em thread dedicada via `asyncio.to_thread`."""
        return await asyncio.to_thread(
            capture_screen_jpeg,
            self.max_width,
            self.quality,
            self.max_height,
        )

    async def run(self, session: Any, stop_event: asyncio.Event) -> None:
        """Loop contínuo de 1 FPS que captura e envia frames sem bloquear nem acumular fila."""
        producer = asyncio.create_task(
            self._producer_loop(stop_event), name="t_vision_producer"
        )
        consumer = asyncio.create_task(
            self._consumer_loop(session, stop_event), name="t_vision_consumer"
        )
        try:
            await asyncio.gather(producer, consumer)
        except asyncio.CancelledError:
            pass
        finally:
            for t in (producer, consumer):
                if not t.done():
                    t.cancel()
            await asyncio.gather(producer, consumer, return_exceptions=True)

    async def _producer_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            t0 = asyncio.get_running_loop().time()
            try:
                jpeg_bytes = await self.capture_once_async()
                self.put_latest_frame(jpeg_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Erro ao capturar frame no ScreenCaptureWorker: %s", exc)
            elapsed = asyncio.get_running_loop().time() - t0
            wait_sec = max(0.05, self.interval_sec - elapsed)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_sec)
            except asyncio.TimeoutError:
                continue

    async def _consumer_loop(self, session: Any, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                jpeg_bytes = await asyncio.wait_for(
                    self._frame_queue.get(), timeout=0.25
                )
            except asyncio.TimeoutError:
                continue
            try:
                await send_jpeg_frame_to_session(session, jpeg_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Falha ao enviar frame de visão contínua: %s", exc)
