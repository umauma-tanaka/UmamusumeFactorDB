"""Windows game-window discovery and screen capture helpers."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import cv2
import numpy as np

from ..core.geometry import Rect
from .scraper_types import ScrollFrame


CaptureBackend = Literal["auto", "mss", "imagegrab"]
CaptureRegion = Literal["client", "window"]


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    window_rect: Rect
    client_rect: Rect
    process_id: int = 0
    process_name: str = ""

    @property
    def area(self) -> int:
        return max(self.window_rect.area, self.client_rect.area)

    def capture_rect(self, region: CaptureRegion = "client") -> Rect:
        rect = self.client_rect if region == "client" else self.window_rect
        return rect if not rect.is_empty else self.window_rect

    def to_dict(self) -> dict[str, object]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "class_name": self.class_name,
            "process_id": self.process_id,
            "process_name": self.process_name,
            "window_rect": list(self.window_rect.as_tuple()),
            "client_rect": list(self.client_rect.as_tuple()),
        }


def list_windows() -> list[WindowInfo]:
    """Return visible top-level Windows windows."""

    if not _is_windows():
        raise RuntimeError("window capture is only supported on Windows")
    _set_dpi_awareness()

    user32 = ctypes.windll.user32
    windows: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_proc(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _get_window_text(hwnd)
        class_name = _get_class_name(hwnd)
        process_id = _get_window_process_id(hwnd)
        process_name = _get_process_name(process_id)
        window_rect = _get_window_rect(hwnd)
        client_rect = _get_client_rect(hwnd)
        if window_rect.is_empty and client_rect.is_empty:
            return True
        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                class_name=class_name,
                window_rect=window_rect,
                client_rect=client_rect,
                process_id=process_id,
                process_name=process_name,
            )
        )
        return True

    user32.EnumWindows(enum_proc, 0)
    return windows


def find_game_window(
    *,
    title_keywords: Sequence[str] = ("umamusume", "ウマ娘"),
    process_name_keywords: Sequence[str] = ("UmamusumePrettyDerby_Jpn",),
    class_name: str = "UnityWndClass",
    minimum_width: int = 480,
    minimum_height: int = 360,
) -> WindowInfo:
    candidates = rank_window_candidates(
        list_windows(),
        title_keywords=title_keywords,
        process_name_keywords=process_name_keywords,
        class_name=class_name,
        minimum_width=minimum_width,
        minimum_height=minimum_height,
    )
    if not candidates:
        keywords = ", ".join(title_keywords)
        process_keywords = ", ".join(process_name_keywords)
        raise RuntimeError(
            "failed to find Steam Umamusume window "
            f"(class={class_name!r}, title_keywords=[{keywords}], "
            f"process_name_keywords=[{process_keywords}])"
        )
    return candidates[0]


def rank_window_candidates(
    windows: Sequence[WindowInfo],
    *,
    title_keywords: Sequence[str],
    process_name_keywords: Sequence[str] = (),
    class_name: str,
    minimum_width: int,
    minimum_height: int,
) -> list[WindowInfo]:
    scored: list[tuple[int, WindowInfo]] = []
    lowered_keywords = tuple(keyword.lower() for keyword in title_keywords if keyword)
    lowered_process_keywords = tuple(
        keyword.lower() for keyword in process_name_keywords if keyword
    )
    for window in windows:
        rect = window.capture_rect("client")
        if rect.width < minimum_width or rect.height < minimum_height:
            continue
        score = 0
        if class_name and window.class_name == class_name:
            score += 100
        title_lower = window.title.lower()
        if any(keyword in title_lower for keyword in lowered_keywords):
            score += 70
        process_lower = window.process_name.lower()
        if any(keyword in process_lower for keyword in lowered_process_keywords):
            score += 120
        if score <= 0:
            continue
        score += min(30, rect.area // 100_000)
        scored.append((score, window))
    scored.sort(key=lambda item: (item[0], item[1].area), reverse=True)
    return [window for _score, window in scored]


class ScreenCaptureSession:
    def __init__(self, backend: CaptureBackend = "auto") -> None:
        self.backend = backend
        self._mss = None
        self._mss_error: Exception | None = None
        if backend in {"auto", "mss"}:
            try:
                import mss

                self._mss = mss.mss()
            except Exception as exc:  # noqa: BLE001 - fallback is intentional.
                self._mss_error = exc
                if backend == "mss":
                    raise RuntimeError(f"mss capture backend is unavailable: {exc}") from exc

    @property
    def active_backend(self) -> str:
        return "mss" if self._mss is not None else "imagegrab"

    def close(self) -> None:
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    def __enter__(self) -> "ScreenCaptureSession":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def capture_rect(self, rect: Rect) -> np.ndarray:
        if rect.is_empty:
            raise ValueError(f"capture rect is empty: {rect}")
        if self._mss is not None:
            shot = self._mss.grab(
                {
                    "left": rect.x0,
                    "top": rect.y0,
                    "width": rect.width,
                    "height": rect.height,
                }
            )
            return np.asarray(shot, dtype=np.uint8)[:, :, :3].copy()
        return _capture_with_imagegrab(rect)


def capture_window_frames(
    window: WindowInfo,
    *,
    duration_sec: float,
    fps: float,
    backend: CaptureBackend = "auto",
    region: CaptureRegion = "client",
    min_frame_diff: float = 1.5,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
) -> tuple[ScrollFrame, ...]:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    capture_rect = window.capture_rect(region)
    if stop_requested is not None and stop_requested():
        return tuple()

    frames: list[ScrollFrame] = []
    last_kept: np.ndarray | None = None
    interval = 1.0 / fps
    started_at = time.perf_counter()
    deadline = started_at + duration_sec
    next_at = started_at
    index = 0
    with ScreenCaptureSession(backend) as session:
        while time.perf_counter() < deadline:
            if stop_requested is not None and stop_requested():
                break
            now = time.perf_counter()
            if now < next_at:
                time.sleep(min(next_at - now, interval))
                continue
            image = session.capture_rect(capture_rect)
            if last_kept is None or _frame_diff_mean(last_kept, image) >= min_frame_diff:
                frames.append(
                    ScrollFrame(
                        image=image,
                        frame_index=len(frames),
                        source_path=f"window:{window.hwnd}:{index}",
                    )
                )
                last_kept = image
                if progress_callback is not None:
                    progress_callback(len(frames), time.perf_counter() - started_at)
            index += 1
            next_at += interval
    return tuple(frames)


def _capture_with_imagegrab(rect: Rect) -> np.ndarray:
    from PIL import ImageGrab

    pil_image = ImageGrab.grab(bbox=rect.as_tuple(), all_screens=True)
    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _frame_diff_mean(previous: np.ndarray, current: np.ndarray) -> float:
    if previous.shape != current.shape:
        return float("inf")
    return float(np.mean(cv2.absdiff(previous, current)))


def _get_window_text(hwnd) -> str:
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _get_class_name(hwnd) -> str:
    user32 = ctypes.windll.user32
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def _get_window_process_id(hwnd) -> int:
    process_id = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    return int(process_id.value)


def _get_process_name(process_id: int) -> str:
    if process_id <= 0:
        return ""

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    process_query_limited_information = 0x1000
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        wintypes.DWORD(process_id),
    )
    if not handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
        if not ok:
            return ""
        return buffer.value.rsplit("\\", 1)[-1]
    finally:
        kernel32.CloseHandle(handle)


def _get_window_rect(hwnd) -> Rect:
    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return Rect(0, 0, 0, 0)
    return Rect(rect.left, rect.top, rect.right, rect.bottom)


def _get_client_rect(hwnd) -> Rect:
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return Rect(0, 0, 0, 0)
    top_left = wintypes.POINT(rect.left, rect.top)
    bottom_right = wintypes.POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return Rect(0, 0, 0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return Rect(0, 0, 0, 0)
    return Rect(top_left.x, top_left.y, bottom_right.x, bottom_right.y)


def _set_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001 - older Windows fallback.
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _is_windows() -> bool:
    return hasattr(ctypes, "windll")
