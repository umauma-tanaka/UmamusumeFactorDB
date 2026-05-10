from __future__ import annotations

import numpy as np

from umafactor.capture import window_capture
from umafactor.capture.window_capture import (
    WindowInfo,
    capture_window_frames,
    rank_window_candidates,
)
from umafactor.core.geometry import Rect


def _candidate_window(
    *,
    hwnd: int,
    title: str,
    class_name: str,
    width: int,
    height: int,
    process_name: str = "",
) -> WindowInfo:
    rect = Rect(10, 20, 10 + width, 20 + height)
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        class_name=class_name,
        window_rect=rect,
        client_rect=rect,
        process_id=hwnd * 10,
        process_name=process_name,
    )


def test_rank_window_candidates_prefers_unity_umamusume_window() -> None:
    windows = [
        _candidate_window(
            hwnd=1,
            title="Other Game",
            class_name="UnityWndClass",
            width=1280,
            height=720,
        ),
        _candidate_window(
            hwnd=2,
            title="Umamusume Pretty Derby",
            class_name="UnityWndClass",
            width=1920,
            height=1080,
            process_name="UmamusumePrettyDerby_Jpn.exe",
        ),
        _candidate_window(
            hwnd=3,
            title="umamusume notes",
            class_name="Notepad",
            width=1920,
            height=1080,
        ),
    ]

    ranked = rank_window_candidates(
        windows,
        title_keywords=("umamusume", "ウマ娘"),
        process_name_keywords=("UmamusumePrettyDerby_Jpn",),
        class_name="UnityWndClass",
        minimum_width=540,
        minimum_height=960,
    )

    assert [window.hwnd for window in ranked] == [2, 3]


def test_rank_window_candidates_rejects_small_windows() -> None:
    windows = [
        _candidate_window(
            hwnd=1,
            title="Umamusume Pretty Derby",
            class_name="UnityWndClass",
            width=800,
            height=600,
        )
    ]

    ranked = rank_window_candidates(
        windows,
        title_keywords=("umamusume",),
        process_name_keywords=("UmamusumePrettyDerby_Jpn",),
        class_name="UnityWndClass",
        minimum_width=540,
        minimum_height=960,
    )

    assert ranked == []


def test_rank_window_candidates_accepts_process_name_match() -> None:
    windows = [
        _candidate_window(
            hwnd=1,
            title="",
            class_name="UnityWindow",
            width=1920,
            height=1080,
            process_name="UmamusumePrettyDerby_Jpn.exe",
        )
    ]

    ranked = rank_window_candidates(
        windows,
        title_keywords=("umamusume",),
        process_name_keywords=("UmamusumePrettyDerby_Jpn",),
        class_name="UnityWndClass",
        minimum_width=540,
        minimum_height=960,
    )

    assert [window.hwnd for window in ranked] == [1]


def test_rank_window_candidates_accepts_steam_landscape_size() -> None:
    windows = [
        _candidate_window(
            hwnd=1,
            title="UmamusumePrettyDerby_Jpn",
            class_name="UnityWndClass",
            width=1385,
            height=779,
            process_name="UmamusumePrettyDerby_Jpn.exe",
        ),
        _candidate_window(
            hwnd=2,
            title="#ウマ娘 | Discord",
            class_name="Chrome_WidgetWin_1",
            width=160,
            height=28,
            process_name="Discord.exe",
        ),
    ]

    ranked = rank_window_candidates(
        windows,
        title_keywords=("umamusume", "ウマ娘"),
        process_name_keywords=("UmamusumePrettyDerby_Jpn",),
        class_name="UnityWndClass",
        minimum_width=480,
        minimum_height=360,
    )

    assert [window.hwnd for window in ranked] == [1]


def test_window_info_capture_rect_falls_back_to_window_rect() -> None:
    window = WindowInfo(
        hwnd=10,
        title="Umamusume Pretty Derby",
        class_name="UnityWndClass",
        window_rect=Rect(0, 0, 1920, 1080),
        client_rect=Rect(0, 0, 0, 0),
    )

    assert window.capture_rect("client") == Rect(0, 0, 1920, 1080)


class _FakeCaptureSession:
    def __init__(self, backend: str = "auto") -> None:
        self.backend = backend
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None

    def capture_rect(self, rect: Rect) -> np.ndarray:
        self.count += 1
        return np.full((rect.height, rect.width, 3), self.count, dtype=np.uint8)


def test_capture_window_frames_can_stop_before_opening_session(monkeypatch) -> None:
    opened = False

    class _UnexpectedSession(_FakeCaptureSession):
        def __init__(self, backend: str = "auto") -> None:
            nonlocal opened
            opened = True
            super().__init__(backend)

    monkeypatch.setattr(window_capture, "ScreenCaptureSession", _UnexpectedSession)

    frames = capture_window_frames(
        _capture_window(),
        duration_sec=1.0,
        fps=30.0,
        stop_requested=lambda: True,
    )

    assert frames == tuple()
    assert opened is False


def test_capture_window_frames_stops_after_button_callback(monkeypatch) -> None:
    stop = False
    progress_events: list[tuple[int, float]] = []

    def _progress(frame_count: int, elapsed_sec: float) -> None:
        nonlocal stop
        progress_events.append((frame_count, elapsed_sec))
        stop = True

    monkeypatch.setattr(window_capture, "ScreenCaptureSession", _FakeCaptureSession)

    frames = capture_window_frames(
        _capture_window(),
        duration_sec=1.0,
        fps=60.0,
        min_frame_diff=0.0,
        stop_requested=lambda: stop,
        progress_callback=_progress,
    )

    assert len(frames) == 1
    assert progress_events
    assert progress_events[0][0] == 1


def _capture_window() -> WindowInfo:
    rect = Rect(0, 0, 20, 10)
    return WindowInfo(
        hwnd=1,
        title="test",
        class_name="UnityWndClass",
        window_rect=rect,
        client_rect=rect,
    )
