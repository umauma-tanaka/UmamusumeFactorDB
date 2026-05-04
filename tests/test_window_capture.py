from __future__ import annotations

from umafactor.capture.window_capture import WindowInfo, rank_window_candidates
from umafactor.core.geometry import Rect


def _window(
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
        _window(
            hwnd=1,
            title="Other Game",
            class_name="UnityWndClass",
            width=1280,
            height=720,
        ),
        _window(
            hwnd=2,
            title="Umamusume Pretty Derby",
            class_name="UnityWndClass",
            width=1920,
            height=1080,
            process_name="UmamusumePrettyDerby_Jpn.exe",
        ),
        _window(
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
        _window(
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
        _window(
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
        _window(
            hwnd=1,
            title="UmamusumePrettyDerby_Jpn",
            class_name="UnityWndClass",
            width=1385,
            height=779,
            process_name="UmamusumePrettyDerby_Jpn.exe",
        ),
        _window(
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
