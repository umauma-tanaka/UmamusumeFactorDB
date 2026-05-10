"""Small topmost control window for live manual capture."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol


class CaptureControl(Protocol):
    def set_waiting(self, *, output_dir: Path, warmup_sec: float) -> None:
        ...

    def wait_warmup(self, seconds: float) -> bool:
        ...

    def start_capture(self, *, duration_sec: float, fps: float) -> None:
        ...

    def update_capture(self, *, frame_count: int, elapsed_sec: float) -> None:
        ...

    def stop_requested(self) -> bool:
        ...

    def mark_processing(self, message: str) -> None:
        ...

    def mark_done(self, message: str) -> None:
        ...

    def mark_error(self, message: str) -> None:
        ...

    def hold(self, seconds: float) -> None:
        ...

    def close(self) -> None:
        ...


class NullCaptureControl:
    def set_waiting(self, *, output_dir: Path, warmup_sec: float) -> None:
        return None

    def wait_warmup(self, seconds: float) -> bool:
        if seconds > 0:
            time.sleep(seconds)
        return True

    def start_capture(self, *, duration_sec: float, fps: float) -> None:
        return None

    def update_capture(self, *, frame_count: int, elapsed_sec: float) -> None:
        return None

    def stop_requested(self) -> bool:
        return False

    def mark_processing(self, message: str) -> None:
        return None

    def mark_done(self, message: str) -> None:
        return None

    def mark_error(self, message: str) -> None:
        return None

    def hold(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def close(self) -> None:
        return None


class TkCaptureControlWindow:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("UmamusumeFactorDB Capture")
        self._root.geometry("380x170+24+64")
        self._root.resizable(False, False)
        self._root.attributes("-topmost", True)
        self._root.protocol("WM_DELETE_WINDOW", self._request_stop)

        self._stop_requested = False
        self._status_var = tk.StringVar(value="準備中")
        self._detail_var = tk.StringVar(value="")
        self._output_var = tk.StringVar(value="")

        frame = ttk.Frame(self._root, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, textvariable=self._status_var, font=("", 15, "bold")).pack(
            anchor="w"
        )
        ttk.Label(frame, textvariable=self._detail_var, justify="left").pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(frame, textvariable=self._output_var, justify="left").pack(
            anchor="w", pady=(6, 0)
        )
        self._button = ttk.Button(frame, text="キャンセル", command=self._request_stop)
        self._button.pack(fill="x", pady=(12, 0))
        self._pump()

    def set_waiting(self, *, output_dir: Path, warmup_sec: float) -> None:
        self._status_var.set("キャプチャ準備中")
        self._detail_var.set(f"{warmup_sec:.1f} 秒後にキャプチャを開始します。")
        self._output_var.set(f"出力: {output_dir}")
        self._button.configure(text="キャンセル")
        self._pump()

    def wait_warmup(self, seconds: float) -> bool:
        deadline = time.perf_counter() + max(0.0, seconds)
        while not self._stop_requested:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            self._detail_var.set(f"{remaining:.1f} 秒後にキャプチャを開始します。")
            self._pump()
            time.sleep(min(0.05, remaining))
        return not self._stop_requested

    def start_capture(self, *, duration_sec: float, fps: float) -> None:
        self._status_var.set("キャプチャ中")
        self._detail_var.set(
            f"ゆっくりスクロールしてください。最大 {duration_sec:.1f} 秒 / {fps:.1f} fps"
        )
        self._button.configure(text="スクロール終了 / 停止")
        self._pump()

    def update_capture(self, *, frame_count: int, elapsed_sec: float) -> None:
        self._detail_var.set(
            f"スクロール中です。終了したら下のボタンを押してください。\n"
            f"経過 {elapsed_sec:.1f} 秒 / 保存フレーム {frame_count}"
        )
        self._pump()

    def stop_requested(self) -> bool:
        self._pump()
        return self._stop_requested

    def mark_processing(self, message: str) -> None:
        self._status_var.set("キャプチャ終了")
        self._detail_var.set(message)
        self._button.configure(text="処理中", state="disabled")
        self._pump()

    def mark_done(self, message: str) -> None:
        self._status_var.set("完了")
        self._detail_var.set(message)
        self._button.configure(text="閉じる", state="normal", command=self.close)
        self._pump()

    def mark_error(self, message: str) -> None:
        self._status_var.set("エラー")
        self._detail_var.set(message)
        self._button.configure(text="閉じる", state="normal", command=self.close)
        self._pump()

    def hold(self, seconds: float) -> None:
        deadline = time.perf_counter() + max(0.0, seconds)
        while time.perf_counter() < deadline:
            self._pump()
            time.sleep(0.05)

    def close(self) -> None:
        try:
            self._root.destroy()
        except Exception:
            pass

    def _request_stop(self) -> None:
        self._stop_requested = True
        self._status_var.set("停止要求を受け付けました")
        self._detail_var.set("現在のフレームまででキャプチャを終了します。")
        self._button.configure(text="停止中", state="disabled")
        self._pump()

    def _pump(self) -> None:
        try:
            self._root.lift()
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            self._stop_requested = True


def create_capture_control(*, enabled: bool) -> CaptureControl:
    if not enabled:
        return NullCaptureControl()
    try:
        return TkCaptureControlWindow()
    except Exception:
        return NullCaptureControl()
