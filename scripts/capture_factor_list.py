"""Capture a Steam Umamusume factor list during manual scrolling.

The flow is intentionally manual for now:
1. Open the inheritance factor tab in the Steam game window.
2. Start this script and wait for the warmup countdown.
3. Slowly scroll the in-game factor list continuously until capture ends.

The captured frames are stitched with the static prototype and optionally sent
through factor-list OCR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_factor_ocr import _write_detected_csv, _write_overlay  # noqa: E402
from umafactor.capture.static_stitch import stitch_static_scroll_frames  # noqa: E402
from umafactor.capture.window_capture import (  # noqa: E402
    capture_window_frames,
    find_game_window,
    list_windows,
    rank_window_candidates,
)
from umafactor.detection.factor_list import (  # noqa: E402
    FactorListTile,
    detect_stitched_factor_list,
)
from umafactor.recognition.factor_list_ocr import (  # noqa: E402
    recognize_factor_list_tile_names,
)


DEFAULT_TITLE_KEYWORDS = ("umamusume", "ウマ娘")
DEFAULT_PROCESS_NAME_KEYWORDS = ("UmamusumePrettyDerby_Jpn",)
DEFAULT_CACHE_DIR = ROOT / "paddleocr_cache"


def main() -> int:
    _configure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_windows:
        _print_windows(args)
        return 0

    output_dir = _resolve_output_dir(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    window = _select_window(args)
    _write_json(output_dir / "window.json", window.to_dict())
    print(
        f"window: hwnd={window.hwnd} pid={window.process_id} "
        f"process={window.process_name!r} class={window.class_name!r} title={window.title!r}"
    )
    print(f"output: {output_dir}")

    if args.warmup > 0:
        print(f"warmup: {args.warmup:.1f}s")
        time.sleep(args.warmup)

    print(
        "capturing: "
        f"duration={args.duration:.1f}s fps={args.fps:.1f} "
        f"backend={args.backend} region={args.region}"
    )
    frames = capture_window_frames(
        window,
        duration_sec=args.duration,
        fps=args.fps,
        backend=args.backend,
        region=args.region,
        min_frame_diff=args.min_frame_diff,
    )
    if not frames:
        raise RuntimeError("no frames captured")
    _write_frames(output_dir / "frames", frames)

    stitch = stitch_static_scroll_frames(
        frames,
        use_scrollbar_hint=args.use_scrollbar_hint,
    )
    stitched_path = output_dir / "stitched.png"
    cv2.imwrite(str(stitched_path), stitch.image)
    _write_json(output_dir / "stitch_metadata.json", stitch.to_metadata())
    _write_json(
        output_dir / "capture_metadata.json",
        {
            "window": window.to_dict(),
            "frame_count": len(frames),
            "requested_backend": args.backend,
            "region": args.region,
            "duration_sec": args.duration,
            "fps": args.fps,
            "min_frame_diff": args.min_frame_diff,
            "use_scrollbar_hint": args.use_scrollbar_hint,
            "source_frames": [frame.to_metadata() for frame in frames],
            "stitched_path": str(stitched_path),
        },
    )

    if not args.skip_ocr:
        _run_factor_ocr(stitch.image, output_dir, args)

    print(f"frames: {len(frames)}")
    print(f"stitched: {stitched_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and stitch a live Steam Umamusume factor list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/live_capture/<timestamp>.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Capture duration in seconds after warmup.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Capture target FPS. Duplicate-like frames are filtered.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=3.0,
        help="Seconds to wait before capture starts.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mss", "imagegrab"],
        default="auto",
        help="Capture backend. auto prefers mss and falls back to Pillow ImageGrab.",
    )
    parser.add_argument(
        "--region",
        choices=["client", "window"],
        default="client",
        help="Capture the game client area or the whole window rectangle.",
    )
    parser.add_argument(
        "--min-frame-diff",
        type=float,
        default=1.5,
        help="Mean pixel difference required to keep a captured frame.",
    )
    parser.add_argument(
        "--window-class",
        default="UnityWndClass",
        help="Expected Win32 window class.",
    )
    parser.add_argument(
        "--window-title-keyword",
        action="append",
        default=list(DEFAULT_TITLE_KEYWORDS),
        help="Title keyword used for window selection. Can be passed multiple times.",
    )
    parser.add_argument(
        "--process-name-keyword",
        action="append",
        default=list(DEFAULT_PROCESS_NAME_KEYWORDS),
        help="Process executable keyword used for window selection. Can be passed multiple times.",
    )
    parser.add_argument(
        "--window-hwnd",
        type=int,
        default=None,
        help="Use an exact HWND from --list-windows output instead of ranked selection.",
    )
    parser.add_argument("--minimum-width", type=int, default=480)
    parser.add_argument("--minimum-height", type=int, default=360)
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List visible windows and ranked game candidates, then exit.",
    )
    parser.add_argument(
        "--use-scrollbar-hint",
        action="store_true",
        help="Use detected scrollbar thumb motion as an optional weak stitch hint.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Only capture and stitch. Do not run factor-list OCR.",
    )
    parser.add_argument(
        "--ocr-crop-target",
        choices=["name", "card"],
        default="card",
        help="Image region passed to OCR for each detected factor tile.",
    )
    parser.add_argument(
        "--crop-variant",
        choices=["current", "wide", "upper", "full"],
        default="current",
        help="Name crop variant used when --ocr-crop-target=name.",
    )
    parser.add_argument(
        "--paddle-cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Project-local PaddleOCR cache/model directory.",
    )
    parser.add_argument("--paddle-lang", default="japan")
    parser.add_argument(
        "--paddle-det-limit-side-len",
        type=int,
        default=128,
        help="PaddleOCR text_det_limit_side_len for per-card OCR.",
    )
    parser.add_argument(
        "--paddle-det-limit-type",
        choices=["min", "max"],
        default="min",
        help="PaddleOCR text_det_limit_type. min avoids shrinking card crops.",
    )
    parser.add_argument("--paddle-det-thresh", type=float, default=None)
    parser.add_argument("--paddle-det-box-thresh", type=float, default=None)
    parser.add_argument("--paddle-det-unclip-ratio", type=float, default=None)
    parser.add_argument("--paddle-rec-score-thresh", type=float, default=None)
    return parser


def _resolve_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ROOT / "outputs" / "live_capture" / timestamp
    return path if path.is_absolute() else ROOT / path


def _select_window(args: argparse.Namespace):
    if args.window_hwnd is not None:
        for window in list_windows():
            if window.hwnd == args.window_hwnd:
                return window
        raise RuntimeError(f"failed to find window by hwnd: {args.window_hwnd}")
    return find_game_window(
        title_keywords=args.window_title_keyword,
        process_name_keywords=args.process_name_keyword,
        class_name=args.window_class,
        minimum_width=args.minimum_width,
        minimum_height=args.minimum_height,
    )


def _print_windows(args: argparse.Namespace) -> None:
    windows = list_windows()
    candidates = rank_window_candidates(
        windows,
        title_keywords=args.window_title_keyword,
        process_name_keywords=args.process_name_keyword,
        class_name=args.window_class,
        minimum_width=args.minimum_width,
        minimum_height=args.minimum_height,
    )
    print("# ranked candidates")
    for index, window in enumerate(candidates):
        print(_format_window(index, window))
    print("# visible windows")
    visible = sorted(windows, key=lambda item: item.area, reverse=True)
    for index, window in enumerate(visible[:80]):
        print(_format_window(index, window))


def _format_window(index: int, window) -> str:
    rect = window.capture_rect("client")
    return (
        f"{index:02d} hwnd={window.hwnd} pid={window.process_id} "
        f"process={window.process_name!r} size={rect.width}x{rect.height} "
        f"class={window.class_name!r} title={window.title!r}"
    )


def _write_frames(output_dir: Path, frames: Sequence[Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        cv2.imwrite(str(output_dir / f"frame_{frame.frame_index:04d}.png"), frame.image)


def _run_factor_ocr(image, output_dir: Path, args: argparse.Namespace) -> None:
    from umafactor.recognition.paddle_ocr_adapter import PaddleFactorOCR

    ocr = PaddleFactorOCR(
        lang=args.paddle_lang,
        mode="ocr",
        cache_dir=_resolve_path(args.paddle_cache_dir),
        text_det_limit_side_len=args.paddle_det_limit_side_len,
        text_det_limit_type=args.paddle_det_limit_type,
        text_det_thresh=args.paddle_det_thresh,
        text_det_box_thresh=args.paddle_det_box_thresh,
        text_det_unclip_ratio=args.paddle_det_unclip_ratio,
        text_rec_score_thresh=args.paddle_rec_score_thresh,
    )
    role_tiles: dict[str, Sequence[FactorListTile]] = {}
    for section_index in range(3):
        try:
            detection = detect_stitched_factor_list(image, section_index=section_index)
        except IndexError:
            continue
        tiles = recognize_factor_list_tile_names(
            image,
            detection.tiles,
            ocr,
            crop_variant=args.crop_variant,
            crop_target=args.ocr_crop_target,
        )
        role_tiles[detection.role] = tiles
        _write_detected_csv(output_dir / f"detected_{detection.role}_factors.csv", tiles)
        _write_overlay(
            output_dir / f"{detection.role}_overlay.png",
            image,
            tiles,
            crop_target=args.ocr_crop_target,
            crop_variant=args.crop_variant,
        )
    _write_json(
        output_dir / "ocr_metadata.json",
        {
            "roles": {role: len(tiles) for role, tiles in role_tiles.items()},
            "ocr_engine": "paddleocr",
            "ocr_crop_target": args.ocr_crop_target,
            "crop_variant": args.crop_variant,
            "paddle_cache_dir": str(_resolve_path(args.paddle_cache_dir)),
            "paddle_params": {
                "text_det_limit_side_len": args.paddle_det_limit_side_len,
                "text_det_limit_type": args.paddle_det_limit_type,
                "text_det_thresh": args.paddle_det_thresh,
                "text_det_box_thresh": args.paddle_det_box_thresh,
                "text_det_unclip_ratio": args.paddle_det_unclip_ratio,
                "text_rec_score_thresh": args.paddle_rec_score_thresh,
            },
        },
    )


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
