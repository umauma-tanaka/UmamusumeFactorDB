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
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from umafactor.capture.static_stitch import (  # noqa: E402
    detect_dynamic_roi,
    stitch_static_scroll_frames,
)
from umafactor.capture.control_window import create_capture_control  # noqa: E402
from umafactor.capture.window_capture import (  # noqa: E402
    capture_window_frames,
    find_game_window,
    list_windows,
    rank_window_candidates,
)
from umafactor.recognition.ocr_protocol import (  # noqa: E402
    DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
    DEFAULT_OCR_MAX_UPSCALE,
    DEFAULT_OCR_MIN_HEIGHT,
    DEFAULT_OCR_MIN_WIDTH,
    DEFAULT_OCR_SHARPEN_STRENGTH,
)


DEFAULT_TITLE_KEYWORDS = ("umamusume", "\u30a6\u30de\u5a18")
DEFAULT_PROCESS_NAME_KEYWORDS = ("UmamusumePrettyDerby_Jpn",)
DEFAULT_CACHE_DIR = ROOT / "paddleocr_cache"
DEFAULT_RAPIDOCR_MODEL_DIR = ROOT / "rapidocr_models"


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

    control = create_capture_control(enabled=not args.no_control_window)
    control.set_waiting(output_dir=output_dir, warmup_sec=args.warmup)
    try:
        if args.warmup > 0:
            print(f"warmup: {args.warmup:.1f}s")
            if not control.wait_warmup(args.warmup):
                print("capture cancelled before start")
                return 130

        print(
            "capturing: "
            f"duration={args.duration:.1f}s fps={args.fps:.1f} "
            f"backend={args.backend} region={args.region}"
        )
        control.start_capture(duration_sec=args.duration, fps=args.fps)
        frames = capture_window_frames(
            window,
            duration_sec=args.duration,
            fps=args.fps,
            backend=args.backend,
            region=args.region,
            min_frame_diff=args.min_frame_diff,
            stop_requested=control.stop_requested,
            progress_callback=lambda frame_count, elapsed_sec: control.update_capture(
                frame_count=frame_count,
                elapsed_sec=elapsed_sec,
            ),
        )
        stopped_by_user = control.stop_requested()
        if not frames:
            raise RuntimeError("no frames captured")
        control.mark_processing("キャプチャを終了しました。画像を書き出しています。")
        _write_frames(output_dir / "frames", frames)
        if args.debug:
            _write_capture_debug(output_dir / "debug", frames, roi_limit=args.debug_roi_limit)

        control.mark_processing("スクロール画像を結合しています。")
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
                "stopped_by_user": stopped_by_user,
                "source_frames": [frame.to_metadata() for frame in frames],
                "stitched_path": str(stitched_path),
            },
        )

        if not args.skip_ocr:
            control.mark_processing("OCRを実行しています。")
            _run_factor_ocr(stitch.image, output_dir, args)

        print(f"frames: {len(frames)}")
        print(f"stitched: {stitched_path}")
        control.mark_done(f"処理が完了しました。\n出力: {output_dir}")
        control.hold(args.control_window_hold_seconds)
        return 0
    except Exception as exc:
        control.mark_error(str(exc))
        control.hold(args.control_window_hold_seconds)
        raise
    finally:
        control.close()


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
        "--no-control-window",
        action="store_true",
        help="Disable the topmost capture status/stop window.",
    )
    parser.add_argument(
        "--control-window-hold-seconds",
        type=float,
        default=2.0,
        help="Seconds to keep the control window visible after completion or error.",
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
        "--debug",
        action="store_true",
        help="Write dynamic ROI masks and cropped ROI samples for stitch debugging.",
    )
    parser.add_argument(
        "--debug-roi-limit",
        type=int,
        default=5,
        help="Number of ROI crop samples to write when --debug is enabled.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Only capture and stitch. Do not run factor-list OCR.",
    )
    parser.add_argument(
        "--submitter-id",
        default="live-capture",
        help="Submitter id written to factor_list_submission.json.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["paddle", "rapidocr"],
        default="rapidocr",
        help="OCR engine used for factor-list recognition.",
    )
    parser.add_argument(
        "--ocr-crop-target",
        choices=["name", "card"],
        default="name",
        help="Image region passed to OCR for each detected factor tile.",
    )
    parser.add_argument(
        "--crop-variant",
        choices=["body_name", "current", "wide", "upper", "full"],
        default="body_name",
        help="Name crop variant used when --ocr-crop-target=name.",
    )
    parser.add_argument(
        "--disable-ocr-preprocess",
        action="store_true",
        help="Disable OCR crop upscaling, contrast normalization, and sharpening.",
    )
    parser.add_argument(
        "--ocr-min-crop-width",
        type=int,
        default=DEFAULT_OCR_MIN_WIDTH,
        help="Minimum OCR crop width after automatic upscaling.",
    )
    parser.add_argument(
        "--ocr-min-crop-height",
        type=int,
        default=DEFAULT_OCR_MIN_HEIGHT,
        help="Minimum OCR crop height after automatic upscaling.",
    )
    parser.add_argument(
        "--ocr-max-upscale",
        type=float,
        default=DEFAULT_OCR_MAX_UPSCALE,
        help="Maximum OCR crop upscale factor.",
    )
    parser.add_argument(
        "--ocr-sharpen-strength",
        type=float,
        default=DEFAULT_OCR_SHARPEN_STRENGTH,
        help="Unsharp mask strength applied to OCR crops. 0 disables sharpening.",
    )
    parser.add_argument(
        "--ocr-contrast-clip-limit",
        type=float,
        default=DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
        help="CLAHE clip limit applied to OCR crop luminance. 0 disables contrast normalization.",
    )
    parser.add_argument(
        "--ocr-execution-mode",
        choices=["sequential", "canvas", "batch", "role_sheet"],
        default="batch",
        help="OCR execution strategy. batch packs prepared card crops into OCR batches.",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=12,
        help="Number of cards per OCR batch or canvas.",
    )
    parser.add_argument(
        "--ocr-canvas-padding",
        type=int,
        default=24,
        help="Padding in pixels between cards when --ocr-execution-mode=canvas.",
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
        help="PaddleOCR text_det_limit_side_len. Default preserves the prepared OCR canvas scale.",
    )
    parser.add_argument(
        "--paddle-det-limit-type",
        choices=["min", "max"],
        default="min",
        help="PaddleOCR text_det_limit_type. min avoids shrinking prepared OCR canvases.",
    )
    parser.add_argument("--paddle-det-thresh", type=float, default=None)
    parser.add_argument("--paddle-det-box-thresh", type=float, default=None)
    parser.add_argument("--paddle-det-unclip-ratio", type=float, default=None)
    parser.add_argument("--paddle-rec-score-thresh", type=float, default=None)
    parser.add_argument(
        "--rapidocr-model-root-dir",
        type=Path,
        default=DEFAULT_RAPIDOCR_MODEL_DIR,
        help="Project-local RapidOCR model directory.",
    )
    parser.add_argument("--rapidocr-text-score", type=float, default=None)
    parser.add_argument("--rapidocr-ocr-version", choices=["PP-OCRv4", "PP-OCRv5"], default="PP-OCRv4")
    parser.add_argument("--rapidocr-lang-type", choices=["japan", "ch"], default="japan")
    parser.add_argument("--rapidocr-model-type", choices=["mobile", "server"], default="mobile")
    parser.add_argument(
        "--rapidocr-rec-img-width",
        type=int,
        choices=[320, 480, 640],
        default=480,
        help="RapidOCR recognition image width; height is fixed at 48.",
    )
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


def _write_capture_debug(output_dir: Path, frames: Sequence[Any], *, roi_limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    roi = detect_dynamic_roi(frames)
    cv2.imwrite(str(output_dir / "dynamic_mask.png"), roi.mask)
    for frame in frames[: max(0, roi_limit)]:
        cv2.imwrite(str(output_dir / f"roi_{frame.frame_index:04d}.png"), roi.crop(frame.image))
    _write_json(
        output_dir / "debug_metadata.json",
        {
            "roi": list(roi.rect.as_tuple()),
            "roi_source": roi.source,
            "frame_count": len(frames),
            "roi_samples": min(max(0, roi_limit), len(frames)),
        },
    )


def _run_factor_ocr(image, output_dir: Path, args: argparse.Namespace) -> None:
    _run_factor_list_pipeline_ocr(image, output_dir, args)


def _run_factor_list_pipeline_ocr(image, output_dir: Path, args: argparse.Namespace) -> None:
    from umafactor.factor_list import FactorOcrOptions, recognize_factor_list_image, to_submission

    debug_dir = output_dir / "factor_list_debug"
    batch_size = args.ocr_batch_size if args.ocr_batch_size > 0 else 12
    rapidocr = args.ocr_engine == "rapidocr"
    result = recognize_factor_list_image(
        image,
        options=FactorOcrOptions(
            use_paddle=True,
            debug_dir=debug_dir,
            enable_overlay=True,
            enable_stitch=False,
            ocr_mode=args.ocr_engine,
            paddle_cache_dir=_resolve_path(args.paddle_cache_dir),
            paddle_lang=args.paddle_lang,
            paddle_mode="recognition",
            paddle_det_limit_side_len=args.paddle_det_limit_side_len,
            paddle_det_limit_type=args.paddle_det_limit_type,
            paddle_det_thresh=args.paddle_det_thresh,
            paddle_det_box_thresh=args.paddle_det_box_thresh,
            paddle_det_unclip_ratio=args.paddle_det_unclip_ratio,
            paddle_rec_score_thresh=args.paddle_rec_score_thresh,
            rapidocr_model_root_dir=_resolve_path(args.rapidocr_model_root_dir),
            rapidocr_text_score=args.rapidocr_text_score,
            preprocess_crop=not args.disable_ocr_preprocess,
            rapidocr_ocr_version=args.rapidocr_ocr_version,
            rapidocr_lang_type=args.rapidocr_lang_type,
            rapidocr_model_type=args.rapidocr_model_type,
            rapidocr_rec_img_shape=(3, 48, args.rapidocr_rec_img_width),
            rapidocr_rec_batch_num=batch_size,
            ocr_crop_target=args.ocr_crop_target,
            crop_variant=args.crop_variant,
            ocr_roi_profiles=("body_name",)
            if rapidocr
            else (
                "card_upper_band",
                "text_band_with_margin",
                "tight_text_roi",
            ),
            ocr_preprocess_modes=("raw_upscaled", "gray_sharpen", "color_text_safe")
            if rapidocr
            else (
                "raw_upscaled",
                "gray_sharpen",
                "color_text_safe",
            ),
            ocr_min_crop_width=args.ocr_min_crop_width,
            ocr_min_crop_height=args.ocr_min_crop_height,
            ocr_max_upscale=args.ocr_max_upscale,
            ocr_sharpen_strength=args.ocr_sharpen_strength,
            ocr_contrast_clip_limit=args.ocr_contrast_clip_limit,
            ocr_execution_mode=args.ocr_execution_mode,
            ocr_batch_size=batch_size,
            ocr_canvas_padding=args.ocr_canvas_padding,
        ),
    )
    submission = to_submission(
        result,
        submitter_id=args.submitter_id,
        image_path=output_dir / "stitched.png",
    )
    _write_json(output_dir / "factor_list_submission.json", submission.to_json_dict())
    _write_json(
        output_dir / "factor_list_ocr_metadata.json",
        {
            "pipeline": "factor-list",
            "factor_count": len(result.factors),
            "needs_review": sum(1 for factor in result.factors if factor.needs_review),
            "debug_dir": str(debug_dir),
            "submission_path": str(output_dir / "factor_list_submission.json"),
            "ocr_engine": args.ocr_engine,
            "card_detector": "card-body",
            "ocr_execution_mode": args.ocr_execution_mode,
            "ocr_batch_size": batch_size,
            "ocr_canvas_padding": args.ocr_canvas_padding,
            "ocr_preprocess": _ocr_preprocess_params(args),
            "paddle_cache_dir": str(_resolve_path(args.paddle_cache_dir)),
            "paddle_params": {
                "mode": "recognition",
                "text_det_limit_side_len": args.paddle_det_limit_side_len,
                "text_det_limit_type": args.paddle_det_limit_type,
                "text_det_thresh": args.paddle_det_thresh,
                "text_det_box_thresh": args.paddle_det_box_thresh,
                "text_det_unclip_ratio": args.paddle_det_unclip_ratio,
                "text_rec_score_thresh": args.paddle_rec_score_thresh,
            },
            "rapidocr_model_root_dir": str(_resolve_path(args.rapidocr_model_root_dir)),
            "rapidocr_text_score": args.rapidocr_text_score,
            "rapidocr_ocr_version": args.rapidocr_ocr_version,
            "rapidocr_lang_type": args.rapidocr_lang_type,
            "rapidocr_model_type": args.rapidocr_model_type,
            "rapidocr_rec_img_shape": [3, 48, args.rapidocr_rec_img_width],
        },
    )


def _ocr_preprocess_params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "enabled": not args.disable_ocr_preprocess,
        "min_crop_width": args.ocr_min_crop_width,
        "min_crop_height": args.ocr_min_crop_height,
        "max_upscale": args.ocr_max_upscale,
        "sharpen_strength": args.ocr_sharpen_strength,
        "contrast_clip_limit": args.ocr_contrast_clip_limit,
    }


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
