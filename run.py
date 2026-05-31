"""CLI エントリポイント。

使い方:
    python run.py <image_path> --submitter <ID>
    python run.py <image_path> --submitter <ID> --dry-run
    python run.py <image_path> --submitter <ID> --debug-crops ./crops
    python run.py <image_path> --submitter <ID> --tab factors_raw_test
    python run.py <image_path> --submitter <ID> --review    # 自信度の低い因子を人間レビュー
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from umafactor.factor_list import FactorOcrOptions, recognize_factor_list_image, to_submission  # noqa: E402
from umafactor.pipeline import analyze_image, apply_review_results  # noqa: E402
from umafactor.review import ReviewQueue  # noqa: E402
from umafactor.schema import SHEET_TAB_NAME  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ウマ娘 継承因子画像 → スプレ書き込み")
    parser.add_argument("image_path", help="解析対象の画像ファイル（PNG/JPEG）")
    parser.add_argument("--submitter", required=True, help="投稿者 ID")
    parser.add_argument("--dry-run", action="store_true", help="スプレに書かず JSON を標準出力")
    parser.add_argument("--debug-crops", default=None, help="切り出し画像を保存するディレクトリ")
    parser.add_argument("--tab", default=SHEET_TAB_NAME, help=f"書き込み先タブ名（既定: {SHEET_TAB_NAME}）")
    parser.add_argument(
        "--ocr-mode",
        choices=["legacy", "factor-list"],
        default="legacy",
        help="OCR pipeline. Default keeps the existing legacy analyze_image flow.",
    )
    parser.add_argument(
        "--factor-ocr-engine",
        choices=["rapidocr", "paddle"],
        default="rapidocr",
        help="OCR engine used only with --ocr-mode factor-list.",
    )
    parser.add_argument(
        "--factor-ocr-execution-mode",
        choices=["sequential", "batch", "role_sheet"],
        default="batch",
        help="OCR execution mode used only with --ocr-mode factor-list.",
    )
    parser.add_argument(
        "--factor-ocr-batch-size",
        type=int,
        default=12,
        help=(
            "OCR batch/sheet split size used only with --ocr-mode factor-list. "
            "Use 0 for one sheet per role."
        ),
    )
    parser.add_argument(
        "--factor-ocr-sheet-max-side",
        type=int,
        default=3600,
        help="Maximum role-sheet side length before PaddleOCR, used by role_sheet mode.",
    )
    parser.add_argument(
        "--factor-ocr-sheet-columns",
        type=int,
        default=None,
        help="Fixed role-sheet column count. Default auto-packs under --factor-ocr-sheet-max-side.",
    )
    parser.add_argument(
        "--paddle-mode",
        choices=["recognition", "ocr"],
        default=None,
        help="PaddleOCR adapter mode for factor-list. role_sheet defaults to ocr.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="自信度の低い因子（赤<0.95 / 白<0.7 / 青<0.95）をポップアップでレビュー",
    )
    parser.add_argument("--review-all", action="store_true", help="全因子をレビューする（デバッグ用）")
    args = parser.parse_args()

    if args.ocr_mode == "factor-list" and not args.dry_run:
        print(
            "factor-list OCR review flow is not connected to the legacy review UI yet; "
            "use --dry-run and inspect debug output before sending to Sheets.",
            file=sys.stderr,
        )
        return 2

    if args.ocr_mode == "factor-list":
        paddle_mode = args.paddle_mode
        if paddle_mode is None:
            paddle_mode = "ocr" if args.factor_ocr_execution_mode == "role_sheet" else "recognition"
        factor_result = recognize_factor_list_image(
            args.image_path,
            options=FactorOcrOptions(
                debug_dir=Path(args.debug_crops) if args.debug_crops else None,
                enable_overlay=bool(args.debug_crops),
                ocr_mode=args.factor_ocr_engine,
                paddle_mode=paddle_mode,
                ocr_execution_mode=args.factor_ocr_execution_mode,
                ocr_batch_size=args.factor_ocr_batch_size,
                ocr_sheet_max_side=args.factor_ocr_sheet_max_side,
                ocr_sheet_columns=args.factor_ocr_sheet_columns,
            ),
        )
        submission = to_submission(
            factor_result,
            submitter_id=args.submitter,
            image_path=args.image_path,
        )
        review_queue = ReviewQueue()
    else:
        submission, review_queue = analyze_image(
            image_path=args.image_path,
            submitter_id=args.submitter,
            debug_crops_dir=args.debug_crops,
        )

    if args.review or args.review_all:
        from umafactor.review_ui import review_queue_interactive

        queue = review_queue if args.review_all else review_queue.filter_uncertain()
        print(f"レビュー対象: {len(queue.items)} 件（自信度が低いもの）")
        if queue.items:
            review_queue_interactive(queue)
            apply_review_results(submission, queue)

    if args.dry_run:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(submission.to_json_dict(), ensure_ascii=False, indent=2))
        return 0

    from umafactor.sheet_writer import append_submission

    result = append_submission(submission, tab_name=args.tab)
    print(f"書き込み完了: submission_id={submission.submission_id}, 応答={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
