"""Factory for selectable factor-list OCR engines."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from ..ocr_protocol import FactorListOCRLike

if TYPE_CHECKING:
    from ...factor_list.types import FactorOcrOptions


def create_factor_list_ocr(options: FactorOcrOptions) -> FactorListOCRLike:
    """Return a cached OCR adapter for the selected engine."""

    if options.ocr_mode == "paddle":
        return _cached_paddle_factor_ocr(
            lang=options.paddle_lang,
            mode=options.paddle_mode,
            cache_dir=str(options.paddle_cache_dir.resolve())
            if options.paddle_cache_dir is not None
            else None,
            text_det_limit_side_len=options.paddle_det_limit_side_len,
            text_det_limit_type=options.paddle_det_limit_type,
            text_det_thresh=options.paddle_det_thresh,
            text_det_box_thresh=options.paddle_det_box_thresh,
            text_det_unclip_ratio=options.paddle_det_unclip_ratio,
            text_rec_score_thresh=options.paddle_rec_score_thresh,
        )
    if options.ocr_mode == "rapidocr":
        return _cached_rapidocr_factor_ocr(
            model_root_dir=str(options.rapidocr_model_root_dir.resolve())
            if options.rapidocr_model_root_dir is not None
            else None,
            text_score=options.rapidocr_text_score,
            ocr_version=options.rapidocr_ocr_version,
            lang_type=options.rapidocr_lang_type,
            model_type=options.rapidocr_model_type,
            rec_img_shape=options.rapidocr_rec_img_shape,
            rec_batch_num=options.rapidocr_rec_batch_num,
        )
    raise ValueError(f"unsupported OCR mode: {options.ocr_mode}")


@lru_cache(maxsize=4)
def _cached_paddle_factor_ocr(
    *,
    lang: str,
    mode: str,
    cache_dir: str | None,
    text_det_limit_side_len: int | None,
    text_det_limit_type: str | None,
    text_det_thresh: float | None,
    text_det_box_thresh: float | None,
    text_det_unclip_ratio: float | None,
    text_rec_score_thresh: float | None,
) -> FactorListOCRLike:
    from ..paddle_ocr_adapter import PaddleFactorOCR

    return PaddleFactorOCR(
        lang=lang,
        mode=mode,  # type: ignore[arg-type]
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        text_det_limit_side_len=text_det_limit_side_len,
        text_det_limit_type=text_det_limit_type,
        text_det_thresh=text_det_thresh,
        text_det_box_thresh=text_det_box_thresh,
        text_det_unclip_ratio=text_det_unclip_ratio,
        text_rec_score_thresh=text_rec_score_thresh,
    )


@lru_cache(maxsize=4)
def _cached_rapidocr_factor_ocr(
    *,
    model_root_dir: str | None,
    text_score: float | None,
    ocr_version: str,
    lang_type: str,
    model_type: str,
    rec_img_shape: tuple[int, int, int],
    rec_batch_num: int,
) -> FactorListOCRLike:
    from ..rapid_ocr_adapter import RapidFactorOCR

    return RapidFactorOCR(
        model_root_dir=Path(model_root_dir) if model_root_dir is not None else None,
        text_score=text_score,
        ocr_version=ocr_version,
        lang_type=lang_type,
        model_type=model_type,
        rec_img_shape=rec_img_shape,
        rec_batch_num=rec_batch_num,
    )
