"""OCR helpers for stitched factor-list tiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol, Sequence

import cv2
import numpy as np

from ..detection.factor_list import FactorListTile


FactorListOcrCropTarget = Literal["name", "card"]
FactorListOcrExecutionMode = Literal["sequential", "canvas", "batch"]

DEFAULT_OCR_MIN_WIDTH = 640
DEFAULT_OCR_MIN_HEIGHT = 160
DEFAULT_OCR_MAX_UPSCALE = 4.0
DEFAULT_OCR_SHARPEN_STRENGTH = 0.45
DEFAULT_OCR_CONTRAST_CLIP_LIMIT = 1.6


class FactorListOCRLike(Protocol):
    def recognize(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_blue(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_red(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_with_parts(self, img_bgr: np.ndarray) -> tuple[str, list[str]]:
        ...


@dataclass(frozen=True)
class OcrCanvasBatch:
    canvas: np.ndarray
    regions: list[tuple[int, int, int, int]]


@dataclass(frozen=True)
class _PreparedOcrCrop:
    tile: FactorListTile
    crop: np.ndarray


def recognize_factor_list_tile_names(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
    ocr: FactorListOCRLike,
    *,
    crop_variant: str = "current",
    crop_target: FactorListOcrCropTarget = "name",
    preprocess_crop: bool = True,
    min_crop_width: int = DEFAULT_OCR_MIN_WIDTH,
    min_crop_height: int = DEFAULT_OCR_MIN_HEIGHT,
    max_upscale: float = DEFAULT_OCR_MAX_UPSCALE,
    sharpen_strength: float = DEFAULT_OCR_SHARPEN_STRENGTH,
    contrast_clip_limit: float = DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
    ocr_execution_mode: FactorListOcrExecutionMode = "sequential",
    canvas_batch_size: int = 12,
    canvas_padding: int = 24,
) -> list[FactorListTile]:
    """Return tiles with raw OCR names filled.

    The output intentionally keeps OCR text as raw_name.  Dictionary correction
    and known/unknown skill handling belong to later review/database stages.
    When canvas_batch_size is 0 or negative, canvas/batch modes process all
    provided tiles at once.  The live flow calls this once per role, so that
    becomes one OCR image each for parent, ancestor1, and ancestor2.
    """

    prepared = _prepare_ocr_crops(
        image,
        tiles,
        crop_variant=crop_variant,
        crop_target=crop_target,
        preprocess_crop=preprocess_crop,
        min_crop_width=min_crop_width,
        min_crop_height=min_crop_height,
        max_upscale=max_upscale,
        sharpen_strength=sharpen_strength,
        contrast_clip_limit=contrast_clip_limit,
    )

    if ocr_execution_mode == "canvas":
        raw_names = _recognize_prepared_crops_by_canvas(
            prepared,
            ocr,
            canvas_batch_size=canvas_batch_size,
            canvas_padding=canvas_padding,
        )
    elif ocr_execution_mode == "batch":
        raw_names = _recognize_prepared_crops_by_batch(prepared, ocr, batch_size=canvas_batch_size)
    elif ocr_execution_mode == "sequential":
        raw_names = [_recognize_one_tile(entry.tile, entry.crop, ocr) for entry in prepared]
    else:
        raise ValueError(f"unknown OCR execution mode: {ocr_execution_mode}")

    return [
        replace(entry.tile, raw_name=raw_name, final_name=raw_name)
        for entry, raw_name in zip(prepared, raw_names)
    ]


def pack_ocr_canvas(
    crops: Sequence[np.ndarray],
    *,
    padding: int = 24,
    background: int = 255,
) -> OcrCanvasBatch:
    """Pack OCR crops into one vertical canvas and retain crop-local regions."""

    if not crops:
        return OcrCanvasBatch(np.zeros((1, 1, 3), dtype=np.uint8), [])

    padding = max(0, int(padding))
    prepared = [_ensure_bgr_u8(crop) for crop in crops]
    max_width = max(crop.shape[1] for crop in prepared)
    total_height = padding
    for crop in prepared:
        total_height += crop.shape[0] + padding

    canvas = np.full(
        (max(1, total_height), max(1, max_width + padding * 2), 3),
        int(np.clip(background, 0, 255)),
        dtype=np.uint8,
    )
    regions: list[tuple[int, int, int, int]] = []
    y = padding
    for crop in prepared:
        h, w = crop.shape[:2]
        x = padding
        canvas[y : y + h, x : x + w] = crop
        regions.append((x, y, x + w, y + h))
        y += h + padding
    return OcrCanvasBatch(canvas, regions)


def _prepare_ocr_crops(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
    *,
    crop_variant: str,
    crop_target: FactorListOcrCropTarget,
    preprocess_crop: bool,
    min_crop_width: int,
    min_crop_height: int,
    max_upscale: float,
    sharpen_strength: float,
    contrast_clip_limit: float,
) -> list[_PreparedOcrCrop]:
    prepared: list[_PreparedOcrCrop] = []
    for tile in tiles:
        crop = crop_factor_list_ocr_region(
            image,
            tile,
            target=crop_target,
            variant=crop_variant,
        )
        if preprocess_crop:
            crop = prepare_factor_list_ocr_crop(
                crop,
                min_width=min_crop_width,
                min_height=min_crop_height,
                max_upscale=max_upscale,
                sharpen_strength=sharpen_strength,
                contrast_clip_limit=contrast_clip_limit,
            )
        prepared.append(_PreparedOcrCrop(tile=tile, crop=crop))
    return prepared


def _recognize_prepared_crops_by_canvas(
    prepared: Sequence[_PreparedOcrCrop],
    ocr: FactorListOCRLike,
    *,
    canvas_batch_size: int,
    canvas_padding: int,
) -> list[str]:
    recognize_canvas = getattr(ocr, "recognize_canvas", None)
    if not callable(recognize_canvas):
        return _recognize_prepared_crops_by_batch(
            prepared,
            ocr,
            batch_size=canvas_batch_size,
        )

    raw_names: list[str] = []
    for chunk in _chunks(list(prepared), _effective_batch_size(prepared, canvas_batch_size)):
        packed = pack_ocr_canvas([entry.crop for entry in chunk], padding=canvas_padding)
        names = list(recognize_canvas(packed.canvas, packed.regions))
        if len(names) != len(chunk):
            names = names[: len(chunk)] + [""] * max(0, len(chunk) - len(names))
        raw_names.extend(str(name) for name in names)
    return raw_names


def _recognize_prepared_crops_by_batch(
    prepared: Sequence[_PreparedOcrCrop],
    ocr: FactorListOCRLike,
    *,
    batch_size: int,
) -> list[str]:
    recognize_many = getattr(ocr, "recognize_many", None)
    if not callable(recognize_many):
        return [_recognize_one_tile(entry.tile, entry.crop, ocr) for entry in prepared]

    raw_names: list[str] = []
    for chunk in _chunks(list(prepared), _effective_batch_size(prepared, batch_size)):
        names = list(recognize_many([entry.crop for entry in chunk]))
        if len(names) != len(chunk):
            names = names[: len(chunk)] + [""] * max(0, len(chunk) - len(names))
        raw_names.extend(str(name) for name in names)
    return raw_names


def _recognize_one_tile(
    tile: FactorListTile,
    crop: np.ndarray,
    ocr: FactorListOCRLike,
) -> str:
    if tile.color == "blue":
        return ocr.recognize_blue(crop)
    if tile.color == "red":
        return ocr.recognize_red(crop)
    if tile.color == "green":
        raw_name, _fragments = ocr.recognize_with_parts(crop)
        return raw_name
    return ocr.recognize(crop)


def _chunks(values: Sequence[_PreparedOcrCrop], size: int) -> list[Sequence[_PreparedOcrCrop]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _effective_batch_size(values: Sequence[_PreparedOcrCrop], requested_size: int) -> int:
    if requested_size <= 0:
        return max(1, len(values))
    return max(1, requested_size)


def prepare_factor_list_ocr_crop(
    crop: np.ndarray,
    *,
    min_width: int = DEFAULT_OCR_MIN_WIDTH,
    min_height: int = DEFAULT_OCR_MIN_HEIGHT,
    max_upscale: float = DEFAULT_OCR_MAX_UPSCALE,
    sharpen_strength: float = DEFAULT_OCR_SHARPEN_STRENGTH,
    contrast_clip_limit: float = DEFAULT_OCR_CONTRAST_CLIP_LIMIT,
) -> np.ndarray:
    """Normalize a per-card OCR crop before passing it to the OCR engine.

    Live Steam captures can produce much smaller card crops than the static
    screenshots used for evaluation.  This keeps OCR input size stable without
    changing detection coordinates or the stitched output image.
    """

    if crop is None or crop.size == 0:
        return crop

    prepared = _ensure_bgr_u8(crop)
    prepared = _upscale_to_min_size(
        prepared,
        min_width=max(1, min_width),
        min_height=max(1, min_height),
        max_upscale=max(1.0, max_upscale),
    )
    prepared = _apply_luminance_clahe(prepared, clip_limit=contrast_clip_limit)
    prepared = _apply_unsharp_mask(prepared, strength=sharpen_strength)
    return prepared


def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def _upscale_to_min_size(
    image: np.ndarray,
    *,
    min_width: int,
    min_height: int,
    max_upscale: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image

    scale = max(1.0, min_width / width, min_height / height)
    scale = min(max_upscale, scale)
    if scale <= 1.01:
        return image

    new_width = max(width + 1, int(round(width * scale)))
    new_height = max(height + 1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def _apply_luminance_clahe(image: np.ndarray, *, clip_limit: float) -> np.ndarray:
    if clip_limit <= 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(4, 4))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _apply_unsharp_mask(image: np.ndarray, *, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)


def crop_factor_list_ocr_region(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    target: FactorListOcrCropTarget = "name",
    variant: str = "current",
) -> np.ndarray:
    x0, y0, x1, y1 = factor_list_ocr_region_bbox(
        image,
        tile,
        target=target,
        variant=variant,
    )
    return image[y0:y1, x0:x1]


def factor_list_ocr_region_bbox(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    target: FactorListOcrCropTarget = "name",
    variant: str = "current",
) -> tuple[int, int, int, int]:
    if target == "name":
        return factor_list_name_region_bbox(image, tile, variant=variant)
    if target == "card":
        return factor_list_card_region_bbox(image, tile)
    raise ValueError(f"unknown OCR crop target: {target}")


def crop_factor_list_card_region(
    image: np.ndarray,
    tile: FactorListTile,
) -> np.ndarray:
    x0, y0, x1, y1 = factor_list_card_region_bbox(image, tile)
    return image[y0:y1, x0:x1]


def factor_list_card_region_bbox(
    image: np.ndarray,
    tile: FactorListTile,
) -> tuple[int, int, int, int]:
    """Return the OCR crop bbox for one factor card.

    The detector's bbox is anchored from the star row and is intentionally
    conservative.  For OCR we trim most of the unrelated left-side context,
    extend far enough to the right to keep long names, and expand vertically to
    include the full text glyph height plus the star row.
    """

    x0, y0, x1, y1 = tile.bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    crop_left = int(round(width * 0.14))
    pad_right = int(round(width * (0.34 if tile.col_index == 0 else 0.20)))
    pad_top = int(round(height * 0.40))
    pad_bottom = int(round(height * 0.20))
    card_x0 = max(0, x0 + crop_left)
    card_x1 = min(image.shape[1], x1 + pad_right)
    card_y0 = max(0, y0 - pad_top)
    card_y1 = min(image.shape[0], y1 + pad_bottom)
    card_x1 = max(card_x0 + 1, card_x1)
    card_y1 = max(card_y0 + 1, card_y1)
    return card_x0, card_y0, card_x1, card_y1


def crop_factor_list_name_region(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    variant: str = "current",
) -> np.ndarray:
    x0, y0, x1, y1 = factor_list_name_region_bbox(image, tile, variant=variant)
    return image[y0:y1, x0:x1]


def factor_list_name_region_bbox(
    image: np.ndarray,
    tile: FactorListTile,
    *,
    variant: str = "current",
) -> tuple[int, int, int, int]:
    """Return the text-bearing upper area bbox from a factor tile."""

    x0, y0, x1, y1 = tile.bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    left_rel, right_rel, top_rel, bottom_rel = _crop_variant_bounds(variant)
    name_x0 = x0 + int(round(width * left_rel))
    name_x1 = x1 - int(round(width * right_rel))
    name_y0 = y0 + int(round(height * top_rel))
    name_y1 = y0 + int(round(height * bottom_rel))
    name_x0 = max(0, min(name_x0, image.shape[1] - 1))
    name_x1 = max(name_x0 + 1, min(name_x1, image.shape[1]))
    name_y0 = max(0, min(name_y0, image.shape[0] - 1))
    name_y1 = max(name_y0 + 1, min(name_y1, image.shape[0]))
    return name_x0, name_y0, name_x1, name_y1


def _crop_variant_bounds(variant: str) -> tuple[float, float, float, float]:
    if variant == "current":
        return 0.16, 0.04, 0.0, 0.68
    if variant == "wide":
        return 0.08, 0.02, 0.0, 0.78
    if variant == "upper":
        return 0.12, 0.03, 0.0, 0.58
    if variant == "full":
        return 0.08, 0.02, 0.0, 1.0
    raise ValueError(f"unknown crop variant: {variant}")
