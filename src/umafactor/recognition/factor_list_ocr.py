"""OCR helpers for stitched factor-list tiles."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Protocol, Sequence

import numpy as np

from ..detection.factor_list import FactorListTile


FactorListOcrCropTarget = Literal["name", "card"]


class FactorListOCRLike(Protocol):
    def recognize(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_blue(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_red(self, img_bgr: np.ndarray) -> str:
        ...

    def recognize_with_parts(self, img_bgr: np.ndarray) -> tuple[str, list[str]]:
        ...


def recognize_factor_list_tile_names(
    image: np.ndarray,
    tiles: Sequence[FactorListTile],
    ocr: FactorListOCRLike,
    *,
    crop_variant: str = "current",
    crop_target: FactorListOcrCropTarget = "name",
) -> list[FactorListTile]:
    """Return tiles with raw OCR names filled.

    The output intentionally keeps OCR text as raw_name.  Dictionary correction
    and known/unknown skill handling belong to later review/database stages.
    """

    recognized: list[FactorListTile] = []
    for tile in tiles:
        crop = crop_factor_list_ocr_region(
            image,
            tile,
            target=crop_target,
            variant=crop_variant,
        )
        if tile.color == "blue":
            raw_name = ocr.recognize_blue(crop)
        elif tile.color == "red":
            raw_name = ocr.recognize_red(crop)
        elif tile.color == "green":
            raw_name, _fragments = ocr.recognize_with_parts(crop)
        else:
            raw_name = ocr.recognize(crop)
        recognized.append(replace(tile, raw_name=raw_name, final_name=raw_name))
    return recognized


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
