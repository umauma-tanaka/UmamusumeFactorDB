"""Data structures for scroll capture and stitching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..core.geometry import Size


@dataclass(frozen=True)
class ScrollFrame:
    image: np.ndarray
    frame_index: int
    source_path: str | None = None
    offset_y: int | None = None

    @property
    def size(self) -> Size:
        return Size.from_image_shape(self.image.shape)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "source_path": self.source_path,
            "offset_y": self.offset_y,
            "width": self.size.width,
            "height": self.size.height,
        }


@dataclass(frozen=True)
class FrameOffset:
    frame_index: int
    offset_y: int
    confidence: float = 1.0
    source: str = "metadata"
    inlier_count: int | None = None
    reject_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "offset_y": self.offset_y,
            "confidence": self.confidence,
            "source": self.source,
            "inlier_count": self.inlier_count,
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True)
class StitchPlacement:
    frame_index: int
    x0: int
    y0: int
    x1: int
    y1: int
    source_path: str | None = None

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "source_path": self.source_path,
            "bbox": [self.x0, self.y0, self.x1, self.y1],
        }


StitchMode = Literal["single_frame", "provided_offsets"]


@dataclass(frozen=True)
class StitchResult:
    image: np.ndarray
    placements: tuple[StitchPlacement, ...]
    offsets: tuple[FrameOffset, ...]
    mode: StitchMode
    skipped: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "skipped": self.skipped,
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "placements": [placement.to_dict() for placement in self.placements],
            "offsets": [offset.to_dict() for offset in self.offsets],
        }
