"""Scroll offset estimation interfaces and metadata-based implementation."""

from __future__ import annotations

from collections.abc import Sequence

from .scraper_types import FrameOffset, ScrollFrame


class MetadataOffsetEstimator:
    """Read absolute y offsets from ScrollFrame metadata.

    This is intentionally not an image-matching algorithm. It provides a stable
    baseline for fixture-driven stitcher tests before AKAZE/RANSAC estimation is
    implemented.
    """

    def estimate(self, frames: Sequence[ScrollFrame]) -> tuple[FrameOffset, ...]:
        if not frames:
            return ()

        offsets: list[FrameOffset] = []
        for index, frame in enumerate(frames):
            if frame.offset_y is None:
                if index == 0:
                    offset_y = 0
                else:
                    raise ValueError(
                        f"frame {frame.frame_index} is missing offset_y metadata"
                    )
            else:
                offset_y = frame.offset_y
            offsets.append(
                FrameOffset(
                    frame_index=frame.frame_index,
                    offset_y=int(offset_y),
                    confidence=1.0,
                    source="metadata",
                )
            )
        return tuple(offsets)
