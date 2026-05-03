"""Scroll-area stitching helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .scraper_types import FrameOffset, ScrollFrame, StitchPlacement, StitchResult
from .scroll_estimator import MetadataOffsetEstimator


def stitch_single_image(image: np.ndarray, source_path: str | None = None) -> StitchResult:
    frame = ScrollFrame(image=image, frame_index=0, source_path=source_path, offset_y=0)
    placement = StitchPlacement(
        frame_index=0,
        x0=0,
        y0=0,
        x1=frame.size.width,
        y1=frame.size.height,
        source_path=source_path,
    )
    offset = FrameOffset(frame_index=0, offset_y=0, source="single_frame")
    return StitchResult(
        image=image.copy(),
        placements=(placement,),
        offsets=(offset,),
        mode="single_frame",
        skipped=True,
    )


class ScrollAreaStitcher:
    def __init__(self, offset_estimator: MetadataOffsetEstimator | None = None) -> None:
        self.offset_estimator = offset_estimator or MetadataOffsetEstimator()

    def stitch(
        self,
        frames: Sequence[ScrollFrame],
        offsets: Sequence[FrameOffset] | None = None,
    ) -> StitchResult:
        if not frames:
            raise ValueError("frames is empty")
        if len(frames) == 1:
            frame = frames[0]
            return stitch_single_image(frame.image, frame.source_path)

        resolved_offsets = tuple(offsets) if offsets is not None else self.offset_estimator.estimate(frames)
        offset_by_frame = {offset.frame_index: offset for offset in resolved_offsets}
        missing = [frame.frame_index for frame in frames if frame.frame_index not in offset_by_frame]
        if missing:
            raise ValueError(f"missing offsets for frames: {missing}")

        absolute_offsets = [offset_by_frame[frame.frame_index].offset_y for frame in frames]
        min_y = min(absolute_offsets)
        max_y = max(offset + frame.size.height for frame, offset in zip(frames, absolute_offsets))
        canvas_height = max_y - min_y
        canvas_width = max(frame.size.width for frame in frames)
        channels = frames[0].image.shape[2] if frames[0].image.ndim == 3 else 1
        shape = (canvas_height, canvas_width, channels) if channels != 1 else (canvas_height, canvas_width)
        canvas = np.zeros(shape, dtype=frames[0].image.dtype)

        placements: list[StitchPlacement] = []
        for frame, absolute_y in zip(frames, absolute_offsets):
            y0 = absolute_y - min_y
            y1 = y0 + frame.size.height
            x1 = frame.size.width
            canvas[y0:y1, 0:x1] = frame.image
            placements.append(
                StitchPlacement(
                    frame_index=frame.frame_index,
                    x0=0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    source_path=frame.source_path,
                )
            )

        return StitchResult(
            image=canvas,
            placements=tuple(placements),
            offsets=resolved_offsets,
            mode="provided_offsets",
            skipped=False,
        )
