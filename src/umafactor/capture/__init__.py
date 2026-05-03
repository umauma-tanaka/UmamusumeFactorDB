"""Image capture and stitching helpers."""

from .scraper_types import FrameOffset, ScrollFrame, StitchPlacement, StitchResult
from .scroll_estimator import MetadataOffsetEstimator
from .stitcher import ScrollAreaStitcher, stitch_single_image

__all__ = [
    "FrameOffset",
    "MetadataOffsetEstimator",
    "ScrollAreaStitcher",
    "ScrollFrame",
    "StitchPlacement",
    "StitchResult",
    "stitch_single_image",
]
