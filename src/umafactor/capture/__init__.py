"""Image capture and stitching helpers."""

from .scraper_types import FrameOffset, ScrollFrame, StitchPlacement, StitchResult
from .scroll_estimator import MetadataOffsetEstimator
from .stitcher import ScrollAreaStitcher, stitch_single_image
from .window_capture import (
    ScreenCaptureSession,
    WindowInfo,
    capture_window_frames,
    find_game_window,
    list_windows,
    rank_window_candidates,
)

__all__ = [
    "FrameOffset",
    "MetadataOffsetEstimator",
    "ScrollAreaStitcher",
    "ScrollFrame",
    "ScreenCaptureSession",
    "StitchPlacement",
    "StitchResult",
    "WindowInfo",
    "capture_window_frames",
    "find_game_window",
    "list_windows",
    "rank_window_candidates",
    "stitch_single_image",
]
