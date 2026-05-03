from __future__ import annotations

import numpy as np
import pytest

from umafactor.capture import (
    MetadataOffsetEstimator,
    ScrollAreaStitcher,
    ScrollFrame,
    stitch_single_image,
)


def _image(value: int, *, height: int = 3, width: int = 2) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_stitch_single_image_marks_stitch_as_skipped() -> None:
    image = _image(7)

    result = stitch_single_image(image, source_path="full.png")

    assert result.skipped is True
    assert result.mode == "single_frame"
    assert result.image is not image
    np.testing.assert_array_equal(result.image, image)
    assert result.placements[0].to_dict() == {
        "frame_index": 0,
        "source_path": "full.png",
        "bbox": [0, 0, 2, 3],
    }
    assert result.to_metadata()["offsets"][0]["source"] == "single_frame"


def test_metadata_offset_estimator_reads_frame_offsets() -> None:
    frames = [
        ScrollFrame(_image(1), frame_index=10),
        ScrollFrame(_image(2), frame_index=11, offset_y=4),
    ]

    offsets = MetadataOffsetEstimator().estimate(frames)

    assert [(offset.frame_index, offset.offset_y) for offset in offsets] == [
        (10, 0),
        (11, 4),
    ]
    assert all(offset.source == "metadata" for offset in offsets)


def test_metadata_offset_estimator_requires_offsets_after_first_frame() -> None:
    frames = [ScrollFrame(_image(1), 0), ScrollFrame(_image(2), 1)]

    with pytest.raises(ValueError, match="missing offset_y"):
        MetadataOffsetEstimator().estimate(frames)


def test_stitcher_places_frames_by_absolute_offsets() -> None:
    frames = [
        ScrollFrame(_image(1, height=3), frame_index=0, offset_y=0),
        ScrollFrame(_image(2, height=3), frame_index=1, offset_y=2),
        ScrollFrame(_image(3, height=2), frame_index=2, offset_y=5),
    ]

    result = ScrollAreaStitcher().stitch(frames)

    assert result.skipped is False
    assert result.mode == "provided_offsets"
    assert result.image.shape == (7, 2, 3)
    np.testing.assert_array_equal(result.image[0:2], _image(1, height=2))
    np.testing.assert_array_equal(result.image[2:5], _image(2, height=3))
    np.testing.assert_array_equal(result.image[5:7], _image(3, height=2))
    assert [placement.to_dict()["bbox"] for placement in result.placements] == [
        [0, 0, 2, 3],
        [0, 2, 2, 5],
        [0, 5, 2, 7],
    ]


def test_stitcher_requires_frames() -> None:
    with pytest.raises(ValueError, match="frames is empty"):
        ScrollAreaStitcher().stitch([])
