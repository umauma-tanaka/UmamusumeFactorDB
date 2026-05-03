from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from umafactor.evaluation.stitch_dataset import load_stitch_case, load_stitch_cases


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((3, 2, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_load_stitch_case_reads_manifest_frames_and_expected_values() -> None:
    case_dir = Path("outputs") / "test_stitch_dataset" / "case_a"
    _write_image(case_dir / "frames" / "000.png", 10)
    _write_image(case_dir / "frames" / "001.png", 20)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "case_id": "case_a",
                "frames": [
                    {"path": "frames/000.png", "frame_index": 3, "offset_y": 0},
                    {"path": "frames/001.png", "frame_index": 4, "offset_y": 2},
                ],
                "expected": {
                    "offsets": {"3": 0, "4": 2},
                    "size": {"width": 2, "height": 5},
                    "stitched": "expected_stitched.png",
                },
            }
        ),
        encoding="utf-8",
    )

    case = load_stitch_case(case_dir)

    assert case.case_id == "case_a"
    assert [frame.frame_index for frame in case.frames] == [3, 4]
    assert [frame.offset_y for frame in case.frames] == [0, 2]
    assert case.frames[0].image.shape == (3, 2, 3)
    assert case.expected.offsets == {3: 0, 4: 2}
    assert case.expected.size == (2, 5)
    assert case.expected.stitched_path == case_dir / "expected_stitched.png"


def test_load_stitch_case_fails_for_unreadable_frame() -> None:
    case_dir = Path("outputs") / "test_stitch_dataset" / "missing_frame"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        json.dumps({"frames": [{"path": "frames/missing.png"}]}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        load_stitch_case(case_dir)


def test_load_stitch_cases_lists_case_directories_only() -> None:
    root = Path("outputs") / "test_stitch_dataset" / "case_list"
    case_dir = root / "case_b"
    ignored_dir = root / "ignored"
    _write_image(case_dir / "frames" / "000.png", 30)
    ignored_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "frames": [{"path": "frames/000.png"}],
                "expected": {"offsets": {"0": 0}},
            }
        ),
        encoding="utf-8",
    )

    cases = load_stitch_cases(root)

    assert [case.case_id for case in cases] == ["case_b"]
