"""Load scroll stitching regression cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from ..capture.scraper_types import ScrollFrame


@dataclass(frozen=True)
class StitchCaseExpected:
    offsets: dict[int, int]
    size: tuple[int, int] | None = None
    stitched_path: Path | None = None


@dataclass(frozen=True)
class StitchCase:
    case_id: str
    case_dir: Path
    frames: tuple[ScrollFrame, ...]
    expected: StitchCaseExpected


def _resolve_case_path(case_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else case_dir / path


def _load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"stitch case image is missing or unreadable: {path}")
    return image


def _parse_expected(case_dir: Path, data: dict[str, Any]) -> StitchCaseExpected:
    expected = data.get("expected", {})
    raw_offsets = expected.get("offsets", {})
    offsets = {int(frame_index): int(offset_y) for frame_index, offset_y in raw_offsets.items()}
    raw_size = expected.get("size")
    size = None
    if raw_size is not None:
        size = (int(raw_size["width"]), int(raw_size["height"]))
    return StitchCaseExpected(
        offsets=offsets,
        size=size,
        stitched_path=_resolve_case_path(case_dir, expected.get("stitched")),
    )


def load_stitch_case(case_dir: Path) -> StitchCase:
    manifest_path = case_dir / "case.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []
    for default_index, frame_data in enumerate(data.get("frames", [])):
        image_path = _resolve_case_path(case_dir, frame_data["path"])
        assert image_path is not None
        frame_index = int(frame_data.get("frame_index", default_index))
        offset_y = frame_data.get("offset_y")
        frames.append(
            ScrollFrame(
                image=_load_image(image_path),
                frame_index=frame_index,
                source_path=str(image_path),
                offset_y=int(offset_y) if offset_y is not None else None,
            )
        )
    return StitchCase(
        case_id=str(data.get("case_id", case_dir.name)),
        case_dir=case_dir,
        frames=tuple(frames),
        expected=_parse_expected(case_dir, data),
    )


def load_stitch_cases(root: Path) -> tuple[StitchCase, ...]:
    return tuple(
        load_stitch_case(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "case.json").exists()
    )
