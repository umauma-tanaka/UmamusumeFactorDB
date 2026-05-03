# Stitch Case Fixtures

This directory is reserved for scroll-stitching regression cases.

Each case uses this layout:

```text
tests/fixtures/stitch_cases/<case_id>/
  case.json
  frames/
    000.png
    001.png
  expected_stitched.png
```

`case.json` schema:

```json
{
  "case_id": "scroll_001",
  "frames": [
    {"path": "frames/000.png", "frame_index": 0, "offset_y": 0},
    {"path": "frames/001.png", "frame_index": 1, "offset_y": 320}
  ],
  "expected": {
    "offsets": {"0": 0, "1": 320},
    "size": {"width": 540, "height": 1280},
    "stitched": "expected_stitched.png"
  }
}
```

`offset_y` is an absolute y position in the stitched image. For future cases where
offsets are not known in advance, omit `offset_y` from frames and keep expected
offsets under `expected.offsets` for evaluation.
