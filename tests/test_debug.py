from __future__ import annotations

import json
from pathlib import Path

from umafactor.core.debug import DebugManifest, StageDebug, write_debug_manifest


def test_debug_manifest_serializes_stage_data() -> None:
    manifest = DebugManifest(
        run_id="run-1",
        case_id="phase0",
        input_files=["input.png"],
        pipeline_version="0.1.0",
        model_versions={"factor": "onnx"},
        stages=[
            StageDebug(
                name="recognition",
                status="ok",
                artifacts={"result": "recognition_results.json"},
                metrics={"total_images": 2},
                message="done",
            )
        ],
        metrics={"golden_matched": True},
    )

    assert manifest.to_dict() == {
        "run_id": "run-1",
        "case_id": "phase0",
        "input_files": ["input.png"],
        "pipeline_version": "0.1.0",
        "model_versions": {"factor": "onnx"},
        "stages": [
            {
                "name": "recognition",
                "status": "ok",
                "artifacts": {"result": "recognition_results.json"},
                "metrics": {"total_images": 2},
                "message": "done",
            }
        ],
        "metrics": {"golden_matched": True},
    }


def test_stage_debug_omits_empty_message() -> None:
    assert "message" not in StageDebug(name="env").to_dict()


def test_write_debug_manifest_creates_parent_directory() -> None:
    path = Path("outputs") / "test_debug" / "manifest.json"
    try:
        write_debug_manifest(
            path,
            DebugManifest(run_id="run-1", case_id="case", metrics={"ok": True}),
        )

        data = json.loads(path.read_text(encoding="utf-8"))
    finally:
        if path.exists():
            path.unlink()
        if path.parent.exists():
            try:
                path.parent.rmdir()
            except OSError:
                pass

    assert data["run_id"] == "run-1"
    assert data["metrics"] == {"ok": True}
