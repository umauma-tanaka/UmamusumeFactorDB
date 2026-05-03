"""Debug manifest primitives and JSON output helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class StageDebug:
    name: str
    status: str = "ok"
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
        }
        if self.message:
            data["message"] = self.message
        return data


@dataclass(frozen=True)
class DebugManifest:
    run_id: str
    case_id: str
    input_files: list[str] = field(default_factory=list)
    pipeline_version: str = ""
    model_versions: Mapping[str, str] = field(default_factory=dict)
    stages: list[StageDebug] = field(default_factory=list)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "input_files": self.input_files,
            "pipeline_version": self.pipeline_version,
            "model_versions": dict(self.model_versions),
            "stages": [stage.to_dict() for stage in self.stages],
            "metrics": dict(self.metrics),
        }


def write_debug_manifest(path: str | Path, manifest: DebugManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

