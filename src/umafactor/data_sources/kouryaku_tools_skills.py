"""Fetch skill master data from kouryaku.tools.

The skills page is rendered by Next.js and only a small subset of cards is
visible initially. The full skill list is embedded in the React Server
Component flight stream, so we decode that stream instead of scraping the
visible card HTML.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import requests


DEFAULT_SKILLS_URL = "https://xn--gck1f423k.xn--1bvt37a.tools/skills"
DEFAULT_USER_AGENT = "UmamusumeFactorDB/1.0"


_NEXT_FLIGHT_PUSH_RE = re.compile(
    r"self\.__next_f\.push\(\[\d+,\"((?:\\.|[^\"\\])*)\"\]\)",
    re.DOTALL,
)


class SkillPageParseError(RuntimeError):
    """Raised when the Next.js skill payload cannot be found or parsed."""


@dataclass(frozen=True)
class SkillMasterEntry:
    id: int
    name: str
    kana: str | None
    description: str
    rarity: int | None
    group_id: int | None
    group_rate: int | None
    grade_value: int | None
    category: int | None
    tag_id: str | None
    icon_id: int | None
    need_skill_point: int | None
    detail_url: str
    icon_url: str | None
    raw: dict[str, Any] | None = None

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kana": self.kana,
            "description": self.description,
            "rarity": self.rarity,
            "group_id": self.group_id,
            "group_rate": self.group_rate,
            "grade_value": self.grade_value,
            "category": self.category,
            "tag_id": self.tag_id,
            "icon_id": self.icon_id,
            "need_skill_point": self.need_skill_point,
            "detail_url": self.detail_url,
            "icon_url": self.icon_url,
        }
        if include_raw:
            out["raw"] = self.raw
        return out


def fetch_skills_html(
    url: str = DEFAULT_SKILLS_URL,
    *,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    response.raise_for_status()
    return response.text


def extract_next_flight_text(html_text: str) -> str:
    """Decode and concatenate Next.js flight stream string chunks."""

    chunks: list[str] = []
    for match in _NEXT_FLIGHT_PUSH_RE.finditer(html_text):
        encoded = match.group(1)
        try:
            chunks.append(json.loads(f'"{encoded}"'))
        except json.JSONDecodeError as exc:
            raise SkillPageParseError(
                f"failed to decode Next.js flight chunk at offset {match.start()}"
            ) from exc

    if not chunks:
        raise SkillPageParseError("Next.js flight chunks were not found")
    return "".join(chunks)


def extract_skills_from_html(html_text: str) -> list[dict[str, Any]]:
    flight_text = extract_next_flight_text(html_text)
    start = flight_text.find('{"skills"')
    if start < 0:
        raise SkillPageParseError("skill payload was not found in Next.js flight stream")

    try:
        payload, _ = json.JSONDecoder().raw_decode(flight_text[start:])
    except json.JSONDecodeError as exc:
        raise SkillPageParseError("failed to parse skill payload JSON") from exc

    skills = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(skills, list):
        raise SkillPageParseError("skill payload does not contain a list")
    return [skill for skill in skills if isinstance(skill, dict)]


def build_skill_master_entries(
    raw_skills: Iterable[dict[str, Any]],
    *,
    source_url: str = DEFAULT_SKILLS_URL,
    include_raw: bool = False,
) -> list[SkillMasterEntry]:
    entries: list[SkillMasterEntry] = []
    base_url = source_url.rstrip("/")

    for raw in raw_skills:
        skill_id = _as_int(raw.get("id"))
        name = _as_str(raw.get("skillName"))
        if skill_id is None or not name:
            continue

        icon_id = _as_int(raw.get("iconId"))
        icon_url = None
        if icon_id is not None:
            icon_url = (
                "https://static.kouryaku.tools/umamusume/images/skills/"
                f"{icon_id}/icon.png?width=96&height=96"
            )

        entries.append(
            SkillMasterEntry(
                id=skill_id,
                name=name,
                kana=_as_optional_str(raw.get("skillNameKana")),
                description=_as_str(raw.get("skillDesc")),
                rarity=_as_int(raw.get("rarity")),
                group_id=_as_int(raw.get("groupId")),
                group_rate=_as_int(raw.get("groupRate")),
                grade_value=_as_int(raw.get("gradeValue")),
                category=_as_int(raw.get("skillCategory")),
                tag_id=_as_optional_str(raw.get("tagId")),
                icon_id=icon_id,
                need_skill_point=_as_int(raw.get("needSkillPoint")),
                detail_url=f"{base_url}/{skill_id}",
                icon_url=icon_url,
                raw=raw if include_raw else None,
            )
        )

    return entries


def build_skill_master_document(
    entries: list[SkillMasterEntry],
    *,
    source_url: str = DEFAULT_SKILLS_URL,
    fetched_at: datetime | None = None,
    include_raw: bool = False,
) -> dict[str, Any]:
    fetched_at = fetched_at or datetime.now(timezone.utc)
    return {
        "source": "kouryaku.tools",
        "source_url": source_url,
        "fetched_at": fetched_at.isoformat(),
        "count": len(entries),
        "skills": [entry.to_dict(include_raw=include_raw) for entry in entries],
    }


def write_skill_master_json(document: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_skill_master_csv(entries: list[SkillMasterEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "name",
        "kana",
        "description",
        "rarity",
        "group_id",
        "group_rate",
        "grade_value",
        "category",
        "tag_id",
        "icon_id",
        "need_skill_point",
        "detail_url",
        "icon_url",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_dict(include_raw=False))


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
