"""Fetch the kouryaku.tools skill list and write a local skill master."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from umafactor.data_sources.kouryaku_tools_skills import (  # noqa: E402
    DEFAULT_SKILLS_URL,
    build_skill_master_document,
    build_skill_master_entries,
    extract_skills_from_html,
    fetch_skills_html,
    write_skill_master_csv,
    write_skill_master_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch kouryaku.tools skill data into local JSON/CSV masters.",
    )
    parser.add_argument("--url", default=DEFAULT_SKILLS_URL)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "models" / "modules" / "skill_master_kouryaku_tools.json",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "models" / "modules" / "skill_master_kouryaku_tools.csv",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args(argv)

    html = fetch_skills_html(args.url, timeout=args.timeout)
    raw_skills = extract_skills_from_html(html)
    entries = build_skill_master_entries(
        raw_skills,
        source_url=args.url,
        include_raw=args.include_raw,
    )
    document = build_skill_master_document(
        entries,
        source_url=args.url,
        include_raw=args.include_raw,
    )

    write_skill_master_json(document, args.json_out)
    print(f"wrote {args.json_out} ({len(entries)} skills)")

    if not args.no_csv:
        write_skill_master_csv(entries, args.csv_out)
        print(f"wrote {args.csv_out} ({len(entries)} skills)")

    for entry in entries[:5]:
        print(f"  {entry.id}: {entry.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
