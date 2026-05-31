from __future__ import annotations

import json

from umafactor.data_sources.kouryaku_tools_skills import (
    build_skill_master_document,
    build_skill_master_entries,
    extract_skills_from_html,
)


def _flight_script(payload: str) -> str:
    encoded = json.dumps(payload, ensure_ascii=False)[1:-1]
    return f'<script>self.__next_f.push([1,"{encoded}"])</script>'


def test_extract_skills_from_next_flight_stream() -> None:
    skill_payload = (
        '14:["$","$L26",null,{"skills":[{'
        '"id":200011,'
        '"rarity":1,'
        '"groupId":20001,'
        '"groupRate":2,'
        '"gradeValue":258,'
        '"skillCategory":0,'
        '"tagId":"401/603",'
        '"iconId":10011,'
        '"needSkillPoint":110,'
        '"skillName":"右回り◎",'
        '"skillDesc":"右回りコースが得意になる",'
        '"skillNameKana":"ミギマワリ◎"'
        '}]}]'
    )
    html = _flight_script("prefix:") + _flight_script(skill_payload)

    skills = extract_skills_from_html(html)

    assert len(skills) == 1
    assert skills[0]["id"] == 200011
    assert skills[0]["skillName"] == "右回り◎"


def test_build_skill_master_document_maps_public_fields() -> None:
    raw_skills = [
        {
            "id": 200011,
            "rarity": 1,
            "groupId": 20001,
            "groupRate": 2,
            "gradeValue": 258,
            "skillCategory": 0,
            "tagId": "401/603",
            "iconId": 10011,
            "needSkillPoint": 110,
            "skillName": "右回り◎",
            "skillDesc": "右回りコースが得意になる",
            "skillNameKana": "ミギマワリ◎",
        }
    ]

    entries = build_skill_master_entries(raw_skills, source_url="https://example.test/skills")
    document = build_skill_master_document(
        entries,
        source_url="https://example.test/skills",
    )

    assert document["count"] == 1
    skill = document["skills"][0]
    assert skill["id"] == 200011
    assert skill["name"] == "右回り◎"
    assert skill["kana"] == "ミギマワリ◎"
    assert skill["detail_url"] == "https://example.test/skills/200011"
    assert skill["icon_url"] == (
        "https://static.kouryaku.tools/umamusume/images/skills/"
        "10011/icon.png?width=96&height=96"
    )
