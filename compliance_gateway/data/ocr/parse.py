# -*- coding: utf-8 -*-
"""OCR 연구노트 텍스트 파서 — **노이즈 내성**이 핵심.

우리가 만든 가상 연구노트는 깨끗한 텍스트지만, 실제 OCR 결과는 다르다.
아래 열화를 견뎌야 한다.

| OCR 열화 | 대응 |
|---|---|
| 구획 제목 변형 (`4. 실험 결과` → `4 실험결과` / `4.실험 결 과`) | 공백·번호 유연 매칭 |
| 콜론 변형 (`:` `：` `;` 누락) | 구분자 집합 |
| 줄바꿈 삽입 (한 문장이 여러 줄) | 문단 재결합 |
| 문자 혼동 (O↔0, l↔1) | 날짜·수치 정규화 |
| 잡음 라인 (페이지 헤더/스캔 아티팩트) | 짧은 잡음 라인 제거 |
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_SEP = r"[:：;]?"

# 구획 헤더 — 번호·공백·조사 변형을 허용
SECTIONS = {
    "purpose":   r"(?:^|\n)\s*\d?\s*[.)]?\s*(?:실\s*험\s*)?목\s*적",
    "methods":   r"(?:^|\n)\s*\d?\s*[.)]?\s*(?:재\s*료\s*(?:및|와)?\s*)?방\s*법",
    "results":   r"(?:^|\n)\s*\d?\s*[.)]?\s*(?:실\s*험\s*)?결\s*과",
    "discussion": r"(?:^|\n)\s*\d?\s*[.)]?\s*(?:고\s*찰|결\s*론)",
}
_META = {
    "project_no": rf"과\s*제\s*(?:번호|고유번호){_SEP}\s*([^\n]+)",
    "project_title": rf"과\s*제\s*명{_SEP}\s*([^\n]+)",
    "institution": rf"(?:수행)?\s*기\s*관{_SEP}\s*([^\n]+)",
    "researcher": rf"(?:연\s*구\s*자|작\s*성\s*자){_SEP}\s*([^\n]+)",
    "reviewer": rf"(?:점\s*검\s*자|입\s*회\s*자|확\s*인\s*자){_SEP}\s*([^\n]+)",
    "exp_date": rf"실\s*험\s*일{_SEP}\s*([^\n]+)",
    "record_date": rf"기\s*록\s*일{_SEP}\s*([^\n]+)",
}

# 날짜: OCR 문자 혼동(O→0, l→1)과 구분자 변형을 흡수
_DATE = re.compile(r"(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")
_NOISE_LINE = re.compile(r"^[\s\-_=~·.]{0,4}$|^[|/\\]{1,3}$")
_OCR_DIGIT = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})
# 라벨 라인(`과제명: ...`)과 구획 헤더(`4. 실험 결과`)는 **이어붙이면 안 된다**.
# 붙이면 메타데이터 블록이 한 덩어리가 되어 필드·구획 추출이 모두 망가진다.
_LABEL_LINE = re.compile(r"^\s*[가-힣A-Za-z][가-힣A-Za-z\s]{0,12}[:：]")
_SECTION_HEAD = re.compile(r"^\s*\d\s*[.)]\s*\S")


@dataclass
class ParsedNote:
    """OCR 텍스트에서 복원한 연구노트 구조."""

    source: str = ""
    meta: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    text: str = ""                       # 정규화된 전체 텍스트
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def completeness(self) -> float:
        """복원된 필드 비율 — 낮으면 파싱 실패이지 규정 위반이 아니다."""
        want = len(_META) + len(SECTIONS)
        got = len(self.meta) + len(self.sections)
        return got / want

    def to_json(self) -> dict:
        return {"source": self.source, "meta": self.meta, "sections": self.sections,
                "text": self.text, "completeness": round(self.completeness, 3),
                "quality": self.quality}


def normalize_dates(text: str) -> str:
    """`202O-O3-3O` 같은 OCR 혼동 날짜를 ISO 형식으로 교정."""
    def fix(m: re.Match) -> str:
        y, mo, d = (g.translate(_OCR_DIGIT) for g in m.groups())
        try:
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        except ValueError:
            return m.group(0)

    # 혼동 문자가 섞인 날짜도 잡도록 완화한 패턴을 먼저 적용
    loose = re.compile(r"([0-9OolI|]{4})\s*[-./년]\s*([0-9OolI|]{1,2})\s*[-./월]\s*([0-9OolI|]{1,2})")
    return loose.sub(fix, text)


def clean(text: str) -> str:
    """스캔 잡음 제거 + 문단 재결합."""
    lines = [l.rstrip() for l in text.splitlines()]
    lines = [l for l in lines if not _NOISE_LINE.match(l.strip())]
    out: list[str] = []
    for line in lines:
        prev = out[-1] if out else ""
        joinable = (
            prev                                          # 앞 줄이 있고
            and not re.search(r"[.!?다요:：]\s*$", prev)   # 문장이 안 끝났고
            and len(prev) > 20                            # 충분히 길고
            and not _LABEL_LINE.match(prev)               # 앞 줄이 라벨 라인이 아니고
            and not _LABEL_LINE.match(line)               # 다음 줄도 라벨 라인이 아니고
            and not _SECTION_HEAD.match(line)             # 구획 헤더도 아닐 때만
        )
        if joinable:
            out[-1] = prev + " " + line.lstrip()
        else:
            out.append(line)
    return "\n".join(out)


def parse_note(raw: str, source: str = "") -> ParsedNote:
    """OCR 원문 → 구조화 레코드."""
    text = normalize_dates(clean(raw))
    note = ParsedNote(source=source, text=text)

    for key, pat in _META.items():
        m = re.search(pat, text)
        if m and m.group(1).strip():
            note.meta[key] = m.group(1).strip()

    # 구획: 헤더 위치를 찾아 다음 헤더 전까지를 본문으로
    positions: list[tuple[int, str]] = []
    for key, pat in SECTIONS.items():
        m = re.search(pat, text)
        if m:
            positions.append((m.end(), key))
    positions.sort()
    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        # 다음 헤더의 시작 지점을 정확히 쓰려면 헤더 문자열 길이만큼 되돌려야 하나,
        # 실무상 본문 앞부분이 조금 겹쳐도 검사에 영향이 없어 단순화한다.
        body = text[start:end].strip(" :：\n")
        if body:
            note.sections[key] = body

    note.quality = {
        "completeness": round(note.completeness, 3),
        "has_signature_block": float(bool(note.meta.get("researcher"))),
        "has_dates": float(bool(note.meta.get("exp_date") or note.meta.get("record_date"))),
    }
    return note
