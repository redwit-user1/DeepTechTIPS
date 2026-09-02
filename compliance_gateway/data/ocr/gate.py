# -*- coding: utf-8 -*-
"""품질 게이트 — 전수 실사에서 측정한 필터를 한 곳에 모은다.

학습·평가에 쓸 행만 통과시킨다. 각 필터의 임계값은 추측이 아니라
전수 5,654,358행 측정에서 나온 값이다:

| 필터 | 근거 | 제거량 |
|---|---|---|
| 정확 중복 | 전수 해시 측정 | 23.0% |
| 길이 200자 미만 | 100자 미만이 7.8%, 200자면 청킹에 부족 | — |
| 문자 비중 30% 미만 | 숫자·기호만 남은 표 잔해 2.0% | — |
| 출판 논문 문서 | 문서 단위 판정 11.4% | 고유 가치 0 |
| 경계 위반 측정값 | pH>14 등 물리 상한 위반 8.5% | 행 제거가 아니라 **표시** |

**경계 위반은 행을 버리지 않고 표시만 한다.** 원인이 문자 오인식이 아니라
표 구조 손실(값-라벨 결합 오류)이므로 그 행의 다른 내용은 멀쩡하다.
수치 사전을 만들 때만 배제하면 된다.

중복 판정은 상태를 들고 있어야 하므로 `Gate` 인스턴스가 필요하다.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from compliance_gateway.data.ocr.csv_source import OcrRow

MIN_CHARS = 200
MIN_LETTER_RATIO = 0.30

_WS = re.compile(r"\s+")
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")

# 물리적 상한이 정해진 양 — 위반이면 값-라벨 결합이 깨진 것이다.
# `scripts/check_ocr_numbers.py` 와 같은 정의를 쓴다.
_N = r"(\d{1,5}(?:[.,]\d+)?)"
BOUNDED: dict[str, tuple[re.Pattern, float, float]] = {
    "pH": (re.compile(rf"pH\s*(?:값)?\s*[:=]?\s*{_N}", re.IGNORECASE), 0, 14),
    "순도": (re.compile(rf"(?:순도|purity)\s*[:=]?\s*{_N}\s*%", re.IGNORECASE), 0, 100),
    "함량": (re.compile(rf"(?:함량|함유량|content)\s*[:=]?\s*{_N}\s*%", re.IGNORECASE), 0, 100),
    "습도": (re.compile(rf"(?:습도|humidity|\bRH\b)\s*[:=]?\s*{_N}\s*%", re.IGNORECASE), 0, 100),
    "수율": (re.compile(rf"(?:수율|yield)\s*[:=]?\s*{_N}\s*%", re.IGNORECASE), 0, 100),
}
for _k, (_r, _lo, _hi) in BOUNDED.items():
    assert not _r.search(""), f"{_k} 빈 대안"


@dataclass
class GateStats:
    seen: int = 0
    passed: int = 0
    dup: int = 0
    too_short: int = 0
    low_letter: int = 0
    paper: int = 0
    bounded_flagged: int = 0

    def as_dict(self) -> dict[str, int | float]:
        d = {k: getattr(self, k) for k in
             ("seen", "passed", "dup", "too_short", "low_letter", "paper",
              "bounded_flagged")}
        d["pass_rate"] = round(self.passed / self.seen, 4) if self.seen else 0.0
        return d


def bounded_violations(text: str) -> list[str]:
    """물리적 상한을 벗어난 측정값의 종류. 비어 있으면 위반 없음."""
    out = []
    for label, (rx, lo, hi) in BOUNDED.items():
        for m in rx.finditer(text):
            try:
                v = float(m.group(1).replace(",", "."))
            except ValueError:
                continue
            if v < lo or v > hi:
                out.append(label)
                break
    return out


@dataclass
class Gate:
    """행 단위 품질 게이트. 중복 판정 때문에 상태를 유지한다.

    `paper_notes` 에 출판 논문으로 판정된 NoteId 해시를 넣어 두면 그 문서의
    모든 행을 배제한다. 비워 두면 논문 필터는 동작하지 않는다 —
    문서 단위 판정은 별도 패스(`scripts/group_by_note.py`)가 필요하기 때문이다.
    """

    min_chars: int = MIN_CHARS
    min_letter_ratio: float = MIN_LETTER_RATIO
    paper_notes: frozenset[int] = frozenset()
    _seen: set[int] = field(default_factory=set, repr=False)
    stats: GateStats = field(default_factory=GateStats)

    @staticmethod
    def norm(text: str) -> str:
        return _WS.sub(" ", text).strip()

    @staticmethod
    def note_key(note_id: str) -> int:
        """`group_by_note.py` 의 `nid()` 와 동일해야 한다."""
        return int.from_bytes(
            hashlib.blake2s(note_id.encode(), digest_size=7).digest(), "big")

    def check(self, row: OcrRow) -> Optional[str]:
        """통과하면 None, 막히면 사유 문자열."""
        self.stats.seen += 1
        if self.paper_notes and self.note_key(row.note_id) in self.paper_notes:
            self.stats.paper += 1
            return "paper"
        t = self.norm(row.text)
        if len(t) < self.min_chars:
            self.stats.too_short += 1
            return "too_short"
        letters = len(_HANGUL.findall(t)) + len(_LATIN.findall(t))
        if letters / len(t) < self.min_letter_ratio:
            self.stats.low_letter += 1
            return "low_letter"
        h = int.from_bytes(
            hashlib.blake2s(t.encode("utf-8", "ignore"), digest_size=8).digest(), "big")
        if h in self._seen:
            self.stats.dup += 1
            return "dup"
        self._seen.add(h)
        self.stats.passed += 1
        return None

    def filter(self, rows: Iterable[OcrRow]) -> Iterator[tuple[OcrRow, list[str]]]:
        """통과한 행과 그 행의 경계 위반 목록을 함께 내보낸다."""
        for row in rows:
            if self.check(row) is not None:
                continue
            viol = bounded_violations(row.text)
            if viol:
                self.stats.bounded_flagged += 1
            yield row, viol


def _demo() -> None:
    """게이트가 실제로 거르는지 최소 확인."""
    mk = lambda i, nid_, txt: OcrRow(id=str(i), note_id=nid_, file_name="x.pdf",
                                     page=0, text=txt)
    long_ko = "이 실험은 상온에서 수행했으며 측정값을 기록한다. " * 8
    g = Gate()
    assert g.check(mk(1, "n1", long_ko)) is None
    assert g.check(mk(2, "n1", long_ko)) == "dup", "중복을 못 잡았다"
    assert g.check(mk(3, "n2", "짧다")) == "too_short"
    assert g.check(mk(4, "n3", "1 2 3 4.5 6,7 " * 40)) == "low_letter"

    paper = frozenset({Gate.note_key("pn")})
    g2 = Gate(paper_notes=paper)
    assert g2.check(mk(5, "pn", long_ko)) == "paper"
    assert g2.check(mk(6, "ok", long_ko)) is None

    assert bounded_violations("pH 15.2 로 측정") == ["pH"]
    assert bounded_violations("pH 7.4, 수율 82%") == []
    assert bounded_violations("수율 340%") == ["수율"]
    assert g.stats.passed == 1 and g.stats.dup == 1
    print("gate demo ok", g.stats.as_dict())


if __name__ == "__main__":
    _demo()
