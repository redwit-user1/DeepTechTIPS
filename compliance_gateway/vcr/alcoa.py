"""ALCOA_Score — 데이터 무결성(제약·바이오 핵심).

ALCOA+ 9속성 중 우선 4속성을 규칙기반으로 체크한다.
목표: 온톨로지(OWL2/SPARQL) 추론으로 전환.

- Attributable (귀속가능): 생성 주체·시점·모델ID 식별 가능
- Accurate     (정확): 수치·단위·조건 일관
- Complete     (완전): 필수 항목 누락 없음(미완결/말줄임 없음)
- Consistent   (일관): 동일 수치의 모순 없음
"""

from __future__ import annotations

import re

_NUM_UNIT = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)[ \t]*(°c|℃|k|mol|mm|nm|μm|ph|%|명|건|개|회|년|세|시간|일|주|개월|점|배)", re.IGNORECASE)
_INCOMPLETE = re.compile(r"(\.\.\.|…|등등|기타\s*등|TBD|to be|작성\s*예정)", re.IGNORECASE)


def _attributable(model_id: str, has_timestamp: bool) -> float:
    score = 0.0
    if model_id and model_id != "unknown":
        score += 0.5
    if has_timestamp:
        score += 0.5
    return score


def _accurate(text: str) -> float:
    """단위가 붙은 수치 비율로 정확성을 근사. 수치만 있고 단위 없으면 감점."""
    bare_numbers = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.%°])", text)
    unit_numbers = _NUM_UNIT.findall(text)
    total = len(bare_numbers) + len(unit_numbers)
    if total == 0:
        return 1.0  # 수치 없는 서술은 정확성 패널티 없음
    return len(unit_numbers) / total


def _complete(text: str) -> float:
    return 0.0 if _INCOMPLETE.search(text) else 1.0


def _consistent(text: str) -> float:
    """같은 단위의 동일 맥락 수치가 서로 다르게 반복되면 감점(단순 휴리스틱)."""
    by_unit: dict[str, set[str]] = {}
    for value, unit in _NUM_UNIT.findall(text):
        by_unit.setdefault(unit.lower(), set()).add(value)
    # 한 단위에 값이 3개 이상 충돌하면 일관성 의심
    conflicts = sum(1 for vals in by_unit.values() if len(vals) >= 3)
    if not by_unit:
        return 1.0
    return max(0.0, 1.0 - conflicts / len(by_unit))


def alcoa_score(text: str, model_id: str = "unknown", has_timestamp: bool = True) -> float:
    """우선 4속성의 평균 [0, 1]."""
    scores = [
        _attributable(model_id, has_timestamp),
        _accurate(text),
        _complete(text),
        _consistent(text),
    ]
    return sum(scores) / len(scores)
