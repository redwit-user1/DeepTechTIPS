"""Halluc — 환각 점수(낮을수록 좋음).

사업계획서가 정의한 복합기만형 환각 3유형을 탐지한다.
1차 구현: 휴리스틱. 목표: NLI 기반 + 외부 DB(ScienceON/NTIS) 실재 대조.

- 유형 A: 가짜 DOI + 실존 저자 (DOI 형식은 유효하나 근거에 없음)
- 유형 B: 정교한 가짜 논문 제목 (근거에 없는 인용)
- 유형 C: 수치 변조 (근거와 다른 수치)
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from compliance_gateway.models import Citation
from compliance_gateway.vcr.source_match import _tokens

# DOI 실재 여부 확인 어댑터: doi -> 실존하면 True
DOIResolver = Callable[[str], bool]

_NUM_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(°c|℃|micro-?m|µm|nm|mm|mg|kg|mol|ph|%|fold|times|배)",
    re.IGNORECASE,
)


def _type_a_b(citations: list[Citation], grounding: tuple[str, ...], resolver: Optional[DOIResolver]) -> float:
    """검증 가능한 인용 중 허위(유형 A/B) 비율.

    - 유형 A: DOI 가 실존하지 않음(resolver False).
    - 유형 B: 인용에 논문 제목(title)이 있는데 근거에 그 핵심 토큰이 전혀 없음.
    저자-연도/URL 만 있는 인용은 그 문자열이 근거 본문에 안 나오는 게 정상이므로
    근거-중첩으로 환각 판정하지 않는다(정상 인용 오탐 방지).
    """
    if not citations:
        return 0.0
    ev_tokens = _tokens(" ".join(grounding))
    checkable = 0
    bad = 0
    for c in citations:
        if c.doi and resolver is not None:
            checkable += 1
            if not resolver(c.doi):
                bad += 1
        elif c.title and grounding:
            checkable += 1
            ttoks = _tokens(c.title)
            if ttoks and not (ttoks & ev_tokens):
                bad += 1
    return bad / checkable if checkable else 0.0


def _type_c(text: str, grounding: tuple[str, ...]) -> float:
    """수치 변조(유형 C) 점수. 근거에 없는 (값,단위) 쌍 비율."""
    if not grounding:
        return 0.0
    claim_pairs = set(_NUM_UNIT.findall(text))
    if not claim_pairs:
        return 0.0
    evidence = " ".join(grounding)
    ev_pairs = set(_NUM_UNIT.findall(evidence))
    # 같은 단위인데 값이 다른 경우 = 변조 의심
    ev_by_unit: dict[str, set[str]] = {}
    for v, u in ev_pairs:
        ev_by_unit.setdefault(u.lower(), set()).add(v)

    tampered = 0
    checked = 0
    for v, u in claim_pairs:
        u = u.lower()
        if u in ev_by_unit:
            checked += 1
            if v not in ev_by_unit[u]:
                tampered += 1
    if checked == 0:
        return 0.0
    return tampered / checked


def halluc(
    text: str,
    citations: list[Citation],
    grounding: tuple[str, ...] = (),
    doi_resolver: Optional[DOIResolver] = None,
) -> float:
    """환각 점수 [0, 1]. 0 = 환각 없음."""
    ab = _type_a_b(citations, grounding, doi_resolver)
    c = _type_c(text, grounding)
    # 두 신호의 최댓값(보수적): 한 유형이라도 강하게 탐지되면 높은 환각.
    return max(ab, c)
