"""SourceMatch — 인용-내용 일치도.

인용한 출처가 실제로 그 주장을 뒷받침하는지 평가한다.
1차 구현: 휴리스틱(주장-근거 컨텍스트 토큰 중첩률).
목표: NLI 모델(entailment/neutral/contradiction) 문장 수준 평가.

NLI 어댑터는 `nli_fn` 인자로 주입한다(미주입 시 휴리스틱 사용).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from compliance_gateway.models import Citation

# (premise=근거, hypothesis=주장) -> entailment 확률 [0,1]
NLIFn = Callable[[str, str], float]

_TOKEN = re.compile(r"[A-Za-z가-힣0-9.]+")
_STOP = {"the", "a", "an", "of", "to", "in", "and", "은", "는", "이", "가", "에", "의", "을", "를"}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text) if t.lower() not in _STOP and len(t) > 1}


def _overlap(claim: str, evidence: str) -> float:
    a, b = _tokens(claim), _tokens(evidence)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def _claim_for_citation(text: str, citation_raw: str) -> str:
    """인용이 귀속하는 **본문 주장**을 추출한다.

    인용이 문장 끝/독립 조각으로 떨어지면 그 조각만으로는 내용이 없다
    (예: "(출처: 기관, 2020, 과제번호 NCT...)"). 이 경우 인용 이전의 본문을
    주장으로 사용해야 NLI 가 근거-주장 일치도를 제대로 평가한다.
    """
    parts = re.split(r"(?<=[.!?])\s+|(?<=[다요])\s+|\n+", text.strip())
    for idx, part in enumerate(parts):
        if citation_raw in part:
            stripped = part.replace(citation_raw, "").strip(" .,;()（）")
            if len(stripped) >= 10:
                return stripped                      # 문장 안에 본문이 함께 있음
            # 인용만 있는 조각 → 직전 본문(없으면 전체)을 주장으로
            prev = " ".join(parts[:idx]).strip()
            return prev if len(prev) >= 10 else text
    return text


def source_match(
    text: str,
    citations: list[Citation],
    grounding: tuple[str, ...] = (),
    nli_fn: Optional[NLIFn] = None,
) -> float:
    """인용들의 평균 일치도 [0, 1]. 인용이 없으면 0.0.

    grounding(RAG로 검색된 근거 텍스트)과 응답 주장을 대조한다.
    nli_fn 주입 시 NLI 기반, 미주입 시 토큰 중첩 휴리스틱.
    """
    if not citations:
        return 0.0
    if not grounding:
        # 대조할 근거가 없으면 검증 불가 → 보수적으로 0.5(중립).
        for c in citations:
            c.match_score = 0.5
            c.verified = None
        return 0.5

    evidence = " ".join(grounding)
    scores: list[float] = []
    for c in citations:
        # 인용이 포함된 문장(=뒷받침되어야 할 주장)을 근거와 대조한다.
        claim = _claim_for_citation(text, c.raw)
        if nli_fn is not None:
            s = max(nli_fn(g, claim) for g in grounding)
        else:
            s = _overlap(claim, evidence)
        c.match_score = s
        c.verified = s >= 0.5
        scores.append(s)

    return sum(scores) / len(scores)
