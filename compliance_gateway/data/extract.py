"""Abstract → 정량 주장 문장 추출.

변조(특히 수치 변조·극성 역전) 대상이 되려면 검증 가능한 정량/방향 정보를
담은 문장이어야 한다. 그런 문장만 골라낸다.
"""

from __future__ import annotations

import re

# 검증 가능한 수치: '8.47', '56.1%', 'IC50 of 8.47', '66.1% of'
_NUMBER = re.compile(r"(?<![\w-])\d+(?:\.\d+)?\s*(?:%|micro-?M|µM|nM|mM|mg|kg|fold|times)?", re.IGNORECASE)
# 방향(극성) 단서 — 극성 역전 음성 생성에 사용
_DIRECTION = re.compile(
    r"\b(increase[sd]?|decrease[sd]?|reduc(?:e|es|ed)|inhibit[s]?|"
    r"higher|lower|raise[sd]?|elevat(?:e|es|ed)|suppress(?:es|ed)?|improv(?:e|es|ed))\b",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def has_quantitative(sentence: str) -> bool:
    """변조 가능한 검증 정보(수치 또는 방향 단서)를 포함하는가."""
    return bool(_find_numbers(sentence)) or bool(_DIRECTION.search(sentence))


def _find_numbers(sentence: str) -> list[re.Match]:
    # '%' / 단위가 붙거나 소수점이 있는 수치만(연도·작은 정수 노이즈 제외)
    out = []
    for m in re.finditer(r"(?<![\w-])(\d+\.\d+|\d+)\s*(%|micro-?M|µM|nM|mM|mg|kg|fold|times)", sentence, re.IGNORECASE):
        out.append(m)
    return out


def claim_sentences(abstract: str, max_claims: int = 6) -> list[str]:
    """정량/방향 정보가 있는 주장 문장 목록."""
    out = [s for s in split_sentences(abstract) if has_quantitative(s)]
    return out[:max_claims]
