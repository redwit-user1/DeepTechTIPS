"""Lexical NLI 베이스라인 — 토큰 중첩.

기존 source_match 휴리스틱과 동일 계열. 극성을 보지 않으므로
'increase' vs 'decrease' 같은 반의 주장도 높게 점수내는 한계가 있다(벤치마크 대조군).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z가-힣0-9.]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "and", "is", "are", "be", "for", "with",
    "that", "this", "by", "on", "as", "at", "or", "from",
    "은", "는", "이", "가", "에", "의", "을", "를", "도", "으로",
}


def tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text) if t.lower() not in _STOP and len(t) > 1}


def lexical_nli(premise: str, hypothesis: str) -> float:
    """가설 토큰이 근거에 등장하는 비율 [0, 1]."""
    h = tokens(hypothesis)
    p = tokens(premise)
    if not h or not p:
        return 0.0
    return len(h & p) / len(h)
