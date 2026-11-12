"""Statistical NLI (v0.5) — TF-IDF 코사인 + 극성(부정·반의어) 처리.

순수 토큰 중첩의 핵심 약점: 근거와 주장이 주제는 같지만 *방향이 반대*인 경우
(예: 근거 "decreases" vs 주장 "raises")도 높게 점수낸다 → CONTRADICT를 못 거른다.

본 스코어러는 (1) 희소 단어에 가중치를 주는 TF-IDF 코사인으로 주제 유사도를 재고,
(2) 부정어/반의어 극성 충돌을 감지해 충돌 시 점수를 강하게 깎는다.
의존성 없이 표준 라이브러리만 사용. 트랜스포머 NLI로 가기 전의 강한 baseline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Optional

from compliance_gateway.nli.lexical import tokens

_NEGATION = {
    "not", "no", "without", "lack", "lacks", "lacking", "fail", "fails", "failed",
    "absence", "absent", "cannot", "neither", "nor", "none", "unable", "un",
    "않", "없", "못",
}

# 도메인(바이오/제약) 빈출 반의 방향쌍. 한쪽이 근거에, 반대쪽이 주장에 나오면 극성 충돌.
_ANTONYMS = [
    ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
    ("raise", "lower"), ("raises", "lowers"), ("raised", "lowered"),
    ("high", "low"), ("higher", "lower"), ("more", "less"), ("greater", "fewer"),
    ("positive", "negative"), ("activate", "inhibit"), ("activates", "inhibits"),
    ("induce", "suppress"), ("induces", "suppresses"), ("promote", "prevent"),
    ("promotes", "prevents"), ("enhance", "reduce"), ("enhances", "reduces"),
    ("gain", "loss"), ("up", "down"), ("upregulate", "downregulate"),
    ("improve", "worsen"), ("improves", "worsens"), ("stimulate", "block"),
]


def _antonym_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for a, b in _ANTONYMS:
        idx[a] = b
        idx[b] = a
    return idx


_ANT = _antonym_index()


class StatisticalNLI:
    """TF-IDF 코사인 + 극성 보정 entailment 스코어러.

    fit() 으로 IDF를 학습한다(미학습 시 IDF=1, 즉 순수 TF 코사인).
    호출 시 (premise, hypothesis) -> [0,1].
    """

    def __init__(self, idf: Optional[dict[str, float]] = None, contradiction_penalty: float = 0.35) -> None:
        self.idf = idf or {}
        self.default_idf = 1.0
        self.contradiction_penalty = contradiction_penalty

    def fit(self, corpus: Iterable[str]) -> "StatisticalNLI":
        docs = [tokens(t) for t in corpus]
        n = len(docs) or 1
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d)
        # smoothed idf
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
        self.default_idf = math.log((n + 1) / 1) + 1.0
        return self

    def _vec(self, text: str) -> dict[str, float]:
        tf = Counter(tokens(text))
        return {w: c * self.idf.get(w, self.default_idf) for w, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[w] * b[w] for w in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _polarity_conflict(self, p_tokens: set[str], h_tokens: set[str]) -> bool:
        # 1) 반의어 쌍 충돌: 근거에 한쪽, 주장에 반대쪽
        for w in h_tokens:
            mate = _ANT.get(w)
            if mate and mate in p_tokens:
                return True
        # 2) 부정 패리티 불일치: 한쪽만 부정문이면 방향이 어긋남
        neg_p = bool(p_tokens & _NEGATION) or self._has_neg_morpheme(p_tokens)
        neg_h = bool(h_tokens & _NEGATION) or self._has_neg_morpheme(h_tokens)
        return neg_p != neg_h

    @staticmethod
    def _has_neg_morpheme(toks: set[str]) -> bool:
        # 한국어 부정 형태소 부분일치
        return any(any(m in t for m in ("않", "없", "못")) for t in toks)

    def __call__(self, premise: str, hypothesis: str) -> float:
        base = self._cosine(self._vec(premise), self._vec(hypothesis))
        if base == 0.0:
            return 0.0
        p_t, h_t = tokens(premise), tokens(hypothesis)
        if self._polarity_conflict(p_t, h_t):
            # 주제는 비슷하나 방향이 반대 → entailment 아님(오히려 contradiction)
            return base * self.contradiction_penalty
        return base
