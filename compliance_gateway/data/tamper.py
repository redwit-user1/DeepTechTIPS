"""변조 전략 — 사업계획서가 정의한 복합기만형 환각 3유형 생성.

- 유형 A: 가짜 DOI + 실존 저자  → fake_doi
- 유형 C: 수치 변조             → numeric_tamper
- (보강) 극성 역전              → polarity_flip (방향 결론 조작)
- (대조) 무출처                 → no_source
"""

from __future__ import annotations

import re
from typing import Optional

_NUMBER = re.compile(r"(?<![\w-])(\d+\.\d+|\d+)(?=\s*(?:%|micro-?M|µM|nM|mM|mg|kg|fold|times))", re.IGNORECASE)

_ANTONYMS = {
    "increase": "decrease", "increases": "decreases", "increased": "decreased",
    "decrease": "increase", "decreases": "increases", "decreased": "increased",
    "higher": "lower", "lower": "higher",
    "reduce": "elevate", "reduces": "elevates", "reduced": "elevated",
    "elevate": "reduce", "elevates": "reduces", "elevated": "reduced",
    "inhibits": "activates", "inhibit": "activate",
    "suppresses": "induces", "suppress": "induce",
    "improves": "worsens", "improve": "worsen", "improved": "worsened",
    "raise": "lower", "raises": "lowers", "raised": "lowered",
}


def tamper_number(sentence: str) -> Optional[str]:
    """첫 번째 정량 수치를 명확히 다른 값으로 변조(유형 C). 변조 불가 시 None."""
    m = _NUMBER.search(sentence)
    if not m:
        return None
    original = m.group(1)
    if "." in original:
        val = float(original)
        new = round(val * 0.5 + 1.0, 2)        # 명확히 다른 값
        new_str = f"{new:g}"
    else:
        val = int(original)
        new = val + 20 if val < 50 else max(1, val // 2)
        new_str = str(new)
    if new_str == original:
        new_str = original + "9"
    return sentence[: m.start(1)] + new_str + sentence[m.end(1):]


def tamper_polarity(sentence: str) -> Optional[str]:
    """방향 단서를 반의어로 치환(결론 역전). 단서 없으면 None."""
    def repl(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower not in _ANTONYMS:
            return word
        sub = _ANTONYMS[lower]
        return sub.capitalize() if word[0].isupper() else sub

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, _ANTONYMS)) + r")\b", re.IGNORECASE)
    new, n = pattern.subn(repl, sentence, count=1)
    return new if n else None


def fake_doi(real_doi: str) -> str:
    """형식은 유효하나 존재하지 않는 DOI(유형 A)."""
    # bioRxiv DOI 패턴을 흉내내되 식별번호를 바꾼다
    return "10.1101/2024.99.99.999999"


# 서지 변조용 실존 저자 풀(다른 논문의 실제 저자 → '실존하지만 틀린' 귀속)
_SWAP_AUTHORS = ("Zhang et al.", "Smith et al.", "Nakamura et al.", "Muller et al.")


def tamper_biblio(citation: str, real_surname: str) -> Optional[str]:
    """실존 DOI 는 그대로 두고 **저자만** 다른 실존 성으로 교체.

    가장 탐지하기 어려운 유형: DOI 존재 여부만 확인하는 검증기(binary resolver)는
    통과시키지만, 실제로는 잘못된 연구자에게 성과를 귀속시킨다.
    → ALCOA+ 'Attributable'(귀속가능) 직접 위반.
    """
    for cand in _SWAP_AUTHORS:
        if not cand.lower().startswith(real_surname.lower()[:3]):
            return citation.replace(f"{real_surname} et al.", cand)
    return None


def tamper_year(citation: str, real_year: int) -> str:
    """실존 DOI + 연도만 변조(서지 드리프트)."""
    return citation.replace(f"({real_year})", f"({real_year - 5})")
