"""가중 메타데이터 유사도 스코어링.

업스트림(Citation-Hallucination-Detection)의 가중치와 임계값을 채택:
  score = 0.60·title_sim + 0.30·author_sim + 0.10·year_sim
  VALID           : score >= 0.92
  PARTIALLY_VALID : score >= 0.70  (같은 논문이나 메타데이터 드리프트)
  HALLUCINATED    : 그 미만

표준 라이브러리(difflib)만 사용 — 외부 의존성 없음.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from compliance_gateway.verify.models import PaperRecord

# 업스트림 채택 가중치·임계값
W_TITLE = 0.60
W_AUTHOR = 0.30
W_YEAR = 0.10
THRESHOLD_VALID = 0.92
THRESHOLD_PARTIAL = 0.70

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def author_similarity(claimed: tuple[str, ...], record: PaperRecord) -> float:
    """첫 저자 성(姓) 일치를 우선 확인(인용은 보통 'Kim et al.' 형태)."""
    if not claimed:
        return 0.0
    rec_surname = record.first_author_surname()
    if not rec_surname:
        return 0.0
    claimed_surname = normalize(claimed[0]).replace(" et al", "").strip()
    # "kim et al." → "kim"
    claimed_surname = claimed_surname.split()[0] if claimed_surname else ""
    if not claimed_surname:
        return 0.0
    if claimed_surname == rec_surname:
        return 1.0
    return SequenceMatcher(None, claimed_surname, rec_surname).ratio()


def year_similarity(claimed: int | None, record_year: int | None) -> float:
    if claimed is None or record_year is None:
        return 0.0
    diff = abs(claimed - record_year)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5      # preprint→저널 게재로 1년 차이는 흔함
    return 0.0


def weighted_score(
    claimed_title: str,
    claimed_authors: tuple[str, ...],
    claimed_year: int | None,
    record: PaperRecord,
) -> tuple[float, tuple[str, ...]]:
    """가중 유사도와 불일치 필드 목록을 반환."""
    t = title_similarity(claimed_title, record.title) if claimed_title else 0.0
    a = author_similarity(claimed_authors, record)
    y = year_similarity(claimed_year, record.year)

    # 인용에 제목이 없는 경우(저자-연도 인용) 제목 가중치를 저자/연도로 재분배
    if not claimed_title:
        total_w = W_AUTHOR + W_YEAR
        score = (W_AUTHOR * a + W_YEAR * y) / total_w if total_w else 0.0
    else:
        score = W_TITLE * t + W_AUTHOR * a + W_YEAR * y

    mismatches = []
    if claimed_title and t < 0.85:
        mismatches.append("title")
    if claimed_authors and a < 0.85:
        mismatches.append("author")
    if claimed_year is not None and y < 1.0:
        mismatches.append("year")
    return score, tuple(mismatches)
