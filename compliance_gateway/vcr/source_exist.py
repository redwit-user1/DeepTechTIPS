"""SourceExist — 인용 존재 여부.

응답 내 식별 가능한 출처(DOI / URL / 'Author et al. (YYYY)')를 추출한다.
1차 구현: 정규식 기반 식별자 파서. 목표: 추출 + 외부 DB(ScienceON/NTIS) 실재 대조.
"""

from __future__ import annotations

import re

from compliance_gateway.models import Citation

# DOI: 10.<registrant>/<suffix>  (RFC 호환 단순화 패턴)
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
# "Kim et al. (2024)" / "Kim et al., 2024" / "Kim & Lee (2023)"
_AUTHOR_YEAR = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+(?:et al\.?|&\s+[A-Z][A-Za-z]+)))[,]?\s*\(?(\d{4})\)?"
)


def extract_citations(text: str) -> list[Citation]:
    """텍스트에서 인용 후보를 추출한다."""
    citations: list[Citation] = []

    for m in _DOI.finditer(text):
        citations.append(Citation(raw=m.group(0), doi=m.group(0)))
    for m in _URL.finditer(text):
        # URL 안에 DOI가 포함된 경우 중복 방지
        if not _DOI.search(m.group(0)):
            citations.append(Citation(raw=m.group(0), url=m.group(0)))
    for m in _AUTHOR_YEAR.finditer(text):
        authors = (m.group(1),)
        year = int(m.group(2))
        citations.append(Citation(raw=m.group(0), authors=authors, year=year))

    return citations


def _split_claims(text: str) -> list[str]:
    """주장 단위(문장)로 분할한다. 한국어 종결어미 + 영문 마침표 기준."""
    parts = re.split(r"(?<=[.!?])\s+|(?<=[다요])\s+|\n+", text.strip())
    return [p for p in (s.strip() for s in parts) if p]


def source_exist(text: str, citations: list[Citation] | None = None) -> float:
    """출처가 달린 주장 비율 [0, 1].

    인용이 단 하나도 없으면 0.0, 모든 주장에 인용이 있으면 1.0에 수렴.
    """
    if citations is None:
        citations = extract_citations(text)
    claims = _split_claims(text)
    if not claims:
        return 0.0
    if not citations:
        return 0.0

    cited = 0
    for claim in claims:
        if any(c.raw in claim for c in citations):
            cited += 1
    # 주장 수 대비 인용 커버리지. 인용이 주장 수보다 많아도 1.0 상한.
    return min(1.0, cited / len(claims))
