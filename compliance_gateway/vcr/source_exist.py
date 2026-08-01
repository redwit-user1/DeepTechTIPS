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


# 같은 참고문헌으로 볼 최대 문자 간격(예: "(Kim et al. (2024); DOI: 10.x/y)")
_MERGE_WINDOW = 60


def extract_citations(text: str, merge: bool = True) -> list[Citation]:
    """텍스트에서 인용 후보를 추출한다.

    merge=True 면 서로 인접한 (저자-연도) + (DOI/URL) 조각을 **하나의 참고문헌**으로
    병합한다. 병합해야 서지 검증기가 'DOI 는 실존하나 저자/연도가 다름'(서지 변조)을
    판정할 수 있다 — 분리돼 있으면 대조할 메타데이터가 없어 무조건 통과된다.
    """
    found: list[tuple[int, int, Citation]] = []

    for m in _DOI.finditer(text):
        doi = m.group(0).rstrip(").,;]")
        found.append((m.start(), m.start() + len(doi), Citation(raw=doi, doi=doi)))
    for m in _URL.finditer(text):
        url = m.group(0).rstrip(").,;]")
        if not _DOI.search(url):
            found.append((m.start(), m.start() + len(url), Citation(raw=url, url=url)))
    for m in _AUTHOR_YEAR.finditer(text):
        found.append(
            (m.start(), m.end(), Citation(raw=m.group(0), authors=(m.group(1),), year=int(m.group(2))))
        )

    found.sort(key=lambda x: x[0])
    if not merge:
        return [c for _, _, c in found]

    merged: list[Citation] = []
    cur_start = cur_end = None
    cur: Citation | None = None
    for start, end, cit in found:
        if cur is not None and start - cur_end <= _MERGE_WINDOW:
            # 같은 참고문헌으로 간주 → 메타데이터 통합
            cur.doi = cur.doi or cit.doi
            cur.url = cur.url or cit.url
            cur.title = cur.title or cit.title
            cur.authors = cur.authors or cit.authors
            cur.year = cur.year or cit.year
            cur_end = max(cur_end, end)
            cur.raw = text[cur_start:cur_end]
            continue
        cur = Citation(raw=cit.raw, doi=cit.doi, url=cit.url, title=cit.title,
                       authors=cit.authors, year=cit.year)
        cur_start, cur_end = start, end
        merged.append(cur)

    return merged


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
