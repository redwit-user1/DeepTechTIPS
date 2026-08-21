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
# ── 국내 R&D 인용 형식 ──────────────────────────────────────────────
# 과제번호 식별자: NCT########, 한국연구재단(2020R1A2C1010001),
# NTIS 과제고유번호(1711123456), 일반 영숫자 과제번호
_KR_TASK_ID = re.compile(
    r"(?:과제\s*번호|과제고유번호|연구과제번호|사업번호|접수번호)\s*[:：]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9\-_.]{4,30})"
)
# 출처 블록 전체: (출처: 서울대학교병원, 2020, 과제번호 NCT04490642)
_KR_SOURCE_BLOCK = re.compile(
    r"[(（]\s*(?:출처|참고|자료|근거)\s*[:：]\s*([^)）]{4,200})[)）]"
)
# 블록 내부에서 기관/연도 분리
_KR_YEAR = re.compile(r"(19|20)\d{2}")
# 한국어 저자 인용: 홍길동 외 (2024) / 김철수 등(2023)
_KR_AUTHOR_YEAR = re.compile(
    r"([가-힣]{2,4}(?:\s*(?:외|등)))\s*[,]?\s*[(（]?((?:19|20)\d{2})[)）]?"
)

# 큰따옴표로 인용된 논문 제목: (cf. "Title of the paper")
# LLM 생성물이 DOI 없이 제목만 인용하는 흔한 형태.
_QUOTED_TITLE = re.compile(r'"([^"]{20,300})"')


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
    for m in _QUOTED_TITLE.finditer(text):
        found.append((m.start(), m.end(), Citation(raw=m.group(0), title=m.group(1))))

    # 국내 R&D: 출처 블록을 하나의 참고문헌으로 인식(기관·연도·과제번호 통합)
    for m in _KR_SOURCE_BLOCK.finditer(text):
        inner = m.group(1)
        task = _KR_TASK_ID.search(inner)
        ym = _KR_YEAR.search(inner)
        # 블록 첫 필드를 수행기관으로 간주
        org = inner.split(",")[0].strip()
        found.append((m.start(), m.end(), Citation(
            raw=m.group(0),
            doi=task.group(1) if task else None,   # 과제번호를 식별자 자리에 매핑
            authors=(org,) if org else (),
            year=int(ym.group(0)) if ym else None,
        )))
    # 블록 밖에 단독으로 쓰인 과제번호
    for m in _KR_TASK_ID.finditer(text):
        if any(st <= m.start() < en for st, en, _ in found):
            continue
        found.append((m.start(), m.end(), Citation(raw=m.group(0), doi=m.group(1))))
    for m in _KR_AUTHOR_YEAR.finditer(text):
        found.append((m.start(), m.end(), Citation(
            raw=m.group(0), authors=(m.group(1),), year=int(m.group(2)))))

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


def _citation_spans(claims: list[str], citations: list[Citation]) -> list[bool]:
    """문장별 '출처가 붙어 있는가' 판정.

    문단 끝의 인용은 **앞선 문장들을 함께 귀속**한다("A. B. C (출처)." → 3문장 모두 커버).
    직전 인용 이후의 미커버 문장들을 현재 인용이 흡수하는 방식.
    """
    covered = [False] * len(claims)
    buffer: list[int] = []
    for i, claim in enumerate(claims):
        buffer.append(i)
        if any(c.raw and c.raw in claim for c in citations):
            for j in buffer:
                covered[j] = True
            buffer = []
    return covered


def source_exist(text: str, citations: list[Citation] | None = None) -> float:
    """출처가 달린 주장 비율 [0, 1].

    인용이 하나도 없으면 0.0.
    """
    if citations is None:
        citations = extract_citations(text)
    claims = _split_claims(text)
    if not claims or not citations:
        return 0.0
    covered = _citation_spans(claims, citations)
    return sum(covered) / len(covered)
