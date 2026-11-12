"""국내 연구과제 → 한국어 R&D 문장 렌더링 + ALCOA+ 속성별 변조.

한국어 문장은 **실제 국내 연구과제 메타데이터**를 R&D 문체로 렌더링한 것이다
(사실은 실데이터, 문장 형태는 템플릿 — data/korean/__init__.py 참조).

변조 유형은 사업계획서의 환각 3유형이 아니라 **ALCOA+ 속성 위반**에 직접 대응시킨다.
국내 R&D 감사(연구노트 점검·GLP)에서 실제로 지적되는 항목이기 때문이다.

| 변조 | 위반 ALCOA+ 속성 | 현실 시나리오 |
|---|---|---|
| sponsor_swap | Attributable (귀속가능) | 타 기관 성과를 자기 성과로 귀속 |
| enrollment_tamper | Accurate (정확) | 등록례수 부풀리기 |
| date_shift | Contemporaneous (동시적) | 수행시점 소급 기재 |
| id_fabrication | Original (원본) | 존재하지 않는 과제번호 |
| no_source | Attributable | 출처 없는 주장 |
"""

from __future__ import annotations

from typing import Optional

from compliance_gateway.data.korean.models import KRResearchRecord

# 다른 국내 기관으로 교체(실존 기관이므로 '존재는 하나 귀속이 틀린' 사례가 된다)
_SWAP_SPONSORS = (
    "고려대학교 안암병원", "울산대학교병원", "전남대학교병원", "부산대학교병원",
)

_SPONSOR_KO = {
    "Seoul National University Hospital": "서울대학교병원",
    "Seoul National University Bundang Hospital": "분당서울대학교병원",
    "Seoul National University Boramae Hospital": "서울시보라매병원",
    "Asan Medical Center": "서울아산병원",
    "Samsung Medical Center": "삼성서울병원",
    "Severance Hospital": "세브란스병원",
    "Gangnam Severance Hospital": "강남세브란스병원",
    "Yonsei University": "연세대학교",
    "Korea University": "고려대학교",
    "Chungnam National University": "충남대학교",
    "Chungnam National University Hospital": "충남대학교병원",
    "GNT Pharma": "지엔티파마",
    "Clinical Research Center for End Stage Renal Disease, Korea": "말기신부전임상연구센터",
}


def _has_final_consonant(word: str) -> bool:
    """마지막 글자에 받침이 있는지 판정(한글 유니코드 조합 규칙)."""
    for ch in reversed(word.strip()):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:          # 한글 음절
            return (code - 0xAC00) % 28 != 0
        if ch.isdigit():
            # 숫자는 읽는 음가의 받침 유무로 판단 (0,1,3,6,7,8 = 받침 있음)
            return ch in "013678"
        if ch.isalnum():
            return True                        # 라틴 문자 등은 받침 있음으로 처리
    return False


def josa(word: str, pair: str) -> str:
    """받침에 맞는 조사 선택. pair 예: '은는', '이가', '을를', '과와'."""
    with_final, without_final = pair[0], pair[1]
    return with_final if _has_final_consonant(word) else without_final


def with_josa(word: str, pair: str) -> str:
    return f"{word}{josa(word, pair)}"


def sponsor_ko(record: KRResearchRecord) -> str:
    """기관명 한국어 표기(미등록 기관은 원문 유지)."""
    return _SPONSOR_KO.get(record.sponsor, record.sponsor)


def _study_type_ko(record: KRResearchRecord) -> str:
    return {"INTERVENTIONAL": "중재연구", "OBSERVATIONAL": "관찰연구"}.get(record.study_type, "연구")


def render_claim(record: KRResearchRecord) -> str:
    """출처 없는 본문 주장(한국어). 검증 대상 사실만 포함."""
    cond = record.conditions[0] if record.conditions else "해당 질환"
    stype = _study_type_ko(record)
    sponsor = sponsor_ko(record)
    parts = [f"{with_josa(sponsor, '은는')} {with_josa(cond, '을를')} 대상으로 {stype}를 수행하였다"]
    if record.enrollment is not None:
        parts.append(f"본 연구에는 총 {record.enrollment:,}명이 등록되었다")
    if record.year:
        parts.append(f"연구 개시 시점은 {record.year}년이다")
    return ". ".join(parts) + "."


def render_citation(record: KRResearchRecord, *, sponsor: Optional[str] = None,
                    year: Optional[int] = None, nct_id: Optional[str] = None) -> str:
    """국내 R&D 문서 관례의 출처 표기."""
    s = sponsor if sponsor is not None else sponsor_ko(record)
    y = year if year is not None else record.year
    i = nct_id if nct_id is not None else record.nct_id
    return f"(출처: {s}, {y}, 과제번호 {i})"


def render_cited(record: KRResearchRecord) -> str:
    """규정 준수 응답 = 주장 + 정확한 출처."""
    return f"{render_claim(record)} {render_citation(record)}"


# ---- ALCOA+ 속성별 변조 --------------------------------------------------

def tamper_sponsor(record: KRResearchRecord) -> str:
    """Attributable 위반 — 수행기관을 다른 실존 국내 기관으로 교체."""
    mine = sponsor_ko(record)
    swap = next((s for s in _SWAP_SPONSORS if s != mine), _SWAP_SPONSORS[0])
    claim = render_claim(record).replace(mine, swap)
    return f"{claim} {render_citation(record, sponsor=swap)}"


def tamper_enrollment(record: KRResearchRecord) -> Optional[str]:
    """Accurate 위반 — 등록례수 부풀리기."""
    if record.enrollment is None:
        return None
    inflated = record.enrollment * 3 + 7
    claim = render_claim(record).replace(f"총 {record.enrollment:,}명", f"총 {inflated:,}명")
    return f"{claim} {render_citation(record)}"


def tamper_date(record: KRResearchRecord) -> Optional[str]:
    """Contemporaneous 위반 — 수행시점 소급 기재."""
    if not record.year:
        return None
    shifted = record.year - 4
    claim = render_claim(record).replace(f"{record.year}년이다", f"{shifted}년이다")
    return f"{claim} {render_citation(record, year=shifted)}"


def tamper_id(record: KRResearchRecord) -> str:
    """Original 위반 — 존재하지 않는 과제번호."""
    return f"{render_claim(record)} {render_citation(record, nct_id='NCT09999999')}"


def strip_source(record: KRResearchRecord) -> str:
    """출처 없는 주장 — 본 과제 정렬 기준의 핵심 비준수 사례."""
    return render_claim(record)
