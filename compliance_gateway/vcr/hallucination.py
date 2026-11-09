"""Halluc — 환각 점수(낮을수록 좋음).

사업계획서가 정의한 복합기만형 환각 3유형을 탐지한다.
1차 구현: 휴리스틱. 목표: NLI 기반 + 외부 DB(ScienceON/NTIS) 실재 대조.

- 유형 A: 가짜 DOI + 실존 저자 (DOI 형식은 유효하나 근거에 없음)
- 유형 B: 정교한 가짜 논문 제목 (근거에 없는 인용)
- 유형 C: 수치 변조 (근거와 다른 수치)
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from compliance_gateway.models import Citation
from compliance_gateway.vcr.source_match import _tokens

# DOI 실재 여부 확인 어댑터: doi -> 실존하면 True
DOIResolver = Callable[[str], bool]

_NUM_UNIT = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?)[ \t]*(°c|℃|micro-?m|µm|nm|mm|mg|kg|mol|ph|%|fold|times|명|건|개|회|년|세|시간|일|주|개월|점|배)",
    re.IGNORECASE,
)


def _type_a_b(citations: list[Citation], grounding: tuple[str, ...], resolver: Optional[DOIResolver]) -> float:
    """검증 가능한 인용 중 허위(유형 A/B) 비율.

    - 유형 A: DOI 가 실존하지 않음(resolver False).
    - 유형 B: 인용에 논문 제목(title)이 있는데 근거에 그 핵심 토큰이 전혀 없음.
    저자-연도/URL 만 있는 인용은 그 문자열이 근거 본문에 안 나오는 게 정상이므로
    근거-중첩으로 환각 판정하지 않는다(정상 인용 오탐 방지).
    """
    if not citations:
        return 0.0
    ev_tokens = _tokens(" ".join(grounding))
    checkable = 0
    bad = 0
    for c in citations:
        if c.doi and resolver is not None:
            checkable += 1
            if not resolver(c.doi):
                bad += 1
        elif c.title and grounding:
            checkable += 1
            ttoks = _tokens(c.title)
            if ttoks and not (ttoks & ev_tokens):
                bad += 1
    return bad / checkable if checkable else 0.0


def _type_c(text: str, grounding: tuple[str, ...]) -> float:
    """수치 변조(유형 C) 점수. 근거에 없는 (값,단위) 쌍 비율."""
    if not grounding:
        return 0.0
    claim_pairs = set(_NUM_UNIT.findall(text))
    if not claim_pairs:
        return 0.0
    evidence = " ".join(grounding)
    ev_pairs = set(_NUM_UNIT.findall(evidence))
    # 같은 단위인데 값이 다른 경우 = 변조 의심
    ev_by_unit: dict[str, set[str]] = {}
    for v, u in ev_pairs:
        ev_by_unit.setdefault(u.lower(), set()).add(v)

    tampered = 0
    checked = 0
    for v, u in claim_pairs:
        u = u.lower()
        if u in ev_by_unit:
            checked += 1
            if v not in ev_by_unit[u]:
                tampered += 1
    if checked == 0:
        return 0.0
    return tampered / checked


# 같은 차원(질량·농도·부피·시간·길이)의 단위 — 값은 같은데 단위만 바뀌면 크기가 10^3 배씩 달라진다.
_UNIT_FAMILY = {u.lower(): f for f, us in {
    "mass": ("ng", "ug", "µg", "μg", "㎍", "mg", "g", "kg"),
    "conc": ("nm", "um", "µm", "μm", "mm", "m"),
    "vol": ("ul", "µl", "μl", "ml", "㎖", "l"),
    "time": ("sec", "초", "min", "분", "hr", "시간", "일", "주", "개월"),
    "len": ("㎛", "cm"),
}.items() for u in us}


def unit_swaps(text: str, evidence: str) -> list[str]:
    """값은 근거와 같은데 단위만 다른 수치(µg→mg 등). 유형 C 의 다른 얼굴.

    `_type_c` 는 "같은 단위·다른 값"만 본다. 단위 바꿈은 값이 같아 어휘·수치 겹침으로는
    오히려 '일치'로 보이고, 인코더 NLI 도 놓친다(한국어 어려운 평가셋: 통과 점수 0.27~0.41).
    단위 환산(1,000 mg → 1 g)은 값이 바뀌므로 걸리지 않는다.

    ponytail: nm/µm 처럼 길이·농도가 겹치는 표기는 농도로 본다 — 오탐이 나면 문맥 판정으로 교체.
    """
    from compliance_gateway.data.ocr.claim_evidence import NUM_UNIT

    ev: dict[str, set[str]] = {}
    for v, u in NUM_UNIT.findall(evidence or ""):
        ev.setdefault(v, set()).add(u.lower())
    out = []
    for v, u in NUM_UNIT.findall(text or ""):
        u = u.lower()
        if v == "1":   # "1초당 → 1분당" 같은 비율 분모 — 단위 바꿈이 아니라 환산이다
            continue
        fam = _UNIT_FAMILY.get(u)
        seen = ev.get(v, set())
        if fam and u not in seen and any(_UNIT_FAMILY.get(x) == fam for x in seen):
            out.append(f"{v}{u}")
    return out


# 단위 → 기준 단위 배율(같은 차원 안에서만 비교). 근사 판정용.
_SCALE = {"ng": 1e-9, "ug": 1e-6, "µg": 1e-6, "μg": 1e-6, "㎍": 1e-6, "mg": 1e-3, "g": 1.0, "kg": 1e3,
          "nm": 1e-9, "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "mm": 1e-3, "m": 1.0,
          "ul": 1e-6, "µl": 1e-6, "μl": 1e-6, "ml": 1e-3, "㎖": 1e-3, "l": 1.0,
          "sec": 1, "초": 1, "min": 60, "분": 60, "hr": 3600, "시간": 3600, "일": 86400, "주": 604800}


def approx_numbers(text: str, evidence: str, tol: float = 0.1) -> list[str]:
    """근거에 **그대로는 없지만** 환산·반올림하면 근거 값과 10% 안쪽인 수치.

    정당한 근사("약 352 MPa", 1,000 mg → 1 g)일 수도, 미묘한 변조일 수도 있다 — 판정이 어렵다.
    작은 인코더는 이런 문장을 확신하며 거부했다(어려운 셋 근사 평균 0.65). 그래서 **LLM 으로 보낼
    문장**을 고르는 데 쓴다(판정 자체는 하지 않는다).
    """
    from compliance_gateway.data.ocr.claim_evidence import NUM_UNIT

    ev = []
    for v, u in NUM_UNIT.findall(evidence or ""):
        try:
            ev.append((float(v), u.lower()))
        except ValueError:
            pass
    exact = {(x, u) for x, u in ev}
    out = []
    for v, u in NUM_UNIT.findall(text or ""):
        u = u.lower()
        try:
            x = float(v)
        except ValueError:
            continue
        if (x, u) in exact or x == 0:
            continue
        for y, w in ev:
            fam_ok = u == w or (_UNIT_FAMILY.get(u) and _UNIT_FAMILY.get(u) == _UNIT_FAMILY.get(w))
            if not fam_ok or y == 0:
                continue
            a = x * _SCALE.get(u, 1.0)
            b = y * _SCALE.get(w, 1.0)
            if abs(a - b) / abs(b) <= tol:
                out.append(f"{v}{u}")
                break
    return out


def _type_a_b_verified(citations: list[Citation], verifier) -> Optional[float]:
    """CitationVerifier 기반 유형 A/B 점수(3-class 가중 감점).

    이분법(있다/없다) 대신 VALID(0.0) / PARTIALLY_VALID(0.5) / HALLUCINATED(1.0)
    로 감점해 '메타데이터 드리프트'(서지 변조)를 부분 반영한다.
    UNVERIFIED(조회 실패)는 분모에서 제외 — 네트워크 문제로 정상 인용을 벌하지 않는다.
    """
    from compliance_gateway.verify.models import CitationStatus

    if not citations:
        return 0.0
    penalties = []
    for c in citations:
        res = verifier.verify(c)
        if res.status is CitationStatus.UNVERIFIED:
            continue
        penalties.append(res.penalty)
    if not penalties:
        return None      # 전부 미검증 → 휴리스틱으로 폴백
    return sum(penalties) / len(penalties)


def halluc(
    text: str,
    citations: list[Citation],
    grounding: tuple[str, ...] = (),
    doi_resolver: Optional[DOIResolver] = None,
    verifier=None,
) -> float:
    """환각 점수 [0, 1]. 0 = 환각 없음.

    verifier(CitationVerifier) 주입 시 3-class 서지 검증을 우선 사용하고,
    미주입·전부 미검증이면 기존 휴리스틱(doi_resolver)으로 폴백한다.
    """
    ab: Optional[float] = None
    if verifier is not None:
        ab = _type_a_b_verified(citations, verifier)
    if ab is None:
        ab = _type_a_b(citations, grounding, doi_resolver)
    c = _type_c(text, grounding)
    if grounding and unit_swaps(text, " ".join(grounding)):
        c = 1.0                      # 값은 같고 단위만 바뀐 수치 — 유형 C
    # 두 신호의 최댓값(보수적): 한 유형이라도 강하게 탐지되면 높은 환각.
    return max(ab, c)
