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


# 단위 표기 정규화 — 대소문자는 살린다(mm 길이 ≠ mM 농도, nm ≠ nM). 같은 단위의 다른 표기
# (ug·μg·㎍, 초·sec)는 하나로 모은다. 처음엔 소문자로 뭉개서 mm/mM 을 같은 단위로 보고,
# "5 sec"↔"5 초" 를 단위 바꿈으로 오판했다.
_ALIAS = {"ug": "µg", "μg": "µg", "㎍": "µg", "uM": "µM", "μM": "µM", "um": "µm", "μm": "µm", "㎛": "µm",
          "uL": "µL", "μL": "µL", "ml": "mL", "㎖": "mL", "sec": "s", "초": "s", "분": "min",
          "hr": "h", "시간": "h", "일": "d", "주": "wk", "개월": "mo"}


def _canon(u: str) -> str:
    return _ALIAS.get(u, u)


# 같은 차원의 단위 — 값은 같은데 단위만 바뀌면 크기가 10^3 배씩 달라진다.
_UNIT_FAMILY = {u: f for f, us in {
    "mass": ("ng", "µg", "mg", "g", "kg"),
    "conc": ("nM", "µM", "mM", "M"),
    "vol": ("µL", "mL", "L"),
    "len": ("nm", "µm", "mm", "cm"),
    "time": ("s", "min", "h", "d", "wk", "mo"),
    "energy": ("Wh", "kWh"),
    "power": ("W", "kW"),
}.items() for u in us}


def unit_swaps(text: str, evidence: str) -> list[str]:
    """값은 근거와 같은데 단위만 다른 수치(µg→mg 등). 유형 C 의 다른 얼굴.

    `_type_c` 는 "같은 단위·다른 값"만 본다. 단위 바꿈은 값이 같아 어휘·수치 겹침으로는
    오히려 '일치'로 보이고, 인코더 NLI 도 놓친다(한국어 어려운 평가셋: 통과 점수 0.27~0.41).
    단위 환산(1,000 mg → 1 g)은 값이 바뀌므로 걸리지 않는다.

    """
    from compliance_gateway.data.ocr.claim_evidence import NUM_UNIT

    ev: dict[str, set[str]] = {}
    for v, u in NUM_UNIT.findall(evidence or ""):
        ev.setdefault(v, set()).add(_canon(u))
    out = []
    for v, u in NUM_UNIT.findall(text or ""):
        u = _canon(u)
        if v == "1":   # "1초당 → 1분당" 같은 비율 분모 — 단위 바꿈이 아니라 환산이다
            continue
        fam = _UNIT_FAMILY.get(u)
        seen = ev.get(v, set())
        if fam and u not in seen and any(_UNIT_FAMILY.get(x) == fam for x in seen):
            out.append(f"{v}{u}")
    return out


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _num_key(x: str) -> str:
    x = x.replace(",", "")
    try:
        return repr(float(x))
    except ValueError:
        return x


def novel_numbers(text: str, evidence: str) -> list[str]:
    """문장에 있지만 근거에는 **그대로 없는** 숫자(쉼표·소수 표기 정규화).

    정당한 근사·환산(5 uM → 5,000 nM, 25% → 4분의 1, 25,000명 → 2.5만 명, 100초 → 1분 40초)과
    수치 변조가 모두 여기에 걸린다. 둘을 가르는 건 수치 추론이라 작은 인코더가 약하다 —
    **LLM 으로 보낼 문장**을 고르는 데 쓴다(판정은 하지 않는다).

    처음엔 '근거와 10% 안쪽인 수치'만 골랐으나(근사 86건 중 30건) 분수·배율·만 단위·복합 시간을
    못 잡았다. 숫자 하나라도 새로우면 보내는 쪽이 어려운 셋 정밀도 84.2 → 93.0% (LLM 호출 8 → 24%).
    """
    seen = {_num_key(m.group(0)) for m in _NUMBER.finditer(evidence or "")}
    return [m.group(0) for m in _NUMBER.finditer(text or "") if _num_key(m.group(0)) not in seen]


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
