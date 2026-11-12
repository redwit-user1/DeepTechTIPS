"""합성 — preprint → 양성/음성 응답 → DPO 쌍 + Gateway 평가 아이템.

각 정량 주장 문장 C(=진짜 근거)에 대해:
  y_w(chosen)  = C + 정확한 인용                        (출처 ✓ 정확 ✓)
  y_l(rejected)= no_source / numeric_tamper / fake_doi / polarity_flip 중 하나
grounding(VCR 대조 근거) = C.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from compliance_gateway.data.extract import claim_sentences
from compliance_gateway.data.models import DPOPair, GatewayEvalItem, Preprint
from compliance_gateway.data.tamper import (
    fake_doi,
    tamper_biblio,
    tamper_number,
    tamper_polarity,
    tamper_year,
)

DEFAULT_SEED = Path("compliance_gateway/data/seed/biorxiv_pharma.json")


def load_seed(path: Path | None = None) -> list[Preprint]:
    path = path or DEFAULT_SEED
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Preprint(**r) for r in rows]


def _question(pp: Preprint, claim: str) -> str:
    return (
        f"In the study on {pp.title.lower()}, what was the reported finding? "
        f"Answer with the source citation."
    )


def _cited(claim: str, pp: Preprint, doi: str | None = None) -> str:
    return f"{claim} ({pp.citation()}; DOI: {doi or pp.doi})"


def _negatives(claim: str, pp: Preprint) -> list[tuple[str, str, str | None]]:
    """(kind, response, halluc_type) 음성 후보 목록(생성 가능한 것만)."""
    out: list[tuple[str, str, str | None]] = []

    # 무출처(대조군)
    out.append(("no_source", claim, None))

    # 수치 변조 (유형 C) — 인용은 그대로 유지(겉보기엔 출처 있음)
    tnum = tamper_number(claim)
    if tnum:
        out.append(("numeric_tamper", _cited(tnum, pp), "C"))

    # 가짜 DOI (유형 A) — 실존 저자 + 형식만 유효한 DOI
    out.append(("fake_doi", _cited(claim, pp, doi=fake_doi(pp.doi)), "A"))

    # 극성 역전 — 결론 방향 조작 + 인용 유지
    tpol = tamper_polarity(claim)
    if tpol and tpol != claim:
        out.append(("polarity_flip", _cited(tpol, pp), "C"))

    # 서지 변조 — 실존 DOI 유지 + 저자만 다른 실존 성으로 교체.
    # DOI 존재 여부만 보는 검증기는 통과시킴 → 3-class 서지 검증이 필요한 케이스.
    # ALCOA+ 'Attributable' 위반.
    cited = _cited(claim, pp)
    tbib = tamper_biblio(cited, pp.first_author)
    if tbib and tbib != cited:
        out.append(("biblio_tamper", tbib, "A"))

    # 연도 드리프트 — 실존 DOI + 연도만 변조
    out.append(("year_drift", tamper_year(cited, pp.year), "A"))

    return out


def build_examples(
    preprints: Iterable[Preprint],
    max_claims: int = 4,
) -> tuple[list[DPOPair], list[GatewayEvalItem]]:
    """DPO 쌍 + Gateway 평가 아이템 생성(VCR 점수는 아직 비움)."""
    pairs: list[DPOPair] = []
    evals: list[GatewayEvalItem] = []

    for pp in preprints:
        for claim in claim_sentences(pp.abstract, max_claims=max_claims):
            prompt = _question(pp, claim)
            chosen = _cited(claim, pp)

            # 양성 Gateway 평가 아이템
            evals.append(
                GatewayEvalItem(
                    query=prompt, response=chosen, grounding=claim,
                    source_doi=pp.doi, label="compliant",
                )
            )

            for kind, rejected, htype in _negatives(claim, pp):
                pairs.append(
                    DPOPair(
                        prompt=prompt, chosen=chosen, rejected=rejected,
                        rejected_kind=kind, grounding=claim, source_doi=pp.doi,
                    )
                )
                evals.append(
                    GatewayEvalItem(
                        query=prompt, response=rejected, grounding=claim,
                        source_doi=pp.doi, label=kind, halluc_type=htype,
                    )
                )

    return pairs, evals
