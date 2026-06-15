"""VCR 집계 — compute_vcr().

VCR(y|x) = w1·SourceExist + w2·SourceMatch + w3·ALCOA_Score + w4·(1 − Halluc)
"""

from __future__ import annotations

from typing import Optional

from compliance_gateway.models import Citation, VCRBreakdown
from compliance_gateway.vcr.alcoa import alcoa_score
from compliance_gateway.vcr.hallucination import DOIResolver, halluc
from compliance_gateway.vcr.source_exist import extract_citations, source_exist
from compliance_gateway.vcr.source_match import NLIFn, source_match

# 도메인 중립 초기 가중치(합=1). VCR v2에서 도메인별 자동 최적화.
DEFAULT_WEIGHTS = {
    "source_exist": 0.25,
    "source_match": 0.30,
    "alcoa": 0.25,
    "halluc": 0.20,
}


def compute_vcr(
    query: str,
    response: str,
    grounding: tuple[str, ...] = (),
    model_id: str = "unknown",
    weights: Optional[dict[str, float]] = None,
    citations: Optional[list[Citation]] = None,
    nli_fn: Optional[NLIFn] = None,
    doi_resolver: Optional[DOIResolver] = None,
) -> VCRBreakdown:
    """응답의 VCR 보상 점수를 계산한다.

    weights 합이 1이 아니면 정규화한다.
    nli_fn / doi_resolver 미주입 시 휴리스틱으로 동작한다.
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    cits = citations if citations is not None else extract_citations(response)

    se = source_exist(response, cits)
    sm = source_match(response, cits, grounding, nli_fn=nli_fn)
    al = alcoa_score(response, model_id=model_id)
    ha = halluc(response, cits, grounding, doi_resolver=doi_resolver)

    vcr = (
        w["source_exist"] * se
        + w["source_match"] * sm
        + w["alcoa"] * al
        + w["halluc"] * (1.0 - ha)
    )

    return VCRBreakdown(
        source_exist=round(se, 4),
        source_match=round(sm, 4),
        alcoa_score=round(al, 4),
        halluc=round(ha, 4),
        vcr=round(vcr, 4),
    )
