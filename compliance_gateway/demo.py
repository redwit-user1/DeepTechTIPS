"""VCR / Compliance Gateway 데모.

    python -m compliance_gateway.demo

규정 준수 응답 vs 복합기만형 환각 응답을 게이트웨이로 비교한다(제약·바이오 도메인).
"""

from __future__ import annotations

from compliance_gateway.models import GatingMode, GenerationRequest
from compliance_gateway.pipeline import ComplianceGateway

# RAG로 검색된 신뢰 근거(grounding)
GROUNDING = (
    "Kim et al. (2024) reported that the reaction rate increased 2.3 times at pH 7.4. "
    "The assay was conducted at 37°C. DOI: 10.1038/s41586-024-01234.",
)

QUERY = "pH 7.4 조건에서 반응속도 변화를 출처와 함께 요약해줘."

# y_w: 출처 ✓ 정확 ✓ (규정 준수)
GOOD = (
    "Kim et al. (2024)에 따르면, pH 7.4에서 반응속도가 2.3배 향상되었다. "
    "해당 실험은 37°C 조건에서 수행되었다. DOI: 10.1038/s41586-024-01234"
)

# y_l: 출처 없음 (비준수)
NO_SOURCE = "pH 7.4에서 반응속도가 향상되는 것으로 알려져 있습니다."

# 유형 C: 수치 변조 (37°C → 25°C)
TAMPERED = (
    "Kim et al. (2024)에 따르면, pH 7.4에서 반응속도가 2.3배 향상되었다. "
    "해당 실험은 25°C 조건에서 수행되었다. DOI: 10.1038/s41586-024-01234"
)


def _run(gw: ComplianceGateway, label: str, response: str) -> None:
    req = GenerationRequest(
        query=QUERY,
        response=response,
        grounding=GROUNDING,
        model_id="gpt-demo",
        mode=GatingMode.POST,
    )
    res = gw.evaluate(req)
    b = res.vcr
    print(f"\n[{label}] decision={res.decision.value}")
    print(f"  VCR={b.vcr}  (exist={b.source_exist} match={b.source_match} "
          f"alcoa={b.alcoa_score} halluc={b.halluc})")
    print(f"  citations={len(res.citations)}  audit={res.audit_hash[:16]}…")


def main() -> None:
    # θ 는 NLI 백엔드에 맞춰 보정한다(VCR v2에서 도메인별 자동 최적화).
    gw = ComplianceGateway(vcr_threshold=0.55)
    print("=== Compliance Gateway 데모 (제약·바이오 / NLI=statistical v0.5 / θ=0.55) ===")
    _run(gw, "규정준수 응답", GOOD)
    _run(gw, "출처없는 응답", NO_SOURCE)
    _run(gw, "수치변조 응답(유형C)", TAMPERED)


if __name__ == "__main__":
    main()
