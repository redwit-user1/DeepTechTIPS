"""Compliance Gateway 파이프라인 테스트."""

from compliance_gateway.models import GatewayDecision, GatingMode, GenerationRequest
from compliance_gateway.pipeline import ComplianceGateway

GROUNDING = (
    "Kim et al. (2024) reported the reaction rate increased 2.3 times at pH 7.4 at 37°C. "
    "DOI: 10.1038/s41586-024-01234.",
)

GOOD = (
    "Kim et al. (2024)에 따르면 pH 7.4에서 반응속도가 2.3배 향상되었다. "
    "37°C 조건에서 수행되었다. DOI: 10.1038/s41586-024-01234"
)
NO_SOURCE = "pH 7.4에서 반응속도가 향상되는 것으로 알려져 있습니다."


def _req(resp: str) -> GenerationRequest:
    return GenerationRequest(
        query="q", response=resp, grounding=GROUNDING,
        model_id="gpt-demo", mode=GatingMode.POST,
    )


def test_good_response_passes():
    gw = ComplianceGateway(vcr_threshold=0.55)
    res = gw.evaluate(_req(GOOD))
    assert res.decision == GatewayDecision.PASS
    assert res.audit_hash and len(res.audit_hash) == 64


def test_no_source_triggers_regenerate_then_block():
    gw = ComplianceGateway(vcr_threshold=0.60, max_regenerations=3)
    # 초기 시도: 재생성 권고
    res0 = gw.evaluate(_req(NO_SOURCE), regenerations=0)
    assert res0.decision == GatewayDecision.REGENERATE
    # 재생성 소진 후: 차단
    res3 = gw.evaluate(_req(NO_SOURCE), regenerations=3)
    assert res3.decision == GatewayDecision.BLOCK


def test_reports_cover_all_stages():
    gw = ComplianceGateway()
    res = gw.evaluate(_req(GOOD))
    stages = {r.stage for r in res.reports}
    assert {"source_binding", "nli_gating", "alcoa_checkpoint", "audit_trail"} <= stages


def test_audit_hash_is_deterministic():
    gw = ComplianceGateway()
    a = gw.evaluate(_req(GOOD)).audit_hash
    b = gw.evaluate(_req(GOOD)).audit_hash
    assert a == b
