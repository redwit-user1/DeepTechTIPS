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


def test_sentence_gating_blocks_unsupported_sentence():
    # VCR 은 인용 문장만 본다. 원문을 옮긴 보고서 문장은 05 단계 문장 게이팅으로만 검증된다.
    from compliance_gateway.models import GatewayDecision, GenerationRequest
    from compliance_gateway.pipeline import ComplianceGateway

    src = "IC 분석 결과 운전유속 5 m/h 에서 Br 제거율이 98 % 이상으로 확인되었음."
    fake = lambda p, h: 0.9 if "98" in h else 0.02          # 98 을 말하면 뒷받침
    gw = ComplianceGateway(vcr_threshold=0.0, nli_fn=fake, sentence_threshold=0.1)
    ok = gw.evaluate(GenerationRequest(query="q", response="운전유속 5 m/h 에서 Br 제거율은 98 % 이상이었다.",
                                       grounding=(src,)))
    bad = gw.evaluate(GenerationRequest(query="q", response="운전유속 5 m/h 에서 Br 제거율은 75 % 수준이었다.",
                                        grounding=(src,)), regenerations=3)
    assert ok.decision is GatewayDecision.PASS
    assert bad.decision is GatewayDecision.BLOCK
    gating = next(r for r in bad.reports if r.stage == "nli_gating")
    assert gating.detail["unsupported_sentences"][0]["score"] == 0.02
    # 게이팅 미설정이면 기존 동작(문장 검사 없음)
    off = ComplianceGateway(vcr_threshold=0.0, nli_fn=fake).evaluate(
        GenerationRequest(query="q", response="Br 제거율은 75 % 수준이었다.", grounding=(src,)))
    assert off.decision is GatewayDecision.PASS


def test_cascade_routes_only_ambiguous_to_llm():
    from compliance_gateway.nli.cascade import CascadeNLI

    enc = lambda p, h: {"a": 0.001, "b": 0.5, "c": 0.95}[h[0]]
    llm_calls = []
    llm = lambda p, h: (llm_calls.append(h), 0.7)[1]
    c = CascadeNLI(enc, llm, lo=0.005, hi=0.8, threshold=0.1)
    assert c("근거", "a 문장") == 0.001        # 인코더가 확신(위반)
    assert c("근거", "c 문장") == 0.95         # 인코더가 확신(통과)
    assert c("근거", "b 문장") == 0.7          # 애매 → LLM
    assert c("10 ug 투여", "b 10 mg 투여") == 0.0   # 단위 바꿈은 규칙이 먼저 잡는다
    assert llm_calls == ["b 문장"] and c.llm_calls == 1
    # 근사·환산 수치는 인코더 확신과 무관하게 LLM 으로
    assert c("수율 18.7 % 증가", "c 수율 약 19 % 증가") == 0.7 and c.llm_calls == 2
