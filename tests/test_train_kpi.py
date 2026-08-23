"""학습 설정·데이터포맷·KPI 하니스 테스트 (GPU 비의존).

무거운 학습 실행(sft/dpo/nli_finetune)은 A100 환경 전용이라 여기서 테스트하지 않는다.
설정·포맷·측정 로직만 검증한다.
"""

import json

from compliance_gateway.eval.kpi import evaluate, load_eval_items, prf
from compliance_gateway.nli.statistical import StatisticalNLI
from compliance_gateway.pipeline import ComplianceGateway
from compliance_gateway.train.config import SFTConfig, DPOConfig, to_dict
from compliance_gateway.train.data_format import (
    dpo_pairs_to_dpo,
    dpo_pairs_to_sft,
    to_sft_record,
)


def test_config_resolves_model_id():
    assert "EXAONE" in SFTConfig(base_model="exaone").model_id()
    assert "Qwen" in DPOConfig(base_model="qwen").model_id()
    # 미등록 키는 그대로 경로로 취급(로컬 모델 경로 지원)
    assert DPOConfig(base_model="/local/path").model_id() == "/local/path"


def test_config_serializable():
    d = to_dict(SFTConfig())
    assert d["lora"]["r"] == 16 and d["epochs"] == 2


def test_sft_record_shape():
    r = to_sft_record("q", "a")
    assert r["messages"][0]["role"] == "user"
    assert r["messages"][1]["content"] == "a"


def test_dpo_format_and_threshold(tmp_path):
    rows = [
        {"prompt": "p1", "chosen": "c1", "rejected": "r1", "vcr_chosen": 0.8, "vcr_rejected": 0.3},
        {"prompt": "p2", "chosen": "c2", "rejected": "r2", "vcr_chosen": 0.5, "vcr_rejected": 0.4},
        {"prompt": "p3", "chosen": "c3", "rejected": "r3", "vcr_chosen": 0.2, "vcr_rejected": 0.6},
    ]
    f = tmp_path / "dpo.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # 임계값 0: chosen>rejected 인 것만(3번째 제외) → 2개
    assert len(dpo_pairs_to_dpo(f, vcr_accept_threshold=0.0)) == 2
    # 임계값 0.7: vcr_chosen>=0.7 인 1개만
    accepted = dpo_pairs_to_dpo(f, vcr_accept_threshold=0.7)
    assert len(accepted) == 1 and accepted[0]["prompt"] == "p1"
    # SFT 재사용: chosen 3개(중복 없음)
    assert len(dpo_pairs_to_sft(f)) == 3


def test_prf_metric():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, False]
    prec, rec, f1 = prf(y_true, y_pred)
    assert prec == 1.0 and rec == 0.5


def test_kpi_harness_on_seed():
    items = load_eval_items(None)  # 시드에서 생성
    resolver_dois = {"10.1101/2024.01.08.574589"}
    from compliance_gateway.data.build_dpo import make_doi_resolver
    from compliance_gateway.data.synth import load_seed
    resolver = make_doi_resolver({p.doi for p in load_seed()})
    gw = ComplianceGateway(vcr_threshold=0.55, nli_fn=StatisticalNLI(), doi_resolver=resolver)
    m = evaluate(gw, items)
    # 합성 데이터에서 위반 탐지 정밀도는 높아야(목표 90%+ 검증 체계 동작)
    assert m["violation_precision"] >= 0.9
    assert 0.0 <= m["compliant_pass_rate"] <= 1.0
    assert "polarity_flip" in m["per_type_detection"]


# ── H100 프로파일 / 서빙 백엔드 ────────────────────────────────────────

def test_h100_is_default_profile_with_fp8():
    from compliance_gateway.train.config import DEFAULT_GPU, GRPOConfig, GPU_PROFILES

    assert DEFAULT_GPU == "h100"
    assert GPU_PROFILES["h100"]["fp8_capable"] is True
    assert GPU_PROFILES["a100"]["fp8_capable"] is False
    assert GRPOConfig().fp8 is True          # H100 기본


def test_validate_flags_fp8_on_a100():
    """A100 에 FP8 을 켜면 경고해야 한다(하드웨어 미지원)."""
    from compliance_gateway.train.config import GRPOConfig, validate

    warns = validate(GRPOConfig(gpu="a100"))
    assert warns and "FP8" in warns[0]
    assert validate(GRPOConfig(gpu="h100")) == []


def test_validate_flags_pointless_4bit_on_h100():
    from compliance_gateway.train.config import SFTConfig, validate

    assert validate(SFTConfig(load_in_4bit=True))
    assert validate(SFTConfig()) == []


def test_served_nli_parses_verdicts():
    from compliance_gateway.serving import ServedNLI

    assert ServedNLI.parse_verdict("SUPPORT") == 1.0
    assert ServedNLI.parse_verdict("CONTRADICT") == 0.0
    assert ServedNLI.parse_verdict("NEUTRAL") == 0.5
    assert ServedNLI.parse_verdict("garbage") == 0.5    # 파싱 실패 → 중립


def test_served_model_normalizes_base_url():
    from compliance_gateway.serving import ServedModel

    assert ServedModel("http://h:8000", "m").base_url == "http://h:8000/v1"
    assert ServedModel("http://h:8000/v1/", "m").base_url == "http://h:8000/v1"


def test_select_nli_requires_model_id_for_endpoint():
    import pytest

    from compliance_gateway.eval.nli_select import select_nli

    with pytest.raises(SystemExit):
        select_nli(endpoint="http://h:8000/v1")          # --nli-model 누락
    fn, name = select_nli()                              # 기본 폴백
    assert name == "statistical-v0.5"


# ── 보상함수 사전 점검 게이트 ──────────────────────────────────────────

def test_citation_spanning_sentence_boundary_is_counted():
    """인용 안의 마침표('Alam et al.')로 문장이 쪼개져도 출처로 인정돼야 한다.

    부분문자열 매칭 시절의 회귀: 병합 인용이 두 조각에 걸쳐 source_exist=0 이 됐다.
    """
    from compliance_gateway.vcr.source_exist import extract_citations, source_exist

    text = ("The most potent compound C3 showed an IC50 of 8.47 micro-M. "
            "(Alam et al. (2024); DOI: 10.1101/2024.01.08.574589)")
    assert source_exist(text, extract_citations(text)) == 1.0


def test_source_exist_across_citation_styles():
    from compliance_gateway.vcr.source_exist import extract_citations, source_exist

    cases = [
        'Claim here. (cf. "A Study of Something Important in Biology")',
        "서울대학교병원은 연구를 수행하였다. (출처: 서울대학교병원, 2020, 과제번호 NCT04490642)",
        "Finding X. (Kim et al. (2024); DOI: 10.1101/2024.01.01.000001)",
    ]
    for t in cases:
        assert source_exist(t, extract_citations(t)) == 1.0, t
    assert source_exist("출처 없는 주장이다.", []) == 0.0


def test_reward_check_detects_degenerate_component():
    """실데이터 KR 은 인용이 항상 실존 → source_exist/halluc 이 상수.

    단일 데이터셋으로 GRPO 를 돌리면 보상 가중치가 낭비된다는 사실을 고정한다.
    """
    import statistics

    from compliance_gateway.nli.statistical import StatisticalNLI
    from compliance_gateway.train.reward_check import _load
    from compliance_gateway.vcr.reward import compute_vcr
    from compliance_gateway.verify import CitationVerifier

    items, registry = _load("kr_real")
    nli, verifier = StatisticalNLI(), CitationVerifier([registry])
    breakdowns = [
        compute_vcr(i["query"], i["response"], grounding=(i["grounding"],),
                    nli_fn=nli, verifier=verifier)
        for i in items[:40]
    ]
    exists = [b.source_exist for b in breakdowns]
    assert statistics.pstdev(exists) < 1e-6      # 상수 = 학습 신호 없음


def test_reward_check_hack_probes_score_low():
    """빈 응답·인용 나열·근거 복붙이 높은 보상을 받으면 안 된다."""
    from compliance_gateway.nli.statistical import StatisticalNLI
    from compliance_gateway.train.reward_check import MAX_HACK_SCORE, _hack_probes, _load
    from compliance_gateway.vcr.reward import compute_vcr
    from compliance_gateway.verify import CitationVerifier

    items, registry = _load("kr_real")
    nli, verifier = StatisticalNLI(), CitationVerifier([registry])
    for name, resp, g in _hack_probes(items):
        v = compute_vcr("q", resp, grounding=(g,), nli_fn=nli, verifier=verifier).vcr
        assert v <= MAX_HACK_SCORE, f"보상 해킹 가능: {name} → {v}"
