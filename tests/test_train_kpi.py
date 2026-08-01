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
