"""NLI 파인튜닝 설정·라벨 규약 테스트 (GPU 비의존).

실제 학습은 A100 전용이지만, **A100 에서 첫 실행에 실패하지 않도록**
데이터 준비·라벨 매핑·엔테일먼트 추출 규약을 GPU 없이 검증한다.
"""

import pytest

from compliance_gateway.eval.scifact import DEFAULT_DIR
from compliance_gateway.nli.transformer import TransformerNLI
from compliance_gateway.train.config import NLIFinetuneConfig
from compliance_gateway.train.nli_finetune import ID2LABEL, LABEL2ID


def test_label_mapping_is_consistent():
    """0=SUPPORT=entailment 규약이 학습/추론 양쪽에서 일치해야 한다."""
    assert LABEL2ID["SUPPORT"] == 0
    assert ID2LABEL[0] == "entailment"
    assert LABEL2ID["CONTRADICT"] == 1
    assert ID2LABEL[1] == "contradiction"


def test_entail_score_resolves_named_labels():
    scores = [{"label": "entailment", "score": 0.8}, {"label": "contradiction", "score": 0.2}]
    assert TransformerNLI._entail_score(scores, "entailment") == 0.8


def test_entail_score_falls_back_to_label_0():
    """id2label 미설정 모델(LABEL_0/LABEL_1)에서도 0.0 을 반환하면 안 된다.

    이 폴백이 없으면 파인튜닝 모델의 모든 점수가 0.0 이 되어 게이트웨이가 무력화된다.
    """
    scores = [{"label": "LABEL_0", "score": 0.77}, {"label": "LABEL_1", "score": 0.23}]
    assert TransformerNLI._entail_score(scores, "entailment") == 0.77


def test_entail_score_never_crashes_on_unknown_labels():
    scores = [{"label": "weird", "score": 0.5}]
    assert TransformerNLI._entail_score(scores, "entailment") == 0.5
    assert TransformerNLI._entail_score([], "entailment") == 0.0


def test_config_has_eval_split_for_validation():
    cfg = NLIFinetuneConfig()
    assert cfg.eval_split == "dev"      # 학습이 됐는지 확인할 수단이 있어야 함


@pytest.mark.skipif(
    not (DEFAULT_DIR / "corpus.jsonl").exists(),
    reason="SciFact 미다운로드",
)
def test_dry_run_validates_data():
    from compliance_gateway.train.nli_finetune import dry_run

    report = dry_run(NLIFinetuneConfig())
    assert report["train_examples"] > 0
    assert report["eval_examples"] > 0
    assert set(report["label_distribution"]) <= {"SUPPORT", "CONTRADICT"}
