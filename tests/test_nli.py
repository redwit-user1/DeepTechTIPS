"""NLI 백엔드 + 벤치마크 지표 테스트 (데이터셋 비의존)."""

from compliance_gateway.eval.benchmark import auc
from compliance_gateway.nli.lexical import lexical_nli
from compliance_gateway.nli.statistical import StatisticalNLI

EVIDENCE = "Treatment with the drug decreased tumor growth at 37 degrees."
SUPPORT = "The drug decreased tumor growth."
CONTRADICT = "The drug increased tumor growth."


def test_lexical_fails_to_reject_contradiction():
    # 토큰 중첩은 반박 주장에도 높은 점수를 줘서 '거르지 못한다'(R1 리스크).
    assert lexical_nli(EVIDENCE, CONTRADICT) >= 0.4


def test_statistical_penalizes_contradiction_vs_lexical():
    nli = StatisticalNLI()
    # 극성(반의어 increase/decrease) 충돌 → support 보다 낮고, lexical 보다도 강하게 깎임
    assert nli(EVIDENCE, SUPPORT) > nli(EVIDENCE, CONTRADICT)
    assert nli(EVIDENCE, CONTRADICT) < lexical_nli(EVIDENCE, CONTRADICT)


def test_statistical_fit_changes_idf():
    nli = StatisticalNLI()
    nli.fit(["the drug decreased tumor", "another unrelated sentence"])
    assert nli.idf  # idf 학습됨
    assert 0.0 <= nli(EVIDENCE, SUPPORT) <= 1.0


def test_auc_metric():
    # 완벽 분리 → 1.0, 역전 → 0.0
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auc([0.5], [0.5]) == 0.5
