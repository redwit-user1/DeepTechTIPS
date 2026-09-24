"""외부 실데이터 평가셋 테스트.

SciFact 원본이 없으면(다운로드 전) 스킵한다 — CI 에서 데이터 없이도 통과.
"""

import pytest

from compliance_gateway.eval.scifact import DEFAULT_DIR
from compliance_gateway.verify.models import PaperRecord
from compliance_gateway.verify.scoring import weighted_score

pytestmark = pytest.mark.skipif(
    not (DEFAULT_DIR / "corpus.jsonl").exists(),
    reason="SciFact 미다운로드 (bash scripts/download_scifact.sh)",
)


def test_external_eval_builds_real_items():
    from compliance_gateway.eval.external import build_external_eval

    items, registry = build_external_eval(split="train", limit=50)
    assert items
    labels = {i["label"] for i in items}
    # 실데이터 라벨은 전문가 주석 기반 — 우리가 변조하지 않음
    assert labels <= {"compliant", "unsupported_claim"}
    # 인용된 논문은 레지스트리에 실존해야 함(서지 검증 통과 → SourceMatch 만 측정)
    assert registry.by_doi("nonexistent") is None


def test_cited_title_is_verifiable():
    """제목 인용이 실제 레지스트리에서 VALID 로 검증되어야 한다."""
    from compliance_gateway.eval.external import build_external_eval
    from compliance_gateway.models import Citation
    from compliance_gateway.verify import CitationStatus, CitationVerifier

    items, registry = build_external_eval(split="train", limit=30)
    verifier = CitationVerifier([registry])
    # 응답에서 인용 제목을 추출해 검증
    from compliance_gateway.vcr.source_exist import extract_citations

    cits = extract_citations(items[0]["response"])
    assert cits and cits[0].title
    assert verifier.verify(cits[0]).status is CitationStatus.VALID


def test_title_only_citation_not_penalized_for_missing_fields():
    """저자·연도가 없는 인용을 결측만으로 감점하면 안 된다(실데이터에서 발견된 버그).

    가용 필드로 가중치를 정규화하므로 제목이 완전히 일치하면 점수는 1.0 이어야 한다.
    """
    rec = PaperRecord(title="Effect of homocysteine lowering on mortality", authors=(), year=None)
    score, mism = weighted_score("Effect of homocysteine lowering on mortality", (), None, rec)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert mism == ()


def test_registry_exact_title_index():
    """정확 제목 인덱스가 선형 스캔보다 먼저 동작해야 한다."""
    from compliance_gateway.verify.verifier import LocalRegistry

    reg = LocalRegistry([
        PaperRecord(title="Alpha Beta Gamma study"),
        PaperRecord(title="Completely different paper"),
    ])
    got = reg.search("alpha beta gamma study")   # 정규화 후 일치
    assert len(got) == 1 and got[0].title == "Alpha Beta Gamma study"
