"""국내 R&D 한국어 데이터셋 테스트."""

import pytest

from compliance_gateway.data.korean import kr_render as R
from compliance_gateway.data.korean.build_kr import build_items, build_registry
from compliance_gateway.data.korean.sources import (
    ArxivAdapter,
    ScienceONAdapter,
    is_korean_institution,
    load_kr_seed,
)
from compliance_gateway.vcr.source_exist import extract_citations


def test_seed_loads_korean_records():
    recs = load_kr_seed()
    assert len(recs) >= 20
    assert all(is_korean_institution(r.sponsor) for r in recs)


def test_foreign_institutions_filtered_out():
    assert is_korean_institution("Seoul National University Hospital")
    assert is_korean_institution("KAIST")
    assert not is_korean_institution("University of Houston")
    assert not is_korean_institution("Cambridge Health Alliance")


def test_josa_selection_by_final_consonant():
    assert R.with_josa("서울대학교병원", "은는") == "서울대학교병원은"   # 받침 O
    assert R.with_josa("연세대학교", "은는") == "연세대학교는"           # 받침 X
    assert R.with_josa("50명", "을를") == "50명을"


def test_korean_citation_is_extracted():
    """국내 R&D 출처 표기(과제번호·기관·연도)가 하나의 인용으로 인식되어야 한다."""
    rec = load_kr_seed()[0]
    cits = extract_citations(R.render_cited(rec))
    assert cits, "한국어 인용이 추출되지 않음"
    c = cits[0]
    assert c.doi == rec.nct_id
    assert c.year == rec.year
    assert c.authors


def test_korean_author_year_citation():
    cits = extract_citations("김철수 외 (2023)에 따르면 효과가 확인되었다.")
    assert cits and cits[0].year == 2023


def test_alcoa_tampering_changes_text():
    rec = next(r for r in load_kr_seed() if r.enrollment and r.year)
    base = R.render_cited(rec)
    assert R.tamper_sponsor(rec) != base       # Attributable
    assert R.tamper_enrollment(rec) != base    # Accurate
    assert R.tamper_date(rec) != base          # Contemporaneous
    assert R.tamper_id(rec) != base            # Original
    assert R.strip_source(rec) != base


def test_build_covers_all_alcoa_attributes():
    evals, pairs = build_items(load_kr_seed())
    attrs = {e["alcoa_violation"] for e in evals if e["alcoa_violation"]}
    assert attrs == {"Attributable", "Accurate", "Contemporaneous", "Original"}
    assert pairs and all(p["lang"] == "ko" for p in pairs)


def test_compliant_scores_above_all_violations():
    """핵심 불변식: 준수 응답의 VCR 이 모든 위반 유형보다 높아야 한다."""
    from compliance_gateway.nli.statistical import StatisticalNLI
    from compliance_gateway.vcr.reward import compute_vcr
    from compliance_gateway.verify import CitationVerifier

    recs = load_kr_seed()
    evals, _ = build_items(recs)
    v = CitationVerifier([build_registry(recs)])
    nli = StatisticalNLI()

    def vcr_of(e):
        return compute_vcr(e["query"], e["response"], grounding=(e["grounding"],),
                           nli_fn=nli, verifier=v).vcr

    ok = [vcr_of(e) for e in evals if e["label"] == "compliant"]
    bad = [vcr_of(e) for e in evals if e["label"] != "compliant"]
    assert min(ok) > max(bad), "준수 응답이 위반보다 낮게 평가됨"


def test_korean_numeric_units_recognized():
    """등록례수 변조(ALCOA+ Accurate) 탐지를 위해 한국어 단위 인식이 필요."""
    from compliance_gateway.vcr.hallucination import _NUM_UNIT

    found = dict(_NUM_UNIT.findall("총 2,927명이 등록되었고 50건이 확인되었다"))
    assert found.get("2,927") == "명"
    assert found.get("50") == "건"


def test_blocked_source_adapters_fail_clearly():
    """차단된 소스는 조용히 실패하지 말고 원인을 알려야 한다."""
    with pytest.raises(RuntimeError, match="API 키"):
        ScienceONAdapter().search("연구데이터")


def test_arxiv_parser_works_offline():
    """arXiv 어댑터 파서는 네트워크 없이도 검증 가능해야 한다."""
    xml = """<feed><entry>
      <id>http://arxiv.org/abs/2401.00001v1</id>
      <title>A Study on Korean R&amp;D Data</title>
      <summary>We present a dataset.</summary>
      <published>2024-01-01T00:00:00Z</published>
      <author><name>Hong Gildong</name></author>
    </entry></feed>"""
    recs = ArxivAdapter.parse(xml)
    assert len(recs) == 1
    assert recs[0]["arxiv_id"] == "2401.00001v1"
    assert "Korean R&D Data" in recs[0]["title"]
    assert recs[0]["authors"] == ("Hong Gildong",)


# ── 실데이터 평가셋 (변조 없음, 출처 기반 라벨) ────────────────────────

def test_real_protocols_are_verbatim_source_text():
    """실데이터셋의 주장 값은 원문 그대로여야 한다(우리가 만들지 않음)."""
    from compliance_gateway.data.korean.real_eval import build, load_protocols

    protocols = load_protocols()
    assert len(protocols) >= 9
    items, _ = build(protocols)

    by_id = {t["nct_id"]: t for t in protocols}
    for it in items:
        if it["label"] != "compliant" or it["field"] != "primary_outcome":
            continue
        t = by_id[it["source_id"]]
        # 실제 primary_outcome 원문이 응답에 그대로 들어 있어야 함
        assert t["primary_outcome"] in it["response"]
        # 근거도 실제 프로토콜 원문
        assert t["brief_summary"] in it["grounding"]


def test_real_negatives_come_from_other_real_trials():
    """음성은 '다른 실제 과제의 원문'이어야 한다 — 규칙 변조가 아님."""
    from compliance_gateway.data.korean.real_eval import build, load_protocols

    protocols = load_protocols()
    items, _ = build(protocols)
    outcomes = {t["primary_outcome"] for t in protocols}

    negatives = [i for i in items
                 if i["label"] == "misattributed_primary_outcome"]
    assert negatives
    for neg in negatives:
        # 잘못 귀속된 값이 실제 다른 과제의 결과변수 중 하나여야 한다
        assert any(o in neg["response"] for o in outcomes)


def test_real_eval_is_balanced_and_covers_alcoa():
    from compliance_gateway.data.korean.real_eval import build, load_protocols

    items, pairs = build(load_protocols())
    n_ok = sum(1 for i in items if i["label"] == "compliant")
    assert n_ok == len(items) - n_ok          # 균형
    attrs = {i["alcoa_violation"] for i in items if i["alcoa_violation"]}
    assert {"Accurate", "Attributable", "Complete"} <= attrs
    assert pairs


def test_real_dataset_is_harder_than_template():
    """실데이터셋은 템플릿 합성셋보다 반드시 어려워야 한다.

    템플릿에서 AUC 1.0 이 나오는 것은 인공물이다. 실데이터 AUC 가 그보다
    낮게 유지되는지 회귀 감시한다(쉬워졌다면 데이터가 오염된 것).
    """
    from compliance_gateway.data.korean.build_kr import build_items as tmpl_build
    from compliance_gateway.data.korean.build_kr import build_registry as tmpl_reg
    from compliance_gateway.data.korean.real_eval import (
        build as real_build, build_registry as real_reg, load_protocols,
    )
    from compliance_gateway.eval.benchmark import auc
    from compliance_gateway.nli.statistical import StatisticalNLI
    from compliance_gateway.vcr.reward import compute_vcr
    from compliance_gateway.verify import CitationVerifier

    nli = StatisticalNLI()

    def auc_of(items, registry):
        v = CitationVerifier([registry])
        pos, neg = [], []
        for e in items:
            s = compute_vcr(e["query"], e["response"], grounding=(e["grounding"],),
                            nli_fn=nli, verifier=v).vcr
            (pos if e["label"] == "compliant" else neg).append(s)
        return auc(pos, neg)

    protocols = load_protocols()
    real_auc = auc_of(real_build(protocols)[0], real_reg(protocols))
    tmpl_items, _ = tmpl_build(load_kr_seed())
    tmpl_auc = auc_of(tmpl_items, tmpl_reg(load_kr_seed()))

    assert real_auc < tmpl_auc, "실데이터가 템플릿보다 쉬우면 데이터 구성이 잘못된 것"
    assert real_auc > 0.5, "완전 무작위면 신호가 전혀 없다는 뜻"
