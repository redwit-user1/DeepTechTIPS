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
