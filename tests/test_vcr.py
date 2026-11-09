"""VCR 보상함수 단위 테스트."""

from compliance_gateway.vcr.reward import compute_vcr
from compliance_gateway.vcr.source_exist import extract_citations

GROUNDING = (
    "Kim et al. (2024) reported the reaction rate increased 2.3 times at pH 7.4. "
    "The assay was conducted at 37°C. DOI: 10.1038/s41586-024-01234.",
)

GOOD = (
    "Kim et al. (2024)에 따르면 pH 7.4에서 반응속도가 2.3배 향상되었다. "
    "37°C 조건에서 수행되었다. DOI: 10.1038/s41586-024-01234"
)
NO_SOURCE = "pH 7.4에서 반응속도가 향상되는 것으로 알려져 있습니다."
TAMPERED = (
    "Kim et al. (2024)에 따르면 pH 7.4에서 반응속도가 2.3배 향상되었다. "
    "25°C 조건에서 수행되었다. DOI: 10.1038/s41586-024-01234"
)


def test_extract_citations_finds_doi_and_author_year():
    cits = extract_citations(GOOD)
    raws = " ".join(c.raw for c in cits)
    assert any(c.doi for c in cits)
    assert "et al" in raws


def test_no_source_scores_zero_exist():
    b = compute_vcr("q", NO_SOURCE, grounding=GROUNDING)
    assert b.source_exist == 0.0
    assert b.source_match == 0.0


def test_good_beats_no_source():
    good = compute_vcr("q", GOOD, grounding=GROUNDING)
    bad = compute_vcr("q", NO_SOURCE, grounding=GROUNDING)
    assert good.vcr > bad.vcr


def test_tampered_numbers_raise_hallucination():
    good = compute_vcr("q", GOOD, grounding=GROUNDING)
    tampered = compute_vcr("q", TAMPERED, grounding=GROUNDING)
    # 25°C 는 근거(37°C)와 불일치 → 환각 점수 상승
    assert tampered.halluc > good.halluc
    assert tampered.vcr < good.vcr


def test_vcr_in_unit_range():
    b = compute_vcr("q", GOOD, grounding=GROUNDING)
    for v in b.as_dict().values():
        assert 0.0 <= v <= 1.0


def test_weights_normalized():
    # 합이 1이 아닌 가중치를 넣어도 정규화되어 동작
    b = compute_vcr(
        "q", GOOD, grounding=GROUNDING,
        weights={"source_exist": 2, "source_match": 2, "alcoa": 2, "halluc": 2},
    )
    assert 0.0 <= b.vcr <= 1.0


def test_unit_swaps():
    # 값은 같고 단위만 바뀐 수치(유형 C 변형)를 잡고, 환산·비율 분모는 통과시킨다
    from compliance_gateway.vcr.hallucination import unit_swaps
    assert unit_swaps("10 mg 투여", "10 ug 투여") == ["10mg"]
    assert unit_swaps("20 mM 처리", "20 uM 처리") == ["20mm"]
    assert unit_swaps("1 g 투여", "1,000 mg 투여") == []
    assert unit_swaps("1분당 60 증가", "1초당 1 증가") == []


def test_approx_numbers_routes_near_values():
    # 근거와 10% 안쪽이지만 그대로는 아닌 수치(근사·환산) — LLM 으로 보낼 문장을 고른다
    from compliance_gateway.vcr.hallucination import approx_numbers
    assert approx_numbers("약 352 MPa", "352.233 MPa") == ["352mpa"]
    assert approx_numbers("2 시간 반응", "120 분 반응") == ["2시간"]
    assert approx_numbers("37 ℃", "37 ℃") == []           # 그대로면 해당 없음
    assert approx_numbers("75 % 수준", "98 % 이상") == []     # 확실히 다른 값은 근사가 아니다
