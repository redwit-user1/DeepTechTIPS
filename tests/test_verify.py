"""인용 검증(3-class) 테스트."""

from compliance_gateway.models import Citation
from compliance_gateway.verify import (
    CitationStatus,
    CitationVerifier,
    LocalRegistry,
    PaperRecord,
)
from compliance_gateway.verify.scoring import (
    THRESHOLD_PARTIAL,
    THRESHOLD_VALID,
    author_similarity,
    title_similarity,
    weighted_score,
    year_similarity,
)
from compliance_gateway.vcr.source_exist import extract_citations

REAL_DOI = "10.1101/2024.01.08.574589"


def _verifier() -> CitationVerifier:
    return CitationVerifier([LocalRegistry.from_seed()])


def test_thresholds_match_upstream():
    assert THRESHOLD_VALID == 0.92
    assert THRESHOLD_PARTIAL == 0.70


def test_title_and_year_similarity():
    assert title_similarity("Propofol binds RyR1", "propofol binds ryr1") > 0.95
    assert year_similarity(2024, 2024) == 1.0
    assert year_similarity(2024, 2023) == 0.5   # preprint→저널 1년차는 관대
    assert year_similarity(2024, 2019) == 0.0


def test_author_similarity_surname_normalization():
    rec = PaperRecord(authors=("Alam, W.", "Khan, H."))
    assert author_similarity(("Alam et al.",), rec) == 1.0
    assert author_similarity(("Zhang et al.",), rec) < 0.5


def test_real_doi_with_correct_metadata_is_valid():
    v = _verifier()
    c = Citation(raw=REAL_DOI, doi=REAL_DOI, authors=("Alam et al.",), year=2024)
    assert v.verify(c).status is CitationStatus.VALID


def test_fake_doi_is_hallucinated():
    v = _verifier()
    c = Citation(raw="x", doi="10.1101/2024.99.99.999999", authors=("Alam et al.",), year=2024)
    r = v.verify(c)
    assert r.status is CitationStatus.HALLUCINATED
    assert r.penalty == 1.0


def test_real_doi_wrong_author_is_partially_valid():
    """binary resolver 가 놓치는 서지 변조 — 3-class 의 존재 이유."""
    v = _verifier()
    c = Citation(raw=REAL_DOI, doi=REAL_DOI, authors=("Zhang et al.",), year=2024)
    r = v.verify(c)
    assert r.status is CitationStatus.PARTIALLY_VALID
    assert "author" in r.mismatches
    assert r.penalty == 0.5


def test_real_doi_wrong_year_flags_mismatch():
    v = _verifier()
    c = Citation(raw=REAL_DOI, doi=REAL_DOI, authors=("Alam et al.",), year=2019)
    r = v.verify(c)
    assert r.status is CitationStatus.PARTIALLY_VALID
    assert "year" in r.mismatches


def test_no_backend_returns_unverified_and_no_penalty():
    v = CitationVerifier([])
    r = v.verify(Citation(raw=REAL_DOI, doi=REAL_DOI))
    assert r.status is CitationStatus.UNVERIFIED
    assert r.penalty == 0.0     # 조회 실패로 정상 인용을 벌하지 않음


def test_backwards_compatible_resolver_adapter():
    resolve = _verifier().as_doi_resolver()
    assert resolve(REAL_DOI) is True
    assert resolve("10.1101/2024.99.99.999999") is False


def test_citation_merging_combines_doi_and_author_year():
    """같은 참고문헌의 저자-연도 + DOI 는 하나로 병합되어야 검증 가능."""
    text = f"Claim here. (Alam et al. (2024); DOI: {REAL_DOI})"
    cits = extract_citations(text)
    assert len(cits) == 1
    c = cits[0]
    assert c.doi == REAL_DOI and c.year == 2024 and c.authors

    # 병합을 끄면 분리된다(회귀 방지)
    assert len(extract_citations(text, merge=False)) == 2
