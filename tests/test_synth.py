"""합성 데이터 파이프라인 테스트 (커밋된 bioRxiv 시드 사용)."""

from compliance_gateway.data.build_dpo import make_doi_resolver
from compliance_gateway.data.models import Preprint
from compliance_gateway.data.synth import build_examples, load_seed
from compliance_gateway.data.tamper import fake_doi, tamper_number, tamper_polarity
from compliance_gateway.nli.statistical import StatisticalNLI
from compliance_gateway.vcr.reward import compute_vcr


def test_seed_loads():
    pps = load_seed()
    assert len(pps) >= 4
    assert all(isinstance(p, Preprint) for p in pps)


def test_preprint_citation_and_year():
    p = Preprint(
        doi="10.1101/x", title="t", authors="Alam, W.; Khan, H.",
        date="2024-01-08", abstract="a",
    )
    assert p.year == 2024
    assert p.citation() == "Alam et al. (2024)"


def test_tamper_number_changes_value():
    out = tamper_number("IC50 of 8.47 micro-M against 5-LOX")
    assert out is not None and "8.47" not in out


def test_tamper_polarity_flips_direction():
    out = tamper_polarity("antibiotics increased NBP exposure by 56.1%")
    assert out is not None and "decreased" in out.lower()


def test_fake_doi_is_not_real():
    assert fake_doi("10.1101/2024.01.08.574589") != "10.1101/2024.01.08.574589"


def test_build_examples_produces_pairs_and_evals():
    pps = load_seed()
    pairs, evals = build_examples(pps, max_claims=4)
    assert pairs and evals
    kinds = {p.rejected_kind for p in pairs}
    assert {"no_source", "fake_doi"} <= kinds


def test_vcr_self_validation_chosen_beats_rejected():
    """핵심 불변식: 3-class 서지 검증 사용 시 VCR(chosen) > VCR(rejected)."""
    from compliance_gateway.verify import CitationVerifier, LocalRegistry

    pps = load_seed()
    pairs, _ = build_examples(pps, max_claims=4)
    verifier = CitationVerifier([LocalRegistry.from_seed()])
    nli = StatisticalNLI()
    wins = 0
    for p in pairs:
        vc = compute_vcr(p.prompt, p.chosen, grounding=(p.grounding,), nli_fn=nli, verifier=verifier).vcr
        vr = compute_vcr(p.prompt, p.rejected, grounding=(p.grounding,), nli_fn=nli, verifier=verifier).vcr
        wins += int(vc > vr)
    # 적어도 90% 이상에서 chosen 우위(통계 NLI 한계로 극성 일부 제외 가능)
    assert wins / len(pairs) >= 0.9


def test_binary_resolver_misses_biblio_tampering():
    """binary DOI resolver 의 구조적 한계 — 3-class 검증기 도입 근거.

    '실존 DOI + 변조된 저자'는 DOI 존재 여부만 보는 검증기를 통과한다.
    이 한계가 CitationVerifier(3-class) 도입의 이유다.
    """
    from compliance_gateway.verify import CitationVerifier, LocalRegistry

    pps = load_seed()
    pairs, _ = build_examples(pps, max_claims=4)
    biblio = [p for p in pairs if p.rejected_kind == "biblio_tamper"]
    assert biblio, "서지 변조 샘플이 생성되어야 함"

    resolver = make_doi_resolver({p.doi for p in pps})
    verifier = CitationVerifier([LocalRegistry.from_seed()])
    p = biblio[0]

    # binary: chosen 과 rejected 를 구분하지 못함(둘 다 DOI 실존)
    v_bin_c = compute_vcr(p.prompt, p.chosen, grounding=(p.grounding,), doi_resolver=resolver).vcr
    v_bin_r = compute_vcr(p.prompt, p.rejected, grounding=(p.grounding,), doi_resolver=resolver).vcr
    assert v_bin_c == v_bin_r

    # 3-class: 서지 변조를 감지해 rejected 를 감점
    v_ver_c = compute_vcr(p.prompt, p.chosen, grounding=(p.grounding,), verifier=verifier).vcr
    v_ver_r = compute_vcr(p.prompt, p.rejected, grounding=(p.grounding,), verifier=verifier).vcr
    assert v_ver_c > v_ver_r


def test_fake_doi_detected_as_hallucination():
    """가짜 DOI(유형 A)는 실제 DOI보다 VCR 이 낮아야 한다."""
    pps = load_seed()
    pp = pps[0]
    resolver = make_doi_resolver({p.doi for p in pps})
    claim = "The most potent compound C3 showed an IC50 of 8.47 micro-M against 5-LOX."
    real = f"{claim} ({pp.citation()}; DOI: {pp.doi})"
    fake = f"{claim} ({pp.citation()}; DOI: {fake_doi(pp.doi)})"
    v_real = compute_vcr("q", real, grounding=(claim,), doi_resolver=resolver)
    v_fake = compute_vcr("q", fake, grounding=(claim,), doi_resolver=resolver)
    assert v_fake.halluc > v_real.halluc
    assert v_fake.vcr < v_real.vcr
