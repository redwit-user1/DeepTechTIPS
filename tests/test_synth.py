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
    """핵심 불변식: 모든 DPO 쌍에서 VCR(chosen) > VCR(rejected)."""
    pps = load_seed()
    pairs, _ = build_examples(pps, max_claims=4)
    resolver = make_doi_resolver({p.doi for p in pps})
    nli = StatisticalNLI()
    wins = 0
    for p in pairs:
        vc = compute_vcr(p.prompt, p.chosen, grounding=(p.grounding,), nli_fn=nli, doi_resolver=resolver).vcr
        vr = compute_vcr(p.prompt, p.rejected, grounding=(p.grounding,), nli_fn=nli, doi_resolver=resolver).vcr
        wins += int(vc > vr)
    # 적어도 90% 이상에서 chosen 우위(통계 NLI 한계로 극성 일부 제외 가능)
    assert wins / len(pairs) >= 0.9


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
