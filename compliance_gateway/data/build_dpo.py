"""DPO 데이터셋 빌드 + VCR 자기검증.

실행:
  python -m compliance_gateway.data.build_dpo
  python -m compliance_gateway.data.build_dpo --seed compliance_gateway/data/seed/biorxiv_pharma.json

산출:
  data/synth/dpo_pairs.jsonl      DPO Preference 쌍(VCR 점수 포함)
  data/synth/gateway_eval.jsonl   Gateway 평가셋
  + 콘솔 리포트(VCR 승률, kind별 마진, Gateway 결정 정확도)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compliance_gateway.data.synth import build_examples, load_seed
from compliance_gateway.models import GatewayDecision, GatingMode, GenerationRequest
from compliance_gateway.nli.statistical import StatisticalNLI
from compliance_gateway.pipeline import ComplianceGateway
from compliance_gateway.vcr.reward import compute_vcr


def make_doi_resolver(known_dois: set[str]):
    """시드에 존재하는 DOI만 실존으로 간주(가짜 DOI 탐지용)."""
    def resolve(doi: str) -> bool:
        return doi in known_dois
    return resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--max-claims", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    preprints = load_seed(Path(args.seed) if args.seed else None)
    pairs, evals = build_examples(preprints, max_claims=args.max_claims)

    known = {pp.doi for pp in preprints}
    resolver = make_doi_resolver(known)
    nli = StatisticalNLI()

    # --- VCR 자기검증: chosen vs rejected ---
    for p in pairs:
        p.vcr_chosen = compute_vcr(
            p.prompt, p.chosen, grounding=(p.grounding,),
            nli_fn=nli, doi_resolver=resolver,
        ).vcr
        p.vcr_rejected = compute_vcr(
            p.prompt, p.rejected, grounding=(p.grounding,),
            nli_fn=nli, doi_resolver=resolver,
        ).vcr

    wins = sum(1 for p in pairs if p.vcr_chosen > p.vcr_rejected)
    win_rate = wins / len(pairs) if pairs else 0.0

    # kind별 평균 마진
    kinds: dict[str, list[float]] = {}
    for p in pairs:
        kinds.setdefault(p.rejected_kind, []).append(p.margin)

    # --- Gateway 결정 검증 ---
    gw = ComplianceGateway(vcr_threshold=args.threshold, nli_fn=nli, doi_resolver=resolver)
    compliant_pass = compliant_total = 0
    violation_caught = violation_total = 0
    for item in evals:
        req = GenerationRequest(
            query=item.query, response=item.response, grounding=(item.grounding,),
            model_id="synthetic", mode=GatingMode.POST,
        )
        res = gw.evaluate(req)
        passed = res.decision == GatewayDecision.PASS
        if item.label == "compliant":
            compliant_total += 1
            compliant_pass += int(passed)
        else:
            violation_total += 1
            violation_caught += int(not passed)

    # --- 출력 ---
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "dpo_pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n")
    with (out_dir / "gateway_eval.jsonl").open("w", encoding="utf-8") as f:
        for item in evals:
            f.write(json.dumps(item.to_json(), ensure_ascii=False) + "\n")

    print(f"preprints={len(preprints)}  DPO pairs={len(pairs)}  eval items={len(evals)}")
    print(f"\n[VCR 자기검증] chosen > rejected 승률: {win_rate:.1%}  ({wins}/{len(pairs)})")
    print("  kind별 평균 VCR 마진(chosen-rejected):")
    for kind, margins in sorted(kinds.items()):
        print(f"    {kind:16s} {sum(margins)/len(margins):+.4f}  (n={len(margins)})")
    print("\n[Gateway 결정 검증]")
    print(f"  compliant PASS:   {compliant_pass}/{compliant_total} "
          f"({compliant_pass/compliant_total:.1%})" if compliant_total else "  (none)")
    print(f"  violation 차단:   {violation_caught}/{violation_total} "
          f"({violation_caught/violation_total:.1%})" if violation_total else "  (none)")
    print(f"\n→ {out_dir}/dpo_pairs.jsonl, gateway_eval.jsonl")


if __name__ == "__main__":
    main()
