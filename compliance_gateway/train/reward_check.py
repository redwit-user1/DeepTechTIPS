"""보상함수 사전 점검 게이트 — GRPO 학습 전 필수 관문.

업스트림 GRPO/RLVR 스킬(wshobson/agents `llm-finetuning`)의 원칙을 구현한다:
> "학습 전에 50~100개 샘플 출력에 보상함수를 돌려 조용한 오정렬을 잡아라."

RL 은 보상함수가 잘못돼도 **조용히** 잘못된 목표를 최적화한다(reward hacking).
GPU 시간을 쓰기 전에 아래를 확인한다.

  1. 분리도    — 준수 응답과 위반 응답의 보상이 실제로 갈리는가 (AUC)
  2. 축퇴      — 모든 샘플이 같은 점수는 아닌가 (학습 신호 없음)
  3. 해킹 취약 — 빈 문자열·인용만·근거 복붙으로 만점을 얻을 수 있는가
  4. 구성요소  — VCR 4요소가 각각 살아 있는가(하나가 상수면 사실상 3요소)

실행:
  python -m compliance_gateway.train.reward_check                 # 국내 실데이터
  python -m compliance_gateway.train.reward_check --dataset synth # bioRxiv 합성
  python -m compliance_gateway.train.reward_check --min-auc 0.8   # 게이트 강화
"""

from __future__ import annotations

import argparse
import statistics
import sys

from compliance_gateway.eval.benchmark import auc
from compliance_gateway.nli.statistical import StatisticalNLI
from compliance_gateway.vcr.reward import compute_vcr
from compliance_gateway.verify import CitationVerifier

# 학습 신호가 존재한다고 보기 위한 최소 조건
MIN_AUC = 0.70            # 준수 > 위반 순위 정확도
MIN_STDEV = 0.02          # 점수 분산(축퇴 방지)
MAX_HACK_SCORE = 0.60     # 해킹 시도 응답이 이보다 높으면 위험


def _load(dataset: str):
    """(items, registry) — items 는 query/response/grounding/label 딕셔너리."""
    if dataset == "kr_real":
        from compliance_gateway.data.korean.real_eval import (
            build, build_registry, load_protocols,
        )
        p = load_protocols()
        return build(p)[0], build_registry(p)
    if dataset == "kr_synth":
        from compliance_gateway.data.korean.build_kr import build_items, build_registry
        from compliance_gateway.data.korean.sources import load_kr_seed
        recs = load_kr_seed()
        return build_items(recs)[0], build_registry(recs)
    if dataset == "mixed":
        # 단일 데이터셋은 VCR 4요소를 다 자극하지 못한다(예: kr_real 은 항상 실존 인용).
        # GRPO 학습 데이터는 반드시 혼합해야 가중치가 낭비되지 않는다.
        from compliance_gateway.verify.verifier import LocalRegistry
        merged_items, merged_records = [], []
        for name in ("kr_real", "synth"):
            it, reg = _load(name)
            merged_items += it
            merged_records += list(getattr(reg, "_records", []))
        return merged_items, LocalRegistry(merged_records)
    if dataset == "synth":
        from compliance_gateway.data.build_dpo import make_doi_resolver  # noqa: F401
        from compliance_gateway.data.synth import build_examples, load_seed
        from compliance_gateway.verify import LocalRegistry
        seed = load_seed()
        evals = build_examples(seed, max_claims=4)[1]
        return [e.to_json() for e in evals], LocalRegistry.from_seed()
    raise SystemExit(f"알 수 없는 dataset: {dataset}")


def _hack_probes(items: list[dict]) -> list[tuple[str, str, str]]:
    """(이름, 응답, 근거) — 보상 해킹 시도. 낮은 점수가 나와야 정상."""
    sample = items[0]
    g = sample["grounding"]
    return [
        ("빈 응답", "", g),
        ("인용만 나열", "(출처: X, 2024, 과제번호 NCT00000000)", g),
        ("근거 그대로 복붙(무출처)", g, g),
        ("무의미 반복", "연구 " * 60, g),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kr_real",
                    choices=["kr_real", "kr_synth", "synth", "mixed"])
    ap.add_argument("--samples", type=int, default=100,
                    help="점검 표본 수(업스트림 권장 50~100)")
    ap.add_argument("--min-auc", type=float, default=MIN_AUC)
    a = ap.parse_args()

    items, registry = _load(a.dataset)
    items = items[: a.samples]
    nli = StatisticalNLI()
    verifier = CitationVerifier([registry])

    def score(it: dict):
        return compute_vcr(it["query"], it["response"], grounding=(it["grounding"],),
                           nli_fn=nli, verifier=verifier)

    breakdowns = [(it["label"], score(it)) for it in items]
    pos = [b.vcr for lbl, b in breakdowns if lbl == "compliant"]
    neg = [b.vcr for lbl, b in breakdowns if lbl != "compliant"]

    print("=" * 64)
    print(f" 보상함수 사전 점검 — dataset={a.dataset}, n={len(items)}")
    print("=" * 64)

    failures: list[str] = []

    # 1) 분리도
    print("\n[1] 분리도 (준수 vs 위반)")
    if pos and neg:
        a_val = auc(pos, neg)
        print(f"    AUC            {a_val:.3f}   (기준 ≥{a.min_auc})")
        print(f"    mean(준수)     {statistics.fmean(pos):.4f}")
        print(f"    mean(위반)     {statistics.fmean(neg):.4f}")
        print(f"    분리도          {statistics.fmean(pos)-statistics.fmean(neg):+.4f}")
        if a_val < a.min_auc:
            failures.append(f"AUC {a_val:.3f} < {a.min_auc} — 보상이 목표를 구분하지 못함")
    else:
        failures.append("준수/위반 한쪽이 없어 분리도를 못 잰다")

    # 2) 축퇴
    print("\n[2] 축퇴 검사 (학습 신호 존재 여부)")
    allv = pos + neg
    sd = statistics.pstdev(allv) if len(allv) > 1 else 0.0
    print(f"    표준편차        {sd:.4f}   (기준 ≥{MIN_STDEV})")
    print(f"    범위            {min(allv):.4f} ~ {max(allv):.4f}")
    if sd < MIN_STDEV:
        failures.append(f"보상 분산 {sd:.4f} 이 너무 작다 — GRPO 어드밴티지가 0 에 수렴")

    # 3) 보상 해킹
    print("\n[3] 보상 해킹 취약성 (낮을수록 안전)")
    for name, resp, g in _hack_probes(items):
        v = compute_vcr("q", resp, grounding=(g,), nli_fn=nli, verifier=verifier).vcr
        flag = "  ⚠️ 위험" if v > MAX_HACK_SCORE else ""
        print(f"    {name:22s} VCR={v:.4f}{flag}")
        if v > MAX_HACK_SCORE:
            failures.append(f"해킹 시도 '{name}' 가 {v:.3f} 획득 — 보상 설계 재검토")

    # 4) 구성요소 생존
    print("\n[4] VCR 구성요소 (상수면 사실상 무효)")
    for comp in ("source_exist", "source_match", "alcoa_score", "halluc"):
        vals = [getattr(b, comp) for _, b in breakdowns]
        csd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        state = "상수(무효)" if csd < 1e-6 else f"σ={csd:.4f}"
        print(f"    {comp:14s} mean={statistics.fmean(vals):.4f}  {state}")
        if csd < 1e-6:
            failures.append(f"구성요소 {comp} 가 상수 — 가중치가 낭비됨")

    print("\n" + "=" * 64)
    if failures:
        print(" ❌ 사전 점검 실패 — GRPO 학습을 시작하지 말 것")
        for f in failures:
            print(f"    · {f}")
        print("\n RL 은 잘못된 보상도 조용히 최적화한다. 여기서 잡는 편이 GPU 시간보다 싸다.")
        sys.exit(1)
    print(" ✅ 사전 점검 통과 — GRPO 학습 진행 가능")
    print("=" * 64)


if __name__ == "__main__":
    main()
