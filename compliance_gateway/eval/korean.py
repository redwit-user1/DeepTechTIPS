"""국내 R&D 한국어 평가 — ALCOA+ 속성별 위반 탐지.

⚠️ **성능 해석 주의 (매우 중요)**

본 평가셋은 실제 국내 연구과제 메타데이터를 한국어로 렌더링하고 **규칙 기반으로 변조**한
합성셋이다. 우리가 만든 변조를 우리가 찾는 구조이므로 수치가 낙관적으로 나온다.
bioRxiv 합성셋(F1 98.2%)과 같은 성격이며, **외부 실데이터(SciFact F1 39.3%)와 직접
비교해서는 안 된다.**

| 평가셋 | 성격 | 용도 |
|---|---|---|
| 합성 KR (본 모듈) | 템플릿+규칙변조 | 한국어 처리 회귀 테스트, ALCOA+ 속성 커버리지 |
| 합성 EN (bioRxiv) | 실논문+규칙변조 | 서지 검증 회귀 테스트 |
| **외부 EN (SciFact)** | **전문가 주석 실데이터** | **KPI 근거** |
| 외부 KR | **미확보** | ScienceON/KCI 수집 필요 (egress 차단) |

→ 한국어 KPI 를 정직하게 주장하려면 **외부 한국어 평가셋이 필요하다**. 현재 미확보.

실행:
  python -m compliance_gateway.eval.korean --sweep
"""

from __future__ import annotations

import argparse

from compliance_gateway.data.korean.build_kr import build_items, build_registry
from compliance_gateway.data.korean.sources import load_kr_seed
from compliance_gateway.eval.kpi import evaluate
from compliance_gateway.nli.statistical import StatisticalNLI
from compliance_gateway.pipeline import ComplianceGateway
from compliance_gateway.verify import CitationVerifier

# 한국어 템플릿은 근거-주장 어휘 중복이 커 VCR 이 전반적으로 높다.
# → 영어(θ≈0.55)와 다른 운영점이 필요하다(VCR v2 도메인별 가중치 최적화의 근거).
DEFAULT_KR_THRESHOLD = 0.94

ALCOA_KO = {
    "Attributable": "귀속가능",
    "Accurate": "정확",
    "Contemporaneous": "동시적",
    "Original": "원본",
}


def build_eval(real: bool = False) -> tuple[list[dict], object, list[dict]]:
    """real=True 면 **실데이터 평가셋**(변조 없음, 출처 기반 라벨)을 쓴다."""
    if real:
        from compliance_gateway.data.korean.real_eval import (
            build as build_real, build_registry as reg_real, load_protocols,
        )
        protocols = load_protocols()
        evals, _ = build_real(protocols)
        items = [{"query": e["query"], "response": e["response"],
                  "grounding": e["grounding"], "label": e["label"]} for e in evals]
        return items, reg_real(protocols), evals

    records = load_kr_seed()
    evals, pairs = build_items(records)
    items = [
        {"query": e["query"], "response": e["response"],
         "grounding": e["grounding"], "label": e["label"]}
        for e in evals
    ]
    return items, build_registry(records), evals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=DEFAULT_KR_THRESHOLD)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--nli", default=None, help="트랜스포머 NLI 경로(선택)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--real", action="store_true",
                    help="실데이터 평가셋 사용(변조 없음 — 성능 주장의 진짜 근거)")
    a = ap.parse_args()

    items, registry, evals = build_eval(real=a.real)
    if a.real:
        print("데이터셋: 실데이터 KR (교차 귀속 오류, 원문 변조 없음)")
    n_ok = sum(1 for i in items if i["label"] == "compliant")
    print(f"국내 R&D 한국어 평가셋: {len(items)}건 (compliant {n_ok} / 위반 {len(items)-n_ok})")

    if a.nli:
        from compliance_gateway.nli.transformer import TransformerNLI
        nli_fn = TransformerNLI(model_name=a.nli, device=a.device)
        print(f"NLI 백엔드: transformer({a.nli})")
    else:
        nli_fn = StatisticalNLI()
        print("NLI 백엔드: statistical-v0.5")

    gw = ComplianceGateway(vcr_threshold=a.threshold, nli_fn=nli_fn,
                           verifier=CitationVerifier([registry]))
    m = evaluate(gw, items)
    print(f"\nθ={a.threshold}")
    print(f"  위반탐지 Precision {m['violation_precision']*100:6.1f}%")
    print(f"  위반탐지 Recall    {m['violation_recall']*100:6.1f}%")
    print(f"  위반탐지 F1        {m['violation_f1']*100:6.1f}%")
    print(f"  compliant PASS     {m['compliant_pass_rate']*100:6.1f}%")

    # ALCOA+ 속성별 탐지율 — 국내 감사 대응 관점의 핵심 지표
    by_attr: dict[str, list[float]] = {}
    for e in evals:
        if not e["alcoa_violation"]:
            continue
        rate = m["per_type_detection"].get(e["label"])
        if rate is not None:
            by_attr.setdefault(e["alcoa_violation"], []).append(rate)
    print("\nALCOA+ 속성별 위반 탐지율:")
    for attr, rates in sorted(by_attr.items()):
        print(f"  {ALCOA_KO.get(attr, attr):8s}({attr:16s}) {sum(rates)/len(rates)*100:6.1f}%")

    if a.sweep:
        print("\n=== 임계값 스윕 ===")
        print(f"{'θ':>6s} {'P':>8s} {'R':>8s} {'F1':>8s} {'PASS':>8s}")
        for th in [x / 100 for x in range(70, 100, 2)]:
            g = ComplianceGateway(vcr_threshold=th, nli_fn=nli_fn,
                                  verifier=CitationVerifier([registry]))
            mm = evaluate(g, items)
            flag = "" if mm["compliant_pass_rate"] >= 0.8 else "  (사용불가)"
            print(f"{th:6.2f} {mm['violation_precision']*100:7.1f}% {mm['violation_recall']*100:7.1f}%"
                  f" {mm['violation_f1']*100:7.1f}% {mm['compliant_pass_rate']*100:7.1f}%{flag}")

    # 임계값 무관 변별력(AUC) — 임계값 착시를 배제한 진짜 지표
    from compliance_gateway.eval.benchmark import auc
    from compliance_gateway.vcr.reward import compute_vcr

    v = CitationVerifier([registry])
    pos, neg = [], []
    for e in items:
        sc = compute_vcr(e["query"], e["response"], grounding=(e["grounding"],),
                         nli_fn=nli_fn, verifier=v).vcr
        (pos if e["label"] == "compliant" else neg).append(sc)
    if pos and neg:
        print(f"\nAUC(임계값 무관 변별력) = {auc(pos, neg):.3f}"
              f"   분리도 = {sum(pos)/len(pos) - sum(neg)/len(neg):+.4f}")

    print("\n" + "=" * 70)
    if a.real:
        print("데이터: 국내 연구기관 실제 프로토콜 원문. 우리가 변조하지 않았고")
        print("        음성은 '다른 실제 과제의 원문을 잘못 귀속'시킨 것이다.")
        print("→ 한국어 성능 주장은 이 수치를 근거로 해야 한다(합성셋 아님).")
    else:
        print("⚠️  이 수치는 규칙 기반 변조 합성셋 기준이다. 낙관 편향이 있으며")
        print("    실데이터(--real)와 직접 비교할 수 없다. 회귀 테스트 용도로만 쓸 것.")
    print("=" * 70)


if __name__ == "__main__":
    main()
