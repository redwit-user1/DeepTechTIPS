"""KPI 측정 하니스 — 과제 목표(성능지표) 대비 현재 성능을 정량 측정.

"측정 없이는 목표 달성을 증명할 수 없다." 사업계획서 성능지표를 측정 가능한
메트릭으로 매핑하고, 동일 코드로 baseline(현재)과 학습 후(A100) 값을 비교한다.

실행:
  python -m compliance_gateway.eval.kpi                 # 합성 시드 기반 baseline
  python -m compliance_gateway.eval.kpi --eval data/synth/gateway_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from compliance_gateway.data.build_dpo import make_doi_resolver
from compliance_gateway.data.synth import build_examples, load_seed
from compliance_gateway.models import GatewayDecision, GatingMode, GenerationRequest
from compliance_gateway.nli.statistical import StatisticalNLI
from compliance_gateway.pipeline import ComplianceGateway


@dataclass
class KPI:
    name: str
    target: str
    value: float
    unit: str = "%"


def prf(y_true: list[bool], y_pred: list[bool]) -> tuple[float, float, float]:
    """positive=True 기준 precision, recall, F1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def load_eval_items(path: Optional[str]) -> list[dict]:
    """gateway_eval.jsonl 로드, 없으면 시드에서 즉석 생성."""
    if path and Path(path).exists():
        with Path(path).open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    _, evals = build_examples(load_seed(), max_claims=4)
    return [e.to_json() for e in evals]


def evaluate(gateway: ComplianceGateway, items: list[dict]) -> dict:
    """Gateway 로 각 아이템 평가 → KPI 원자료 산출."""
    y_true_violation: list[bool] = []   # 실제 위반 여부
    y_pred_violation: list[bool] = []   # Gateway 가 위반으로 판정(PASS 아님)
    per_type: dict[str, list[bool]] = {}
    compliant_pass = compliant_total = 0

    for it in items:
        req = GenerationRequest(
            query=it["query"], response=it["response"],
            grounding=(it["grounding"],), model_id="eval", mode=GatingMode.POST,
        )
        res = gateway.evaluate(req)
        flagged = res.decision != GatewayDecision.PASS
        is_violation = it["label"] != "compliant"

        y_true_violation.append(is_violation)
        y_pred_violation.append(flagged)
        if is_violation:
            per_type.setdefault(it["label"], []).append(flagged)
        else:
            compliant_total += 1
            compliant_pass += int(not flagged)

    prec, rec, f1 = prf(y_true_violation, y_pred_violation)
    return {
        "violation_precision": prec,
        "violation_recall": rec,
        "violation_f1": f1,
        "compliant_pass_rate": compliant_pass / compliant_total if compliant_total else 0.0,
        "per_type_detection": {k: sum(v) / len(v) for k, v in per_type.items()},
        "n_items": len(items),
    }


def to_kpis(m: dict) -> list[KPI]:
    return [
        KPI("규정위반 탐지 정밀도(Precision)", "90+", round(m["violation_precision"] * 100, 1)),
        KPI("규정위반 탐지 Recall", "-", round(m["violation_recall"] * 100, 1)),
        KPI("규정위반 탐지 F1", "-", round(m["violation_f1"] * 100, 1)),
        KPI("compliant 통과율(오탐 역지표)", "높을수록", round(m["compliant_pass_rate"] * 100, 1)),
    ]


def _print_report(m: dict, backend: str) -> None:
    print(f"=== KPI 측정 (NLI backend: {backend}, n={m['n_items']}) ===\n")
    print(f"{'KPI':32s} {'목표':>8s} {'현재':>8s}")
    print("-" * 52)
    for k in to_kpis(m):
        print(f"{k.name:32s} {k.target:>8s} {k.value:>7.1f}{k.unit}")
    print("\n환각 유형별 탐지율:")
    for t, r in sorted(m["per_type_detection"].items()):
        print(f"  {t:16s} {r*100:5.1f}%")
    print("\n* 현재는 통계 NLI(v0.5) baseline. A100에서 트랜스포머 NLI 주입 시 재측정 → KPI 개선 확인.")


def _print_comparison(rows: list[tuple[str, dict]]) -> None:
    """서지 검증 백엔드 간 KPI 비교표."""
    print("\n=== 백엔드 비교 ===")
    names = [n for n, _ in rows]
    print(f"{'지표':22s}" + "".join(f"{n:>18s}" for n in names))
    print("-" * (22 + 18 * len(names)))
    for key, label in [
        ("violation_precision", "위반탐지 Precision"),
        ("violation_recall", "위반탐지 Recall"),
        ("violation_f1", "위반탐지 F1"),
        ("compliant_pass_rate", "compliant PASS"),
    ]:
        print(f"{label:22s}" + "".join(f"{m[key]*100:17.1f}%" for _, m in rows))

    all_types = sorted({t for _, m in rows for t in m["per_type_detection"]})
    print("\n환각 유형별 탐지율:")
    print(f"{'유형':22s}" + "".join(f"{n:>18s}" for n in names))
    for t in all_types:
        print(f"{t:22s}" + "".join(
            f"{m['per_type_detection'].get(t, 0)*100:17.1f}%" for _, m in rows
        ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None, help="gateway_eval.jsonl 경로(없으면 시드 생성)")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--compare", action="store_true",
                    help="binary resolver vs 3-class verifier 비교")
    a = ap.parse_args()

    items = load_eval_items(a.eval)
    nli = StatisticalNLI()
    from compliance_gateway.verify import CitationVerifier, LocalRegistry

    verifier = CitationVerifier([LocalRegistry.from_seed()])
    gw = ComplianceGateway(vcr_threshold=a.threshold, nli_fn=nli, verifier=verifier)
    m = evaluate(gw, items)
    _print_report(m, backend="statistical-v0.5 + 3-class verifier")

    if a.compare:
        gw_bin = ComplianceGateway(
            vcr_threshold=a.threshold, nli_fn=nli,
            doi_resolver=make_doi_resolver({pp.doi for pp in load_seed()}),
        )
        _print_comparison([("binary", evaluate(gw_bin, items)), ("3-class", m)])


if __name__ == "__main__":
    main()
