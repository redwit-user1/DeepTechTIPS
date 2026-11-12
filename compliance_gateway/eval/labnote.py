"""연구노트 기록 무결성 평가.

⚠️ **가상 데이터 기반** — 규칙 생성물이므로 KPI 근거가 아니다. 회귀 테스트용.

VCR(출처 기반 생성물 검증)과 달리 연구노트는 1차 기록이라 인용이 없다.
실제로 VCR 로 평가하면 정상 노트조차 전부 차단된다(VCR 0.42).
따라서 `integrity/` 의 기록 무결성 검사로 평가한다.

실행:
  python -m compliance_gateway.eval.labnote
  python -m compliance_gateway.eval.labnote --compare-vcr   # VCR 과 비교
"""

from __future__ import annotations

import argparse

from compliance_gateway.data.labnote.generate import VIOLATIONS, build_dataset
from compliance_gateway.eval.benchmark import auc
from compliance_gateway.eval.kpi import prf
from compliance_gateway.integrity import check_lab_note

# 무결성 점수 임계값 — 1.0 미만이면 지침 위반이 하나라도 있다는 뜻
INTEGRITY_THRESHOLD = 0.999


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=INTEGRITY_THRESHOLD)
    ap.add_argument("--compare-vcr", action="store_true",
                    help="VCR(출처 기반)로 평가하면 어떻게 되는지 대조")
    a = ap.parse_args()

    _, items, _ = build_dataset(a.notes)
    n_ok = sum(1 for i in items if i["label"] == "compliant")
    print(f"연구노트 평가셋: {len(items)}건 (준수 {n_ok} / 위반 {len(items)-n_ok})")
    print("데이터: 가상(SYNTHETIC) — 실제 연구기록 아님\n")

    y_true, y_pred, pos, neg = [], [], [], []
    per_type: dict[str, list[bool]] = {}
    for it in items:
        rep = check_lab_note(it["response"], it["grounding"])
        flagged = rep.overall < a.threshold
        is_violation = it["label"] != "compliant"
        y_true.append(is_violation); y_pred.append(flagged)
        (neg if is_violation else pos).append(rep.overall)
        if is_violation:
            per_type.setdefault(it["label"], []).append(flagged)

    prec, rec, f1 = prf(y_true, y_pred)
    pass_rate = sum(1 for t, p in zip(y_true, y_pred) if not t and not p) / max(1, y_true.count(False))
    print("=== 기록 무결성 검사 (integrity/) ===")
    print(f"  위반탐지 Precision {prec*100:6.1f}%")
    print(f"  위반탐지 Recall    {rec*100:6.1f}%")
    print(f"  위반탐지 F1        {f1*100:6.1f}%")
    print(f"  준수노트 PASS      {pass_rate*100:6.1f}%")
    print(f"  AUC                {auc(pos, neg):6.3f}")

    print("\n위반 유형별 탐지율 (ALCOA+ 속성):")
    for kind, hits in sorted(per_type.items()):
        attr = VIOLATIONS[kind][0]
        print(f"  {kind:18s} {sum(hits)/len(hits)*100:5.1f}%  → {attr}")

    if a.compare_vcr:
        from compliance_gateway.nli.statistical import StatisticalNLI
        from compliance_gateway.vcr.reward import compute_vcr
        nli = StatisticalNLI()
        vp, vn = [], []
        for it in items:
            v = compute_vcr(it["query"], it["response"],
                            grounding=(it["grounding"],), nli_fn=nli).vcr
            (vn if it["label"] != "compliant" else vp).append(v)
        print("\n=== 대조: VCR(출처 기반)로 평가하면 ===")
        print(f"  AUC {auc(vp, vn):.3f}   준수 평균 {sum(vp)/len(vp):.4f}   위반 평균 {sum(vn)/len(vn):.4f}")
        print("  → 연구노트는 인용이 없어 SourceExist/SourceMatch 가 구조적으로 낮다.")
        print("    정상 노트조차 통과하지 못하므로 VCR 은 이 과제에 부적합하다.")

    print("\n" + "=" * 64)
    print("⚠️  가상 데이터 기준이다. 규칙으로 만든 위반을 규칙으로 찾으므로")
    print("    낙관 편향이 있다. 실제 연구노트 확보 시 재측정이 필요하다.")
    print("=" * 64)


if __name__ == "__main__":
    main()
