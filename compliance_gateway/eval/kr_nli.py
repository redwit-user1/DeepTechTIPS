"""한국어 NLI 평가셋(연구노트 원문, 의미 판단 라벨) 채점.

입력 jsonl: {premise, hypothesis, label: entail|neutral|contradict, kind}
  kind = orig(원 주장) / paraphrase(같은 뜻 다른 말) / altered(뜻을 바꾼 주장) / other_doc(다른 문서 근거)

지표: entail vs 나머지의 AUC, 그리고 kind 별 평균 점수. 표면 누설 상한도 함께 출력해
"이 셋은 표면으로 풀리지 않는다"를 같이 보인다.

    python -m compliance_gateway.eval.kr_nli data/labeling/kr_nli_eval.jsonl --nli <모델>
    python -m compliance_gateway.eval.kr_nli data/labeling/kr_nli_eval.jsonl --llm Qwen/Qwen3.8-27B
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from compliance_gateway.eval.benchmark import auc


def score_all(rows: list[dict], a) -> tuple[list[float], str]:
    pairs = [(r["premise"], r["hypothesis"]) for r in rows]
    if a.llm:
        from compliance_gateway.nli.llm_judge import LLMJudgeNLI
        return LLMJudgeNLI(a.llm, device=a.device).batch(pairs), f"llm-judge({a.llm})"
    from compliance_gateway.eval.nli_select import select_nli
    fn, name = select_nli(a.nli, None, None, a.device)
    return [fn(p, h) for p, h in pairs], name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--nli", default=None, help="트랜스포머 NLI 경로/ID (없으면 통계 v0.5)")
    ap.add_argument("--llm", default=None, help="LLM 판정기 모델 ID")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump", default=None, help="행별 점수를 jsonl 로 저장(캐스케이드 분석용)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.path, encoding="utf-8")]
    from compliance_gateway.data.ocr.claim_evidence import surface_leak
    leak = surface_leak([{"claim": r["hypothesis"], "evidence": r["premise"], "label": r["label"]} for r in rows])
    scores, name = score_all(rows, a)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for r, sc in zip(rows, scores):
                print(json.dumps({"id": r["id"], "kind": r["kind"], "label": r["label"], "score": sc}), file=f)

    pos = [s for s, r in zip(scores, rows) if r["label"] == "entail"]
    neg = [s for s, r in zip(scores, rows) if r["label"] != "entail"]
    by = defaultdict(list)
    for s, r in zip(scores, rows):
        by[(r["kind"], r["label"])].append(s)

    print(f"백엔드: {name}   n={len(rows)} (entail {len(pos)} / 그 외 {len(neg)})   표면 누설 상한 {leak:.1%}")
    print(f"AUC(entail vs 나머지) = {auc(pos, neg):.3f}")
    for lab in ("contradict", "neutral"):
        n = [s for s, r in zip(scores, rows) if r["label"] == lab]
        if n:
            print(f"  AUC(entail vs {lab:10s}) = {auc(pos, n):.3f}  (n={len(n)})")
    # KPI 운영점: 위반(=entail 아님) 탐지. 정상(entail) 통과율 ≥ 90% 를 지키는 최대 F1
    best = None
    for th in sorted(set(round(s, 3) for s in scores)):
        tp = sum(1 for s in neg if s < th); fp = sum(1 for s in pos if s < th)
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / len(neg) if neg else 0.0
        pas = 1 - fp / len(pos) if pos else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if pas >= 0.9 and (best is None or f1 > best[0]):
            best = (f1, th, p, r, pas)
    if best:
        print(f"운영점(정상통과≥90%): θ={best[1]:.3f} 위반탐지 P {best[2]:.1%} R {best[3]:.1%} F1 {best[0]:.1%} PASS {best[4]:.1%}")
    print("kind/label 별 평균 점수:")
    for k in sorted(by):
        v = by[k]
        print(f"  {k[0]:11s} {k[1]:10s} n={len(v):3d}  mean={sum(v)/len(v):.3f}")


if __name__ == "__main__":
    main()
