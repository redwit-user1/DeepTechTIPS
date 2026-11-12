"""SourceMatch/NLI 벤치마크 — SciFact 기반.

각 NLI 백엔드가 SUPPORT 를 CONTRADICT 보다 높게 점수내는 능력을 측정한다.

지표:
  AUC            : SUPPORT > CONTRADICT 로 올바르게 순위매길 확률 (Mann-Whitney U)
  Acc@best       : 최적 임계값에서의 이분 정확도
  Acc@0.5        : 운영 임계값(0.5) 정확도
  mean(SUPPORT)  / mean(CONTRADICT) : 라벨별 평균 점수(분리도 직관)

실행:
  python -m compliance_gateway.eval.benchmark --split train
  python -m compliance_gateway.eval.benchmark --split dev --transformer  # HF/로컬 모델 가능 시
"""

from __future__ import annotations

import argparse
from typing import Callable

from compliance_gateway.eval.scifact import NLIExample, load_scifact
from compliance_gateway.nli.lexical import lexical_nli
from compliance_gateway.nli.statistical import StatisticalNLI

NLIFn = Callable[[str, str], float]


def auc(pos: list[float], neg: list[float]) -> float:
    """SUPPORT(pos) 점수가 CONTRADICT(neg)보다 클 확률. tie=0.5."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def best_threshold_accuracy(pos: list[float], neg: list[float]) -> tuple[float, float]:
    cand = sorted(set(pos + neg))
    best_acc, best_t = 0.0, 0.5
    total = len(pos) + len(neg)
    for t in cand:
        tp = sum(1 for p in pos if p >= t)
        tn = sum(1 for n in neg if n < t)
        acc = (tp + tn) / total
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_acc, best_t


def accuracy_at(pos: list[float], neg: list[float], t: float) -> float:
    total = len(pos) + len(neg)
    if total == 0:
        return float("nan")
    tp = sum(1 for p in pos if p >= t)
    tn = sum(1 for n in neg if n < t)
    return (tp + tn) / total


def score_backend(name: str, fn: NLIFn, examples: list[NLIExample]) -> dict:
    pos = [fn(e.evidence, e.claim) for e in examples if e.label == "SUPPORT"]
    neg = [fn(e.evidence, e.claim) for e in examples if e.label == "CONTRADICT"]
    acc_best, t_best = best_threshold_accuracy(pos, neg)
    return {
        "backend": name,
        "n_support": len(pos),
        "n_contradict": len(neg),
        "auc": round(auc(pos, neg), 4),
        "acc_best": round(acc_best, 4),
        "t_best": round(t_best, 4),
        "acc_0.5": round(accuracy_at(pos, neg, 0.5), 4),
        "mean_support": round(sum(pos) / len(pos), 4) if pos else float("nan"),
        "mean_contradict": round(sum(neg) / len(neg), 4) if neg else float("nan"),
    }


def _print_table(rows: list[dict]) -> None:
    cols = ["backend", "auc", "acc_best", "t_best", "acc_0.5", "mean_support", "mean_contradict"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--max", type=int, default=None, help="최대 예시 수")
    ap.add_argument("--transformer", action="store_true", help="트랜스포머 NLI 포함(HF/로컬 필요)")
    ap.add_argument("--model", default=None, help="트랜스포머 모델 ID/경로")
    args = ap.parse_args()

    examples = load_scifact(split=args.split, max_examples=args.max)
    print(f"SciFact[{args.split}]  examples={len(examples)} "
          f"(SUPPORT={sum(e.label=='SUPPORT' for e in examples)}, "
          f"CONTRADICT={sum(e.label=='CONTRADICT' for e in examples)})\n")

    stat = StatisticalNLI().fit([e.evidence for e in examples] + [e.claim for e in examples])

    rows = [
        score_backend("lexical (baseline)", lexical_nli, examples),
        score_backend("statistical (v0.5)", stat, examples),
    ]

    if args.transformer:
        from compliance_gateway.nli.transformer import load_default
        tnli = load_default(model_name=args.model)
        rows.append(score_backend("transformer", tnli, examples))

    _print_table(rows)


if __name__ == "__main__":
    main()
