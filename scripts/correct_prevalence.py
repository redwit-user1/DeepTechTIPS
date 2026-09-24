#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""혼동행렬 역산으로 실제 비율을 보정한다.

## 왜 가능한가

두 측정이 각각 다른 것을 알려준다.

1. `retriage_full.py` — 전 구간에서 **분류기가 예측한** 분포 `P(pred)`.
   앞부분 표본과 최대 괴리 0.2p 로, 구간 편향이 없음이 확인됐다.
2. 라벨셋 판정 — 각 예측 버킷 안에서 **실제로 무엇이었는가** `P(gold | pred)`.
   라벨셋이 **예측 버킷 기준 층화 추출**이므로 이 조건부 확률은 편향 없이 추정된다.

따라서

    P(gold=g) = Σ_p P(gold=g | pred=p) · P(pred=p)

분류기가 일관되게 틀리면 그 틀림 자체를 보정할 수 있다.
노이즈가 아니라 계통 오차이기 때문이다.

## 한계

`gold` 이 사람 검수가 아니라 **모델 2차 판정**이다. 보정값은 그 판정을
정답으로 가정한 결과이며, 사람 검수로 다시 세워야 최종 확정된다.
표본이 작은 버킷은 조건부 확률의 분산이 크므로 신뢰구간을 함께 낸다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RESEARCH = {"experimental", "analysis", "literature", "planning"}
GITHUB = {"code_source", "code_commit", "code_meta", "code_docs"}


def macro(b: str) -> str:
    return "연구내용" if b in RESEARCH else "GitHub" if b in GITHUB else "기타"


def read_rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8-sig").splitlines(True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return list(csv.DictReader(lines[i:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label_csv")
    ap.add_argument("retriage_json")
    ap.add_argument("--col", default="model_label")
    ap.add_argument("--json", default="ocr_corrected.json")
    a = ap.parse_args()

    rows = [r for r in read_rows(Path(a.label_csv)) if (r.get(a.col) or "").strip()]
    rt = json.loads(Path(a.retriage_json).read_text(encoding="utf-8"))
    p_pred: dict[str, float] = rt["bucket_pct"]
    TOT = 5_654_358

    # P(gold | pred) — 예측 버킷별 층화 추출이므로 편향 없음
    cond: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        pred = r["pred_bucket"].strip()
        gold = r[a.col].strip()
        gold = pred if gold.lower() == "ok" else gold
        cond[pred][gold] += 1

    def correct(fold):
        """fold 로 접은 라벨 체계에서 P(gold) 를 보정한다."""
        out: Counter[str] = Counter()
        var: Counter[str] = Counter()
        for pred, w in p_pred.items():
            c = cond.get(pred)
            if not c:                      # 판정 표본이 없는 버킷 → 예측을 그대로 신뢰
                out[fold(pred)] += w
                continue
            n = sum(c.values())
            folded: Counter[str] = Counter()
            for g, k in c.items():
                folded[fold(g)] += k
            for g, k in folded.items():
                p = k / n
                out[g] += w * p
                var[g] += (w ** 2) * p * (1 - p) / n     # 이항 분산
        return out, var, {p: sum(c.values()) for p, c in cond.items()}

    print("=" * 74)
    print(f" 혼동행렬 역산 보정 — 판정 {len(rows)}건 · 전 구간 예측 {rt['judged_rows']:,}행")
    print("=" * 74)
    print("  주의: gold 가 사람 검수가 아니라 모델 2차 판정이다.")
    print("        사람 검수로 다시 세워야 확정된다.\n")

    for name, fold in (("3대분류", macro), ("10개 버킷", lambda b: b)):
        out, var, ns = correct(fold)
        print(f"[{name}]  관측(분류기) → 보정  ±는 95% 구간")
        for g, v in out.most_common():
            obs = sum(w for p, w in p_pred.items() if fold(p) == g)
            ci = 1.96 * math.sqrt(var[g])
            d = (v - obs) * 100
            print(f"  {g:14s} {obs*100:6.2f}% → {v*100:6.2f}% (±{ci*100:.2f})"
                  f" {d:+6.2f}p   약 {int(v*TOT):>9,}행")
        print()

    out_m, var_m, ns = correct(macro)
    out_b, var_b, _ = correct(lambda b: b)

    print("[예측 버킷별 판정 표본 수] 표본이 적으면 보정도 불안정하다")
    for p, n in sorted(ns.items(), key=lambda kv: -kv[1]):
        top = cond[p].most_common(1)[0]
        print(f"  {p:14s} {n:>3}건   최다 실제값: {top[0]} {top[1]/n*100:.0f}%")

    Path(a.json).write_text(json.dumps({
        "labeled": len(rows), "label_col": a.col,
        "observed_pct": p_pred,
        "corrected_macro": {k: round(v, 4) for k, v in out_m.items()},
        "corrected_macro_ci95": {k: round(1.96 * math.sqrt(v), 4) for k, v in var_m.items()},
        "corrected_macro_rows": {k: int(v * TOT) for k, v in out_m.items()},
        "corrected_bucket": {k: round(v, 4) for k, v in out_b.items()},
        "corrected_bucket_rows": {k: int(v * TOT) for k, v in out_b.items()},
        "samples_per_pred": ns,
        "caveat": "gold 는 사람 검수가 아니라 모델 2차 판정이다",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
