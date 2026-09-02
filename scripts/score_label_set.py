#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""사람이 채운 라벨셋으로 분류기 정확도를 낸다.

이 스크립트가 돌아야 비로소 지금까지의 모든 분류 수치를 방어할 수 있다.
그전까지 46.5%·22.7%·11.4% 는 정확도가 아니라 분류기의 출력일 뿐이다.

**전체 정확도를 그냥 평균내면 안 된다.** 라벨셋은 버킷별 층화 추출이라
문헌·인용(실제 2.2%)이 표본의 10%를 차지한다. 실제 비율로 가중해야
코퍼스 전체의 정확도가 된다.

원문은 출력하지 않는다 — 혼동행렬과 지표만 낸다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

# 코퍼스 실제 비율(표본 60,000행 판정). 가중평균에 쓴다.
CORPUS_PCT = {
    "experimental": 0.3319, "analysis": 0.0964, "literature": 0.0221,
    "planning": 0.0150, "code_meta": 0.0709, "code_source": 0.0415,
    "code_commit": 0.0244, "code_docs": 0.0091, "fragment": 0.3836,
    "admin": 0.0051,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label_csv")
    ap.add_argument("--col", default="label",
                    help="판정 칸. label=사람 검수, model_label=모델 2차 판정")
    ap.add_argument("--macro", action="store_true",
                    help="연구/GitHub/기타 3대분류로 접어서 채점")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    # 배너는 **헤더 앞부분만** 걷어낸다. 전체 줄에서 `#` 시작 줄을 거르면
    # 인용 필드 안의 본문 줄까지 지워져 CSV 가 깨진다(실제로 깨졌다).
    lines = Path(a.label_csv).read_text(encoding="utf-8-sig").splitlines(True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    rows = list(csv.DictReader(lines[i:]))

    labeled = [r for r in rows if (r.get(a.col) or "").strip()]
    if not labeled:
        raise SystemExit(f"{a.col} 칸이 비어 있다. 채운 뒤 다시 실행하라.")
    if a.col != "label":
        print(f"  [주의] `{a.col}` 기준 채점이다. 사람 검수(label)가 아니므로")
        print( "         정확도가 아니라 **일치도**로만 읽어야 한다.")

    RESEARCH = {"experimental", "analysis", "literature", "planning"}
    GITHUB = {"code_source", "code_commit", "code_meta", "code_docs"}

    def fold(b):
        if not a.macro:
            return b
        return "연구내용" if b in RESEARCH else "GitHub" if b in GITHUB else "기타"

    conf: dict[tuple[str, str], int] = Counter()
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # [맞음, 전체]
    dom_ok = dom_n = 0

    for r in labeled:
        pred = r["pred_bucket"].strip()
        gold = r[a.col].strip()
        gold = pred if gold.lower() == "ok" else gold
        pred, gold = fold(pred), fold(gold)
        conf[(gold, pred)] += 1
        per[gold][1] += 1
        if gold == pred:
            per[gold][0] += 1
        d = (r.get("domain_label") or "").strip()
        if d:
            dom_n += 1
            if d.lower() == "ok" or d == r.get("pred_domain", "").strip():
                dom_ok += 1

    n = len(labeled)
    raw = sum(v[0] for v in per.values()) / n

    # 실제 코퍼스 비율로 가중 — 층화 추출 보정
    wsum = num = 0.0
    for b, (ok, tot) in per.items():
        w = (sum(v for k, v in CORPUS_PCT.items() if fold(k) == b)
             if a.macro else CORPUS_PCT.get(b))
        if w and tot:
            num += w * (ok / tot)
            wsum += w
    weighted = num / wsum if wsum else 0.0

    print("=" * 68)
    label_kind = "사람 검수" if a.col == "label" else "모델 2차 판정"
    scope = "3대분류" if a.macro else "10개 버킷"
    print(f" 분류기 일치도 — {label_kind} {n}건 / 전체 {len(rows)}건 · {scope}")
    print("=" * 68)
    print(f"\n  표본 정확도(비가중)   {raw*100:5.1f}%   <- 층화 추출이라 그대로 쓰면 안 된다")
    print(f"  코퍼스 가중 정확도    {weighted*100:5.1f}%   <- 이게 방어 가능한 수치다")
    if dom_n:
        print(f"  분야 정확도           {dom_ok/dom_n*100:5.1f}%  ({dom_n}건 판정)")

    print("\n[버킷별] 재현율 = 그 버킷의 정답 중 분류기가 맞힌 비율")
    for b in sorted(per, key=lambda k: -per[k][1]):
        ok, tot = per[b]
        pred_tot = sum(c for (g, p), c in conf.items() if p == b)
        prec = ok / pred_tot if pred_tot else 0.0
        print(f"  {b:14s} 재현율 {ok/tot*100:5.1f}% ({ok:>3}/{tot:<3})"
              f"  정밀도 {prec*100:5.1f}%  실제비율 {sum(v for k,v in CORPUS_PCT.items() if fold(k)==b)*100:4.1f}%")

    print("\n[주요 혼동] 정답 → 예측 (5건 이상)")
    for (g, p), c in sorted(conf.items(), key=lambda kv: -kv[1]):
        if g != p and c >= 5:
            print(f"  {g:14s} → {p:14s} {c:>3}건")

    out = {
        "labeled": n, "total": len(rows),
        "raw_accuracy": round(raw, 4), "weighted_accuracy": round(weighted, 4),
        "domain_accuracy": round(dom_ok / dom_n, 4) if dom_n else None,
        "per_bucket": {b: {"recall": round(ok / tot, 4), "n": tot}
                       for b, (ok, tot) in per.items()},
        "confusion": {f"{g}->{p}": c for (g, p), c in conf.items()},
    }
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"\n-> {a.json}")

    print("\n" + "=" * 68)
    if weighted >= 0.80:
        print(" 판정: 분류기 수치를 방어할 수 있다. 근거-주장 쌍 생성으로 진행.")
    elif weighted >= 0.60:
        print(" 판정: 경계선. 혼동이 큰 버킷을 고치고 재측정하라.")
    else:
        print(" 판정: 분류기를 신뢰할 수 없다. 지금까지의 모든 비율을 철회해야 한다.")
    print("=" * 68)


if __name__ == "__main__":
    main()
