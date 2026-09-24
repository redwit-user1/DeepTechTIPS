#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""'조각(fragment)' 버킷 재검토 — 정말 버릴 것인가.

내용 중심 판정에서 60% 가 fragment 로 분류됐는데 **중앙값이 533자**였다.
짧아서가 아니라 키워드 임계값(신호 3개 이상)을 못 넘어 밀린 것이므로,
정말 무가치인지 특성으로 확인한다. 원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from triage_ocr_content import classify_content

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")
_NUM_ANY = re.compile(r"\d+(?:[.,]\d+)?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=40000)
    a = ap.parse_args()

    n_frag = 0
    sig_hist: Counter[int] = Counter()      # 신호 총합 분포
    len_buckets: Counter[str] = Counter()
    ko_buckets: Counter[str] = Counter()
    num_rich = 0
    line_short = 0
    total_signals: Counter[str] = Counter()

    for row in iter_rows(Path(a.csv_path), limit=a.rows):
        b, sig = classify_content(row.text)
        if b != "fragment":
            continue
        n_frag += 1
        t = row.text.strip()

        s = sum(v for k, v in sig.items() if k in
                ("num_unit", "exp", "analysis", "lit", "plan"))
        sig_hist[min(s, 6)] += 1
        for k in ("num_unit", "exp", "analysis", "lit", "plan"):
            if sig[k]:
                total_signals[k] += 1

        L = len(t)
        len_buckets["<80자" if L < 80 else
                    "80~300" if L < 300 else
                    "300~1000" if L < 1000 else
                    "1000~3000" if L < 3000 else "3000+"] += 1

        h, la = len(_HANGUL.findall(t)), len(_LATIN.findall(t))
        r = h / max(1, h + la)
        ko_buckets["한글중심(>60%)" if r > 0.6 else
                   "혼합" if r > 0.2 else "영문중심(<20%)"] += 1

        if len(_NUM_ANY.findall(t)) >= 10:
            num_rich += 1
        lines = t.splitlines() or [t]
        if sum(1 for l in lines if len(l.strip()) <= 15) / len(lines) > 0.7:
            line_short += 1

    print("=" * 62)
    print(f" fragment 버킷 재검토 — {n_frag:,}건")
    print("=" * 62)

    print("\n[길이 분포] — '짧아서' 밀린 것인가")
    for k in ("<80자", "80~300", "300~1000", "1000~3000", "3000+"):
        c = len_buckets[k]
        print(f"  {k:12s} {c:>7,} ({c/max(1,n_frag)*100:5.1f}%) {'█'*int(c/max(1,n_frag)*30)}")

    print("\n[연구 신호 개수] — 0개면 정말 내용 없음, 1~2개면 임계값 미달")
    for s in range(7):
        c = sig_hist[s]
        tag = "신호 없음" if s == 0 else ("임계값 미달" if s < 3 else "임계값 충족*")
        print(f"  {s}개{'+' if s == 6 else ' '} {c:>7,} ({c/max(1,n_frag)*100:5.1f}%)  {tag}")
    print("  * 3개 이상인데 fragment 인 것은 신호가 여러 종류에 흩어진 경우")

    print("\n[신호 종류별 출현]")
    for k, c in total_signals.most_common():
        print(f"  {k:10s} {c:>7,} ({c/max(1,n_frag)*100:5.1f}%)")

    print("\n[언어]")
    for k, c in ko_buckets.most_common():
        print(f"  {k:16s} {c:>7,} ({c/max(1,n_frag)*100:5.1f}%)")

    print(f"\n[형태] 숫자 10개 이상 {num_rich:,} ({num_rich/max(1,n_frag)*100:.1f}%) — 표·측정값 가능성")
    print(f"       짧은 줄 위주  {line_short:,} ({line_short/max(1,n_frag)*100:.1f}%) — 표·목차 가능성")

    zero = sig_hist[0]
    print("\n" + "=" * 62)
    print(f" 판정: 신호 0개는 {zero/max(1,n_frag)*100:.1f}% 뿐이다.")
    if zero / max(1, n_frag) < 0.5:
        print(" → 대부분은 '내용이 없어서'가 아니라 **분류기가 못 잡은 것**이다.")
        print("   키워드 임계값 방식의 한계이며, 이 구간을 버리면 안 된다.")
    print("=" * 62)


if __name__ == "__main__":
    main()
