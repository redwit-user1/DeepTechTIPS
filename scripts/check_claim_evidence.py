#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""문서 안에 '근거'와 '주장'이 같이 있는가.

## 왜 이걸 재는가

VCR 의 `SourceMatch` 를 두고 "인용된 논문 본문이 코퍼스에 없으니 원리적으로
불가능"이라고 했다. **외부 인용에 대해서는 맞지만 그게 전부가 아니다.**

연구 기록은 그 자체로 근거와 주장을 같이 담는다 —
같은 문서 안에 측정값(실험 기록)이 있고 그로부터 끌어낸 해석(분석·해석)이 있다.
`(주장, 근거, 라벨)` 삼중항을 **사람 라벨 없이** 만들 수 있다는 뜻이다:

- 양성: 주장 + 같은 문서의 측정값
- 음성(쉬움): 주장 + 다른 문서의 측정값
- 음성(어려움): 주장 + 숫자를 교란한 측정값  ← VCR 이 실제로 잡아야 할 실패 모드

이게 성립하려면 **한 문서 안에 두 종류가 같이 있어야 한다.** 그 비율을 잰다.
안 나오면 이 계획은 폐기다.

NoteId 해시로 문서를 솎는다(행이 아니라 문서를 솎아야 그룹이 안 깨진다).
원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from triage_ocr_content import classify_content

BUCKETS = ["experimental", "analysis", "literature", "planning"]
BIT = {b: 1 << i for i, b in enumerate(BUCKETS)}
EXP, ANA = BIT["experimental"], BIT["analysis"]
CNT_SHIFT = 12
MASK = (1 << CNT_SHIFT) - 1


def nid(s: str) -> int:
    return int.from_bytes(hashlib.blake2s(s.encode(), digest_size=7).digest(), "big")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--doc-stride", type=int, default=32)
    ap.add_argument("--json", default="ocr_claim_evidence.json")
    a = ap.parse_args()

    acc: dict[int, int] = {}
    exp_rows: dict[int, int] = {}
    ana_rows: dict[int, int] = {}
    n = kept = 0

    for row in iter_rows(Path(a.csv_path)):
        n += 1
        key = nid(row.note_id)
        if key % a.doc_stride:
            continue
        kept += 1
        b, _ = classify_content(row.text)
        bit = BIT.get(b, 0)
        prev = acc.get(key, 0)
        acc[key] = (((prev >> CNT_SHIFT) + 1) << CNT_SHIFT) | ((prev & MASK) | bit)
        if bit == EXP:
            exp_rows[key] = exp_rows.get(key, 0) + 1
        elif bit == ANA:
            ana_rows[key] = ana_rows.get(key, 0) + 1

    docs = len(acc)
    both = both_rows = 0
    only_exp = only_ana = neither = 0
    pair_capacity = 0        # 문서별 (주장 x 근거) 조합 수의 합
    pages_of_both: Counter[str] = Counter()

    for key, packed in acc.items():
        cnt = packed >> CNT_SHIFT
        m = packed & MASK
        he, ha = bool(m & EXP), bool(m & ANA)
        if he and ha:
            both += 1
            both_rows += cnt
            pair_capacity += exp_rows.get(key, 0) * ana_rows.get(key, 0)
            pages_of_both["1" if cnt == 1 else "2-4" if cnt <= 4 else
                          "5-9" if cnt <= 9 else "10-29" if cnt <= 29 else "30+"] += 1
        elif he:
            only_exp += 1
        elif ha:
            only_ana += 1
        else:
            neither += 1

    SC = 5_654_358 / max(1, kept)
    print("=" * 70)
    print(f" 문서 내 근거-주장 동거율")
    print(f" 전수 5,654,358행 중 표본 {kept:,}행(문서 1/{a.doc_stride}) → 문서 {docs:,}개")
    print("=" * 70)

    print("\n[문서 구성]")
    for lbl, c in (("근거+주장 둘 다", both), ("실험 기록만", only_exp),
                   ("분석·해석만", only_ana), ("둘 다 없음", neither)):
        print(f"  {lbl:16s} {c:>8,} ({c/max(1,docs)*100:5.1f}%)"
              f" {'#'*int(c/max(1,docs)*32)}")

    print(f"\n[근거+주장 문서의 규모]")
    print(f"  문서            {both:,}개  → 전체 약 {int(both*a.doc_stride):,}개")
    print(f"  행              {both_rows:,}  → 전체 약 {int(both_rows*SC):,}행")
    print(f"  주장x근거 조합   {pair_capacity:,}  → 전체 약 {int(pair_capacity*a.doc_stride):,}쌍")

    print(f"\n[근거+주장 문서의 페이지 수]")
    for k in ("1", "2-4", "5-9", "10-29", "30+"):
        c = pages_of_both[k]
        print(f"  {k:>6s}페이지 {c:>8,} ({c/max(1,both)*100:5.1f}%)")

    print("\n" + "=" * 70)
    if both / max(1, docs) >= 0.05:
        print(f" 판정: 성립한다. 사람 라벨 없이 약 {int(pair_capacity*a.doc_stride):,}쌍의")
        print(f"        (주장, 근거) 학습 데이터를 만들 수 있다.")
    else:
        print(f" 판정: 동거율 {both/max(1,docs)*100:.1f}% 로 너무 낮다. 계획 폐기.")
    print("=" * 70)

    Path(a.json).write_text(json.dumps({
        "rows_total": n, "rows_sampled": kept, "doc_stride": a.doc_stride,
        "documents": docs,
        "both": both, "only_experimental": only_exp, "only_analysis": only_ana,
        "neither": neither,
        "both_pct": round(both / max(1, docs), 4),
        "both_docs_estimated": int(both * a.doc_stride),
        "both_rows_estimated": int(both_rows * SC),
        "pair_capacity_sampled": pair_capacity,
        "pair_capacity_estimated": int(pair_capacity * a.doc_stride),
        "pages_of_both": dict(pages_of_both),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
