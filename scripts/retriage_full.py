#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""버킷·분야 전 구간 재판정 — 앞부분 편향 제거.

## 왜 다시 하는가

지금까지의 버킷·분야 수치는 **앞부분 연속 60,000행**에서 나왔다.
문제는 표본 크기가 아니라(60,000이면 비율 추정에 충분하다) **연속 구간**이라는 점이다.
업로드 순서·계정·시기와 상관될 수 있어 코퍼스 전체를 대표하지 못한다.

그래서 파일 **전체를 순회**하되 N행마다 하나씩 판정한다. 표본 크기는 비슷해도
전 구간에 균등히 퍼지므로 구간 편향이 사라진다.

전수 판정(5,654,358행 전부)은 하지 않는다 — `classify_content` 가 행당 4,000자를
훑으므로 9시간 이상 걸리고, 얻는 것은 소수점 자릿수뿐이다.

## 비교 대상

기존 앞부분 표본 결과를 나란히 출력해 **얼마나 달라졌는지**를 바로 보게 한다.
차이가 크면 지금까지의 모든 비율을 갱신해야 한다.

원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from classify_domain import classify_domain
from triage_ocr_content import classify_content

# 앞부분 60,000행 기준 기존 수치 — 비교용
HEAD_BUCKET = {
    "experimental": 0.3319, "analysis": 0.0964, "literature": 0.0221,
    "planning": 0.0150, "code_meta": 0.0709, "code_source": 0.0415,
    "code_commit": 0.0244, "code_docs": 0.0091, "fragment": 0.3836,
    "admin": 0.0051,
}
HEAD_DOMAIN = {   # 연구 버킷 대비
    "미분류": 0.3700, "생명·보건의료": 0.2267, "정보·통신": 0.0993,
    "재료·소재": 0.0719, "화학·화공": 0.0609, "기계·제조": 0.0438,
    "에너지·환경": 0.0409, "전기·전자·반도체": 0.0353, "농림수산·식품": 0.0262,
    "건설·교통": 0.0150, "기초·자연과학": 0.0101,
}
RESEARCH = {"experimental", "analysis", "literature", "planning"}
CODE = {"code_source", "code_commit", "code_meta", "code_docs"}


def arrow(new: float, old: float) -> str:
    d = (new - old) * 100
    if abs(d) < 0.5:
        return "  ~   "
    return f" {d:+5.1f}p"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--json", default="ocr_retriage.json")
    a = ap.parse_args()

    n = m = 0
    bucket: Counter[str] = Counter()
    domain: Counter[str] = Counter()
    dom_by_bucket: dict[str, Counter] = {}
    blen: Counter[str] = Counter()

    for row in iter_rows(Path(a.csv_path)):
        n += 1
        if n % a.stride:
            continue
        m += 1
        b, _ = classify_content(row.text)
        bucket[b] += 1
        blen[b] += len(row.text)
        if b in RESEARCH:
            d, _ = classify_domain(row.text)
            domain[d] += 1
            dom_by_bucket.setdefault(b, Counter())[d] += 1

    TOT = 5_654_358
    res_n = sum(bucket[b] for b in RESEARCH)
    code_n = sum(bucket[b] for b in CODE)

    print("=" * 74)
    print(f" 버킷·분야 전 구간 재판정 — 전수 {n:,}행 중 {m:,}행 판정 (1/{a.stride})")
    print(f" 비교 대상: 앞부분 연속 60,000행 기준 기존 수치")
    print("=" * 74)

    print(f"\n[내용 버킷]  {'비율':>8s} {'변화':>7s}  {'전체 추정':>11s}  평균길이")
    for b, c in bucket.most_common():
        p = c / max(1, m)
        print(f"  {b:14s} {p*100:6.2f}% {arrow(p, HEAD_BUCKET.get(b, 0))}"
              f"  {int(p*TOT):>10,}행  {blen[b]//max(1,c):>5,}자")

    rp, cp = res_n / max(1, m), code_n / max(1, m)
    print(f"\n  연구 내용 소계   {rp*100:6.2f}%  (기존 46.54%) → 약 {int(rp*TOT):,}행")
    print(f"  GitHub  소계   {cp*100:6.2f}%  (기존 14.59%) → 약 {int(cp*TOT):,}행")

    print(f"\n[연구 분야]  연구 버킷 {res_n:,}행 대비")
    for d, c in domain.most_common():
        p = c / max(1, res_n)
        print(f"  {d:16s} {p*100:6.2f}% {arrow(p, HEAD_DOMAIN.get(d, 0))}"
              f"  {int(p*rp*TOT):>10,}행")
    known = 1 - domain["미분류"] / max(1, res_n)
    if known:
        top4 = sum(sorted((c for d, c in domain.items() if d != "미분류"),
                          reverse=True)[:4]) / max(1, res_n)
        print(f"\n  분류된 행 기준 최대 분야 점유 "
              f"{max((c for d,c in domain.items() if d!='미분류'), default=0)/max(1,res_n)/known*100:.1f}%")
        print(f"  상위 4개 분야 점유 {top4/known*100:.1f}%")

    print("\n[유형별 주도 분야]")
    for b in ("experimental", "analysis", "literature", "planning"):
        cc = dom_by_bucket.get(b)
        if not cc:
            continue
        t = sum(cc.values())
        top = ", ".join(f"{k} {v/t*100:.0f}%" for k, v in cc.most_common(3))
        print(f"  {b:14s} → {top}")

    # 앞부분 표본과의 최대 괴리
    gaps = sorted(((abs(bucket[b]/max(1,m) - p), b) for b, p in HEAD_BUCKET.items()),
                  reverse=True)
    print("\n" + "=" * 74)
    print(f" 최대 괴리: {gaps[0][1]} {gaps[0][0]*100:.1f}p")
    if gaps[0][0] < 0.02:
        print(" 판정: 앞부분 표본이 전 구간을 잘 대표했다. 기존 수치 유지 가능.")
    elif gaps[0][0] < 0.05:
        print(" 판정: 경미한 편향. 수치를 이 결과로 갱신한다.")
    else:
        print(" 판정: 유의한 편향. 앞부분 기준 수치를 전부 폐기하고 이 결과로 대체한다.")
    print("=" * 74)

    Path(a.json).write_text(json.dumps({
        "total_rows": n, "judged_rows": m, "stride": a.stride,
        "bucket": dict(bucket),
        "bucket_pct": {b: round(c / max(1, m), 4) for b, c in bucket.items()},
        "bucket_estimated": {b: int(c / max(1, m) * TOT) for b, c in bucket.items()},
        "avg_len": {b: blen[b] // max(1, bucket[b]) for b in bucket},
        "research_pct": round(rp, 4), "code_pct": round(cp, 4),
        "domain": dict(domain),
        "domain_pct": {d: round(c / max(1, res_n), 4) for d, c in domain.items()},
        "domain_by_bucket": {b: dict(cc) for b, cc in dom_by_bucket.items()},
        "head_bucket_pct": HEAD_BUCKET,
        "max_gap": {"bucket": gaps[0][1], "delta": round(gaps[0][0], 4)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
