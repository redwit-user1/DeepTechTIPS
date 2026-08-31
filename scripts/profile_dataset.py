#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""데이터셋 분포 — 학습셋으로 가공할 수 있는가.

지금까지 잰 적 없는 것들을 전수로 잰다. 특히 **중복도**는 코퍼스 가치를
직접 깎는 요소인데 한 번도 측정하지 않았다.

| 측정 | 왜 필요한가 |
|---|---|
| 정확 중복 | 같은 문서 재업로드·페이지 반복. 학습 전 제거 대상 |
| 템플릿 반복 | 앞부분이 같은 행(머리글·양식). 실질 정보량을 줄인다 |
| 길이 분포 | 청킹 전략과 폐기 임계값을 정한다 |
| 언어 혼합 | 한국어 비중이 실제로 얼마인가 |
| 확장자 교차 | docx 가 pdf 보다 깨끗하다는 사실을 활용하려면 필요 |
| 토큰 추정 | 사전학습 예산 산정 |

중복 측정은 **전수여야 한다.** 1/N 표본으로 재면 중복 쌍이 같은 표본에
들어올 확률이 1/N^2 이라 중복률이 심하게 과소평가된다.

원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows

PREFIX_CAP = 3_000_000       # 템플릿 카운터 상한(메모리 방어)
PREFIX_CHARS = 120

_WS = re.compile(r"\s+")
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")

# 문자 종류별 대략적인 chars/token — 현대 BPE(Llama3/Qwen2 계열) 관측 범위.
# 토크나이저가 없어 실측 불가하므로 **구간**으로 낸다. 점추정은 하지 않는다.
CPT = {"hangul": (1.3, 1.8), "latin": (3.5, 4.5), "digit": (1.5, 2.5), "other": (3.0, 4.5)}


def norm(t: str) -> str:
    return _WS.sub(" ", t).strip()


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2s(s.encode("utf-8", "ignore"),
                                          digest_size=8).digest(), "big")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--json", default="ocr_dataset_profile.json")
    a = ap.parse_args()

    n = 0
    seen: set[int] = set()
    dup_rows = 0
    prefix: Counter[int] = Counter()
    prefix_capped = False

    len_hist: Counter[str] = Counter()
    lang_hist: Counter[str] = Counter()
    ext_rows: Counter[str] = Counter()
    ext_chars: Counter[str] = Counter()
    ext_dup: Counter[str] = Counter()

    c_total = c_han = c_lat = c_dig = 0
    empty = 0
    # 품질 티어: 길이 200자+ & 한글 또는 영문 비중 30%+ & 중복 아님
    tier_ok = 0
    tier_ok_chars = 0

    for row in iter_rows(Path(a.csv_path)):
        n += 1
        t = norm(row.text)
        L = len(t)
        ext = Path(row.file_name).suffix.lower() or "(없음)"
        ext_rows[ext] += 1
        ext_chars[ext] += L

        if not L:
            empty += 1
            len_hist["0(빈행)"] += 1
            continue

        hh = h64(t)
        is_dup = hh in seen
        if is_dup:
            dup_rows += 1
            ext_dup[ext] += 1
        else:
            seen.add(hh)

        if len(prefix) < PREFIX_CAP:
            prefix[h64(t[:PREFIX_CHARS])] += 1
        else:
            prefix_capped = True

        len_hist["1-99" if L < 100 else "100-299" if L < 300 else
                 "300-999" if L < 1000 else "1000-2999" if L < 3000 else
                 "3000-9999" if L < 10000 else "10000+"] += 1

        han, lat, dig = (len(_HANGUL.findall(t)), len(_LATIN.findall(t)),
                         len(_DIGIT.findall(t)))
        c_total += L; c_han += han; c_lat += lat; c_dig += dig
        letters = han + lat
        r = han / letters if letters else 0.0
        lang_hist["문자없음" if not letters else
                  "한글중심(>70%)" if r > 0.7 else
                  "혼합" if r > 0.3 else "영문중심(<30%)"] += 1

        if L >= 200 and letters / L >= 0.3 and not is_dup:
            tier_ok += 1
            tier_ok_chars += L

    uniq = len(seen)
    tmpl_rep = sum(c - 1 for c in prefix.values() if c > 1)

    def tok_range(han, lat, dig, other):
        lo = han / CPT["hangul"][1] + lat / CPT["latin"][1] + \
             dig / CPT["digit"][1] + other / CPT["other"][1]
        hi = han / CPT["hangul"][0] + lat / CPT["latin"][0] + \
             dig / CPT["digit"][0] + other / CPT["other"][0]
        return lo, hi

    c_oth = c_total - c_han - c_lat - c_dig
    lo, hi = tok_range(c_han, c_lat, c_dig, c_oth)

    print("=" * 70)
    print(f" 데이터셋 분포 — 전수 {n:,}행")
    print("=" * 70)

    print(f"\n[중복] 전수 측정")
    print(f"  고유 행          {uniq:>10,} ({uniq/max(1,n)*100:5.1f}%)")
    print(f"  정확 중복 행     {dup_rows:>10,} ({dup_rows/max(1,n)*100:5.1f}%)  <- 학습 전 제거")
    print(f"  빈 행            {empty:>10,} ({empty/max(1,n)*100:5.1f}%)")
    print(f"  템플릿 반복(앞{PREFIX_CHARS}자 동일, 정확중복 포함)"
          f" {tmpl_rep:>10,} ({tmpl_rep/max(1,n)*100:5.1f}%)"
          + ("  [카운터 상한 도달]" if prefix_capped else ""))

    print(f"\n[길이 분포] 정규화 후")
    for k in ("0(빈행)", "1-99", "100-299", "300-999", "1000-2999", "3000-9999", "10000+"):
        c = len_hist[k]
        print(f"  {k:12s} {c:>10,} ({c/max(1,n)*100:5.1f}%) {'#'*int(c/max(1,n)*40)}")

    print(f"\n[언어]")
    for k, c in lang_hist.most_common():
        print(f"  {k:16s} {c:>10,} ({c/max(1,n)*100:5.1f}%) {'#'*int(c/max(1,n)*36)}")
    print(f"  문자 구성: 한글 {c_han/max(1,c_total)*100:.1f}% / "
          f"영문 {c_lat/max(1,c_total)*100:.1f}% / "
          f"숫자 {c_dig/max(1,c_total)*100:.1f}% / 기타 {c_oth/max(1,c_total)*100:.1f}%")

    print(f"\n[확장자별]")
    print(f"  {'확장자':10s} {'행':>10s} {'문자':>14s} {'평균':>7s} {'중복률':>7s}")
    for e, c in ext_rows.most_common(8):
        print(f"  {e:10s} {c:>10,} {ext_chars[e]:>14,} "
              f"{ext_chars[e]//max(1,c):>7,} {ext_dup[e]/max(1,c)*100:>6.1f}%")

    print(f"\n[총량]")
    print(f"  전체 문자        {c_total:>14,}자")
    print(f"  추정 토큰        {lo/1e6:>10,.0f}M ~ {hi/1e6:,.0f}M  (토크나이저 미보유, 구간추정)")
    print(f"\n[학습 가능 티어] 200자+ & 문자비중 30%+ & 중복 아님")
    print(f"  행               {tier_ok:>10,} ({tier_ok/max(1,n)*100:5.1f}%)")
    print(f"  문자             {tier_ok_chars:>14,}자 ({tier_ok_chars/max(1,c_total)*100:.1f}%)")
    tlo, thi = tok_range(tier_ok_chars * c_han / max(1, c_total),
                         tier_ok_chars * c_lat / max(1, c_total),
                         tier_ok_chars * c_dig / max(1, c_total),
                         tier_ok_chars * c_oth / max(1, c_total))
    print(f"  추정 토큰        {tlo/1e6:>10,.0f}M ~ {thi/1e6:,.0f}M")

    Path(a.json).write_text(json.dumps({
        "total_rows": n, "unique_rows": uniq,
        "duplicate_rows": dup_rows, "duplicate_pct": round(dup_rows / max(1, n), 4),
        "empty_rows": empty,
        "template_repeat_rows": tmpl_rep, "template_capped": prefix_capped,
        "len_hist": dict(len_hist), "lang_hist": dict(lang_hist),
        "chars": {"total": c_total, "hangul": c_han, "latin": c_lat,
                  "digit": c_dig, "other": c_oth},
        "token_estimate": {"low_M": round(lo / 1e6), "high_M": round(hi / 1e6)},
        "ext_rows": dict(ext_rows.most_common(12)),
        "ext_chars": {e: ext_chars[e] for e, _ in ext_rows.most_common(12)},
        "ext_dup_pct": {e: round(ext_dup[e] / max(1, c), 4)
                        for e, c in ext_rows.most_common(12)},
        "trainable_rows": tier_ok, "trainable_chars": tier_ok_chars,
        "trainable_token_M": {"low": round(tlo / 1e6), "high": round(thi / 1e6)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
