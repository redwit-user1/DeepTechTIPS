#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""자산 서사를 무너뜨릴 수 있는 세 가지를 전수 패스로 확인한다.

1. **집중도** — "2,000개사"라지만 상위 소수가 대부분이면 서사가 무너진다.
   CSV 에 기업 식별자가 없으므로 **파일명 명명규칙**을 프록시로 쓴다.
   조직마다 문서 이름 짓는 관습이 다르다는 성질을 이용한다. 완전한 대체물은
   아니며(한 회사가 여러 규칙을 쓰거나 그 반대일 수 있다) **분포의 모양**을
   보기 위한 것이다.

2. **OCR 숫자 신뢰도** — 정답이 없으므로 정확도는 못 잰다. 대신 **원본이
   무엇이었든 틀린 게 확실한 것**만 센다(다중 소수점, 숫자 속 letter,
   pH>14, 수율>100%). 이것은 오류율의 **하한선**이다.

3. **시간 분포** — 본문의 완전한 날짜(Y-M-D)에서 연도를 뽑는다.
   인용 연도와 섞이지 않도록 4자리 단독 연도는 따로 센다.

성능: 파일 전체를 훑되 비싼 검사는 stride 로 솎는다. 앞부분 편향을 없애는 게
목적이므로 전수 순회 자체를 생략하지 않는다. 원문·파일명은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows

SALT = os.environ.get("GOONO_DEID_SALT", "goono-deid-v1")
KEY_CAP = 2_000_000          # 키가 이보다 많아지면 신규 등록 중단(메모리 방어)

# ── 1. 조직 프록시 키 ────────────────────────────────────────────────
_NONWORD = re.compile(r"[^0-9A-Za-z가-힣]+")
_DIGITS = re.compile(r"\d+")


def org_key(file_name: str) -> str:
    """파일명 → 조직 프록시 키(해시). 숫자·확장자·구분자를 지우고 앞 12자."""
    stem = Path(file_name).stem
    norm = _DIGITS.sub("", _NONWORD.sub("", stem)).lower()[:12]
    if len(norm) < 3:
        return "(짧음)"
    return hashlib.blake2s(f"{SALT}|{norm}".encode(), digest_size=6).hexdigest()


# ── 2. 확실한 OCR 숫자 오류만 ────────────────────────────────────────
NUM_TOKEN = re.compile(r"\d[\d.,]*")
# 소수점 두 개 이상 — 어떤 수도 이렇게 생기지 않는다
MULTI_DOT = re.compile(r"\d+\.\d*\.\d")
# 숫자 사이에 낀 letter — O/0, l/1, S/5, B/8 혼동의 흔적
DIGIT_LETTER = re.compile(r"\d[OolIiSsBZzg]\d")
# 숫자가 공백으로 쪼개진 흔적(표 OCR 에서 흔함)
SPLIT_NUM = re.compile(r"\b\d\s\d\s\d\b")
# 물리적으로 불가능
BAD_PH = re.compile(r"pH\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
BAD_YIELD = re.compile(r"(?:수율|yield)\s*[:=]?\s*(\d{1,4}(?:\.\d+)?)\s*%", re.IGNORECASE)
BAD_PCT = re.compile(r"(?<![\d.])(\d{3,5}(?:\.\d+)?)\s*%")

# ── 3. 날짜 ─────────────────────────────────────────────────────────
FULL_DATE = re.compile(r"(20[0-3]\d)\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")
BARE_YEAR = re.compile(r"(?<![\d.])((?:19|20)\d{2})\s*(?:년|\)|\.|,|\s)")

for _n, _r in (("MULTI_DOT", MULTI_DOT), ("DIGIT_LETTER", DIGIT_LETTER),
               ("SPLIT_NUM", SPLIT_NUM), ("FULL_DATE", FULL_DATE),
               ("BARE_YEAR", BARE_YEAR), ("NUM_TOKEN", NUM_TOKEN)):
    assert not _r.search(""), f"{_n} 빈 대안"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--stride", type=int, default=25,
                    help="비싼 검사를 N행마다 1회 수행(기본 25)")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--json", default="ocr_risks.json")
    a = ap.parse_args()

    n = 0                       # 전수
    m = 0                       # 정밀 검사한 행
    orgs: Counter[str] = Counter()
    org_capped = False
    ext: Counter[str] = Counter()

    num_tokens = 0
    err_multi = err_letter = err_split = 0
    err_ph = err_yield = err_pct = 0
    rows_with_err = 0
    ph_seen = yield_seen = 0

    years: Counter[int] = Counter()
    bare_years: Counter[int] = Counter()
    rows_dated = 0

    for row in iter_rows(Path(a.csv_path), limit=a.max_rows):
        n += 1
        # ① 전수 — 싸다
        if len(orgs) < KEY_CAP:
            orgs[org_key(row.file_name)] += 1
        else:
            org_capped = True
        ext[Path(row.file_name).suffix.lower() or "(없음)"] += 1

        if n % a.stride:
            continue
        m += 1
        t = row.text
        # ② 숫자 신뢰도
        num_tokens += len(NUM_TOKEN.findall(t))
        e1 = len(MULTI_DOT.findall(t))
        e2 = len(DIGIT_LETTER.findall(t))
        e3 = len(SPLIT_NUM.findall(t))
        err_multi += e1
        err_letter += e2
        err_split += e3
        e4 = e5 = 0
        for mm in BAD_PH.finditer(t):
            ph_seen += 1
            try:
                if float(mm.group(1)) > 14:
                    e4 += 1
            except ValueError:
                pass
        for mm in BAD_YIELD.finditer(t):
            yield_seen += 1
            try:
                if float(mm.group(1)) > 100:
                    e5 += 1
            except ValueError:
                pass
        e6 = len(BAD_PCT.findall(t))
        err_ph += e4
        err_yield += e5
        err_pct += e6
        if e1 + e2 + e3 + e4 + e5 + e6:
            rows_with_err += 1
        # ③ 시간
        found = FULL_DATE.findall(t)
        if found:
            rows_dated += 1
            for y, _mo, _d in found:
                years[int(y)] += 1
        for y in BARE_YEAR.findall(t):
            bare_years[int(y)] += 1

    # ── 집중도 ──
    tot = sum(orgs.values())
    ranked = orgs.most_common()

    def share(k):
        return sum(c for _, c in ranked[:k]) / max(1, tot)

    counts = sorted(orgs.values())
    cum = 0
    g = 0
    for c in counts:
        cum += c
        g += cum
    gini = 1 - 2 * g / (len(counts) * max(1, cum)) + 1 / max(1, len(counts))

    print("=" * 70)
    print(f" 코퍼스 리스크 점검 — 전수 {n:,}행 / 정밀 {m:,}행 (1/{a.stride})")
    print("=" * 70)

    print(f"\n(1) 집중도 — 조직 프록시 키 {len(orgs):,}개"
          + ("  [상한 도달 · 절단됨]" if org_capped else ""))
    for k in (1, 5, 10, 50, 100, 500):
        if k <= len(ranked):
            s = share(k)
            print(f"  상위 {k:>4,}개 누적 점유율   {s*100:5.1f}% {'#'*int(s*34)}")
    print(f"  Gini 계수                {gini:.3f}   (0=완전균등, 1=완전집중)")
    print(f"  키당 중앙값 {counts[len(counts)//2]:,}행 / 최대 {ranked[0][1]:,}행")

    print("\n  [파일 형식]")
    for e, c in ext.most_common(8):
        print(f"    {e:10s} {c:>9,} ({c/max(1,n)*100:5.1f}%)")

    print(f"\n(2) OCR 숫자 신뢰도 — 숫자 토큰 {num_tokens:,}개 (정밀 {m:,}행)")
    tot_err = err_multi + err_letter + err_split + err_ph + err_yield + err_pct
    for lbl, c in (("다중 소수점(1.2.3)", err_multi), ("숫자 속 letter(1O0)", err_letter),
                   ("숫자 분리(1 2 3)", err_split), ("pH > 14", err_ph),
                   ("수율 > 100%", err_yield), ("퍼센트 >= 100%", err_pct)):
        print(f"  {lbl:22s} {c:>8,}")
    per1k = tot_err / max(1, num_tokens) * 1000
    print(f"  -- 합계 {tot_err:,}건 = 숫자 1,000개당 {per1k:.2f}건 (오류율 하한선)")
    print(f"  오류 포함 행 {rows_with_err:,} ({rows_with_err/max(1,m)*100:.1f}%)")
    if ph_seen:
        print(f"  참고: pH 언급 {ph_seen:,}건 중 불가능 {err_ph:,}"
              f" ({err_ph/max(1,ph_seen)*100:.1f}%)")
    if yield_seen:
        print(f"        수율 언급 {yield_seen:,}건 중 불가능 {err_yield:,}"
              f" ({err_yield/max(1,yield_seen)*100:.1f}%)")

    print(f"\n(3) 시간 분포 — 완전한 날짜 포함 {rows_dated:,}행"
          f" ({rows_dated/max(1,m)*100:.1f}%)")
    ytot = sum(years.values())
    for y in sorted(years):
        if years[y] >= ytot * 0.005:
            print(f"  {y}  {years[y]:>7,} ({years[y]/max(1,ytot)*100:5.1f}%)"
                  f" {'#'*int(years[y]/max(1,ytot)*40)}")
    recent = sum(c for y, c in years.items() if y >= 2022)
    print(f"  2022년 이후 {recent:,} / {ytot:,} = {recent/max(1,ytot)*100:.1f}%")
    print("\n  [단독 4자리 연도 — 인용 연도가 섞인다. 대조용]")
    btot = sum(bare_years.values())
    for y in sorted(bare_years):
        if bare_years[y] >= btot * 0.02:
            print(f"    {y}  {bare_years[y]:>7,} ({bare_years[y]/max(1,btot)*100:5.1f}%)")

    Path(a.json).write_text(json.dumps({
        "total_rows": n, "profiled_rows": m, "stride": a.stride,
        "org_keys": len(orgs), "org_capped": org_capped,
        "org_share": {f"top{k}": round(share(k), 4)
                      for k in (1, 5, 10, 50, 100, 500) if k <= len(ranked)},
        "org_gini": round(gini, 4),
        "org_max_rows": ranked[0][1],
        "ext": dict(ext.most_common(12)),
        "num_tokens": num_tokens,
        "errors": {"multi_dot": err_multi, "digit_letter": err_letter,
                   "split_num": err_split, "ph_gt14": err_ph,
                   "yield_gt100": err_yield, "pct_ge100": err_pct},
        "error_per_1k_numbers": round(per1k, 3),
        "rows_with_error_pct": round(rows_with_err / max(1, m), 4),
        "ph_mentions": ph_seen, "yield_mentions": yield_seen,
        "rows_dated": rows_dated,
        "year_hist": {str(k): v for k, v in sorted(years.items())},
        "bare_year_hist": {str(k): v for k, v in sorted(bare_years.items())},
        "since_2022_pct": round(recent / max(1, ytot), 4),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
