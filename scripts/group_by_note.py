#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NoteId 로 실제 그룹핑해서 문서 단위 판정을 한다.

## 왜 새로 짜는가

`csv_source.iter_notes` 는 같은 NoteId 행이 **인접해 있다**고 가정하고
"직전 NoteId 가 바뀌면 flush" 한다. 그런데 이 파일은 그렇지 않다 —
400,000행을 조사하니 직전 행과 NoteId 가 같은 경우가 **3건(0.0%)** 뿐이었다.
그래서 `iter_notes` 를 쓰면 문서 = 행이 되어 그룹핑이 전혀 일어나지 않는다.
(문서 5,654,300개 / 평균 1.0페이지라는 결과가 그 증거였다.)

## 대신 하는 것

전체를 한 번 훑으며 **NoteId 별로 표지 비트마스크를 OR 누적**한다.
한 문서의 어느 페이지에든 표지가 있으면 그 문서 전체가 그 성격이다.
문서 하나를 int 하나로 압축해 담는다:

    [ paper 7bit | record 4bit | rowcount ]

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
from check_paper_contamination import PAPER, RECORD, PREFILTER

ROW_SCAN = 1500          # 페이지당 훑을 앞부분. 표지는 페이지 머리에 몰려 있다
CNT_SHIFT = 11
MASK_BITS = (1 << CNT_SHIFT) - 1

P_KEYS = list(PAPER)
R_KEYS = list(RECORD)
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def nid(s: str) -> int:
    return int.from_bytes(hashlib.blake2s(s.encode(), digest_size=7).digest(), "big")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--doc-stride", type=int, default=8,
                    help="NoteId 해시 기준 문서 표본 간격. 행이 아니라 문서를 솎으므로 "
                         "그룹핑이 깨지지 않는다")
    ap.add_argument("--json", default="ocr_paper_grouped.json")
    a = ap.parse_args()

    acc: dict[int, int] = {}          # note -> packed
    lang_acc: dict[int, int] = {}     # note -> (han<<12 | lat), 첫 페이지 기준
    n = 0
    kept = 0

    # **문서 단위 해시 표본.** 행을 솎으면 한 문서가 조각나 그룹핑이 깨진다.
    # NoteId 해시로 고르면 그 문서의 모든 페이지가 통째로 들어오거나 빠진다.
    # 딕셔너리 두 개 x 300만 항목으로 메모리가 터졌던 것을 이걸로 해결한다.
    for row in iter_rows(Path(a.csv_path)):
        n += 1
        key = nid(row.note_id)
        if key % a.doc_stride:
            continue
        kept += 1
        t = row.text[:ROW_SCAN]
        pm = rm = 0
        if PREFILTER.search(t):
            for i, k in enumerate(P_KEYS):
                if PAPER[k].search(t):
                    pm |= 1 << i
            for i, k in enumerate(R_KEYS):
                if RECORD[k].search(t):
                    rm |= 1 << i
        prev = acc.get(key, 0)
        packed = ((prev >> CNT_SHIFT) + 1) << CNT_SHIFT
        packed |= (prev & MASK_BITS) | pm | (rm << 7)
        acc[key] = packed
        if key not in lang_acc:
            lang_acc[key] = (min(len(_HANGUL.findall(t)), 4095) << 12) | \
                            min(len(_LATIN.findall(t)), 4095)

    docs = len(acc)
    verdict: Counter[str] = Counter()
    v_rows: Counter[str] = Counter()
    pages_hist: Counter[str] = Counter()
    marker: Counter[str] = Counter()
    lang_v: Counter[tuple[str, str]] = Counter()
    order = ["출판 논문 추정", "논문 가능성", "1차 기록", "표지 없음"]

    for key, packed in acc.items():
        cnt = packed >> CNT_SHIFT
        pm = packed & 0x7F
        rm = (packed >> 7) & 0xF
        np_ = bin(pm).count("1")
        nr = bin(rm).count("1")
        for i, k in enumerate(P_KEYS):
            if pm >> i & 1:
                marker[k] += 1
        if np_ >= 2 and np_ >= nr:
            v = "출판 논문 추정"
        elif np_ and not nr:
            v = "논문 가능성"
        elif nr:
            v = "1차 기록"
        else:
            v = "표지 없음"
        verdict[v] += 1
        v_rows[v] += cnt
        pages_hist["1" if cnt == 1 else "2-4" if cnt <= 4 else
                   "5-9" if cnt <= 9 else "10-29" if cnt <= 29 else "30+"] += 1
        la = lang_acc[key]
        han, lat = la >> 12, la & 0xFFF
        letters = han + lat
        L = ("문자없음" if not letters else "한글중심" if han / letters > 0.7
             else "혼합" if han / letters > 0.3 else "영문중심")
        lang_v[(L, v)] += cnt

    tot = sum(v_rows.values())
    print("=" * 72)
    print(f" 문서 단위 판정 — NoteId 실제 그룹핑")
    print(f" 전수 {n:,}행 중 표본 {kept:,}행(문서 1/{a.doc_stride})"
          f" → 문서 {docs:,}개 · 문서당 평균 {kept/max(1,docs):.2f}페이지")
    print("=" * 72)

    print("\n[문서 페이지 수 분포]")
    for k in ("1", "2-4", "5-9", "10-29", "30+"):
        c = pages_hist[k]
        print(f"  {k:>6s}페이지 {c:>9,} ({c/max(1,docs)*100:5.1f}%)"
              f" {'#'*int(c/max(1,docs)*36)}")

    print("\n[판정] 문서 기준 / 행 기준")
    for v in order:
        d, r = verdict[v], v_rows[v]
        print(f"  {v:12s} 문서 {d:>9,} ({d/max(1,docs)*100:5.1f}%)  "
              f"행 {r:>9,} ({r/max(1,tot)*100:5.1f}%) {'#'*int(r/max(1,tot)*28)}")

    print("\n[언어 x 판정 — 행 기준]")
    print(f"  {'언어':8s} {'행':>10s} " + " ".join(f"{v[:6]:>8s}" for v in order))
    for L in ["영문중심", "혼합", "한글중심", "문자없음"]:
        t = sum(lang_v[(L, v)] for v in order)
        if t:
            print(f"  {L:8s} {t:>10,} " +
                  " ".join(f"{lang_v[(L,v)]/t*100:>7.1f}%" for v in order))

    print("\n[출판물 표지 — 문서 기준]")
    for k, c in marker.most_common():
        print(f"  {k:12s} {c:>9,} ({c/max(1,docs)*100:5.1f}%)")

    pr = v_rows["출판 논문 추정"] / max(1, tot)
    mr = (v_rows["출판 논문 추정"] + v_rows["논문 가능성"]) / max(1, tot)
    print("\n" + "=" * 72)
    print(f" 고유성 보정: 출판 논문 {pr*100:.1f}% 제외 → {int((1-pr)*tot):,}행")
    print(f"            보수적('논문 가능성' 포함) → {int((1-mr)*tot):,}행")
    print("=" * 72)

    Path(a.json).write_text(json.dumps({
        "rows_total": n, "rows_sampled": kept, "doc_stride": a.doc_stride,
        "documents": docs, "avg_pages": round(kept / max(1, docs), 3),
        "pages_hist": dict(pages_hist),
        "verdict_docs": dict(verdict), "verdict_rows": dict(v_rows),
        "verdict_row_pct": {k: round(v / max(1, tot), 4) for k, v in v_rows.items()},
        "marker_doc_pct": {k: round(v / max(1, docs), 4) for k, v in marker.items()},
        "lang_x_verdict_rows": {f"{L}|{v}": lang_v[(L, v)]
                                for L in ["영문중심", "혼합", "한글중심", "문자없음"]
                                for v in order if lang_v[(L, v)]},
        "paper_row_pct": round(pr, 4),
        "paper_or_maybe_row_pct": round(mr, 4),
        "unique_rows": int((1 - pr) * tot),
        "unique_rows_conservative": int((1 - mr) * tot),
        "row_scan_chars": ROW_SCAN,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
