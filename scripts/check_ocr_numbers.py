#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR 숫자 오염률 — 물리적으로 경계가 있는 양만 본다.

`check_corpus_risks.py` 의 숫자 검사는 오탐이 심했다:

| 신호 | 왜 못 쓰나 |
|---|---|
| 다중 소수점 `1.2.3` | 버전(v1.2.3)·절번호(3.1.2)·날짜(2024.03.15)가 전부 매치 |
| 퍼센트 >= 100% | 증가율 250%, 회수율 105% 는 정상값이다 |

그래서 **원본이 무엇이었든 틀린 게 확실한 것**만 남긴다 —
물리적 상한이 정해진 양(pH 0~14, 순도·함량·습도 0~100%)의 위반.
이건 오탐이 거의 없고, 위반율이 곧 **해당 문맥의 숫자 오염률 추정치**다.

`숫자 분리(1 2 3)` 는 숫자 오류가 아니라 **표 구조 손실**이므로 따로 센다.
원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows

_N = r"(\d{1,5}(?:[.,]\d+)?)"

# 물리적 상한이 있는 양 — (정규식, 하한, 상한)
BOUNDED: dict[str, tuple[re.Pattern, float, float]] = {
    "pH (0~14)": (re.compile(rf"pH\s*(?:값)?\s*[:=]?\s*{_N}", re.IGNORECASE), 0, 14),
    "순도 (0~100%)": (re.compile(rf"(?:순도|purity)\s*[:=]?\s*{_N}\s*%", re.IGNORECASE), 0, 100),
    "함량 (0~100%)": (re.compile(rf"(?:함량|함유량|content)\s*[:=]?\s*{_N}\s*%", re.IGNORECASE), 0, 100),
    "습도 (0~100%)": (re.compile(rf"(?:습도|humidity|\bRH\b)\s*[:=]?\s*{_N}\s*%", re.IGNORECASE), 0, 100),
    "수율 (0~100%)": (re.compile(rf"(?:수율|yield)\s*[:=]?\s*{_N}\s*%", re.IGNORECASE), 0, 100),
    "비율 (0~100%)": (re.compile(rf"(?:비율|점유율|백분율)\s*[:=]?\s*{_N}\s*%"), 0, 100),
}
# 숫자 속 letter — O/0, l/1, S/5 혼동. 오탐이 거의 없다
DIGIT_LETTER = re.compile(r"\d[OolIiSsBZ]\d")
# 표 구조 손실 — 숫자 오류가 아니라 별개 문제
SPLIT_NUM = re.compile(r"\b\d\s\d\s\d\b")
NUM_TOKEN = re.compile(r"\d[\d.,]*")
# 원본이 디지털인지 스캔인지 가르는 단서: 스캔본은 문자 오인식 흔적이 많다
SCAN_ARTIFACT = re.compile(r"[│┃┆╎▯□■◇○●]|[가-힣][A-Za-z][가-힣]|�")

for _n, _r in (("DIGIT_LETTER", DIGIT_LETTER), ("SPLIT_NUM", SPLIT_NUM),
               ("NUM_TOKEN", NUM_TOKEN), ("SCAN_ARTIFACT", SCAN_ARTIFACT)):
    assert not _r.search(""), f"{_n} 빈 대안"
for _k, (_r, _lo, _hi) in BOUNDED.items():
    assert not _r.search(""), f"{_k} 빈 대안"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--json", default="ocr_numbers.json")
    a = ap.parse_args()

    n = m = 0
    seen: Counter[str] = Counter()
    viol: Counter[str] = Counter()
    by_ext_seen: Counter[str] = Counter()
    by_ext_viol: Counter[str] = Counter()
    num_tokens = dl = sp = 0
    rows_split = 0
    scanish = 0

    for row in iter_rows(Path(a.csv_path)):
        n += 1
        if n % a.stride:
            continue
        m += 1
        t = row.text
        ext = Path(row.file_name).suffix.lower() or "(없음)"
        num_tokens += len(NUM_TOKEN.findall(t))
        dl += len(DIGIT_LETTER.findall(t))
        s = len(SPLIT_NUM.findall(t))
        sp += s
        if s:
            rows_split += 1
        if SCAN_ARTIFACT.search(t):
            scanish += 1
        for label, (rx, lo, hi) in BOUNDED.items():
            for mm in rx.finditer(t):
                try:
                    v = float(mm.group(1).replace(",", "."))
                except ValueError:
                    continue
                seen[label] += 1
                by_ext_seen[ext] += 1
                if v < lo or v > hi:
                    viol[label] += 1
                    by_ext_viol[ext] += 1

    tot_seen = sum(seen.values())
    tot_viol = sum(viol.values())
    print("=" * 70)
    print(f" OCR 숫자 오염률 — 전수 {n:,}행 / 정밀 {m:,}행 (1/{a.stride})")
    print("=" * 70)

    print("\n[물리적 상한 위반 — 오탐이 거의 없는 신호]")
    for k in BOUNDED:
        s_, v_ = seen[k], viol[k]
        if not s_:
            print(f"  {k:16s}  언급 없음")
            continue
        r = v_ / s_
        print(f"  {k:16s} 언급 {s_:>7,}  위반 {v_:>6,}  = {r*100:5.2f}%"
              f" {'#'*int(r*60)}")
    if tot_seen:
        print(f"  {'합계':16s} 언급 {tot_seen:>7,}  위반 {tot_viol:>6,}"
              f"  = {tot_viol/tot_seen*100:5.2f}%   <- 숫자 오염률 추정치")

    print("\n[파일 형식별 위반율 — 스캔본일수록 높아야 한다]")
    for e, s_ in by_ext_seen.most_common(6):
        if s_ >= 50:
            print(f"  {e:10s} 언급 {s_:>7,}  위반율 {by_ext_viol[e]/s_*100:5.2f}%")

    print(f"\n[보조 신호]")
    print(f"  숫자 속 letter(1O0)   {dl:>8,}  = 숫자 1,000개당 "
          f"{dl/max(1,num_tokens)*1000:.2f}건")
    print(f"  숫자 분리(1 2 3)      {sp:>8,}  포함 행 {rows_split:,}"
          f" ({rows_split/max(1,m)*100:.1f}%)  <- 숫자 오류 아님. 표 구조 손실")
    print(f"  스캔 아티팩트 포함 행  {scanish:>8,} ({scanish/max(1,m)*100:.1f}%)")

    Path(a.json).write_text(json.dumps({
        "total_rows": n, "profiled_rows": m, "stride": a.stride,
        "bounded_seen": dict(seen), "bounded_violations": dict(viol),
        "violation_rate": (round(tot_viol / tot_seen, 4) if tot_seen else None),
        "by_ext_seen": dict(by_ext_seen.most_common(12)),
        "by_ext_violation_rate": {e: round(by_ext_viol[e] / s, 4)
                                  for e, s in by_ext_seen.most_common(12) if s >= 50},
        "num_tokens": num_tokens,
        "digit_letter": dl, "digit_letter_per_1k": round(dl / max(1, num_tokens) * 1000, 3),
        "split_num": sp, "rows_with_split_pct": round(rows_split / max(1, m), 4),
        "scan_artifact_pct": round(scanish / max(1, m), 4),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
