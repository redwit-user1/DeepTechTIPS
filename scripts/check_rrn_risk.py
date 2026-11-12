#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""주민등록번호 검출 신뢰도 확인 — **값을 출력하지 않고** 체크섬으로만 판정.

`\\d{6}-[1-4]\\d{6}` 패턴은 과제번호·전화번호·날짜 조합에서 오탐할 수 있다.
실제 주민등록번호는 **마지막 자리가 검증번호(체크섬)** 이므로,
체크섬 통과율로 진짜 여부를 가늠할 수 있다.

  - 통과율이 무작위 수준(~1/11 ≈ 9%)이면 대부분 오탐
  - 통과율이 높으면 **실제 고유식별정보**일 가능성이 크다 → 즉시 조치 필요

출력에는 원문·번호가 일절 포함되지 않는다.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows

RRN = re.compile(r"(\d{6})\s*[-–]\s*([1-4]\d{6})")
_WEIGHTS = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)


def rrn_checksum_ok(front: str, back: str) -> bool:
    """주민등록번호 검증번호 규칙."""
    digits = [int(c) for c in front + back]
    if len(digits) != 13:
        return False
    total = sum(d * w for d, w in zip(digits[:12], _WEIGHTS))
    return (11 - (total % 11)) % 10 == digits[12]


def plausible_birthdate(front: str, gender: str) -> bool:
    """앞 6자리가 실제 생년월일 형식인가(월 01~12, 일 01~31)."""
    mm, dd = int(front[2:4]), int(front[4:6])
    return 1 <= mm <= 12 and 1 <= dd <= 31


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=200000)
    a = ap.parse_args()

    total = checksum_ok = date_ok = both = 0
    ext_hits: Counter[str] = Counter()
    rows_with_hit = 0

    for row in iter_rows(Path(a.csv_path), limit=a.rows):
        hits = RRN.findall(row.text)
        if not hits:
            continue
        rows_with_hit += 1
        ext_hits[row.ext] += len(hits)
        for front, back in hits:
            total += 1
            c = rrn_checksum_ok(front, back)
            d = plausible_birthdate(front, back[0])
            checksum_ok += c
            date_ok += d
            both += (c and d)

    print("=" * 62)
    print(" 주민등록번호 패턴 검증 (값 미출력)")
    print("=" * 62)
    print(f"\n표본 {a.rows:,}행 중 검출 행 {rows_with_hit:,}건 / 패턴 {total:,}개")
    if total == 0:
        print("\n검출 없음.")
        return

    print(f"\n  체크섬 통과      {checksum_ok:>6,} / {total:,}  ({checksum_ok/total*100:5.1f}%)")
    print(f"  생년월일 형식     {date_ok:>6,} / {total:,}  ({date_ok/total*100:5.1f}%)")
    print(f"  둘 다 통과       {both:>6,} / {total:,}  ({both/total*100:5.1f}%)")
    print(f"\n  무작위 기대치는 체크섬 약 9% (1/11)")

    print("\n  검출된 원본 파일 형식:")
    for ext, c in ext_hits.most_common(8):
        print(f"    {ext:10s} {c:,}건")

    print("\n" + "=" * 62)
    if both / total > 0.5:
        print(" 🚨 실제 주민등록번호가 다량 포함된 것으로 판단됩니다.")
        print("    개인정보보호법상 고유식별정보입니다. 학습·복제 전에")
        print("    반드시 제거하고, 원본 보관 경위도 확인하세요.")
    elif both / total > 0.15:
        print(" ⚠️  일부는 실제 주민등록번호일 가능성이 있습니다.")
        print("    비식별화를 필수로 적용하고 표본을 직접 확인하세요.")
    else:
        print(" ✅ 대부분 오탐으로 보입니다(과제번호·전화·날짜 조합 등).")
        print("    그래도 비식별화 규칙은 유지하는 편이 안전합니다.")
    print("=" * 62)


if __name__ == "__main__":
    main()
