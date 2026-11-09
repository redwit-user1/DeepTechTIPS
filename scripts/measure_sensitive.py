#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""민감정보(건강) 문서 비율 전수 측정 — 집계만 출력한다(원문 미출력).

    python scripts/measure_sensitive.py ocr_goono_ocr_texts.csv
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.gate import Gate
from compliance_gateway.data.ocr.sensitive import sensitive_signals


def main() -> None:
    rows = docs = 0
    seen: set[int] = set()
    flagged: set[int] = set()
    why: Counter[str] = Counter()
    for r in iter_rows(Path(sys.argv[1])):
        rows += 1
        k = Gate.note_key(r.note_id)
        seen.add(k)
        s = sensitive_signals(r.text)
        if s:
            flagged.add(k)
            why.update(s)
    docs = len(seen)
    print(f"행 {rows:,} / 문서 {docs:,}")
    print(f"민감 표지 행 사유: {dict(why)}")
    print(f"민감 문서 {len(flagged):,} ({len(flagged)/max(1, docs):.2%}) — 문서 단위로 학습에서 제외")


if __name__ == "__main__":
    main()
