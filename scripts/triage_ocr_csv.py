#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR 코퍼스 용도 판정(triage) — 어떤 부분집합이 무엇에 쓸모 있는가.

프로파일링 결과 이 코퍼스는 **연구노트 양식이 아니라** 연구자들이 업로드한
각종 연구자료(PDF/PPTX/XLSX)였다. 그렇다면 통째로 한 용도에 쓸 수 없고,
**부분집합별로 용도를 나눠야** 한다. 이 스크립트가 그 비율을 잰다.

판정 버킷:
  labnote_like  연구노트 구조 단서 다수 → 기록 무결성 검사(`integrity/`) 대상
  cited         참고문헌·DOI 등 인용 포함 → VCR(출처 검증) 대상
  korean_prose  한국어 서술 중심 → SLM 도메인 적응·RAG 대상
  tabular       표/수치 중심 → 구조 인식·수치 검증 대상
  low_value     너무 짧거나 잡음 → 제외

원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_SHORT_LINE = re.compile(r"^\s*\S{0,12}\s*$")

# 연구노트 구조 단서
NOTE_MARKERS = [
    r"과제\s*(?:명|번호)", r"연구자|작성자", r"점검자|입회자|확인자",
    r"실험\s*일|기록\s*일", r"실험\s*목적", r"재료\s*(?:및|와)?\s*방법",
    r"실험\s*결과", r"고\s*찰|차기\s*계획",
]
# 인용 단서 — VCR 이 의미를 갖는 조건
CITE_MARKERS = [
    r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+",        # DOI
    r"references?|참고\s*문헌|인용\s*문헌",
    r"[A-Z][A-Za-z]+\s+et\s+al\.?,?\s*\(?\d{4}",   # Author et al. (YYYY)
    r"[가-힣]{2,4}\s*(?:외|등)\s*[,]?\s*\(?\d{4}",  # 홍길동 외 (2024)
    r"\[\d{1,3}\]",                                # [12] 번호 인용
]
_NOTE_RE = [re.compile(p) for p in NOTE_MARKERS]
_CITE_RE = [re.compile(p, re.IGNORECASE) for p in CITE_MARKERS]

MIN_CHARS = 200


def classify(text: str) -> tuple[str, int, int]:
    """(버킷, 노트단서수, 인용단서수)."""
    if len(text.strip()) < MIN_CHARS:
        return "low_value", 0, 0

    note_hits = sum(1 for r in _NOTE_RE if r.search(text))
    cite_hits = sum(1 for r in _CITE_RE if r.search(text))

    lines = text.splitlines() or [text]
    short_ratio = sum(1 for l in lines if _SHORT_LINE.match(l)) / len(lines)
    h, l = len(_HANGUL.findall(text)), len(_LATIN.findall(text))
    ko_ratio = h / max(1, h + l)

    if note_hits >= 4:
        return "labnote_like", note_hits, cite_hits
    if cite_hits >= 2:
        return "cited", note_hits, cite_hits
    if short_ratio > 0.6:
        return "tabular", note_hits, cite_hits
    if ko_ratio >= 0.4:
        return "korean_prose", note_hits, cite_hits
    return "low_value", note_hits, cite_hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    buckets: Counter[str] = Counter()
    by_ext: dict[str, Counter] = {}
    chars: Counter[str] = Counter()
    n = 0

    for row in iter_rows(Path(a.csv_path), limit=a.rows):
        n += 1
        b, _nh, _ch = classify(row.text)
        buckets[b] += 1
        chars[b] += len(row.text)
        by_ext.setdefault(row.ext, Counter())[b] += 1

    print("=" * 66)
    print(f" OCR 코퍼스 용도 판정 — 표본 {n:,}행")
    print("=" * 66)

    label = {
        "labnote_like": "연구노트형 → 기록 무결성 검사",
        "cited":        "인용 포함  → VCR 출처 검증",
        "korean_prose": "한국어 서술 → SLM 도메인 적응·RAG",
        "tabular":      "표/수치형   → 구조·수치 검증",
        "low_value":    "저가치     → 제외",
    }
    print(f"\n{'버킷':30s} {'행수':>9s} {'비율':>7s} {'전체추정':>11s}")
    print("-" * 62)
    EST_TOTAL = 6_520_029
    for b in ("labnote_like", "cited", "korean_prose", "tabular", "low_value"):
        c = buckets[b]
        pct = c / max(1, n)
        print(f"{label[b]:30s} {c:>9,} {pct*100:6.1f}% {int(pct*EST_TOTAL):>10,}행")

    print(f"\n[버킷별 평균 본문 길이]")
    for b in ("labnote_like", "cited", "korean_prose", "tabular"):
        if buckets[b]:
            print(f"  {b:14s} {chars[b]//buckets[b]:,}자")

    print(f"\n[확장자별 주요 버킷]")
    for ext, cc in sorted(by_ext.items(), key=lambda kv: -sum(kv[1].values()))[:6]:
        tot = sum(cc.values())
        top = ", ".join(f"{k} {v/tot*100:.0f}%" for k, v in cc.most_common(3))
        print(f"  {ext:10s} ({tot:,}행) → {top}")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "sampled_rows": n,
            "buckets": dict(buckets),
            "bucket_pct": {k: round(v / max(1, n), 4) for k, v in buckets.items()},
            "estimated_total_rows": EST_TOTAL,
            "estimated_by_bucket": {k: int(v / max(1, n) * EST_TOTAL) for k, v in buckets.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
