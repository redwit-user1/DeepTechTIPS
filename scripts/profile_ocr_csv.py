#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOONO OCR CSV 프로파일러 — **원문을 출력하지 않고** 구성만 파악한다.

실제 고객 연구데이터이므로 본문·파일명을 그대로 찍지 않는다.
통계와 분포만 산출하므로 결과를 공유해도 비교적 안전하다.

  python scripts/profile_ocr_csv.py ocr_goono_ocr_texts.csv --rows 20000
  python scripts/profile_ocr_csv.py ocr_goono_ocr_texts.csv --rows 20000 --json p.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows

# 연구노트 구조 단서 (한국어 지침 용어)
NOTE_MARKERS = {
    "과제": r"과제\s*(?:명|번호|고유번호)",
    "연구자/서명": r"연구자|작성자|서\s*명",
    "점검자": r"점검자|입회자|확인자",
    "일자": r"실험\s*일|기록\s*일|작성\s*일|\d{4}[-./]\d{1,2}[-./]\d{1,2}",
    "실험목적": r"실험\s*목적|연구\s*목적|목\s*적",
    "재료·방법": r"재료\s*(?:및|와)?\s*방법|실험\s*방법",
    "결과": r"실험\s*결과|결\s*과",
    "고찰": r"고\s*찰|결\s*론|차기\s*계획",
}
OCR_NOISE = {
    "치환문자(�)": r"�",
    "고립자모": r"[ㄱ-ㅎㅏ-ㅣ]{2,}",
    "숫자-문자 혼동": r"\b(?=[0-9O]*O)(?=[0-9O]*[0-9])[0-9O]{3,}\b",
    "과다공백": r"[ \t]{6,}",
}
# 위험 신호 — **건수만** 세고 값은 저장하지 않는다
PII_PATTERNS = {
    "주민등록번호형": r"\d{6}\s*[-–]\s*[1-4]\d{6}",
    "휴대전화": r"01[016789][-. ]?\d{3,4}[-. ]?\d{4}",
    "이메일": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "역할+인명": r"(?:연구자|작성자|점검자|성명)\s*[:：]?\s*[가-힣]{2,4}",
}

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
# 표/스프레드시트형 본문 — 짧은 줄이 대량 반복
_SHORT_LINE = re.compile(r"^\s*\S{0,12}\s*$")


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=20000, help="표본 행 수")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    path = Path(a.csv_path)
    if not path.exists():
        raise SystemExit(f"경로 없음: {path}")

    total_bytes = path.stat().st_size
    print("=" * 66)
    print(f" GOONO OCR CSV 프로파일: {path.name}")
    print("=" * 66)
    print(f"\n파일 크기 {human(total_bytes)}  |  표본 {a.rows:,}행\n")

    ext_c: Counter[str] = Counter()
    marker_c: Counter[str] = Counter()
    noise_c: Counter[str] = Counter()
    pii_c: Counter[str] = Counter()
    pages_per_note: Counter[str] = Counter()
    lens: list[int] = []
    hangul = latin = 0
    empty = 0
    tabular = 0
    n = 0
    sampled_bytes = 0

    for row in iter_rows(path, limit=a.rows):
        n += 1
        t = row.text
        sampled_bytes += len(t.encode("utf-8", "ignore"))
        ext_c[row.ext] += 1
        pages_per_note[row.note_id] += 1
        if not t.strip():
            empty += 1
            continue
        lens.append(len(t))
        hangul += len(_HANGUL.findall(t))
        latin += len(_LATIN.findall(t))

        lines = t.splitlines()
        if lines and sum(1 for l in lines if _SHORT_LINE.match(l)) / len(lines) > 0.6:
            tabular += 1

        for k, p in NOTE_MARKERS.items():
            if re.search(p, t):
                marker_c[k] += 1
        for k, p in OCR_NOISE.items():
            if re.search(p, t):
                noise_c[k] += 1
        for k, p in PII_PATTERNS.items():
            c = len(re.findall(p, t))
            if c:
                pii_c[k] += c

    filled = max(1, n - empty)
    avg_row = sampled_bytes / max(1, n)
    est_rows = int(total_bytes / avg_row) if avg_row else 0

    print(f"[규모 추정] 표본 평균 행 크기 {human(avg_row)} → 전체 약 {est_rows:,}행")
    print(f"           고유 NoteId(표본) {len(pages_per_note):,}건, "
          f"노트당 평균 {n/max(1,len(pages_per_note)):.1f}페이지")
    print(f"           빈 본문 {empty:,}행 ({empty/max(1,n)*100:.1f}%)")

    if lens:
        lens.sort()
        print(f"\n[본문 길이] 중앙값 {lens[len(lens)//2]:,}자 / "
              f"평균 {statistics.fmean(lens):,.0f}자 / 최대 {lens[-1]:,}자")
        ratio = hangul / max(1, hangul + latin)
        print(f"[언어] 한글 비율 {ratio:.1%} → "
              f"{'한국어 중심' if ratio > 0.5 else '영문 중심' if ratio < 0.2 else '혼합'}")
        print(f"[형태] 표/스프레드시트형 추정 {tabular:,}행 ({tabular/filled*100:.1f}%)")

    print("\n[원본 파일 확장자] — 무엇이 업로드됐는가")
    for ext, c in ext_c.most_common(12):
        print(f"  {ext:12s} {c:>7,}행 ({c/max(1,n)*100:5.1f}%)")

    print("\n[연구노트 구조 단서] (본문 있는 행 대비)")
    for k in NOTE_MARKERS:
        c = marker_c[k]
        print(f"  {k:12s} {c/filled*100:5.1f}% {'█' * int(c/filled*20)}")
    struct = sum(marker_c.values()) / (filled * len(NOTE_MARKERS))
    print(f"  → 구조성 지수 {struct:.2f}")

    print("\n[OCR 품질 신호]")
    for k in OCR_NOISE:
        print(f"  {k:16s} {noise_c[k]/filled*100:5.1f}%")

    print("\n[개인정보 위험] ⚠️ 건수만 — 값은 수집하지 않음")
    if pii_c:
        for k, c in pii_c.most_common():
            print(f"  {k:16s} {c:,}건 (표본 {n:,}행 기준)")
        print("  → 학습·공유 전 비식별화 필수")
    else:
        print("  표본에서 검출 없음")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "file": path.name, "total_bytes": total_bytes,
            "sampled_rows": n, "estimated_rows": est_rows,
            "unique_notes_in_sample": len(pages_per_note),
            "empty_rows": empty, "tabular_rows": tabular,
            "hangul_ratio": round(hangul / max(1, hangul + latin), 4),
            "ext": dict(ext_c.most_common(20)),
            "structure_index": round(struct, 3),
            "note_markers_pct": {k: round(marker_c[k] / filled, 3) for k in NOTE_MARKERS},
            "ocr_noise_pct": {k: round(noise_c[k] / filled, 3) for k in OCR_NOISE},
            "pii_counts": dict(pii_c),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {a.json} 저장 (원문 미포함)")


if __name__ == "__main__":
    main()
