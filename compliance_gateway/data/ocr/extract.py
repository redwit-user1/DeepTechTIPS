# -*- coding: utf-8 -*-
"""OCR 코퍼스 부분집합 추출 — 용도별로 나눠서 뽑는다.

판정 결과(`scripts/triage_ocr_csv.py`) 이 코퍼스는 단일 용도가 아니었다.
따라서 버킷별로 추출해 각기 다른 파이프라인에 넣는다.

| 버킷 | 추정 규모 | 쓰임 |
|---|---|---|
| `cited` | 약 3.4만행 | **한국어 VCR 평가셋** — 최대 공백이던 항목 |
| `korean_prose` | 약 153만행 | SLM 도메인 적응·RAG |
| `labnote_like` | 약 4천행 | 기록 무결성 검사 검수 표본 |
| `tabular` | 약 131만행 | 표·수치 검증 |

추출 시 **비식별화가 항상 선행**된다(끄는 옵션 없음).

실행:
  python -m compliance_gateway.data.ocr.extract ocr_goono_ocr_texts.csv \\
      --bucket cited --limit 2000 --out data/real/cited.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.deidentify import audit, deidentify

# 판정 로직은 triage 스크립트와 공유한다(중복 정의 방지)
import sys

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from triage_ocr_csv import classify  # noqa: E402

BUCKETS = ("cited", "korean_prose", "labnote_like", "tabular")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--bucket", choices=BUCKETS, required=True)
    ap.add_argument("--limit", type=int, default=1000, help="추출할 최대 건수")
    ap.add_argument("--scan", type=int, default=None, help="스캔할 최대 행 수")
    ap.add_argument("--out", default=None, help="JSONL 저장 경로(미지정 시 dry-run)")
    ap.add_argument("--min-chars", type=int, default=200)
    a = ap.parse_args()

    salt = os.getenv("GOONO_DEID_SALT") or "goono-deid-v1"
    if salt == "goono-deid-v1":
        print("⚠️  기본 salt 사용 중 — 운영에서는 GOONO_DEID_SALT 를 설정하세요.\n")

    out_rows: list[dict] = []
    deid_total: Counter[str] = Counter()
    residual: Counter[str] = Counter()
    scanned = 0
    lens: list[int] = []

    for row in iter_rows(Path(a.csv_path), limit=a.scan):
        scanned += 1
        bucket, note_hits, cite_hits = classify(row.text)
        if bucket != a.bucket or len(row.text) < a.min_chars:
            continue

        d = deidentify(row.text, salt=salt)
        deid_total.update(d.counts)
        residual.update(audit(d.text))
        lens.append(len(d.text))

        out_rows.append({
            "id": row.id,
            "note_id": row.note_id,
            "source_ext": row.ext,          # 파일명 자체는 저장하지 않는다(식별 위험)
            "page": row.page,
            "bucket": bucket,
            "note_markers": note_hits,
            "cite_markers": cite_hits,
            "text": d.text,
            "deidentified": True,
            "lang": "ko",
        })
        if len(out_rows) >= a.limit:
            break

    print(f"스캔 {scanned:,}행 → '{a.bucket}' {len(out_rows):,}건 추출")
    if lens:
        lens.sort()
        print(f"  본문 길이: 중앙값 {lens[len(lens)//2]:,}자 / 최대 {lens[-1]:,}자")
    print(f"  비식별화: {sum(deid_total.values()):,}건 치환 {dict(deid_total)}")
    print(f"  잔존 위험: {dict(residual) if residual else '없음'}")

    if a.out and out_rows:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n→ {p} ({len(out_rows):,}건, 비식별화 완료)")
    elif out_rows:
        print("\n(dry-run — 저장하려면 --out)")


if __name__ == "__main__":
    main()
