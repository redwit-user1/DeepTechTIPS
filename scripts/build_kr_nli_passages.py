#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""한국어 NLI 평가셋 v2 — **원문 단락 + 그 안의 수치 결과 문장** 추출.

v1(같은 문서의 근거 문장 ↔ 주장) 은 의미 판정 결과 뒷받침이 160쌍 중 1쌍(0.6%)이었다.
같은 문서라도 근거 문장은 대개 다른 단계(조건·캡션·계획)를 말한다. 그래서 평가 단위를
제품의 실제 상황으로 바꾼다: **원문 단락(premise) → LLM 이 옮겨 쓴 문장(hypothesis)**.
이 스크립트는 단락과 그 안의 결과 문장만 뽑고, 가설 문장 작성·판정은 판정자가 한다.

필터(v1 라벨러 지적 반영): 목적·계획 문장, 커밋 로그, 문장 중간에서 잘린 것 제외.

    python scripts/build_kr_nli_passages.py ocr_goono_ocr_texts.csv --docs 150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from compliance_gateway.data.ocr.claim_evidence import NUM_UNIT, NOT_CLAIM, is_claim, sentences
from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.deidentify import audit, deidentify

_HANGUL = re.compile(r"[가-힣]")
PLAN = re.compile(r"목적|목표|계획|예정|하고자|위하여|위해서|할 것|제안|요청|필요하다")
ENDED = re.compile(r"(?:다|음|함|됨|임|였음|었음)\s*\.?\s*$")   # 문장이 끝까지 있는가
COMMIT = re.compile(r"\bcommit\b|github\.com|merge|pull request", re.IGNORECASE)


def nid(s: str) -> int:
    return int.from_bytes(hashlib.blake2s(s.encode(), digest_size=7).digest(), "big")


def korean(s: str, th: float = 0.5) -> bool:
    letters = sum(c.isalpha() for c in s)
    return letters > 0 and len(_HANGUL.findall(s)) / letters >= th


def result_sentence(s: str) -> bool:
    return (is_claim(s) and korean(s) and bool(NUM_UNIT.search(s)) and bool(ENDED.search(s))
            and not PLAN.search(s) and not COMMIT.search(s) and not NOT_CLAIM.search(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--docs", type=int, default=150)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--residue", type=int, default=5, help="학습셋(0)·v1 후보(3)와 다른 나머지")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", default="data/labeling/kr_nli_passages.jsonl")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if not str(out).startswith(str(Path.cwd().resolve() / "data")):
        sys.exit(f"거부: 출력은 data/ 안이어야 한다. 받은 값: {a.out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    found: dict[int, tuple[str, str]] = {}
    for row in iter_rows(Path(a.csv_path)):
        key = nid(row.note_id)
        if key % a.stride != a.residue or key in found:
            continue
        ss = sentences(row.text)
        for i, s in enumerate(ss):
            if result_sentence(s):
                passage = " ".join(ss[max(0, i - 2): i + 2])
                if 80 <= len(passage) <= 700 and korean(passage, 0.4):
                    found[key] = (passage, s)
                break

    rng = random.Random(a.seed)
    keys = list(found)
    rng.shuffle(keys)
    n = drop = 0
    with out.open("w", encoding="utf-8") as f:
        for k in keys:
            if n >= a.docs:
                break
            p, c = (deidentify(x).text for x in found[k])
            if audit(p) or audit(c):
                drop += 1
                continue
            f.write(json.dumps({"id": f"{k:014x}", "doc": f"{k:014x}", "passage": p, "claim": c},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"후보 문서 {len(found):,} → 단락 {n} (PII 폐기 {drop}) → {out}")


if __name__ == "__main__":
    main()
