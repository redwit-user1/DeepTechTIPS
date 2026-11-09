#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""한국어 NLI 평가셋 **후보** 추출 — 라벨은 사람(또는 지정된 판정자)이 붙인다.

자동 생성 학습셋은 규칙으로 만든 라벨이라 규칙으로 맞힐 수 있었다(연구노트 2026-11-09).
평가셋은 라벨을 **의미 판단**으로 붙여야 순환이 끊긴다. 이 스크립트는 판정할 쌍만 뽑는다.

- 학습셋(문서 해시 % 8 == 0)과 겹치지 않게 다른 나머지(기본 3)를 쓴다.
- 주장은 한글 비중 50% 이상만 — 한국어 평가셋이다.
- 문서마다 (주장, 같은 문서 근거) 1쌍 + (주장, 어휘 겹치는 다른 문서 근거) 1쌍.
- 비식별화 강제, 출력은 data/ 안에서만(gitignore).

    python scripts/build_kr_nli_candidates.py ocr_goono_ocr_texts.csv --docs 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.claim_evidence import NoteBuffer, supports, words
from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.deidentify import audit, deidentify
from compliance_gateway.data.ocr.sensitive import sensitive_signals

_HANGUL = re.compile(r"[가-힣]")


def nid(s: str) -> int:
    return int.from_bytes(hashlib.blake2s(s.encode(), digest_size=7).digest(), "big")


def korean(s: str) -> bool:
    letters = sum(c.isalpha() for c in s)
    return letters > 0 and len(_HANGUL.findall(s)) / letters >= 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--docs", type=int, default=400, help="뽑을 문서 수(문서당 2쌍)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--residue", type=int, default=3, help="학습셋(0)과 다른 해시 나머지")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", default="data/labeling/kr_nli_candidates.jsonl")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if not str(out).startswith(str(Path.cwd().resolve() / "data")):
        sys.exit(f"거부: 출력은 data/ 안이어야 한다. 받은 값: {a.out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(a.seed)
    notes: dict[int, NoteBuffer] = {}
    bad: set[int] = set()          # 민감(건강) 행이 하나라도 있는 문서 — 통째로 뺀다
    for row in iter_rows(Path(a.csv_path)):
        key = nid(row.note_id)
        if key % a.stride == a.residue:
            if sensitive_signals(row.text):
                bad.add(key)
            notes.setdefault(key, NoteBuffer()).add(row.text)

    usable = [k for k, v in notes.items()
              if k not in bad and v.usable and any(korean(c) for c in v.claims)]
    rng.shuffle(usable)
    index: dict[str, list[tuple[int, int]]] = {}
    for k in usable:
        for j, e in enumerate(notes[k].evidence):
            for w in words(e):
                index.setdefault(w, []).append((k, j))

    stats: Counter[str] = Counter()
    rows = []
    for k in usable:
        if stats["docs"] >= a.docs:
            break
        nb = notes[k]
        claims = [c for c in nb.claims if korean(c)]
        rng.shuffle(claims)
        for claim in claims:
            # 중복 행(코퍼스 23%) 때문에 같은 문장이 주장·근거 양쪽에 들어올 수 있다 — 자기 짝짓기 제외
            same = [e for e in nb.evidence if e != claim and supports(claim, e)]
            hits: Counter[tuple[int, int]] = Counter()
            for w in words(claim):
                post = index.get(w, ())
                for kj in (post if len(post) <= 2000 else rng.sample(post, 2000)):
                    if kj[0] != k:
                        hits[kj] += 1
            other = [kj for kj, c in hits.items() if c >= 2 and notes[kj[0]].evidence[kj[1]] != claim]
            if not same or not other:
                continue
            ok, oj = rng.choice(other)
            pair = [("same_doc", rng.choice(same)), ("other_doc", notes[ok].evidence[oj])]
            dec = [(src, deidentify(claim).text, deidentify(e).text) for src, e in pair]
            if any(audit(c) or audit(e) for _, c, e in dec):
                stats["pii_dropped"] += 1
                continue
            for src, c, e in dec:
                rows.append({"id": f"{k:014x}-{src}", "doc": f"{k:014x}", "source": src,
                             "premise": e, "hypothesis": c})
            stats["docs"] += 1
            break

    rng.shuffle(rows)   # 판정자가 출처(같은/다른 문서)를 순서로 추측하지 못하게
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"후보 {len(rows)}쌍 / 문서 {stats['docs']} / PII 폐기 {stats['pii_dropped']} → {out}")


if __name__ == "__main__":
    main()
