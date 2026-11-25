#!/usr/bin/env python3
"""제안서 과제 학습 데이터용 주제 목록 — KCI 이공 · 의약 초록의 국문 제목(주제)과 영문 제목(OpenAlex 검색어).

평가셋(heldout)에 든 문서와 WF-A 평가 주제는 뺀다. 주제만 가져오고 근거는 OpenAlex 검색 결과를 쓴다
(WF-A 와 같은 입력 분포: 국문 주제 + 영문 초록 5건).

    python scripts/proposal_topics.py --passages data/kci/passages_0.jsonl data/kci/passages_1.jsonl \
        --corpus ../data/corpus/kci --exclude data/kci/heldout.jsonl --n 900 --out ../data/prop_topics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", nargs="+", required=True)
    ap.add_argument("--corpus", required=True, help="KCI 원본 jsonl 폴더(id, title, title_en)")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = set()
    for p in a.passages:
        for line in open(p, encoding="utf-8"):
            d = json.loads(line).get("doc", "")
            if d.startswith("kci"):
                ids.add(d[3:])
    for p in a.exclude:
        for line in open(p, encoding="utf-8"):
            d = json.loads(line).get("doc", "")
            ids.discard(d[3:] if d.startswith("kci") else d)

    meta = {}
    for f in sorted(glob.glob(os.path.join(a.corpus, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("id") in ids and r.get("title") and r.get("title_en"):
                meta[r["id"]] = (r["title"].strip(), r["title_en"].strip())

    rows = []
    for i, (ko, en) in meta.items():
        en = re.sub(r"\s+", " ", re.sub(r"[^\w\s\-]", " ", en)).strip()
        ko = re.sub(r"\s+", " ", ko)
        if 3 <= len(en.split()) <= 25 and 8 <= len(ko) <= 80:
            rows.append((i, ko, " ".join(en.split()[:12])))
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    json.dump([[ko, en, i] for i, ko, en in rows], open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"후보 {len(ids)} · 제목 있음 {len(meta)} · 저장 {len(rows)} → {a.out}")


if __name__ == "__main__":
    main()
