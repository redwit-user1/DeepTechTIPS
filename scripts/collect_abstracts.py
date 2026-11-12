"""초록 수집 — KoreaScience(국문) / arXiv(영문) → JSONL.

    python scripts/collect_abstracts.py koreascience 임상시험 GLP --pages 5
    python scripts/collect_abstracts.py arxiv 'all:"Seoul National University" AND cat:q-bio*' --pages 3

출력: data/abstracts/<source>.jsonl (이어쓰기, id 중복 제거).
용도: 외부 한국어 평가셋·RAG. 학습 투입 전 라이선스 확인(sources.py docstring).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from compliance_gateway.data.korean.sources import ArxivAdapter, KoreaScienceAdapter

DELAY = {"koreascience": 2.0, "arxiv": 3.0}   # arXiv 약관 3초 1회. KS 는 명시 없음 → 보수적
PAGE = {"koreascience": 10, "arxiv": 50}


def fetch(source: str, query: str, page: int) -> list[dict]:
    if source == "koreascience":
        return KoreaScienceAdapter().search(query, page=page, size=PAGE[source])
    rows = ArxivAdapter().search(query, max_results=PAGE[source], start=page * PAGE[source])
    return [{"id": r["arxiv_id"], "url": r["id"], "title": r["title"], "abstract": r["summary"],
             "authors": r["authors"], "year": r["published"][:4], "doi": ""} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=list(DELAY))
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    out = a.out or Path(f"data/abstracts/{a.source}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()

    added = 0
    with out.open("a", encoding="utf-8") as f:
        for q in a.queries:
            for page in range(a.pages):
                rows = fetch(a.source, q, page)
                time.sleep(DELAY[a.source])
                for r in rows:
                    if r["id"] in seen or not r["abstract"]:
                        continue
                    seen.add(r["id"])
                    r.update(source=a.source, query=q, fetched=date.today().isoformat())
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    added += 1
                print(f"{q!r} p{page}: {len(rows)}건 → 누적 신규 {added}")
                if len(rows) < PAGE[a.source]:
                    break   # 마지막 페이지
    print(f"{out}: +{added} (총 {len(seen)})")


if __name__ == "__main__":
    main()
