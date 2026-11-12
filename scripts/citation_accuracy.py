#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 생성 콘텐츠 출처 정확률(KPI 90%+) — WF-A 초안의 DOI 인용을 검사한다.

정확 = 인용한 DOI 가 (1) 에이전트가 실제로 찾은 근거 목록에 있고 (2) Crossref 에서 실재가 확인된다.
(1) 만 보면 실재하지 않는 근거를 근거로 받아도 통과하고, (2) 만 보면 실재하지만 찾지 않은(지어낸) 인용이 통과한다.

    python scripts/citation_accuracy.py data/agent/wf_a_*.jsonl [--offline]

한계: 인용이 그 문장을 뒷받침하는지(내용 일치)는 여기서 보지 않는다 — 그건 게이트웨이 SourceMatch 몫.
계획서 정의(ScienceON·NTIS 대조)와 출처 DB 가 다르다.
"""

from __future__ import annotations

import argparse
import json

from compliance_gateway.verify.backends import CrossRefBackend


def score(rows: list[dict], resolve) -> dict:
    total = in_src = real = both = 0
    for r in rows:
        known = {d.lower() for d in r.get("source_dois", [])}
        for doi in r.get("cited_dois", []):
            total += 1
            a = doi.lower() in known
            b = resolve(doi)
            in_src += a
            real += b
            both += a and b
    return {"cited": total, "in_sources": in_src, "resolvable": real, "accurate": both,
            "accuracy": both / total if total else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--offline", action="store_true", help="실재 조회 생략(근거 목록 포함 여부만)")
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.files for l in open(f, encoding="utf-8") if l.strip()]
    cr = CrossRefBackend()
    m = score(rows, (lambda d: True) if a.offline else (lambda d: cr.by_doi(d) is not None))
    print(json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
