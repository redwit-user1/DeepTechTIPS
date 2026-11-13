#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KCI 국문 초록 → Instruction 생성용 원문 (GPU 불필요).

연구노트 단락은 짧다(중앙값 158자) — '보고서 문단' 과제가 문체 바꿔 쓰기에 그친다(연구노트 §38).
KCI 초록은 중앙값 약 1,000자라 여러 사실을 묶어 써야 하고, 공개 자료라 학습 데이터를 밖에 내놓을 수 있다.

    python scripts/build_kci_passages.py --corpus ~/research-data-analysis/data/corpus/kci --out data/kci --shards 2

- 이공·의약·농수해양 분야만(인문·사회·예술 초록은 실험 조건 추출 과제가 성립하지 않는다)
- 국문 초록 400~1,500자, 한글이 공백 제외 글자의 절반 이상
- 초록이 같은 글은 한 번만. 순서는 sha256(id) — 달·분야에 치우치지 않게 섞는다
- 출력: passages_{0..n-1}.jsonl  {"doc": "kci<id>", "passage": 초록, "category", "year"}
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

STEM = {
    "공학", "전자/정보통신공학", "컴퓨터학", "토목공학", "전기공학", "기계공학", "재료공학", "안전공학", "건축공학", "금속공학",
    "화학공학", "환경공학", "산업공학", "원자력공학", "조선공학", "항공우주공학", "자원공학", "고분자공학", "생물공학", "의공학",
    "자연과학", "자연과학일반", "기타자연과학", "화학", "생물학", "물리학", "수학", "통계학", "지구과학", "지질학", "천문학", "대기과학", "해양학",
    "의약학", "기타의약학", "내과학", "일반외과학", "산부인과학", "치의학", "한의학", "약학", "간호학", "수의학", "방사선과학", "물리치료학",
    "예방의학", "정신과학", "소아과학", "피부과학", "안과학", "이비인후과학", "신경과학", "임상병리학", "재활의학", "면역학", "병리학",
    "농수해양", "농수해양학", "농학", "임학", "축산학", "수산학", "식품과학", "원예학", "조경학", "생활과학",
}


def usable(abstract: str) -> bool:
    if not 400 <= len(abstract) <= 1500:
        return False
    body = re.sub(r"\s", "", abstract)
    return len(re.findall(r"[가-힣]", body)) >= 0.5 * len(body)


def collect(corpus: str) -> tuple[list[dict], Counter]:
    seen, rows, why = set(), [], Counter()
    for f in sorted(glob.glob(str(Path(corpus).expanduser() / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                why["깨진 줄"] += 1
                continue
            a = re.sub(r"\s+", " ", r.get("abstract") or "").strip()
            if r.get("category") not in STEM:
                why["분야 제외"] += 1
            elif not usable(a):
                why["길이·언어 제외"] += 1
            elif (key := hashlib.sha256(re.sub(r"\s", "", a).encode("utf-8")).hexdigest()) in seen:
                why["중복"] += 1
            else:
                seen.add(key)
                why["사용"] += 1
                rows.append({"doc": f"kci{r['id']}", "passage": a, "category": r["category"], "year": r.get("year", "")})
    rows.sort(key=lambda r: hashlib.sha256(r["doc"].encode("utf-8")).hexdigest())
    return rows, why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows, why = collect(a.corpus)
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        raise SystemExit(f"거부: 쓸 수 있는 초록이 0건이다 — {dict(why)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(a.shards):
        with open(out / f"passages_{k}.jsonl", "w", encoding="utf-8") as f:
            for r in rows[k:: a.shards]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    L = sorted(len(r["passage"]) for r in rows)
    print(f"{dict(why)} → {len(rows)}건 · {a.shards}조각 · 길이 중앙값 {L[len(L) // 2]}자 · "
          f"분야 {Counter(r['category'] for r in rows).most_common(8)}")


if __name__ == "__main__":
    main()
