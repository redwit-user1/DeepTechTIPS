#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""사람 검수용 라벨셋 500건을 만든다 — 모든 후속 작업의 선행 조건.

## 왜 필요한가

지금까지 낸 분류 수치(연구 내용 46.5%, 생명·보건의료 22.7%, 논문 혼입 11.4%)는
전부 **"분류기가 어떻게 나눴는지"이지 정확도가 아니다.** 정답셋이 0건이기 때문이다.
그리고 근거-주장 쌍은 버킷 분류가 맞다는 전제 위에 서 있으므로, 분류가 틀리면
만들어지는 학습 데이터가 전부 쓰레기가 된다.

## 이 스크립트만 원문을 담는다

다른 모든 분석 스크립트는 통계만 출력한다. 이건 사람이 읽고 판정해야 하므로
발췌를 담을 수밖에 없다. 대신 세 겹으로 막는다:

1. **비식별화 강제** — 끌 수 있는 옵션이 없다. 잔여 PII 는 검사해서 보고한다.
2. **출력 경로 강제** — `data/` 밖으로는 쓰지 못한다(`data/` 는 gitignore 대상).
3. **파일명 미포함** — 확장자만 남긴다. 파일명은 그 자체로 식별 위험이다.

## 표본 설계

버킷별 **층화 추출**. 비율대로 뽑으면 조각(38.4%)이 표본의 3분의 1을 먹고
문헌·인용(2.2%)은 11건밖에 안 나와서 그 버킷의 정확도를 못 잰다.
각 버킷에서 최소 30건씩 뽑아 **버킷별로** 정확도를 낼 수 있게 한다.
전체 정확도가 필요하면 실제 비율로 가중평균하면 된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.deidentify import audit, deidentify
from compliance_gateway.data.ocr.gate import Gate, bounded_violations
from classify_domain import classify_domain
from triage_ocr_content import classify_content

# 버킷별 최소 할당. 합계가 목표치보다 작으면 남는 만큼 실제 비율로 채운다.
QUOTA = {
    "experimental": 90, "analysis": 70, "literature": 50, "planning": 40,
    "code_source": 30, "code_commit": 30, "code_meta": 30, "code_docs": 30,
    "fragment": 90, "admin": 30,
}
EXCERPT = 1200          # 사람이 판정하기에 충분한 분량. 전문은 담지 않는다
RESERVOIR = 40          # 버킷당 후보 저수지 배수 — 앞부분 편향을 줄인다

BANNER = (
    "# GOONO OCR 분류 검수 라벨셋 — 대외 반출 금지\n"
    "# 비식별화 적용됨. 원본 파일명 미포함. 발췌는 앞 1,200자.\n"
    "# 판정: label 칸에 올바른 버킷명을 적는다. 예측이 맞으면 'ok'.\n"
    "# 버킷: experimental analysis literature planning\n"
    "#       code_source code_commit code_meta code_docs fragment admin\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--scan", type=int, default=400_000,
                    help="후보를 고를 스캔 범위. 넓을수록 편향이 준다")
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--out", default="data/labeling/label_set_500.csv")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    root = Path.cwd().resolve()
    if not str(out).startswith(str(root / "data")):
        sys.exit(f"거부: 출력은 data/ 안이어야 한다(gitignore 대상). 받은 값: {a.out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(a.seed)
    gate = Gate()
    pools: dict[str, list] = {}
    seen_by_bucket: Counter[str] = Counter()

    # 저수지 표본 — 각 버킷의 후보를 스캔 범위 전체에 걸쳐 균등하게 모은다
    for row, _viol in gate.filter(iter_rows(Path(a.csv_path), limit=a.scan)):
        b, _ = classify_content(row.text)
        seen_by_bucket[b] += 1
        cap = QUOTA.get(b, 30) * RESERVOIR
        pool = pools.setdefault(b, [])
        if len(pool) < cap:
            pool.append(row)
        else:
            j = rng.randrange(seen_by_bucket[b])
            if j < cap:
                pool[j] = row

    # 할당량만큼 뽑되, 부족한 버킷의 몫은 큰 버킷으로 넘긴다
    picked: list[tuple[str, object]] = []
    for b, q in QUOTA.items():
        pool = pools.get(b, [])
        take = min(q, len(pool))
        for row in rng.sample(pool, take):
            picked.append((b, row))
    short = a.n - len(picked)
    if short > 0:
        rest = [(b, r) for b, pool in pools.items() for r in pool
                if (b, r) not in picked]
        rng.shuffle(rest)
        picked.extend(rest[:short])
    picked = picked[:a.n]
    rng.shuffle(picked)

    residual = 0
    counts: Counter[str] = Counter()
    dom_counts: Counter[str] = Counter()
    rows_out = []
    for i, (bucket, row) in enumerate(picked, 1):
        res = deidentify(row.text[:EXCERPT])
        left = audit(res.text)
        if left:
            residual += 1
        dom, _ = classify_domain(row.text)
        counts[bucket] += 1
        dom_counts[dom] += 1
        rows_out.append({
            "no": i,
            "pred_bucket": bucket,
            "pred_domain": dom,
            "ext": Path(row.file_name).suffix.lower() or "(없음)",
            "chars": len(row.text),
            "bounded_violation": "|".join(bounded_violations(row.text)),
            "excerpt": res.text,
            "label": "",            # 사람이 채운다
            "domain_label": "",     # 사람이 채운다
            "note": "",
        })

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        f.write(BANNER)
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "n": len(rows_out), "seed": a.seed, "scan": a.scan,
        "excerpt_chars": EXCERPT,
        "bucket_counts": dict(counts), "domain_counts": dict(dom_counts),
        "residual_pii_rows": residual,
        "gate_stats": gate.stats.as_dict(),
        "quota": QUOTA,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f" 라벨셋 {len(rows_out)}건 생성 — {out}")
    print("=" * 66)
    print(f"\n[게이트] {gate.stats.as_dict()}")
    print("\n[버킷별 할당]")
    for b, c in counts.most_common():
        print(f"  {b:14s} {c:>4}건  (스캔에서 관측 {seen_by_bucket[b]:,})")
    print("\n[분야 분포]")
    for d, c in dom_counts.most_common(6):
        print(f"  {d:14s} {c:>4}건")
    print(f"\n[비식별화] 잔여 PII 행 {residual}건", "← 0이어야 한다" if residual else "OK")
    print(f"\n메타: {meta}")
    print("\n다음: label / domain_label 칸을 사람이 채운 뒤")
    print("      python scripts/score_label_set.py <이 파일>")


if __name__ == "__main__":
    main()
