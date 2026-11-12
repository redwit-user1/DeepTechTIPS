#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전 구간 단순 무작위 표본 — 비율 추정 전용.

## 왜 층화를 쓰지 않는가

1차 라벨셋은 **버킷 균등 층화**였다. 버킷별 정확도를 재려면 그게 맞다.
그런데 전체 비율을 보정하는 데 쓰자 문제가 드러났다 — 코퍼스 질량이
균등하지 않아 판정 1건의 영향이 버킷마다 20배 차이 났다
(fragment 1.20p/건 vs literature 0.09p/건).

**단순 무작위 표본은 자동으로 질량 비례다.** 혼동행렬 역산이 아예 필요 없고
`P(gold)` 가 직접 나온다. 표준오차도 `sqrt(p(1-p)/n)` 로 단순하다.

두 표본이 두 목적을 나눠 갖는다:

| 표본 | 설계 | 용도 |
|---|---|---|
| 1차 500건 | 버킷 균등 층화 | 버킷별 재현율·정밀도 |
| 2차 (이 파일) | **전 구간 무작위** | **코퍼스 비율** |

## 게이트를 걸지 않는다

전 구간 재판정(`retriage_full.py`)이 게이트 없이 돌았으므로, 비교 가능하려면
여기서도 걸지 않는다. 게이트 통과분의 비율이 필요하면 별도로 뽑아야 한다.

저수지 표본(reservoir sampling)으로 한 번만 훑는다. 원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.deidentify import audit, deidentify
from classify_domain import classify_domain
from triage_ocr_content import classify_content

EXCERPT = 900
_WS = re.compile(r"\s+")

BANNER = (
    "# GOONO OCR 무작위 표본 — 대외 반출 금지\n"
    "# 전 구간 단순 무작위. 비율 추정 전용(층화 아님).\n"
    "# 비식별화 적용됨. 원본 파일명 미포함.\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", default="data/labeling/random_sample.csv")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if not str(out).startswith(str(Path.cwd().resolve() / "data")):
        sys.exit(f"거부: 출력은 data/ 안이어야 한다. 받은 값: {a.out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(a.seed)
    pool: list = []
    n = 0
    for row in iter_rows(Path(a.csv_path)):
        n += 1
        if len(pool) < a.n:
            pool.append(row)
        else:
            j = rng.randrange(n)
            if j < a.n:
                pool[j] = row

    residual = 0
    pred: Counter[str] = Counter()
    rows_out = []
    for i, row in enumerate(sorted(pool, key=lambda r: r.id), 1):
        b, _ = classify_content(row.text)
        d, _ = classify_domain(row.text)
        pred[b] += 1
        res = deidentify(row.text[:EXCERPT])
        if audit(res.text):
            residual += 1
        rows_out.append({
            "no": i, "pred_bucket": b, "pred_domain": d,
            "ext": Path(row.file_name).suffix.lower() or "(없음)",
            "chars": len(row.text),
            "excerpt": res.text,
            "model_label": "", "label": "", "note": "",
        })

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        f.write(BANNER)
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)

    # 판정용 압축 뷰(사람/모델이 읽는 용도)
    view = out.with_name(out.stem + "_view.txt")
    with view.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(f"[{r['no']}|{r['pred_bucket']}|{r['ext']}|{r['chars']}] "
                    f"{_WS.sub(' ', r['excerpt'])[:300]}\n")

    out.with_suffix(".meta.json").write_text(json.dumps({
        "n": len(rows_out), "scanned_rows": n, "seed": a.seed,
        "design": "simple random (reservoir), 전 구간, 게이트 미적용",
        "pred_bucket": dict(pred),
        "pred_bucket_pct": {k: round(v / len(rows_out), 4) for k, v in pred.items()},
        "residual_pii_rows": residual, "excerpt_chars": EXCERPT,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f" 무작위 표본 {len(rows_out)}건 / 전수 {n:,}행")
    print("=" * 66)
    print("\n[예측 버킷 분포] 전 구간 재판정치와 맞아야 표본이 건전하다")
    for b, c in pred.most_common():
        print(f"  {b:14s} {c:>4}건 ({c/len(rows_out)*100:5.2f}%)")
    print(f"\n[비식별화] 잔여 PII 행 {residual}건", "← 0이어야 한다" if residual else "OK")
    print(f"\n→ {out}\n→ {view}")


if __name__ == "__main__":
    main()
