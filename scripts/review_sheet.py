#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""사람 검수 시트 — Claude 라벨과 사람 판정의 일치도(κ)를 잰다.

평가셋 라벨은 Claude 가 작성·판정했다. 사람 판정과 일치하는지 재야 KPI 근거가 된다.
검수자는 Claude 라벨을 **보지 않는다**(정답 키는 별도 파일).

    python scripts/review_sheet.py make  [--n 100]   # data/labeling/review_100.csv (+ .key.json)
    python scripts/review_sheet.py score data/labeling/review_100.csv

시트는 엑셀용 UTF-8(BOM). human_label 에 entail / neutral / contradict / unusable 중 하나를 쓴다.
판정 기준(검수자 안내):
  entail     근거만 보고 문장이 참이라고 확인된다
  contradict 근거와 양립할 수 없다(수치·방향·대상·단위가 다름)
  neutral    근거로 확인도 반박도 안 된다(덧붙인 원인·일반화·확신 등)
  unusable   판정 불가(근거가 깨짐 등)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("entail", "neutral", "contradict", "unusable")
SRC = Path("data/labeling/kr_psg_eval.jsonl")


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's κ — 우연 일치를 뺀 일치도."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def make(n: int, seed: int) -> None:
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    rng = random.Random(seed)
    pick = []
    for kind, rs in sorted(by.items()):   # kind 별 균등 층화
        pick += rng.sample(rs, min(len(rs), n // len(by)))
    rng.shuffle(pick)
    out = SRC.parent / f"review_{n}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "premise(근거)", "hypothesis(문장)", "human_label", "memo"])
        for i, r in enumerate(pick, 1):
            w.writerow([i, r["premise"], r["hypothesis"], "", ""])
    key = {i: {"id": r["id"], "label": r["label"], "kind": r["kind"]} for i, r in enumerate(pick, 1)}
    out.with_suffix(".key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"시트 {len(pick)}행 → {out}\n정답 키(검수자에게 주지 말 것) → {out.with_suffix('.key.json')}")


def score(sheet: Path) -> None:
    key = json.loads(sheet.with_suffix(".key.json").read_text(encoding="utf-8"))
    human, claude, kinds = [], [], []
    with sheet.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            h = (row["human_label"] or "").strip().lower()
            if h not in LABELS:
                continue
            k = key[str(row["no"])]
            human.append(h); claude.append(k["label"]); kinds.append(k["kind"])
    if not human:
        raise SystemExit("판정된 행이 없다 — human_label 을 채울 것")
    agree = sum(h == c for h, c in zip(human, claude))
    print(f"판정 {len(human)}행  일치 {agree/len(human):.1%}  Cohen κ {kappa(human, claude):.3f}")
    print("kind 별 일치:")
    for kd in sorted(set(kinds)):
        idx = [i for i, x in enumerate(kinds) if x == kd]
        print(f"  {kd:12s} {sum(human[i] == claude[i] for i in idx)}/{len(idx)}")
    print("혼동(Claude → 사람):", dict(Counter(f"{c}→{h}" for h, c in zip(human, claude) if h != c)))
    print("판단: κ ≥ 0.8 이면 이 평가셋을 KPI 근거로 쓸 수 있다. 0.6 미만이면 라벨 재작업.")


def _demo() -> None:
    assert abs(kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) - 1.0) < 1e-9
    assert kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0   # 우연 수준
    print("review_sheet demo ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--n", type=int, default=100); m.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score"); s.add_argument("sheet", type=Path)
    sub.add_parser("demo")
    a = ap.parse_args()
    {"make": lambda: make(a.n, a.seed), "score": lambda: score(a.sheet), "demo": _demo}[a.cmd]()
