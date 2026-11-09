#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""판정 캐스케이드 시뮬레이션 — 규칙 → 인코더 → (애매 구간만) LLM.

    0단계  unit_swaps (값 같고 단위만 바뀜)                → 위반
    1단계  인코더 점수 s:  s < lo → 위반,  s ≥ hi → 통과
    2단계  lo ≤ s < hi 만 LLM 판정: q < θ → 위반

행별 점수(`kr_nli --dump`)를 읽어 정상 통과율 ≥ 90% 를 지키는 최대 F1 구성을 찾는다.
**임계값을 같은 평가셋에서 고르므로 낙관적이다** — 홀드아웃 확인은 단락 2분할로 한다.

    python scripts/cascade_sim.py --scores <dir> --enc klue
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compliance_gateway.vcr.hallucination import unit_swaps

SETS = ("kr_psg_eval", "kr_hard_eval")


def load(scores: Path, enc: str, llm: str = "qwen") -> list[dict]:
    rows = []
    for s in SETS:
        src = {r["id"]: r for r in map(json.loads, open(f"data/labeling/{s}.jsonl", encoding="utf-8"))}
        e = {r["id"]: r["score"] for r in map(json.loads, open(scores / f"{s}_{enc}.jsonl", encoding="utf-8"))}
        q = {r["id"]: r["score"] for r in map(json.loads, open(scores / f"{s}_{llm}.jsonl", encoding="utf-8"))}
        for i, r in src.items():
            rows.append({"set": s, "doc": r["doc"], "kind": r["kind"], "y": r["label"] != "entail",
                         "rule": bool(unit_swaps(r["hypothesis"], r["premise"])), "e": e[i], "q": q[i]})
    return rows


def run(rows, lo, hi, th, use_rule=True):
    pred, calls = [], 0
    for r in rows:
        if use_rule and r["rule"]:
            pred.append(True); continue
        if lo is None:                       # LLM 단독
            calls += 1; pred.append(r["q"] < th); continue
        if hi is None:                       # 인코더 단독(th = 인코더 임계)
            pred.append(r["e"] < th); continue
        if r["e"] < lo:
            pred.append(True)
        elif r["e"] >= hi:
            pred.append(False)
        else:
            calls += 1; pred.append(r["q"] < th)
    tp = sum(p and r["y"] for p, r in zip(pred, rows)); fp = sum(p and not r["y"] for p, r in zip(pred, rows))
    neg = sum(r["y"] for r in rows); pos = len(rows) - neg
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / neg if neg else 0.0
    return {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0,
            "PASS": 1 - fp / pos if pos else 0.0, "llm_calls": calls / len(rows)}


GRID_T = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
GRID_LO = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
GRID_HI = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.01]


def best(rows, mode, use_rule):
    cand = []
    if mode == "llm":
        cand = [((None, None, t), run(rows, None, None, t, use_rule)) for t in GRID_T]
    elif mode == "enc":
        cand = [((0, None, t), run(rows, 0, None, t, use_rule)) for t in GRID_T + [0.9, 0.95, 0.99]]
    else:
        cand = [((lo, hi, t), run(rows, lo, hi, t, use_rule))
                for lo in GRID_LO for hi in GRID_HI if hi > lo for t in GRID_T]
    ok = [c for c in cand if c[1]["PASS"] >= 0.9]
    return max(ok, key=lambda c: (c[1]["F1"], -c[1]["llm_calls"])) if ok else None


def fmt(m):
    return f"P {m['P']:.1%} R {m['R']:.1%} F1 {m['F1']:.1%} PASS {m['PASS']:.1%} LLM호출 {m['llm_calls']:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--enc", default="klue")
    ap.add_argument("--llm", default="qwen", help="LLM 점수 파일 태그(qwen=v1 프롬프트, qwen2=v2)")
    a = ap.parse_args()
    rows = load(a.scores, a.enc, a.llm)
    docs = sorted({r["doc"] for r in rows})
    fold = {d: i % 2 for i, d in enumerate(docs)}
    for mode, rule in (("enc", False), ("enc", True), ("llm", False), ("llm", True), ("cascade", True)):
        cfg, m = best(rows, mode, rule) or (None, None)
        if not m:
            print(f"{mode:8s} 규칙{'O' if rule else 'X'}: PASS≥90% 운영점 없음"); continue
        per = {s: fmt(run([r for r in rows if r["set"] == s], *cfg, use_rule=rule)) for s in SETS}
        # 홀드아웃: 단락 절반에서 고른 구성을 나머지 절반에 적용
        tr = [r for r in rows if fold[r["doc"]] == 0]; te = [r for r in rows if fold[r["doc"]] == 1]
        c2 = best(tr, mode, rule)
        ho = fmt(run(te, *c2[0], use_rule=rule)) if c2 else "-"
        print(f"\n[{mode} · 규칙{'O' if rule else 'X'}] 구성(lo,hi,θ)={cfg}\n  전체   {fmt(m)}")
        for s, v in per.items():
            print(f"  {s:13s} {v}")
        print(f"  홀드아웃  {ho}")


if __name__ == "__main__":
    main()
