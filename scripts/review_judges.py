#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""판정 모델 검수 시트 — 판정 모델(Phi-4 · Kanana · Midm)의 문장 판정을 사람 판정과 맞춰 본다 (연구노트 §48).

지금까지의 모델 비교는 Phi-4 판정이 기준이다. 그 판정이 맞는지는 사람이 본 적이 없다. 특히 조건 추출의
'항목: 값' 줄은 값이 원문에 그대로 있어도 판정 모델이 자주 탈락시킨다. 판정 모델끼리 갈린 문장을 골고루 뽑아
사람 두 명이 따로 판정한다. 검수자는 판정 모델의 점수를 **보지 않는다**(정답 키는 별도 파일).

    python scripts/review_judges.py make  [--per-cell 10] [--models base_hint sft_v4 dpo_v4]
    python scripts/review_judges.py score data/labeling/review_judges_A.csv [data/labeling/review_judges_B.csv]

시트는 엑셀용 UTF-8(BOM). human_label 칸에 1 / 0 / x 중 하나를 쓴다.
  1  원문만 보고 이 문장(줄)이 맞다고 확인된다. 반올림 · '약' · 올바른 단위 환산 · 같은 뜻의 다른 표현은 1.
     '항목: 값' 줄은 값이 원문에 있고 항목 이름이 그 값을 가리키는 말로 무리가 없으면 1.
  0  원문과 다르거나(값 · 단위 · 대상 · 방향), 원문으로 확인할 수 없다(덧붙인 원인 · 효과 · 일반화, 추정을 확정으로).
     '항목: 값' 줄은 값이 원문에 없거나 다른 항목의 값이면 0.
  x  판정 불가(원문이 깨졌다, 문장이 잘렸다 등).

층: 원천(연구노트 / KCI) × 모양(항목 줄 / 문장) × Phi-4(통과 / 탈락) × 나머지 둘(둘 다 통과 / 갈림 / 둘 다 탈락).
층마다 같은 수를 뽑고 층의 전체 크기를 키에 적어 둔다 — score 가 그 크기로 가중해 전체에서의 정확도를 추정한다.
연구노트 원문이 들어가므로 시트와 키는 data/ 아래에만 두고 커밋하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

THR = {"phi4": 0.986, "kanana": 0.651, "midm": 0.202}      # 라벨 평가셋으로 보정한 임계값(연구노트 §44)
SRC = {"eval2": "note", "evalk": "kci"}
KV = re.compile(r"^[-•*\d.\s]*[^:：]{1,40}[:：]\s*\S")
KV_JUDGES, KV_THR = ("phi4", "kanana", "midm", "qwen"), 0.5      # 항목 줄 전용 판정(§48) — 임계값은 아직 보정 전
OUT = Path("data/labeling")


def stratum(src: str, text: str, ok: dict[str, bool]) -> str:
    others = ok["kanana"] + ok["midm"]
    return "/".join((src, "kv" if KV.match(text) else "prose", "phi+" if ok["phi4"] else "phi-",
                     ("oth-", "split", "oth+")[others]))


def collect(root: Path, models: list[str]) -> dict[tuple[str, str], dict]:
    """(원문, 문장) → 항목. 세 판정 모델의 점수를 모은다. 같은 문장이 여러 모델 응답에 나오면 하나로."""
    items: dict[tuple[str, str], dict] = {}
    for pre, src in SRC.items():
        for m in models:
            rows = {}
            for tag in THR:
                p = root / f"judged_{tag}" / f"{pre}_{m}.jsonl"
                rows[tag] = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
            kv = {}                                   # 항목 줄 전용 판정 점수(있으면) — judge_all.py --kv
            for tag in KV_JUDGES:
                p = root / "kv" / tag / f"{pre}_{m}.kv.jsonl"
                if p.exists():
                    kv[tag] = {x["i"]: x["kv"] for x in map(json.loads, p.open(encoding="utf-8"))}
            for i, r in enumerate(rows["phi4"]):
                for j, s in enumerate(r["sents"]):
                    key = (r["passage"], s["text"])
                    if key in items or len(s["text"]) < 4:
                        continue
                    sc = {t: rows[t][i]["sents"][j]["llm"] for t in THR}
                    ok = {t: sc[t] >= THR[t] for t in THR}
                    items[key] = {"passage": r["passage"], "text": s["text"], "task": r["task"], "model": m,
                                  "scores": sc, "ok": ok, "stratum": stratum(src, s["text"], ok),
                                  "kv": {t: d[i][j] for t, d in kv.items() if i in d and d[i][j] is not None}}
    return items


def make(a) -> None:
    items = collect(Path(a.root), a.models)
    by = defaultdict(list)
    for it in items.values():
        by[it["stratum"]].append(it)
    rng, picked = random.Random(a.seed), []
    for name in sorted(by):
        pool = sorted(by[name], key=lambda it: (it["passage"], it["text"]))
        for it in rng.sample(pool, min(a.per_cell, len(pool))):
            picked.append({**it, "population": len(pool)})
    rng.shuffle(picked)
    for k, it in enumerate(picked):
        it["id"] = f"J{k + 1:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review_judges.key.json").write_text(json.dumps(
        {it["id"]: {k: it[k] for k in ("stratum", "population", "task", "model", "scores", "ok", "kv")} for it in picked},
        ensure_ascii=False, indent=1), encoding="utf-8")
    for who in "AB":                                   # 두 검수자에게 서로 다른 순서로
        order = picked[:]
        random.Random(a.seed + ord(who)).shuffle(order)
        with (OUT / f"review_judges_{who}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "원문", "문장", "human_label", "메모"])
            for it in order:
                w.writerow([it["id"], it["passage"], it["text"], "", ""])
    print(f"{len(picked)}건 (층 {len(by)}개, 후보 {len(items)}건) → {OUT}/review_judges_A.csv, _B.csv, .key.json")
    for name in sorted(by):
        print(f"  {name:28s} 전체 {len(by[name]):6d} · 뽑음 {min(a.per_cell, len(by[name]))}")


def read_sheet(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r["human_label"].strip().lower() for r in csv.DictReader(f) if r["human_label"].strip()}


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def judge_stats(gold: dict[str, bool], key: dict) -> dict[str, dict]:
    """판정 모델별 정확도 — 층 크기로 가중(전체 문장에서의 추정)과 표본 그대로를 함께."""
    cell = defaultdict(list)
    for i in gold:
        cell[key[i]["stratum"]].append(i)
    weight = {i: key[i]["population"] / len(ids) for ids in cell.values() for i in ids}
    out = {}
    for t in THR:
        c, raw = Counter(), Counter()
        for i, g in gold.items():
            k = ("t" if key[i]["ok"][t] == g else "f") + ("p" if key[i]["ok"][t] else "n")
            c[k] += weight[i]
            raw[k] += 1
        tot = sum(c.values())
        out[t] = {"acc": (c["tp"] + c["tn"]) / tot, "precision": c["tp"] / max(1e-9, c["tp"] + c["fp"]),
                  "recall": c["tp"] / max(1e-9, c["tp"] + c["fn"]), "raw_acc": (raw["tp"] + raw["tn"]) / len(gold),
                  "false_reject": c["fn"] / max(1e-9, c["tp"] + c["fn"]), "false_accept": c["fp"] / max(1e-9, c["fp"] + c["tn"])}
    return out


def kv_stats(gold: dict[str, bool], key: dict) -> None:
    """항목 줄 전용 판정(KV_PROMPT)이 사람 판정과 얼마나 맞는지 — 표본 그대로, 임계값 몇 개에서."""
    sub = {i: g for i, g in gold.items() if key[i].get("kv")}
    if not sub:
        return
    print(f"[항목 줄 전용 판정] n={len(sub)}")
    for t in KV_JUDGES:
        have = {i: g for i, g in sub.items() if t in key[i]["kv"]}
        if not have:
            continue
        cells = []
        for thr in (0.2, KV_THR, 0.8, 0.95):
            acc = sum((key[i]["kv"][t] >= thr) == g for i, g in have.items()) / len(have)
            cells.append(f"{thr}: {acc:.1%}")
        pos = [i for i, g in have.items() if g]
        neg = [i for i, g in have.items() if not g]
        print(f"  {t:7s} 정확도 " + " · ".join(cells) + f" | {KV_THR} 에서 맞는 줄 탈락 "
              f"{sum(key[i]['kv'][t] < KV_THR for i in pos) / max(1, len(pos)):.1%} · 틀린 줄 통과 "
              f"{sum(key[i]['kv'][t] >= KV_THR for i in neg) / max(1, len(neg)):.1%}")


def score(a) -> None:
    key = json.loads((Path(a.sheets[0]).parent / "review_judges.key.json").read_text(encoding="utf-8"))
    sheets = [read_sheet(p) for p in a.sheets]
    for p, s in zip(a.sheets, sheets):
        bad = {v for v in s.values() if v not in ("1", "0", "x")}
        if bad:
            raise SystemExit(f"거부: {p} 의 human_label 에 1 / 0 / x 가 아닌 값 {sorted(bad)}")
        print(f"{p}: {len(s)}건 · {dict(Counter(s.values()))}")
    if len(sheets) == 2:
        both = [i for i in sheets[0] if i in sheets[1]]
        print(f"두 검수자 공통 {len(both)}건 · 일치 {sum(sheets[0][i] == sheets[1][i] for i in both) / len(both):.1%} · "
              f"κ = {kappa([sheets[0][i] for i in both], [sheets[1][i] for i in both]):.3f}")
        gold = {i: sheets[0][i] == "1" for i in both if sheets[0][i] == sheets[1][i] and sheets[0][i] != "x"}
        print(f"기준: 두 사람이 일치한 {len(gold)}건(판정 불가 제외)")
    else:
        gold = {i: v == "1" for i, v in sheets[0].items() if v != "x"}
        print(f"기준: 한 사람 판정 {len(gold)}건(판정 불가 제외) — κ 는 두 번째 시트가 있어야 잰다")
    for name, sub in (("전체", gold),
                      ("항목 줄", {i: g for i, g in gold.items() if "/kv/" in key[i]["stratum"]}),
                      ("문장", {i: g for i, g in gold.items() if "/prose/" in key[i]["stratum"]})):
        if not sub:
            continue
        print(f"[{name}] n={len(sub)} · 사람이 '맞다'고 본 비율(표본) {sum(sub.values()) / len(sub):.1%}")
        for t, s in judge_stats(sub, key).items():
            print(f"  {t:7s} 정확도 {s['acc']:.1%}(표본 그대로 {s['raw_acc']:.1%}) · 정밀도 {s['precision']:.1%} · 재현율 {s['recall']:.1%} · "
                  f"맞는 문장을 탈락시킨 비율 {s['false_reject']:.1%} · 틀린 문장을 통과시킨 비율 {s['false_accept']:.1%}")
    kv_stats(gold, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--root", default="data/judged_x", help="judged_<판정 모델>/ 폴더들이 있는 곳")
    m.add_argument("--models", nargs="+", default=["base_hint", "sft_v4", "dpo_v4"])
    m.add_argument("--per-cell", type=int, default=10)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("sheets", nargs="+")
    a = ap.parse_args()
    {"make": make, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
