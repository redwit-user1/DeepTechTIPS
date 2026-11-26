#!/usr/bin/env python3
"""AI 패널 검수 → 판정 기준(연구노트 §63). 사람 검수 대신 계열이 다른 AI 검수자 넷의 독립 판정을 모은다.

검수자: AI-1(review_judges_AI.csv), AI-2(review_judges_AI_independent.csv, 앞 결과를 보지 않고 다시 함),
Grok(review_judges_Grok.tsv), Claude(review_judges_Claude_b1~b4.tsv, 정답 키 · 다른 검수 · 모델 점수를 보지 않음).
판정 모델(Qwen · Kanana · Midm · EEVE · Phi-4)과 계열이 겹치지 않는다.

기준(gold): 과반(넷이면 셋, 다섯이면 셋)이 같은 값(1 또는 0)을 낸 항목. 모두 같은 것만 쓰는 엄격 기준도 함께 낸다.
출력: review_judges_panel.csv / review_judges_panel_strict.csv (review_judges.py score 가 읽는 시트 형식)

    python scripts/review_panel.py data/labeling
    python scripts/review_judges.py score data/labeling/review_judges_panel.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


def read_csv(p: Path, col: str) -> dict[str, str]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r[col].strip().lower() for r in csv.DictReader(f)}


def read_tsv(paths: list[Path], col: str) -> dict[str, str]:
    out = {}
    for p in paths:
        lines = p.read_text(encoding="utf-8-sig").splitlines()
        head = lines[0].split("\t")
        for line in lines[1:]:
            if line.strip():
                r = dict(zip(head, line.split("\t")))
                out[r["id"]] = r[col].strip().lower()
    return out


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(rows: list[list[str]], cats=("1", "0", "x")) -> float:
    n, m = len(rows), len(rows[0])
    p_j = {c: sum(r.count(c) for r in rows) / (n * m) for c in cats}
    p_i = [(sum(r.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for r in rows]
    pbar, pe = sum(p_i) / n, sum(v * v for v in p_j.values())
    return (pbar - pe) / (1 - pe)


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/labeling")
    raters = {
        "AI-1": read_csv(d / "review_judges_AI.csv", "ai_label"),
        "AI-2": read_csv(d / "review_judges_AI_independent.csv", "ai_label"),
        "Grok": read_tsv([d / "review_judges_Grok.tsv"], "grok_label"),
        "Claude": read_tsv(sorted(d.glob("review_judges_Claude_b*.tsv")), "label"),
    }
    g2 = sorted(d.glob("review_judges_Grok2_b*.tsv"))       # 다섯째 검수자(있으면): Grok 을 앞 결과 없이 다시
    if g2:
        raters["Grok2"] = read_tsv(g2, "label")
    with (d / "review_judges_A.csv").open(encoding="utf-8-sig", newline="") as f:
        sheet = {r["id"]: r for r in csv.DictReader(f)}
    ids = sorted(sheet)
    for name, lab in raters.items():
        miss = [i for i in ids if lab.get(i) not in ("0", "1", "x")]
        print(f"{name:7s} {len(lab)}건 {dict(Counter(lab.get(i) for i in ids))}" + (f" · 빠짐/이상값 {len(miss)}" if miss else ""))
        if miss:
            raise SystemExit(f"거부: {name} 의 판정이 빠지거나 이상한 항목 {miss[:5]}")
    names = list(raters)
    print("\n쌍별 일치 · κ")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = [raters[a][k] for k in ids], [raters[b][k] for k in ids]
            print(f"  {a:6s} - {b:6s} 일치 {sum(p == q for p, q in zip(x, y)) / len(ids):.1%} · κ {kappa(x, y):.3f}")
    rows = [[raters[n][k] for n in names] for k in ids]
    print(f"Fleiss κ (넷) {fleiss(rows):.3f}")

    votes = {k: Counter(raters[n][k] for n in names) for k in ids}
    m = len(names)
    for tag, need in (("panel", m // 2 + 1 if m > 4 else 3), ("panel_strict", m)):
        out = d / f"review_judges_{tag}.csv"
        lab = {}
        for k in ids:
            top, c = votes[k].most_common(1)[0]
            lab[k] = top if c >= need and top in ("0", "1") else ("x" if c >= need else "")
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "원문", "문장", "human_label", "메모"])
            for k in ids:
                w.writerow([k, sheet[k]["원문"], sheet[k]["문장"], lab[k], " ".join(f"{n}={raters[n][k]}" for n in names)])
        c = Counter(lab.values())
        print(f"{tag}: 기준 {c['1'] + c['0']}건(1={c['1']} · 0={c['0']}) · 판정불가 {c['x']} · 합의 안 됨 {c['']} → {out}")


if __name__ == "__main__":
    main()
