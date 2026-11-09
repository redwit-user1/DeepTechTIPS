#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""근거·주장 NLI 학습셋 생성 — 사람 라벨 없이.

사업계획서 대응:
- `ROADMAP.md` M2 "GOONO 70K → (질문, 답변, 출처) 트리플, 5만건 목표"
- `ARCHITECTURE.md` "(질문, 답변, 출처) 트리플 **자동추출**"
- `VCR_SPEC.md` 환각 유형 C "수치 변조(37°C → 25°C)"

세 가지 라벨을 만든다.

| 라벨 | 구성 | 학습시키는 것 |
|---|---|---|
| `entail` | 주장 + 같은 문서의 근거 | SourceMatch |
| `neutral` | 주장 + **다른 문서**의 근거 | 무관한 출처 거부 |
| `contradict` | 주장 + **수치를 교란한** 근거 | Halluc 유형 C |

## 왜 행 분류기를 안 쓰는가

`analysis` 버킷 정밀도가 28.6% 라 행 단위로는 주장을 못 고른다.
문장 단위 정규식은 그 실패에 의존하지 않는다.

## 메모리

문서 전문을 들고 있지 않는다. 행을 읽는 즉시 근거·주장 문장만 뽑아
문서당 최대 12개씩만 버퍼에 남긴다.

비식별화는 **출력 직전에 강제**된다. 끄는 옵션이 없다.
출력은 `data/` 안에서만 허용한다(gitignore 대상).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.claim_evidence import (
    NoteBuffer, numbers, perturb, shared_numbers, supports, surface_leak, words,
)

# 표면 특징만으로 이 정확도를 넘으면 학습셋으로 쓰지 않는다(3분류 우연 = 33%).
LEAK_MAX = 0.60
from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.deidentify import audit, deidentify
from compliance_gateway.data.ocr.sensitive import sensitive_signals


def nid(s: str) -> int:
    return int.from_bytes(hashlib.blake2s(s.encode(), digest_size=7).digest(), "big")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--doc-stride", type=int, default=8,
                    help="NoteId 해시 기준 문서 표본. 행이 아니라 문서를 솎아야 "
                         "한 문서의 페이지가 흩어지지 않는다")
    ap.add_argument("--max-pairs-per-doc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", default="data/nli/claim_evidence.jsonl")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if not str(out).startswith(str(Path.cwd().resolve() / "data")):
        sys.exit(f"거부: 출력은 data/ 안이어야 한다. 받은 값: {a.out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(a.seed)
    notes: dict[int, NoteBuffer] = {}
    bad: set[int] = set()          # 민감(건강) 행이 하나라도 있는 문서 — 통째로 뺀다
    n = kept = 0

    for row in iter_rows(Path(a.csv_path)):
        n += 1
        key = nid(row.note_id)
        if key % a.doc_stride:
            continue
        kept += 1
        if sensitive_signals(row.text):
            bad.add(key)
        notes.setdefault(key, NoteBuffer()).add(row.text)

    usable = [k for k, v in notes.items() if k not in bad and v.usable]
    rng.shuffle(usable)

    # 다른 문서 근거의 역색인 — neutral 을 **어휘가 겹치는** 근거에서 고르기 위함.
    # 1차 셋은 무작위 문서에서 골라 neutral 의 97% 가 중첩 0 이었다
    # ("단어가 겹치면 entail" 만 배우게 된다).
    index: dict[str, list[tuple[int, int]]] = {}
    for key in usable:
        for j, e in enumerate(notes[key].evidence):
            for w in words(e):
                index.setdefault(w, []).append((key, j))

    def hard_neutral(claim: str, own: int) -> str | None:
        """다른 문서의 근거 중 내용어 2개+ 공유, 주장 수치는 공유하지 않는 것."""
        hits: Counter[tuple[int, int]] = Counter()
        for w in sorted(words(claim), key=lambda w: len(index.get(w, ()))):
            post = index.get(w, ())
            # ponytail: 흔한 단어 목록은 표본 2000 개로 자른다 — 정확한 최근접이 필요하면 BM25 로 교체
            for kj in (post if len(post) <= 2000 else rng.sample(post, 2000)):
                if kj[0] != own:
                    hits[kj] += 1
        nums = numbers(claim)
        cand = [kj for kj, c in hits.items() if c >= 2
                and not (nums & numbers(notes[kj[0]].evidence[kj[1]]))]
        if not cand:
            return None
        k, j = rng.choice(cand)
        return notes[k].evidence[j]

    stats: Counter[str] = Counter()
    rows: list[dict] = []
    residual = 0
    for key in usable:
        nb = notes[key]
        pairs = min(a.max_pairs_per_doc, len(nb.claims))
        for claim in rng.sample(nb.claims, pairs):
            # 같은 문서라는 것만으로는 부족하다. 실제로 뒷받침할 만한
            # 근거만 고른다(어휘/수치 공유). 없으면 이 주장은 버린다.
            cand = [e for e in nb.evidence if supports(claim, e)]
            if not cand:
                stats["no_support"] += 1
                continue
            # 주장과 **같은 값**을 공유하는 근거만 쓴다. 그 값을 바꿔야 모순이 성립한다.
            cand = [(e, s) for e in cand if (s := shared_numbers(claim, e))]
            if not cand:
                stats["no_shared_number"] += 1
                continue
            ev, shared = rng.choice(cand)
            bad, changes = perturb(ev, rng, targets=shared)
            neu = hard_neutral(claim, key)
            if bad == ev or neu is None:
                stats["perturb_failed" if bad == ev else "no_hard_neutral"] += 1
                continue
            triples = [("entail", ev, None), ("neutral", neu, None), ("contradict", bad, changes)]
            dec = [(lab, deidentify(claim).text, deidentify(e).text, ch) for lab, e, ch in triples]
            # 삼중항 단위로 버린다 — 하나만 빠지면 라벨 균형이 깨진다
            if any(audit(c) or audit(e) for _, c, e, _ in dec):
                residual += 1
                continue
            for lab, c, e, ch in dec:
                rows.append({"claim": c, "evidence": e, "label": lab,
                             "doc": f"{key:014x}", "changes": ch})
                stats[lab] += 1

    leak = surface_leak(rows)
    written = len(rows)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "rows_scanned": n, "rows_kept": kept, "doc_stride": a.doc_stride,
        "docs_buffered": len(notes), "docs_usable": len(usable),
        "usable_pct": round(len(usable) / max(1, len(notes)), 4),
        "pairs": dict(stats), "written": written,
        "residual_pii_dropped": residual, "seed": a.seed,
        "surface_leak": round(leak, 4), "leak_max": LEAK_MAX,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f" 근거·주장 학습셋 — {written:,}건")
    print("=" * 66)
    print(f"\n  전수 {n:,}행 → 표본 {kept:,}행(문서 1/{a.doc_stride})")
    print(f"  버퍼 문서 {len(notes):,}개 중 사용 가능 {len(usable):,}개"
          f" ({len(usable)/max(1,len(notes))*100:.1f}%)")
    print("\n[라벨 분포]")
    for k in ("entail", "neutral", "contradict"):
        print(f"  {k:12s} {stats[k]:>8,}")
    for k, msg in (("no_support", "뒷받침 근거 없어 폐기"),
                   ("no_shared_number", "주장과 공유 수치 없어 폐기"),
                   ("no_hard_neutral", "어휘 겹치는 neutral 없어 폐기"),
                   ("perturb_failed", "교란 실패로 폐기")):
        if stats[k]:
            print(f"  {msg} {stats[k]:,}건")
    print(f"\n[비식별화] 잔여 PII 로 폐기 삼중항 {residual:,}건",
          "← 0 이 아니면 확인 필요" if residual else "OK")
    verdict = "OK" if leak <= LEAK_MAX else "✗ 학습 금지 — 표면 특징으로 라벨이 샌다"
    print(f"[누설 게이트] 표면 특징 상한 정확도 {leak*100:.1f}% (기준 ≤{LEAK_MAX*100:.0f}%) {verdict}")
    est = int(len(usable) * a.doc_stride)
    print(f"\n  전체 코퍼스 환산: 사용 가능 문서 약 {est:,}개"
          f" → 약 {int(written * a.doc_stride):,}건 생성 가능")
    print(f"\n→ {out}\n→ {meta}")


if __name__ == "__main__":
    main()
