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

from compliance_gateway.data.ocr.claim_evidence import NoteBuffer, perturb, supports
from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.deidentify import audit, deidentify


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
    n = kept = 0

    for row in iter_rows(Path(a.csv_path)):
        n += 1
        key = nid(row.note_id)
        if key % a.doc_stride:
            continue
        kept += 1
        notes.setdefault(key, NoteBuffer()).add(row.text)

    usable = [k for k, v in notes.items() if v.usable]
    rng.shuffle(usable)

    stats: Counter[str] = Counter()
    residual = 0
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for i, key in enumerate(usable):
            nb = notes[key]
            # 다른 문서의 근거 — 쉬운 음성
            other = notes[usable[(i + 1 + rng.randrange(len(usable) - 1)) % len(usable)]]
            pairs = min(a.max_pairs_per_doc, len(nb.claims))
            for claim in rng.sample(nb.claims, pairs):
                # 같은 문서라는 것만으로는 부족하다. 실제로 뒷받침할 만한
                # 근거만 고른다(어휘/수치 공유). 없으면 이 주장은 버린다.
                cand = [e for e in nb.evidence if supports(claim, e)]
                if not cand:
                    stats["no_support"] += 1
                    continue
                ev = rng.choice(cand)
                bad, changes = perturb(ev, rng)
                if bad == ev:            # 교란이 실패하면 하드 네거티브를 만들 수 없다
                    stats["perturb_failed"] += 1
                    continue
                triples = [
                    ("entail", ev, None),
                    ("neutral", rng.choice(other.evidence), None),
                    ("contradict", bad, changes),
                ]
                for label, evidence, ch in triples:
                    dc, de = deidentify(claim), deidentify(evidence)
                    if audit(dc.text) or audit(de.text):
                        residual += 1
                        continue
                    f.write(json.dumps({
                        "claim": dc.text, "evidence": de.text, "label": label,
                        "doc": f"{key:014x}",
                        "changes": ch,
                    }, ensure_ascii=False) + "\n")
                    stats[label] += 1
                    written += 1

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "rows_scanned": n, "rows_kept": kept, "doc_stride": a.doc_stride,
        "docs_buffered": len(notes), "docs_usable": len(usable),
        "usable_pct": round(len(usable) / max(1, len(notes)), 4),
        "pairs": dict(stats), "written": written,
        "residual_pii_dropped": residual, "seed": a.seed,
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
                   ("perturb_failed", "교란 실패로 폐기")):
        if stats[k]:
            print(f"  {msg} {stats[k]:,}건")
    print(f"\n[비식별화] 잔여 PII 로 폐기 {residual:,}건",
          "← 0 이 아니면 확인 필요" if residual else "OK")
    est = int(len(usable) * a.doc_stride)
    print(f"\n  전체 코퍼스 환산: 사용 가능 문서 약 {est:,}개"
          f" → 약 {int(written * a.doc_stride):,}건 생성 가능")
    print(f"\n→ {out}\n→ {meta}")


if __name__ == "__main__":
    main()
