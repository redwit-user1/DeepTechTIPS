"""학습 데이터 포맷 변환 — 순수 Python(테스트 가능).

합성 파이프라인 산출(data/synth/*.jsonl)을 TRL 학습 포맷으로 변환한다.
- SFT:  {"messages": [{"role":"user",...},{"role":"assistant",...}]}
- DPO:  {"prompt": ..., "chosen": ..., "rejected": ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_sft_record(prompt: str, completion: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    }


def dpo_pairs_to_sft(dpo_path: str | Path) -> list[dict]:
    """DPO 쌍의 chosen(규정준수 응답)을 SFT 지시 데이터로 재사용."""
    out = []
    seen: set[tuple[str, str]] = set()
    for row in read_jsonl(dpo_path):
        key = (row["prompt"], row["chosen"])
        if key in seen:
            continue
        seen.add(key)
        out.append(to_sft_record(row["prompt"], row["chosen"]))
    return out


def dpo_pairs_to_dpo(
    dpo_path: str | Path,
    vcr_accept_threshold: float = 0.0,
) -> list[dict]:
    """TRL DPO 포맷으로 변환. vcr_accept_threshold>0 이면 고품질 쌍만 채택.

    채택 규칙(사업계획서 p.20 순환오류 방지):
      - vcr_chosen >= threshold (충분히 좋은 양성)
      - vcr_chosen > vcr_rejected (선호 방향 일관)
    """
    out = []
    for row in read_jsonl(dpo_path):
        vc = row.get("vcr_chosen", 1.0)
        vr = row.get("vcr_rejected", 0.0)
        if vcr_accept_threshold > 0 and vc < vcr_accept_threshold:
            continue
        if vc <= vr:
            continue
        out.append(
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
        )
    return out


def write_jsonl(records: list[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)
