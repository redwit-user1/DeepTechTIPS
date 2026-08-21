"""국내 R&D 한국어 평가셋 / DPO 데이터 빌드.

실행:
  python -m compliance_gateway.data.korean.build_kr
  python -m compliance_gateway.data.korean.build_kr --out data/synth/kr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compliance_gateway.data.korean import kr_render as R
from compliance_gateway.data.korean.models import KRResearchRecord
from compliance_gateway.data.korean.sources import load_kr_seed
from compliance_gateway.verify.models import PaperRecord
from compliance_gateway.verify.verifier import LocalRegistry

# 변조 유형 → 위반 ALCOA+ 속성
ALCOA_VIOLATION = {
    "sponsor_swap": "Attributable",
    "enrollment_tamper": "Accurate",
    "date_shift": "Contemporaneous",
    "id_fabrication": "Original",
    "no_source": "Attributable",
}


def build_registry(records: list[KRResearchRecord]) -> LocalRegistry:
    """국내 과제번호 레지스트리 — 과제번호를 DOI 자리에 매핑."""
    return LocalRegistry([
        PaperRecord(doi=r.nct_id, title=r.title,
                    authors=(R.sponsor_ko(r),), year=r.year, source="kr_trials")
        for r in records
    ])


def build_items(records: list[KRResearchRecord]) -> tuple[list[dict], list[dict]]:
    """(Gateway 평가 아이템, DPO 쌍) 생성."""
    evals: list[dict] = []
    pairs: list[dict] = []

    for rec in records:
        grounding = R.render_claim(rec)          # 진짜 사실(근거)
        chosen = R.render_cited(rec)
        query = f"{R.sponsor_ko(rec)}의 해당 연구 내용을 출처와 함께 요약하라."

        evals.append({
            "query": query, "response": chosen, "grounding": grounding,
            "source_id": rec.nct_id, "label": "compliant",
            "alcoa_violation": None, "lang": "ko",
        })

        negatives: list[tuple[str, str | None]] = [
            ("no_source", R.strip_source(rec)),
            ("sponsor_swap", R.tamper_sponsor(rec)),
            ("enrollment_tamper", R.tamper_enrollment(rec)),
            ("date_shift", R.tamper_date(rec)),
            ("id_fabrication", R.tamper_id(rec)),
        ]
        for kind, text in negatives:
            if not text or text == chosen:
                continue
            evals.append({
                "query": query, "response": text, "grounding": grounding,
                "source_id": rec.nct_id, "label": kind,
                "alcoa_violation": ALCOA_VIOLATION[kind], "lang": "ko",
            })
            pairs.append({
                "prompt": query, "chosen": chosen, "rejected": text,
                "rejected_kind": kind, "alcoa_violation": ALCOA_VIOLATION[kind],
                "grounding": grounding, "source_id": rec.nct_id, "lang": "ko",
            })
    return evals, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth/kr")
    a = ap.parse_args()

    records = load_kr_seed()
    evals, pairs = build_items(records)
    registry = build_registry(records)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("kr_eval.jsonl", evals), ("kr_dpo_pairs.jsonl", pairs)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for e in evals if e["label"] == "compliant")
    print(f"국내 연구과제 {len(records)}건 → 평가 {len(evals)}건 "
          f"(compliant {n_ok} / 위반 {len(evals)-n_ok}), DPO {len(pairs)}쌍")
    print("\nALCOA+ 속성별 위반 사례 수:")
    counts: dict[str, int] = {}
    for e in evals:
        if e["alcoa_violation"]:
            counts[e["alcoa_violation"]] = counts.get(e["alcoa_violation"], 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k:18s} {v}")
    print(f"\n→ {out}/kr_eval.jsonl, kr_dpo_pairs.jsonl")
    print(f"   레지스트리 과제번호 {len(records)}건")


if __name__ == "__main__":
    main()
