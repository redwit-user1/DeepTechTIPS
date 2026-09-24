"""국내 R&D **실데이터** 평가셋 — 변조 없음, 라벨은 출처(provenance)에서 도출.

## 합성셋과의 결정적 차이

| | 합성 KR (`build_kr.py`) | **실데이터 KR (본 모듈)** |
|---|---|---|
| 주장 문장 | 템플릿 생성 | **실제 프로토콜 원문** |
| 오류 생성 | 규칙 기반 변조(우리가 만듦) | **다른 실제 과제의 원문을 잘못 귀속** |
| 라벨 근거 | 우리 변조 규칙 | **실제 출처 일치 여부** |
| 난이도 | 낮음(규칙을 검증기가 앎) | **높음(같은 도메인의 진짜 문장)** |

핵심: 음성 사례가 **같은 도메인 국내 연구과제의 실제 문장**이다.
규칙으로 만든 티가 나지 않으므로, 패턴 매칭이 아니라 실제 의미 근거 확인이 필요하다.
SciFact 외부 평가와 같은 방식(교차 문서 귀속 오류)을 국내 R&D 문서에 적용한 것.

## 검증 항목 (ALCOA+ 매핑)

| 항목 | 양성 | 음성(실제 오귀속) | ALCOA+ |
|---|---|---|---|
| primary_outcome | 해당 과제의 실제 주요결과변수 | **다른 과제의 실제 결과변수** | Accurate |
| enrollment | 해당 과제의 실제 등록례수 | **다른 과제의 실제 등록례수** | Accurate |
| eligibility | 해당 과제의 실제 선정기준 | **다른 과제의 실제 선정기준** | Complete |
| institution | 실제 수행기관 목록 내 기관 | **다른 과제의 실제 수행기관** | Attributable |

실행:
  python -m compliance_gateway.data.korean.real_eval --out data/synth/kr_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from compliance_gateway.verify.models import PaperRecord
from compliance_gateway.verify.verifier import LocalRegistry

SEED = Path("compliance_gateway/data/korean/seed/kr_protocols.json")

# 검증 항목 → 위반 ALCOA+ 속성
ATTR = {
    "primary_outcome": "Accurate",
    "enrollment": "Accurate",
    "eligibility": "Complete",
    "institution": "Attributable",
}


def load_protocols(path: Optional[Path] = None) -> list[dict]:
    return json.loads(Path(path or SEED).read_text(encoding="utf-8"))


def grounding_of(t: dict) -> str:
    """근거 = 해당 과제의 실제 프로토콜 서술(원문 그대로)."""
    parts = [t["brief_summary"]]
    if t.get("detailed_description"):
        parts.append(t["detailed_description"])
    return " ".join(parts)


def citation_of(t: dict) -> str:
    """국내 R&D 출처 표기 — 실제 기관·연도·과제번호."""
    year = t["start_date"][:4]
    return f"(출처: {t['sponsor_ko']}, {year}, 과제번호 {t['nct_id']})"


def _carrier(field: str, value: str, t: dict) -> str:
    """주장 문장. **검증 대상 값은 실제 원문**이고 전달 문구만 최소 템플릿."""
    c = citation_of(t)
    if field == "primary_outcome":
        return f"본 연구의 주요 결과변수는 다음과 같다: {value} {c}"
    if field == "enrollment":
        return f"본 연구에는 총 {value}명이 등록되었다. {c}"
    if field == "eligibility":
        return f"본 연구의 주요 선정기준은 다음과 같다: {value} {c}"
    if field == "institution":
        return f"본 연구는 {value}에서 수행되었다. {c}"
    raise ValueError(field)


def build(protocols: list[dict]) -> tuple[list[dict], list[dict]]:
    """실데이터 평가 아이템 + DPO 쌍 생성.

    음성은 **다른 과제의 실제 값**을 가져와 잘못 귀속시킨 것이다(원문 변조 없음).
    """
    items: list[dict] = []
    pairs: list[dict] = []
    n = len(protocols)

    for i, t in enumerate(protocols):
        g = grounding_of(t)
        other = protocols[(i + 1) % n]          # 짝수: 다른 실제 과제
        far = protocols[(i + 2) % n]

        specs = [
            ("primary_outcome", str(t["primary_outcome"]), str(other["primary_outcome"])),
            ("enrollment", f"{t['enrollment']:,}", f"{other['enrollment']:,}"),
            ("eligibility", str(t["inclusion"]), str(far["inclusion"])),
        ]
        # 기관 귀속: 실제 사이트 목록에 없는 '다른 과제의 실제 기관'
        own_sites = set(t["sites"])
        wrong_site = next((s for s in far["sites"] + other["sites"] if s not in own_sites), None)
        if wrong_site:
            specs.append(("institution", t["sites"][0], wrong_site))

        for field, good, bad in specs:
            if good == bad:
                continue
            chosen = _carrier(field, good, t)
            rejected = _carrier(field, bad, t)
            query = f"{t['sponsor_ko']} 연구({t['nct_id']})의 {field} 항목을 출처와 함께 기술하라."

            items.append({"query": query, "response": chosen, "grounding": g,
                          "source_id": t["nct_id"], "label": "compliant",
                          "field": field, "alcoa_violation": None, "lang": "ko"})
            items.append({"query": query, "response": rejected, "grounding": g,
                          "source_id": t["nct_id"], "label": f"misattributed_{field}",
                          "field": field, "alcoa_violation": ATTR[field], "lang": "ko"})
            pairs.append({"prompt": query, "chosen": chosen, "rejected": rejected,
                          "rejected_kind": f"misattributed_{field}",
                          "alcoa_violation": ATTR[field], "grounding": g,
                          "source_id": t["nct_id"], "lang": "ko"})
    return items, pairs


def build_registry(protocols: list[dict]) -> LocalRegistry:
    """실제 과제번호 레지스트리(서지 검증은 통과 → 의미 근거만 측정)."""
    return LocalRegistry([
        PaperRecord(doi=t["nct_id"], title=t["title"], authors=(t["sponsor_ko"],),
                    year=int(t["start_date"][:4]), source="kr_protocols")
        for t in protocols
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth/kr_real")
    a = ap.parse_args()

    protocols = load_protocols()
    items, pairs = build(protocols)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("kr_real_eval.jsonl", items), ("kr_real_dpo.jsonl", pairs)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for i in items if i["label"] == "compliant")
    print(f"국내 실제 프로토콜 {len(protocols)}건 → 평가 {len(items)}건 "
          f"(compliant {n_ok} / 오귀속 {len(items)-n_ok}), DPO {len(pairs)}쌍")
    by_field: dict[str, int] = {}
    for i in items:
        if i["alcoa_violation"]:
            by_field[i["field"]] = by_field.get(i["field"], 0) + 1
    print("\n검증 항목별 오귀속 사례:")
    for k, v in sorted(by_field.items()):
        print(f"  {k:18s} {v:3d}  → ALCOA+ {ATTR[k]}")
    print(f"\n→ {out}/kr_real_eval.jsonl, kr_real_dpo.jsonl")


if __name__ == "__main__":
    main()
