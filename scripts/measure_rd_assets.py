#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연구분야 특화 AI 용 '가공 가능성' 측정.

VCR(출처 검증)이 아니라 **연구 수행을 돕는 AI** 관점에서 이 코퍼스에
무엇이 실제로 들어 있는지 센다. 계획을 세우기 전에 전제부터 측정한다.

논문에는 없고 연구노트에만 있는 것 네 가지를 찾는다:

1. **음성 결과** — 실패·미검출·재현 불가. 논문이 체계적으로 배제하는 정보.
2. **측정값 밀도** — 수치+단위. 분야별 정상범위 사전(numeric prior)의 재료.
3. **프로토콜 구조** — 시약·조건·절차. 실험 명세 카드로 뽑을 수 있는가.
4. **궤적** — 한 노트 안의 여러 페이지. 가설→실험→해석의 흐름이 잡히는가.

원문은 출력하지 않는다. 집계만.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from classify_domain import classify_domain
from triage_ocr_content import SCAN_CHARS, classify_content

RESEARCH = {"experimental", "analysis", "literature", "planning"}

# ① 음성 결과 — 논문에 안 실리는 것
NEGATIVE = re.compile(
    r"실패|안\s?됐|안\s?됨|되지\s*않(?:았|음|는)|미검출|검출\s*(?:되지\s*않|안\s?됨)|"
    r"재현\s*(?:이\s*)?(?:안|불가|되지)|불가능|오염(?:됨|되었|이\s*발생)|"
    r"수율\s*(?:이\s*)?(?:낮|저조|떨어)|효율\s*(?:이\s*)?(?:낮|저조)|"
    r"반응\s*(?:이\s*)?없|무반응|관찰되지\s*않|나타나지\s*않|"
    r"\bfailed?\b|\bunsuccessful\b|not\s+detect|no\s+(?:signal|band|peak|growth|change|effect)|"
    r"contaminat|\bnegative\s+(?:result|control\s+fail)",
    re.IGNORECASE,
)
# 재시도 — 실패 뒤에 오는 신호
RETRY = re.compile(
    r"재실험|다시\s*(?:실험|측정|수행)|재측정|재시도|조건\s*변경|"
    r"repeat(?:ed)?\s+(?:the\s+)?experiment|retry|re-?run|re-?test",
    re.IGNORECASE,
)

# ② 측정값 — 수치 + 단위
UNIT = re.compile(
    r"\d+(?:[.,]\d+)?\s*"
    r"(?:mg|kg|[munμ]?g|[munμ]?[lL]|mL|°C|℃|\bK\b|rpm|min|hr?\b|sec\b|"
    r"[munμ]?m\b|nm|Å|%|ppm|pH|[munμ]?M\b|mol|Hz|kHz|MHz|[MG]?Pa|bar|psi|"
    r"[mk]?V\b|[mu]?A\b|[kM]?W\b|[kK]?Da|CFU|OD\d*|rpm|xg\b|배\b|회\b)",
)
# ③ 프로토콜 구조
PROTOCOL = re.compile(
    r"시약|reagent|장비|기기|instrument|조건|condition|절차|procedure|프로토콜|protocol|"
    r"방법|method|전처리|배지|medium|buffer|완충|희석|dilut|incubat|배양|교반|stir|"
    r"원심\s*분리|centrifug|정제|purif|세척|wash|건조|dry|가열|heat|냉각|cool|"
    r"투입|주입|inject|첨가|add(?:ition)?\b",
    re.IGNORECASE,
)
# 반복·통계 — 재현성 기록
REPLICATE = re.compile(
    r"\bn\s*=\s*\d+|반복\s*(?:측정|실험)?|triplicate|duplicate|replicate|"
    r"[2-9]\s*회\s*(?:반복|측정|실험)|평균\s*±|±\s*\d|표준\s*편차|std\s*dev|\bSD\b|\bSEM\b",
    re.IGNORECASE,
)
# ④ 날짜 — 궤적 정렬 가능성
DATE = re.compile(
    r"20\d{2}[-./년]\s*\d{1,2}[-./월]\s*\d{1,2}|"
    r"\d{1,2}[-./]\d{1,2}[-./]20\d{2}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--json", default="ocr_rd_assets.json")
    a = ap.parse_args()

    n = n_res = 0
    neg = retry = proto = repl = dated = 0
    neg_by_dom: Counter[str] = Counter()
    res_by_dom: Counter[str] = Counter()
    unit_hist: Counter[str] = Counter()
    unit_by_dom: dict[str, int] = Counter()
    proto_rich = 0          # 프로토콜 신호 3+ & 측정값 3+ → 명세 카드 후보
    notes: Counter[str] = Counter()
    EST = 6_520_029

    for row in iter_rows(Path(a.csv_path), limit=a.rows):
        n += 1
        notes[row.note_id] += 1
        bucket, _ = classify_content(row.text)
        if bucket not in RESEARCH:
            continue
        n_res += 1
        t = row.text[:SCAN_CHARS]
        dom, _ = classify_domain(row.text)
        res_by_dom[dom] += 1

        nu = len(UNIT.findall(t))
        unit_hist["0" if nu == 0 else "1-2" if nu < 3 else
                  "3-9" if nu < 10 else "10-29" if nu < 30 else "30+"] += 1
        unit_by_dom[dom] += nu

        has_neg = bool(NEGATIVE.search(t))
        if has_neg:
            neg += 1
            neg_by_dom[dom] += 1
            if RETRY.search(t):
                retry += 1
        np_ = len(PROTOCOL.findall(t))
        if np_ >= 3:
            proto += 1
            if nu >= 3:
                proto_rich += 1
        if REPLICATE.search(t):
            repl += 1
        if DATE.search(t):
            dated += 1

    multi = sum(1 for v in notes.values() if v >= 3)
    pr = lambda c: f"{c:>7,} ({c/max(1,n_res)*100:5.1f}%)  약 {int(c/max(1,n)*EST):>9,}행"

    print("=" * 70)
    print(f" 연구분야 AI 용 가공 가능성 — 표본 {n:,}행 / 연구 버킷 {n_res:,}행")
    print("=" * 70)

    print("\n① 음성 결과 — 논문에 실리지 않는 정보")
    print(f"  실패·미검출 기록      {pr(neg)}")
    print(f"  └ 재시도까지 기록     {pr(retry)}")

    print("\n② 측정값 밀도 — 수치+단위 개수 분포")
    for k in ("0", "1-2", "3-9", "10-29", "30+"):
        c = unit_hist[k]
        print(f"  {k:>6s}개 {c:>7,} ({c/max(1,n_res)*100:5.1f}%) {'█'*int(c/max(1,n_res)*34)}")
    ex = sum(unit_hist[k] for k in ("3-9", "10-29", "30+"))
    print(f"  추출 가능(3개+)      {pr(ex)}")

    print("\n③ 프로토콜 구조")
    print(f"  절차 신호 3개+        {pr(proto)}")
    print(f"  └ 측정값도 3개+       {pr(proto_rich)}   ← 실험 명세 카드 후보")
    print(f"  반복·통계 기록        {pr(repl)}")

    print("\n④ 궤적")
    print(f"  고유 노트             {len(notes):,}개 / 3페이지 이상 {multi:,}개 "
          f"({multi/max(1,len(notes))*100:.1f}%)")
    print(f"  날짜 포함             {pr(dated)}")

    print("\n[분야별 음성 결과 비율]")
    for d, c in res_by_dom.most_common(7):
        if d == "미분류":
            continue
        r = neg_by_dom[d] / max(1, c)
        avg_u = unit_by_dom[d] / max(1, c)
        print(f"  {d:14s} 음성 {r*100:5.1f}%   평균 측정값 {avg_u:5.1f}개/행")

    Path(a.json).write_text(json.dumps({
        "sampled_rows": n, "research_rows": n_res, "estimated_total": EST,
        "negative_results": neg, "negative_pct": round(neg / max(1, n_res), 4),
        "negative_estimated": int(neg / max(1, n) * EST),
        "with_retry": retry, "retry_estimated": int(retry / max(1, n) * EST),
        "unit_hist": dict(unit_hist),
        "extractable_measurements": ex,
        "extractable_estimated": int(ex / max(1, n) * EST),
        "protocol_signal": proto, "protocol_card_candidates": proto_rich,
        "protocol_card_estimated": int(proto_rich / max(1, n) * EST),
        "replicate_records": repl, "dated_rows": dated,
        "unique_notes": len(notes), "notes_3plus_pages": multi,
        "negative_by_domain": {d: round(neg_by_dom[d] / max(1, c), 4)
                               for d, c in res_by_dom.items()},
        "units_per_row_by_domain": {d: round(unit_by_dom[d] / max(1, c), 2)
                                    for d, c in res_by_dom.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
