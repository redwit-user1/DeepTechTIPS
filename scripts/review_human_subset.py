#!/usr/bin/env python3
"""AI 패널이 확정하지 못한 문장만 사람에게 묻는 시트(연구노트 §63).

대상: 패널 넷 중 셋 이상이 같은 값을 내지 못한 문장 + 패널이 판정불가(x)로 본 문장 + 패널이 '틀림(0)'으로 본 문장.
시트에는 AI 판정을 보여 주지 않는다(먼저 본 답에 끌리지 않게). 순서는 섞는다.

    python scripts/review_human_subset.py make  data/labeling        # openpyxl 필요(시스템 python)        → review_judges_human46.xlsx / .csv
    python scripts/review_human_subset.py merge data/labeling        → review_judges_final.csv (사람 답이 패널을 덮음)
    python scripts/review_judges.py score data/labeling/review_judges_final.csv
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path


def read(p: Path) -> dict[str, dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def make(d: Path) -> None:
    panel = read(d / "review_judges_panel.csv")
    ids = [i for i, r in panel.items() if r["human_label"] in ("", "x", "0")]
    random.Random(7).shuffle(ids)
    head = ["번호", "id", "원문", "문장", "human_label", "메모"]
    rows = [[n, i, panel[i]["원문"], panel[i]["문장"], "", ""] for n, i in enumerate(ids, 1)]
    with (d / "review_judges_human46.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "검수"
    ws.append(head)
    for r in rows:
        ws.append(r)
    dv = DataValidation(type="list", formula1='"1,0,x"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(rows) + 1}")
    for col, width in zip("ABCDEF", [6, 8, 90, 60, 13, 40]):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows():
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
            c.font = Font(name="맑은 고딕", size=11)
    for c in ws[1]:
        c.fill = PatternFill("solid", fgColor="203864")
        c.font = Font(name="맑은 고딕", color="FFFFFF", bold=True)
    for r in range(2, len(rows) + 2):
        ws.row_dimensions[r].height = 120
        ws.cell(r, 5).fill = PatternFill("solid", fgColor="FFF2CC")
    ws.freeze_panes = "C2"
    g = wb.create_sheet("안내")
    for line in [
        ["원문만 보고 문장이 맞는지 판정해 주세요. 외부 지식은 쓰지 않습니다."],
        ["1 = 원문으로 확인됨. 반올림, '약', 맞는 단위 환산, 같은 뜻의 바꿔 쓰기는 1"],
        ["0 = 원문과 다름(값 · 단위 · 대상 · 방향 · 이상/이하), 또는 원문으로 확인 안 됨(덧붙인 원인 · 효과 · 일반화, 추측을 단정으로)"],
        ["x = 판정 불가(원문이 깨졌거나 문장이 잘림)"],
        ["'항목: 값' 줄은 값이 원문에 있고 항목 이름이 그 값을 가리키면 1. 값이 없거나 다른 대상의 값이면 0"],
        ["human_label 칸(노란색)에 1 / 0 / x 중 하나만 적어 주세요. 이유는 메모 칸에 자유롭게."],
    ]:
        g.append(line)
    g.column_dimensions["A"].width = 120
    wb.save(d / "review_judges_human46.xlsx")
    print(f"{len(rows)}문장 → {d / 'review_judges_human46.xlsx'}")


def merge(d: Path) -> None:
    panel = read(d / "review_judges_panel.csv")
    src = d / "review_judges_human46.xlsx"
    from openpyxl import load_workbook

    ws = load_workbook(src)["검수"]
    human = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[1] and r[4] is not None and str(r[4]).strip():
            v = str(r[4]).strip().lower().replace(".0", "")
            if v not in ("1", "0", "x"):
                raise SystemExit(f"거부: {r[1]} 의 human_label '{r[4]}' — 1 / 0 / x 만")
            human[r[1]] = v
    # 원본 240문장 시트(review_judges_A.csv)에 직접 적은 답도 쓴다. 두 곳에 다 있으면 46문장 시트가 우선
    with (d / "review_judges_A.csv").open(encoding="utf-8-sig", newline="") as f:
        extra = {r["id"]: r["human_label"].strip().lower() for r in csv.DictReader(f) if r["human_label"].strip()}
    for i, v in extra.items():
        human.setdefault(i, v)
    print(f"원본 시트의 사람 답 {len(extra)}건 더함 → 사람 답 합계 {len(human)}")
    both = [i for i in human if panel[i]["human_label"] in ("0", "1") and human[i] in ("0", "1")]
    if both:
        print(f"패널이 정한 문장 중 사람도 본 {len(both)}건: 일치 {sum(panel[i]['human_label'] == human[i] for i in both)} · "
              f"패널 1 → 사람 0 {sum(panel[i]['human_label'] == '1' and human[i] == '0' for i in both)} · "
              f"패널 0 → 사람 1 {sum(panel[i]['human_label'] == '0' and human[i] == '1' for i in both)}")
    todo = [i for i, r in panel.items() if r["human_label"] in ("", "x", "0")]
    print(f"사람 답 {len(human)} / {len(todo)}")
    agree0 = [i for i in todo if panel[i]["human_label"] == "0" and i in human]
    if agree0:
        print(f"패널 '틀림' {len(agree0)}건 중 사람도 0: {sum(human[i] == '0' for i in agree0)}")
    with (d / "review_judges_final.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "원문", "문장", "human_label", "메모"])
        for i, r in panel.items():
            lab = human[i] if i in human else (r["human_label"] if i not in todo else "")
            w.writerow([i, r["원문"], r["문장"], lab, ("사람" if i in human else "패널") + " " + r["메모"]])
    print(f"→ {d / 'review_judges_final.csv'}")


if __name__ == "__main__":
    {"make": make, "merge": merge}[sys.argv[1]](Path(sys.argv[2] if len(sys.argv) > 2 else "data/labeling"))
