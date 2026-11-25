#!/usr/bin/env python3
"""검수 시트(240문장)에 증류 인코더 점수를 매긴다 — AI 패널 기준으로 인코더 v3 대 v6a 를 비교하려고(연구노트 §63).

    /usr/bin/python scripts/enc_review.py data/labeling/review_judges_A.csv data/labeling/enc_review.json \
        checkpoints/nli_distill checkpoints/nli_distill_v6a checkpoints/nli_distill_v6a_s1 checkpoints/nli_distill_v6a_s2
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compliance_gateway.nli.transformer import TransformerNLI  # noqa: E402
from compliance_gateway.pipeline import _windows  # noqa: E402

sheet, out, *students = sys.argv[1:]
with open(sheet, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
res = {r["id"]: {} for r in rows}
for st in students:
    enc = TransformerNLI(st, device="cuda")
    name = Path(st).name
    for r in rows:
        res[r["id"]][name] = round(max(float(enc(w, r["문장"])) for w in _windows(r["원문"])), 5)
    print(name, "done", flush=True)
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(out, len(res))
