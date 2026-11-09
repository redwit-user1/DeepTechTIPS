#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen 판정 → 1차 인코더 증류용 학습셋 생성 (GPU).

    gen    단락마다 Qwen 이 보고서식 문장 8종을 쓴다(의도 kind 포함)
    label  Qwen v2 판정기가 각 (단락, 문장)을 채점 → 의도와 판정이 **일치하는 것만** 남긴다

자동 학습셋 v1·v2 는 규칙으로 만든 라벨이라 규칙과 순환했다(연구노트 2026-11-09 §3).
여기서는 문장을 LLM 이 쓰고, 라벨은 **별도 판정 패스**가 정한다. 둘이 어긋나면 버린다.

    python scripts/distill.py gen   --passages data/distill/passages.jsonl --out data/distill/gen.jsonl
    python scripts/distill.py label --gen data/distill/gen.jsonl --out data/distill/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GEN_PROMPT = """다음은 연구노트 원문 단락이다. 이 단락을 보고서로 옮길 때 나올 수 있는 한국어 문장을 아래 종류별로 하나씩 써라.
각 줄은 JSON 한 줄: {{"kind": "...", "text": "..."}}. 해당 종류를 만들 수 없으면 그 줄은 쓰지 마라.

- faithful: 원문 표현에 가깝게, 원문이 말한 사실만
- paraphrase: 같은 뜻을 다른 어휘·구조로(수치·단위 그대로)
- approx: 반올림·'약'·올바른 단위 환산 등 원문이 여전히 뒷받침하는 근사
- altered_number: 원문의 수치 하나를 확실히 다른 값으로(단위 유지) — 원문과 모순
- altered_direction: 증가↔감소, 높다↔낮다, 이상↔이하, 비교 대상 뒤바꿈 — 원문과 모순
- unit_swap: 같은 수치에 크기가 다른 단위(µg↔mg, mM↔µM, 분↔시간) — 원문과 모순
- hedge_drop: 원문이 추정(판단됨·보임)일 때 그것을 확정(확인됨·입증됨)으로 — 원문만으로는 확인 불가
- unsupported: 원문 표현을 대부분 유지하되 원문에 없는 원인·효과·일반화를 하나 덧붙임

원문:
{passage}"""

INTENT = {"faithful": "entail", "paraphrase": "entail", "approx": "entail",
          "altered_number": "contradict", "altered_direction": "contradict", "unit_swap": "contradict",
          "hedge_drop": "neutral", "unsupported": "neutral"}
_JSON = re.compile(r"\{[^{}]*\"kind\"[^{}]*\}")


def gen(a) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = [json.loads(l) for l in open(a.passages, encoding="utf-8")]
    done = {json.loads(l)["doc"] for l in open(a.out, encoding="utf-8")} if Path(a.out).exists() else set()
    rows = [r for r in rows if r["doc"] not in done]          # 중단 후 재개
    with open(a.out, "a", encoding="utf-8") as f:
        for i in range(0, len(rows), a.batch):
            chunk = rows[i: i + a.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": GEN_PROMPT.format(passage=r["passage"])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=700, do_sample=True, temperature=0.7, top_p=0.9)
            for r, o in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
                for js in _JSON.findall(tok.decode(o, skip_special_tokens=True)):
                    try:
                        h = json.loads(js)
                    except json.JSONDecodeError:
                        continue
                    if h.get("kind") in INTENT and len(h.get("text", "")) >= 10:
                        f.write(json.dumps({"doc": r["doc"], "premise": r["passage"], "hypothesis": h["text"],
                                            "kind": h["kind"], "intent": INTENT[h["kind"]]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{min(i + a.batch, len(rows))}/{len(rows)}", flush=True)


def label(a) -> None:
    from compliance_gateway.nli.llm_judge import LLMJudgeNLI

    rows = [json.loads(l) for l in open(a.gen, encoding="utf-8")]
    scores = LLMJudgeNLI(a.model, batch_size=32).batch([(r["premise"], r["hypothesis"]) for r in rows])
    kept = drop = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r, s in zip(rows, scores):
            # 의도와 판정이 일치하는 것만 — 확신 구간(0.8 이상 / 0.2 이하)
            agree = (r["intent"] == "entail" and s >= 0.8) or (r["intent"] != "entail" and s <= 0.2)
            if not agree:
                drop += 1
                continue
            f.write(json.dumps({**r, "label": r["intent"], "teacher": round(s, 4)}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"판정 {len(rows)} → 채택 {kept} / 불일치·애매 폐기 {drop}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--passages", required=True); g.add_argument("--out", required=True)
    g.add_argument("--batch", type=int, default=16)
    l = sub.add_parser("label"); l.add_argument("--gen", required=True); l.add_argument("--out", required=True)
    for p in (g, l):
        p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    a = ap.parse_args()
    {"gen": gen, "label": label}[a.cmd](a)
