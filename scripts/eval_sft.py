#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT 평가 — 학습에 안 쓴 연구노트 단락으로 '근거에 충실한 생성 비율'을 기반모델과 비교 (GPU).

KPI '보고서 자동 생성 정확도'의 **대리 지표**: 생성한 응답의 모든 문장이 원문 단락에 뒷받침되고(CascadeNLI)
개인정보·민감 서술이 없는 비율. 문장이 맞는지(사실 충실)만 보고, 보고서로서의 완결성·유용성은 보지 않는다.

    split  문서 해시로 10% 를 평가용으로 떼고 학습셋에서 뺀다(같은 문서의 다른 과제도 함께 뺀다)
    run    모델(+LoRA 어댑터)로 평가 지시에 답하고 게이트로 심사

    python scripts/eval_sft.py split --gen data/instr/gen.jsonl --sft data/instr/sft.jsonl --out data/instr
    python scripts/eval_sft.py run --model Qwen/Qwen3.5-9B [--adapter ckpt] --heldout data/instr/heldout.jsonl --out data/instr/eval_base.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_instructions import prompt_of, screen  # noqa: E402


def heldout_doc(doc: str, mod: int = 10) -> bool:
    return int(doc, 16) % mod == 0


def split(a) -> None:
    out = Path(a.out)
    gen = [json.loads(l) for l in open(a.gen, encoding="utf-8")]
    seen, held = set(), []
    for r in gen:
        key = (r["doc"], r["task"], r.get("question", ""))
        if heldout_doc(r["doc"]) and key not in seen:
            seen.add(key)
            held.append({"doc": r["doc"], "task": r["task"], "passage": r["passage"],
                         "prompt": prompt_of(r["task"], r.get("question", ""), r["passage"])})
    n_tr = 0
    with open(out / "sft_train.jsonl", "w", encoding="utf-8") as f:
        for l in open(a.sft, encoding="utf-8"):
            if not heldout_doc(json.loads(l)["source"]["doc"]):
                f.write(l)
                n_tr += 1
    with open(out / "heldout.jsonl", "w", encoding="utf-8") as f:
        for h in held:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"학습 {n_tr} · 평가 지시 {len(held)}(문서 {len({h['doc'] for h in held})})")


def run(a) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from compliance_gateway.nli.cascade import CascadeNLI

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if a.adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, a.adapter).eval()
    rows = [json.loads(l) for l in open(a.heldout, encoding="utf-8")]
    answers = []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i: i + a.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=400, do_sample=False)
        answers += [tok.decode(s, skip_special_tokens=True).strip() for s in o[:, enc["input_ids"].shape[1]:]]
        print(f"{min(i + a.batch, len(rows))}/{len(rows)}", flush=True)
    del m
    torch.cuda.empty_cache()
    nli = CascadeNLI.default()
    by: dict[str, Counter] = {}
    with open(a.out, "w", encoding="utf-8") as f:
        for r, ans in zip(rows, answers):
            reason = screen({"passage": r["passage"], "answer": ans, "question": ""}, nli, nli.threshold) if ans else "empty"
            by.setdefault(r["task"], Counter())[reason or "pass"] += 1
            f.write(json.dumps({**r, "answer": ans, "reason": reason or "pass"}, ensure_ascii=False) + "\n")
    tot = sum(by.values(), Counter())
    print(f"충실 생성률 {tot['pass'] / sum(tot.values()):.1%}  {dict(tot)}")
    for t, c in by.items():
        print(f"  {t:10s} {c['pass'] / sum(c.values()):.1%}  {dict(c)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("--gen", required=True); s.add_argument("--sft", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--adapter")
    r.add_argument("--heldout", required=True); r.add_argument("--out", required=True)
    r.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    {"split": split, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
