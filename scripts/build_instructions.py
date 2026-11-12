#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R&D Instruction 데이터 — 연구노트 단락 → (지시, 응답, 출처) 트리플, **게이트웨이 통과분만** (GPU).

    gen   단락마다 Qwen 이 4종 과제(요약·조건 추출·보고서 문단·질의응답)의 지시와 응답을 쓴다
    gate  응답의 **모든 문장**을 원문 단락과 대조(CascadeNLI: 단위 바꿈 규칙 → 새 숫자는 LLM → 인코더 →
          애매 구간 LLM)해 하나라도 근거 없으면 버린다. 개인정보 잔존·민감 서술도 버린다.

AI 가 만든 데이터를 그대로 학습하면 순환오류가 난다(사업계획서 p.20). 게이트웨이를 데이터 필터로 쓰는 것이
이 과제의 Compliance Data Flywheel 이다 — 학습 데이터부터 규정준수 검증을 거친다.

    python scripts/build_instructions.py gen  --passages data/distill/passages.jsonl --out data/instr/gen.jsonl
    python scripts/build_instructions.py gate --gen data/instr/gen.jsonl --out data/instr/sft.jsonl
    python scripts/build_instructions.py dpo  --screened data/instr/sft.jsonl.screened.jsonl --out data/instr/dpo.jsonl
    python -m compliance_gateway.train.sft --dataset data/instr/sft.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

TASKS = {
    "summary": "다음 연구노트 단락의 실험 결과를 3문장 이내로 요약하라.",
    "conditions": "다음 연구노트 단락에서 실험 조건(시료·농도·온도·시간·장비 등)을 '항목: 값' 줄로 추출하라. 원문에 없는 항목은 쓰지 마라.",
    "report": "다음 연구노트 단락을 연차보고서 '연구개발 수행 내용' 문단으로 다시 써라. 원문에 없는 수치·해석·평가를 더하지 마라.",
    "qa": None,   # 질문도 모델이 만든다
}

GEN_PROMPT = """다음은 연구노트 원문 단락이다. 아래 과제마다 응답을 써라. 응답은 **원문이 말한 사실만** 담고,
원문에 없는 수치·원인·평가·일반화를 덧붙이지 마라. 원문의 수치·단위는 그대로 옮겨라.
각 줄은 JSON 한 줄: {{"task": "...", "question": "...", "answer": "..."}}

- task "summary": 실험 결과 3문장 이내 요약 (question 은 빈 문자열)
- task "conditions": 실험 조건을 '항목: 값' 줄로 (question 은 빈 문자열, 줄바꿈은 \\n)
- task "report": 연차보고서 '연구개발 수행 내용' 문단 (question 은 빈 문자열)
- task "qa": 원문만으로 답할 수 있는 구체적 질문 하나와 그 답

원문:
{passage}"""

_JSON = re.compile(r"\{[^{}]*\"task\"[^{}]*\}")


def prompt_of(task: str, question: str, passage: str) -> str:
    """학습용 사용자 프롬프트. 원문을 함께 준다 — 근거 기반 생성을 배우게 한다(출처 없는 생성 금지)."""
    ask = question if task == "qa" else TASKS[task]
    return f"{ask}\n\n[연구노트]\n{passage}"


def gen(a) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = [json.loads(l) for l in open(a.passages, encoding="utf-8")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["doc"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    rows = [r for r in rows if r["doc"] not in done]          # 중단 후 재개
    with out.open("a", encoding="utf-8") as f:
        for i in range(0, len(rows), a.batch):
            chunk = rows[i: i + a.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": GEN_PROMPT.format(passage=r["passage"])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                o = m.generate(**enc, max_new_tokens=900, do_sample=True, temperature=0.6, top_p=0.9,
                               num_return_sequences=a.samples)
            for k, seq in enumerate(o[:, enc["input_ids"].shape[1]:]):
                r, sample = chunk[k // a.samples], k % a.samples
                for js in _JSON.findall(tok.decode(seq, skip_special_tokens=True)):
                    try:
                        h = json.loads(js)
                    except json.JSONDecodeError:
                        continue
                    if h.get("task") in TASKS and len(h.get("answer", "")) >= 10 and (h["task"] != "qa" or h.get("question")):
                        f.write(json.dumps({"doc": r["doc"], "passage": r["passage"], "task": h["task"],
                                            "question": h.get("question", ""), "answer": h["answer"],
                                            "sample": sample}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{min(i + a.batch, len(rows))}/{len(rows)}", flush=True)


def screen(row: dict, nli, threshold: float) -> str:
    """버릴 사유(통과면 빈 문자열). nli(premise, hypothesis) -> 지지 확률."""
    from compliance_gateway.data.ocr.deidentify import audit
    from compliance_gateway.data.ocr.sensitive import sensitive_signals
    from compliance_gateway.pipeline import _split_all, _windows

    text = row["answer"] + ("\n" + row["question"] if row.get("question") else "")
    if audit(text):
        return "pii"
    if sensitive_signals(text):
        return "sensitive"
    chunks = _windows(row["passage"])
    for sent in _split_all(row["answer"]):
        if max(nli(c, sent) for c in chunks) < threshold:
            return "unsupported"
    return ""


def gate(a) -> None:
    from compliance_gateway.nli.cascade import CascadeNLI
    from compliance_gateway.train.data_format import to_sft_record

    nli = CascadeNLI.default()
    rows = [json.loads(l) for l in open(a.gen, encoding="utf-8")]
    why: Counter[str] = Counter()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # 통과·탈락 모두 사유와 함께 남긴다 — 같은 지시의 통과/탈락 응답이 DPO 선호쌍이 된다(`dpo`)
    with open(a.out, "w", encoding="utf-8") as f, open(a.out + ".screened.jsonl", "w", encoding="utf-8") as sf:
        for i, r in enumerate(rows):
            reason = screen(r, nli, nli.threshold)
            why[reason or "pass"] += 1
            sf.write(json.dumps({**r, "reason": reason or "pass"}, ensure_ascii=False) + "\n")
            if not reason:
                rec = to_sft_record(prompt_of(r["task"], r["question"], r["passage"]), r["answer"])
                rec["source"] = {"doc": r["doc"], "task": r["task"],
                                 "passage_sha256": hashlib.sha256(r["passage"].encode("utf-8")).hexdigest()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(rows)} {dict(why)}", flush=True)
    print(f"게이트 {len(rows)} → {dict(why)}")


def dpo_pairs(screened: list[dict]) -> list[dict]:
    """같은 (문서, 과제)의 통과 응답 = chosen, '근거 없는 문장' 탈락 응답 = rejected.

    질의응답은 표본마다 질문이 달라 지시가 같지 않으므로 뺀다. 개인정보·민감 탈락은 선호의 문제가 아니라 뺀다.
    rejected 가 '근거에 없는 내용을 덧붙인 답'이라 규정준수(출처 충실) 정렬에 바로 쓰인다.
    """
    from collections import defaultdict

    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: {"pass": [], "unsupported": []})
    for r in screened:
        if r["task"] != "qa" and r["reason"] in ("pass", "unsupported"):
            groups[(r["doc"], r["task"])][r["reason"]].append(r)
    out = []
    for (doc, task), g in groups.items():
        for good, bad in zip(g["pass"], g["unsupported"]):
            out.append({"prompt": prompt_of(task, "", good["passage"]), "chosen": good["answer"],
                        "rejected": bad["answer"], "source": {"doc": doc, "task": task}})
    return out


def dpo(a) -> None:
    rows = [json.loads(l) for l in open(a.screened, encoding="utf-8")]
    pairs = dpo_pairs(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"선호쌍 {len(pairs)} ← 심사 {len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--passages", required=True); g.add_argument("--out", required=True)
    g.add_argument("--batch", type=int, default=16); g.add_argument("--model", default="Qwen/Qwen3.8-27B")
    g.add_argument("--samples", type=int, default=1, help="지시당 표본 수(2 이상이면 DPO 선호쌍이 생긴다)")
    t = sub.add_parser("gate"); t.add_argument("--gen", required=True); t.add_argument("--out", required=True)
    d = sub.add_parser("dpo"); d.add_argument("--screened", required=True); d.add_argument("--out", required=True)
    a = ap.parse_args()
    for p in (getattr(a, "out", ""),):
        if p and not str(Path(p).resolve()).startswith(str(Path.cwd().resolve() / "data")):
            raise SystemExit(f"거부: 출력은 data/ 안이어야 한다(연구노트 원문 포함). 받은 값: {p}")
    {"gen": gen, "gate": gate, "dpo": dpo}[a.cmd](a)


if __name__ == "__main__":
    main()
