#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""제안서 과제 학습 데이터 시범 생성(연구노트 §62) — WF-A 와 같은 프롬프트 · 같은 게이트웨이로 교사(기반 모델)의 답을 걸러 SFT 예로 만든다.

§61: 단락 과제로만 학습한 어댑터는 제안서 초안(절 구성 · 문장마다 DOI)을 쓰지 못한다. 그 과제를 학습 데이터에 넣으면 되는지 보려는 시범.

단계(주제마다): 갭 분석 → 초안 → 게이트웨이 판정. 걸린 초안은 한 번 재작성하고 다시 판정한다.
채택 기준(모두): 게이트웨이 통과 · 인용 DOI 3개 이상 · 인용이 모두 근거 목록 안 · 제목(#)으로 시작.
출력 jsonl 한 줄 = 학습 예 하나 {"messages": [...], "kind": gap|draft|regen, "topic": ...} — 통과한 주제의 갭 분석 · 초안, 재작성으로 통과한 경우 재작성 예.

    python scripts/gen_proposals.py --model Qwen/Qwen3.8-27B --topics ../data/prop_topics.json \
        --search-cache ../data/prop_search.json --out data/instr/prop_train.jsonl --log ../logs/prop_gen.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

from compliance_gateway.agent.wf_a import DRAFT_PROMPT, GAP_PROMPT, REGEN_PROMPT, SYSTEM, _source_text, feedback
from compliance_gateway.models import GenerationRequest
from compliance_gateway.pipeline import ComplianceGateway
from compliance_gateway.vcr.source_exist import extract_citations
from compliance_gateway.verify.models import PaperRecord


def accept(res, draft: str, known: set[str]) -> bool:
    dois = [c.doi.lower() for c in extract_citations(draft) if c.doi]
    return (res.decision.value == "pass" and len(dois) >= 3 and all(d in known for d in dois)
            and draft.lstrip().startswith("#"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--topics", required=True, help="[[국문 주제, 검색어, id], ...]")
    ap.add_argument("--search-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--min-abstracts", type=int, default=3)
    ap.add_argument("--skip-log", nargs="*", default=[], help="이미 생성한 주제(앞 묶음의 --log)는 건너뛴다")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = json.load(open(a.search_cache, encoding="utf-8"))
    done = {json.loads(l)["topic"] for p in a.skip_log for l in open(p, encoding="utf-8") if l.strip()}
    items = []
    for t in json.load(open(a.topics, encoding="utf-8")):
        if t[0] in done:
            continue
        recs = [PaperRecord(**{**r, "authors": tuple(r["authors"])}) for r in cache.get(t[1], [])]
        if sum(1 for r in recs if r.abstract) < a.min_abstracts or sum(1 for r in recs if r.doi) < 3:
            continue
        grounding = tuple(_source_text(i, r) for i, r in enumerate(recs, 1) if r.title or r.abstract)
        items.append({"topic": t[0], "query": t[1], "records": recs, "sources": "\n".join(grounding), "grounding": grounding})
    print(f"주제 {len(items)}개(초록 {a.min_abstracts}건 이상)", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()

    def gen(prompts: list[str]) -> list[str]:
        out = []
        for i in range(0, len(prompts), a.batch):
            texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for p in prompts[i: i + a.batch]]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                o = m.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
            out += [tok.decode(s, skip_special_tokens=True).strip() for s in o[:, enc["input_ids"].shape[1]:]]
            print(f"  {min(i + a.batch, len(prompts))}/{len(prompts)} {time.strftime('%H:%M')}", flush=True)
        return out

    gw = ComplianceGateway()
    msg = lambda p, ans: [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}, {"role": "assistant", "content": ans}]

    gp = [GAP_PROMPT.format(topic=it["topic"], sources=it["sources"]) for it in items]
    for it, p, g in zip(items, gp, gen(gp)):
        it["gap_prompt"], it["gap"] = p, g
    dp = [DRAFT_PROMPT.format(topic=it["topic"], sources=it["sources"], gap=it["gap"]) for it in items]
    for it, p, d in zip(items, dp, gen(dp)):
        it["draft_prompt"], it["draft"] = p, d
        it["known"] = {(r.doi or "").lower() for r in it["records"] if r.doi}
        it["res"] = gw.evaluate(GenerationRequest(query=it["topic"], response=d, grounding=it["grounding"], model_id=a.model))
        it["ok"] = accept(it["res"], d, it["known"])

    redo = [it for it in items if not it["ok"]]
    rp = [REGEN_PROMPT.format(vcr=it["res"].vcr.vcr, threshold=gw.vcr_threshold, problems=feedback(it["res"], it["records"]),
                              sources=it["sources"], draft=it["draft"]) for it in redo]
    for it, p, d in zip(redo, rp, gen(rp)):
        res = gw.evaluate(GenerationRequest(query=it["topic"], response=d, grounding=it["grounding"], model_id=a.model), regenerations=1)
        it["regen_prompt"], it["regen"], it["regen_ok"], it["regen_vcr"] = p, d, accept(res, d, it["known"]), res.vcr.vcr

    n = {"gap": 0, "draft": 0, "regen": 0}
    with open(a.out, "w", encoding="utf-8") as f, open(a.log, "w", encoding="utf-8") as lg:
        for it in items:
            rows = []
            if it["ok"]:
                rows = [("gap", it["gap_prompt"], it["gap"]), ("draft", it["draft_prompt"], it["draft"])]
            elif it.get("regen_ok"):
                rows = [("gap", it["gap_prompt"], it["gap"]), ("regen", it["regen_prompt"], it["regen"])]
            for kind, p, ans in rows:
                f.write(json.dumps({"messages": msg(p, ans), "kind": kind, "topic": it["topic"], "source": "proposal"}, ensure_ascii=False) + "\n")
                n[kind] += 1
            lg.write(json.dumps({"topic": it["topic"], "ok": it["ok"], "vcr": it["res"].vcr.vcr, "decision": it["res"].decision.value,
                                 "draft_len": len(it["draft"]), "cited": len([c for c in extract_citations(it["draft"]) if c.doi]),
                                 "regen_ok": it.get("regen_ok"), "regen_vcr": it.get("regen_vcr")}, ensure_ascii=False) + "\n")
    ok = sum(it["ok"] for it in items)
    rok = sum(bool(it.get("regen_ok")) for it in items)
    print(f"채택: 첫 초안 {ok}/{len(items)}, 재작성으로 {rok}/{len(redo)} · 학습 예 {n}")


if __name__ == "__main__":
    main()
