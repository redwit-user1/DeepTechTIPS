#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WF-A 를 GPU 의 로컬 Qwen 으로 여러 주제에 돌려 자동화율을 잰다(서빙 엔드포인트 없이).

연구자 최종 승인(gate)은 '승인'으로 가정한다 — 자동화율 분모에서 빠지는 단계다.
게이트웨이 차단 후 사람 검토(block_review)는 거절로 둔다 — 자동화 실패로 센다.

    python scripts/run_wf_a_local.py --model Qwen/Qwen3.8-27B --out logs/wf_a.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

from compliance_gateway.agent.dag import automation_rate
from compliance_gateway.agent.wf_a import run_wf_a
from compliance_gateway.verify.backends import OpenAlexBackend

# (과제 주제, OpenAlex 검색어) — 초기 집중 도메인(제약·바이오·연구데이터 무결성)
TOPICS = [
    ("항체-약물 접합체(ADC)의 링커 안정성 개선", "antibody drug conjugate linker stability"),
    ("오가노이드 기반 간독성 예측 모델", "liver organoid hepatotoxicity prediction"),
    ("mRNA 지질나노입자 전달체의 비강 투여", "lipid nanoparticle mRNA intranasal delivery"),
    ("전자연구노트의 데이터 무결성(ALCOA+) 자동 점검", "data integrity ALCOA electronic laboratory notebook"),
    ("GLP 독성시험 데이터의 감사추적 자동화", "GLP toxicology study audit trail"),
    ("마이크로바이옴 기반 염증성 장질환 바이오마커", "microbiome biomarker inflammatory bowel disease"),
    ("CAR-T 세포치료제의 사이토카인 방출 증후군 예측", "CAR-T cytokine release syndrome prediction"),
    ("대형 언어모델의 과학 문헌 인용 환각 탐지", "citation hallucination large language model scientific"),
    ("펩타이드 의약품의 경구 생체이용률 향상", "oral bioavailability peptide drugs permeation enhancer"),
    ("임상시험 프로토콜 이탈의 자동 탐지", "clinical trial protocol deviation detection"),
]


def local_llm(model_id: str, max_new_tokens: int = 1500):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()

    def llm(prompt: str, system: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=5)
    a = ap.parse_args()
    llm = local_llm(a.model)
    approve = lambda node, ctx: node.gate          # 최종 승인은 승인, 차단 검토는 거절
    logs = []
    with open(a.out, "w", encoding="utf-8") as f:
        for topic, query in TOPICS:
            t0 = time.time()
            res = run_wf_a(topic, llm=llm, approve=approve, model_id=a.model,
                           search_fn=lambda _t, q=query: OpenAlexBackend().search(q, limit=a.limit))
            v = res.ctx.get("verify") or {}
            rec = {"topic": topic, "status": res.status, "rate": automation_rate(res.log), "log": res.log,
                   "decision": v.get("decision"), "vcr": v.get("vcr"), "regenerations": v.get("regenerations"),
                   "n_sources": len((res.ctx.get("search") or {}).get("grounding", [])), "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            logs += res.log
            print(f"{topic[:24]:26s} {res.status:14s} 자동화 {rec['rate']:.0%} 결정 {rec['decision']} VCR {rec['vcr']} "
                  f"재생성 {rec['regenerations']} 근거 {rec['n_sources']}", flush=True)
    print(f"전체 자동화율(단계 가중) {automation_rate(logs):.1%}  주제 {len(TOPICS)}")


if __name__ == "__main__":
    main()
