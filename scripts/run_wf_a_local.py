#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WF-A 를 GPU 의 로컬 Qwen 으로 여러 주제에 돌려 자동화율을 잰다(서빙 엔드포인트 없이).

연구자 최종 승인(gate)은 '승인'으로 가정한다 — 자동화율 분모에서 빠지는 단계다.
게이트웨이 차단 후 사람 검토(block_review)는 거절로 둔다 — 자동화 실패로 센다.

    python scripts/run_wf_a_local.py --model Qwen/Qwen3.8-27B --out logs/wf_a.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

from compliance_gateway.agent.dag import automation_rate
from compliance_gateway.agent.wf_a import run_wf_a
from compliance_gateway.agent.wf_a import search_papers
from compliance_gateway.vcr.source_exist import extract_citations
from compliance_gateway.verify.models import PaperRecord

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


def local_llm(model_id: str, max_new_tokens: int = 1500, adapter: str | None = None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if adapter:
        # LoRA 어댑터는 학습(`train/loader.py`)과 같은 클래스로 올려야 키가 맞는다(연구노트 §39) — 안 실리면 멈춘다
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
        m = PeftModel.from_pretrained(m, adapter).eval()
        b = [float(p.detach().abs().sum()) for n, p in m.named_parameters() if "lora_B" in n]
        if not b or not all(b):
            raise SystemExit(f"거부: 어댑터가 실리지 않았다 — lora_B {len(b)}개 중 {sum(v == 0 for v in b)}개가 0")
        print(f"[어댑터] lora_B {len(b)}개 모두 적재: {adapter}", flush=True)
    else:
        m = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()

    def llm(prompt: str, system: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return llm


def save_search(path: str, limit: int = 5) -> None:
    """검색만 미리 해 둔다. OpenAlex 무료 한도는 **IP 단위**라 공용 GPU 서버 IP 는 자주 소진된다(429)."""
    cache = {q: [dataclasses.asdict(r) for r in search_papers(q, limit=limit)] for _, q in TOPICS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print({q: len(v) for q, v in cache.items()})


def cached_search(path: str):
    cache = json.load(open(path, encoding="utf-8"))
    return lambda q: [PaperRecord(**{**r, "authors": tuple(r["authors"])}) for r in cache.get(q, [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--adapter", help="LoRA 어댑터 경로 — SLM v1 + Gateway v1 + 에이전트 v1 통합 실행")
    ap.add_argument("--search-cache", help="save_search 로 만든 검색 결과 JSON(없으면 OpenAlex 실시간)")
    a = ap.parse_args()
    search = cached_search(a.search_cache) if a.search_cache else (lambda q: search_papers(q, limit=a.limit))
    llm = local_llm(a.model, adapter=a.adapter)
    approve = lambda node, ctx: node.gate          # 최종 승인은 승인, 차단 검토는 거절
    logs = []
    with open(a.out, "w", encoding="utf-8") as f:
        for topic, query in TOPICS:
            t0 = time.time()
            res = run_wf_a(topic, llm=llm, approve=approve, model_id=a.model + ("+" + a.adapter.rstrip("/").rsplit("/", 1)[-1] if a.adapter else ""),
                           search_fn=lambda _t, q=query: search(q))
            v = res.ctx.get("verify") or {}
            rec = {"topic": topic, "status": res.status, "rate": automation_rate(res.log), "log": res.log,
                   "decision": v.get("decision"), "vcr": v.get("vcr"), "regenerations": v.get("regenerations"),
                   "n_sources": len((res.ctx.get("search") or {}).get("grounding", [])), "seconds": round(time.time() - t0, 1),
                   # 출처 정확률 KPI 계산용: 초안·인용 DOI·근거 DOI
                   "draft": v.get("draft", ""),
                   "cited_dois": [c.doi for c in extract_citations(v.get("draft", "")) if c.doi],
                   "source_dois": [r.doi for r in (res.ctx.get("search") or {}).get("records", []) if r.doi]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            logs += res.log
            print(f"{topic[:24]:26s} {res.status:14s} 자동화 {rec['rate']:.0%} 결정 {rec['decision']} VCR {rec['vcr']} "
                  f"재생성 {rec['regenerations']} 근거 {rec['n_sources']}", flush=True)
    print(f"전체 자동화율(단계 가중) {automation_rate(logs):.1%}  주제 {len(TOPICS)}")


if __name__ == "__main__":
    main()
