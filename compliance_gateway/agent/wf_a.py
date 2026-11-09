"""WF-A 과제 기획 워크플로우 — 탐색·기획 에이전트(아키텍처 ③-①) 첫 작동판.

    goal(목표 설정) → search(선행연구 탐색) → gap(갭 분석, LLM) → draft(제안서 초안, LLM)
    → verify(Gateway 검증) → block_review(차단 시에만, 사람) → approve(연구자 승인, 사람)

- verify: 초안을 `ComplianceGateway.evaluate` 에 넣는다. 근거 = 탐색한 논문의 제목·DOI·초록.
  REGENERATE 면 뒷받침 안 된 문장을 알려주고 다시 쓰게 한다(게이트웨이 `max_regenerations` 까지).
  BLOCK 이면 `block_review` 사람 노드로 넘어간다.
- approve 는 항상 사람 노드다 — 에이전트는 스스로 승인하지 않는다(아키텍처 ③ HITL 필수).

주입점: `llm(prompt, system) -> str`, `search_fn(topic) -> list[PaperRecord]`, `gateway`, `approve`.
기본 LLM 환경변수: `GOONO_SERVING_URL`(OpenAI 호환 엔드포인트), `GOONO_SERVING_MODEL`(모델 ID).
인증키는 ServedModel 규약대로 `OPENAI_API_KEY` 또는 `KT_API_KEY`.

    python -m compliance_gateway.agent.wf_a "주제" --base-url http://host:8000/v1 --model Qwen/Qwen3.8-27B
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Optional

from compliance_gateway.agent.dag import Approve, Node, RunResult, automation_rate, run_dag
from compliance_gateway.models import GatewayDecision, GenerationRequest
from compliance_gateway.pipeline import ComplianceGateway
from compliance_gateway.verify.backends import OpenAlexBackend
from compliance_gateway.verify.models import PaperRecord

LLM = Callable[[str, str], str]
SearchFn = Callable[[str], list]

SYSTEM = (
    "당신은 국가 R&D 과제 기획을 돕는 연구 보조 에이전트다. 선행연구에 관한 사실은 [근거] 에 있는 것만 쓰고, "
    "그 문장 끝에 (DOI: ...) 로 출처를 단다. 근거에 없는 수치·결과를 지어내지 않는다."
)
GAP_PROMPT = (
    "주제: {topic}\n\n[근거]\n{sources}\n\n"
    "위 선행연구로 이미 밝혀진 것과 아직 비어 있는 것(연구 갭)을 항목별로 정리하라. 밝혀진 것에는 DOI 를 단다."
)
DRAFT_PROMPT = (
    "주제: {topic}\n\n[근거]\n{sources}\n\n[갭 분석]\n{gap}\n\n"
    "위 내용으로 R&D 과제 제안서 초안을 써라: 1) 연구 필요성(선행연구 문장마다 DOI) 2) 연구 목표 "
    "3) 연구 내용 4) 기대 효과."
)
REGEN_PROMPT = (
    "아래 초안이 출처 검증(Compliance Gateway)을 통과하지 못했다(VCR {vcr:.2f}). 재작성하라.\n"
    "근거가 뒷받침하지 않는 문장:\n{unsupported}\n\n[근거]\n{sources}\n\n[초안]\n{draft}\n\n"
    "뒷받침되지 않는 문장은 근거에 맞게 고치거나 지우고, 근거 문장에는 (DOI: ...) 를 달아 초안 전체를 다시 써라."
)


def _source_text(i: int, r: PaperRecord) -> str:
    head = f"[{i}] {r.title}" + (f" ({r.year})" if r.year else "") + (f" DOI: {r.doi}" if r.doi else "")
    return f"{head}. {r.abstract}".strip()


def build_wf_a(llm: LLM, search_fn: SearchFn, gateway: ComplianceGateway,
               model_id: str = "unknown", search_backoff: float = 2.0) -> list[Node]:
    def sources(ctx: dict) -> str:
        return "\n".join(ctx["search"]["grounding"])

    def goal(ctx: dict) -> str:
        topic = " ".join(str(ctx.get("topic", "")).split())
        if not topic:
            raise ValueError("주제가 비었다")
        return topic

    def search(ctx: dict) -> dict:
        records = search_fn(ctx["goal"])
        grounding = tuple(_source_text(i, r) for i, r in enumerate(records, 1) if r.title or r.abstract)
        if not grounding:   # OpenAlex 는 네트워크 실패도 빈 목록으로 돌려준다 → 재시도 대상
            raise RuntimeError("선행연구 0건 — 근거 없이 기획하지 않는다")
        return {"records": records, "grounding": grounding}

    def gap(ctx: dict) -> str:
        return llm(GAP_PROMPT.format(topic=ctx["goal"], sources=sources(ctx)), SYSTEM)

    def draft(ctx: dict) -> str:
        return llm(DRAFT_PROMPT.format(topic=ctx["goal"], sources=sources(ctx), gap=ctx["gap"]), SYSTEM)

    def verify(ctx: dict) -> dict:
        text, regen = ctx["draft"], 0
        while True:
            res = gateway.evaluate(GenerationRequest(
                query=ctx["goal"], response=text, grounding=ctx["search"]["grounding"],
                model_id=model_id), regenerations=regen)
            unsupported = next(r for r in res.reports if r.stage == "nli_gating").detail.get(
                "unsupported_sentences", [])
            if res.decision is not GatewayDecision.REGENERATE:
                break
            regen += 1
            listed = "\n".join(f"- {u['sentence']}" for u in unsupported) or "- (문장 게이팅 미설정 — 인용·출처 일치도 미달)"
            text = llm(REGEN_PROMPT.format(vcr=res.vcr.vcr, unsupported=listed, sources=sources(ctx),
                                           draft=text), SYSTEM)
        return {"decision": res.decision.value, "draft": text, "regenerations": regen,
                "vcr": res.vcr.vcr, "unsupported": unsupported, "audit_hash": res.audit_hash}

    def approved(ctx: dict) -> dict:
        v = ctx["verify"]
        return {"draft": v["draft"], "audit_hash": v["audit_hash"],
                "gateway_override": ctx.get("block_review") is not None}

    return [
        Node("goal", goal),
        Node("search", search, deps=("goal",), retries=2, backoff=search_backoff),
        Node("gap", gap, deps=("search",), retries=1),
        Node("draft", draft, deps=("gap",), retries=1),
        Node("verify", verify, deps=("draft",)),
        Node("block_review", deps=("verify",), human=True,
             when=lambda ctx: ctx["verify"]["decision"] == GatewayDecision.BLOCK.value),
        Node("approve", approved, deps=("verify", "block_review"), human=True),
    ]


def served_llm(base_url: Optional[str] = None, model: Optional[str] = None,
               max_tokens: int = 2048) -> LLM:
    """ServedModel 래퍼. 인자가 없으면 GOONO_SERVING_URL / GOONO_SERVING_MODEL."""
    from compliance_gateway.serving.openai_client import ServedModel

    base_url = base_url or os.getenv("GOONO_SERVING_URL")
    model = model or os.getenv("GOONO_SERVING_MODEL")
    if not base_url or not model:
        raise ValueError("LLM 엔드포인트 미설정 — GOONO_SERVING_URL / GOONO_SERVING_MODEL")
    m = ServedModel(base_url, model)
    return lambda prompt, system: m.generate(prompt, system=system, max_tokens=max_tokens)


def run_wf_a(topic: str, llm: Optional[LLM] = None, search_fn: Optional[SearchFn] = None,
             gateway: Optional[ComplianceGateway] = None, approve: Optional[Approve] = None,
             model_id: str = "unknown") -> RunResult:
    """WF-A 실행. approve 미지정이면 승인 단계에서 `pending_human` 으로 멈춘다."""
    nodes = build_wf_a(
        llm or served_llm(),
        search_fn or (lambda t: OpenAlexBackend().search(t, limit=5)),
        gateway or ComplianceGateway(),
        model_id=model_id,
    )
    return run_dag(nodes, ctx={"topic": topic}, approve=approve)


def _ask(node: Node, ctx: dict) -> bool:
    """CLI 승인 — 초안을 보여주고 연구자에게 묻는다. 비대화형(stdin 없음)이면 거절 → pending_human."""
    v = ctx["verify"]
    print(f"\n----- 제안서 초안 (Gateway {v['decision']}, VCR {v['vcr']}, 재생성 {v['regenerations']}회) -----")
    print(v["draft"])
    for u in v["unsupported"]:
        print(f"  ! 뒷받침 안 됨({u['score']}): {u['sentence']}")
    question = "Gateway 가 차단했다. 연구자 책임으로 진행할까요?" if node.name == "block_review" else "초안을 승인할까요?"
    try:
        return input(f"[{node.name}] {question} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="WF-A 과제 기획 워크플로우")
    ap.add_argument("topic", help="과제 주제")
    ap.add_argument("--base-url", default=os.getenv("GOONO_SERVING_URL"), help="OpenAI 호환 엔드포인트")
    ap.add_argument("--model", default=os.getenv("GOONO_SERVING_MODEL"), help="서빙 모델 ID")
    ap.add_argument("--limit", type=int, default=5, help="선행연구 탐색 건수(OpenAlex)")
    a = ap.parse_args(argv)
    if not a.base_url or not a.model:
        ap.error("--base-url / --model 또는 GOONO_SERVING_URL / GOONO_SERVING_MODEL 이 필요하다")

    res = run_wf_a(a.topic, llm=served_llm(a.base_url, a.model),
                   search_fn=lambda t: OpenAlexBackend().search(t, limit=a.limit),
                   approve=_ask, model_id=a.model)
    print("\n단계 기록")
    for e in res.log:
        err = f"  {e['error']}" if "error" in e else ""
        print(f"  {e['node']:<13} {e['status']:<8} 시도 {e['attempts']}  {e['seconds']:.2f}s{err}")
    print(f"상태: {res.status}   자동화율: {automation_rate(res.log):.1%}")


if __name__ == "__main__":
    main()
