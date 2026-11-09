"""R&D 에이전트 v1 테스트 — DAG·메모리·라우터·WF-A. 가짜 llm/search/approve 로 결정적으로."""

import json
import re
import threading

import pytest

from compliance_gateway.agent import router as router_mod
from compliance_gateway.agent.dag import Node, automation_rate, run_dag
from compliance_gateway.agent.memory import ProjectMemory, SensitiveContentError
from compliance_gateway.agent.router import LOCAL_CONTEXT_TOKENS, route
from compliance_gateway.agent.wf_a import build_wf_a, run_wf_a
from compliance_gateway.pipeline import ComplianceGateway
from compliance_gateway.verify.models import PaperRecord

YES = lambda node, ctx: True          # noqa: E731


# ---- DAG ---------------------------------------------------------------
def test_dag_order_and_parallel():
    order = []
    barrier = threading.Barrier(2, timeout=5)    # b·c 가 동시에 돌지 않으면 깨진다

    def step(name, wait=False):
        def fn(ctx):
            if wait:
                barrier.wait()
            order.append(name)
            return name.upper()
        return fn

    res = run_dag([
        Node("a", step("a")),
        Node("b", step("b", wait=True), deps=("a",)),
        Node("c", step("c", wait=True), deps=("a",)),
        Node("d", lambda ctx: ctx["b"] + ctx["c"], deps=("b", "c")),
    ])
    assert res.status == "done"
    assert order[0] == "a" and set(order[1:]) == {"b", "c"}
    assert res.ctx["d"] == "BC"
    assert [e["status"] for e in res.log] == ["auto"] * 4
    assert {"node", "status", "attempts", "seconds"} <= res.log[0].keys()


def test_dag_skip_keeps_dependents_running():
    res = run_dag([
        Node("a", lambda ctx: 1),
        Node("opt", lambda ctx: 1 / 0, deps=("a",), when=lambda ctx: ctx["a"] > 5),
        Node("z", lambda ctx: ctx["opt"] is None, deps=("opt",)),
    ])
    assert res.status == "done"
    assert {e["node"]: e["status"] for e in res.log} == {"a": "auto", "opt": "skipped", "z": "auto"}
    assert res.ctx["z"] is True


def test_dag_retry_then_fail_stops_run():
    calls = {"n": 0}

    def flaky(ctx):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("일시 오류")
        return "ok"

    ok = run_dag([Node("f", flaky, retries=2, backoff=0.001)])
    assert ok.status == "done" and ok.log[0]["attempts"] == 3 and ok.log[0]["status"] == "auto"

    bad = run_dag([Node("f", lambda ctx: 1 / 0, retries=1), Node("g", lambda ctx: 1, deps=("f",))])
    assert bad.status == "failed"
    assert bad.log == [{**bad.log[0], "node": "f", "status": "failed", "attempts": 2}]
    assert "ZeroDivisionError" in bad.log[0]["error"] and "g" not in bad.ctx


def test_dag_human_node_pending_without_approval():
    nodes = [Node("a", lambda ctx: 1), Node("ok", lambda ctx: "approved", deps=("a",), human=True),
             Node("after", lambda ctx: 2, deps=("ok",))]
    res = run_dag(nodes)                               # 승인 콜백 없음 → 스스로 승인하지 않는다
    assert res.status == "pending_human"
    assert [(e["node"], e["status"]) for e in res.log] == [("a", "auto"), ("ok", "pending")]
    assert "ok" not in res.ctx and "after" not in res.ctx

    seen = []
    res2 = run_dag(nodes, approve=lambda node, ctx: seen.append((node.name, ctx["a"])) or True)
    assert res2.status == "done" and seen == [("ok", 1)]
    assert res2.ctx["ok"] == "approved"
    assert [e["status"] for e in res2.log] == ["auto", "human", "auto"]


def test_dag_rejects_cycles_and_unknown_deps():
    from graphlib import CycleError

    with pytest.raises(CycleError):
        run_dag([Node("a", deps=("b",)), Node("b", deps=("a",))])
    with pytest.raises(ValueError):
        run_dag([Node("a", deps=("nope",))])


def test_automation_rate_math():
    log = [{"status": s} for s in ["auto", "auto", "auto", "human", "failed", "pending", "skipped", "skipped"]]
    assert automation_rate(log) == 3 / 6                # skipped 는 분모에서 제외
    assert automation_rate([{"status": "auto"}] * 17 + [{"status": "human"}] * 3) == 0.85
    assert automation_rate([]) == 0.0
    assert automation_rate([{"status": "skipped"}]) == 0.0


# ---- 메모리 L1/L2 --------------------------------------------------------
def test_memory_refuses_sensitive_and_deidentifies(tmp_path):
    path = tmp_path / "l2.jsonl"
    mem = ProjectMemory(path, salt="test-salt")
    mem.l1["draft"] = "초안"                             # L1 은 그냥 dict

    with pytest.raises(SensitiveContentError) as e:
        mem.promote("관찰", "환자 12명 중 3명에서 증상 완화", "임상 결과")
    assert "환자" not in str(e.value)                    # 오류 메시지에 원문을 남기지 않는다
    assert not path.exists()

    rec = mem.promote("제안서 v1", "연구자: 김민준 확인, 연락처 010-1234-5678. pH 7.4 반응속도 2.3배", "연구자 승인")
    raw = path.read_text(encoding="utf-8")
    assert "김민준" not in raw and "010-1234-5678" not in raw
    assert "연구자_" in rec["value"] and rec["deidentified"] >= 2

    mem.promote("공정 메모", "용접 전류 270A 에서 열영향부 폭이 좁아짐", "재현 조건")
    assert len(raw.splitlines()) == 1 and len(path.read_text(encoding="utf-8").splitlines()) == 2  # append-only
    hits = mem.recall("반응속도 pH")
    assert [h["key"] for h in hits] == ["제안서 v1"]
    assert mem.recall("용접") and mem.recall("없는단어") == [] and mem.recall("") == []
    assert all(json.loads(l)["ts"] for l in path.read_text(encoding="utf-8").splitlines())


# ---- 라우터 -------------------------------------------------------------
def test_router_sensitive_goes_local():
    r = route("환자 12명 중 3명에서 증상 완화 — 후속 과제 제안서 초안을 써라", task="draft")
    assert r.target == "local" and "민감" in r.reason


def test_router_deidentifies_before_api_and_routes_by_task():
    r = route("연구자: 김민준 의 촉매 반응속도 결과를 바탕으로 제안서 초안", task="draft")
    assert r.target == "api" and "김민준" not in r.text and r.est_cost_usd > 0
    assert route("이 문장을 분류하라", task="classify").target == "local"
    # 띄어쓰기 없는 한글 연속열은 deidentify 정규식이 O(n²) 라 자연 문장으로 만든다
    long_text = "촉매 반응속도 결과를 정리한다. " * int(LOCAL_CONTEXT_TOKENS * 1.6 / 17)
    assert route(long_text, task="classify").target == "api"           # 로컬 컨텍스트 초과
    assert route("초안", task="draft", max_cost_usd=0.0).target == "local"  # 비용 상한


def test_router_residual_pii_goes_local(monkeypatch):
    monkeypatch.setattr(router_mod, "audit", lambda text: {"휴대전화": 1})
    r = route("일반 공정 서술", task="draft")
    assert r.target == "local" and "잔존" in r.reason


# ---- WF-A ---------------------------------------------------------------
PAPERS = [
    PaperRecord(doi="10.1234/abc", title="Catalyst kinetics at physiological pH", year=2024,
                abstract="반응속도가 pH 7.4 에서 2.3배 향상되었다."),
    PaperRecord(doi="10.5678/xyz", title="Enzyme stability", year=2023, abstract="37°C 에서 안정성이 유지되었다."),
]
UNSUPPORTED = "본 기술은 수율을 99% 향상시킨다."
GROUNDED = "선행연구에 따르면 pH 7.4 에서 반응속도가 2.3배 향상되었다 (DOI: 10.1234/abc)."


def _nums(t):
    return re.findall(r"\d+(?:\.\d+)?", t)


def _fake_nli(premise, hypothesis):
    """가설의 숫자가 모두 근거에 있으면 뒷받침(1.0), 아니면 0.0."""
    n = _nums(hypothesis)
    return 1.0 if n and all(x in _nums(premise) for x in n) else 0.0


def _gateway():
    # VCR 은 끄고(0.0) 문장 게이팅만으로 판정 — 숫자 지어내기를 잡는지 본다
    return ComplianceGateway(vcr_threshold=0.0, nli_fn=_fake_nli, sentence_threshold=0.5, max_regenerations=2)


class FakeLLM:
    def __init__(self, drafts):
        self.drafts = list(drafts)        # 초안 → 재작성 순서로 돌려줄 응답
        self.prompts = []

    def __call__(self, prompt, system):
        self.prompts.append(prompt)
        if "[갭 분석]" in prompt or "재작성하라" in prompt:
            return self.drafts.pop(0)
        return "갭: 37°C 이외 온도에서의 반응속도는 보고되지 않았다."


def test_wf_a_regenerates_unsupported_claim_then_passes():
    llm = FakeLLM([UNSUPPORTED, GROUNDED])
    searched, approvals = [], []
    res = run_wf_a("  촉매 반응속도  개선 ", llm=llm,
                   search_fn=lambda t: searched.append(t) or PAPERS, gateway=_gateway(),
                   approve=lambda node, ctx: approvals.append(node.name) or True)

    assert res.status == "done" and searched == ["촉매 반응속도 개선"]
    v = res.ctx["verify"]
    assert v["decision"] == "pass" and v["regenerations"] == 1 and v["draft"] == GROUNDED
    assert "99%" in llm.prompts[-1] and "10.1234/abc" in llm.prompts[-1]   # 재작성 프롬프트에 위반 문장·근거
    assert approvals == ["approve"]                                     # 차단 없음 → 승인 1회
    assert res.ctx["approve"]["draft"] == GROUNDED and not res.ctx["approve"]["gateway_override"]
    assert {e["node"]: e["status"] for e in res.log} == {
        "goal": "auto", "search": "auto", "gap": "auto", "draft": "auto",
        "verify": "auto", "block_review": "skipped", "approve": "human"}
    assert automation_rate(res.log) == 5 / 6


def test_wf_a_block_goes_to_human_and_never_self_approves():
    llm = FakeLLM([UNSUPPORTED] * 3)                    # 재생성 2회를 다 써도 근거 없음 → BLOCK
    res = run_wf_a("촉매", llm=llm, search_fn=lambda t: PAPERS, gateway=_gateway())
    assert res.ctx["verify"]["decision"] == "block" and res.ctx["verify"]["regenerations"] == 2
    assert res.status == "pending_human" and res.log[-1] == {**res.log[-1], "node": "block_review",
                                                              "status": "pending"}
    assert "approve" not in res.ctx

    res2 = run_wf_a("촉매", llm=FakeLLM([UNSUPPORTED] * 3), search_fn=lambda t: PAPERS,
                    gateway=_gateway(), approve=YES)
    assert res2.status == "done" and res2.ctx["approve"]["gateway_override"]
    assert automation_rate(res2.log) == 5 / 7


def test_wf_a_no_literature_fails_after_retries():
    nodes = build_wf_a(FakeLLM([]), search_fn=lambda t: [], gateway=_gateway(), search_backoff=0)
    res = run_dag(nodes, ctx={"topic": "촉매"}, approve=YES)
    assert res.status == "failed"
    assert res.log[-1]["node"] == "search" and res.log[-1]["attempts"] == 3


def test_openalex_abstract_is_rebuilt_from_inverted_index():
    from compliance_gateway.verify.backends import OpenAlexBackend

    rec = OpenAlexBackend._to_record({
        "doi": "https://doi.org/10.1234/abc", "title": "T", "publication_year": 2024,
        "abstract_inverted_index": {"rate": [1], "Reaction": [0], "doubled": [2], "rate.": [4], "again": [3]},
    })
    assert rec.doi == "10.1234/abc" and rec.abstract == "Reaction rate doubled again rate."
    assert OpenAlexBackend._to_record({"title": "no abstract"}).abstract == ""
