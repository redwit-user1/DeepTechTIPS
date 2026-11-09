"""R&D 에이전트 + DAG 오케스트레이션 (아키텍처 ③) — 1년차 v1.

  dag.py     순차·병렬·조건부·재시도·HITL 실행기 + 자동화율 KPI
  memory.py  프로젝트 메모리 L1(세션) / L2(append-only JSONL, 비식별화 관문)
  router.py  Hybrid Inference Router v1 — local(온프레미스 SLM) / api 규칙 라우팅
  wf_a.py    WF-A 과제 기획 워크플로우(탐색 → 갭 분석 → 초안 → Gateway 검증 → 연구자 승인)

표준 라이브러리만 쓴다(LangGraph 미도입).
"""

from compliance_gateway.agent.dag import Node, RunResult, automation_rate, run_dag
from compliance_gateway.agent.memory import ProjectMemory, SensitiveContentError
from compliance_gateway.agent.router import Route, route

__all__ = ["Node", "RunResult", "automation_rate", "run_dag",
           "ProjectMemory", "SensitiveContentError", "Route", "route"]
