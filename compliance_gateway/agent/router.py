"""Hybrid Inference Router v1 — 생성 작업을 온프레미스 SLM(local) 과 외부 API(api) 중 어디로 보낼지.

규칙(위에서부터 먼저 걸리는 것):
  1. 건강 민감정보 표지 → local (외부 전송 금지)
  2. 비식별화 후에도 개인정보 잔존(`audit`) → local
  3. 입력이 로컬 컨텍스트를 넘음 → api
  4. 긴 생성·추론 작업이고 예상 API 비용이 상한 이하 → api
  5. 그 외(짧은 정형 작업·비용 초과) → local

api 로 보낼 때는 `Route.text`(비식별화본)를 보낸다 — 원문을 보내지 않는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from compliance_gateway.data.ocr.deidentify import audit, deidentify
from compliance_gateway.data.ocr.sensitive import sensitive_signals

LOCAL_CONTEXT_TOKENS = 32_000
# 긴 생성·추론 작업 → 예상 출력 토큰. 여기 없는 작업(classify·extract·nli 등)은 local.
API_TASKS = {"draft": 1500, "gap_analysis": 800, "plan": 800}
# USD / 1M 토큰 — 가정값. 계약 단가로 교체할 것
PRICE_IN, PRICE_OUT = 3.0, 15.0


@dataclass(frozen=True)
class Route:
    target: str          # "local" | "api"
    reason: str
    text: str            # 실제로 보낼 텍스트(api 면 비식별화본)
    est_cost_usd: float  # api 로 보냈을 때 예상 비용(local 이어도 비교용으로 채움)


def route(text: str, task: str, max_cost_usd: float = 0.10) -> Route:
    tokens = int(len(text) / 1.5) + 1             # 한국어 ≈ 1.5자/토큰 (pipeline.WINDOW_CHARS 와 같은 가정)
    cost = (tokens * PRICE_IN + API_TASKS.get(task, 300) * PRICE_OUT) / 1e6
    signals = sensitive_signals(text)
    if signals:
        return Route("local", f"민감정보 표지 {signals} — 외부 전송 금지", text, cost)
    deid = deidentify(text, salt=os.getenv("GOONO_DEID_SALT") or "goono-deid-v1")
    residual = audit(deid.text)
    if residual:
        return Route("local", f"비식별화 후 개인정보 잔존 {sorted(residual)}", text, cost)
    if tokens > LOCAL_CONTEXT_TOKENS:
        return Route("api", f"입력 ≈{tokens} 토큰 — 로컬 컨텍스트({LOCAL_CONTEXT_TOKENS}) 초과", deid.text, cost)
    if task in API_TASKS and cost <= max_cost_usd:
        return Route("api", f"{task}: 긴 생성·추론, 예상 ${cost:.4f} ≤ ${max_cost_usd}", deid.text, cost)
    why = f"예상 ${cost:.4f} > 상한 ${max_cost_usd}" if task in API_TASKS else f"{task}: 짧은 정형 작업"
    return Route("local", why, text, cost)
