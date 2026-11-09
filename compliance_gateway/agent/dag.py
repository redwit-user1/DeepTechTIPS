"""DAG 실행기 — 순차·병렬·조건부·재시도·Human-in-the-loop (표준 라이브러리만).

아키텍처 ③ 은 LangGraph 를 상정했지만, 1년차 v1 은 `graphlib.TopologicalSorter` +
`ThreadPoolExecutor` 로 충분하다. 노드 = (이름, fn(ctx)->출력, 선행노드).

- 준비된(선행이 모두 끝난) 노드는 스레드 풀에서 **동시에** 돈다.
- `when(ctx)` 가 거짓이면 건너뛴다(skipped). 건너뛴 노드의 후속은 그대로 실행된다 —
  ctx[이름] = None 으로 보고 판단한다.
- `retries=n` 이면 예외 시 최대 n 번 더 시도한다(`backoff` 초부터 2배씩 대기).
- `human=True` 노드는 `approve(node, ctx)` 가 참일 때만 fn 을 실행한다. 승인 콜백이 없거나
  거절하면 실행이 `pending_human` 으로 멈춘다(에이전트는 스스로 승인하지 않는다).
- 실패·보류가 나면 새 노드는 띄우지 않고, 이미 돌던 노드만 마저 기록한 뒤 멈춘다.

ctx 쓰기는 메인 스레드에서만 일어나고, 워커에는 제출 시점 ctx 의 얕은 복사본을 넘긴다
(출력은 반환값으로만 전달) — 잠금이 필요 없다.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Any, Callable, Optional

Approve = Callable[["Node", dict], bool]


@dataclass
class Node:
    name: str
    fn: Optional[Callable[[dict], Any]] = None     # human 노드는 생략 가능(출력 = True)
    deps: tuple[str, ...] = ()
    when: Optional[Callable[[dict], bool]] = None  # 거짓이면 skipped
    retries: int = 0
    backoff: float = 0.0                           # 재시도 전 대기(초), 매번 2배 — 429 대응
    human: bool = False


@dataclass
class RunResult:
    status: str          # done | failed | pending_human
    ctx: dict
    log: list[dict]      # 단계 기록 {node, status, attempts, seconds[, error]}


def _execute(node: Node, ctx: dict) -> tuple[dict, Any]:
    """fn 을 재시도 포함 실행하고 (단계 기록, 출력) 을 돌려준다."""
    t0 = time.perf_counter()
    err: Optional[Exception] = None
    for attempt in range(1, node.retries + 2):
        try:
            out = node.fn(ctx) if node.fn else True
            return {"node": node.name, "status": "auto", "attempts": attempt,
                    "seconds": round(time.perf_counter() - t0, 3)}, out
        except Exception as e:  # 노드 실패는 실행 결과로 기록한다(전파하지 않음)
            err = e
            if attempt <= node.retries and node.backoff:
                time.sleep(node.backoff * 2 ** (attempt - 1))
    return {"node": node.name, "status": "failed", "attempts": node.retries + 1,
            "seconds": round(time.perf_counter() - t0, 3),
            "error": f"{type(err).__name__}: {err}"}, None


def run_dag(nodes: list[Node], ctx: Optional[dict] = None,
            approve: Optional[Approve] = None, max_workers: int = 4) -> RunResult:
    """노드를 의존 순서대로 실행한다. ctx 는 초기 입력이자 노드 출력 저장소(L1)."""
    by_name = {n.name: n for n in nodes}
    if len(by_name) != len(nodes):
        raise ValueError("노드 이름 중복")
    unknown = {d for n in nodes for d in n.deps} - by_name.keys()
    if unknown:
        raise ValueError(f"정의되지 않은 선행 노드: {sorted(unknown)}")
    ts = TopologicalSorter({n.name: n.deps for n in nodes})
    ts.prepare()                                   # 순환이면 CycleError

    ctx = {} if ctx is None else ctx
    log: list[dict] = []
    status = "done"

    def record(entry: dict, out: Any) -> None:
        nonlocal status
        log.append(entry)
        if entry["status"] == "failed":
            status = "failed"
        else:
            ctx[entry["node"]] = out
            ts.done(entry["node"])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        running: dict = {}
        while status == "done" and ts.is_active():
            for name in ts.get_ready():
                node = by_name[name]
                if node.when is not None and not node.when(ctx):
                    record({"node": name, "status": "skipped", "attempts": 0, "seconds": 0.0}, None)
                elif node.human:
                    # 사람 대기는 메인 스레드에서(대화형 input() 가능). 그동안 워커는 계속 돈다.
                    t0 = time.perf_counter()
                    if approve is None or not approve(node, ctx):
                        log.append({"node": name, "status": "pending", "attempts": 0,
                                    "seconds": round(time.perf_counter() - t0, 3)})
                        status = "pending_human"
                        break
                    entry, out = _execute(node, ctx)
                    entry["seconds"] = round(time.perf_counter() - t0, 3)
                    if entry["status"] == "auto":
                        entry["status"] = "human"
                    record(entry, out)
                else:
                    running[pool.submit(_execute, node, dict(ctx))] = name
                if status != "done":
                    break
            if status != "done" or not running:
                continue
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                del running[fut]
                record(*fut.result())
        # 멈춘 경우: 이미 돌던 노드는 끝까지 기다려 기록한다
        for fut in running:
            entry, out = fut.result()
            log.append(entry)
            if entry["status"] != "failed":
                ctx[entry["node"]] = out
    return RunResult(status=status, ctx=ctx, log=log)


def automation_rate(log: list[dict]) -> float:
    """에이전트 워크플로우 자동화율 (KPI 목표 85%+).

    자동화율 = auto / (auto + human + failed + pending)

    - 분자 `auto`: 사람 개입 없이 완료된 단계. 재시도 끝에 성공해도 auto 다.
    - 분모 = **시도된 단계**: skipped 를 뺀 모든 기록.
      · human  — 사람이 승인해야 끝난 단계(필수 승인 게이트 포함) → 자동 아님
      · failed — 재시도를 다 써도 실패 → 자동 아님
      · pending — 사람 대기로 멈춘 단계 → 자동 아님
      · skipped — 조건 미충족으로 실행할 필요가 없던 단계 → 분모에서 제외
    - 실행이 멈춰 기록조차 없는 후속 단계는 시도되지 않았으므로 분모에 없다.
    - 여러 실행을 합산하려면 로그를 이어 붙여 넘긴다(단계 가중 평균). 시도 0건이면 0.0.
    """
    attempted = [e for e in log if e["status"] != "skipped"]
    if not attempted:
        return 0.0
    return sum(e["status"] == "auto" for e in attempted) / len(attempted)
