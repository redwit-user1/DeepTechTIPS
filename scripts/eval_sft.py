#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT 평가 — 학습에 안 쓴 연구노트 단락으로 '근거에 충실한 생성 비율'을 기반모델과 비교 (GPU).

KPI '보고서 자동 생성 정확도'의 **대리 지표**: 생성한 응답의 모든 문장이 원문 단락에 뒷받침되고(CascadeNLI)
개인정보·민감 서술이 없는 비율. 문장이 맞는지(사실 충실)만 보고, 보고서로서의 완결성·유용성은 보지 않는다.
그래서 **정보량 지표**(길이·문장 수·원문 수치 재현율)를 함께 낸다 — 게이트는 문장이 많을수록 떨어지기 쉬워
(연구노트 §38: 50자 미만 93% → 300자 이상 77%) 통과분만 학습한 모델은 '짧게 써서 통과'할 수 있다.

    split    평가용 문서 묶음 10% 를 떼고 학습셋에서 뺀다
    run      모델(+LoRA 어댑터)로 평가 지시에 답하고 게이트로 심사
    rejudge  이미 생성한 응답을 다른 판정기(LLM 단독)로 다시 심사 — 학습 필터와 평가가 같은 판정기라는 순환 점검
    compare  같은 지시에 대한 두 결과를 짝지어 비교(McNemar 정확 검정 · 문서 단위 부트스트랩 신뢰구간)

    python scripts/eval_sft.py split --gen data/instr/gen2.jsonl --sft data/instr/sft2.jsonl \
        --exclude data/instr/gen.jsonl --train-out data/instr/sft2_train.jsonl --heldout-out data/instr/heldout2.jsonl
    python scripts/eval_sft.py run --model Qwen/Qwen3.5-9B [--adapter ckpt] --heldout data/instr/heldout2.jsonl --out data/instr/eval_base.jsonl
    python scripts/eval_sft.py compare data/instr/eval_base.jsonl data/instr/eval_sft_v1.jsonl

분할 규칙(§38 — v4 에서 평가셋이 0건이었던 원인 수정):
- 문서 ID 를 수로 읽어 나머지를 보지 않는다. 1차 단락 1,303개는 문서 ID 가 전부 홀수라 `int(doc,16) % 10 == 0` 이
  한 번도 참이 아니었다 → sha256(묶음 키) 로 나눈다.
- 같은 단락이 여러 문서에 있다(2차 5,854 단락 중 398개가 963개 문서에 공유). 단락을 공유하는 문서는 한 묶음으로
  같은 쪽에 둔다 — 문서 단위로만 나누면 같은 원문이 학습·평가 양쪽에 들어간다.
- `--exclude`: 이미 학습에 쓴 생성 파일. 그 문서·단락과 겹치는 묶음은 평가셋에서 뺀다(이전 어댑터도 깨끗하게 평가).
- 평가셋이나 학습셋이 비면 멈춘다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_instructions import prompt_of  # noqa: E402

_NUM = re.compile(r"\d+(?:\.\d+)?")

# 프롬프트만으로 맞춘 기준선(--style-hint). 기반모델 탈락의 상당수는 형식(마크다운 47% · 머리말 30%)과 덧붙인 해설이다 —
# 지시 한 줄로 고쳐지는 몫과 학습이 실제로 더한 몫을 가른다. 짝 비교를 위해 저장하는 prompt 는 원래 지시 그대로 둔다.
STYLE_HINT = "\n\n머리말·맺음말·마크다운 없이 답만 간결하게 써라. 원문에 없는 설명·해석·추측을 덧붙이지 마라."


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _read(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def heldout_key(key: str, mod: int = 10) -> bool:
    """묶음 키의 sha256 으로 나눈다 — ID 의 수치 성질(홀짝 등)에 의존하지 않는다."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % mod == 0


def doc_groups(rows: list[dict]) -> dict[str, str]:
    """문서 → 묶음 키. 공백을 지운 단락이 같은 문서끼리 한 묶음(union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first: dict[str, str] = {}
    for r in rows:
        d, p = find(r["doc"]), _norm(r["passage"])
        if p in first:
            a, b = sorted((d, find(first[p])))
            parent[b] = a
        else:
            first[p] = r["doc"]
    return {d: find(d) for d in parent}


def split(a) -> None:
    gen = _read(a.gen)
    group = doc_groups(gen)
    seen_docs, seen_passages = set(), set()
    for path in a.exclude or []:
        for r in _read(path):
            seen_docs.add(r["doc"])
            seen_passages.add(_norm(r["passage"]))
    dirty = {group[r["doc"]] for r in gen if r["doc"] in seen_docs or _norm(r["passage"]) in seen_passages}
    held_groups = {g for g in set(group.values()) if heldout_key(g, a.mod) and g not in dirty}

    seen, held = set(), []
    for r in gen:
        key = (_norm(r["passage"]), r["task"], r.get("question", ""))
        if group[r["doc"]] in held_groups and key not in seen:
            seen.add(key)
            held.append({"doc": r["doc"], "group": group[r["doc"]], "task": r["task"], "passage": r["passage"],
                         "prompt": prompt_of(r["task"], r.get("question", ""), r["passage"], r.get("kind", "note"))})

    train, dup, uniq = [], 0, set()
    for l in open(a.sft, encoding="utf-8"):
        rec = json.loads(l)
        if group.get(rec["source"]["doc"], rec["source"]["doc"]) in held_groups:
            continue
        h = hashlib.sha256(json.dumps(rec["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if h in uniq:                      # 다중 표본이 같은 응답을 낸 경우 — 같은 예를 두 번 학습하지 않는다
            dup += 1
            continue
        uniq.add(h)
        train.append(l if l.endswith("\n") else l + "\n")

    if not held or not train:
        raise SystemExit(f"거부: 평가 {len(held)}건 · 학습 {len(train)}건 — 빈 분할로는 학습·평가하지 않는다 "
                         f"(묶음 {len(set(group.values()))} · 평가 묶음 {len(held_groups)} · 제외 묶음 {len(dirty)})")
    Path(a.train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.train_out, "w", encoding="utf-8") as f:
        f.writelines(train)
    with open(a.heldout_out, "w", encoding="utf-8") as f:
        for h in held:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"학습 {len(train)}(중복 제거 {dup}) · 평가 지시 {len(held)}(묶음 {len(held_groups)} · "
          f"문서 {len({h['doc'] for h in held})}) · 이미 학습에 쓰여 제외한 묶음 {len(dirty)} · "
          f"과제 {dict(Counter(h['task'] for h in held))}")


# ── 심사 ────────────────────────────────────────────────────────────────────

def judge(passage: str, answer: str, nli, threshold: float) -> dict:
    """게이트(`build_instructions.screen`)와 같은 판정 + 문장 단위 집계와 정보량."""
    from compliance_gateway.data.ocr.deidentify import audit
    from compliance_gateway.data.ocr.sensitive import sensitive_signals
    from compliance_gateway.pipeline import _split_all, _windows

    nums = set(_NUM.findall(passage))
    info = {"chars": len(answer), "n_sent": 0, "n_supported": 0,
            "num_recall": (len(nums & set(_NUM.findall(answer))) / len(nums)) if nums else None}
    if not answer:
        return {"reason": "empty", **info}
    chunks, sents = _windows(passage), _split_all(answer)
    info["n_sent"] = len(sents)
    info["n_supported"] = sum(max(nli(c, s) for c in chunks) >= threshold for s in sents)
    if audit(answer):
        reason = "pii"
    elif sensitive_signals(answer):
        reason = "sensitive"
    else:
        reason = "pass" if info["n_supported"] == len(sents) else "unsupported"
    return {"reason": reason, **info}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("거부: 평가 지시가 0건이다 — heldout 파일을 확인할 것(split 이 비었는가)")

    def line(name: str, rs: list[dict]) -> str:
        k, n = sum(r["reason"] == "pass" for r in rs), len(rs)
        lo, hi = wilson(k, n)
        ns, sup = sum(r["n_sent"] for r in rs), sum(r["n_supported"] for r in rs)
        chars = sorted(r["chars"] for r in rs)[n // 2]
        rec = [r["num_recall"] for r in rs if r["num_recall"] is not None]
        return (f"{name:10s} 충실 생성률 {k / n:.1%} [{lo:.1%}, {hi:.1%}] n={n} · 문장 지지율 {sup / max(1, ns):.1%} · "
                f"길이 중앙값 {chars}자 · 문장 {ns / n:.1f} · 수치 재현율 {sum(rec) / max(1, len(rec)):.1%} · "
                f"{dict(Counter(r['reason'] for r in rs))}")

    print(line("전체", rows))
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    for t, rs in by.items():
        print("  " + line(t, rs))


def _nli(kind: str):
    from compliance_gateway.nli.cascade import CascadeNLI

    nli = CascadeNLI.default()
    if kind == "llm":                      # 인코더를 건너뛰고 모든 문장을 LLM 판정기로 — 캐스케이드와 독립된 두 번째 의견
        nli = CascadeNLI(lambda p, h: (nli.lo + nli.hi) / 2, nli.llm, nli.lo, nli.hi, nli.threshold)
    return nli


def _screen_all(rows: list[dict], kind: str, out: str) -> None:
    nli = _nli(kind)
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            r.update(judge(r["passage"], r["answer"], nli, nli.threshold), judge=kind)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"심사 {i + 1}/{len(rows)}", flush=True)
    print(f"판정기 {kind} · LLM 호출 {nli.llm_calls}/{nli.calls}")
    summarize(rows)


def check_adapter_loaded(model) -> int:
    """어댑터 가중치가 실제로 실렸는지 — lora_B 가 전부 0 인 모듈이 있으면 멈춘다(키 불일치로 안 실린 것)."""
    b = [(n, float(p.detach().abs().sum())) for n, p in model.named_parameters() if "lora_B" in n]
    dead = [n for n, v in b if v == 0.0]
    if not b or dead:
        raise SystemExit(f"거부: 어댑터가 실리지 않았다 — lora_B {len(b)}개 중 {len(dead)}개가 0 (예: {dead[:2]}). "
                         "학습 때와 같은 모델 클래스로 올렸는지 확인할 것")
    print(f"[어댑터] lora_B {len(b)}개 모두 적재")
    return len(b)


def run(a) -> None:
    rows = _read(a.heldout)
    if not rows:
        raise SystemExit(f"거부: {a.heldout} 가 비어 있다")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    # 학습(`train/loader.py`)과 **같은 클래스**로 올린다. 비전 통합 클래스(AutoModelForImageTextToText)는 모듈 경로가
    # model.language_model.layers.* 라, CausalLM 으로 학습한 어댑터(model.layers.*)의 키가 하나도 맞지 않는다 —
    # PEFT 는 경고만 내고 초기값(lora_B = 0, 효과 없음)으로 둔다. v5 첫 실행이 그렇게 '기반모델'을 두 번 쟀다(§38).
    try:
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if a.adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, a.adapter).eval()
        check_adapter_loaded(m)
    for i in range(0, len(rows), a.batch):
        chunk = rows[i: i + a.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"] + (STYLE_HINT if a.style_hint else "")}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
        for r, s in zip(chunk, o[:, enc["input_ids"].shape[1]:]):
            r["answer"] = tok.decode(s, skip_special_tokens=True).strip()
        print(f"{min(i + a.batch, len(rows))}/{len(rows)}", flush=True)
    del m
    torch.cuda.empty_cache()
    _screen_all(rows, a.judge, a.out)


def rejudge(a) -> None:
    rows = _read(a.inp)
    if a.limit and len(rows) > a.limit:    # 같은 시드 → 여러 결과 파일에서 같은 지시가 뽑힌다(짝 비교 가능)
        rows = random.Random(0).sample(rows, a.limit)
    prev = [r["reason"] for r in rows]
    _screen_all(rows, a.judge, a.out)
    agree = sum((p == "pass") == (r["reason"] == "pass") for p, r in zip(prev, rows))
    both = Counter(((p == "pass"), (r["reason"] == "pass")) for p, r in zip(prev, rows))
    print(f"이전 판정과 일치 {agree / len(rows):.1%} · 이전만 통과 {both[(True, False)]} · 새 판정만 통과 {both[(False, True)]}")


# ── 짝 비교 ──────────────────────────────────────────────────────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """불일치 쌍 b(앞만 통과)·c(뒤만 통과)의 양측 정확 검정 p 값."""
    n = b + c
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a_rows: list[dict], b_rows: list[dict], boot: int = 2000, seed: int = 0) -> dict:
    ka = {r["prompt"]: r for r in a_rows}
    pairs = [(ka[r["prompt"]], r) for r in b_rows if r["prompt"] in ka]
    if not pairs:
        raise SystemExit("거부: 두 결과에 공통 지시가 없다")
    ok = lambda r: r["reason"] == "pass"  # noqa: E731
    b = sum(ok(x) and not ok(y) for x, y in pairs)
    c = sum(ok(y) and not ok(x) for x, y in pairs)
    by_group = defaultdict(list)            # 같은 문서 묶음의 지시는 독립이 아니다 → 묶음 단위로 재표집
    for x, y in pairs:
        by_group[x.get("group", x["doc"])].append(ok(y) - ok(x))
    groups, rng, diffs = list(by_group.values()), random.Random(seed), []
    for _ in range(boot):
        s = [d for g in rng.choices(groups, k=len(groups)) for d in g]
        diffs.append(sum(s) / len(s))
    diffs.sort()
    n = len(pairs)
    return {"n": n, "a": sum(ok(x) for x, _ in pairs) / n, "b": sum(ok(y) for _, y in pairs) / n,
            "a_only": b, "b_only": c, "p": mcnemar_exact(b, c),
            "ci": (diffs[int(0.025 * boot)], diffs[int(0.975 * boot) - 1]),
            "chars": (sorted(x["chars"] for x, _ in pairs)[n // 2], sorted(y["chars"] for _, y in pairs)[n // 2])}


def compare(a) -> None:
    A, B = _read(a.a), _read(a.b)
    r = paired(A, B)
    print(f"짝 {r['n']} · 앞 {r['a']:.1%} → 뒤 {r['b']:.1%} (차 {r['b'] - r['a']:+.1%}p, 95% 구간 [{r['ci'][0]:+.1%}, {r['ci'][1]:+.1%}]) · "
          f"앞만 통과 {r['a_only']} · 뒤만 통과 {r['b_only']} · McNemar p={r['p']:.4f} · 길이 중앙값 {r['chars'][0]} → {r['chars'][1]}자")
    for t in sorted({x["task"] for x in A}):
        s = paired([x for x in A if x["task"] == t], [x for x in B if x["task"] == t])
        print(f"  {t:10s} 짝 {s['n']} · {s['a']:.1%} → {s['b']:.1%} [{s['ci'][0]:+.1%}, {s['ci'][1]:+.1%}] p={s['p']:.4f} · 길이 {s['chars'][0]} → {s['chars'][1]}자")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("--gen", required=True); s.add_argument("--sft", required=True)
    s.add_argument("--exclude", nargs="*", help="이미 학습에 쓴 생성 파일(gen 형식) — 겹치는 묶음은 평가셋에서 뺀다")
    s.add_argument("--train-out", required=True); s.add_argument("--heldout-out", required=True)
    s.add_argument("--mod", type=int, default=10)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--adapter")
    r.add_argument("--heldout", required=True); r.add_argument("--out", required=True)
    r.add_argument("--max-new-tokens", type=int, default=400, help="긴 원문(초록)의 보고서 문단은 400 토큰에서 잘린다")
    r.add_argument("--style-hint", action="store_true", help="지시 끝에 형식·충실 안내 한 줄을 붙인다(프롬프트 기준선)")
    r.add_argument("--batch", type=int, default=16); r.add_argument("--judge", choices=("cascade", "llm"), default="cascade")
    j = sub.add_parser("rejudge"); j.add_argument("--in", dest="inp", required=True); j.add_argument("--out", required=True)
    j.add_argument("--judge", choices=("cascade", "llm"), default="llm"); j.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    {"split": split, "run": run, "rejudge": rejudge, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
