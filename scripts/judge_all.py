#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""응답 전수 판정 — 문장마다 인코더·LLM 점수를 **둘 다** 남기고, 질의응답은 질문 적합성까지 (GPU).

게이트(`build_instructions.gate`)는 통과/탈락 한 줄만 남겨서, 판정기를 고치거나 임계값을 바꾸려면 GPU 로
다시 돌려야 했다. 여기서는 원점수를 저장한다 — 사람 검수 뒤 임계값·라우팅을 바꿔 다시 거르는 일은 GPU 없이 된다.
캐스케이드는 LLM 을 한 쌍씩 불렀는데(배치 미사용) 여기서는 길이순 배치로 묶는다.

    python scripts/judge_all.py --out-dir data/judged data/instr/sft2.jsonl.screened.jsonl data/instr/eval2_base.jsonl ...

입력 행: passage, answer, task, 그리고 question 또는 prompt(지시 + "\\n\\n[연구노트]\\n" + 원문).
출력 행: 입력 + sents[{text, enc, llm, novel_num, unit_swap}] + qa_ok(질의응답만) + pii + sensitive.
  enc  증류 인코더(학생)의 지지 확률 — 800자 창 최대값
  llm  Qwen3.8-27B 판정 P(Yes) — 800자 창 최대값
  qa_ok  (질문, 답)을 원문과 함께 본 P(Yes)

    python scripts/judge_all.py report data/judged/*.jsonl      # GPU 불필요 — 판정기 엇갈림·과제별 통과율
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LO, HI, THR = 0.005, 0.98, 0.01          # CascadeNLI 기본값과 같다(증류 인코더 기준)
QA_THR = 0.5


def question_of(row: dict) -> str:
    if row.get("question"):
        return row["question"]
    if row.get("task") != "qa" or not row.get("prompt"):
        return ""
    return re.split(r"\n\n\[(?:연구노트|논문 초록)\]\n", row["prompt"], maxsplit=1)[0]      # 지시 + 원문 꼬리표(prompt_of)


def cascade_score(s: dict) -> float:
    """저장한 점수로 캐스케이드 판정을 재현한다(`nli/cascade.py` 와 같은 순서)."""
    if s["unit_swap"]:
        return 0.0
    if s["novel_num"]:
        return s["llm"]
    return s["enc"] if (s["enc"] < LO or s["enc"] >= HI) else s["llm"]


def verdicts(row: dict) -> dict:
    """한 응답의 판정 세 가지: 캐스케이드 · LLM 단독 · (질의응답이면) 질문 적합성."""
    clean = not row["pii"] and not row["sensitive"] and bool(row["sents"])
    return {"cascade": clean and all(cascade_score(s) >= THR for s in row["sents"]),
            "llm": clean and all(s["llm"] >= THR for s in row["sents"]),
            "qa_ok": row.get("qa_ok") is None or row["qa_ok"] >= QA_THR}


def judge_file(path: str, out: Path, enc, llm) -> None:
    from compliance_gateway.data.ocr.deidentify import audit
    from compliance_gateway.data.ocr.sensitive import sensitive_signals
    from compliance_gateway.nli.llm_judge import QA_PROMPT
    from compliance_gateway.pipeline import _split_all, _windows
    from compliance_gateway.vcr.hallucination import novel_numbers, unit_swaps

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    done = sum(1 for _ in open(out, encoding="utf-8")) if out.exists() else 0     # 중단 후 재개
    print(f"{path}: {len(rows)}행 (이미 {done})", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for i in range(done, len(rows), 2000):
            chunk = rows[i: i + 2000]
            pairs, where, qa = [], [], []
            for k, r in enumerate(chunk):
                wins = _windows(r["passage"])
                r["sents"] = [{"text": s, "novel_num": bool(novel_numbers(s, r["passage"])),
                               "unit_swap": bool(unit_swaps(s, r["passage"]))} for s in _split_all(r.get("answer", ""))]
                for j, s in enumerate(r["sents"]):
                    for w in wins:
                        pairs.append((w, s["text"]))
                        where.append((k, j))
                text = r.get("answer", "") + ("\n" + r["question"] if r.get("question") else "")
                r["pii"], r["sensitive"] = bool(audit(text)), bool(sensitive_signals(text))
                q = question_of(r)
                if r.get("task") == "qa" and q and r.get("answer"):
                    qa.append((k, (r["passage"], f"질문: {q}\n답: {r['answer']}")))
            for name, scores in (("llm", llm.batch(pairs)), ("enc", [enc(p, h) for p, h in pairs])):
                for (k, j), v in zip(where, scores):
                    s = chunk[k]["sents"][j]
                    s[name] = max(s.get(name, 0.0), round(float(v), 5))
            for (k, _), v in zip(qa, llm.batch([p for _, p in qa], prompt=QA_PROMPT)):
                chunk[k]["qa_ok"] = round(float(v), 5)
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            llm._cache.clear()
            print(f"  {min(i + 2000, len(rows))}/{len(rows)}", flush=True)


def report(paths: list[str]) -> None:
    for path in paths:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        by = defaultdict(Counter)
        for r in rows:
            v = verdicts(r)
            c = by[r.get("task", "?")]
            c["n"] += 1
            c["cascade"] += v["cascade"]; c["llm"] += v["llm"]
            c["cascade_only"] += v["cascade"] and not v["llm"]; c["llm_only"] += v["llm"] and not v["cascade"]
            if r.get("qa_ok") is not None:
                c["qa_n"] += 1; c["qa_ok"] += v["qa_ok"]; c["final"] += v["cascade"] and v["qa_ok"]
            else:
                c["final"] += v["cascade"]
        tot = sum(by.values(), Counter())
        print(f"\n{Path(path).name}  n={tot['n']}")
        for t, c in [("전체", tot), *sorted(by.items())]:
            qa = f" · 질문 적합 {c['qa_ok'] / c['qa_n']:.1%}" if c["qa_n"] else ""
            print(f"  {t:10s} 캐스케이드 {c['cascade'] / c['n']:.1%} · LLM 단독 {c['llm'] / c['n']:.1%} · "
                  f"캐스케이드만 통과 {c['cascade_only'] / c['n']:.1%} · LLM 만 통과 {c['llm_only'] / c['n']:.1%}{qa} · "
                  f"최종(캐스케이드+질문 적합) {c['final'] / c['n']:.1%}")


def filter_sft(judged: str, sft: str, out: str) -> None:
    """질문에 맞지 않는 질의응답을 학습셋에서 뺀다. (지시, 응답)이 judged 의 탈락 행과 같은 예를 지운다."""
    sys.path.insert(0, str(Path(__file__).parent))
    from build_instructions import prompt_of

    bad = set()
    for l in open(judged, encoding="utf-8"):
        r = json.loads(l)
        if r.get("task") == "qa" and r.get("qa_ok") is not None and r["qa_ok"] < QA_THR:
            bad.add((prompt_of("qa", r["question"], r["passage"], r.get("kind", "note")), r["answer"]))
    kept = dropped = 0
    with open(out, "w", encoding="utf-8") as f:
        for l in open(sft, encoding="utf-8"):
            m = json.loads(l)["messages"]
            if (m[0]["content"], m[-1]["content"]) in bad:
                dropped += 1
            else:
                f.write(l if l.endswith("\n") else l + "\n")
                kept += 1
    if not kept:
        raise SystemExit("거부: 남은 학습 예가 0건이다")
    print(f"질문 부적합 질의응답 {len(bad)}종 · 학습셋에서 제외 {dropped} · 남은 예 {kept}")


def export_sft(out: str, judged: list[str]) -> None:
    """판정을 통과한(캐스케이드 + 질문 적합) 응답을 SFT 기록으로. `build_instructions.gate` 가 쓰는 형식과 같다."""
    import hashlib

    sys.path.insert(0, str(Path(__file__).parent))
    from build_instructions import prompt_of

    n = kept = 0
    with open(out, "w", encoding="utf-8") as f:
        for path in judged:
            for l in open(path, encoding="utf-8"):
                r = json.loads(l)
                n += 1
                v = verdicts(r)
                if v["cascade"] and v["qa_ok"]:
                    kept += 1
                    f.write(json.dumps({
                        "messages": [{"role": "user", "content": prompt_of(r["task"], r.get("question", ""), r["passage"], r.get("kind", "note"))},
                                     {"role": "assistant", "content": r["answer"]}],
                        "source": {"doc": r["doc"], "task": r["task"],
                                   "passage_sha256": hashlib.sha256(r["passage"].encode("utf-8")).hexdigest()}},
                        ensure_ascii=False) + "\n")
    if not kept:
        raise SystemExit("거부: 통과한 응답이 0건이다")
    print(f"SFT 기록 {kept} ← 판정 {n} ({kept / n:.1%})")


def distill_rows(judged: list[str], exclude: set[str], easy: int = 10000, seed: int = 0) -> list[dict]:
    """LLM 판정을 인코더 학습 라벨로 — 생성 응답의 문장 분포로 학생을 다시 가르친다(§40 엇갈림).

    - LLM 이 확신하는 문장만: P(Yes) ≥ 0.9 → entail, ≤ 0.1 → neutral(근거로 확인 불가). 그 사이는 버린다.
    - 인코더가 이미 맞히는 쉬운 지지 문장(enc ≥ 0.98)은 easy 건만 표본. 나머지(엇갈리거나 인코더가 애매)는 전부.
    - 원문이 한 창(800자)에 들어가는 행만 — 저장한 점수는 창 최대값이라 어느 창인지 모른다.
    - exclude(공백 제거 원문)에 있는 원문은 뺀다 — 생성 모델 평가셋의 원문으로 판정기를 가르치지 않는다.
    """
    import random
    import re

    hard, soft = [], []
    for path in judged:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if len(r["passage"]) > 800 or re.sub(r"\s+", "", r["passage"]) in exclude:
                continue
            for s in r["sents"]:
                if 0.1 < s["llm"] < 0.9 or s["unit_swap"]:
                    continue
                row = {"doc": r["doc"], "premise": r["passage"], "hypothesis": s["text"],
                       "label": "entail" if s["llm"] >= 0.9 else "neutral", "teacher": s["llm"], "kind": "generated"}
                (soft if row["label"] == "entail" and s["enc"] >= 0.98 else hard).append(row)
    random.Random(seed).shuffle(soft)
    return hard + soft[:easy]


def distill(out: str, base: str, heldout: str, judged: list[str]) -> None:
    import re
    from collections import Counter

    exclude = {re.sub(r"\s+", "", json.loads(l)["passage"]) for l in open(heldout, encoding="utf-8")}
    rows = distill_rows(judged, exclude)
    with open(out, "w", encoding="utf-8") as f:
        for l in open(base, encoding="utf-8"):
            f.write(l if l.endswith("\n") else l + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"증류 학습셋: 기존 {sum(1 for _ in open(base, encoding='utf-8'))} + 생성 응답 문장 {len(rows)} {dict(Counter(r['label'] for r in rows))}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        return report(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "export":      # export <sft_out.jsonl> <judged.jsonl>...
        return export_sft(sys.argv[2], sys.argv[3:])
    if len(sys.argv) > 1 and sys.argv[1] == "distill":     # distill <out.jsonl> <기존 학습셋> <평가 지시(heldout)> <judged.jsonl>...
        return distill(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    if len(sys.argv) > 1 and sys.argv[1] == "filter":      # filter <judged.jsonl> <sft_in.jsonl> <sft_out.jsonl>
        return filter_sft(*sys.argv[2:5])
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--student", default="checkpoints/nli_distill")
    ap.add_argument("--llm", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out_dir)
    if not str(out.resolve()).startswith(str(Path.cwd().resolve() / "data")):
        raise SystemExit(f"거부: 출력은 data/ 안이어야 한다(연구노트 원문 포함). 받은 값: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from compliance_gateway.nli.llm_judge import LLMJudgeNLI
    from compliance_gateway.nli.transformer import TransformerNLI

    if not Path(a.student).exists():
        raise SystemExit(f"거부: 증류 인코더가 없다 — {a.student}")
    enc, llm = TransformerNLI(a.student, device="cuda"), LLMJudgeNLI(a.llm, device="cuda", batch_size=a.batch)
    for path in a.files:
        judge_file(path, out / Path(path).name, enc, llm)
    print("JUDGE_DONE")


if __name__ == "__main__":
    main()
