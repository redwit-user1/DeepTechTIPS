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

NL = chr(10)
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


KV_LINE = re.compile(r"^[-•*\d.\s]*([^:：]{1,40})[:：]\s*(\S.*)$")


def kv_split(text: str) -> tuple[str, str] | None:
    """'항목: 값' 줄이면 (항목, 값). 문장이면 None."""
    m = KV_LINE.match(text.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def kv_file(path: str, out: Path, llm) -> None:
    """이미 판정한 파일의 조건 추출 응답에서 항목 줄만 KV_PROMPT 로 다시 묻는다 (연구노트 §48).

    출력은 줄마다 {"i": 행 번호, "kv": [문장별 점수 또는 null]} — 원문을 다시 싣지 않는다.
    """
    from compliance_gateway.nli.llm_judge import KV_PROMPT
    from compliance_gateway.pipeline import _windows

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    pairs, where, result = [], [], {}
    for i, r in enumerate(rows):
        if r.get("task") != "conditions":
            continue
        result[i] = [None] * len(r["sents"])
        for j, s in enumerate(r["sents"]):
            kv = kv_split(s["text"])
            if kv:
                for w in _windows(r["passage"]):
                    pairs.append((w, f"항목: {kv[0]}\n값: {kv[1]}"))
                    where.append((i, j))
    for (i, j), v in zip(where, llm.batch(pairs, prompt=KV_PROMPT)):
        result[i][j] = max(result[i][j] or 0.0, round(float(v), 5))
    with out.open("w", encoding="utf-8") as f:
        for i in sorted(result):
            f.write(json.dumps({"i": i, "kv": result[i]}) + "\n")
    print(f"{path}: 조건 추출 {len(result)}행 · 항목 줄 {len(set(where))}개", flush=True)


KV_THR = 0.5        # 항목 줄 전용 판정의 임계값 — 사람 검수 전의 잠정값(§48)


def kv_row_ok(kv_scores: list[list], sent_ok: list[bool]) -> bool:
    """조건 추출 응답 하나의 통과 여부. kv_scores: 판정 모델별 [문장별 전용 점수 또는 None], sent_ok: 문장별 기존 합의 판정.

    항목 줄은 전용 판정 모델 중 하나를 뺀 나머지가 통과시키면(셋 이하이면 모두) 통과, 문장은 기존 판정을 따른다.
    """
    need = len(kv_scores) - 1 if len(kv_scores) > 3 else len(kv_scores)
    for j, old in enumerate(sent_ok):
        col = [k[j] for k in kv_scores]
        if col[0] is None:
            if not old:
                return False
        elif sum(v >= KV_THR for v in col) < need:
            return False
    return bool(sent_ok)


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


def distill_multi_rows(judges: dict[str, tuple[list[str], float]], exclude: set[str],
                       cap: int = 60000, seed: int = 0) -> tuple[list[dict], Counter]:
    """여러 판정 모델의 합의를 인코더 학습 라벨로 (연구노트 §53).

    v4 까지는 Qwen 하나의 확신(P ≥ 0.9 / ≤ 0.1)을 라벨로 썼다 — 인코더가 Qwen 의 관대함을 물려받는다(§44).
    - 모든 판정 모델이 자기 임계값 이상 → entail, 과반이 미만 → neutral(선호쌍의 rejected 와 같은 기준). 그 사이는 버린다.
      (셋 모두 미만인 문장은 59만 중 1,116개뿐이다 — Qwen 의 임계값 0.016 이 거의 아무것도 탈락시키지 않는다.)
    - '항목: 값' 줄은 뺀다 — 기존 프롬프트의 판정이 믿을 수 없다(§48).
    - 원문이 한 창(800자)에 들어가는 행만, exclude(평가 원문)는 뺀다 — distill_rows 와 같다.
    - neutral 은 전부, entail 은 neutral 의 2배까지 표본. 합쳐서 cap 을 넘으면 같은 비율로 줄인다.
    """
    import random

    names = list(judges)
    stats: Counter = Counter()
    pos, neg = [], []
    for k in range(len(judges[names[0]][0])):
        handles = [open(judges[n][0][k], encoding="utf-8") for n in names]
        for lines in zip(*handles):
            rows = [json.loads(l) for l in lines]
            r = rows[0]
            if any(x["answer"] != r["answer"] or len(x["sents"]) != len(r["sents"]) for x in rows):
                raise SystemExit(f"거부: 판정 파일의 행이 서로 다르다 — {judges[names[0]][0][k]}")
            if len(r["passage"]) > 800 or re.sub(r"\s+", "", r["passage"]) in exclude:
                continue
            for j, s in enumerate(r["sents"]):
                if s["unit_swap"] or kv_split(s["text"]):
                    stats["항목 줄 · 단위 뒤바뀜 제외"] += 1
                    continue
                votes = [x["sents"][j]["llm"] >= judges[n][1] for n, x in zip(names, rows)]
                if all(votes) or sum(votes) * 2 < len(votes):
                    row = {"doc": r["doc"], "premise": r["passage"], "hypothesis": s["text"],
                           "label": "entail" if all(votes) else "neutral", "kind": "consensus"}
                    (pos if all(votes) else neg).append(row)
                stats[f"지지 판정 {sum(votes)}/{len(votes)}"] += 1
        for h in handles:
            h.close()
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos = pos[: 2 * len(neg)]
    if len(pos) + len(neg) > cap:
        f = cap / (len(pos) + len(neg))
        pos, neg = pos[: int(len(pos) * f)], neg[: int(len(neg) * f)]
    return pos + neg, stats


def distill_multi_cli(args: list[str]) -> None:
    """distill-multi <out.jsonl> <기존 학습셋> <heldout1,heldout2> <scores 디렉터리> <파일1,파일2> <이름=판정디렉터리[=덤프꼬리]>..."""
    out, base, heldouts, scores, files, specs = args[0], args[1], args[2], args[3], args[4].split(","), args[5:]
    exclude = set()
    for h in heldouts.split(","):
        exclude |= {re.sub(r"\s+", "", json.loads(l)["passage"]) for l in open(h, encoding="utf-8")}
    rows, stats = distill_multi_rows(_judges_from(scores, files, specs), exclude)
    if not rows:
        raise SystemExit("거부: 합의 라벨이 0건이다")
    with open(out, "w", encoding="utf-8") as f:
        for l in open(base, encoding="utf-8"):
            f.write(l if l.endswith(NL) else l + NL)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + NL)
    print(f"문장별 합의: {dict(sorted(stats.items()))}")
    print(f"합의 증류 학습셋: 기존 {sum(1 for _ in open(base, encoding='utf-8'))} + 합의 문장 {len(rows)} {dict(Counter(r['label'] for r in rows))}")


def dpo_consensus(judged: list[str], exclude: set[str]) -> list[dict]:
    """두 판정기가 **모두** 동의한 선호쌍 — 같은 (원문, 과제)의 응답 가운데.

    chosen   캐스케이드 통과 + 모든 문장의 LLM 점수 ≥ 0.5
    rejected 캐스케이드 탈락 + LLM 점수 < 0.5 인 문장이 하나 이상(개인정보·민감 탈락은 선호의 문제가 아니라 뺀다)
    캐스케이드만 탈락시킨 응답(§40: 8~15%)은 오탈락일 수 있어 rejected 로 쓰지 않는다. 질의응답은 표본마다 질문이 달라 뺀다.
    exclude(공백 제거 원문)에 있는 평가용 원문은 뺀다.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from build_instructions import prompt_of

    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: {"good": [], "bad": []})
    for path in judged:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            key = re.sub(r"\s+", "", r["passage"])
            if r.get("task") == "qa" or key in exclude or r["pii"] or r["sensitive"] or not r["sents"]:
                continue
            v = verdicts(r)
            low = any(s["llm"] < 0.5 for s in r["sents"])
            if v["cascade"] and not low:
                groups[(key, r["task"])]["good"].append(r)
            elif not v["cascade"] and low:
                groups[(key, r["task"])]["bad"].append(r)
    out = []
    for (_, task), g in groups.items():
        for good, bad in zip(g["good"], g["bad"]):
            if good["answer"] != bad["answer"]:
                out.append({"prompt": prompt_of(task, "", good["passage"], good.get("kind", "note")),
                            "chosen": good["answer"], "rejected": bad["answer"],
                            "source": {"doc": good["doc"], "task": task}})
    return out


def dpo(out: str, heldouts: str, judged: list[str]) -> None:
    exclude = set()
    for h in heldouts.split(","):
        exclude |= {re.sub(r"\s+", "", json.loads(l)["passage"]) for l in open(h, encoding="utf-8")}
    pairs = dpo_consensus(judged, exclude)
    if not pairs:
        raise SystemExit("거부: 선호쌍이 0건이다")
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    lc = sorted(len(p["chosen"]) for p in pairs); lr = sorted(len(p["rejected"]) for p in pairs)
    print(f"합의 선호쌍 {len(pairs)} · 과제 {dict(Counter(p['source']['task'] for p in pairs))} · "
          f"길이 중앙값 chosen {lc[len(lc) // 2]}자 / rejected {lr[len(lr) // 2]}자 · "
          f"chosen 이 더 짧은 쌍 {sum(len(p['chosen']) < len(p['rejected']) for p in pairs) / len(pairs):.0%}")


def calibrate(dumps: list[str]) -> float:
    """라벨이 있는 평가셋의 점수 덤프(kr_nli --dump)에서 판정 임계값을 고른다 — 지지/비지지 균형 정확도가 최대인 값.

    판정 모델마다 P(Yes) 눈금이 다르다(§44: 같은 평가셋의 운영점이 Midm 0.04~0.32, Phi-4 0.44~0.99).
    0.5 로 일괄 자르면 모델에 따라 지나치게 엄하거나 느슨하다.
    """
    pos, neg = [], []
    for d in dumps:
        for l in open(d, encoding="utf-8"):
            r = json.loads(l)
            (pos if r["label"] == "entail" else neg).append(r["score"])
    if not pos or not neg:
        raise SystemExit(f"거부: 보정용 점수가 없다 — {dumps}")
    best, best_acc = 0.5, -1.0
    for t in sorted(set(pos + neg)):
        acc = (sum(s >= t for s in pos) / len(pos) + sum(s < t for s in neg) / len(neg)) / 2
        if acc > best_acc:
            best, best_acc = t, acc
    return best


def dpo_multi(judges: dict[str, tuple[list[str], float]], exclude: set[str]) -> tuple[list[dict], Counter]:
    """여러 판정 모델의 합의로 선호쌍을 만든다. judges: 이름 → (같은 순서의 판정 파일들, 임계값).

    chosen   **모든** 판정 모델이 전 문장 지지(임계값 이상)
    rejected 판정 모델 과반이 지지하지 않는 문장이 있다고 봄
    한 판정 모델로 고른 쌍으로 정렬하면 그 모델에 맞추는 법을 배운다(§44: DPO 이득이 다른 판정기에서 사라짐).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from build_instructions import prompt_of

    names = list(judges)
    n_files = {len(judges[n][0]) for n in names}
    assert len(n_files) == 1, "판정 모델마다 파일 수가 같아야 한다"
    stats: Counter = Counter()
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: {"good": [], "bad": []})
    for k in range(n_files.pop()):
        handles = [open(judges[n][0][k], encoding="utf-8") for n in names]
        for lines in zip(*handles):
            rows = [json.loads(l) for l in lines]
            r = rows[0]
            if any(x["answer"] != r["answer"] for x in rows):
                raise SystemExit(f"거부: 판정 파일의 행 순서가 다르다 — {judges[names[0]][0][k]}")
            key = re.sub(r"\s+", "", r["passage"])
            if key in exclude or r["pii"] or r["sensitive"] or not r["sents"]:
                continue
            q = ""
            if r.get("task") == "qa":            # 같은 질문의 표본끼리만 짝짓는다. 질문이 없는 행은 뺀다
                q = (r.get("question") or "").strip()
                if not q:
                    continue
            ok = [all(s["llm"] >= judges[n][1] for s in x["sents"]) for n, x in zip(names, rows)]
            good, bad = all(ok), sum(ok) * 2 < len(ok)
            if q:                                # 질의응답은 질문 적합성도 합의로: 모두 적합해야 chosen, 과반이 부적합이면 rejected
                fit = [x.get("qa_ok") is not None and x["qa_ok"] >= QA_THR for x in rows]
                good, bad = good and all(fit), bad or sum(fit) * 2 < len(fit)
                stats[f"질의응답 적합 판정 {sum(fit)}/{len(fit)}"] += 1
            stats[f"통과 판정 {sum(ok)}/{len(ok)}"] += 1
            if good:
                groups[(key, r["task"], q)]["good"].append(r)
            elif bad:
                groups[(key, r["task"], q)]["bad"].append(r)
        for h in handles:
            h.close()
    out = []
    for (_, task, q), g in groups.items():
        for good, bad in zip(g["good"], g["bad"]):
            if good["answer"] != bad["answer"]:
                out.append({"prompt": prompt_of(task, q, good["passage"], good.get("kind", "note")),
                            "chosen": good["answer"], "rejected": bad["answer"],
                            "source": {"doc": good["doc"], "task": task}})
    return out, stats


def _judges_from(scores: str, files: list[str], specs: list[str]) -> dict[str, tuple[list[str], float]]:
    judges = {}
    for spec in specs:
        name, d, *tail = spec.split("=")
        tag = tail[0] if tail else name
        thr = calibrate([f"{scores}/{s}_{tag}.jsonl" for s in ("kr_psg_eval", "kr_hard_eval", "kr_abs_eval")])
        judges[name] = ([f"{d}/{f}" for f in files], thr)
        print(f"판정 모델 {name}: 임계값 {thr:.3f}")
    return judges


def export_multi_cli(args: list[str]) -> None:
    """export-multi <sft_out.jsonl> <heldout1,heldout2> <scores 디렉터리> <파일1,파일2> <이름=판정디렉터리[=덤프꼬리]>...

    **모든** 판정 모델이 전 문장을 지지한 응답만 SFT 기록으로 낸다(Qwen 하나로 거른 학습셋과 비교하기 위한 것, §45).
    질의응답은 첫 판정 모델의 질문 적합성(qa_ok)도 통과해야 한다. 평가용 원문은 뺀다. 같은 (지시, 응답)은 한 번만.
    """
    import hashlib

    sys.path.insert(0, str(Path(__file__).parent))
    from build_instructions import prompt_of

    out, heldouts, scores, files, specs = args[0], args[1], args[2], args[3].split(","), args[4:]
    kv_dirs = [x for spec in specs if spec.startswith("kv=") for x in spec[3:].split(",")]     # kv=<디렉터리,디렉터리,...>
    specs = [spec for spec in specs if not spec.startswith("kv=")]
    exclude = set()
    for h in heldouts.split(","):
        exclude |= {re.sub(r"\s+", "", json.loads(l)["passage"]) for l in open(h, encoding="utf-8")}
    judges = _judges_from(scores, files, specs)
    names = list(judges)
    stats, seen, kept = Counter(), set(), Counter()
    with open(out, "w", encoding="utf-8") as f:
        for k in range(len(files)):
            handles = [open(judges[n][0][k], encoding="utf-8") for n in names]
            kvs = [{x["i"]: x["kv"] for x in map(json.loads, open(f"{d}/{Path(files[k]).stem}.kv.jsonl", encoding="utf-8"))}
                   for d in kv_dirs]
            for i, lines in enumerate(zip(*handles)):
                rows = [json.loads(l) for l in lines]
                r = rows[0]
                if any(x["answer"] != r["answer"] for x in rows):
                    raise SystemExit(f"거부: 판정 파일의 행 순서가 다르다 — {files[k]}")
                if re.sub(r"\s+", "", r["passage"]) in exclude:
                    stats["평가 원문 제외"] += 1
                    continue
                if r["pii"] or r["sensitive"] or not r["sents"]:
                    stats["개인정보·민감·빈 응답"] += 1
                    continue
                ok = [all(s["llm"] >= judges[n][1] for s in x["sents"]) for n, x in zip(names, rows)]
                stats[f"통과 판정 {sum(ok)}/{len(ok)}"] += 1
                passed = all(ok)
                if kvs and r["task"] == "conditions" and all(i in d for d in kvs):
                    # 항목 줄은 전용 판정으로(§48) — 기존 프롬프트는 값이 원문에 그대로 있는 줄도 3건 중 1건을 탈락시켰다
                    sent_ok = [all(x["sents"][j]["llm"] >= judges[n][1] for n, x in zip(names, rows)) for j in range(len(r["sents"]))]
                    passed = kv_row_ok([d[i] for d in kvs], sent_ok)
                    stats["조건 추출: 전용 판정 통과" if passed else "조건 추출: 전용 판정 탈락"] += 1
                if not passed or (r.get("qa_ok") is not None and r["qa_ok"] < QA_THR):
                    continue
                prompt = prompt_of(r["task"], r.get("question", ""), r["passage"], r.get("kind", "note"))
                key = hashlib.sha256((prompt + "\0" + r["answer"]).encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                kept[r["task"]] += 1
                f.write(json.dumps({
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": r["answer"]}],
                    "source": {"doc": r["doc"], "task": r["task"],
                               "passage_sha256": hashlib.sha256(r["passage"].encode("utf-8")).hexdigest()}},
                    ensure_ascii=False) + "\n")
            for h in handles:
                h.close()
    if not kept:
        raise SystemExit("거부: 합의 통과 응답이 0건이다")
    print(f"응답별 합의: {dict(sorted(stats.items()))}")
    print(f"합의 통과 SFT 기록 {sum(kept.values())} · 과제 {dict(kept)}")


def dpo_multi_cli(args: list[str]) -> None:
    """dpo-multi <out.jsonl> <heldout1,heldout2> <scores 디렉터리> <파일1,파일2> <이름=판정디렉터리[=덤프꼬리]>..."""
    out, heldouts, scores, files, specs = args[0], args[1], args[2], args[3].split(","), args[4:]
    exclude = set()
    for h in heldouts.split(","):
        exclude |= {re.sub(r"\s+", "", json.loads(l)["passage"]) for l in open(h, encoding="utf-8")}
    judges = {}
    for spec in specs:
        name, d, *tail = spec.split("=")
        tag = tail[0] if tail else name
        thr = calibrate([f"{scores}/{s}_{tag}.jsonl" for s in ("kr_psg_eval", "kr_hard_eval", "kr_abs_eval")])
        judges[name] = ([f"{d}/{f}" for f in files], thr)
        print(f"판정 모델 {name}: 임계값 {thr:.3f}")
    pairs, stats = dpo_multi(judges, exclude)
    if not pairs:
        raise SystemExit("거부: 선호쌍이 0건이다")
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    lc = sorted(len(p["chosen"]) for p in pairs); lr = sorted(len(p["rejected"]) for p in pairs)
    print(f"응답별 합의: {dict(sorted(stats.items()))}")
    print(f"다중 판정 합의 선호쌍 {len(pairs)} · 과제 {dict(Counter(p['source']['task'] for p in pairs))} · "
          f"길이 중앙값 chosen {lc[len(lc) // 2]}자 / rejected {lr[len(lr) // 2]}자 · "
          f"chosen 이 더 짧은 쌍 {sum(len(p['chosen']) < len(p['rejected']) for p in pairs) / len(pairs):.0%}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "distill-multi":
        return distill_multi_cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "dpo-multi":
        return dpo_multi_cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "export-multi":
        return export_multi_cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "dpo":         # dpo <out.jsonl> <heldout1,heldout2> <judged.jsonl>...
        return dpo(sys.argv[2], sys.argv[3], sys.argv[4:])
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
    ap.add_argument("--kv", action="store_true", help="이미 판정한 파일의 조건 추출 항목 줄만 KV_PROMPT 로 다시 묻는다 → <이름>.kv.jsonl")
    a = ap.parse_args()
    out = Path(a.out_dir)
    if not str(out.resolve()).startswith(str(Path.cwd().resolve() / "data")):
        raise SystemExit(f"거부: 출력은 data/ 안이어야 한다(연구노트 원문 포함). 받은 값: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from compliance_gateway.nli.llm_judge import LLMJudgeNLI
    from compliance_gateway.nli.transformer import TransformerNLI

    if a.kv:
        llm = LLMJudgeNLI(a.llm, device="cuda", batch_size=a.batch)
        for path in a.files:
            kv_file(path, out / (Path(path).stem + ".kv.jsonl"), llm)
        print("KV_DONE")
        return
    if not Path(a.student).exists():
        raise SystemExit(f"거부: 증류 인코더가 없다 — {a.student}")
    enc, llm = TransformerNLI(a.student, device="cuda"), LLMJudgeNLI(a.llm, device="cuda", batch_size=a.batch)
    for path in a.files:
        judge_file(path, out / Path(path).name, enc, llm)
    print("JUDGE_DONE")


if __name__ == "__main__":
    main()
