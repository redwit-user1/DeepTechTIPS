"""R&D Instruction 게이트 — 근거 없는 문장·개인정보·민감 서술이 하나라도 있으면 버린다."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_instructions", Path("scripts/build_instructions.py"))
bi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bi)

PASSAGE = "시료 A 를 37℃ 에서 24시간 배양했다. 흡광도는 0.82 로 대조군 0.41 보다 높았다."


def fake_nli(premise: str, hypothesis: str) -> float:
    # 문장 속 숫자가 모두 근거에 있으면 지지 — 테스트용
    import re
    return 1.0 if all(n in premise for n in re.findall(r"\d+(?:\.\d+)?", hypothesis)) else 0.0


def row(answer, task="summary", question=""):
    return {"doc": "d", "passage": PASSAGE, "task": task, "question": question, "answer": answer}


def test_screen_keeps_grounded_and_drops_the_rest():
    assert bi.screen(row("37℃ 에서 24시간 배양한 시료 A 의 흡광도는 0.82 였다."), fake_nli, 0.5) == ""
    assert bi.screen(row("흡광도는 0.82 였다. 대조군은 0.55 였다."), fake_nli, 0.5) == "unsupported", "한 문장이라도 근거 없으면 버린다"
    assert bi.screen(row("담당자 메일 kim@redwit.io 로 문의"), fake_nli, 0.5) == "pii"
    assert bi.screen(row("환자 3명에서 증상이 호전되었다", task="qa", question="결과는?"), fake_nli, 0.5) == "sensitive"


def test_prompt_carries_the_source():
    p = bi.prompt_of("qa", "배양 온도는?", PASSAGE)
    assert p.startswith("배양 온도는?") and PASSAGE in p
    assert "요약" in bi.prompt_of("summary", "", PASSAGE)


def test_dpo_pairs_from_gate_outcomes():
    rows = [
        {**row("A 결과", "summary"), "reason": "pass"},
        {**row("A 결과에 근거 없는 해석", "summary"), "reason": "unsupported"},
        {**row("B 조건", "conditions"), "reason": "pass"},                     # 탈락 짝 없음 → 쌍 없음
        {**row("질문1 답", "qa", "Q1"), "reason": "pass"},
        {**row("질문2 답", "qa", "Q2"), "reason": "unsupported"},               # 질의응답은 지시가 달라 제외
        {**row("메일 노출", "report"), "reason": "pii"},
        {**row("보고서", "report"), "reason": "pass"},                          # 개인정보 탈락은 선호쌍 아님
    ]
    pairs = bi.dpo_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "A 결과" and pairs[0]["rejected"].endswith("해석") and PASSAGE in pairs[0]["prompt"]


def _eval_sft():
    spec2 = importlib.util.spec_from_file_location("eval_sft", Path("scripts/eval_sft.py"))
    es = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(es)
    return es


def _write_split_inputs(tmp_path, rows):
    """rows: (doc, passage). gen 과 sft 를 같은 문서로 만든다."""
    import json
    gen = tmp_path / "gen.jsonl"; sft = tmp_path / "sft.jsonl"
    gen.write_text("".join(json.dumps({"doc": d, "task": "summary", "question": "", "passage": p, "answer": "x"},
                                      ensure_ascii=False) + "\n" for d, p in rows), encoding="utf-8")
    sft.write_text("".join(json.dumps({"messages": [{"role": "user", "content": p}], "source": {"doc": d}},
                                      ensure_ascii=False) + "\n" for d, p in rows), encoding="utf-8")
    return type("A", (), {"gen": str(gen), "sft": str(sft), "exclude": None, "mod": 10,
                          "train_out": str(tmp_path / "train.jsonl"), "heldout_out": str(tmp_path / "held.jsonl")})


def _split_result(a):
    import json
    train = [json.loads(l)["source"]["doc"] for l in open(a.train_out, encoding="utf-8")]
    held = [json.loads(l) for l in open(a.heldout_out, encoding="utf-8")]
    return train, held


def test_eval_split_works_when_every_doc_id_is_odd(tmp_path):
    """v4 회귀: 1차 단락의 문서 ID 는 전부 홀수라 int(doc,16) % 10 == 0 이 한 번도 참이 아니었다(평가 0건)."""
    es = _eval_sft()
    docs = [f"{2 * i + 1:014x}" for i in range(400)]
    a = _write_split_inputs(tmp_path, [(d, f"단락 {d}") for d in docs])
    es.split(a)
    train, held = _split_result(a)
    held_docs = {h["doc"] for h in held}
    assert 20 <= len(held_docs) <= 70, "약 10% 가 평가용"
    assert not held_docs & set(train) and len(held_docs) + len(train) == 400


def test_eval_split_keeps_docs_sharing_a_passage_together(tmp_path):
    es = _eval_sft()
    rows = []
    for i in range(300):                       # 문서 둘이 같은 단락(공백만 다름)을 공유
        rows += [(f"a{i:04x}", f"공유 단락 {i}"), (f"b{i:04x}", f"공유  단락 {i}")]
    a = _write_split_inputs(tmp_path, rows)
    es.split(a)
    train, held = _split_result(a)
    held_i = {h["doc"][1:] for h in held}
    assert held_i and not held_i & {d[1:] for d in train}, "같은 단락의 다른 문서가 학습에 남으면 안 된다"
    assert len(held) == len(held_i), "같은 단락·같은 과제의 지시는 한 번만"


def test_eval_split_excludes_already_trained_passages_and_refuses_empty(tmp_path):
    import json

    import pytest
    es = _eval_sft()
    rows = [(f"{i:04x}", f"단락 {i}") for i in range(200)]
    a = _write_split_inputs(tmp_path, rows)
    es.split(a)
    _, held = _split_result(a)
    seen = tmp_path / "seen.jsonl"             # 평가용으로 뽑힌 단락을 '이미 학습에 썼다'고 하면 모두 빠져야 한다
    seen.write_text("".join(json.dumps({"doc": "다른문서", "passage": h["passage"]}, ensure_ascii=False) + "\n" for h in held),
                    encoding="utf-8")
    a.exclude = [str(seen)]
    with pytest.raises(SystemExit, match="빈 분할"):
        es.split(a)


def test_eval_judge_counts_sentences_and_numbers():
    es = _eval_sft()
    r = es.judge(PASSAGE, "흡광도는 0.82 였다. 대조군은 0.55 였다.", fake_nli, 0.5)
    assert r["reason"] == "unsupported" and (r["n_sent"], r["n_supported"]) == (2, 1)
    assert r["num_recall"] == 0.25, "원문 수치 37·24·0.82·0.41 중 0.82 만 옮겼다"
    assert es.judge(PASSAGE, "", fake_nli, 0.5)["reason"] == "empty"
    assert es.judge(PASSAGE, "37℃ 에서 24시간 배양했고 흡광도는 0.82, 대조군은 0.41 이었다.", fake_nli, 0.5)["num_recall"] == 1.0


def test_eval_paired_comparison():
    import pytest
    es = _eval_sft()
    assert es.mcnemar_exact(0, 0) == 1.0 and es.mcnemar_exact(5, 5) == 1.0
    assert es.mcnemar_exact(0, 10) == pytest.approx(2 / 2 ** 10)
    mk = lambda i, ok: {"prompt": f"p{i}", "doc": f"d{i}", "task": "summary", "reason": "pass" if ok else "unsupported", "chars": 10}  # noqa: E731
    a = [mk(i, i < 50) for i in range(100)]
    b = [mk(i, i < 80) for i in range(100)]
    r = es.paired(a, b)
    assert (r["a"], r["b"], r["a_only"], r["b_only"]) == (0.5, 0.8, 0, 30) and r["p"] < 1e-6
    assert 0.2 < r["ci"][0] < 0.3 < r["ci"][1] < 0.4
    with pytest.raises(SystemExit):
        es.summarize([])


def test_eval_refuses_adapter_that_did_not_load():
    """v5 회귀: 모델 클래스가 학습 때와 달라 어댑터 키가 안 맞으면 PEFT 는 경고만 내고 lora_B=0 으로 둔다."""
    import pytest
    es = _eval_sft()

    class P:
        def __init__(self, v): self.v = v
        def detach(self): return self
        def abs(self): return self
        def sum(self): return self.v

    class M:
        def __init__(self, vals): self.vals = vals
        def named_parameters(self): return [(f"layers.{i}.q_proj.lora_B.default.weight", P(v)) for i, v in enumerate(self.vals)]

    assert es.check_adapter_loaded(M([0.3, 1.2])) == 2
    with pytest.raises(SystemExit, match="실리지 않았다"):
        es.check_adapter_loaded(M([0.3, 0.0]))
    with pytest.raises(SystemExit):
        es.check_adapter_loaded(M([]))


def _judge_all():
    spec3 = importlib.util.spec_from_file_location("judge_all", Path("scripts/judge_all.py"))
    ja = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(ja)
    return ja


def test_judge_all_reproduces_cascade_from_stored_scores():
    ja = _judge_all()
    s = lambda enc, llm, novel=False, swap=False: {"text": "x", "enc": enc, "llm": llm, "novel_num": novel, "unit_swap": swap}  # noqa: E731
    assert ja.cascade_score(s(0.99, 0.0)) == 0.99, "인코더가 확신하면 LLM 을 보지 않는다"
    assert ja.cascade_score(s(0.001, 1.0)) == 0.001
    assert ja.cascade_score(s(0.5, 0.9)) == 0.9, "애매 구간은 LLM"
    assert ja.cascade_score(s(0.99, 0.2, novel=True)) == 0.2, "근거에 없는 숫자는 LLM"
    assert ja.cascade_score(s(0.99, 0.9, swap=True)) == 0.0
    row = {"pii": False, "sensitive": False, "sents": [s(0.001, 1.0), s(0.99, 0.9)], "qa_ok": 0.2}
    assert ja.verdicts(row) == {"cascade": False, "llm": True, "qa_ok": False}
    assert ja.verdicts({"pii": False, "sensitive": False, "sents": []})["cascade"] is False, "빈 응답은 통과가 아니다"
    assert ja.question_of({"task": "qa", "prompt": bi.prompt_of("qa", "배양 온도는?", PASSAGE)}) == "배양 온도는?"
    assert ja.question_of({"task": "summary", "prompt": bi.prompt_of("summary", "", PASSAGE)}) == ""


def test_judge_all_filter_drops_off_question_answers(tmp_path):
    import json
    ja = _judge_all()
    judged = tmp_path / "judged.jsonl"; sft = tmp_path / "sft.jsonl"; out = tmp_path / "out.jsonl"
    rows = [{**row("0.41 이다", "qa", "흡광도는?"), "qa_ok": 0.1},      # 대조군 값을 답함 → 제외
            {**row("0.82 이다", "qa", "흡광도는?"), "qa_ok": 0.9},
            {**row("요약", "summary"), "qa_ok": None}]
    judged.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    recs = [{"messages": [{"role": "user", "content": bi.prompt_of(r["task"], r["question"], r["passage"])},
                          {"role": "assistant", "content": r["answer"]}]} for r in rows]
    sft.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in recs), encoding="utf-8")
    ja.filter_sft(str(judged), str(sft), str(out))
    kept = [json.loads(l)["messages"][1]["content"] for l in open(out, encoding="utf-8")]
    assert kept == ["0.82 이다", "요약"]


def test_judge_all_kv_asks_only_item_lines_of_conditions(tmp_path):
    import json
    from compliance_gateway.nli.llm_judge import KV_PROMPT
    ja = _judge_all()
    assert ja.kv_split("- 배양 온도: 37℃") == ("배양 온도", "37℃")
    assert ja.kv_split("시료를 37℃ 에서 배양했다.") is None
    sent = lambda t: {"text": t, "llm": 0.1}  # noqa: E731
    rows = [{**row("x", "conditions"), "sents": [sent("배양 온도: 37℃"), sent("배양했다."), sent("시간: 99시간")]},
            {**row("y", "summary"), "sents": [sent("온도: 37℃")]}]
    src = tmp_path / "eval.jsonl"; out = tmp_path / "eval.kv.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    asked = []

    class Fake:
        def batch(self, pairs, prompt=None):
            assert prompt is KV_PROMPT
            asked.extend(h for _, h in pairs)
            return [1.0 if h.split("값: ")[1] in p else 0.0 for p, h in pairs]
    ja.kv_file(str(src), out, Fake())
    got = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert got == [{"i": 0, "kv": [1.0, None, 0.0]}], "조건 추출 행의 항목 줄만 묻고, 문장은 비워 둔다"
    assert asked[0] == "항목: 배양 온도" + chr(10) + "값: 37℃"


def test_kv_row_ok_uses_item_judges_for_lines_and_old_verdict_for_sentences():
    ja = _judge_all()
    four = lambda *cols: [list(c) for c in zip(*cols)]  # noqa: E731 — 문장별 (판정 넷의 점수) → 판정별 목록
    assert ja.kv_row_ok(four((0.9, 0.9, 0.9, 0.1)), [False]), "항목 줄: 넷 중 셋이 통과시키면 기존 판정이 탈락이어도 통과"
    assert not ja.kv_row_ok(four((0.9, 0.9, 0.1, 0.1)), [True])
    assert not ja.kv_row_ok(four((0.9, 0.9, 0.9, 0.9), (None, None, None, None)), [True, False]), "문장은 기존 판정"
    assert ja.kv_row_ok(four((0.9, 0.9, 0.9, 0.9), (None, None, None, None)), [False, True])
    assert not ja.kv_row_ok([[0.9], [0.9], [0.1]], [True]), "판정이 셋 이하이면 모두 통과해야 한다"
    assert not ja.kv_row_ok([[], [], [], []], []), "빈 응답은 통과가 아니다"


def test_sample_question_takes_teacher_questions_only_in_qa_mode():
    spec2 = importlib.util.spec_from_file_location("eval_sft_q", Path("scripts/eval_sft.py"))
    es = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(es)
    assert es.sample_question(row("요약", "summary"), qa=False) == ""
    assert es.sample_question(row("요약", "summary"), qa=True) is None
    assert es.sample_question(row("37℃", "qa", "배양 온도는?"), qa=False) is None
    assert es.sample_question(row("37℃", "qa", "배양 온도는?"), qa=True) == "배양 온도는?"
    assert es.sample_question({**row("37℃", "qa", "배양 온도는?"), "qa_ok": 0.1}, qa=True) is None, "원문에 맞지 않는 질문은 쓰지 않는다"
    assert es.sample_question(row("37℃", "qa", ""), qa=True) is None


def test_dpo_multi_pairs_qa_only_within_the_same_question(tmp_path):
    import json
    ja = _judge_all()

    def judged(name, verdicts):          # verdicts: [(질문, 답, 문장 점수, qa_ok)]
        rows = [{**row(ans, "qa", q), "pii": False, "sensitive": False, "qa_ok": fit,
                 "sents": [{"text": ans, "llm": sc}]} for q, ans, sc, fit in verdicts]
        p = tmp_path / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
        return ([str(p)], 0.5)
    base = [("배양 온도는?", "37℃ 이다", 0.9, 0.9),      # 모두 통과 + 적합 → chosen
            ("배양 온도는?", "24시간 이다", 0.9, 0.1),   # 원문에 있는 값이지만 질문과 다름 → rejected
            ("흡광도는?", "0.99 이다", 0.1, 0.9),         # 다른 질문의 탈락 — 위 chosen 과 짝이 되면 안 된다
            ("", "질문 없는 행", 0.9, 0.9)]
    pairs, stats = ja.dpo_multi({n: judged(n, base) for n in ("a", "b", "c")}, set())
    assert [(p["chosen"], p["rejected"]) for p in pairs] == [("37℃ 이다", "24시간 이다")]
    assert "배양 온도는?" in pairs[0]["prompt"] and pairs[0]["source"]["task"] == "qa"


def test_distill_multi_labels_only_unanimous_sentences(tmp_path):
    import json
    ja = _judge_all()

    def judged(name, scores):            # scores: 문장별 점수
        sents = [{"text": t, "llm": v, "unit_swap": False} for t, v in
                 zip(("배양했다.", "흡광도가 높았다.", "원인은 온도다.", "온도: 37℃"), scores)]
        rows = [{**row("x"), "sents": sents}, {**row("y"), "passage": "긴" * 900, "sents": sents[:1]}]
        p = tmp_path / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
        return ([str(p)], 0.5)
    judges = {"a": judged("a", (0.9, 0.9, 0.1, 0.9)), "b": judged("b", (0.9, 0.1, 0.1, 0.9)), "c": judged("c", (0.9, 0.9, 0.2, 0.9))}
    rows, stats = ja.distill_multi_rows(judges, set())
    assert sorted((r["hypothesis"], r["label"]) for r in rows) == [("배양했다.", "entail"), ("원인은 온도다.", "neutral")], \
        "갈린 문장 · 항목 줄 · 800자 넘는 원문은 빠진다"
    assert ja.distill_multi_rows(judges, {"".join(PASSAGE.split())})[0] == [], "평가 원문은 뺀다"


def test_enc_file_rescoring_keeps_llm_scores(tmp_path):
    import json
    ja = _judge_all()
    rows = [{**row("x"), "sents": [{"text": "37℃ 에서 배양했다.", "enc": 0.0, "llm": 0.7}, {"text": "99℃ 였다.", "enc": 1.0, "llm": 0.2}]}]
    src = tmp_path / "j.jsonl"; out = tmp_path / "o.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    ja.enc_file(str(src), out, fake_nli)
    got = json.loads(out.read_text(encoding="utf-8"))["sents"]
    assert [(s["enc"], s["llm"]) for s in got] == [(1.0, 0.7), (0.0, 0.2)]


def test_prompt_kind_changes_wording_only_for_other_sources():
    assert bi.prompt_of("summary", "", PASSAGE) == bi.prompt_of("summary", "", PASSAGE, "note"), "기본값은 기존 데이터와 글자까지 같다"
    p = bi.prompt_of("summary", "", PASSAGE, "abstract")
    assert "연구노트" not in p and "논문 초록" in p and "[논문 초록]" in p and PASSAGE in p
    assert "연구노트" not in bi._kind(bi.GEN_PROMPT, "abstract") and "{passage}" in bi._kind(bi.GEN_PROMPT, "abstract")


def test_kci_passage_filter():
    spec4 = importlib.util.spec_from_file_location("build_kci_passages", Path("scripts/build_kci_passages.py"))
    kp = importlib.util.module_from_spec(spec4); spec4.loader.exec_module(kp)
    assert kp.usable("가나다라마 " * 100) and not kp.usable("가나다") and not kp.usable("abcde " * 100)
    assert not kp.usable("가" * 1501)


def test_llm_judge_pins_the_date_for_templates_that_take_one():
    from compliance_gateway.nli.llm_judge import JUDGE_DATE, LLMJudgeNLI

    class Tok:
        def __init__(self, template):
            self.chat_template, self.seen = template, None

        def apply_chat_template(self, msg, **kw):
            self.seen = kw
            return "x"
    j = LLMJudgeNLI("m")
    j._tok = Tok("{{ date_string }} {{ messages }}")
    j._text("근거", "주장")
    assert j._tok.seen["date_string"] == JUDGE_DATE, "날짜를 받는 대화 틀에는 고정 날짜를 넘긴다"
    j._tok = Tok("{{ messages }}")
    j._text("근거", "주장")
    assert "date_string" not in j._tok.seen
