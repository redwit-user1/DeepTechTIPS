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
