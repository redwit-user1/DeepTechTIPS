"""판정 모델 검수 시트 — 층 나누기, 시트에 점수가 새지 않는지, 층 크기 가중."""

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

spec = importlib.util.spec_from_file_location("review_judges", Path("scripts/review_judges.py"))
rj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rj)


def test_stratum_separates_item_lines_from_sentences():
    ok = {"phi4": False, "kanana": True, "midm": True}
    assert rj.stratum("kci", "처리 농도: 5 mM", ok) == "kci/kv/phi-/oth+"
    assert rj.stratum("note", "흡광도는 0.82 로 높았다.", {**ok, "midm": False}) == "note/prose/phi-/split"


def _judged(root: Path, tag: str, name: str, scores: list[float]) -> None:
    d = root / f"judged_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"passage": f"{name} 원문 {i} 온도 37℃", "task": "conditions",
             "sents": [{"text": f"온도: 37℃ ({i})", "llm": s}]} for i, s in enumerate(scores)]
    (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_make_hides_scores_and_records_population(tmp_path, monkeypatch):
    for pre in ("eval2", "evalk"):
        _judged(tmp_path, "phi4", f"{pre}_m", [0.99, 0.99, 0.99, 0.1])
        _judged(tmp_path, "kanana", f"{pre}_m", [0.9, 0.9, 0.9, 0.9])
        _judged(tmp_path, "midm", f"{pre}_m", [0.9, 0.9, 0.9, 0.9])
    monkeypatch.setattr(rj, "OUT", tmp_path / "labeling")
    rj.make(Namespace(root=str(tmp_path), models=["m"], per_cell=2, seed=0))
    with (tmp_path / "labeling" / "review_judges_A.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"id", "원문", "문장", "human_label", "메모"}          # 판정 점수 · 모델 이름이 시트에 없다
    key = json.loads((tmp_path / "labeling" / "review_judges.key.json").read_text(encoding="utf-8"))
    assert len(rows) == len(key) == 6                                             # 층 4개: 통과 3건 중 2건 + 탈락 1건, 원천 둘
    assert {v["population"] for v in key.values() if v["stratum"].endswith("phi+/oth+")} == {3}


def test_judge_stats_weights_by_stratum_size():
    # 큰 층(1000)에서는 판정이 맞고 작은 층(10)에서는 틀린다 → 가중 정확도는 표본 그대로(50%)보다 훨씬 높다
    def k(stratum, pop, ok):
        return {"stratum": stratum, "population": pop, "ok": {t: ok for t in rj.THR}}
    key = {"a": k("big", 1000, True), "b": k("small", 10, True)}
    s = rj.judge_stats({"a": True, "b": False}, key)["phi4"]
    assert s["raw_acc"] == 0.5 and s["acc"] > 0.98
    assert abs(s["false_accept"] - 1.0) < 1e-6 and s["false_reject"] == 0


def test_kappa_is_zero_for_chance_and_one_for_identity():
    assert rj.kappa(["1", "0", "1", "0"], ["1", "0", "1", "0"]) == 1.0
    assert abs(rj.kappa(["1", "1", "0", "0"], ["1", "0", "1", "0"])) < 1e-9
