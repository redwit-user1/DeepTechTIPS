# -*- coding: utf-8 -*-
"""연구노트 가상 데이터 생성기 + ALCOA+ 위반 변형.

위반 유형은 **국가연구개발 연구노트 지침의 실제 필수 요건**에서 도출했다.
연구노트 점검·감사에서 실제로 지적되는 항목들이다.

실행:
  python -m compliance_gateway.data.labnote.generate --notes 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

from compliance_gateway.data.labnote.models import LabNote
from compliance_gateway.data.labnote.templates import (
    DOMAINS, INSTITUTIONS_BY_DOMAIN, RESEARCHERS, REVIEWERS,
)

# 위반 유형 → 위반 ALCOA+ 속성 + 감사 지적 사유
VIOLATIONS = {
    "missing_signature":  ("Attributable",    "연구자 서명 누락 — 기록 주체를 특정할 수 없음"),
    "missing_reviewer":   ("Attributable",    "점검자(입회자) 서명 누락 — 제3자 확인 부재"),
    "backdated":          ("Contemporaneous", "실험일보다 기록일이 지연 — 소급 작성 의심"),
    "overwritten":        ("Original",        "정정 이력 없이 수치 변경 — 원본 식별 불가"),
    "missing_units":      ("Accurate",        "수치 단위 누락 — 값의 해석이 불가능"),
    "incomplete":         ("Complete",        "결과 미기재/미완결 표현 — 기재 누락"),
    "inconsistent":       ("Consistent",      "본문 수치와 고찰 수치 불일치"),
}

_NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)[ \t]*(℃|%|μM|mM|N/m|ms|시간|nm|㎛|mm|rpm)")


def _perturb(value: str, factor: float) -> str:
    """수치를 **반드시 달라지도록** 변형한다.

    단순히 배수를 곱하고 반올림하면 원값으로 되돌아갈 수 있다
    (예: 0.1 × 0.6 = 0.06 → '.1f' 포맷 시 다시 '0.1').
    그 경우 위반이 주입되지 않아 '미탐'으로 오인된다.
    """
    digits = len(value.split(".")[1]) if "." in value else 0
    for d in (digits, digits + 1, digits + 2):
        out = f"{float(value) * factor:.{d}f}"
        if float(out) != float(value):
            return out
    return f"{float(value) + 1:.{digits}f}"


def _mk_note(rng: random.Random, idx: int) -> LabNote:
    """규정을 준수하는 정상 연구노트 1건 생성."""
    domain = rng.choice(list(DOMAINS))
    spec = DOMAINS[domain]
    exp = rng.choice(spec["experiments"])
    exp_day = date(2024, 1, 1) + timedelta(days=rng.randint(0, 500))
    researcher = rng.choice(RESEARCHERS)
    reviewer = rng.choice(REVIEWERS)
    return LabNote(
        note_id=f"SYN-{exp_day.year}-{idx:04d}",
        project_no=f"SYN-RND-{exp_day.year}-{rng.randint(1, 99):03d}",
        project_title=rng.choice(spec["projects"]),
        domain=domain,
        researcher=researcher,
        institution=rng.choice(INSTITUTIONS_BY_DOMAIN[domain]),   # 도메인 정합
        page=rng.randint(1, 200),
        exp_date=exp_day.isoformat(),
        record_date=exp_day.isoformat(),               # 당일 기록 = 동시성 충족
        reviewer=reviewer,
        review_date=(exp_day + timedelta(days=rng.randint(1, 3))).isoformat(),
        title=exp["title"], purpose=exp["purpose"], methods=exp["methods"],
        results=exp["results"], discussion=exp["discussion"],
    )


def _apply_violation(note: LabNote, kind: str, rng: random.Random) -> LabNote:
    """정상 노트에 단일 ALCOA+ 위반을 주입한다."""
    from copy import deepcopy

    n = deepcopy(note)
    attr, _reason = VIOLATIONS[kind]

    if kind == "missing_signature":
        n.researcher = ""
    elif kind == "missing_reviewer":
        n.reviewer, n.review_date = None, None
    elif kind == "backdated":
        exp = date.fromisoformat(n.exp_date)
        n.record_date = (exp + timedelta(days=rng.randint(21, 90))).isoformat()
    elif kind == "overwritten":
        # 정정 이력 없이 결과 수치만 바뀐 상태(원본 식별 불가)
        m = _NUM_UNIT.search(n.results)
        if m:
            new = _perturb(m.group(1), 1.4)
            n.results = n.results.replace(m.group(0), f"{new} {m.group(2)}", 1)
        n.correction = None
    elif kind == "missing_units":
        n.results = _NUM_UNIT.sub(lambda m: m.group(1), n.results)
    elif kind == "incomplete":
        # 소수점이 아니라 **문장 경계**에서 자른다("42.5" 중간에서 끊기지 않도록)
        first = re.split(r"(?<=다)\.\s+", n.results)[0].rstrip(". ")
        n.results = first + ". 이하 결과는 추후 기재 예정(TBD)."
        n.discussion = "…"
    elif kind == "inconsistent":
        # 본문은 그대로 두고 고찰의 수치만 다르게 → 내부 모순
        m = _NUM_UNIT.search(n.results)
        if m:
            other = f"{_perturb(m.group(1), 0.6)} {m.group(2)}"
            n.discussion = (f"본 실험의 대표값은 {other}로 확인되었다. " + n.discussion)

    n.violations = (kind,)
    n.alcoa_violated = (attr,)
    return n


def generate_notes(count: int = 40, seed: int = 20260826) -> list[LabNote]:
    """정상 연구노트 목록 생성(재현 가능하도록 시드 고정)."""
    rng = random.Random(seed)
    return [_mk_note(rng, i + 1) for i in range(count)]


def build_dataset(count: int = 40, seed: int = 20260826):
    """(노트 전체, Gateway 평가 아이템, DPO 쌍) 생성.

    평가/DPO 는 '연구노트 규정 준수 점검' 과제로 구성한다:
      질의 = 이 연구노트가 연구노트 작성 지침을 준수하는가
      정상 = 준수 노트 원문 / 위반 = 동일 노트에 단일 위반을 주입한 원문
    """
    rng = random.Random(seed + 1)
    notes = generate_notes(count, seed)
    all_notes: list[LabNote] = []
    items: list[dict] = []
    pairs: list[dict] = []

    kinds = list(VIOLATIONS)
    for note in notes:
        all_notes.append(note)
        query = (f"다음 연구노트가 연구노트 작성 지침(ALCOA+)을 준수하는지 점검하고 "
                 f"위반 사항이 있으면 근거와 함께 지적하라. [{note.note_id}]")
        base_text = note.render()
        items.append({
            "query": query, "response": base_text, "grounding": base_text,
            "note_id": note.note_id, "label": "compliant",
            "alcoa_violation": None, "domain": note.domain, "lang": "ko",
        })
        # 노트당 위반 2종 주입. 템플릿에 따라 적용 불가한 유형이 있으므로
        # (예: 결과부에 단위 수치가 없으면 overwritten 주입 불가) 실제로
        # 원문이 바뀌는 유형만 후보로 삼는다.
        applicable = [k for k in kinds if _apply_violation(note, k, rng).render() != base_text]
        for kind in rng.sample(applicable, min(2, len(applicable))):
            bad = _apply_violation(note, kind, rng)
            all_notes.append(bad)
            bad_text = bad.render()
            items.append({
                "query": query, "response": bad_text, "grounding": base_text,
                "note_id": note.note_id, "label": kind,
                "alcoa_violation": bad.alcoa_violated[0], "domain": note.domain, "lang": "ko",
            })
            pairs.append({
                "prompt": query, "chosen": base_text, "rejected": bad_text,
                "rejected_kind": kind, "alcoa_violation": bad.alcoa_violated[0],
                "grounding": base_text, "note_id": note.note_id, "lang": "ko",
            })
    return all_notes, items, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", type=int, default=40, help="정상 연구노트 수")
    ap.add_argument("--out", default="data/synth/labnote")
    ap.add_argument("--seed", type=int, default=20260826)
    a = ap.parse_args()

    notes, items, pairs = build_dataset(a.notes, a.seed)
    out = Path(a.out)
    (out / "txt").mkdir(parents=True, exist_ok=True)

    # 1) 원문 txt — 사람이 읽고 검수할 수 있게
    for n in notes:
        suffix = "OK" if not n.violations else n.violations[0]
        (out / "txt" / f"{n.note_id}_{suffix}.txt").write_text(n.render(), encoding="utf-8")
    # 2) 구조화 JSONL
    for name, rows in (("labnote_notes.jsonl", [n.to_json() for n in notes]),
                       ("labnote_eval.jsonl", items),
                       ("labnote_dpo.jsonl", pairs)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for i in items if i["label"] == "compliant")
    chars = sum(len(n.render()) for n in notes)
    print(f"연구노트 {len(notes)}건 (정상 {a.notes} / 위반 {len(notes)-a.notes}), "
          f"원문 {chars:,}자")
    print(f"평가 {len(items)}건 (준수 {n_ok} / 위반 {len(items)-n_ok}), DPO {len(pairs)}쌍")
    counts: dict[str, int] = {}
    for i in items:
        if i["alcoa_violation"]:
            counts[i["alcoa_violation"]] = counts.get(i["alcoa_violation"], 0) + 1
    print("\nALCOA+ 속성별 위반:")
    for k, v in sorted(counts.items()):
        print(f"  {k:16s} {v:3d}")
    print(f"\n→ {out}/  (txt/ 원문, labnote_eval.jsonl, labnote_dpo.jsonl)")


if __name__ == "__main__":
    main()
