"""연구노트 가상 데이터셋 + 기록 무결성 검사 테스트."""

import random

from compliance_gateway.data.labnote.generate import (
    VIOLATIONS, _apply_violation, build_dataset, generate_notes,
)
from compliance_gateway.data.labnote.models import SYNTHETIC_BANNER
from compliance_gateway.data.labnote.templates import INSTITUTIONS_BY_DOMAIN
from compliance_gateway.integrity import check_lab_note


def test_all_notes_marked_synthetic():
    """연구노트는 법적 증빙 기록이다. 가상 데이터가 실제로 오인되면 안 된다."""
    for note in generate_notes(10):
        assert note.synthetic is True
        assert SYNTHETIC_BANNER in note.render()
        assert note.note_id.startswith("SYN-")
        assert note.project_no.startswith("SYN-")
        assert note.to_json()["synthetic"] is True


def test_domain_and_institution_are_consistent():
    """정상 노트에 내부 모순이 있으면 안 된다(바이오 과제 + 신소재연구본부 등)."""
    for note in generate_notes(30):
        assert note.institution in INSTITUTIONS_BY_DOMAIN[note.domain]


def test_compliant_notes_pass_integrity_check():
    """정상 노트는 ALCOA+ 6속성을 모두 충족해야 한다(오탐 0)."""
    for note in generate_notes(30):
        text = note.render()
        rep = check_lab_note(text, text)
        assert rep.overall == 1.0, f"{note.note_id} 오탐: {rep.violated} / {rep.findings}"


def test_each_violation_detected_on_its_own_attribute():
    """위반 1종 → ALCOA+ 1속성. 미탐·오탐 모두 0이어야 한다."""
    misses, false_positives = [], []
    for note in generate_notes(20):
        base = note.render()
        rng = random.Random(7)
        for kind, (attr, _) in VIOLATIONS.items():
            bad = _apply_violation(note, kind, rng)
            if bad.render() == base:
                continue                      # 템플릿상 주입 불가
            rep = check_lab_note(bad.render(), base)
            if attr not in rep.violated:
                misses.append((note.note_id, kind))
            false_positives += [(kind, v) for v in rep.violated if v != attr]
    assert not misses, f"미탐: {misses[:5]}"
    assert not false_positives, f"오탐: {false_positives[:5]}"


def test_dimensionless_metrics_are_not_unit_violations():
    """F1·AUC 등 무차원 지표는 단위가 없는 게 정상 — 감점하면 안 된다."""
    text = (f"{SYNTHETIC_BANNER}\n실험일: 2024-01-01\n연구자: 홍길동 (서명)\n"
            "기록일: 2024-01-01\n점검자: 김철수 (서명) / 점검일: 2024-01-02\n\n"
            "4. 실험 결과\nFP32 모델의 F1은 0.912, INT8은 F1 0.897로 측정되었다.\n")
    rep = check_lab_note(text)
    assert rep.scores["Accurate"] == 1.0, rep.findings


def test_derived_values_in_discussion_are_not_inconsistency():
    """'지연시간을 59% 단축' 같은 파생값은 결과부에 없어도 모순이 아니다."""
    text = (f"{SYNTHETIC_BANNER}\n실험일: 2024-01-01\n연구자: 홍길동 (서명)\n"
            "기록일: 2024-01-01\n점검자: 김철수 (서명) / 점검일: 2024-01-02\n\n"
            "4. 실험 결과\nFP32는 48.3 ms, INT8은 19.6 ms였다.\n\n"
            "5. 고찰 및 차기 계획\nINT8이 지연시간을 59% 단축하여 배포 후보로 적합하다.\n")
    rep = check_lab_note(text)
    assert rep.scores["Consistent"] == 1.0, rep.findings


def test_backdated_record_flags_contemporaneous():
    note = generate_notes(1)[0]
    bad = _apply_violation(note, "backdated", random.Random(1))
    rep = check_lab_note(bad.render(), note.render())
    assert "Contemporaneous" in rep.violated
    assert any("소급" in f for f in rep.findings)


def test_build_dataset_is_balanced_and_covers_alcoa():
    notes, items, pairs = build_dataset(20)
    assert all(n.synthetic for n in notes)
    attrs = {i["alcoa_violation"] for i in items if i["alcoa_violation"]}
    assert len(attrs) >= 5              # ALCOA+ 다수 속성 커버
    assert pairs and all(p["lang"] == "ko" for p in pairs)
    # chosen 은 준수 원문, rejected 는 위반 원문
    assert all(p["chosen"] != p["rejected"] for p in pairs)


def test_vcr_is_unsuitable_for_lab_notes():
    """연구노트는 인용이 없어 VCR 로는 평가할 수 없다 — 설계 판단을 고정한다.

    이 사실이 바뀌면(예: VCR 이 무인용 텍스트를 잘 다루게 되면) 이 테스트가 실패해
    integrity/ 와 vcr/ 의 역할 분담을 재검토하게 된다.
    """
    from compliance_gateway.eval.benchmark import auc
    from compliance_gateway.nli.statistical import StatisticalNLI
    from compliance_gateway.vcr.reward import compute_vcr

    _, items, _ = build_dataset(15)
    nli = StatisticalNLI()
    pos, neg = [], []
    for it in items:
        v = compute_vcr(it["query"], it["response"],
                        grounding=(it["grounding"],), nli_fn=nli).vcr
        (neg if it["label"] != "compliant" else pos).append(v)
    vcr_auc = auc(pos, neg)

    integ_pos = [check_lab_note(i["response"], i["grounding"]).overall
                 for i in items if i["label"] == "compliant"]
    integ_neg = [check_lab_note(i["response"], i["grounding"]).overall
                 for i in items if i["label"] != "compliant"]
    integrity_auc = auc(integ_pos, integ_neg)

    assert integrity_auc > vcr_auc, "기록 무결성 검사가 VCR 보다 나아야 한다"
    assert sum(pos) / len(pos) < 0.6, "VCR 은 정상 노트도 낮게 준다(인용 부재)"
