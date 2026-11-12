"""규정 온톨로지 · 감사추적 · 접근통제."""

from datetime import date, datetime, timezone

import pytest

from compliance_gateway.integrity.access import AccessRequest, decide
from compliance_gateway.integrity.audit import AuditChain
from compliance_gateway.ontology.checks import (
    REGISTRY, TEXT_CHECKS, amendment_trail, check_document, constraints_for, coverage, deviation_documented,
    labnote_cover, report_elements, retention_findings, retention_until, signature_manifest, single_project,
    specimen_labels, to_turtle)
from compliance_gateway.ontology.clauses import ALCOA, CLAUSES


def test_inventory_is_consistent():
    ids = [(c.reg, c.id) for c in CLAUSES]
    assert len(ids) == len(set(ids))
    for c in CLAUSES:
        assert c.kind in ("record", "system", "org"), c.id
        assert set(c.alcoa) <= set(ALCOA), c.id
        assert set(c.checks) <= set(REGISTRY), (c.id, c.checks)
        if c.kind == "org":
            assert not c.checks, f"{c.id}: 조직 절차에 자동 점검을 붙이면 커버리지가 부풀려진다"


def test_coverage_numbers_and_turtle():
    m = coverage()["전체"]
    assert m["clauses"] == len(CLAUSES) and m["modeled"] == 1.0
    assert 0 < m["automated"] <= m["with_platform"] <= 1
    ttl = to_turtle()
    assert ttl.count("a go:Clause") == len(CLAUSES) and "go:checkedBy" in ttl


def test_signature_manifest():
    ok = "확인자 서명: 인명_1A2B 2026-11-09 14:30 (검토 확인)"
    assert signature_manifest(ok) == []
    bad = signature_manifest("연구자 서명: 홍길동 2026-11-09")
    assert any("시각" in f for f in bad) and any("의미" in f for f in bad)
    assert signature_manifest("서명 없는 본문") and signature_manifest("본문뿐") == ["[11.50(a)/58.185(b)] 서명 표시 없음"]


def test_amendment_trail():
    assert amendment_trail("정정: 25 mM → 2.5 mM, 사유: 오기, 2026-11-09, 서명 인명_1A2B") == []
    f = amendment_trail("정정: 25 mM → 2.5 mM")
    assert len(f) == 3
    assert amendment_trail("시약을 수정 없이 사용했다") == [], "본문의 '수정'은 정정 항목이 아니다"


def test_report_elements_and_deviation():
    missing = report_elements("목적: A. 방법: B. 결론: C.")
    assert any("58.185(a)(3)" in f for f in missing) and not any("(a)(11)" in f for f in missing)
    assert deviation_documented("계획서 이탈이 있었다.") and not deviation_documented("계획서 이탈은 책임자 승인 후 진행했다.")


def test_labnote_checks():
    cover = "기관명: 레드윗 연구소\n과제명: X\n노트 번호 No. 12\n3 / 120 쪽"
    assert labnote_cover(cover) == []
    assert len(labnote_cover("실험 내용만 있음")) == 4
    assert specimen_labels("시료 S-0123 2026-11-09 채취") == [] and specimen_labels("시료를 희석") != []
    assert single_project("과제명: A\n과제명: B") and not single_project("과제명: A\n과제명: A")


def test_retention():
    assert retention_until(date(2026, 12, 31)) == date(2056, 12, 31)
    assert retention_until(date(2028, 2, 29), 30) == date(2058, 2, 28)
    assert retention_findings(date(2026, 12, 31), date(2040, 1, 1))
    assert not retention_findings(date(2026, 12, 31), date(2057, 1, 1))


def test_check_document_and_constraints():
    assert {k for c in CLAUSES for k in c.checks if k in TEXT_CHECKS}
    f = check_document("목적만 있는 보고서", "report", skip=("onto.signature",))
    assert f and not any("11.50" in x for x in f)
    p = constraints_for("report")
    assert "58.185(a)(3)" in p and "대신 서명하지 말 것" in p


def test_audit_chain_detects_tampering_and_enforces_sequence():
    t = datetime(2026, 11, 10, tzinfo=timezone.utc)
    ch = AuditChain()
    ch.append("u1", "create", "n1", "pH 7.4", now=t)
    ch.append("u1", "sign", "n1", "pH 7.4", meaning="작성", now=t)
    ch.append("u1", "modify", "n1", "pH 7.2", reason="측정 오기", now=t)
    ch.append("u2", "view", "n1", now=t)
    assert ch.verify() == [] and ch.current("n1") == "pH 7.2"
    assert ch.entries[0]["content"] == "pH 7.4", "변경이 이전 값을 가리지 않는다"
    assert ch.signatures("n1", "pH 7.4") and not ch.signatures("n1", "pH 7.2"), "서명은 서명한 내용에만 붙는다"
    with pytest.raises(ValueError):
        ch.append("u1", "modify", "n1", "pH 7.0")               # 사유 없음
    with pytest.raises(ValueError):
        ch.append("u1", "sign", "n1", "pH 7.2")                 # 의미 없음
    with pytest.raises(ValueError):
        ch.append("u1", "create", "n1", "x")                    # 순서: 이미 있음
    with pytest.raises(ValueError):
        ch.append("u1", "modify", "n9", "x", reason="r")        # 순서: 없는 기록
    ch.entries[0]["content"] = "pH 9.9"
    assert any("내용 해시" in p for p in ch.verify())


def test_audit_chain_roundtrip(tmp_path):
    ch = AuditChain()
    ch.append("u1", "create", "n1", "a")
    ch.append("u3", "access_denied", "n1", reason="과제 비소속")
    ch.dump(tmp_path / "a.jsonl")
    back = AuditChain.load(tmp_path / "a.jsonl")
    assert back.verify() == [] and back.head() == ch.head()


# 접근통제 시나리오 — 조항에서 기대 결과를 먼저 적고(오른쪽 근거) 구현과 대조한다.
# KPI '접근통제 정확도'의 v1 측정용. 자체 작성 명세 대비 일치율이며 침투 시험이 아니다.
R = AccessRequest
SCENARIOS = [
    (R("a", "recorder", "create", True), True, "과제 소속 기록자 작성"),
    (R("a", "recorder", "create", False), False, "11.10(d) 비소속"),
    (R("a", "recorder", "modify", True, is_owner=True), True, "작성자 수정(사유는 감사추적)"),
    (R("a", "recorder", "modify", True, is_owner=False), False, "11.10(g) 타인 기록 수정"),
    (R("a", "recorder", "modify", True, is_owner=True, state="signed"), True, "서명 후 수정은 사유와 함께 — 서명은 새 내용에 안 붙음"),
    (R("a", "recorder", "delete", True, is_owner=True), True, "본인 초안 삭제"),
    (R("a", "recorder", "delete", True, is_owner=True, state="signed"), False, "58.130(e) 서명 기록 삭제 불가"),
    (R("a", "recorder", "sign", True, is_owner=True, meaning="작성"), True, "작성 서명"),
    (R("a", "recorder", "sign", True, is_owner=True, meaning="승인"), False, "기록자 자기 승인 불가"),
    (R("a", "recorder", "sign", True, is_owner=False, meaning="작성"), False, "타인 기록 작성 서명"),
    (R("b", "reviewer", "sign", True, is_owner=False, meaning="확인"), True, "지침7②2 확인자 서명"),
    (R("b", "reviewer", "sign", True, is_owner=True, meaning="확인"), False, "기록자·확인자 분리"),
    (R("b", "reviewer", "sign", True, is_owner=False, meaning="작성"), False, "확인자 서명 의미"),
    (R("b", "reviewer", "modify", True, is_owner=False), False, "확인자는 수정 불가"),
    (R("b", "reviewer", "create", True), False, "기록 작성은 기록자"),
    (R("b", "reviewer", "view", True), True, "과제 소속 열람"),
    (R("b", "reviewer", "sign", False, meaning="확인"), False, "비소속 서명"),
    (R("m", "manager", "view", False), True, "지침10② 관리 부서 열람"),
    (R("m", "manager", "modify", False), False, "관리 부서 내용 변경 불가"),
    (R("m", "manager", "delete", False, state="archived", retention_expired=True), True, "지침12① 보존기간 경과 폐기"),
    (R("m", "manager", "delete", False, state="archived"), False, "지침10④ 보존기간 내 폐기 불가"),
    (R("m", "manager", "sign", False, meaning="승인"), False, "관리 부서 서명 불가"),
    (R("v", "viewer", "view", True), True, "열람자 열람"),
    (R("v", "viewer", "view", False), False, "비소속 열람"),
    (R("v", "viewer", "modify", True), False, "열람자 수정"),
    (R("g", "agent", "create", True), True, "에이전트 초안"),
    (R("g", "agent", "modify", True, state="signed"), False, "에이전트는 초안만"),
    (R("g", "agent", "sign", True, meaning="승인"), False, "연구자 승인 필수"),
    (R("g", "agent", "delete", True), False, "에이전트 삭제 불가"),
    (R("g", "agent", "view", True, sensitive=True, location="api"), False, "민감 기록 외부 API 금지"),
    (R("g", "agent", "view", True, sensitive=True, location="onprem"), True, "민감 기록 온프레미스 열람"),
    (R("a", "recorder", "view", True, sensitive=True, location="api"), False, "사람도 민감 기록을 외부로 못 보냄"),
    (R("a", "recorder", "modify", True, is_owner=True, state="archived"), False, "보관 기록 변경 불가"),
    (R("x", "admin", "view", True), False, "알 수 없는 역할"),
]


def test_access_control_scenarios():
    wrong = [(why, r) for r, want, why in SCENARIOS if decide(r).allow != want]
    assert not wrong, wrong
