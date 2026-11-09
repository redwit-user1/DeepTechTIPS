"""OCR 연구노트 수확 파이프라인 테스트 (비식별화·파싱)."""

from compliance_gateway.data.ocr.deidentify import audit, deidentify
from compliance_gateway.data.ocr.parse import clean, normalize_dates, parse_note

SAMPLE = """과제번호: RND-2024-017
과제명: 고내열 복합소재 경화 거동 규명
수행기관: OO연구원 신소재본부
실험일: 2024-03-05
연구자: 김민준 (서명)
기록일: 2024-03-05
점검자: 조성민 책임연구원 (서명)

2. 실험 목적
경화온도가 유리전이온도에 미치는 영향을 확인한다.

3. 재료 및 방법
DGEBA 수지와 DDS 경화제를 1:1로 혼합하였다.

4. 실험 결과
120℃ 시편의 Tg는 132.4℃로 측정되었다.

5. 고찰 및 차기 계획
가교밀도 증가로 판단된다.
"""


# ── 비식별화 ────────────────────────────────────────────────────────

def test_deidentify_is_irreversible_and_consistent():
    """같은 이름은 같은 가명으로(문서 간 추적 가능), 원본은 복원 불가."""
    a = deidentify("연구자: 김민준", salt="s1").text
    b = deidentify("연구자: 김민준", salt="s1").text
    assert a == b                       # 일관성
    assert "김민준" not in a            # 원본 제거
    c = deidentify("연구자: 김민준", salt="s2").text
    assert a != c                       # salt 가 다르면 다른 가명


def test_deidentify_preserves_structure():
    """구조를 보존해야 ALCOA+ Attributable 검사가 계속 성립한다.

    이름을 지워버리면 정상 노트가 '서명 누락'으로 오판된다.
    """
    from compliance_gateway.integrity import check_lab_note

    out = deidentify(SAMPLE).text
    assert "연구자:" in out and "점검자:" in out
    rep = check_lab_note(out)
    assert rep.scores["Attributable"] == 1.0, rep.findings


def test_deidentify_removes_contact_and_id():
    out = deidentify("연락처: 010-1234-5678 / a.b@c.re.kr / 900101-1234567").text
    assert "010-1234-5678" not in out
    assert "a.b@c.re.kr" not in out
    assert "900101-1234567" not in out


def test_audit_ignores_already_pseudonymized():
    """가명을 위험으로 세면 비식별화가 끝나도 경고가 남아 판단을 흐린다."""
    assert audit(deidentify(SAMPLE).text) == {}
    assert audit("연구자: 박지훈")           # 미처리 원문은 여전히 감지


# ── OCR 노이즈 내성 ─────────────────────────────────────────────────

def test_normalize_ocr_confused_dates():
    """O↔0, l↔1 혼동이 섞인 날짜를 교정한다."""
    assert "2024-03-30" in normalize_dates("실험일: 2O24-O3-3O")
    assert "2024-11-01" in normalize_dates("기록일: 2024년 11월 1일")


def test_clean_rejoins_broken_lines():
    """OCR 이 문장 중간에 삽입한 줄바꿈을 되붙인다."""
    broken = "경화온도가 유리전이온도에 미치는 영향을 확인하고자\n하였으며 결과는 다음과 같다."
    assert "확인하고자 하였으며" in clean(broken)


def test_parse_extracts_meta_and_sections():
    note = parse_note(SAMPLE, source="a.txt")
    assert note.meta["project_no"].startswith("RND-2024")
    assert note.meta["researcher"].startswith("김민준")
    assert note.meta["exp_date"].startswith("2024-03-05")
    assert {"purpose", "methods", "results", "discussion"} <= set(note.sections)
    assert note.completeness > 0.8


def test_parse_tolerates_header_variants():
    """`4. 실험 결과` → `4 실험결과`, 콜론 변형 등 OCR 변형을 견딘다."""
    noisy = SAMPLE.replace("4. 실험 결과", "4 실험결과").replace("연구자:", "연 구 자 ")
    note = parse_note(noisy)
    assert "results" in note.sections
    assert note.meta.get("researcher", "").startswith("김민준")


def test_low_completeness_signals_parse_failure_not_violation():
    """구조가 없는 텍스트는 '위반'이 아니라 '파싱 실패'로 분류돼야 한다."""
    note = parse_note("이것은 연구노트가 아닌 임의의 문단입니다. 특별한 구조가 없습니다.")
    assert note.completeness < 0.4


def test_claim_evidence_selfcheck():
    # 교란 띄어쓰기 보존·공유 수치 한정·부등호 제외·표면 누설 감지
    from compliance_gateway.data.ocr.claim_evidence import _demo
    _demo()


def test_sensitive_and_gate_selfcheck():
    # 민감정보(건강) 표지 + 게이트의 문서 단위 차단
    from compliance_gateway.data.ocr import gate, sensitive
    sensitive._demo()
    gate._demo()


def test_deidentify_names_next_to_titles():
    # 직함·호칭 앞뒤 실명(표본 2만 행에 838건 잔존했다) — 기관명·일반어는 건드리지 않는다
    from compliance_gateway.data.ocr.deidentify import deidentify
    for t in ("홍길동 변리사가 검토함", "변리사 홍길동 의견", "책임연구원 이수진 확인", "장혜식 교수님", "이영희 씨 요청"):
        assert deidentify(t).text != t, t
    for t in ("참여 연구원 3명", "선임연구원이 확인", "한국화학연구원 협력", "대표 이사의 승인",
              "연구소장 승인", "원장님 결재", "고객님의 요청", "공동대표자",
              "연구원 모집을 시작", "연구원 채용 공고", "연구원 이하의 인력"):
        assert deidentify(t).text == t, t
    # 복성 4음절·조사 붙은 이름(Codex 교차검증 지적), 가명 재치환 없음
    for t in ("연구원 남궁민준 확인", "연구원 김민준과 협의", "연구원 남궁민준과 협의"):
        out = deidentify(t)
        assert "민준" not in out.text and out.passes == 1, out.text
    assert deidentify("연구원 김민준과 협의").text.count("_") == 1


def test_deidentify_is_linear_on_long_unspaced_text():
    # 이메일 규칙이 띄어쓰기 없는 긴 한글에서 O(n²) 였다(5만 자 78초) — 라우터가 사용자 입력에 돌린다
    import time
    from compliance_gateway.data.ocr.deidentify import deidentify
    t = time.perf_counter()
    deidentify("가나다라마바사" * 5000)
    assert time.perf_counter() - t < 2.0
    assert "이메일_" in deidentify("메일 kim.r-d+lab@redwit.co.kr 로").text
