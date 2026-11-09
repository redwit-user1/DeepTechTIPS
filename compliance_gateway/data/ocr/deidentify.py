# -*- coding: utf-8 -*-
"""연구노트 비식별화 — 학습·공유 전 필수 단계.

연구노트는 **미공개 연구내용·개인정보·영업비밀**을 담는 법적 기록이다.
그대로 학습에 쓰거나 외부로 내보내면 안 된다.

## 설계 원칙

1. **되돌릴 수 없게(비가역)** — 해시 기반 가명 치환. 원본 복원 불가.
2. **구조는 보존** — `연구자: 김민준` → `연구자: 연구자_A3F2`.
   ALCOA+ `Attributable` 검사가 "서명이 있는가"를 계속 판정할 수 있어야 한다.
   단순 삭제하면 정상 노트가 위반으로 오판된다.
3. **일관성** — 같은 이름은 같은 가명으로. 문서 간 동일인 추적이 가능해야
   `Consistent` 검사가 의미를 갖는다.
4. **감사 가능** — 무엇을 몇 건 치환했는지 보고한다(값은 남기지 않음).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# 흔한 성씨(인구 상위). 직함 앞뒤 2~4자가 성씨로 시작할 때만 인명으로 본다.
_SURNAME = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉사부"
# 성씨 글자로 시작하는 일반어 — 직함 앞에 자주 온다("공동 연구원", "참여 연구원" 등)
_NOT_NAME = ("(?:공동|참여|담당|외부|소속|해당|전체|신규|기존|주관|위탁|객원|학생|대학원|연구|기술|"
             "실험|분석|품질|개발|기획|영업|관리|생산|이번|이전|정부|조사|장비|고객|전문|선임|수석|"
             "책임|주임|박사|교수|대표|원장|소장|사업|부서|본부|센터|각|모든|현|전|신임|"
             "인턴|파견|계약|정규|비정규|우리|귀사|당사|협력|연구소|고객사|협력사|회사|"
             # 직함 **뒤**에 자주 오는 일반어("교수 연구실", "대표 문의")
             "연구실|연구팀|연구진|회의|의견|검토|요청|확인|승인|면담|미팅|문의|보고|지시|결재|"
             # 직함·서식 라벨 자체(표본 2만 행 상위 오탐)
             "석사|이름|성명|서명|주소|"
             "구분|현황|이미지|소속|직위|직급|명칭|대상|비고|기관|"
             # 2026-11-09 재측정(조사·4음절 허용 후) 상위 오탐
             "오전|오후|제목|변경|제조일|채용|모집|공고|국토|경제|공학|기계|유사한|어머)(?![가-힣])"
             # 접두만 봐도 이름이 아닌 것: 직함 접두어("선임연구원"), 기관명("한국화학연구원"),
             # 직함 자체("이사의", "사장", "선생님")
             "|(?:선임|책임|수석|주임|박사후|석사후|전임|사외|사내|지도|선생|사장|부사장|이사|"
             "한국|서울|부산|대구|광주|대전|울산|세종|경기|강원|충남|충북|전남|전북|경남|경북|제주|"
             "국립|고등|첨단|미래|산업|연구|기반|고객|사용자|회원|공동|박사|전무|상무|담당|참여|학생|선배|후배|"
             # 직함 + 님("원장님", "부장님", "부모님")
             "차장|원장|부장|소장|강사|팀장|실장|과장|대리|사원|기사|부모|인건비|전공|선택|부재|기술)")

# 치환 규칙: (이름, 패턴, 가명 접두)  — 그룹 1이 있으면 그 부분만 치환
RULES: list[tuple[str, re.Pattern, str]] = [
    ("주민등록번호", re.compile(r"\d{6}\s*[-–]\s*[1-4]\d{6}"), "주민번호"),
    ("휴대전화", re.compile(r"01[016789][-. ]?\d{3,4}[-. ]?\d{4}"), "연락처"),
    ("이메일", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "이메일"),
    # 역할 라벨 뒤의 인명만 치환한다(본문 속 일반 한글은 건드리지 않음)
    ("연구자명", re.compile(r"(?<=연구자[:： ])\s*([가-힣]{2,4})"), "연구자"),
    ("작성자명", re.compile(r"(?<=작성자[:： ])\s*([가-힣]{2,4})"), "작성자"),
    ("점검자명", re.compile(r"(?<=점검자[:： ])\s*([가-힣]{2,4})"), "점검자"),
    ("확인자명", re.compile(r"(?<=확인자[:： ])\s*([가-힣]{2,4})"), "확인자"),
    ("성명", re.compile(r"(?<=성명[:： ])\s*([가-힣]{2,4})"), "성명"),
    # 직함·호칭에 붙은 실명(2026-11-09 평가셋 작성 중 "○○○ 변리사", "연구원 ○○○" 잔존 확인).
    # 성씨로 시작하는 2~4자 + 직함. "참여 연구원"·"공동 연구원" 같은 일반어는 _NOT_NAME 으로 제외.
    ("직함인명", re.compile(
        r"(?<![가-힣])((?!" + _NOT_NAME + r")[" + _SURNAME + r"][가-힣]{1,3}(?<![의을를은는와과로에]))\s?"
        r"(?=(?:책임|선임|수석|주임)?(?:연구원|변리사|변호사|박사|교수|대표|이사|팀장|실장|과장|부장|"
        r"원장|소장|센터장|님|씨)(?![가-힣]{2}))"), "인명"),
    # 직함 뒤 이름: 복성(남궁·황보…) 4음절, 3음절 이름 + 조사("김민준과"), 2음절은 조사 없이만.
    # 2음절 + 조사까지 받으면 "연구원 모집을" 같은 일반어가 걸린다. (Codex 교차검증 지적: 4음절·조사 누락)
    ("인명직함", re.compile(
        r"(?:연구원|변리사|변호사|박사|교수|대표|팀장|실장)\s((?!" + _NOT_NAME + r")"
        r"(?:(?:(?:남궁|황보|제갈|선우|독고|사공|서문)[가-힣]{1,2}|[" + _SURNAME + r"][가-힣]{2})"
        r"(?<![의을를은는와과로에이가])(?=(?:은|는|이|가|을|를|과|와|의|에게|께서|께|도|만)?(?![가-힣_]))"
        # 2음절. `_` 제외: 가명 "인명_XXXX" 의 "인명" 이 다시 이름으로 잡히던 문제
        r"|[" + _SURNAME + r"][가-힣](?![가-힣_])))"), "인명"),
]


@dataclass
class DeidentifyResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)
    mapping_size: int = 0
    passes: int = 1                 # 잔존 제거까지 실제로 돌린 패스 수

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _pseudonym(value: str, prefix: str, salt: str) -> str:
    """되돌릴 수 없는 가명. 같은 값 → 같은 가명(문서 간 일관성)."""
    h = hashlib.sha256((salt + "::" + value.strip()).encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:4].upper()}"


# 1회 패스로 부족한 경우가 실제로 있다(인접·중첩 매치).
# 실데이터에서 치환 후에도 11자리 전화번호가 남는 사례가 확인됐고,
# 2차 패스에서 잡혔다. 개인정보가 남으면 안 되므로 **잔존이 0이 될 때까지 반복**한다.
MAX_PASSES = 5


def deidentify(text: str, salt: str = "goono-deid-v1") -> DeidentifyResult:
    """개인정보를 비가역 가명으로 치환한다.

    salt 는 배포마다 바꾸고 **외부에 공개하지 않는다**
    (공개되면 후보군 대조로 재식별 시도가 가능해진다).

    잔존이 없을 때까지(최대 `MAX_PASSES`) 반복 적용한다.
    """
    counts: dict[str, int] = {}
    mapping: dict[str, str] = {}

    def one_pass(t: str) -> str:
        for name, pat, prefix in RULES:
            def repl(m: re.Match, _name=name, _prefix=prefix) -> str:
                raw = m.group(1) if m.groups() else m.group(0)
                key = f"{_prefix}:{raw.strip()}"
                if key not in mapping:
                    mapping[key] = _pseudonym(raw, _prefix, salt)
                counts[_name] = counts.get(_name, 0) + 1
                whole = m.group(0)
                return whole.replace(raw, mapping[key]) if m.groups() else mapping[key]

            t = pat.sub(repl, t)
        return t

    passes = 0
    for _ in range(MAX_PASSES):
        text = one_pass(text)
        passes += 1
        if not audit(text):
            break

    return DeidentifyResult(text=text, counts=counts, mapping_size=len(mapping),
                            passes=passes)


# 이미 치환된 가명(연구자_9565 등)은 위험이 아니다.
# 인명 패턴은 한글만 캡처하므로 가명의 앞부분("연구자")만 잡힌다
# → 매치 **직후** 에 `_XXXX` 가 오는지로 판정해야 한다.
_PSEUDO_TAIL = re.compile(r"^_[0-9A-F]{4}\b")


def audit(text: str) -> dict[str, int]:
    """치환하지 않고 **잔존 위험만** 센다(비식별화 후 검증용).

    가명 자체를 위험으로 세면 비식별화가 끝나도 경고가 남아 판단을 흐린다.
    """
    out: dict[str, int] = {}
    for name, pat, _prefix in RULES:
        remaining = 0
        for m in pat.finditer(text):
            if _PSEUDO_TAIL.match(text[m.end():m.end() + 8]):
                continue                       # 이미 가명 처리됨
            remaining += 1
        if remaining:
            out[name] = remaining
    return out
