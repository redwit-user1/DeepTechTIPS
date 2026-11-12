"""규정 조항 인벤토리 — 온톨로지 L1(규정) · L2(조항·객체) · L3(ALCOA+ 속성) · L4(checked_by).

원문 대조(2026-11-10):
  - FDA 21 CFR Part 11 — Cornell LII(law.cornell.edu/cfr/text/21/11.10 등). eCFR 은 자동 접근 차단.
  - FDA 21 CFR Part 58(GLP) — Cornell LII 58.35·58.130·58.185·58.190·58.195.
    OECD GLP 원칙(Directive 2004/10/EC Annex I)은 원문 접근 실패 → 미반영(다음 차수).
  - 국가연구개발사업 연구노트 지침 — 과기정통부 고시 제2020-104호(시행 2021-01-01).
    정부 역할(제4·13·14조)·정의(제1~3조)는 기록·시스템 요건이 아니라 제외했다.

요약은 번역·의역이다. 법적 판단의 근거는 원문이다.

kind
  record — 기록·보고서 **본문**에 대한 요건. 텍스트로 자동 점검 가능해야 한다.
  system — 시스템 기능 요건(감사추적·접근통제·전자서명·보존).
  org    — 조직 절차(교육·서면 정책·QA 조직). 자동 점검 대상이 아니다 — 모델링만 한다.
checks   — 이 레포에 구현·테스트된 점검(registry 키). 비면 미구현.
platform — GOONO 서비스가 제공한다고 알려진 기능(전자서명·시점인증·권한). **이 레포에서 검증하지 않았다.**
applies  — Pre-Gen 제약·Post-Gen 점검을 적용할 문서 유형(labnote | report).
"""

from __future__ import annotations

from dataclasses import dataclass

ALCOA = ("Attributable", "Legible", "Contemporaneous", "Original", "Accurate",
         "Complete", "Consistent", "Enduring", "Available")

P11 = "FDA 21 CFR Part 11"
P58 = "FDA 21 CFR Part 58 (GLP)"
KR = "국가연구개발사업 연구노트 지침(과기정통부 고시 제2020-104호)"


@dataclass(frozen=True)
class Clause:
    id: str
    reg: str
    kind: str                       # record | system | org
    obj: str                        # L2 객체: record | signature | audit_trail | access | archive | report | system | org
    alcoa: tuple[str, ...]
    summary: str
    checks: tuple[str, ...] = ()
    platform: bool = False
    applies: tuple[str, ...] = ()


def C(id, reg, kind, obj, alcoa, summary, checks=(), platform=False, applies=()):
    return Clause(id, reg, kind, obj, tuple(alcoa.split()) if isinstance(alcoa, str) else alcoa,
                  summary, tuple(checks), platform, tuple(applies))


CLAUSES: tuple[Clause, ...] = (
    # ---- 21 CFR Part 11 -------------------------------------------------------------
    C("11.10(a)", P11, "system", "system", "Accurate Consistent", "시스템 밸리데이션(정확성·신뢰성·일관된 성능, 무효·변조 기록 식별)"),
    C("11.10(b)", P11, "system", "record", "Legible Available", "사람이 읽을 수 있는 형태와 전자 형태로 정확·완전한 사본 생성", ["audit.chain"]),
    C("11.10(c)", P11, "system", "archive", "Enduring Available", "보존기간 내내 정확·신속하게 조회되도록 기록 보호", ["onto.retention"], platform=True),
    C("11.10(d)", P11, "system", "access", "Attributable", "시스템 접근을 권한자로 제한", ["access.abac"], platform=True),
    C("11.10(e)", P11, "system", "audit_trail", "Attributable Contemporaneous Original Enduring",
      "보안·컴퓨터 생성·시각 기록 감사추적. 변경이 이전 기록을 가리지 않을 것", ["audit.chain"]),
    C("11.10(f)", P11, "system", "system", "Consistent", "허용된 단계 순서를 강제하는 운영 점검", ["audit.chain"]),
    C("11.10(g)", P11, "system", "access", "Attributable", "권한자만 사용·서명·변경하도록 권한 점검", ["access.abac"], platform=True),
    C("11.10(h)", P11, "system", "system", "Accurate", "입력 출처(장치)의 유효성 점검"),
    C("11.10(i)", P11, "org", "org", "", "개발·운영·사용자의 교육·훈련·경험"),
    C("11.10(j)", P11, "org", "org", "Attributable", "전자서명 행위 책임에 관한 서면 정책"),
    C("11.10(k)(1)", P11, "org", "org", "", "시스템 문서의 배포·접근·사용 통제"),
    C("11.10(k)(2)", P11, "org", "org", "Original", "시스템 문서 개정·변경 통제와 시간순 감사추적"),
    C("11.30", P11, "system", "system", "Original", "개방형 시스템: 암호화·전자서명 표준 등 추가 통제"),
    C("11.50(a)(1)", P11, "record", "signature", "Attributable", "서명 표시에 서명자 성명", ["onto.signature"], applies=["labnote", "report"]),
    C("11.50(a)(2)", P11, "record", "signature", "Contemporaneous", "서명 표시에 서명 일시", ["onto.signature"], applies=["labnote", "report"]),
    C("11.50(a)(3)", P11, "record", "signature", "Attributable", "서명 표시에 서명의 의미(검토·승인·책임·작성)", ["onto.signature"], applies=["labnote", "report"]),
    C("11.50(b)", P11, "record", "signature", "Legible Enduring", "서명 표시도 기록과 같은 통제·사람이 읽는 형태에 포함", ["onto.signature", "audit.chain"], applies=["labnote", "report"]),
    C("11.70", P11, "system", "signature", "Original Attributable", "서명이 기록에 연결돼 떼어내기·복사·이전으로 위조 불가", ["audit.chain"]),
    C("11.100(a)", P11, "system", "signature", "Attributable", "전자서명은 1인 고유, 재사용·재배정 금지", platform=True),
    C("11.100(b)", P11, "org", "org", "Attributable", "서명 부여 전 본인 확인"),
    C("11.100(c)", P11, "org", "org", "", "전자서명의 법적 효력을 FDA 에 인증"),
    C("11.200(a)(1)", P11, "system", "signature", "Attributable", "비생체 전자서명은 2개 이상 식별 요소(ID+비밀번호)", platform=True),
    C("11.200(a)(1)(i)", P11, "system", "signature", "Attributable", "연속 세션: 첫 서명 전 요소, 이후 1개 이상", platform=True),
    C("11.200(a)(1)(ii)", P11, "system", "signature", "Attributable", "비연속 서명: 매번 전 요소", platform=True),
    C("11.200(a)(2)", P11, "system", "signature", "Attributable", "실소유자만 사용", platform=True),
    C("11.200(a)(3)", P11, "system", "signature", "Attributable", "타인 사용에는 2인 이상 공모가 필요하도록 관리", platform=True),
    C("11.200(b)", P11, "system", "signature", "Attributable", "생체 서명은 실소유자만 사용 가능하게 설계"),
    C("11.300(a)", P11, "system", "access", "Attributable", "ID·비밀번호 조합의 고유성", platform=True),
    C("11.300(b)", P11, "system", "access", "Attributable", "ID·비밀번호 주기적 점검·회수·갱신", platform=True),
    C("11.300(c)", P11, "org", "org", "", "분실·유출 토큰 무효화·재발급 절차"),
    C("11.300(d)", P11, "system", "access", "Attributable", "무단 사용 방지·시도 탐지·즉시 보고", ["access.abac", "audit.chain"]),
    C("11.300(e)", P11, "org", "org", "", "토큰·카드 초기·주기 시험"),
    # ---- 21 CFR Part 58 (GLP) --------------------------------------------------------
    C("58.130(e)-기록", P58, "record", "record", "Attributable Legible Contemporaneous Accurate",
      "데이터는 직접·신속·판독 가능하게 기록, 기록일 기재·기록자 서명", ["integrity.attributable", "integrity.contemporaneous"], applies=["labnote"]),
    C("58.130(e)-정정", P58, "record", "record", "Original Attributable",
      "변경은 원래 기재를 가리지 않고 사유·일자·서명", ["onto.amendment", "integrity.original"], applies=["labnote", "report"]),
    C("58.130(e)-자동", P58, "system", "audit_trail", "Attributable Original",
      "자동 수집 데이터: 입력 책임자 식별, 변경 시 사유·일자·책임자", ["audit.chain"]),
    C("58.130(c)", P58, "record", "record", "Attributable", "검체에 시험계·시험·성상·채취일 식별", ["onto.specimen"], applies=["labnote"]),
    C("58.185(a)(1)", P58, "record", "report", "Complete", "최종보고서: 시험기관명·주소, 개시·종료일", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(2)", P58, "record", "report", "Complete", "목적·절차(시험계획 변경 포함)", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(3)", P58, "record", "report", "Complete", "통계 방법", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(4)", P58, "record", "report", "Complete Accurate", "시험·대조물질 식별(명칭·CAS·순도·조성)", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(5)", P58, "record", "report", "Complete", "투여 조건에서 시험·대조물질 안정성", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(6)", P58, "record", "report", "Complete", "방법", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(7)", P58, "record", "report", "Complete", "시험계 기술", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(8)", P58, "record", "report", "Complete", "용량·투여 방식·경로·기간", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(9)", P58, "record", "report", "Complete Accurate", "데이터 품질·무결성에 영향을 줄 수 있는 상황", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(10)", P58, "record", "report", "Attributable", "시험책임자·참여 과학자·감독자 명시", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(11)", P58, "record", "report", "Accurate Consistent",
      "데이터 변환·계산, 요약·분석·결론 — 원자료와 일치", ["onto.report_elements", "gateway.nli", "vcr.halluc"], applies=["report"]),
    C("58.185(a)(12)", P58, "record", "report", "Attributable", "참여 과학자 개별 보고서(서명·일자)", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(13)", P58, "record", "report", "Available", "검체·원자료·최종보고서 보관 위치", ["onto.report_elements"], applies=["report"]),
    C("58.185(a)(14)", P58, "record", "report", "Attributable", "QA 진술서(서명)", ["onto.report_elements"], applies=["report"]),
    C("58.185(b)", P58, "record", "signature", "Attributable Contemporaneous", "최종보고서에 시험책임자 서명·일자", ["onto.signature"], applies=["report"]),
    C("58.185(c)", P58, "record", "report", "Original Attributable", "정정·추가는 변경 부분·사유·서명·일자를 적은 개정으로만", ["onto.amendment"], applies=["report"]),
    C("58.190(a)", P58, "system", "archive", "Enduring", "원자료·문서·계획서·보고서·검체 보관", platform=True),
    C("58.190(b)", P58, "system", "archive", "Enduring Available", "질서 있는 보관·신속 조회, 열화 최소화", platform=True),
    C("58.190(c)", P58, "org", "org", "Attributable", "보관소 책임자 지정"),
    C("58.190(d)", P58, "system", "access", "Attributable", "권한자만 보관소 출입", ["access.abac"], platform=True),
    C("58.190(e)", P58, "system", "archive", "Available", "신속 조회를 위한 색인", platform=True),
    C("58.195(b)", P58, "system", "archive", "Enduring", "보존기간(승인 후 2년·제출 후 5년·종료 후 2년 중 최단)", ["onto.retention"]),
    C("58.195(c)", P58, "org", "org", "Enduring", "불안정 검체는 평가 가능한 동안만 보관"),
    C("58.195(d)", P58, "org", "org", "Available", "QA 마스터 스케줄·계획서·점검 기록 보관"),
    C("58.195(e)", P58, "org", "org", "", "교육·경력 요약·직무 기술 보관"),
    C("58.195(f)", P58, "org", "org", "", "장비 유지·교정·점검 기록 보관"),
    C("58.195(g)", P58, "system", "archive", "Original", "원본 또는 진본 사본 허용", ["audit.chain"]),
    C("58.195(h)", P58, "org", "org", "Available", "시설 폐쇄 시 의뢰자 보관소로 이관·FDA 통보"),
    C("58.35(a)", P58, "org", "org", "", "QA 조직은 시험 수행과 독립"),
    C("58.35(b)(1)", P58, "org", "org", "", "전체 시험 마스터 스케줄"),
    C("58.35(b)(2)", P58, "org", "org", "", "시험계획서 사본 보관"),
    C("58.35(b)(3)", P58, "org", "org", "Attributable", "주기 점검·서명된 점검 기록·무결성 문제 즉시 보고"),
    C("58.35(b)(4)", P58, "org", "org", "", "경영진·책임자에 정기 서면 보고"),
    C("58.35(b)(5)", P58, "record", "report", "Consistent", "계획서·SOP 이탈은 승인·문서화된 것만", ["onto.deviation"], applies=["report", "labnote"]),
    C("58.35(b)(6)", P58, "record", "report", "Accurate Consistent",
      "최종보고서가 방법·원자료를 정확히 반영하는지 검토", ["gateway.nli", "vcr.halluc"], applies=["report"]),
    C("58.35(b)(7)", P58, "record", "report", "Attributable", "QA 진술서(점검일·보고 사항) 작성·서명", ["onto.report_elements"], applies=["report"]),
    C("58.35(c)", P58, "org", "org", "", "QA 책임·기록·색인 방법 서면화"),
    C("58.35(d)", P58, "org", "org", "", "FDA 의 QA 절차 열람·점검 수행 확인"),
    # ---- 국가연구개발사업 연구노트 지침 ---------------------------------------------
    C("KR5①", KR, "org", "org", "", "기관은 연구노트 작성·보관·관리 자체규정 마련·운영"),
    C("KR5②", KR, "org", "org", "", "작성 환경·교육·인센티브 등 활성화"),
    C("KR5③", KR, "org", "org", "", "과제관리·연속성·지재권 보호에 활용, 연구자 통제 목적 금지"),
    C("KR6", KR, "org", "org", "", "연구자는 자체규정에 따라 작성·관리 의무 이행"),
    C("KR7①1", KR, "system", "signature", "Attributable", "전자연구노트: 기록자 전자서명인증 기능", platform=True),
    C("KR7①2", KR, "system", "audit_trail", "Contemporaneous", "전자연구노트: 기록 날짜·시각 자동기록", ["audit.chain"], platform=True),
    C("KR7①3", KR, "system", "audit_trail", "Original", "전자연구노트: 기록 위·변조 확인 기능", ["audit.chain"], platform=True),
    C("KR7②1", KR, "record", "record", "Attributable Complete", "서면연구노트: 기관명·과제명·일련번호·쪽번호가 적힌 제본", ["onto.labnote_cover"], applies=["labnote"]),
    C("KR7②2", KR, "record", "signature", "Attributable Contemporaneous", "서면연구노트: 기록자·확인자 서명 및 기록·서명 날짜",
      ["integrity.attributable", "integrity.contemporaneous"], applies=["labnote"]),
    C("KR7②3", KR, "org", "org", "", "서면연구노트: 기관장이 정한 기타 사항"),
    C("KR8①", KR, "org", "org", "", "협약에 따라 소속 연구자 작성 관리(보고서로 갈음 가능 과제 예외)"),
    C("KR8②", KR, "record", "record", "Consistent", "과제마다 연구노트 구분 작성", ["onto.single_project"], applies=["labnote"]),
    C("KR8③", KR, "org", "org", "", "참여 기관마다 각각 작성"),
    C("KR8④", KR, "org", "org", "Attributable", "연구자별 또는 공동 작성, 모든 연구자 작성 원칙"),
    C("KR8⑤", KR, "record", "record", "Accurate Complete Consistent",
      "위조·변조 없이 객관적 사실 기록, 제3자 재현 가능하도록", ["gateway.nli", "vcr.halluc", "integrity.complete", "integrity.consistent", "integrity.accurate"],
      applies=["labnote", "report"]),
    C("KR8⑥1", KR, "record", "record", "Original Attributable", "서면: 수정·삭제·자료 부착 시 날짜·서명", ["onto.amendment", "integrity.original"], applies=["labnote"]),
    C("KR8⑥2", KR, "record", "record", "Complete", "서면: 빈 공간에 사선·여백 표시(이미지 필요)"),
    C("KR9①", KR, "org", "org", "", "연구노트 권리는 작성 관리 기관 소유"),
    C("KR10①", KR, "org", "org", "Enduring", "소유 기관장의 보관·관리"),
    C("KR10②", KR, "org", "org", "", "보관·관리 담당 부서 지정"),
    C("KR10③", KR, "org", "org", "Available", "과제 종료·중단·참여 종료 시 연구노트 제출"),
    C("KR10④", KR, "system", "archive", "Enduring", "보존기간: 과제 종료일부터 30년(유형별 달리 정할 수 있음)", ["onto.retention"]),
    C("KR11①", KR, "org", "org", "Available", "열람 대상·범위 자체규정, 기록자 열람권 보장"),
    C("KR11②", KR, "system", "audit_trail", "Attributable", "열람 관리대장 구비", ["audit.chain"]),
    C("KR11③", KR, "org", "org", "", "자체규정에 따른 공개"),
    C("KR11④", KR, "org", "org", "", "성과 제출·평가·연구자 보호·지재권 활용 사용권 보장"),
    C("KR11⑤", KR, "org", "org", "", "사용 용도 외 사용·누설·유출·양도·매매 금지"),
    C("KR12①", KR, "system", "archive", "Enduring", "보존기간 경과 후 폐기 가능", ["onto.retention"]),
    C("KR12②", KR, "org", "org", "", "보존가치 없으면 기간 전 폐기(자체규정)"),
)
