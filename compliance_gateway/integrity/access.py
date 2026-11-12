"""연구기록 접근통제 v1 — RBAC(역할) + ABAC(과제 소속·기록 상태·민감도·실행 위치).

근거 조항: Part 11 §11.10(d)(g)(권한자만 접근·서명·변경), §11.300(d)(무단 시도 탐지·보고),
Part 58 §58.190(d)(권한자만 보관소 출입), 연구노트 지침 제10조②(관리 부서)·제11조(열람)·제12조(폐기),
아키텍처 원칙 "에이전트는 판단하지 않는다 — 연구자 승인 필수".

거부도 기록한다(`AuditChain` 에 access_denied) — 무단 시도 탐지(11.300(d)).
"""

from __future__ import annotations

from dataclasses import dataclass

ROLES = ("recorder", "reviewer", "manager", "viewer", "agent")
ACTIONS = ("create", "modify", "delete", "sign", "view")


@dataclass(frozen=True)
class AccessRequest:
    user: str
    role: str                     # recorder 기록자 | reviewer 확인자 | manager 관리부서 | viewer 열람자 | agent AI
    action: str
    in_project: bool              # 요청자가 기록의 과제에 소속됐는가
    is_owner: bool = False        # 요청자가 기록 작성자인가
    state: str = "draft"          # draft | signed | archived
    sensitive: bool = False       # 민감(건강)·개인정보 포함 기록
    location: str = "onprem"      # onprem | api — 요청이 실행될 곳(외부 API 로 내보내는가)
    retention_expired: bool = False
    meaning: str = ""             # sign 일 때: 작성 | 확인 | 승인


@dataclass(frozen=True)
class AccessDecision:
    allow: bool
    reason: str


def decide(r: AccessRequest) -> AccessDecision:
    """순서가 곧 우선순위다 — 먼저 걸리는 거부 사유를 돌려준다."""
    def no(reason: str) -> AccessDecision:
        return AccessDecision(False, reason)

    if r.role not in ROLES or r.action not in ACTIONS:
        return no("알 수 없는 역할·행위")
    if r.sensitive and r.location != "onprem":
        return no("민감 기록은 외부 API 로 내보낼 수 없음(온프레미스 전용)")
    if r.role == "manager":
        if r.action == "view":
            return AccessDecision(True, "관리 부서 열람(지침 10②)")
        if r.action == "delete" and r.state == "archived" and r.retention_expired:
            return AccessDecision(True, "보존기간 경과 기록 폐기(지침 12①)")
        return no("관리 부서는 열람·보존기간 경과 폐기만 가능(내용 변경·서명 불가)")
    if not r.in_project:
        return no("과제 비소속 — 권한 없음(11.10(d))")
    if r.state == "archived" and r.action != "view":
        return no("보관 중 기록은 열람만 가능")
    if r.action == "view":
        return AccessDecision(True, "과제 소속 열람(지침 11①)")
    if r.role == "viewer":
        return no("열람자는 열람만 가능")
    if r.role == "agent":
        if r.action in ("sign", "delete"):
            return no("에이전트는 서명·삭제 불가 — 연구자 승인 필수")
        if r.state != "draft":
            return no("에이전트는 초안만 작성·수정")
        return AccessDecision(True, "에이전트 초안 작성")
    if r.action == "delete":
        return no("삭제 불가 — 정정은 사유를 남긴 수정으로(58.130(e))") if r.state != "draft" or not r.is_owner \
            else AccessDecision(True, "작성자 본인 초안 삭제")
    if r.action == "create":
        return AccessDecision(True, "과제 소속 기록 작성") if r.role == "recorder" else no("기록 작성은 기록자만")
    if r.action == "modify":
        if r.role != "recorder" or not r.is_owner:
            return no("수정은 작성자 본인만(11.10(g))")
        return AccessDecision(True, "작성자 수정(사유는 감사추적에 기록)")
    # sign
    if r.role == "recorder":
        return AccessDecision(True, "작성자 서명") if r.is_owner and r.meaning == "작성" else no("기록자는 본인 기록에 '작성' 서명만")
    if r.role == "reviewer":
        if r.is_owner:
            return no("본인 기록을 확인 서명할 수 없음(기록자·확인자 분리)")
        return AccessDecision(True, "확인자 서명") if r.meaning in ("확인", "승인") else no("확인자 서명 의미는 확인·승인")
    return no("규칙 없음 — 기본 거부")
