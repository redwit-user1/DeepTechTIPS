"""조항별 점검 · 커버리지 · Pre-Gen 제약 · Turtle 내보내기.

점검은 모두 `(text) -> list[str]`(지적 사항, 비면 통과). 지적 앞에 조항 ID 를 붙여 감사 대응에 쓴다.
키워드 기반이라 **있음/없음**만 본다 — 내용의 옳고 그름은 게이트웨이(NLI·수치)가 맡는다.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from typing import Callable, Optional

from compliance_gateway.ontology.clauses import ALCOA, CLAUSES, Clause

# ---- 서명 표시 (11.50(a), 58.185(b)) --------------------------------------------------
_SIGN_LINE = re.compile(r"서명|signed|signature", re.IGNORECASE)
_NAME = re.compile(r"[가-힣]{2,4}(?![가-힣])|인명_[0-9A-F]{4}|[A-Z][a-z]+\s[A-Z][a-z]+")
_DATE = re.compile(r"\d{4}\s*[-./년]\s*\d{1,2}\s*[-./월]\s*\d{1,2}")
_TIME = re.compile(r"\d{1,2}\s*[:시]\s*\d{2}")
_MEANING = re.compile(r"작성|검토|승인|확인|책임|authorship|author|review|approv|responsib", re.IGNORECASE)
_SIGN_WORDS = re.compile(r"서명|확인자|기록자|작성자|검토자|승인자|일시|날짜|일자|시각")


def signature_manifest(text: str) -> list[str]:
    """서명 줄마다 성명·일자·시각·의미가 있는지."""
    lines = [l for l in (text or "").splitlines() if _SIGN_LINE.search(l)]
    if not lines:
        return ["[11.50(a)/58.185(b)] 서명 표시 없음"]
    out = []
    for l in lines:
        tag = l.strip()[:30]
        if not _NAME.search(_SIGN_WORDS.sub(" ", l)):
            out.append(f"[11.50(a)(1)] 서명자 성명 없음: {tag}")
        if not _DATE.search(l):
            out.append(f"[11.50(a)(2)] 서명 일자 없음: {tag}")
        elif not _TIME.search(l):
            out.append(f"[11.50(a)(2)] 서명 시각 없음(일자만): {tag}")
        if not _MEANING.search(l):
            out.append(f"[11.50(a)(3)] 서명 의미(작성·검토·승인) 없음: {tag}")
    return out


# ---- 정정 기록 (58.130(e), 58.185(c), 지침 제8조⑥1) ------------------------------------
_AMEND = re.compile(r"^\s*[\[(<]?\s*(?:정정|수정\s*이력|변경\s*이력|삭제|개정|amendment|correction)\b", re.IGNORECASE | re.MULTILINE)
_REASON = re.compile(r"사유|이유|reason|because|때문", re.IGNORECASE)
_SIGNER = re.compile(r"서명|확인|initial|signed|인명_[0-9A-F]{4}|[가-힣]{2,4}\s*\(인\)", re.IGNORECASE)


def amendment_trail(text: str) -> list[str]:
    """정정·삭제·개정 항목마다 사유·일자·서명이 있는지. 정정 항목이 없으면 통과."""
    out = []
    for m in _AMEND.finditer(text or ""):
        line = text[m.start(): text.find("\n", m.end()) if "\n" in text[m.end():] else len(text)]
        tag = line.strip()[:30]
        if not _REASON.search(line):
            out.append(f"[58.130(e)/58.185(c)] 정정 사유 없음: {tag}")
        if not _DATE.search(line):
            out.append(f"[58.130(e)/지침8⑥1] 정정 일자 없음: {tag}")
        if not _SIGNER.search(line):
            out.append(f"[58.130(e)/지침8⑥1] 정정자 서명 없음: {tag}")
    return out


# ---- 계획서·SOP 이탈 (58.35(b)(5)) ----------------------------------------------------
_DEVIATION = re.compile(r"이탈|일탈|계획\s*변경|deviation", re.IGNORECASE)
_DEV_DOC = re.compile(r"승인|사유|보고|문서화|approv|reason|document", re.IGNORECASE)


def deviation_documented(text: str) -> list[str]:
    """이탈을 언급한 문장에 승인·사유가 함께 있는지. 이탈 언급이 없으면 통과."""
    sents = re.split(r"(?<=[.!?。])\s+|\n", text or "")
    return [f"[58.35(b)(5)] 이탈 승인·사유 없음: {s.strip()[:30]}"
            for s in sents if _DEVIATION.search(s) and not _DEV_DOC.search(s)]


# ---- 검체 식별 (58.130(c)) · 과제 구분 (지침 제8조②) -----------------------------------
_SPECIMEN = re.compile(r"검체|시료|샘플|specimen|sample", re.IGNORECASE)
_SPEC_ID = re.compile(r"(?:시료|검체|샘플|sample|specimen)\s*(?:번호|ID|No\.?|#)|[A-Z]{1,4}-?\d{2,}", re.IGNORECASE)
_PROJECT = re.compile(r"과제\s*명\s*[:：]\s*(.+)")


def specimen_labels(text: str) -> list[str]:
    """검체를 다루면 검체 식별자와 일자가 있어야 한다. 검체 언급이 없으면 통과."""
    t = text or ""
    if not _SPECIMEN.search(t):
        return []
    out = []
    if not _SPEC_ID.search(t):
        out.append("[58.130(c)] 검체 식별번호 없음")
    if not _DATE.search(t):
        out.append("[58.130(c)] 검체 채취·기록 일자 없음")
    return out


def single_project(text: str) -> list[str]:
    """한 연구노트에 과제명이 둘 이상이면 지적(과제마다 구분 작성)."""
    names = {m.group(1).strip() for m in _PROJECT.finditer(text or "")}
    return [f"[지침8②] 한 노트에 과제 {len(names)}개 — 과제마다 구분 작성"] if len(names) > 1 else []


# ---- 최종보고서 필수 항목 (58.185(a)) -------------------------------------------------
REPORT_ELEMENTS: dict[str, tuple[str, ...]] = {
    "58.185(a)(1)": (r"시험\s*기관|연구\s*기관|수행\s*기관|facility", r"개시|시작|종료|initiat|complet"),
    "58.185(a)(2)": (r"목적|objective", r"절차|방법|계획서|protocol|procedure"),
    "58.185(a)(3)": (r"통계|statistic",),
    "58.185(a)(4)": (r"시험\s*물질|대조\s*물질|시료|test article|control article|CAS",),
    "58.185(a)(5)": (r"안정성|stability",),
    "58.185(a)(6)": (r"방법|method",),
    "58.185(a)(7)": (r"시험계|시험\s*동물|세포주|test system|동물|균주",),
    "58.185(a)(8)": (r"용량|투여|농도|처리\s*(?:시간|기간)|dosage|dose|route|duration",),
    "58.185(a)(9)": (r"품질|무결성|영향을\s*(?:준|줄|미친|미칠)|이탈|일탈|deviation|circumstance|integrity",),
    "58.185(a)(10)": (r"책임자|study director|담당자|참여\s*연구원",),
    "58.185(a)(11)": (r"결론|요약|고찰|분석\s*결과|conclusion|summary",),
    "58.185(a)(12)": (r"개별\s*보고|분담\s*보고|individual report|참여자\s*보고",),
    "58.185(a)(13)": (r"보관\s*(?:위치|장소)|보존\s*(?:위치|장소)|저장\s*위치|archive|stored at",),
    "58.185(a)(14)": (r"신뢰성\s*보증|품질\s*보증|QA|quality assurance",),
}
_ELEM_NAME = {"58.185(a)(1)": "시험기관·개시/종료일", "58.185(a)(2)": "목적·절차", "58.185(a)(3)": "통계 방법",
              "58.185(a)(4)": "시험·대조물질", "58.185(a)(5)": "물질 안정성", "58.185(a)(6)": "방법",
              "58.185(a)(7)": "시험계", "58.185(a)(8)": "용량·투여·기간", "58.185(a)(9)": "데이터 품질 영향 상황",
              "58.185(a)(10)": "책임자·참여자", "58.185(a)(11)": "요약·분석·결론", "58.185(a)(12)": "참여자 개별 보고",
              "58.185(a)(13)": "자료 보관 위치", "58.185(a)(14)": "QA 진술", }


def report_elements(text: str) -> list[str]:
    t = text or ""
    return [f"[{cid}] 최종보고서 항목 누락: {_ELEM_NAME[cid]}"
            for cid, pats in REPORT_ELEMENTS.items() if not all(re.search(p, t, re.IGNORECASE) for p in pats)]


# ---- 서면연구노트 표지 (지침 제7조②1) -------------------------------------------------
_COVER = {
    "기관명": r"기관명|연구소|연구원(?![가-힣])|대학교|주식회사|㈜|\(주\)|Inc\.|Co\.,",
    "과제명": r"과제\s*명|과제\s*번호|연구\s*과제|project",
    "일련번호": r"일련\s*번호|노트\s*(?:번호|No)|No\.\s*\d+|#\s*\d+",
    "쪽번호": r"\d+\s*/\s*\d+\s*(?:쪽|page)?|(?:page|p\.)\s*\d+|-\s*\d+\s*-|\d+\s*쪽",
}


def labnote_cover(text: str) -> list[str]:
    return [f"[지침7②1] 연구노트 {k} 없음" for k, p in _COVER.items() if not re.search(p, text or "", re.IGNORECASE)]


# ---- 보존기간 (지침 제10조④·제12조①, 58.195(b), 11.10(c)) --------------------------------
def retention_until(end: date, years: int = 30) -> date:
    """보존 만료일. 2월 29일 종료는 2월 28일로."""
    try:
        return end.replace(year=end.year + years)
    except ValueError:
        return end.replace(year=end.year + years, day=28)


def retention_findings(end: date, disposal: Optional[date], years: int = 30) -> list[str]:
    """예정 폐기일이 보존기간 안이면 지적. 국내 연구노트 30년, GLP(58.195(b))는 2·5년 중 해당 기간을 넣는다."""
    until = retention_until(end, years)
    if disposal is not None and disposal < until:
        return [f"[지침10④/58.195(b)] 보존기간({years}년, {until}까지) 전 폐기 예정 {disposal}"]
    return []


# ---- 점검 레지스트리 -----------------------------------------------------------------
# 문서 텍스트만으로 돌릴 수 있는 점검. 그 밖의 키는 다른 모듈에 구현돼 있다(값 = 위치).
TEXT_CHECKS: dict[str, Callable[[str], list[str]]] = {
    "onto.signature": signature_manifest,
    "onto.amendment": amendment_trail,
    "onto.report_elements": report_elements,
    "onto.labnote_cover": labnote_cover,
    "onto.deviation": deviation_documented,
    "onto.specimen": specimen_labels,
    "onto.single_project": single_project,
}
REGISTRY: dict[str, str] = {
    **{k: f"compliance_gateway.ontology.checks.{f.__name__}" for k, f in TEXT_CHECKS.items()},
    "onto.retention": "compliance_gateway.ontology.checks.retention_findings",
    "audit.chain": "compliance_gateway.integrity.audit.AuditChain",
    "access.abac": "compliance_gateway.integrity.access.decide",
    "integrity.attributable": "compliance_gateway.integrity.record.RecordIntegrityChecker._attributable",
    "integrity.contemporaneous": "compliance_gateway.integrity.record.RecordIntegrityChecker._contemporaneous",
    "integrity.original": "compliance_gateway.integrity.record.RecordIntegrityChecker._original",
    "integrity.accurate": "compliance_gateway.integrity.record.RecordIntegrityChecker._accurate",
    "integrity.complete": "compliance_gateway.integrity.record.RecordIntegrityChecker._complete",
    "integrity.consistent": "compliance_gateway.integrity.record.RecordIntegrityChecker._consistent",
    "gateway.nli": "compliance_gateway.pipeline.ComplianceGateway._unsupported_sentences",
    "vcr.halluc": "compliance_gateway.vcr.hallucination.halluc",
}


def check_document(text: str, doc_type: str, skip: tuple[str, ...] = ()) -> list[str]:
    """문서 유형에 걸리는 조항의 텍스트 점검을 모두 돌린다(Post-Gen). 지적 중복 제거."""
    keys = {k for c in CLAUSES if doc_type in c.applies for k in c.checks if k in TEXT_CHECKS} - set(skip)
    out: list[str] = []
    for k in sorted(keys):
        for f in TEXT_CHECKS[k](text):
            if f not in out:
                out.append(f)
    return out


# ---- Pre-Gen 제약 (생성 전에 규정을 프롬프트로 주입) -----------------------------------
def constraints_for(doc_type: str) -> str:
    """문서 유형에 걸리는 record 조항을 생성 지시문으로. 서명은 사람이 하므로 자리만 남기게 한다."""
    lines = [f"- [{c.id}] {c.summary}" for c in CLAUSES if doc_type in c.applies]
    return ("다음 규정 요건을 지켜 작성하라. 근거에 없는 수치·사실은 쓰지 말고, 모르는 항목은 '확인 필요'로 남겨라. "
            "서명·일시는 사람이 채울 자리만 남겨라(대신 서명하지 말 것).\n" + "\n".join(lines))


# ---- 커버리지 -----------------------------------------------------------------------
def coverage(clauses: tuple[Clause, ...] = CLAUSES) -> dict:
    """규정별 커버리지.

    modeled   : 인벤토리 조항 중 L3(ALCOA+)·L2 객체가 붙은 비율 — 온톨로지 표현 범위.
    automated : 자동 점검 대상(record+system) 중 **이 레포에 구현된 점검**이 연결된 비율 — 실질 지표.
    with_platform : automated 에 GOONO 플랫폼 기능(미검증)을 더한 비율 — 참고치.
    """
    out = {}
    for reg in [None] + sorted({c.reg for c in clauses}):
        cs = [c for c in clauses if reg is None or c.reg == reg]
        auto = [c for c in cs if c.kind in ("record", "system")]
        out[reg or "전체"] = {
            "clauses": len(cs),
            "modeled": sum(bool(c.obj) for c in cs) / len(cs),
            "automatable": len(auto),
            "automated": sum(bool(c.checks) for c in auto) / max(1, len(auto)),
            "with_platform": sum(bool(c.checks) or c.platform for c in auto) / max(1, len(auto)),
            "uncovered": [c.id for c in auto if not c.checks and not c.platform],
        }
    return out


def alcoa_coverage(clauses: tuple[Clause, ...] = CLAUSES) -> dict[str, tuple[int, int]]:
    """ALCOA+ 속성별 (점검 연결 조항 수, 전체 조항 수)."""
    tot, cov = Counter(), Counter()
    for c in clauses:
        for a in c.alcoa:
            tot[a] += 1
            cov[a] += bool(c.checks)
    return {a: (cov[a], tot[a]) for a in ALCOA}


# ---- Turtle 내보내기 (Protégé 등에서 열람) ---------------------------------------------
def to_turtle(clauses: tuple[Clause, ...] = CLAUSES) -> str:
    def iri(s: str) -> str:
        return "go:" + re.sub(r"[^0-9A-Za-z가-힣_]", "_", s)

    def lit(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    out = ["@prefix go: <https://goono.ai/ontology/compliance#> .",
           "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", ""]
    for a in ALCOA:
        out.append(f"{iri(a)} a go:AlcoaAttribute .")
    for reg in sorted({c.reg for c in clauses}):
        out.append(f"{iri(reg)} a go:Regulation ; rdfs:label {lit(reg)} .")
    for k in sorted(REGISTRY):
        out.append(f"{iri('check_' + k)} a go:Check ; rdfs:label {lit(k)} ; go:implementedBy {lit(REGISTRY[k])} .")
    for c in clauses:
        props = [f"a go:Clause", f"rdfs:label {lit(c.id + ' ' + c.summary)}", f"go:partOf {iri(c.reg)}",
                 f"go:kind {lit(c.kind)}", f"go:object {lit(c.obj)}"]
        props += [f"go:requires {iri(a)}" for a in c.alcoa]
        props += [f"go:checkedBy {iri('check_' + k)}" for k in c.checks]
        if c.platform:
            props.append("go:providedByPlatform true")
        out.append(f"{iri(c.reg + '_' + c.id)} " + " ;\n    ".join(props) + " .")
    return "\n".join(out) + "\n"
