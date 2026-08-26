# -*- coding: utf-8 -*-
"""연구노트 기록 무결성 검사 — ALCOA+ 속성별 텍스트 검증.

국가연구개발 연구노트 지침의 필수 요건을 ALCOA+ 로 매핑해 구현한다.
각 검사는 (점수 0~1, 지적사항) 을 돌려주며, 감사 대응을 위해
**왜 위반인지 문장으로 설명**한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# 서명·일자 블록
_RESEARCHER = re.compile(r"연구자\s*[:：]\s*(.*)")
_REVIEWER = re.compile(r"점검자\s*[:：]\s*(.*)")
_EXP_DATE = re.compile(r"실험일\s*[:：]\s*(\d{4}-\d{2}-\d{2})")
_REC_DATE = re.compile(r"기록일\s*[:：]\s*(\d{4}-\d{2}-\d{2})")
_NO_SIGN = re.compile(r"\(\s*서명\s*없음\s*\)|미기재|없음")

# 수치·단위 (한국어 R&D 표기 포함)
_NUM_UNIT = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?)[ \t]*(℃|°C|%p|%|μM|mM|nM|N/m|ms|초|분|시간|일|주|개월|년|"
    r"nm|㎛|μm|mm|cm|mg|kg|g|mL|L|rpm|명|건|회|배|점)"
)
# 숫자 뒤의 쉼표·마침표는 문장부호이지 단위가 아니다 → 제외하면 안 된다
_BARE_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w%℃])")
_INCOMPLETE = re.compile(r"TBD|추후\s*기재|작성\s*예정|…|\.\.\.|미정|해당\s*없음\s*$")
# 무차원 지표 — 단위가 없는 것이 정상이다(F1, AUC, pH, 상관계수 등)
_DIMENSIONLESS = re.compile(
    r"(F1|AUC|ROC|정확도|재현율|정밀도|pH|지수|상관계수|R2|R\^?2|p-?value|비율|OD\d*|"
    r"Child[- ]?Pugh|ECOG|점수|score)\s*(?:는|은|이|가|을|를)?\s*$", re.IGNORECASE)
# 파생·비교 표현 — 결과값을 재진술한 게 아니라 계산해서 말한 것이므로 대조 대상이 아니다
_DERIVED = re.compile(r"단축|절감|향상|개선|증가|감소|대비|배\s|상회|하회|수준")
# 재진술 표현 — 결과를 다시 말한 경우에만 일관성 대조를 한다
_RESTATE = re.compile(r"확인되었|측정되었|산출되었|나타났|기록되었|대표값")
# 향후 계획 문장 — 새 수치가 등장해도 모순이 아니다
_PLAN_MARKER = re.compile(r"차기|향후|예정|계획|검토할|추가로|제안한다|필요하다")
_CORRECTION = re.compile(r"정정\s*이력|정정\s*사유|수정\s*사유|원본\s*보존")

# 동시성 허용 지연(일) — 지침상 실험 당일 기록이 원칙, 통상 3일 이내를 허용치로 본다
CONTEMPORANEOUS_LIMIT_DAYS = 3


@dataclass
class IntegrityReport:
    """ALCOA+ 속성별 무결성 결과."""

    scores: dict[str, float] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0

    @property
    def violated(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.scores.items() if v < 1.0)

    def as_dict(self) -> dict:
        return {"overall": round(self.overall, 4),
                "scores": {k: round(v, 4) for k, v in self.scores.items()},
                "violated": list(self.violated), "findings": list(self.findings)}


class RecordIntegrityChecker:
    """연구노트 텍스트의 ALCOA+ 무결성을 점검한다."""

    def __init__(self, contemporaneous_limit: int = CONTEMPORANEOUS_LIMIT_DAYS) -> None:
        self.limit = contemporaneous_limit

    # ---- Attributable: 누가 기록했는가 -------------------------------
    def _attributable(self, text: str, rep: IntegrityReport) -> None:
        score = 1.0
        m = _RESEARCHER.search(text)
        if not m or not m.group(1).strip() or _NO_SIGN.search(m.group(1)):
            score -= 0.6
            rep.findings.append("[Attributable] 연구자 서명이 없어 기록 주체를 특정할 수 없음")
        r = _REVIEWER.search(text)
        if not r or not r.group(1).strip() or _NO_SIGN.search(r.group(1)):
            score -= 0.4
            rep.findings.append("[Attributable] 점검자(입회자) 서명 누락 — 제3자 확인 부재")
        rep.scores["Attributable"] = max(0.0, score)

    # ---- Contemporaneous: 언제 기록했는가 ----------------------------
    def _contemporaneous(self, text: str, rep: IntegrityReport) -> None:
        e, r = _EXP_DATE.search(text), _REC_DATE.search(text)
        if not e or not r:
            rep.scores["Contemporaneous"] = 0.0
            rep.findings.append("[Contemporaneous] 실험일 또는 기록일 미기재")
            return
        delay = (date.fromisoformat(r.group(1)) - date.fromisoformat(e.group(1))).days
        if delay < 0:
            rep.scores["Contemporaneous"] = 0.0
            rep.findings.append(f"[Contemporaneous] 기록일이 실험일보다 빠름({delay}일) — 기재 오류")
        elif delay <= self.limit:
            rep.scores["Contemporaneous"] = 1.0
        else:
            # 지연이 클수록 감점(30일 이상이면 0점)
            rep.scores["Contemporaneous"] = max(0.0, 1.0 - (delay - self.limit) / 30)
            rep.findings.append(
                f"[Contemporaneous] 실험일 대비 기록일 {delay}일 지연 — 소급 작성 의심")

    # ---- Original: 원본을 식별할 수 있는가 ---------------------------
    def _original(self, text: str, grounding: Optional[str], rep: IntegrityReport) -> None:
        """정정 이력 없이 **값이 바뀐** 경우만 위반으로 본다.

        값이 단순히 *사라진* 것은 단위 누락(Accurate)이나 기재 누락(Complete) 소관이다.
        여기서 함께 잡으면 한 위반이 여러 속성을 오염시켜 진단이 흐려진다.
        """
        if grounding is None:
            rep.scores["Original"] = 1.0
            return

        def by_unit(t: str) -> dict[str, set[str]]:
            out: dict[str, set[str]] = {}
            for v, u in _NUM_UNIT.findall(t):
                out.setdefault(u, set()).add(v)
            return out

        # **결과부 단위로 비교**한다. 문서 전체로 비교하면, 바뀐 값이 방법부 등
        # 다른 구획에도 남아 있을 때 "사라진 값 없음"이 되어 변조를 놓친다.
        cur, ref = by_unit(self._results_section(text)), by_unit(self._results_section(grounding))
        # **교체**만 변조로 본다: 원본에만 있는 값과 현재에만 있는 값이 동시에 존재.
        #   값이 빠지기만 하면(단위 누락·기재 누락) → Accurate/Complete 소관
        #   값이 더해지기만 하면(고찰의 모순 수치) → Consistent 소관
        modified = []
        for u in ref:
            if u not in cur:
                continue
            gone, added = ref[u] - cur[u], cur[u] - ref[u]
            if gone and added:
                modified.append(f"{sorted(gone)[0]}{u}→{sorted(added)[0]}{u}")
        if modified and not _CORRECTION.search(text):
            rep.scores["Original"] = 0.0
            rep.findings.append(
                f"[Original] 정정 이력 없이 수치가 변경됨({', '.join(modified[:3])}) — 원본 식별 불가")
        else:
            rep.scores["Original"] = 1.0

    # ---- Accurate: 값이 해석 가능한가 --------------------------------
    def _accurate(self, text: str, rep: IntegrityReport) -> None:
        body = self._results_section(text)
        with_unit = len(_NUM_UNIT.findall(body))
        stripped = _NUM_UNIT.sub(" ", body)
        # 무차원 지표(F1 0.912 등)는 단위가 없는 게 정상 → 감점 대상에서 제외
        bare = len([
            m for m in _BARE_NUM.finditer(stripped)
            if not _DIMENSIONLESS.search(stripped[max(0, m.start() - 20):m.start()])
        ])
        total = with_unit + bare
        if total == 0:
            rep.scores["Accurate"] = 1.0
            return
        ratio = with_unit / total
        rep.scores["Accurate"] = ratio
        if ratio < 0.7:
            rep.findings.append(
                f"[Accurate] 결과부 수치 중 단위 미표기 {bare}건 — 값의 해석이 불가능")

    # ---- Complete: 누락이 없는가 -------------------------------------
    def _complete(self, text: str, rep: IntegrityReport) -> None:
        if _INCOMPLETE.search(text):
            rep.scores["Complete"] = 0.0
            rep.findings.append("[Complete] 미완결 표현(TBD/추후 기재/… ) 발견 — 기재 누락")
        else:
            rep.scores["Complete"] = 1.0

    # ---- Consistent: 내부 모순이 없는가 ------------------------------
    def _consistent(self, text: str, rep: IntegrityReport) -> None:
        """고찰이 결과를 **재진술**할 때만 대조한다.

        "차기 실험에서 17.5% 조건을 검토" 같은 향후 계획은 새 수치가 나오는 게 정상이므로
        제외한다. 이를 구분하지 않으면 정상 노트가 대량 오탐된다.
        """
        results = set(_NUM_UNIT.findall(self._results_section(text)))
        disc_text = self._section(text, "5. 고찰")
        restated = " ".join(
            sent for sent in re.split(r"(?<=다)\.\s*", disc_text)
            if _RESTATE.search(sent)                  # 결과를 다시 말한 문장만
            and not _PLAN_MARKER.search(sent)         # 향후 계획 제외
            and not _DERIVED.search(sent)             # 계산된 파생값 제외
        )
        disc = set(_NUM_UNIT.findall(restated))
        by_unit: dict[str, set[str]] = {}
        for v, u in results:
            by_unit.setdefault(u, set()).add(v)
        conflict = [f"{v}{u}" for v, u in disc if u in by_unit and v not in by_unit[u]]
        if conflict:
            rep.scores["Consistent"] = 0.0
            rep.findings.append(
                f"[Consistent] 고찰의 수치({', '.join(conflict[:3])})가 결과부와 불일치")
        else:
            rep.scores["Consistent"] = 1.0

    # ---- 구획 추출 ----------------------------------------------------
    @staticmethod
    def _section(text: str, header: str) -> str:
        m = re.search(rf"{re.escape(header)}[^\n]*\n(.*?)(?=\n\d\.\s|\Z)", text, re.DOTALL)
        return m.group(1) if m else ""

    def _results_section(self, text: str) -> str:
        return self._section(text, "4. 실험 결과") or text

    def check(self, text: str, grounding: Optional[str] = None) -> IntegrityReport:
        rep = IntegrityReport()
        self._attributable(text, rep)
        self._contemporaneous(text, rep)
        self._original(text, grounding, rep)
        self._accurate(text, rep)
        self._complete(text, rep)
        self._consistent(text, rep)
        return rep


def check_lab_note(text: str, grounding: Optional[str] = None) -> IntegrityReport:
    """편의 함수."""
    return RecordIntegrityChecker().check(text, grounding)
