"""연구노트 레코드 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# 가상 데이터 표식 — 실제 기록으로 오인되지 않도록 모든 레코드에 부착
SYNTHETIC_PREFIX = "SYN"
SYNTHETIC_BANNER = "※ 본 기록은 시스템 검증용 가상 데이터입니다(실제 연구기록 아님)."


@dataclass
class LabNote:
    """연구노트 1페이지 분량 기록."""

    note_id: str                  # SYN-2024-0001-p012
    project_no: str               # SYN-RND-2024-017
    project_title: str
    domain: str                   # 화학·재료 / 바이오 / 공학·ICT
    researcher: str               # 가상 연구자명
    institution: str
    page: int

    exp_date: str                 # 실험 수행일 YYYY-MM-DD
    record_date: str              # 기록일 (동시성 판정 기준)
    reviewer: Optional[str] = None
    review_date: Optional[str] = None

    title: str = ""               # 실험 제목
    purpose: str = ""             # 실험 목적
    methods: str = ""             # 재료 및 방법
    results: str = ""             # 실험 결과
    discussion: str = ""          # 고찰 및 차기 계획
    correction: Optional[str] = None   # 정정 이력(원본 보존 여부)

    synthetic: bool = True
    violations: tuple[str, ...] = ()   # 위반 유형(정상이면 빈 튜플)
    alcoa_violated: tuple[str, ...] = ()

    def signature_block(self) -> str:
        sig = f"연구자: {self.researcher} (서명)" if self.researcher else "연구자: (서명 없음)"
        rec = f"기록일: {self.record_date}" if self.record_date else "기록일: (미기재)"
        if self.reviewer:
            rev = f"점검자: {self.reviewer} (서명) / 점검일: {self.review_date}"
        else:
            rev = "점검자: (서명 없음)"
        return f"{sig}\n{rec}\n{rev}"

    def render(self) -> str:
        """연구노트 원문 텍스트(txt) 렌더링."""
        lines = [
            SYNTHETIC_BANNER,
            "=" * 58,
            f"[연구노트] {self.note_id}   (p.{self.page})",
            f"과제번호: {self.project_no}",
            f"과제명: {self.project_title}",
            f"수행기관: {self.institution}",
            f"실험일: {self.exp_date}",
            self.signature_block(),
            "=" * 58,
            "",
            f"1. 실험 제목\n{self.title}",
            "",
            f"2. 실험 목적\n{self.purpose}",
            "",
            f"3. 재료 및 방법\n{self.methods}",
            "",
            f"4. 실험 결과\n{self.results}",
            "",
            f"5. 고찰 및 차기 계획\n{self.discussion}",
        ]
        if self.correction:
            lines += ["", f"6. 정정 이력\n{self.correction}"]
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "note_id": self.note_id, "project_no": self.project_no,
            "project_title": self.project_title, "domain": self.domain,
            "researcher": self.researcher, "institution": self.institution,
            "page": self.page, "exp_date": self.exp_date, "record_date": self.record_date,
            "reviewer": self.reviewer, "review_date": self.review_date,
            "title": self.title, "purpose": self.purpose, "methods": self.methods,
            "results": self.results, "discussion": self.discussion,
            "correction": self.correction, "text": self.render(),
            "synthetic": True,
            "violations": list(self.violations),
            "alcoa_violated": list(self.alcoa_violated),
            "lang": "ko",
        }
