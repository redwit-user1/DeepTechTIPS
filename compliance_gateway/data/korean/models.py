"""국내 연구과제 레코드 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KRResearchRecord:
    """국내 연구기관의 연구과제 레코드(실데이터).

    ALCOA+ 관점 매핑:
      sponsor    → Attributable (귀속가능: 누가 수행했는가)
      start_date → Contemporaneous (동시적: 언제 수행했는가)
      enrollment → Accurate (정확: 수치가 맞는가)
      nct_id     → Original (원본: 등록 원본 식별자)
    """

    nct_id: str
    title: str
    sponsor: str                       # 수행기관
    conditions: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    enrollment: Optional[int] = None
    start_date: str = ""
    study_type: str = ""
    phase: Optional[str] = None
    source: str = "clinicaltrials.gov"

    @property
    def year(self) -> Optional[int]:
        return int(self.start_date[:4]) if self.start_date[:4].isdigit() else None

    def citation(self) -> str:
        """국내 R&D 문서 관례의 인용 표기."""
        parts = [self.sponsor]
        if self.year:
            parts.append(f"{self.year}")
        return f"{', '.join(parts)}; {self.nct_id}"

    @classmethod
    def from_dict(cls, d: dict) -> "KRResearchRecord":
        return cls(
            nct_id=d["nct_id"], title=d["title"], sponsor=d["sponsor"],
            conditions=tuple(d.get("conditions") or ()),
            interventions=tuple(d.get("interventions") or ()),
            enrollment=d.get("enrollment"), start_date=d.get("start_date", ""),
            study_type=d.get("study_type", ""), phase=d.get("phase"),
        )
