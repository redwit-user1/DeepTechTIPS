"""합성 데이터 파이프라인 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Preprint:
    doi: str
    title: str
    authors: str          # "Last, F.; Last2, F2; ..."
    date: str             # YYYY-MM-DD
    abstract: str
    category: str = ""
    license: str = ""

    @property
    def year(self) -> int:
        return int(self.date[:4])

    @property
    def first_author(self) -> str:
        first = self.authors.split(";")[0].strip()
        return first.split(",")[0].strip() or "Anonymous"

    def citation(self) -> str:
        """'Alam et al. (2024)' 형태의 인용 문자열."""
        suffix = " et al." if ";" in self.authors else ""
        return f"{self.first_author}{suffix} ({self.year})"


@dataclass
class DPOPair:
    """DPO Preference 쌍. chosen=규정준수(y_w), rejected=비준수(y_l)."""

    prompt: str
    chosen: str
    rejected: str
    rejected_kind: str          # no_source | numeric_tamper | fake_doi | polarity_flip
    grounding: str              # 진짜 근거(출처 본문 문장)
    source_doi: str
    vcr_chosen: float = 0.0
    vcr_rejected: float = 0.0

    @property
    def margin(self) -> float:
        return round(self.vcr_chosen - self.vcr_rejected, 4)

    def to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "rejected_kind": self.rejected_kind,
            "grounding": self.grounding,
            "source_doi": self.source_doi,
            "vcr_chosen": self.vcr_chosen,
            "vcr_rejected": self.vcr_rejected,
            "margin": self.margin,
        }


@dataclass
class GatewayEvalItem:
    """Gateway 평가 아이템. label=compliant 면 PASS, 그 외는 차단/재생성 기대."""

    query: str
    response: str
    grounding: str
    source_doi: str
    label: str                  # compliant | no_source | numeric_tamper | fake_doi | polarity_flip
    halluc_type: Optional[str] = None   # A | B | C (사업계획서 정의 환각 유형)

    def to_json(self) -> dict:
        return {
            "query": self.query,
            "response": self.response,
            "grounding": self.grounding,
            "source_doi": self.source_doi,
            "label": self.label,
            "halluc_type": self.halluc_type,
        }
