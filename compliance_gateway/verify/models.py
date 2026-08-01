"""인용 검증 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CitationStatus(str, Enum):
    """3-class 인용 판정.

    VALID            : 실존 논문 + 메타데이터 일치
    PARTIALLY_VALID  : 같은 논문이나 메타데이터 불일치(저자·연도·제목 드리프트)
                       → 환각 유형 C(변조)의 서지 버전. 감사 시 지적 대상.
    HALLUCINATED     : 어떤 DB 에서도 매칭 실패 → 유형 A/B
    UNVERIFIED       : 조회 불가(오프라인·레이트리밋) → 판정 보류(차단 아님)
    """

    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    HALLUCINATED = "hallucinated"
    UNVERIFIED = "unverified"


@dataclass
class PaperRecord:
    """서지 DB(CrossRef/OpenAlex/S2/로컬) 의 정규화된 논문 레코드."""

    doi: Optional[str] = None
    title: str = ""
    authors: tuple[str, ...] = ()
    year: Optional[int] = None
    source: str = "local"          # crossref | openalex | semantic_scholar | local

    def first_author_surname(self) -> str:
        if not self.authors:
            return ""
        first = self.authors[0]
        # "Alam, W." / "W. Alam" / "Alam" 모두 성(姓)으로 정규화
        return (first.split(",")[0] if "," in first else first.split()[-1]).strip().lower()


@dataclass
class VerificationResult:
    """검증 결과 — 감사추적에 기록되는 단위."""

    status: CitationStatus
    score: float = 0.0                       # 가중 메타데이터 유사도 [0,1]
    matched: Optional[PaperRecord] = None
    stage: str = "none"                      # exact | fuzzy | llm | none
    mismatches: tuple[str, ...] = ()         # 불일치 필드 (title/author/year)
    detail: dict = field(default_factory=dict)

    @property
    def is_hallucinated(self) -> bool:
        return self.status is CitationStatus.HALLUCINATED

    @property
    def penalty(self) -> float:
        """Halluc 점수 기여도 [0,1]. 1.0 = 완전 환각.

        PARTIALLY_VALID 는 부분 감점(메타데이터 드리프트 — 논문은 실존).
        UNVERIFIED 는 감점하지 않는다(조회 실패로 정상 인용을 벌하지 않음).
        """
        return {
            CitationStatus.VALID: 0.0,
            CitationStatus.PARTIALLY_VALID: 0.5,
            CitationStatus.HALLUCINATED: 1.0,
            CitationStatus.UNVERIFIED: 0.0,
        }[self.status]
