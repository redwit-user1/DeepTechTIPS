"""Compliance Gateway 데이터 모델.

표준 라이브러리(dataclasses)만 사용해 의존성 없이 동작한다.
NLI 모델·블록체인 등 외부 연동은 추후 어댑터로 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GatingMode(str, Enum):
    """모델 유형별 게이팅 경로.

    LOGIT: 자체 LoRA — 토큰 생성 중 in-process 개입(최고 정밀도).
    POST:  외부 API(GPT/Claude) — 로짓 미접근 → 생성 후 텍스트 기반 비동기 검증.
    """

    LOGIT = "logit"
    POST = "post"


class GatewayDecision(str, Enum):
    PASS = "pass"          # 검증 통과 → 사용자 노출
    REGENERATE = "regen"   # 임계값 미달 → 자동 재생성(최대 N회)
    BLOCK = "block"        # 재생성 실패 → 차단 / Human-in-the-loop


@dataclass
class Citation:
    """응답에서 추출한 인용/출처 한 건."""

    raw: str                          # 원문 인용 문자열
    doi: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    authors: tuple[str, ...] = ()
    year: Optional[int] = None
    # SourceMatch 단계에서 채워짐: 인용이 실제 주장을 뒷받침하는지
    verified: Optional[bool] = None
    match_score: float = 0.0


@dataclass
class GenerationRequest:
    """게이트웨이 입력."""

    query: str
    response: str
    mode: GatingMode = GatingMode.POST
    model_id: str = "unknown"
    # 인용이 대조될 신뢰 출처 컨텍스트(RAG로 검색된 근거 텍스트)
    grounding: tuple[str, ...] = ()
    domain: str = "pharma_bio"        # 초기 집중 도메인


@dataclass
class VCRBreakdown:
    """VCR 보상함수 세부 점수."""

    source_exist: float
    source_match: float
    alcoa_score: float
    halluc: float                      # 환각 점수(낮을수록 좋음)
    vcr: float                         # 가중 집계 결과 [0, 1]

    def as_dict(self) -> dict[str, float]:
        return {
            "source_exist": self.source_exist,
            "source_match": self.source_match,
            "alcoa_score": self.alcoa_score,
            "halluc": self.halluc,
            "vcr": self.vcr,
        }


@dataclass
class StageReport:
    """파이프라인 각 단계의 산출."""

    stage: str
    ok: bool
    detail: dict = field(default_factory=dict)


@dataclass
class GatewayResult:
    """게이트웨이 최종 결과 — 블록체인 감사추적에 기록되는 단위."""

    decision: GatewayDecision
    vcr: VCRBreakdown
    citations: list[Citation]
    reports: list[StageReport]
    audit_hash: Optional[str] = None   # 감사로그 해시(블록체인 시점인증)
    regenerations: int = 0
