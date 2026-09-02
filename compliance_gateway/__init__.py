"""GOONO AI — Compliance Gateway.

규정준수(Compliance) 기반 Agentic AI 오케스트레이션 엔진의 1순위 모듈.
모든 AI 출력이 통과하는 7단계 검증 파이프라인과 핵심 IP인 VCR 보상함수를 제공한다.
"""

from compliance_gateway.models import (
    Citation,
    GatewayResult,
    GenerationRequest,
    StageReport,
    VCRBreakdown,
)
from compliance_gateway.vcr.reward import DEFAULT_WEIGHTS, compute_vcr

__all__ = [
    "Citation",
    "GatewayResult",
    "GenerationRequest",
    "StageReport",
    "VCRBreakdown",
    "DEFAULT_WEIGHTS",
    "compute_vcr",
]
