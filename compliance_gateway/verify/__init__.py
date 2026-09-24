"""인용 검증(Citation Verification) — 3단계 파이프라인.

업스트림 참고: Citation-Hallucination-Detection (Vikranth3140) 의 3단계 구조
(exact lookup → fuzzy retrieval → LLM verification)와 3-class 판정을 채택하고,
본 과제의 ALCOA+/VCR 요구에 맞춰 재구현했다. → docs/UPSTREAM_TECH.md

기존 `DOIResolver = Callable[[str], bool]` 대비 개선점:
  - 존재/부재 이분법 → **VALID / PARTIALLY_VALID / HALLUCINATED** 3-class
  - PARTIALLY_VALID 가 '메타데이터 드리프트'(환각 유형 C)를 포착
  - 가중 메타데이터 유사도(title .60 / author .30 / year .10)로 정밀 판정
"""

from compliance_gateway.verify.models import (
    CitationStatus,
    PaperRecord,
    VerificationResult,
)
from compliance_gateway.verify.verifier import CitationVerifier, LocalRegistry

__all__ = [
    "CitationStatus",
    "PaperRecord",
    "VerificationResult",
    "CitationVerifier",
    "LocalRegistry",
]
