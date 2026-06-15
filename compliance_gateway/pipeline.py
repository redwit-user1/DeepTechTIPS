"""Compliance Gateway — 7단계 파이프라인 오케스트레이션.

01 사용자 질의 → 02 Pre-Gen 제약주입 → 03 SLM/API 생성 → 04 출처 바인딩
→ 05 NLI 게이팅 → 06 ALCOA+ 체크 → 07 블록체인 감사추적

본 모듈은 03(생성) 결과를 입력으로 받아 04~07을 수행한다.
03은 SLM/API 어댑터가 담당하며, 02는 GenerationRequest 구성 전에 적용된다.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from compliance_gateway.models import (
    Citation,
    GatewayDecision,
    GatewayResult,
    GatingMode,
    GenerationRequest,
    StageReport,
    VCRBreakdown,
)
from compliance_gateway.vcr.hallucination import DOIResolver
from compliance_gateway.vcr.reward import compute_vcr
from compliance_gateway.vcr.source_exist import extract_citations
from compliance_gateway.vcr.source_match import NLIFn


class ComplianceGateway:
    """모든 AI 출력이 통과하는 검증 게이트웨이.

    Parameters
    ----------
    vcr_threshold:
        게이팅 통과 임계값 θ. VCR 이 이 값 미만이면 재생성/차단.
    max_regenerations:
        임계값 미달 시 자동 재생성 최대 횟수.
    nli_fn / doi_resolver:
        외부 어댑터(미주입 시 휴리스틱).
    """

    def __init__(
        self,
        vcr_threshold: float = 0.60,
        max_regenerations: int = 3,
        nli_fn: Optional[NLIFn] = None,
        doi_resolver: Optional[DOIResolver] = None,
    ) -> None:
        self.vcr_threshold = vcr_threshold
        self.max_regenerations = max_regenerations
        # 기본 NLI 백엔드 = 통계적 v0.5(극성 처리). 운영 환경에서는
        # TransformerNLI 를 주입해 교체한다(HF/온프레미스 모델).
        if nli_fn is None:
            from compliance_gateway.nli.statistical import StatisticalNLI
            nli_fn = StatisticalNLI()
        self.nli_fn = nli_fn
        self.doi_resolver = doi_resolver

    # ---- 04 출처 바인딩 -------------------------------------------------
    def _source_binding(self, req: GenerationRequest) -> tuple[list[Citation], StageReport]:
        citations = extract_citations(req.response)
        report = StageReport(
            stage="source_binding",
            ok=bool(citations),
            detail={"n_citations": len(citations), "mode": req.mode.value},
        )
        return citations, report

    # ---- 05 NLI 게이팅 + 06 ALCOA+ (VCR 집계) --------------------------
    def _score(self, req: GenerationRequest, citations: list[Citation]) -> VCRBreakdown:
        return compute_vcr(
            query=req.query,
            response=req.response,
            grounding=req.grounding,
            model_id=req.model_id,
            citations=citations,
            nli_fn=self.nli_fn,
            doi_resolver=self.doi_resolver,
        )

    # ---- 07 블록체인 감사추적 (해시 생성) ------------------------------
    def _audit_hash(self, req: GenerationRequest, vcr: VCRBreakdown) -> str:
        """감사로그 해시. 실제 배포에서는 GOONO 시점인증(허가형 블록체인)에 기록."""
        payload = {
            "model_id": req.model_id,
            "query": req.query,
            "response": req.response,
            "vcr": vcr.as_dict(),
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _decide(self, vcr: float, regenerations: int) -> GatewayDecision:
        if vcr >= self.vcr_threshold:
            return GatewayDecision.PASS
        if regenerations < self.max_regenerations:
            return GatewayDecision.REGENERATE
        return GatewayDecision.BLOCK

    def evaluate(self, req: GenerationRequest, regenerations: int = 0) -> GatewayResult:
        """생성 완료된 응답(03)에 대해 04~07을 수행한다.

        mode=POST(API): 사후 검증. mode=LOGIT(자체 LoRA): 동일 집계를 사용하되
        실제 배포에서는 토큰 생성 중 개입 훅이 추가된다(M5).
        """
        reports: list[StageReport] = []

        citations, sb_report = self._source_binding(req)
        reports.append(sb_report)

        vcr = self._score(req, citations)
        reports.append(
            StageReport(
                stage="nli_gating",
                ok=vcr.vcr >= self.vcr_threshold,
                detail={"vcr": vcr.vcr, "threshold": self.vcr_threshold, "halluc": vcr.halluc},
            )
        )
        reports.append(
            StageReport(
                stage="alcoa_checkpoint",
                ok=vcr.alcoa_score >= 0.5,
                detail={"alcoa_score": vcr.alcoa_score},
            )
        )

        decision = self._decide(vcr.vcr, regenerations)
        audit_hash = self._audit_hash(req, vcr)
        reports.append(
            StageReport(stage="audit_trail", ok=True, detail={"audit_hash": audit_hash})
        )

        return GatewayResult(
            decision=decision,
            vcr=vcr,
            citations=citations,
            reports=reports,
            audit_hash=audit_hash,
            regenerations=regenerations,
        )


__all__ = ["ComplianceGateway", "GatewayDecision", "GatingMode", "GenerationRequest"]
