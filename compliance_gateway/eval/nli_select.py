"""NLI 백엔드 선택 — 로컬 트랜스포머 / 서빙 엔드포인트 / 통계 폴백."""

from __future__ import annotations

from typing import Optional


def select_nli(
    nli_model: Optional[str] = None,
    endpoint: Optional[str] = None,
    endpoint_model: Optional[str] = None,
    device: str = "cuda",
):
    """(nli_fn, 백엔드 이름) 반환.

    우선순위: 서빙 엔드포인트 > 로컬 트랜스포머 > 통계 v0.5
    엔드포인트는 학습 환경 없이 추론만 가능한 경우(AI Serv)를 위한 경로다.
    """
    if endpoint:
        from compliance_gateway.serving import ServedNLI

        if not endpoint_model:
            raise SystemExit("--nli-endpoint 사용 시 --nli-model 로 모델 ID 를 지정하세요.")
        return ServedNLI(endpoint, endpoint_model), f"served({endpoint_model})"
    if nli_model:
        from compliance_gateway.nli.transformer import TransformerNLI

        return TransformerNLI(model_name=nli_model, device=device), f"transformer({nli_model})"
    from compliance_gateway.nli.statistical import StatisticalNLI

    return StatisticalNLI(), "statistical-v0.5"


def add_nli_args(ap) -> None:
    """평가 CLI 공통 인자."""
    ap.add_argument("--nli", default=None,
                    help="로컬 트랜스포머 NLI 경로/ID (예: checkpoints/nli)")
    ap.add_argument("--nli-endpoint", default=None,
                    help="OpenAI 호환 서빙 엔드포인트 (예: http://host:8000/v1)")
    ap.add_argument("--nli-model", default=None, help="서빙 모델 ID")
    ap.add_argument("--device", default="cuda", help="로컬 NLI 디바이스")
