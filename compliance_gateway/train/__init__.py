"""학습 파이프라인 — SFT / DPO / GRPO / NLI 파인튜닝.

A100 2장(HF 접근 가능) 환경 대상. 무거운 의존성(torch/transformers/trl/peft)은
실행 시점에 지연 임포트하므로, 설정·데이터 포맷 모듈은 GPU 없이도 import·테스트된다.

실행: docs/A100_PLAN.md 참조.
"""

from compliance_gateway.train import config, data_format

__all__ = ["config", "data_format"]
