"""합성 데이터 파이프라인 — 실제 bioRxiv 논문 → DPO Preference 쌍 + Gateway 평가셋.

GOONO 자체 데이터가 없는 단계에서, 제약·바이오 도메인 정합 데이터를 자동 구축한다.

흐름:
  bioRxiv preprint(실데이터 시드)
    → 정량 주장 문장 추출(extract)
    → 양성 y_w(출처+정확) / 음성 y_l(무출처·수치변조·가짜DOI·극성역전) 생성(synth)
    → DPO Preference 쌍 + Gateway 평가 아이템(JSONL)
    → VCR 로 자기검증(VCR(y_w) > VCR(y_l) 승률)
"""

from compliance_gateway.data.models import DPOPair, GatewayEvalItem, Preprint
from compliance_gateway.data.synth import build_examples, load_seed

__all__ = ["Preprint", "DPOPair", "GatewayEvalItem", "build_examples", "load_seed"]
