"""NLI 백엔드 — SourceMatch / Halluc 의 entailment 스코어러.

모두 시그니처 `nli_fn(premise, hypothesis) -> float [0,1]` 를 따른다.
premise = 신뢰 근거(인용 출처 본문), hypothesis = 응답의 주장.
높을수록 "근거가 주장을 뒷받침함(entailment)".

- lexical:     토큰 중첩 베이스라인(기존 휴리스틱과 동일 계열)
- statistical: TF-IDF 코사인 + 극성(부정·반의어) 처리 = NLI v0.5 (의존성 없음)
- transformer: 사전학습 NLI 모델 백엔드 (HF/온프레미스 환경에서 활성)
"""

from compliance_gateway.nli.lexical import lexical_nli
from compliance_gateway.nli.statistical import StatisticalNLI

__all__ = ["lexical_nli", "StatisticalNLI"]
