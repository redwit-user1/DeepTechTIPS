"""VCR — Verifiable Compliance Reward.

VCR(y|x) = w1·SourceExist + w2·SourceMatch + w3·ALCOA_Score + w4·(1 − Halluc)

본 과제의 핵심 IP. 1차 구현은 의존성 없는 휴리스틱이며, 각 컴포넌트는
추후 NLI 모델·외부 DB 대조·온톨로지 추론으로 교체 가능하도록 분리돼 있다.
"""
