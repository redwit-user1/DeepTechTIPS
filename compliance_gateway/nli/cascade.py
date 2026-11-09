"""판정 캐스케이드 — 규칙 → 인코더 NLI → (애매 구간만) LLM 판정기.

한국어 연구노트 평가(연구노트 2026-11-09 §18, 2-fold 홀드아웃)에서
LLM 호출 9% 로 F1 90.4%, 26% 로 93.2%. 인코더 단독은 단위 바꿈·단정화를 놓치고
LLM 단독(96.6%)은 비싸다.

반환값은 "근거가 문장을 뒷받침할 확률"로 해석되는 점수다. 인코더가 확신하는 구간은
인코더 점수를, 애매 구간은 LLM 점수를 그대로 돌려주므로 **임계값 하나(기본 0.1)**로
게이팅한다(lo < 임계값 < hi 이므로 인코더 구간의 판정과 일치한다).

    nli = CascadeNLI.default(device="cuda")
    ComplianceGateway(nli_fn=nli, sentence_threshold=CascadeNLI.THRESHOLD)
"""

from __future__ import annotations

from typing import Callable, Optional

from compliance_gateway.vcr.hallucination import unit_swaps

NLIFn = Callable[[str, str], float]


class CascadeNLI:
    # 26% 호출 예산 구성(홀드아웃 F1 93.2%). 호출을 줄이려면 (0.05, 0.5) → 약 9%.
    LO, HI, THRESHOLD = 0.005, 0.8, 0.1

    def __init__(self, encoder: NLIFn, llm: Optional[NLIFn] = None,
                 lo: float = LO, hi: float = HI) -> None:
        assert lo < self.THRESHOLD < hi, "lo < 임계값 < hi 여야 인코더 구간 판정이 일관된다"
        self.encoder, self.llm, self.lo, self.hi = encoder, llm, lo, hi
        self.llm_calls = self.calls = 0

    def __call__(self, premise: str, hypothesis: str) -> float:
        self.calls += 1
        if unit_swaps(hypothesis, premise):      # 값은 같고 단위만 바뀜 → 모순
            return 0.0
        s = self.encoder(premise, hypothesis)
        if self.llm is None or s < self.lo or s >= self.hi:
            return s
        self.llm_calls += 1
        return self.llm(premise, hypothesis)

    @classmethod
    def default(cls, device: str = "cuda", llm: bool = True) -> "CascadeNLI":
        from compliance_gateway.nli.transformer import TransformerNLI

        enc = TransformerNLI("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device=device)
        judge = None
        if llm:
            from compliance_gateway.nli.llm_judge import LLMJudgeNLI
            judge = LLMJudgeNLI("Qwen/Qwen3.8-27B", device=device)
        return cls(enc, judge)
