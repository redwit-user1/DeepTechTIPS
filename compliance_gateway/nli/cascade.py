"""판정 캐스케이드 — 규칙 → 인코더 NLI → (애매 구간·근사 수치만) LLM 판정기.

    1. 단위 바꿈(값 같고 단위만 다름) → 위반
    2. 근사·환산 수치(근거와 10% 안쪽이지만 그대로는 아님) → LLM
    3. 인코더 점수 s: s < lo → 위반, s ≥ hi → 통과, 그 사이 → LLM

한국어 연구노트 평가(연구노트 2026-11-09 §25, 2-fold 홀드아웃): 1차 인코더를 **Qwen 판정 증류 모델**로
두면 LLM 호출 8% 로 F1 95.4%(P 93.8%), 어려운 셋 89.9%. 학습 없는 DeBERTa 는 같은 예산에서 90.3%.

반환값은 "근거가 문장을 뒷받침할 확률"로 해석되는 점수다. 인코더가 확신하는 구간은
인코더 점수를, 애매 구간은 LLM 점수를 그대로 돌려주므로 **임계값 하나(기본 0.1)**로
게이팅한다(lo < 임계값 < hi 이므로 인코더 구간의 판정과 일치한다).

    nli = CascadeNLI.default(device="cuda")
    ComplianceGateway(nli_fn=nli, sentence_threshold=CascadeNLI.THRESHOLD)
"""

from __future__ import annotations

from typing import Callable, Optional

from compliance_gateway.vcr.hallucination import approx_numbers, unit_swaps

NLIFn = Callable[[str, str], float]


class CascadeNLI:
    # 증류 인코더 기준(홀드아웃 F1 95.4%, LLM 8%). 학습 없는 DeBERTa 면 (0.005, 0.8, 0.1).
    LO, HI, THRESHOLD = 0.005, 0.98, 0.005
    STUDENT = "checkpoints/nli_distill"     # scripts/distill.py → nli_finetune 산출물(NAS)

    def __init__(self, encoder: NLIFn, llm: Optional[NLIFn] = None,
                 lo: float = LO, hi: float = HI, threshold: float = THRESHOLD) -> None:
        assert lo <= threshold < hi, "lo ≤ 임계값 < hi 여야 인코더 구간 판정이 일관된다"
        self.encoder, self.llm, self.lo, self.hi, self.threshold = encoder, llm, lo, hi, threshold
        self.llm_calls = self.calls = 0

    def __call__(self, premise: str, hypothesis: str) -> float:
        self.calls += 1
        if unit_swaps(hypothesis, premise):      # 값은 같고 단위만 바뀜 → 모순
            return 0.0
        if self.llm is not None and approx_numbers(hypothesis, premise):
            self.llm_calls += 1                  # 근사·환산 수치 — 인코더가 확신하며 틀리는 구간
            return self.llm(premise, hypothesis)
        s = self.encoder(premise, hypothesis)
        if self.llm is None or s < self.lo or s >= self.hi:
            return s
        self.llm_calls += 1
        return self.llm(premise, hypothesis)

    @classmethod
    def default(cls, device: str = "cuda", llm: bool = True, student: str = STUDENT) -> "CascadeNLI":
        """증류 모델이 있으면 1차로 쓰고, 없으면 학습 없는 DeBERTa-large(임계값도 그에 맞춤)."""
        from pathlib import Path

        from compliance_gateway.nli.transformer import TransformerNLI

        judge = None
        if llm:
            from compliance_gateway.nli.llm_judge import LLMJudgeNLI
            judge = LLMJudgeNLI("Qwen/Qwen3.8-27B", device=device)
        if Path(student).exists():
            return cls(TransformerNLI(student, device=device), judge)
        enc = TransformerNLI("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device=device)
        return cls(enc, judge, lo=0.005, hi=0.8, threshold=0.1)
