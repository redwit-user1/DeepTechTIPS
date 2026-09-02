"""Transformer NLI 백엔드 (사전학습 모델).

HuggingFace 또는 온프레미스 모델 경로가 접근 가능한 환경에서 활성화된다.
현재 개발 환경은 네트워크 정책상 huggingface.co 가 차단(403)되어 즉시 로드 불가.
→ SourceMatch/Halluc 의 운영 목표 백엔드이며, 함수 시그니처가 동일하므로
   `ComplianceGateway(nli_fn=...)` 에 그대로 주입해 통계적 baseline 을 교체한다.

권장 모델(제약·바이오 우선):
  - 과학 도메인:  scifact 파인튜닝 모델 / biomed NLI
  - 범용 NLI:     DeBERTa-v3 (MNLI+FEVER+ANLI) 계열
온프레미스(에어갭): 4-bit 양자화 모델을 로컬 경로로 지정.
"""

from __future__ import annotations

from typing import Optional


class TransformerNLI:
    """premise→hypothesis entailment 확률을 반환하는 트랜스포머 백엔드.

    Parameters
    ----------
    model_name:
        HF 모델 ID 또는 로컬 디렉터리 경로(에어갭 배포 시).
    device:
        "cpu" / "cuda".
    entailment_label:
        엔테일먼트에 해당하는 라벨명(모델별 상이).
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: str = "cpu",
        entailment_label: str = "entailment",
        cache_size: int = 200_000,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.entailment_label = entailment_label
        self.cache_size = cache_size
        self._cache: dict[tuple[str, str], float] = {}
        self._pipe = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as e:  # pragma: no cover - 환경 의존
            raise RuntimeError(
                "transformers/torch 미설치. `pip install transformers torch` 후, "
                "HF 접근 가능 또는 로컬 모델 경로(model_name)를 지정하세요."
            ) from e
        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
            top_k=None,
        )

    @staticmethod
    def _entail_score(scores: list[dict], want: str) -> float:
        """엔테일먼트 확률 추출.

        라벨 명명은 모델마다 다르다:
          - MNLI 계열: 'entailment' / 'neutral' / 'contradiction'
          - id2label 미설정 파인튜닝 모델: 'LABEL_0' / 'LABEL_1'
        이름 매칭 실패 시 **인덱스 0 을 엔테일먼트로 간주**해 폴백한다
        (본 프로젝트 파인튜닝은 0=SUPPORT 규약. train/nli_finetune.py 참조).
        """
        for s in scores:
            if str(s["label"]).lower() == want.lower():
                return float(s["score"])
        for s in scores:
            if str(s["label"]).upper() in ("LABEL_0", "0"):
                return float(s["score"])
        return float(scores[0]["score"]) if scores else 0.0

    def __call__(self, premise: str, hypothesis: str) -> float:
        key = (premise, hypothesis)
        if key in self._cache:
            return self._cache[key]
        self._ensure_loaded()
        assert self._pipe is not None
        out = self._pipe({"text": premise, "text_pair": hypothesis})
        scores = out[0] if (out and isinstance(out[0], list)) else out
        val = self._entail_score(scores, self.entailment_label)
        # 임계값 스윕은 동일 쌍을 반복 평가하므로 캐시 효과가 크다.
        if len(self._cache) < self.cache_size:
            self._cache[key] = val
        return val


def load_default(model_name: Optional[str] = None, device: str = "cpu") -> TransformerNLI:
    """기본 트랜스포머 NLI 로더. (환경에서 HF/로컬 접근 가능 시)"""
    return TransformerNLI(model_name=model_name or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)
