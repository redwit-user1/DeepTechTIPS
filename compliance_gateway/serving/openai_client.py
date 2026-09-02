"""OpenAI 호환 서빙 클라이언트 (의존성 없음).

설계 원칙: 서빙 장애가 파이프라인 전체를 죽이면 안 된다.
  - 재시도 + 지수 백오프
  - 실패 시 예외 대신 안전한 기본값(NLI 는 0.5=중립)을 돌려주는 옵션
  - 응답 캐시(임계값 스윕에서 동일 쌍 반복 호출 방지)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional


def _api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("KT_API_KEY")


class ServedModel:
    """OpenAI 호환 chat/completions 클라이언트.

    Parameters
    ----------
    base_url : 예) "http://<host>:8000/v1"
    model    : 서빙 중인 모델 ID (`/v1/models` 로 확인)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model = model
        self.api_key = api_key or _api_key()
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    # ---- 저수준 호출 -----------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:300]
                last = RuntimeError(f"HTTP {e.code}: {detail}")
                if e.code < 500 and e.code != 429:
                    break               # 4xx(429 제외)는 재시도해도 동일
            except Exception as e:       # 네트워크·타임아웃
                last = e
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        raise RuntimeError(f"서빙 호출 실패 ({url}): {last}")

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("id", "") for m in data.get("data", [])]

    # ---- 생성 -----------------------------------------------------------
    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 512) -> str:
        """Gateway 7단계 중 03(생성)에 사용."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        out = self._post("/chat/completions", {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": max_tokens,
        })
        return out["choices"][0]["message"]["content"]


# NLI 판정 프롬프트 — 출력 형식을 강하게 제약해 파싱 실패를 줄인다.
_NLI_SYSTEM = (
    "You are a strict scientific fact-verification system. "
    "Given EVIDENCE and a CLAIM, decide whether the evidence supports the claim. "
    "Answer with exactly one word: SUPPORT, CONTRADICT, or NEUTRAL. No explanation."
)
_NLI_USER = "EVIDENCE:\n{premise}\n\nCLAIM:\n{hypothesis}\n\nAnswer (SUPPORT/CONTRADICT/NEUTRAL):"

_VERDICT_SCORE = {"SUPPORT": 1.0, "NEUTRAL": 0.5, "CONTRADICT": 0.0}


class ServedNLI:
    """서빙 모델에 entailment 판정을 위임하는 NLI 백엔드.

    `nli_fn(premise, hypothesis) -> float` 시그니처를 만족하므로
    `ComplianceGateway(nli_fn=ServedNLI(...))` 로 바로 주입된다.

    로컬 트랜스포머 NLI 가 정확도·비용 면에서 우수하지만, 학습 환경이 없고
    추론 서빙만 있는 경우(AI Serv)의 현실적 대안이다.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        cache_size: int = 200_000,
        fail_value: float = 0.5,
        timeout: float = 60.0,
    ) -> None:
        self.client = ServedModel(base_url, model, api_key=api_key,
                                  timeout=timeout, temperature=0.0)
        self.cache: dict[tuple[str, str], float] = {}
        self.cache_size = cache_size
        self.fail_value = fail_value      # 호출 실패 시 중립(정상 인용을 벌하지 않음)
        self.failures = 0

    @staticmethod
    def parse_verdict(text: str) -> float:
        up = (text or "").strip().upper()
        for verdict, score in _VERDICT_SCORE.items():
            if up.startswith(verdict) or verdict in up:
                return score
        return 0.5

    def __call__(self, premise: str, hypothesis: str) -> float:
        key = (premise, hypothesis)
        if key in self.cache:
            return self.cache[key]
        try:
            out = self.client.generate(
                _NLI_USER.format(premise=premise[:4000], hypothesis=hypothesis[:1000]),
                system=_NLI_SYSTEM, max_tokens=8,
            )
            score = self.parse_verdict(out)
        except Exception:
            self.failures += 1
            score = self.fail_value
        if len(self.cache) < self.cache_size:
            self.cache[key] = score
        return score
