"""서빙 엔드포인트 연동 — OpenAI 호환 API.

KT Cloud AI Nexus 의 **AI Serv**(추론 전용, GPU 슬라이싱)처럼 학습은 불가하고
추론 엔드포인트만 있는 환경에서도 전체 파이프라인이 동작하도록 한다.
vLLM(`trl vllm-serve`, `vllm serve`), TGI, OpenAI, 사내 게이트웨이 등
OpenAI 호환 API 라면 모두 같은 코드로 붙는다.

용도 2가지:
  1. **생성 백엔드** — Gateway 7단계 중 03(생성). Model-Agnostic 설계의 실증.
  2. **NLI 백엔드** — 로컬 트랜스포머가 없을 때 서빙 모델에 판정을 위임.

의존성 없음(표준 urllib). 인증키는 환경변수 사용:
  OPENAI_API_KEY 또는 KT_API_KEY
"""

from compliance_gateway.serving.openai_client import ServedModel, ServedNLI

__all__ = ["ServedModel", "ServedNLI"]
