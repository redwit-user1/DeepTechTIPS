"""LLM 판정 NLI 백엔드 — 지시 따르기 LLM 에게 "근거가 주장을 뒷받침하는가"를 묻는다.

점수 = 다음 토큰에서 P(Yes) / (P(Yes) + P(No)). 생성하지 않고 로짓만 보므로 빠르고
결정적이다. `TransformerNLI` 와 같은 시그니처(premise, hypothesis) -> float 라
`ComplianceGateway(nli_fn=...)` 에 그대로 주입된다.

    LLMJudgeNLI("Qwen/Qwen3.8-27B", device="cuda").batch(pairs)
"""

from __future__ import annotations

# v2(2026-11-09): v1 은 정당한 근사·환산을 반박으로 봤다(어려운 셋 round_entail 평균 0.63).
PROMPT = (
    "다음 '근거'만 보고 '주장'이 참이라고 확인할 수 있는지 판정하라.\n"
    "- 근거가 주장을 직접 뒷받침하면 Yes.\n"
    "- 값이 같은 다른 표현도 뒷받침이다: 반올림·'약'을 붙인 근사, 올바른 단위 환산"
    "(1,000 mg = 1 g, 60 분 = 1 시간), 분수↔백분율, 같은 범위를 넘지 않는 하한/상한 표현.\n"
    "- 근거와 모순되면 No: 다른 값, 같은 값이지만 크기가 다른 단위(µg↔mg), 반대 방향, "
    "뒤바뀐 대상·조건, 뒤집힌 부등호(이상↔이하).\n"
    "- 근거만으로 확인할 수 없으면 No: 근거에 없는 원인·효과·일반화를 덧붙였거나, "
    "근거는 추정(판단됨·보임·가능성)인데 주장은 확정(확인됨·입증됨)으로 말한 경우.\n"
    "Yes 또는 No 한 단어로만 답하라.\n\n"
    "근거: {premise}\n주장: {hypothesis}"
)


# 질의응답 적합성(2026-11-12, 연구노트 §39): 문장 게이팅은 답이 원문에 있는지만 본다. 한 조건의 값을 물었는데
# 같은 단락의 다른 조건 값을 답해도 통과했다. 질문·답을 함께 준다 — {hypothesis} 자리에 "질문: … / 답: …"(줄바꿈으로 구분).
QA_PROMPT = (
    "다음 '연구노트'만 보고 '답'이 '질문'에 대한 올바른 답인지 판정하라.\n"
    "- 질문이 묻는 대상·조건·시점에 해당하는 내용을 연구노트대로 답했으면 Yes.\n"
    "- 연구노트에 있는 내용이라도 질문이 묻는 것과 다른 대상·조건·시점의 값이면 No.\n"
    "- 답이 연구노트와 다르거나, 연구노트로 확인할 수 없거나, 질문에 답하지 않으면 No.\n"
    "Yes 또는 No 한 단어로만 답하라.\n\n"
    "연구노트: {premise}\n{hypothesis}"
)


class LLMJudgeNLI:
    def __init__(self, model_name: str = "Qwen/Qwen3.8-27B", device: str = "cuda",
                 batch_size: int = 16, max_chars: int = 8000) -> None:
        # 근거가 잘리면 뒤쪽 근거를 못 봐 '미지지'로 오판한다(누출이 아니라 오탐 쪽). 2000 자는
        # 연구노트 한 페이지도 잘랐다. Qwen3.8 문맥은 충분 — 메모리가 모자라면 batch_size 를 줄인다.
        # 게이트웨이 문장 게이팅은 근거를 800 자 창(pipeline.WINDOW_CHARS)으로 이미 나눠 넣는다.
        self.model_name, self.device = model_name, device
        self.batch_size, self.max_chars = batch_size, max_chars
        self._m = self._tok = None
        self._cache: dict[tuple[str, str], float] = {}

    def _load(self) -> None:
        if self._m is not None:
            return
        import torch
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._tok.padding_side = "left"   # 마지막 위치 로짓을 읽으므로 왼쪽 패딩
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        if not getattr(self._tok, "chat_template", None):
            # Mistral 계열은 대화 틀이 토크나이저가 아니라 프로세서에 있다(연구노트 §44: 다른 계열 판정 모델)
            try:
                from transformers import AutoProcessor
                self._tok.chat_template = AutoProcessor.from_pretrained(self.model_name).chat_template
            except Exception:
                pass
        try:   # Qwen3.5+ 는 비전 통합 모델 — 텍스트만 넣어도 된다
            from transformers import AutoModelForImageTextToText as Auto
            self._m = Auto.from_pretrained(self.model_name, dtype=torch.bfloat16, device_map=self.device)
        except (ValueError, KeyError):
            from transformers import AutoModelForCausalLM as Auto
            self._m = Auto.from_pretrained(self.model_name, dtype=torch.bfloat16, device_map=self.device)
        self._m.eval()
        ids = lambda w: self._tok.encode(w, add_special_tokens=False)[0]
        self._yes, self._no = ids("Yes"), ids("No")

    def _text(self, premise: str, hypothesis: str, prompt: str = PROMPT) -> str:
        msg = [{"role": "user", "content": prompt.format(
            premise=premise[: self.max_chars], hypothesis=hypothesis[: self.max_chars])}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if not getattr(self._tok, "chat_template", None):
            return f"[INST] {msg[0]['content']} [/INST]"      # 대화 틀을 끝내 못 찾은 경우의 최소 형식
        try:
            return self._tok.apply_chat_template(msg, enable_thinking=False, **kw)
        except TypeError:
            return self._tok.apply_chat_template(msg, **kw)

    def batch(self, pairs: list[tuple[str, str]], prompt: str = PROMPT) -> list[float]:
        """prompt 를 바꾸면 같은 모델로 다른 판정(예: QA_PROMPT)을 한다. 캐시는 프롬프트별."""
        import torch

        self._load()
        key = (lambda p: p) if prompt is PROMPT else (lambda p: (prompt, *p))
        todo = [p for p in dict.fromkeys(pairs) if key(p) not in self._cache]
        todo.sort(key=lambda p: len(p[0]) + len(p[1]))      # 길이순 — 패딩 낭비를 줄인다
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i: i + self.batch_size]
            enc = self._tok([self._text(*p, prompt=prompt) for p in chunk], return_tensors="pt", padding=True).to(self._m.device)
            with torch.no_grad():
                logits = self._m(**enc).logits[:, -1, :].float()
            two = torch.softmax(logits[:, [self._yes, self._no]], dim=-1)[:, 0].tolist()
            self._cache.update(zip(map(key, chunk), two))
        return [self._cache[key(p)] for p in pairs]

    def __call__(self, premise: str, hypothesis: str) -> float:
        return self.batch([(premise, hypothesis)])[0]
