"""모델 로더 — Unsloth 우선, 미설치 시 표준 HF 로 자동 폴백.

Unsloth 채택 근거(2026 동향 조사): LoRA/GRPO 학습 가속, RL VRAM 대폭 절감,
장문맥 학습 3배 속도·30% 메모리 절감. 멀티GPU 는 accelerate/torchrun DDP.
→ docs/UPSTREAM_TECH.md

주의: Unsloth 는 API 가 빠르게 변하므로, 실패 시 조용히 HF 경로로 폴백해
학습이 중단되지 않게 한다(로그로 어느 경로인지 항상 표시).
"""

from __future__ import annotations

from typing import Any


def load_model_and_tokenizer(
    model_id: str,
    max_seq_len: int = 2048,
    load_in_4bit: bool = False,
    use_unsloth: bool = True,
    bf16: bool = True,
) -> tuple[Any, Any, bool]:
    """(model, tokenizer, unsloth_used) 반환."""
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_len,
                load_in_4bit=load_in_4bit,
                dtype=None,          # 자동(A100 → bf16)
            )
            print(f"[loader] Unsloth 경로 사용: {model_id}")
            return model, tokenizer, True
        except Exception as e:  # pragma: no cover - 환경 의존
            print(f"[loader] Unsloth 사용 불가({e.__class__.__name__}: {e}) → HF 폴백")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kw = dict(torch_dtype=torch.bfloat16 if bf16 else torch.float16, device_map="auto", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except (ValueError, KeyError):
        # Qwen3.5+ (Qwen3.8-27B 포함)는 비전 통합 클래스(Qwen3_5ForConditionalGeneration)라
        # CausalLM 자동 매핑이 없을 수 있다. 텍스트만 학습해도 이 클래스로 로드한다.
        # 주의(미검증): LoRA target_modules 가 비전 타워의 gate/up/down_proj 에도 붙을 수 있다 — 첫 학습 때 확인.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    print(f"[loader] HF 경로 사용: {model_id}")
    return model, tokenizer, False


def apply_lora(model: Any, lora_cfg, unsloth_used: bool, max_seq_len: int = 2048) -> Any:
    """Unsloth 경로면 네이티브 PEFT 부착, 아니면 호출부에서 peft_config 사용."""
    if not unsloth_used:
        return model
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,     # 0 이어야 Unsloth 고속 경로
        bias=getattr(lora_cfg, "bias", "none"),
        target_modules=list(lora_cfg.target_modules),
        use_gradient_checkpointing="unsloth",   # 장문맥 메모리 절감
        max_seq_length=max_seq_len,
    )
