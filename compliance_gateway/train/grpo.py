"""GRPO(RLVR) — VCR 을 검증가능 보상함수로 사용하는 자기진화 학습.

사업계획서 핵심 수학적 기여:
    L_GRPO = -E_q[ Σ_t min(r_t·Â_t, clip(r_t,1-ε,1+ε)·Â_t) - β·D_KL(π_θ‖π_ref) ]
    여기서 R(o_i) = VCR(o_i)  ← 본 과제의 기여

RLVR 정의상 보상은 '학습된 보상모델'이 아니라 **결정론적·검증가능** 함수여야 한다.
VCR 은 (출처 존재/일치·ALCOA+·환각) 모두 기계 검증 가능하므로 RLVR 요건을 만족한다.

A100 2장 권장 구성(2026 커뮤니티 베스트프랙티스):
  - vLLM server 모드: GPU0=생성(vLLM), GPU1=학습 → 생성/학습 간섭 제거
  - num_generations=8, temperature=0.8, beta=0.04

실행(2 GPU, server 모드):
  # 터미널 1 (생성 서버)
  CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model <model_id>
  # 터미널 2 (학습)
  CUDA_VISIBLE_DEVICES=1 python -m compliance_gateway.train.grpo --vllm-mode server

단일 GPU(colocate):
  python -m compliance_gateway.train.grpo --vllm-mode colocate
"""

from __future__ import annotations

import argparse
from typing import Optional

from compliance_gateway.train.config import GRPOConfig


def make_vcr_reward_fn(
    verifier=None,
    nli_fn=None,
    rescale_to_signed: bool = True,
):
    """VCR 기반 보상함수 팩토리 (TRL GRPOTrainer 시그니처).

    TRL 은 `reward_func(prompts, completions, **kwargs) -> list[float]` 를 기대한다.
    grounding 은 데이터셋 컬럼으로 전달되어 kwargs 로 들어온다.

    rescale_to_signed=True 면 VCR[0,1] → [-1,1] 로 변환(RLVR 권장 범위).
    """
    from compliance_gateway.vcr.reward import compute_vcr

    def reward_func(prompts, completions, **kwargs) -> list[float]:
        groundings = kwargs.get("grounding") or [""] * len(completions)
        rewards: list[float] = []
        for prompt, completion, grounding in zip(prompts, completions, groundings):
            # completion 이 chat 포맷일 수 있음
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            g = (grounding,) if isinstance(grounding, str) and grounding else ()
            v = compute_vcr(prompt if isinstance(prompt, str) else str(prompt),
                            text, grounding=g, nli_fn=nli_fn, verifier=verifier).vcr
            rewards.append(v * 2.0 - 1.0 if rescale_to_signed else v)
        return rewards

    reward_func.__name__ = "vcr_reward"
    return reward_func


def run(cfg: GRPOConfig, dataset_path: str, verifier=None) -> None:
    from datasets import load_dataset
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    from compliance_gateway.train.loader import load_model_and_tokenizer

    model, tokenizer, _ = load_model_and_tokenizer(
        cfg.model_id(), max_seq_len=cfg.max_prompt_len + cfg.max_completion_len,
    )
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    reward_fn = make_vcr_reward_fn(
        verifier=verifier, rescale_to_signed=cfg.reward_rescale_to_signed
    )

    args = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        learning_rate=cfg.lr,
        beta=cfg.beta,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        max_prompt_length=cfg.max_prompt_len,
        max_completion_length=cfg.max_completion_len,
        use_vllm=cfg.use_vllm,
        vllm_mode=cfg.vllm_mode,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        logging_steps=5,
        save_strategy="epoch",
    )
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=dataset,
        reward_funcs=[reward_fn], processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[GRPO] saved → {cfg.output_dir}")
    print(f"[GRPO] 롤백 기준: VCR < {cfg.kl_rollback_threshold} → DPO 모델로 복귀")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="exaone")
    ap.add_argument("--dataset", default="data/synth/grpo_prompts.jsonl")
    ap.add_argument("--output", default="checkpoints/grpo")
    ap.add_argument("--vllm-mode", default="server", choices=["server", "colocate"])
    ap.add_argument("--num-generations", type=int, default=8)
    a = ap.parse_args()
    cfg = GRPOConfig(base_model=a.base, output_dir=a.output,
                     vllm_mode=a.vllm_mode, num_generations=a.num_generations)

    # 서지 검증기 주입(오프라인 로컬 레지스트리 기본)
    from compliance_gateway.verify import CitationVerifier, LocalRegistry
    verifier = CitationVerifier([LocalRegistry.from_seed()])
    run(cfg, a.dataset, verifier=verifier)


if __name__ == "__main__":
    main()
