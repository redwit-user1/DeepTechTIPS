"""SFT — 도메인 LoRA 지시 미세조정 (TRL SFTTrainer).

A100 실행:
  python -m compliance_gateway.train.sft --base exaone \
      --dataset data/synth/sft.jsonl --output checkpoints/sft

의존성(A100 환경): pip install "compliance-gateway[train]"
"""

from __future__ import annotations

import argparse

from compliance_gateway.train.config import SFTConfig


def run(cfg: SFTConfig) -> None:
    # 지연 임포트: torch/transformers/trl/peft 는 A100 환경에서만 필요.
    import torch  # noqa: F401
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id(), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id(), torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")

    peft_config = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
    )
    args = TRLSFTConfig(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr,
        max_seq_length=cfg.max_seq_len, bf16=cfg.bf16, logging_steps=10, save_strategy="epoch",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                         peft_config=peft_config, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[SFT] saved → {cfg.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="exaone")
    ap.add_argument("--dataset", default="data/synth/sft.jsonl")
    ap.add_argument("--output", default="checkpoints/sft")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    run(SFTConfig(base_model=a.base, dataset_path=a.dataset, output_dir=a.output, epochs=a.epochs))


if __name__ == "__main__":
    main()
