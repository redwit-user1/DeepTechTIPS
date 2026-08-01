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
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    from compliance_gateway.train.loader import apply_lora, load_model_and_tokenizer

    model, tokenizer, unsloth_used = load_model_and_tokenizer(
        cfg.model_id(), max_seq_len=cfg.max_seq_len,
        load_in_4bit=cfg.load_in_4bit, use_unsloth=cfg.use_unsloth, bf16=cfg.bf16,
    )
    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")

    # Unsloth 경로는 네이티브 PEFT 부착, HF 경로는 TRL 에 peft_config 전달
    model = apply_lora(model, cfg.lora, unsloth_used, max_seq_len=cfg.max_seq_len)
    peft_config = None if unsloth_used else LoraConfig(
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
